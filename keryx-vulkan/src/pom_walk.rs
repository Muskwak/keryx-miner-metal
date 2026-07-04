//! GPU Proof-of-Model walk: dispatch the `pom_walk` compute shader over a resident weight blob to
//! find the lowest nonce in a batch whose `pom_pow_value <= target`. The folds are byte-identical
//! to `src/pom.rs`, so a nonce found here builds a `PomProof` the node accepts.
//!
//! Two shader variants, picked at runtime by [`Vk::supports_shader_int64`]:
//!   - `pom_walk.comp` — native `uint64_t` arithmetic. Used whenever the device supports it (all
//!     desktop AMD/NVIDIA GPUs, and some mobile GPUs).
//!   - `pom_walk_i32.comp` — the same math hand-emulated over `uvec2` (lo, hi) pairs. Used when
//!     `shaderInt64` is unavailable (confirmed missing on Qualcomm Adreno 740 by direct device
//!     query, despite otherwise-modern Vulkan 1.3 support). Costs more ALU work per nonce, so it's
//!     only used where the native path genuinely isn't an option.
//!
//! Both variants read/write the exact same byte layout for the weight blob and shard address
//! table (a little-endian `uint64_t` and a `uvec2(lo, hi)` occupy the same 8 bytes), so the host
//! side only needs to change how the *push constants* are packed, not how buffers are uploaded.

use crate::{GpuBuffer, Kernel, Vk};
use std::io::Cursor;

/// SPIR-V for the native (`uint64_t`) PoM walk variant, compiled from `shaders/pom_walk.comp`.
const POM_WALK_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_walk.spv"));
/// SPIR-V for the `shaderInt64`-less (`uvec2`-emulated) variant, from `shaders/pom_walk_i32.comp`.
const POM_WALK_I32_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_walk_i32.spv"));

/// POM_WALK_STEPS — must match `pom::POM_WALK_STEPS` and the node.
pub const POM_WALK_STEPS: u32 = 256;

/// Native push-constant block — layout MUST match the `Push` block in `pom_walk.comp` (std430:
/// twelve u64 at 0..96, then three u32 at 96, 100, 104; total 112 bytes incl. tail pad). The weight
/// blob is split into power-of-two-sized shards (each a separate device-address buffer, sized to
/// the device's real `maxMemoryAllocationSize`); the shader maps a chunk to its shard via
/// `shard_shift`/`shard_mask` and reads the shard's GPU address from a small bound address table.
#[repr(C)]
#[derive(Clone, Copy)]
struct PomPush {
    p: [u64; 4],
    t: [u64; 4],
    timestamp: u64,
    n_chunks: u64,
    start_nonce: u64,
    shard_mask: u64, // chunks_per_shard - 1
    k: u32,
    batch: u32,
    shard_shift: u32, // log2(chunks_per_shard)
}

/// Emulated push-constant block — layout MUST match the `Push` block in `pom_walk_i32.comp`.
/// Every field that's a genuine arbitrary 64-bit value (hash/target words, timestamp, nonce) is a
/// `[u32; 2]` (lo, hi) pair; `n_chunks`/`shard_mask` fit in `u32` for any realistic model (see the
/// shader's `mod64` comment) and stay plain `u32`, same as `k`/`batch`/`shard_shift`. All ten pair
/// fields are declared first (each naturally 8-byte-aligned when packed from offset 0) followed by
/// the five plain `u32` fields, so this repr(C) layout matches GLSL's std430 push-constant layout
/// byte-for-byte without needing an explicit `#[repr(align(8))]` wrapper — see the compile-time
/// size assertion below, which would catch any future reordering that broke this.
#[repr(C)]
#[derive(Clone, Copy)]
struct PomPush32 {
    p: [[u32; 2]; 4],
    t: [[u32; 2]; 4],
    timestamp: [u32; 2],
    start_nonce: [u32; 2],
    n_chunks: u32,
    shard_mask: u32,
    k: u32,
    batch: u32,
    shard_shift: u32,
}

const _: () = assert!(std::mem::size_of::<PomPush32>() == 100);

fn split_u64(v: u64) -> [u32; 2] {
    [(v & 0xFFFF_FFFF) as u32, (v >> 32) as u32]
}

const NO_WINNER: u32 = 0xFFFF_FFFF;

/// Chunks per shard: 2^25 × 32 B = 1 GiB, comfortably under the AMD 2 GiB `maxMemoryAllocationSize`
/// single-allocation cap, with headroom for driver overhead. Power of two so the shader maps a
/// chunk to (shard, offset) with a shift + mask instead of 64-bit divide.
const SHARD_CHUNKS: u64 = 1 << 25;

/// Max nonces per GPU dispatch. The walk is latency-bound (256 dependent reads/nonce), so a full
/// 1<<20 batch in one dispatch can exceed the Windows TDR watchdog (~2 s) and lose the device.
/// 65,536 keeps each dispatch to a few ms on device-local VRAM while staying far above launch cost.
const MAX_DISPATCH_NONCES: u32 = 1 << 16;

/// `mod64`'s binary long division (in `pom_walk_i32.comp`) assumes the divisor (`n_chunks`) is
/// strictly less than 2^31 — see that shader's comment. 2^31 32-byte chunks would be a ~64 GiB
/// weight blob, far beyond anything this miner ever loads, but the guard is here so a future
/// pathological input fails loudly instead of silently mis-mining.
const MAX_N_CHUNKS_FOR_I32_MOD: u64 = 1 << 31;

/// Resident GPU PoM miner: the weight blob lives in a storage buffer; `mine` re-dispatches the
/// walk over nonce batches. Build once per mining tier (the weight blob is large).
pub struct PomWalkGpu {
    vk: Vk,
    kernel: Kernel,
    shards: Vec<GpuBuffer>, // weight blob split into ≤1 GiB device-address buffers
    addr_table: GpuBuffer,  // bound SSBO: one u64 GPU address per shard
    winner: GpuBuffer,
    n_chunks: u64,
    shard_chunks: u64,
    use_i32: bool, // true: shaderInt64-less emulated variant; false: native uint64_t variant
}

impl PomWalkGpu {
    /// Upload the canonical weight blob (`weight_words` = the model's quant bytes as little-endian
    /// u64 words, `n_chunks * 4` of them) and compile the walk kernel on the GPU.
    pub fn new(weight_words: &[u64], n_chunks: u64) -> Result<Self, String> {
        Self::new_sharded(weight_words, n_chunks, SHARD_CHUNKS)
    }

    /// Like [`new`](Self::new) but with an explicit shard size (chunks per shard, power of two).
    /// Lets tests force a multi-shard layout without multi-GiB allocations.
    pub fn new_sharded(weight_words: &[u64], n_chunks: u64, shard_chunks: u64) -> Result<Self, String> {
        if n_chunks == 0 || weight_words.len() as u64 != n_chunks * 4 {
            return Err(format!(
                "weight blob size mismatch: {} words for {} chunks (expected {})",
                weight_words.len(),
                n_chunks,
                n_chunks * 4
            ));
        }
        if !shard_chunks.is_power_of_two() {
            return Err(format!("shard_chunks must be a power of two, got {shard_chunks}"));
        }
        let vk = Vk::new()?;
        let use_i32 = !vk.supports_shader_int64();
        if use_i32 && n_chunks >= MAX_N_CHUNKS_FOR_I32_MOD {
            return Err(format!(
                "n_chunks={n_chunks} exceeds the shaderInt64-less shader's mod64 limit ({MAX_N_CHUNKS_FOR_I32_MOD})"
            ));
        }
        let (spv_bytes, push_size) = if use_i32 {
            (POM_WALK_I32_SPV, std::mem::size_of::<PomPush32>() as u32)
        } else {
            (POM_WALK_SPV, std::mem::size_of::<PomPush>() as u32)
        };
        let spirv = ash::util::read_spv(&mut Cursor::new(spv_bytes)).map_err(|e| e.to_string())?;
        // Two descriptor bindings: the winner buffer and the shard address table. The (large) weight
        // shards are reached by device address, not bound as descriptors.
        let kernel = vk.make_kernel(&spirv, 2, push_size)?;

        // Split the blob on chunk boundaries into device-address shards; collect their GPU addresses.
        let n_shards = n_chunks.div_ceil(shard_chunks);
        let mut shards: Vec<GpuBuffer> = Vec::with_capacity(n_shards as usize);
        let mut addrs: Vec<u64> = Vec::with_capacity(n_shards as usize);
        for s in 0..n_shards {
            let first_word = (s * shard_chunks * 4) as usize;
            let last_word = (((s + 1) * shard_chunks).min(n_chunks) * 4) as usize;
            let slice = &weight_words[first_word..last_word];
            // Device-local VRAM (staged copy): the walk's random reads are ~100x faster here than
            // host-visible memory — host-visible overran the TDR watchdog → DEVICE_LOST.
            let (buf, addr) = vk.create_device_local_address_buffer(words_as_bytes(slice))?;
            shards.push(buf);
            addrs.push(addr);
        }

        // Address table (tiny — one u64 per shard) bound as a normal storage buffer at binding 1.
        let addr_table = vk.create_buffer((addrs.len() * 8) as u64)?;
        vk.write_buffer(&addr_table, words_as_bytes(&addrs));
        let winner = vk.create_buffer(4)?;

        Ok(Self { vk, kernel, shards, addr_table, winner, n_chunks, shard_chunks, use_i32 })
    }

    /// Name of the GPU the miner is running on.
    pub fn device_name(&self) -> &str {
        self.vk.device_name()
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_chunks
    }

    /// Search nonces `[start, start + batch)`. Returns the lowest winning nonce, or None.
    ///
    /// The batch is ground in `MAX_DISPATCH_NONCES`-sized sub-dispatches, in increasing nonce order,
    /// so no single GPU dispatch runs long enough to trip the Windows TDR watchdog (DEVICE_LOST).
    /// Sub-batches are ascending, so the first one with any winner holds the global lowest nonce —
    /// returning there is identical to grinding the whole batch, and skips the rest.
    pub fn mine(&self, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u32) -> Option<u64> {
        let mut done: u32 = 0;
        while done < batch {
            let sub = (batch - done).min(MAX_DISPATCH_NONCES);
            self.vk.write_buffer(&self.winner, &NO_WINNER.to_le_bytes());
            let start_nonce = start + done as u64;
            let shard_mask = self.shard_chunks - 1;
            let shard_shift = self.shard_chunks.trailing_zeros();

            let groups = sub.div_ceil(64); // local_size_x = 64
            if self.use_i32 {
                let p = words4(pre_pow_hash);
                let t = words4(target_le);
                let push = PomPush32 {
                    p: [split_u64(p[0]), split_u64(p[1]), split_u64(p[2]), split_u64(p[3])],
                    t: [split_u64(t[0]), split_u64(t[1]), split_u64(t[2]), split_u64(t[3])],
                    timestamp: split_u64(timestamp),
                    start_nonce: split_u64(start_nonce),
                    n_chunks: self.n_chunks as u32,
                    shard_mask: shard_mask as u32,
                    k: POM_WALK_STEPS,
                    batch: sub,
                    shard_shift,
                };
                self.vk.dispatch(&self.kernel, &[&self.winner, &self.addr_table], push_bytes(&push), groups);
            } else {
                let push = PomPush {
                    p: words4(pre_pow_hash),
                    t: words4(target_le),
                    timestamp,
                    n_chunks: self.n_chunks,
                    start_nonce,
                    shard_mask,
                    k: POM_WALK_STEPS,
                    batch: sub,
                    shard_shift,
                };
                self.vk.dispatch(&self.kernel, &[&self.winner, &self.addr_table], push_bytes(&push), groups);
            }

            let mut out = [0u8; 4];
            self.vk.read_buffer(&self.winner, &mut out);
            if let offset @ 0..=0xFFFF_FFFE = u32::from_le_bytes(out) {
                return Some(start + done as u64 + offset as u64);
            }
            done += sub;
        }
        None
    }
}

impl Drop for PomWalkGpu {
    fn drop(&mut self) {
        self.vk.destroy_buffer(&self.winner);
        self.vk.destroy_buffer(&self.addr_table);
        for shard in &self.shards {
            self.vk.destroy_buffer(shard);
        }
        self.vk.destroy_kernel(&self.kernel);
    }
}

/// 32 LE bytes → 4 u64 words (matches `pom::pph_words` / `words4`).
pub fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

fn words_as_bytes(words: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words)) }
}

fn push_bytes<T>(p: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(p as *const T as *const u8, std::mem::size_of::<T>()) }
}
