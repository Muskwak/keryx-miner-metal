//! Apple Silicon (Metal) PoM GPU miner — parity path for `pom_gpu.rs` (CUDA).
//!
//! Same public free-function surface as the CUDA module (`install`/`uninstall`/`is_installed`/
//! `is_loading`/`mine`/`current_tier`/`ensure_installed`/`set_mining_tier`) so callers in
//! main.rs / miner.rs / slm.rs are backend-agnostic. Under the hood:
//!
//!   * Weights are packed into ONE Metal buffer in canonical (name-sorted) GGUF tensor order —
//!     the same order the CUDA gather is built over — so a chunk at global index `off` lives
//!     at bytes `[off*32 .. off*32+32]`. The Metal kernel drops the per-tensor (bases, prefix)
//!     binary search the CUDA path needs, because candle 0.9 does not publicly expose
//!     `QMetalStorage::buffer` from a `QTensor` — so we cannot do zero-dup here yet. Cost is one
//!     extra weight-sized allocation in unified memory; acceptable while Metal stays opt-in.
//!   * `metal/pom_mine.metal` is `include_str!`d and compiled the first time the miner loads.
//!     One `ComputePipeline` cached per miner instance.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use log::info;

use candle_core::quantized::{gguf_file, QTensor};
use candle_core::Device;
use candle_metal_kernels::metal::{
    create_command_buffer, Buffer, CommandQueue, CommandSemaphore, ComputePipeline,
    Device as MtlDevice, MTLResourceOptions,
};
use objc2_metal::{MTLResourceOptions as ObjcMTLResourceOptions, MTLSize};

const METAL_SRC: &str = include_str!("../metal/pom_mine.metal");
const CHUNK_BYTES: usize = 32;
const THREADGROUP_SIZE: usize = 256;

/// Shared-storage buffers: CPU and GPU see the same unified-memory backing, so the host can
/// write uniforms / read the winner without a blit copy. See candle's own RESOURCE_OPTIONS
/// for the same choice on candle's transient buffers.
const SHARED_STORAGE: MTLResourceOptions =
    ObjcMTLResourceOptions(ObjcMTLResourceOptions::StorageModeShared.bits());

pub struct PomGpuMiner {
    device: MtlDevice,
    queue: CommandQueue,
    pipeline: ComputePipeline,
    weights: Buffer,
    n_total_chunks: u64,
    _tensors: Vec<QTensor>, // kept alive so any device-side buffers aren't dropped early
}

// Matches PomUniforms in metal/pom_mine.metal — field order and padding are load-bearing.
#[repr(C)]
struct Uniforms {
    n_total_chunks: u64,
    k_steps: u32,
    _pad0: u32,
    p0: u64, p1: u64, p2: u64, p3: u64,
    time_: u64,
    t0: u64, t1: u64, t2: u64, t3: u64,
    nonce_base: u64,
    n_nonces: u32,
    _pad1: u32,
}

impl PomGpuMiner {
    /// Load the mining model's GGUF into a candle Metal device, pack it into a single Metal
    /// buffer, compile the kernel. Heavy — call once per (device, model).
    pub fn load(gguf_path: &str, device_id: usize) -> candle_core::Result<Self> {
        let cdev = Device::new_metal(device_id)?;
        let mdev = match &cdev {
            Device::Metal(m) => m.metal_device().clone(),
            _ => return Err(candle_core::Error::Msg("PoM Metal: not a Metal device".into())),
        };

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical name-sorted order — matches pom-rt-builder / the node R_T

        let mut tensors: Vec<QTensor> = Vec::with_capacity(names.len());
        let mut packed: Vec<u8> = Vec::new();
        for name in &names {
            let qt = content.tensor(&mut file, name, &cdev)?;
            let n = qt.storage_size_in_bytes();
            if n < CHUNK_BYTES {
                tensors.push(qt);
                continue;
            }
            let bytes = qt.data()?;
            let usable = (n / CHUNK_BYTES) * CHUNK_BYTES; // defensive vs any ragged tail
            packed.extend_from_slice(&bytes[..usable]);
            tensors.push(qt);
        }
        let n_total_chunks = (packed.len() / CHUNK_BYTES) as u64;
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM Metal: model produced 0 chunks".into()));
        }

        let weights = mdev
            .new_buffer_with_data(
                packed.as_ptr() as *const _,
                packed.len(),
                SHARED_STORAGE,
            )
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: weights buffer: {e}")))?;

        let library = mdev
            .new_library_with_source(METAL_SRC, None)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: compile: {e}")))?;
        let func = library
            .get_function("pom_mine", None)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: get_function: {e}")))?;
        let pipeline = mdev
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: pipeline: {e}")))?;

        let queue = mdev
            .new_command_queue()
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: command queue: {e}")))?;

        info!(
            "PoM Metal: packed {} chunks ({} MiB) on device {}",
            n_total_chunks,
            packed.len() / (1024 * 1024),
            device_id
        );

        Ok(Self { device: mdev, queue, pipeline, weights, n_total_chunks, _tensors: tensors })
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_total_chunks
    }

    /// Search nonces in `[start, start + batch)`. Returns the lowest winning nonce, or `None`.
    /// Batch must fit in `u32` (POM_BATCH is 1<<20 — comfortably below the limit); this is what
    /// lets the winner atomic stay a 32-bit tid.
    pub fn mine(
        &self,
        pre_pow_hash: &[u8; 32],
        timestamp: u64,
        target_le: &[u8; 32],
        start: u64,
        batch: u64,
    ) -> candle_core::Result<Option<u64>> {
        if batch > u32::MAX as u64 {
            return Err(candle_core::Error::Msg("PoM Metal: batch exceeds u32".into()));
        }
        if batch == 0 {
            return Ok(None);
        }
        let p = words4(pre_pow_hash);
        let t = words4(target_le);
        let uniforms = Uniforms {
            n_total_chunks: self.n_total_chunks,
            k_steps: crate::pom::POM_WALK_STEPS,
            _pad0: 0,
            p0: p[0], p1: p[1], p2: p[2], p3: p[3],
            time_: timestamp,
            t0: t[0], t1: t[1], t2: t[2], t3: t[3],
            nonce_base: start,
            n_nonces: batch as u32,
            _pad1: 0,
        };

        let uniforms_buf = self
            .device
            .new_buffer_with_data(
                &uniforms as *const _ as *const _,
                std::mem::size_of::<Uniforms>(),
                SHARED_STORAGE,
            )
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: uniforms buffer: {e}")))?;
        let winner_init: u32 = u32::MAX;
        let winner_buf = self
            .device
            .new_buffer_with_data(
                &winner_init as *const _ as *const _,
                std::mem::size_of::<u32>(),
                SHARED_STORAGE,
            )
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: winner buffer: {e}")))?;

        let semaphore = Arc::new(CommandSemaphore::new());
        let cmd = create_command_buffer(&self.queue, semaphore)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: command buffer: {e}")))?;
        let enc = cmd.compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&self.weights), 0);
        enc.set_buffer(1, Some(&uniforms_buf), 0);
        enc.set_buffer(2, Some(&winner_buf), 0);
        let grid = MTLSize { width: batch as usize, height: 1, depth: 1 };
        let tg = MTLSize { width: THREADGROUP_SIZE, height: 1, depth: 1 };
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Safe: shared-storage buffer, contents is CPU-visible, sync has completed.
        let w = unsafe { *(winner_buf.contents() as *const u32) };
        Ok(if w == u32::MAX { None } else { Some(start + w as u64) })
    }
}

fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

// ─── Per-device miner registry ────────────────────────────────────────────────────────────
//
// Same shape as the CUDA path: one PomGpuMiner per device_id, callers hold no direct handle,
// they route everything through the free helpers below.

fn miners() -> &'static Mutex<HashMap<u32, Arc<PomGpuMiner>>> {
    static MINERS: OnceLock<Mutex<HashMap<u32, Arc<PomGpuMiner>>>> = OnceLock::new();
    MINERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn index_build_lock() -> &'static Mutex<()> {
    static INDEX_BUILD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    INDEX_BUILD_LOCK.get_or_init(|| Mutex::new(()))
}

pub fn install(device_id: u32, m: PomGpuMiner) {
    if let Ok(mut g) = miners().lock() {
        g.insert(device_id, Arc::new(m));
    }
}

pub fn uninstall(device_id: u32) {
    if let Ok(mut g) = miners().lock() {
        g.remove(&device_id);
    }
}

pub fn is_installed(device_id: u32) -> bool {
    miners().lock().map(|g| g.contains_key(&device_id)).unwrap_or(false)
}

static LOADING: AtomicUsize = AtomicUsize::new(0);

pub fn is_loading() -> bool {
    LOADING.load(Ordering::Relaxed) > 0
}

pub fn mine(
    device_id: u32,
    pre_pow_hash: &[u8; 32],
    timestamp: u64,
    target_le: &[u8; 32],
    start: u64,
    batch: u64,
) -> Option<u64> {
    let miner = {
        let g = miners().lock().ok()?;
        g.get(&device_id)?.clone()
    };
    miner.mine(pre_pow_hash, timestamp, target_le, start, batch).ok().flatten()
}

static MINING_TIER: OnceLock<([u8; 32], String)> = OnceLock::new();

pub fn set_mining_tier(model_id: [u8; 32], gguf_path: String) {
    let _ = MINING_TIER.set((model_id, gguf_path));
}

pub fn current_tier(daa: u64) -> Option<u8> {
    let (model_id, _) = MINING_TIER.get()?;
    crate::models::pom_tier_index(model_id, daa)
}

pub fn ensure_installed(device_id: u32, daa: u64) -> bool {
    if is_installed(device_id) {
        return true;
    }
    LOADING.fetch_add(1, Ordering::Relaxed);
    let ok = ensure_installed_inner(device_id, daa);
    LOADING.fetch_sub(1, Ordering::Relaxed);
    ok
}

fn ensure_installed_inner(device_id: u32, daa: u64) -> bool {
    let (model_id, gguf) = match MINING_TIER.get() {
        Some(x) => x,
        None => return false,
    };
    if crate::pom::active_index().is_none() {
        let _guard = match index_build_lock().lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if crate::pom::active_index().is_none() {
            let tier = match crate::models::pom_tier_index(model_id, daa) {
                Some(t) => t,
                None => return false,
            };
            info!("PoM Metal: building shared host weight index (gpu{}) — this can take a while…", device_id);
            match crate::pom::WeightIndex::build_from_gguf(gguf) {
                Ok(idx) => {
                    info!("PoM Metal: shared host index ready — N={} chunks", idx.n_chunks);
                    crate::pom::set_index(idx, tier);
                }
                Err(e) => {
                    log::error!("PoM Metal: shared host index build failed on gpu{}: {}", device_id, e);
                    return false;
                }
            }
        }
    }
    let gm = match PomGpuMiner::load(gguf, device_id as usize) {
        Ok(m) => m,
        Err(e) => {
            log::error!("PoM Metal[gpu{}]: load failed: {}", device_id, e);
            return false;
        }
    };
    let n = gm.n_chunks();
    if let Some((idx, _)) = crate::pom::active_index() {
        if n != idx.n_chunks {
            log::error!(
                "PoM Metal[gpu{}]: packed N={} != shared index N={} — refusing to mine",
                device_id, n, idx.n_chunks
            );
            return false;
        }
    }
    install(device_id, gm);
    info!("PoM Metal[gpu{}]: miner ready — N={} chunks resident (matches shared index)", device_id, n);
    true
}
