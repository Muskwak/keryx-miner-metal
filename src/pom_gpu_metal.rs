//! Apple Silicon (Metal) PoM GPU miner — parity path for `pom_gpu.rs` (CUDA).
//!
//! Same public free-function surface as the CUDA module (`install`/`uninstall`/`is_installed`/
//! `is_loading`/`mine`/`current_tier`/`ensure_installed`/`set_mining_tier`) so callers in
//! main.rs / miner.rs / slm.rs are backend-agnostic. Under the hood this is a **zero-dup**
//! walk over candle's own resident `MTLBuffer`s: no packed weight blob, no host copy of the
//! quantized bytes. Instead we build two small side tables once at load time:
//!
//!   * `prefix`  — cumulative 32-byte-chunk count per tensor in canonical (name-sorted)
//!     GGUF order, length `n_tensors + 1`. Matches the layout `pom-rt-builder` / the node's
//!     `R_T` root are built over, so a global chunk index is the same address here and there.
//!   * `addrs`   — `MTLBuffer.gpuAddress()` per tensor. In Metal 3 (Apple Silicon is always
//!     Tier-2 argument buffers) these are plain 64-bit pointers the kernel reinterprets to
//!     `device const ulong*`. We call `use_resource` on each tensor buffer before dispatch
//!     so the driver keeps them GPU-resident even though nothing else binds them.
//!
//! The MTLBuffer handles are the very ones candle-core holds inside `QMetalStorage`; we get
//! at them through the vendored `QTensor::metal_storage()` accessor added in
//! `vendor/candle-core/src/quantized/mod.rs`. Cloning a `Buffer` just bumps the objc2 retain
//! count — no data is copied.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use log::info;

use candle_core::quantized::gguf_file;
use candle_core::Device;
use candle_metal_kernels::metal::{
    create_command_buffer, Buffer, CommandQueue, CommandSemaphore, ComputePipeline,
    Device as MtlDevice, MTLResourceOptions,
};
use objc2_metal::{
    MTLBuffer as _, MTLDevice as _, MTLResourceOptions as ObjcMTLResourceOptions, MTLResourceUsage,
    MTLSize,
};

const METAL_SRC: &str = include_str!("../metal/pom_mine.metal");
const CHUNK_BYTES: usize = 32;
const THREADGROUP_SIZE: usize = 256;

/// Shared-storage buffers: CPU and GPU see the same unified-memory backing, so the host can
/// write uniforms / read the winner without a blit copy. Same choice candle uses for its
/// own transient buffers.
const SHARED_STORAGE: MTLResourceOptions =
    ObjcMTLResourceOptions(ObjcMTLResourceOptions::StorageModeShared.bits());

pub struct PomGpuMiner {
    device: MtlDevice,
    queue: CommandQueue,
    pipeline: ComputePipeline,
    /// Prefix sums of chunk counts across tensors, length `n_tensors + 1`, in chunks.
    prefix_buf: Buffer,
    /// GPU addresses of the resident per-tensor MTLBuffers, length `n_tensors`.
    addrs_buf: Buffer,
    /// Clones of the candle-owned per-tensor MTLBuffers. Kept for two reasons:
    ///   (a) they hold the retain count that keeps the unified-memory backing alive, and
    ///   (b) we hand them to `use_resource` on every dispatch so the driver marks them
    ///       resident even though the kernel binds only the addrs/prefix tables.
    resources: Vec<Buffer>,
    n_total_chunks: u64,
    n_tensors: u32,
}

// Matches PomUniforms in metal/pom_mine.metal — field order and padding are load-bearing.
#[repr(C)]
struct Uniforms {
    n_total_chunks: u64,
    k_steps: u32,
    n_tensors: u32,
    p0: u64, p1: u64, p2: u64, p3: u64,
    time_: u64,
    t0: u64, t1: u64, t2: u64, t3: u64,
    nonce_base: u64,
    n_nonces: u32,
    _pad: u32,
}

impl PomGpuMiner {
    /// Load the mining model's GGUF into a candle Metal device and build the bindless walk
    /// tables. Heavy — call once per (device, model).
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

        let mut resources: Vec<Buffer> = Vec::with_capacity(names.len());
        let mut addrs: Vec<u64> = Vec::with_capacity(names.len());
        let mut prefix: Vec<u64> = Vec::with_capacity(names.len() + 1);
        prefix.push(0);
        let mut cum: u64 = 0;

        for name in &names {
            let qt = content.tensor(&mut file, name, &cdev)?;
            let n_bytes = qt.storage_size_in_bytes();
            if n_bytes < CHUNK_BYTES {
                // Skip tiny tensors (biases, norms, etc.) — same behaviour as the CUDA gather.
                continue;
            }
            let qmet = qt.metal_storage().ok_or_else(|| {
                candle_core::Error::Msg("PoM Metal: QTensor has no Metal storage".into())
            })?;
            let buf: Buffer = qmet.buffer().clone();
            let addr = buf.as_ref().gpuAddress();
            let n_chunks = (n_bytes / CHUNK_BYTES) as u64;
            cum += n_chunks;
            prefix.push(cum);
            addrs.push(addr);
            resources.push(buf);
        }
        let n_total_chunks = cum;
        let n_tensors = resources.len() as u32;
        if n_total_chunks == 0 || n_tensors == 0 {
            return Err(candle_core::Error::Msg("PoM Metal: model produced 0 chunks".into()));
        }

        let prefix_buf = mdev
            .new_buffer_with_data(
                prefix.as_ptr() as *const _,
                std::mem::size_of_val(&prefix[..]),
                SHARED_STORAGE,
            )
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: prefix buffer: {e}")))?;
        let addrs_buf = mdev
            .new_buffer_with_data(
                addrs.as_ptr() as *const _,
                std::mem::size_of_val(&addrs[..]),
                SHARED_STORAGE,
            )
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: addrs buffer: {e}")))?;

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
            "PoM Metal: {} tensors, {} chunks (~{} MiB) resident on device {} (zero-dup)",
            n_tensors,
            n_total_chunks,
            (n_total_chunks as usize * CHUNK_BYTES) / (1024 * 1024),
            device_id
        );

        Ok(Self {
            device: mdev,
            queue,
            pipeline,
            prefix_buf,
            addrs_buf,
            resources,
            n_total_chunks,
            n_tensors,
        })
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
            n_tensors: self.n_tensors,
            p0: p[0], p1: p[1], p2: p[2], p3: p[3],
            time_: timestamp,
            t0: t[0], t1: t[1], t2: t[2], t3: t[3],
            nonce_base: start,
            n_nonces: batch as u32,
            _pad: 0,
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
        enc.set_buffer(0, Some(&self.prefix_buf), 0);
        enc.set_buffer(1, Some(&self.addrs_buf), 0);
        enc.set_buffer(2, Some(&uniforms_buf), 0);
        enc.set_buffer(3, Some(&winner_buf), 0);
        // Bindless: the resident per-tensor buffers are dereffed via raw gpuAddress inside
        // the kernel, so nothing here binds them to a slot. We must still tell the driver
        // they'll be read — otherwise it can page them out or omit residency tracking.
        for buf in &self.resources {
            enc.use_resource(buf, MTLResourceUsage::Read);
        }
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

/// Total usable GPU memory (MB) of every Metal device, in candle's `Device::new_metal` order — the
/// Metal analogue of the CUDA path's `query_all_gpus_vram`, so an entry `(id, mb)` is the budget of
/// the device the miner would mine/serve on for that `id`. Apple Silicon exposes a single
/// unified-memory GPU (id 0). Sourced from Metal's `recommendedMaxWorkingSetSize` — the driver's own
/// budget for resident resources (≈75% of unified RAM) rather than total RAM — so it matches what
/// candle/PoM can actually keep resident before eviction, the role CUDA's per-device `total_mem`
/// plays on a discrete card. Returns an empty vec when no Metal device is present. Never panics — a
/// device-open failure is caught and treated as "no device".
pub fn query_all_gpus_vram() -> Vec<(usize, u64)> {
    std::panic::catch_unwind(|| {
        let cdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        let bytes = match &cdev {
            // `recommendedMaxWorkingSetSize` is an MTLDevice property (bytes); reach the objc2
            // protocol object through candle's wrapper the same way the gather reads
            // `buffer.as_ref().gpuAddress()` in `PomGpuMiner::load`.
            Device::Metal(m) => m.metal_device().as_ref().recommendedMaxWorkingSetSize(),
            _ => return Vec::new(),
        };
        vec![(0usize, bytes / (1024 * 1024))]
    })
    .unwrap_or_default()
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

/// Per-device mining-tier identity for rebuilds: `device_id -> (model_id, gguf_path)`. Apple Silicon
/// is single-GPU (device 0), but the map is keyed by device to stay signature-compatible with the
/// CUDA path (where a heterogeneous rig mines a different tier per GPU), so main.rs/miner.rs/slm.rs
/// call the same free functions on both backends.
static MINING_TIERS: OnceLock<Mutex<HashMap<u32, ([u8; 32], String)>>> = OnceLock::new();

fn mining_tiers() -> &'static Mutex<HashMap<u32, ([u8; 32], String)>> {
    MINING_TIERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record a device's mining tier so its miner can be rebuilt after an inference swapped the model away.
pub fn set_mining_tier(device_id: u32, model_id: [u8; 32], gguf_path: String) {
    if let Ok(mut g) = mining_tiers().lock() {
        g.insert(device_id, (model_id, gguf_path));
    }
}

/// PoM tier index of a device's mining model at a given block DAA. Recomputed per block (not frozen at
/// index-build time) so the tier reindexing at the very-light hardfork (H2) is applied at the exact
/// boundary rather than from a stale build-time value.
pub fn current_tier(device_id: u32, daa: u64) -> Option<u8> {
    let model_id = mining_tiers().lock().ok()?.get(&device_id).map(|(id, _)| *id)?;
    crate::models::pom_tier_index(&model_id, daa)
}

/// The Metal device that mines `model_id` (from the per-device tier assignment), if any — the Metal
/// peer of the CUDA `device_for_model`. Inference is routed to the device already holding the model so
/// only that device pauses mining. Returns the lowest matching `device_id`; `None` when nothing mines
/// it. On Apple Silicon this is device 0 whenever a tier is assigned.
pub fn device_for_model(model_id: &[u8; 32]) -> Option<u32> {
    let g = mining_tiers().lock().ok()?;
    g.iter().filter(|(_, (id, _))| id == model_id).map(|(dev, _)| *dev).min()
}

/// Models that failed to load on a given device: `(device_id, model_id)`. Once banlisted, that device
/// never retries that model (avoids a hot-spin reloading a model that doesn't fit); the OOM handler
/// downgrades to a smaller downloaded tier instead. On Apple Silicon an "OOM" is a unified-memory
/// allocation failure — real on a small Mac loading a large tier alongside the resident inference engine.
static OOM_BANLIST: OnceLock<Mutex<HashSet<(u32, [u8; 32])>>> = OnceLock::new();

fn oom_banlist() -> &'static Mutex<HashSet<(u32, [u8; 32])>> {
    OOM_BANLIST.get_or_init(|| Mutex::new(HashSet::new()))
}

fn is_oom_banlisted(device_id: u32, model_id: &[u8; 32]) -> bool {
    oom_banlist().lock().map(|g| g.contains(&(device_id, *model_id))).unwrap_or(false)
}

fn oom_banlist_add(device_id: u32, model_id: [u8; 32]) {
    if let Ok(mut g) = oom_banlist().lock() {
        g.insert((device_id, model_id));
    }
}

/// After a device fails to load its assigned tier (OOM), reassign it to the largest
/// **already-downloaded** PoM model strictly smaller than the failed one not itself banlisted here — so
/// a Mac whose unified-memory budget was optimistic mines a smaller tier instead of idling. The Metal
/// peer of the CUDA `downgrade_after_oom`. Returns true if a downgrade was applied; no extra prefetch
/// is needed since the candidate set is the already-downloaded served union.
fn downgrade_after_oom(device_id: u32, failed_model: &[u8; 32], daa: u64) -> bool {
    let Some(failed_tier) = crate::models::pom_tier_index(failed_model, daa) else {
        return false;
    };
    let pick = crate::slm::served_pom_specs()
        .into_iter()
        .filter_map(|s| crate::models::pom_tier_index(&s.model_id, daa).map(|t| (t, s)))
        .filter(|(t, s)| *t < failed_tier && !is_oom_banlisted(device_id, &s.model_id))
        .max_by_key(|(t, _)| *t);
    match pick {
        Some((tier, spec)) => {
            let gguf = crate::slm::gguf_path_for(spec).to_string_lossy().into_owned();
            info!("PoM Metal[gpu{}]: OOM on tier {} — downgrading to tier {} ({}).", device_id, failed_tier, tier, spec.name);
            set_mining_tier(device_id, spec.model_id, gguf);
            true
        }
        None => {
            log::warn!("PoM Metal[gpu{}]: OOM and no smaller downloaded tier available — this device will not mine PoM (lower the tier flag or add RAM).", device_id);
            false
        }
    }
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
    let (model_id, gguf) = match mining_tiers().lock().ok().and_then(|g| g.get(&device_id).cloned()) {
        Some(x) => x,
        None => return false,
    };
    // This device's tier at the current block DAA (recomputed per block, H2-gated).
    let tier = match crate::models::pom_tier_index(&model_id, daa) {
        Some(t) => t,
        None => return false,
    };
    if is_oom_banlisted(device_id, &model_id) {
        return false; // this model OOM'd on this device before — don't retry (avoids a hot reload spin).
    }
    // Build THIS tier's possession index once (host, heavy) — deferred from boot so the pre-PoM legacy
    // phase starts immediately, and keyed by tier so a rig mining several tiers builds one index per
    // distinct tier (shared across every device on that tier).
    if crate::pom::active_index_for_tier(tier).is_none() {
        let _guard = match index_build_lock().lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if crate::pom::active_index_for_tier(tier).is_none() {
            info!("PoM Metal: building host weight index for tier {} (gpu{}) — this can take a while…", tier, device_id);
            match crate::pom::WeightIndex::build_from_gguf(&gguf) {
                Ok(idx) => {
                    info!("PoM Metal: tier {} host index ready — N={} chunks", tier, idx.n_chunks);
                    crate::pom::set_index(tier, idx);
                }
                Err(e) => {
                    log::error!("PoM Metal: host index build failed for tier {} on gpu{}: {}", tier, device_id, e);
                    return false;
                }
            }
        }
    }
    // Load the Metal miner (a bindless walk over candle's resident MTLBuffers). A unified-memory
    // allocation failure surfaces as an Err — or, defensively, a panic; catch both so the OOM handler
    // can banlist + downgrade instead of crashing the mining thread or hot-spinning on a model that
    // doesn't fit this device.
    let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        PomGpuMiner::load(&gguf, device_id as usize)
    }));
    let gm = match loaded {
        Ok(Ok(gm)) => gm,
        Ok(Err(e)) => {
            log::error!("PoM Metal[gpu{}]: device miner build failed: {} — banlisting this model and downgrading.", device_id, e);
            oom_banlist_add(device_id, model_id);
            downgrade_after_oom(device_id, &model_id, daa);
            return false;
        }
        Err(_) => {
            log::error!("PoM Metal[gpu{}]: device miner load panicked (likely OOM) — banlisting this model and downgrading.", device_id);
            oom_banlist_add(device_id, model_id);
            downgrade_after_oom(device_id, &model_id, daa);
            return false;
        }
    };
    let n = gm.n_chunks();
    // N-guard: the gather must match the host index, else blocks would be rejected.
    if let Some(idx) = crate::pom::active_index_for_tier(tier) {
        if n != idx.n_chunks {
            log::error!("PoM Metal[gpu{}]: resident N={} != tier {} index N={} — refusing to mine", device_id, n, tier, idx.n_chunks);
            return false;
        }
    }
    install(device_id, gm);
    info!("PoM Metal[gpu{}]: miner ready — N={} chunks resident (matches shared index)", device_id, n);
    true
}
