//! Proof-of-Model GPU mining — runs the `pom_mine` kernel on the resident weight blob to find a
//! winning nonce. Foundation for the live mining loop (§6/3b).
//!
//! On macOS (Apple Silicon) the PoM kernel runs via a custom Metal compute shader; on CUDA GPUs
//! it runs via the original PTX kernel through candle's CUDA context.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use log::info;

// ── Platform-agnostic parts ──────────────────────────────────────────────────

/// Mining-tier identity for rebuilds: (model_id, gguf_path). Set once at startup.
static MINING_TIER: OnceLock<([u8; 32], String)> = OnceLock::new();

/// Record the mining tier so the miner can be rebuilt after an inference swapped the model away.
pub fn set_mining_tier(model_id: [u8; 32], gguf_path: String) {
    let _ = MINING_TIER.set((model_id, gguf_path));
}

/// PoM tier index of the mining model at a given block DAA.
pub fn current_tier(daa: u64) -> Option<u8> {
    let (model_id, _) = MINING_TIER.get()?;
    crate::models::pom_tier_index(model_id, daa)
}

/// True while the GPU miner is being (re)built — a heavy one-time model load that blocks the
/// mining worker. The PoW stall watchdog treats this like an inference pause, not a crash.
static LOADING: AtomicUsize = AtomicUsize::new(0);

/// Whether a PoM model load/rebuild is in progress (worker intentionally paused, not stalled).
pub fn is_loading() -> bool {
    LOADING.load(Ordering::Relaxed) > 0
}

// ── CUDA implementation ──────────────────────────────────────────────────────

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
use candle_core::quantized::{gguf_file, QTensor};
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
use candle_core::{CudaDevice, Device};

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine.ptx"));
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
const CHUNK_BYTES: usize = 32;

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub struct PomGpuMiner {
    cuda: CudaDevice,
    stream: Arc<CudaStream>,
    bases_dev: CudaSlice<u64>,
    prefix_dev: CudaSlice<u64>,
    t_count: u32,
    n_total_chunks: u64,
    _tensors: Vec<QTensor>,
    _shared: Vec<Arc<QTensor>>,
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl PomGpuMiner {
    pub fn load(gguf_path: &str, device_id: usize) -> candle_core::Result<Self> {
        let device = Device::new_cuda(device_id)?;
        let cuda = match &device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: not a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort();

        let mut tensors: Vec<QTensor> = Vec::with_capacity(names.len());
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        for name in &names {
            let qt = content.tensor(&mut file, name, &device)?;
            let chunks = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u64;
            if chunks == 0 {
                tensors.push(qt);
                continue;
            }
            bases.push(qt.device_ptr()? as usize as u64);
            prefix.push(prefix.last().unwrap() + chunks);
            tensors.push(qt);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: model produced 0 chunks".into()));
        }

        let bases_dev = stream.clone_htod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = stream.clone_htod(&prefix).map_err(candle_core::Error::wrap)?;
        let _ = cuda.get_or_load_custom_func("pom_mine", "pom_mine_mod", PTX)?;

        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: tensors, _shared: Vec::new() })
    }

    pub fn load_shared(
        gguf_path: &str,
        device: &Device,
        shared: &std::collections::HashMap<String, Arc<QTensor>>,
    ) -> candle_core::Result<Self> {
        let cuda = match device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: shared load requires a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort();

        let mut raw: Vec<QTensor> = Vec::new();
        let mut kept_shared: Vec<Arc<QTensor>> = Vec::new();
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        let mut shared_hits = 0usize;
        for name in &names {
            let (ptr, chunks) = if let Some(qt) = shared.get(name) {
                let c = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u64;
                let p = qt.device_ptr()? as usize as u64;
                kept_shared.push(qt.clone());
                shared_hits += 1;
                (p, c)
            } else {
                let qt = content.tensor(&mut file, name, device)?;
                let c = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u64;
                if c == 0 {
                    raw.push(qt);
                    continue;
                }
                let p = qt.device_ptr()? as usize as u64;
                raw.push(qt);
                (p, c)
            };
            if chunks == 0 {
                continue;
            }
            bases.push(ptr);
            prefix.push(prefix.last().unwrap() + chunks);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: shared load produced 0 chunks".into()));
        }
        info!("PoM zero-dup gather: {} shared tensors, {} raw-loaded, N={} chunks", shared_hits, raw.len(), n_total_chunks);

        let bases_dev = stream.clone_htod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = stream.clone_htod(&prefix).map_err(candle_core::Error::wrap)?;
        let _ = cuda.get_or_load_custom_func("pom_mine", "pom_mine_mod", PTX)?;

        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: raw, _shared: kept_shared })
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_total_chunks
    }

    pub fn mine(&self, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64) -> candle_core::Result<Option<u64>> {
        let p = words4(pre_pow_hash);
        let t = words4(target_le);
        let k = crate::pom::POM_WALK_STEPS;
        let winner = self.stream.clone_htod(&[u64::MAX]).map_err(candle_core::Error::wrap)?;
        let grid = ((batch + 255) / 256) as u32;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };

        let func = self.cuda.get_or_load_custom_func("pom_mine", "pom_mine_mod", PTX)?;
        let mut b = func.builder();
        b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&self.n_total_chunks).arg(&k)
            .arg(&p[0]).arg(&p[1]).arg(&p[2]).arg(&p[3]).arg(&timestamp)
            .arg(&t[0]).arg(&t[1]).arg(&t[2]).arg(&t[3])
            .arg(&start).arg(&batch).arg(&winner);
        unsafe { b.launch(cfg).map_err(candle_core::Error::wrap)?; }
        self.stream.synchronize().map_err(candle_core::Error::wrap)?;

        let w = self.stream.clone_dtoh(&winner).map_err(candle_core::Error::wrap)?[0];
        Ok(if w == u64::MAX { None } else { Some(w) })
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn miners() -> &'static Mutex<HashMap<u32, Arc<PomGpuMiner>>> {
    static MINERS: OnceLock<Mutex<HashMap<u32, Arc<PomGpuMiner>>>> = OnceLock::new();
    MINERS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn index_build_lock() -> &'static Mutex<()> {
    static INDEX_BUILD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    INDEX_BUILD_LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn install(device_id: u32, m: PomGpuMiner) {
    if let Ok(mut g) = miners().lock() {
        g.insert(device_id, Arc::new(m));
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn remove_device_entry<T>(map: &mut HashMap<u32, T>, device_id: u32) {
    map.remove(&device_id);
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn uninstall(device_id: u32) {
    if let Ok(mut g) = miners().lock() {
        remove_device_entry(&mut g, device_id);
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn is_installed(device_id: u32) -> bool {
    miners().lock().map(|g| g.contains_key(&device_id)).unwrap_or(false)
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn mine(device_id: u32, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64) -> Option<u64> {
    let miner = {
        let g = miners().lock().ok()?;
        g.get(&device_id)?.clone()
    };
    miner.mine(pre_pow_hash, timestamp, target_le, start, batch).ok().flatten()
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn ensure_installed(device_id: u32, daa: u64) -> bool {
    if is_installed(device_id) {
        return true;
    }
    LOADING.fetch_add(1, Ordering::Relaxed);
    let ok = ensure_installed_inner(device_id, daa);
    LOADING.fetch_sub(1, Ordering::Relaxed);
    ok
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn cuda_gpu_id(d: &Device) -> Option<usize> {
    match d.location() {
        candle_core::DeviceLocation::Cuda { gpu_id } => Some(gpu_id),
        _ => None,
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
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
            info!("PoM: building shared host weight index (gpu{}) — this can take a while…", device_id);
            match crate::pom::WeightIndex::build_from_gguf(gguf) {
                Ok(idx) => {
                    info!("PoM: shared host index ready — N={} chunks", idx.n_chunks);
                    crate::pom::set_index(idx, tier);
                }
                Err(e) => {
                    log::error!("PoM: shared host index build failed on gpu{}: {}", device_id, e);
                    return false;
                }
            }
        }
    }
    let m = match crate::slm::pom_shared(model_id) {
        Some((inf_dev, shared)) if cuda_gpu_id(&inf_dev) == Some(device_id as usize) => {
            info!("PoM[gpu{}]: zero-dup — sharing the inference engine's resident weights (no 2nd VRAM copy)", device_id);
            PomGpuMiner::load_shared(gguf, &inf_dev, &shared)
        }
        _ => PomGpuMiner::load(gguf, device_id as usize),
    };
    match m {
        Ok(gm) => {
            let n = gm.n_chunks();
            if let Some((idx, _)) = crate::pom::active_index() {
                if n != idx.n_chunks {
                    log::error!("PoM[gpu{}]: gather N={} != shared index N={} — refusing to mine", device_id, n, idx.n_chunks);
                    return false;
                }
            }
            install(device_id, gm);
            info!("PoM[gpu{}]: GPU miner ready — N={} chunks resident (matches shared index)", device_id, n);
            true
        }
        Err(e) => {
            log::error!("PoM[gpu{}]: device miner build failed: {}", device_id, e);
            false
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remove_device_entry_only_clears_target_device() {
        let mut map: HashMap<u32, &str> = HashMap::new();
        map.insert(0, "gpu0-miner");
        map.insert(1, "gpu1-miner");
        map.insert(2, "gpu2-miner");
        remove_device_entry(&mut map, 0);
        assert!(!map.contains_key(&0));
        assert_eq!(map.get(&1), Some(&"gpu1-miner"));
        assert_eq!(map.get(&2), Some(&"gpu2-miner"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn remove_device_entry_on_missing_device_is_a_no_op() {
        let mut map: HashMap<u32, &str> = HashMap::new();
        map.insert(1, "gpu1-miner");
        remove_device_entry(&mut map, 0);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"gpu1-miner"));
    }
}

// ── Metal implementation (Apple Silicon) ──────────────────────────────────

#[cfg(any(target_os = "macos", target_os = "ios"))]
use candle_core::quantized::gguf_file;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use candle_core::Device;

#[cfg(any(target_os = "macos", target_os = "ios"))]
const MSL_SOURCE: &str = include_str!("../cuda/pom_mine.metal");
#[cfg(any(target_os = "macos", target_os = "ios"))]
const CHUNK_BYTES: usize = 32;

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

/// Parameter block matching the `PomParams` struct in the Metal shader. Packed via
/// `encoder.set_bytes` so that each dispatch carries its own nonce range and target.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[allow(non_snake_case)]
#[repr(C)]
struct PomParams {
    T: u32,
    K: u32,
    n_total_chunks: u64,
    p0: u64, p1: u64, p2: u64, p3: u64,
    time_: u64,
    t0: u64, t1: u64, t2: u64, t3: u64,
    nonce_base: u64,
    n_nonces: u64,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[allow(dead_code)]
pub struct PomGpuMiner {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
    all_data: metal::Buffer,
    base_offsets_buf: metal::Buffer,
    prefix_buf: metal::Buffer,
    winner_buf: metal::Buffer,
    t_count: u32,
    n_total_chunks: u64,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl PomGpuMiner {
    pub fn load(gguf_path: &str, _device_id: usize) -> candle_core::Result<Self> {
        let device = metal_device()?;
        let queue = device.new_command_queue();

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort();

        let cpu = Device::Cpu;

        // Pass 1: measure each tensor's byte length (dropped immediately after) to size
        // the Metal buffer up front. Previously this accumulated every tensor into one
        // CPU-side `all_data: Vec<u8>` (~1 GB for a 1.7B model) before a single GPU
        // upload — for a moment both the full Vec AND the freshly-copied Metal buffer
        // were resident simultaneously (~2 GB peak), which on an iPhone's tighter memory
        // budget got the app killed by the OS right after "got block template" (this is
        // the first time ensure_installed's model load runs). Streaming tensor-by-tensor
        // below keeps only one tensor's bytes live at a time alongside the single
        // full-size Metal buffer.
        let mut base_offsets: Vec<u64> = Vec::with_capacity(names.len());
        let mut prefix: Vec<u64> = vec![0];
        let mut tensor_lens: Vec<usize> = Vec::with_capacity(names.len());
        let mut total_bytes: u64 = 0;
        for name in &names {
            let qt = content.tensor(&mut file, name, &cpu)?;
            let bytes = qt.data()?;
            let n = bytes.len();
            tensor_lens.push(n);
            let chunks = n / CHUNK_BYTES;
            if chunks == 0 {
                continue;
            }
            base_offsets.push(total_bytes);
            total_bytes += n as u64;
            prefix.push(prefix.last().unwrap() + chunks as u64);
        }

        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: model produced 0 chunks".into()));
        }
        let t_count = base_offsets.len() as u32;

        let opts = metal::MTLResourceOptions::StorageModeShared;
        let all_data_buf = device.new_buffer(total_bytes, opts);
        let dst = all_data_buf.contents() as *mut u8;

        // Pass 2: re-read each tensor and copy its bytes directly into the Metal buffer
        // at its offset. Re-reading from disk is a one-time cost per app launch (only
        // runs the first time a block template arrives) and is worth halving peak RAM.
        let mut write_off: u64 = 0;
        for (name, &n) in names.iter().zip(tensor_lens.iter()) {
            if n / CHUNK_BYTES == 0 {
                continue;
            }
            let qt = content.tensor(&mut file, name, &cpu)?;
            let bytes = qt.data()?;
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.add(write_off as usize), bytes.len());
            }
            write_off += n as u64;
        }

        let base_offsets_buf = device.new_buffer_with_data(
            base_offsets.as_ptr() as *const std::ffi::c_void,
            (base_offsets.len() * std::mem::size_of::<u64>()) as u64,
            opts,
        );

        let prefix_buf = device.new_buffer_with_data(
            prefix.as_ptr() as *const std::ffi::c_void,
            (prefix.len() * std::mem::size_of::<u64>()) as u64,
            opts,
        );

        let winner_buf = device.new_buffer(std::mem::size_of::<u64>() as u64, opts);
        unsafe { *(winner_buf.contents() as *mut u64) = u64::MAX; }

        let library = compile_metal_library(&device)?;
        let function = library
            .get_function("pom_mine", None)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: kernel 'pom_mine' not found: {e}")))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: pipeline build failed: {e}")))?;

        Ok(Self {
            device,
            queue,
            pipeline,
            all_data: all_data_buf,
            base_offsets_buf,
            prefix_buf,
            winner_buf,
            t_count,
            n_total_chunks,
        })
    }

    pub fn load_shared(
        gguf_path: &str,
        _device: &Device,
        _shared: &std::collections::HashMap<String, std::sync::Arc<candle_core::quantized::QTensor>>,
    ) -> candle_core::Result<Self> {
        log::info!("PoM Metal: zero-dup sharing not available (Metal buffer not accessible via candle); loading standalone");
        Self::load(gguf_path, 0)
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_total_chunks
    }

    pub fn mine(
        &self,
        pre_pow_hash: &[u8; 32],
        timestamp: u64,
        target_le: &[u8; 32],
        start: u64,
        batch: u64,
    ) -> candle_core::Result<Option<u64>> {
        let p = words4(pre_pow_hash);
        let t = words4(target_le);
        let k = crate::pom::POM_WALK_STEPS;

        let params = PomParams {
            T: self.t_count,
            K: k,
            n_total_chunks: self.n_total_chunks,
            p0: p[0], p1: p[1], p2: p[2], p3: p[3],
            time_: timestamp,
            t0: t[0], t1: t[1], t2: t[2], t3: t[3],
            nonce_base: start,
            n_nonces: batch,
        };

        unsafe { *(self.winner_buf.contents() as *mut u64) = u64::MAX; }

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.all_data), 0);
        encoder.set_buffer(1, Some(&self.base_offsets_buf), 0);
        encoder.set_buffer(2, Some(&self.prefix_buf), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<PomParams>() as u64,
            &params as *const PomParams as *const std::ffi::c_void,
        );
        encoder.set_buffer(4, Some(&self.winner_buf), 0);

        let wgs = 256u64;
        let wg_count = ((batch + wgs - 1) / wgs) as u64;
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(wg_count, 1, 1),
            metal::MTLSize::new(wgs, 1, 1),
        );

        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let w = unsafe { *(self.winner_buf.contents() as *const u64) };
        Ok(if w == u64::MAX { None } else { Some(w) })
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn metal_device() -> candle_core::Result<metal::Device> {
    metal::Device::system_default()
        .ok_or_else(|| candle_core::Error::Msg("PoM Metal: no system default Metal device".into()))
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn compile_metal_library(device: &metal::Device) -> candle_core::Result<metal::Library> {
    let opts = metal::CompileOptions::new();
    // Disable fast-math to match the CUDA kernel's IEEE-754 compliant arithmetic.
    opts.set_fast_math_enabled(false);
    device
        .new_library_with_source(MSL_SOURCE, &opts)
        .map_err(|e| candle_core::Error::Msg(format!("PoM Metal: shader compilation failed: {e}")))
}

// ── Miners registry (Metal backend) ────────────────────────────────────────

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn miners() -> &'static std::sync::Mutex<std::collections::HashMap<u32, std::sync::Arc<PomGpuMiner>>> {
    static MINERS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<u32, std::sync::Arc<PomGpuMiner>>>> =
        std::sync::OnceLock::new();
    MINERS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn remove_device_entry<T>(map: &mut std::collections::HashMap<u32, T>, device_id: u32) {
    map.remove(&device_id);
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn index_build_lock() -> &'static std::sync::Mutex<()> {
    static INDEX_BUILD_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    INDEX_BUILD_LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn install(device_id: u32, m: PomGpuMiner) {
    if let Ok(mut g) = miners().lock() {
        g.insert(device_id, std::sync::Arc::new(m));
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn uninstall(device_id: u32) {
    if let Ok(mut g) = miners().lock() {
        remove_device_entry(&mut g, device_id);
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn is_installed(device_id: u32) -> bool {
    miners()
        .lock()
        .map(|g| g.contains_key(&device_id))
        .unwrap_or(false)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
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

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn ensure_installed(device_id: u32, daa: u64) -> bool {
    if is_installed(device_id) {
        return true;
    }
    LOADING.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let ok = ensure_installed_inner(device_id, daa);
    LOADING.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    ok
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
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
            info!(
                "PoM: building shared host weight index (metal{}) — this can take a while…",
                device_id
            );
            match crate::pom::WeightIndex::build_from_gguf(gguf) {
                Ok(idx) => {
                    info!("PoM: shared host index ready — N={} chunks", idx.n_chunks);
                    crate::pom::set_index(idx, tier);
                }
                Err(e) => {
                    log::error!("PoM: shared host index build failed on metal{}: {}", device_id, e);
                    return false;
                }
            }
        }
    }
    let m = PomGpuMiner::load(gguf, device_id as usize);
    match m {
        Ok(gm) => {
            let n = gm.n_chunks();
            if let Some((idx, _)) = crate::pom::active_index() {
                if n != idx.n_chunks {
                    log::error!(
                        "PoM[metal{}]: gather N={} != shared index N={} — refusing to mine",
                        device_id, n, idx.n_chunks
                    );
                    return false;
                }
            }
            install(device_id, gm);
            info!(
                "PoM[metal{}]: GPU miner ready — N={} chunks resident (matches shared index)",
                device_id, n
            );
            true
        }
        Err(e) => {
            log::error!("PoM[metal{}]: device miner build failed: {}", device_id, e);
            false
        }
    }
}
