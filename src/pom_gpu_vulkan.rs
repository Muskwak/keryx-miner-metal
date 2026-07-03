//! Android (Adreno/Mali) PoM GPU miner — Vulkan backend.
//!
//! Same public free-function surface as the CUDA (`pom_gpu.rs`) and Metal (`pom_gpu_metal.rs`)
//! backends (`install`/`uninstall`/`is_installed`/`is_loading`/`mine`/`current_tier`/
//! `ensure_installed`/`set_mining_tier`/`device_for_model`) so `android.rs` calls `crate::pom_gpu`
//! exactly like `ios.rs` does — the only difference is which module `lib.rs` aliases that name to.
//!
//! Unlike the Metal backend's zero-dup walk (which borrows candle's own resident `MTLBuffer`s),
//! this backend packs the GGUF's quantized tensors into a plain `Vec<u64>` host blob (via
//! `candle_core`'s CPU device, canonical name-sorted order — identical layout to
//! [`crate::pom::WeightIndex`]) and uploads it to VRAM through `keryx_vulkan::pom_walk::PomWalkGpu`.
//! candle has no Vulkan backend, so there is no resident GPU tensor to borrow here the way there is
//! on Metal — this is the same approach the desktop Vulkan (RDNA3) fork uses.
//!
//! Mobile-specific: `keryx_vulkan`'s weight blob is split into power-of-two shards sized well
//! under the device's real `maxMemoryAllocationSize`, queried at runtime — desktop AMD's ~2 GiB
//! single-allocation cap doesn't hold on a phone GPU, which can report far less.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::quantized::gguf_file;
use candle_core::Device;
use keryx_vulkan::pom_walk::PomWalkGpu;
use log::info;

fn miners() -> &'static Mutex<HashMap<u32, Arc<PomWalkGpu>>> {
    static MINERS: OnceLock<Mutex<HashMap<u32, Arc<PomWalkGpu>>>> = OnceLock::new();
    MINERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn index_build_lock() -> &'static Mutex<()> {
    static INDEX_BUILD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    INDEX_BUILD_LOCK.get_or_init(|| Mutex::new(()))
}

pub fn install(device_id: u32, m: PomWalkGpu) {
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
    let batch = batch.min(u32::MAX as u64) as u32;
    miner.mine(pre_pow_hash, timestamp, target_le, start, batch)
}

/// Per-device mining-tier identity for rebuilds: `device_id -> (model_id, gguf_path)`. Android is
/// single-GPU (device 0), but keyed by device to stay signature-compatible with the other backends.
static MINING_TIERS: OnceLock<Mutex<HashMap<u32, ([u8; 32], String)>>> = OnceLock::new();

fn mining_tiers() -> &'static Mutex<HashMap<u32, ([u8; 32], String)>> {
    MINING_TIERS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn set_mining_tier(device_id: u32, model_id: [u8; 32], gguf_path: String) {
    if let Ok(mut g) = mining_tiers().lock() {
        g.insert(device_id, (model_id, gguf_path));
    }
}

/// PoM tier index of a device's mining model at a given block DAA. Recomputed per block (not
/// frozen at index-build time) so the tier reindexing at the very-light hardfork (H2) is applied
/// at the exact boundary rather than from a stale build-time value.
pub fn current_tier(device_id: u32, daa: u64) -> Option<u8> {
    let model_id = mining_tiers().lock().ok()?.get(&device_id).map(|(id, _)| *id)?;
    crate::models::pom_tier_index(&model_id, daa)
}

pub fn device_for_model(model_id: &[u8; 32]) -> Option<u32> {
    let g = mining_tiers().lock().ok()?;
    g.iter().filter(|(_, (id, _))| id == model_id).map(|(dev, _)| *dev).min()
}

/// Models that failed to load on a given device: `(device_id, model_id) -> reason`. Once
/// banlisted, that device never retries that model — avoids a hot-spin reloading a model that
/// doesn't fit; the allocation-failure handler downgrades to a smaller downloaded tier instead.
/// Keeping the reason (not just a `HashSet` membership bit) lets every retry re-log why, since the
/// mining loop retries every 500ms and the original failure otherwise scrolls out of the on-screen
/// log's last-20-lines window almost immediately.
static OOM_BANLIST: OnceLock<Mutex<HashMap<(u32, [u8; 32]), String>>> = OnceLock::new();

fn oom_banlist() -> &'static Mutex<HashMap<(u32, [u8; 32]), String>> {
    OOM_BANLIST.get_or_init(|| Mutex::new(HashMap::new()))
}

fn is_oom_banlisted(device_id: u32, model_id: &[u8; 32]) -> bool {
    oom_banlist().lock().map(|g| g.contains_key(&(device_id, *model_id))).unwrap_or(false)
}

fn oom_banlist_add(device_id: u32, model_id: [u8; 32], reason: String) {
    if let Ok(mut g) = oom_banlist().lock() {
        g.insert((device_id, model_id), reason);
    }
}

/// After a device fails to load its assigned tier (allocation failure), reassign it to the largest
/// **already-downloaded** PoM model strictly smaller than the failed one and not itself banlisted —
/// so a phone whose GPU memory budget was optimistic mines a smaller tier instead of idling.
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
            info!(
                "PoM(vulkan)[gpu{}]: allocation failure on tier {} — downgrading to tier {} ({}).",
                device_id, failed_tier, tier, spec.name
            );
            set_mining_tier(device_id, spec.model_id, gguf);
            true
        }
        None => {
            log::warn!(
                "PoM(vulkan)[gpu{}]: allocation failure and no smaller downloaded tier available — \
                 this device will not mine PoM.",
                device_id
            );
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
    let tier = match crate::models::pom_tier_index(&model_id, daa) {
        Some(t) => t,
        None => return false,
    };
    if is_oom_banlisted(device_id, &model_id) {
        if let Ok(g) = oom_banlist().lock() {
            if let Some(reason) = g.get(&(device_id, model_id)) {
                log::error!(
                    "PoM(vulkan)[gpu{}]: model already banlisted, not retrying — {}",
                    device_id, reason
                );
            }
        }
        return false;
    }
    // Build this tier's possession index once (host, heavy) — shared across every device mining
    // that tier, matching the CUDA/Metal backends' bookkeeping.
    if crate::pom::active_index_for_tier(tier).is_none() {
        let _guard = match index_build_lock().lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if crate::pom::active_index_for_tier(tier).is_none() {
            info!("PoM(vulkan): building host weight index for tier {} (gpu{}) — this can take a while…", tier, device_id);
            match crate::pom::WeightIndex::build_from_gguf(&gguf) {
                Ok(idx) => {
                    info!("PoM(vulkan): tier {} host index ready — N={} chunks", tier, idx.n_chunks);
                    crate::pom::set_index(tier, idx);
                }
                Err(e) => {
                    log::error!("PoM(vulkan): host index build failed for tier {} on gpu{}: {}", tier, device_id, e);
                    return false;
                }
            }
        }
    }

    info!("PoM(vulkan)[gpu{}]: loading weight blob…", device_id);
    let (words, n_chunks) = match load_weight_words(&gguf) {
        Ok(v) => v,
        Err(e) => {
            log::error!("PoM(vulkan)[gpu{}]: weight blob load failed: {}", device_id, e);
            return false;
        }
    };

    // N-guard: the gather must match the host index, else blocks would be rejected.
    if let Some(idx) = crate::pom::active_index_for_tier(tier) {
        if n_chunks != idx.n_chunks {
            log::error!(
                "PoM(vulkan)[gpu{}]: blob N={} != tier {} index N={} — refusing to mine",
                device_id, n_chunks, tier, idx.n_chunks
            );
            return false;
        }
    }

    let shard_chunks = pick_shard_chunks(n_chunks);
    let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        PomWalkGpu::new_sharded(&words, n_chunks, shard_chunks)
    }));
    let gm = match loaded {
        Ok(Ok(gm)) => gm,
        Ok(Err(e)) => {
            log::error!(
                "PoM(vulkan)[gpu{}]: device miner build failed: {} — banlisting this model and downgrading.",
                device_id, e
            );
            oom_banlist_add(device_id, model_id, format!("device miner build failed: {e}"));
            downgrade_after_oom(device_id, &model_id, daa);
            return false;
        }
        Err(_) => {
            log::error!(
                "PoM(vulkan)[gpu{}]: device miner load panicked (likely an allocation failure) — \
                 banlisting this model and downgrading.",
                device_id
            );
            oom_banlist_add(
                device_id,
                model_id,
                "device miner load panicked (likely an allocation failure)".to_string(),
            );
            downgrade_after_oom(device_id, &model_id, daa);
            return false;
        }
    };
    info!(
        "PoM(vulkan)[gpu{}]: miner ready on {} — N={} chunks resident (shard={} chunks)",
        device_id,
        gm.device_name(),
        n_chunks,
        shard_chunks
    );
    install(device_id, gm);
    true
}

/// Read the GGUF's quantized tensors in canonical (name-sorted) order and pack their full 32-byte
/// chunks into little-endian u64 words — the exact layout `pom::WeightIndex` indexes. Returns
/// `(words, n_chunks)` with `words.len() == n_chunks * 4`. Runs on the CPU candle device; Android
/// has no candle GPU backend, so this is a plain host-RAM parse, not a resident-tensor borrow.
fn load_weight_words(gguf_path: &str) -> candle_core::Result<(Vec<u64>, u64)> {
    let device = Device::Cpu;
    let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    names.sort(); // canonical order — must match WeightIndex

    let mut words: Vec<u64> = Vec::new();
    let mut n_chunks: u64 = 0;
    for name in &names {
        let qt = content.tensor(&mut file, name, &device)?;
        let bytes = qt.data()?;
        let full = bytes.len() / 32; // drop any sub-chunk remainder, like WeightIndex
        if full == 0 {
            continue;
        }
        for w in bytes[..full * 32].chunks_exact(8) {
            words.push(u64::from_le_bytes(w.try_into().unwrap()));
        }
        n_chunks += full as u64;
    }
    if n_chunks == 0 {
        return Err(candle_core::Error::Msg("PoM(vulkan): model produced 0 chunks".into()));
    }
    Ok((words, n_chunks))
}

/// Pick a power-of-two shard size (chunks per shard) that comfortably fits the device's real
/// `maxMemoryAllocationSize`, queried at runtime — desktop AMD's ~2 GiB single-allocation headroom
/// assumption does not hold on mobile GPUs, which can report far less. Falls back to a
/// conservative 256 MiB default if the probe fails or the limit is unreported (0); clamps to the
/// original desktop default (1 GiB, `1<<25` chunks) as an upper bound and 2 MiB as a lower bound
/// so a pathologically small `n_chunks` (or a huge reported limit) never picks something silly.
fn pick_shard_chunks(n_chunks: u64) -> u64 {
    const CHUNK_BYTES: u64 = 32;
    const MIN_SHARD_CHUNKS: u64 = 1 << 16; // 2 MiB
    const MAX_SHARD_CHUNKS: u64 = 1 << 25; // 1 GiB (desktop AMD default)
    const FALLBACK_SHARD_BYTES: u64 = 256 * 1024 * 1024; // 256 MiB

    let max_alloc = keryx_vulkan::Vk::new().map(|vk| vk.max_alloc_bytes()).unwrap_or(0);
    let budget_bytes = if max_alloc == 0 { FALLBACK_SHARD_BYTES } else { max_alloc / 2 }; // headroom
    let mut shard_chunks = (budget_bytes / CHUNK_BYTES).next_power_of_two();
    // next_power_of_two can overshoot the budget (rounds up); step back down if so.
    if shard_chunks * CHUNK_BYTES > budget_bytes && shard_chunks > 1 {
        shard_chunks /= 2;
    }
    // shard_chunks may exceed n_chunks — new_sharded's div_ceil then yields a single shard, which
    // is fine, it only needs to be a power of two, not related in size to n_chunks.
    shard_chunks.clamp(MIN_SHARD_CHUNKS, MAX_SHARD_CHUNKS)
}
