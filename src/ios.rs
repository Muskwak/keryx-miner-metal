use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread;

use tokio::sync::watch;

use crate::models::{self, ModelSpec, Tier, VERY_LIGHT_ACTIVATION_DAA};
use crate::pom;
use crate::pom_gpu;
use crate::proto::kaspad_message::Payload;
use crate::proto::rpc_client::RpcClient;
use crate::proto::{
    GetBlockTemplateRequestMessage, KaspadMessage, NotifyNewBlockTemplateRequestMessage,
    SubmitBlockRequestMessage,
};

static GRPC_ADDRESS: OnceLock<String> = OnceLock::new();
static MINING_ADDRESS: OnceLock<String> = OnceLock::new();
static NONCES_FOUND: AtomicU64 = AtomicU64::new(0);
static LAST_LOG: OnceLock<Mutex<String>> = OnceLock::new();
static RUNNING: AtomicBool = AtomicBool::new(false);
static STOP_TX: OnceLock<watch::Sender<bool>> = OnceLock::new();
/// Set true once the heavy one-time PoM model load (index + Metal upload) has
/// succeeded, so we log it once and don't re-attempt on every template.
static INSTALLED_OK: AtomicBool = AtomicBool::new(false);
/// Total GPU mining batches dispatched — drives the heartbeat log.
static BATCH_COUNT: AtomicU64 = AtomicU64::new(0);

const BATCH_SIZE: u64 = 1 << 20;
const HEARTBEAT_BATCHES: u64 = 16;

/// Same mechanism as the desktop CLI's `--devfund-percent` (src/cli.rs,
/// src/client/grpc.rs::get_block_template): out of every 10_000 block-template
/// requests, `DEVFUND_PERCENT` of them pay to `DEVFUND_ADDRESS` instead of the
/// user's mining address. Floored at 2% (200/10_000) — not user-configurable
/// on iOS, matching the desktop's forced minimum in `parse_devfund_percent`.
const DEVFUND_ADDRESS: &str = "keryx:qpcptntu45n0xtyq60apnwnhpkta0ujzt5sy3uk5v6nrjvxlqhamjyc882jj3";
const DEVFUND_PERCENT: u16 = 200;
static DEVFUND_CTR: AtomicU16 = AtomicU16::new(0);

/// Picks the pay_address for the next GetBlockTemplateRequest, rotating a
/// fraction of requests to the devfund address. Mirrors
/// `GrpcClient::get_block_template`'s counter/modulo-10_000 logic.
fn next_pay_address(mining_addr: &str) -> String {
    let counter = DEVFUND_CTR.load(Ordering::SeqCst);
    let addr = if counter <= DEVFUND_PERCENT {
        DEVFUND_ADDRESS.to_string()
    } else {
        mining_addr.to_string()
    };
    let _ = DEVFUND_CTR.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| Some((v + 1) % 10_000));
    addr
}

fn log_msg(msg: &str) {
    let log = LAST_LOG.get_or_init(|| Mutex::new(String::new()));
    if let Ok(mut log) = log.lock() {
        log.push_str(msg);
        log.push('\n');
        let len = log.len();
        if len > 65536 {
            *log = log.split_off(len - 32768);
        }
    }
}

/// Bridges the `log` crate into the on-screen status log. On iOS there is no
/// `main()` to call `env_logger::init()` (that only runs in the desktop binary),
/// so every `log::info!/warn!/error!` inside pom.rs / pom_gpu.rs / slm.rs was a
/// silent no-op — which is why a failing `ensure_installed()` looked like the
/// miner just bouncing between "got block template" and "requesting" with no
/// visible reason. This forwards those records to `log_msg` so the actual error
/// (index build failure, model load OOM, chunk-count mismatch, …) is visible.
struct IosLogger;

impl log::Log for IosLogger {
    fn enabled(&self, meta: &log::Metadata) -> bool {
        // Always surface warnings/errors. For info/debug, only forward our own
        // mining-relevant modules — otherwise tonic/h2/hyper flood the UI.
        meta.level() <= log::Level::Warn
            || meta.target().contains("pom")
            || meta.target().contains("slm")
            || meta.target().contains("keryx")
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            log_msg(&format!("[{}] {}", record.level(), record.args()));
        }
    }

    fn flush(&self) {}
}

static IOS_LOGGER: IosLogger = IosLogger;
static LOGGER_SET: OnceLock<()> = OnceLock::new();

/// Idempotent: safe to call from both `keryx_miner_initialize` and
/// `keryx_miner_start` (whichever runs first wins; `set_logger` errors if
/// already set, which we ignore).
fn install_log_bridge() {
    LOGGER_SET.get_or_init(|| {
        if log::set_logger(&IOS_LOGGER).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
    });
}

static MODEL_BASE: OnceLock<std::path::PathBuf> = OnceLock::new();

/// The root that holds one sub-directory per tier-model. Defaults to
/// `<exe_dir>/keryx-models` (desktop fallback) until Swift overrides it via
/// `keryx_miner_set_doc_path` with the app's sandboxed Documents URL.
fn model_dir() -> std::path::PathBuf {
    MODEL_BASE
        .get_or_init(|| {
            let mut p = std::env::current_exe().unwrap_or_default();
            p.pop();
            p.push("keryx-models");
            p
        })
        .clone()
}

/// Called once from the SwiftUI App with the sandbox documents URL.
/// We append "keryx-models/" so downloads are isolated from user files.
/// Must be called before the first `model_dir()` access (i.e. before
/// `keryx_miner_initialize`/`keryx_miner_start`) or it has no effect.
#[no_mangle]
pub extern "C" fn keryx_miner_set_doc_path(path: *const std::ffi::c_char) -> bool {
    let c_str = unsafe { CStr::from_ptr(path) };
    let s = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };
    let mut p = std::path::PathBuf::from(s);
    p.push("keryx-models");
    let _ = std::fs::create_dir_all(&p);
    // First call wins (matches MODEL_BASE's OnceLock semantics) — fine since
    // Swift only calls this once, at app launch.
    let _ = MODEL_BASE.set(p);
    true
}

/// iOS only ever mines the `--very-light` tier (Qwen3-1.7B, ~1-2 GB) — it's the
/// only tier that fits alongside iOS in an iPhone's RAM. This downloads the
/// model (if needed) and registers it with `pom_gpu` so `ensure_installed` can
/// build the weight index. Idempotent: `pom_gpu::set_mining_tier` is a
/// OnceLock, so calling this from both `keryx_miner_initialize` (at launch)
/// and `keryx_miner_start` (defensively, in case initialize wasn't called)
/// is safe as long as both target the same tier.
fn ensure_mining_model_ready() -> bool {
    let specs: &[&ModelSpec] = models::specs_for(VERY_LIGHT_ACTIVATION_DAA, Tier::VeryLight);
    let Some(spec) = specs.first() else {
        return false;
    };
    let Some(gguf_path) = download_model(spec) else {
        return false;
    };
    pom_gpu::set_mining_tier(spec.model_id, gguf_path.to_string_lossy().into_owned());
    true
}

/// Called from the SwiftUI App on launch so the (multi-GB) model download
/// happens while the user is looking at the UI, not after they tap Start.
#[no_mangle]
pub extern "C" fn keryx_miner_initialize() -> bool {
    install_log_bridge();
    ensure_mining_model_ready()
}

/// Serializes model downloads: `keryx_miner_initialize` (launch, background
/// thread) and `keryx_miner_start` (defensive fallback) can both call
/// `download_model` for the same file — without this, a Start tap mid-launch
/// download would race two writers on the same path.
static DOWNLOAD_LOCK: Mutex<()> = Mutex::new(());

/// Downloads (with resume/retry, via `slm::download_file`) the model's GGUF
/// from the same IPFS gateway the desktop miner uses, and returns its path
/// once complete. Layout mirrors the desktop's `<dir>/model.gguf` + `.ok`
/// sentinel, just rooted under the iOS sandbox's model_dir() instead of
/// `<exe_dir>/models/`.
fn download_model(spec: &ModelSpec) -> Option<std::path::PathBuf> {
    let _guard = DOWNLOAD_LOCK.lock();
    let dir = model_dir().join(spec.dir_name);
    let gguf_path = dir.join("model.gguf");
    let ok_file = dir.join(".ok");
    if ok_file.exists() && gguf_path.exists() {
        log_msg(&format!("ios: model '{}' already downloaded", spec.name));
        return Some(gguf_path);
    }
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log_msg(&format!("ios: model dir create error: {e}"));
        return None;
    }
    let _ = std::fs::remove_file(&ok_file); // clear stale flag before (re)downloading

    let url = crate::slm::ipfs_url(spec.weight_cids[0]);
    log_msg(&format!("ios: downloading model '{}' from {} …", spec.name, url));
    if let Err(e) = crate::slm::download_file(&url, &gguf_path) {
        log_msg(&format!("ios: model download failed: {e}"));
        return None;
    }
    let _ = std::fs::write(&ok_file, b"ok");
    log_msg(&format!("ios: model '{}' downloaded", spec.name));
    Some(gguf_path)
}

#[no_mangle]
pub extern "C" fn keryx_miner_connect(address: *const std::ffi::c_char) -> bool {
    let c_str = unsafe { CStr::from_ptr(address) };
    let addr = match c_str.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return false,
    };
    let _ = GRPC_ADDRESS.set(addr);
    log_msg(&format!("ios: gRPC address set"));
    true
}

#[no_mangle]
pub extern "C" fn keryx_miner_set_mining_address(address: *const std::ffi::c_char) -> bool {
    let c_str = unsafe { CStr::from_ptr(address) };
    let addr = match c_str.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return false,
    };
    let _ = MINING_ADDRESS.set(addr);
    true
}

#[no_mangle]
pub extern "C" fn keryx_miner_start() -> bool {
    install_log_bridge();
    if RUNNING.swap(true, Ordering::SeqCst) {
        return false;
    }
    let address = match GRPC_ADDRESS.get() {
        Some(a) => a.clone(),
        None => return false,
    };
    let mining_addr = MINING_ADDRESS.get().cloned().unwrap_or_else(|| "keryx:ios:miner".into());
    let (stop_tx, stop_rx) = watch::channel(false);
    let _ = STOP_TX.set(stop_tx);

    log_msg("ios: starting mining runtime…");
    log_msg(&format!(
        "ios: devfund enabled, mining {:.2}% of the time to {}",
        DEVFUND_PERCENT as f64 / 100.0,
        DEVFUND_ADDRESS
    ));

    // Defensive: normally already done by keryx_miner_initialize() at app
    // launch, but cover the case where Swift skipped that call.
    if !ensure_mining_model_ready() {
        log_msg("ios: FAILED to download model — cannot mine");
        RUNNING.store(false, Ordering::SeqCst);
        return false;
    }

    thread::spawn(move || {
        let rt = match tokio::runtime::Runtime::new() {
            Ok(r) => r,
            Err(e) => {
                log_msg(&format!("ios: tokio runtime creation failed: {e}"));
                RUNNING.store(false, Ordering::SeqCst);
                return;
            }
        };
        rt.block_on(mining_loop(address, mining_addr, stop_rx));
        RUNNING.store(false, Ordering::SeqCst);
        log_msg("ios: mining loop exited");
    });

    true
}

async fn mining_loop(grpc_addr: String, mining_addr: String, mut stop_rx: watch::Receiver<bool>) {
    let endpoint_str = if grpc_addr.contains("://") {
        grpc_addr.clone()
    } else {
        format!("grpc://{}", grpc_addr)
    };

    let endpoint = match tonic::transport::Endpoint::new(endpoint_str) {
        Ok(e) => e,
        Err(e) => {
            log_msg(&format!("ios: invalid gRPC endpoint: {e}"));
            return;
        }
    };

    let mut client = match RpcClient::connect(endpoint).await {
        Ok(c) => c,
        Err(e) => {
            log_msg(&format!("ios: gRPC connect failed: {e}"));
            return;
        }
    };

    // Large outbound buffer (matches the desktop's 1024) so a burst of template
    // requests + a block submission never queues up behind a full channel and
    // stalls template delivery.
    let (req_tx, req_rx) = tokio::sync::mpsc::channel::<KaspadMessage>(1024);
    let response = match client
        .message_stream(tokio_stream::wrappers::ReceiverStream::new(req_rx))
        .await
    {
        Ok(r) => r.into_inner(),
        Err(e) => {
            log_msg(&format!("ios: gRPC stream open failed: {e}"));
            return;
        }
    };

    tokio::pin!(response);

    // Job channel: this async receiver hands the *latest* block template to the
    // blocking GPU mining thread. `crate::watch` coalesces — if templates arrive
    // faster than the GPU grinds a batch, the miner just picks up the newest, so
    // it can never build a backlog or mine a stale template. This mirrors the
    // desktop's block_channel (watch::Sender) → launch_gpu_miner design.
    let (job_tx, job_rx) =
        crate::watch::channel::<Option<std::sync::Arc<crate::pow::State>>>(None);

    // The GPU miner runs on its own OS thread: pom_gpu::mine() is a *blocking*
    // call and must not run on the async executor — doing so previously starved
    // this receive loop, so it fell behind the stream and mined stale templates.
    let worker_req_tx = req_tx.clone();
    let worker = std::thread::spawn(move || mining_worker(job_rx, worker_req_tx));

    // Subscribe to new block templates
    let _ = req_tx
        .send(KaspadMessage {
            payload: Some(Payload::NotifyNewBlockTemplateRequest(NotifyNewBlockTemplateRequestMessage {})),
        })
        .await;

    // Request first template
    let _ = req_tx
        .send(KaspadMessage {
            payload: Some(Payload::GetBlockTemplateRequest(GetBlockTemplateRequestMessage {
                pay_address: next_pay_address(&mining_addr),
                extra_data: format!("keryx-miner-ios/{}", env!("CARGO_PKG_VERSION")),
                inference_result: String::new(),
            })),
        })
        .await;

    // Throttle template logging: the node emits many templates/sec, which would
    // otherwise drown the heartbeat and everything else in the 20-line log view.
    let mut last_tmpl_log: Option<std::time::Instant> = None;

    loop {
        tokio::select! {
            msg = response.message() => {
                match msg {
                    Ok(Some(m)) => {
                        if let Some(payload) = m.payload {
                            match payload {
                                Payload::GetBlockTemplateResponse(r) => {
                                    if let Some(block) = r.block {
                                        let daa = block.header.as_ref().map(|h| h.daa_score).unwrap_or(0);
                                        // Build the PoW/PoM State on the (cheap) async side, exactly
                                        // like the desktop's process_block, then publish it to the
                                        // miner thread. The watch coalesces intermediate templates.
                                        match crate::pow::State::new(0, crate::pow::BlockSeed::FullBlock(Box::new(block))) {
                                            Ok(s) => {
                                                let _ = job_tx.send(Some(std::sync::Arc::new(s)));
                                                let stale = last_tmpl_log
                                                    .map_or(true, |t| t.elapsed() >= std::time::Duration::from_secs(2));
                                                if stale {
                                                    log_msg(&format!("ios: mining on latest template DAA={daa}"));
                                                    last_tmpl_log = Some(std::time::Instant::now());
                                                }
                                            }
                                            Err(e) => log_msg(&format!("ios: bad template DAA={daa}: {e}")),
                                        }
                                    }
                                }
                                Payload::NewBlockTemplateNotification(_) => {
                                    // A new block landed — pull a fresh template. (No log: fires
                                    // many times/sec; the throttled "mining on latest template"
                                    // line above is the visible signal.)
                                    let _ = req_tx.send(KaspadMessage {
                                        payload: Some(Payload::GetBlockTemplateRequest(GetBlockTemplateRequestMessage {
                                            pay_address: next_pay_address(&mining_addr),
                                            extra_data: format!("keryx-miner-ios/{}", env!("CARGO_PKG_VERSION")),
                                            inference_result: String::new(),
                                        })),
                                    }).await;
                                }
                                Payload::SubmitBlockResponse(r) => {
                                    let reject = r.reject_reason();
                                    log_msg(&format!("ios: submit block response: reject={reject:?}"));
                                }
                                _ => {}
                            }
                        }
                    }
                    Ok(None) => {
                        log_msg("ios: gRPC stream closed");
                        break;
                    }
                    Err(e) => {
                        log_msg(&format!("ios: gRPC recv error: {e}"));
                        break;
                    }
                }
            }
            _ = stop_rx.changed() => {
                if *stop_rx.borrow() {
                    log_msg("ios: stop requested");
                    break;
                }
            }
        }
    }

    // Dropping the job sender closes the channel; the worker sees it on its next
    // batch boundary (or wakes from wait_for_change) and exits. Then join it.
    drop(job_tx);
    let _ = worker.join();
    log_msg("ios: mining loop ended");
}

/// Blocking GPU mining thread — a faithful port of the desktop's
/// `launch_gpu_miner` PoM branch (src/miner.rs). Reads the latest template from
/// the coalescing `watch` channel, grinds one `pom_gpu::mine` batch at a time on
/// a persistent (never-reset) nonce cursor, and submits winning blocks via the
/// shared outbound channel. Runs on its own OS thread so the blocking `mine`
/// call never starves the async gRPC receiver.
fn mining_worker(
    mut job_rx: crate::watch::Receiver<Option<std::sync::Arc<crate::pow::State>>>,
    req_tx: tokio::sync::mpsc::Sender<KaspadMessage>,
) {
    // Persistent nonce cursor: advances by BATCH_SIZE each batch and is NOT reset
    // per template (each (template, nonce) is an independent PoM trial). Random-ish
    // start so relaunches don't all begin at nonce 0.
    let mut pom_nonce: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    let mut state: Option<std::sync::Arc<crate::pow::State>> = None;

    loop {
        if !RUNNING.load(Ordering::Relaxed) {
            break;
        }

        // No job yet → block until the receiver publishes one (or the channel closes).
        if state.is_none() {
            match job_rx.wait_for_change() {
                Ok(s) => state = s,
                Err(_) => break, // sender dropped → stop
            }
            continue;
        }
        let s = state.clone().unwrap();
        let daa = s.daa_score;

        // One-time heavy model load (index build + Metal GPU upload). A `false`
        // here is surfaced (the log bridge forwards the underlying error); we back
        // off and retry rather than spin invisibly.
        if !INSTALLED_OK.load(Ordering::Relaxed) {
            log_msg(&format!("ios: loading PoM model into GPU (one-time) at DAA={daa}…"));
            if pom_gpu::ensure_installed(0, daa) {
                INSTALLED_OK.store(true, Ordering::Relaxed);
                log_msg("ios: PoM model installed — mining now active");
            } else {
                log_msg("ios: ERROR ensure_installed returned false (see [ERROR]/[WARN] above) — retrying");
                std::thread::sleep(std::time::Duration::from_millis(500));
                if let Ok(Some(ns)) = job_rx.get_changed() {
                    if ns.is_some() {
                        state = ns;
                    }
                }
                continue;
            }
        }

        let (index, tier) = match pom::active_index() {
            Some(x) => x,
            None => {
                log_msg("ios: ERROR active_index() None after install — retrying");
                std::thread::sleep(std::time::Duration::from_millis(200));
                continue;
            }
        };

        let mut pph = [0u8; 32];
        pph.copy_from_slice(&s.pow_hash_header[..32]);
        let timestamp = u64::from_le_bytes(s.pow_hash_header[32..40].try_into().unwrap());
        let target_le = s.target.to_le_bytes();

        let t0 = std::time::Instant::now();
        let found = pom_gpu::mine(0, &pph, timestamp, &target_le, pom_nonce, BATCH_SIZE);
        pom_nonce = pom_nonce.wrapping_add(BATCH_SIZE);

        let batches = BATCH_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if batches % HEARTBEAT_BATCHES == 0 {
            let secs = t0.elapsed().as_secs_f64().max(1e-6);
            let mhs = (BATCH_SIZE as f64 / secs) / 1e6;
            log_msg(&format!("ios: mining… {batches} batches, {mhs:.2} MH/s"));
        }

        if let Some(winning_nonce) = found {
            log_msg(&format!("ios: PoM winner nonce={winning_nonce}"));
            if let Some(crate::pow::BlockSeed::FullBlock(found_block)) =
                s.generate_block_if_pom(winning_nonce, index, *tier)
            {
                NONCES_FOUND.fetch_add(1, Ordering::Relaxed);
                log_msg("ios: submitting block…");
                let msg = KaspadMessage {
                    payload: Some(Payload::SubmitBlockRequest(SubmitBlockRequestMessage {
                        block: Some(*found_block),
                        allow_non_daa_blocks: false,
                    })),
                };
                let _ = req_tx.blocking_send(msg);
            }
            // This template is consumed — wait for a fresh one.
            state = None;
        } else {
            // No winner: swap to a fresher template if one arrived (coalesced),
            // else keep grinding the current one with the advanced nonce cursor.
            match job_rx.get_changed() {
                Ok(Some(ns)) => {
                    if ns.is_some() {
                        state = ns;
                    }
                }
                Ok(None) => {}
                Err(_) => break, // sender dropped → stop
            }
        }
    }
    log_msg("ios: mining worker exited");
}

#[no_mangle]
pub extern "C" fn keryx_miner_stop() {
    log_msg("ios: stopping miner…");
    RUNNING.store(false, Ordering::SeqCst);
    if let Some(tx) = STOP_TX.get() {
        let _ = tx.send(true);
    }
}

#[no_mangle]
pub extern "C" fn keryx_miner_status() -> *mut std::ffi::c_char {
    let running = RUNNING.load(Ordering::Relaxed);
    let nonces = NONCES_FOUND.load(Ordering::Relaxed);
    let log = LAST_LOG.get_or_init(|| Mutex::new(String::new()));
    let log_content = log.lock().ok().map(|l| l.clone()).unwrap_or_default();
    let last_lines: Vec<&str> = log_content.lines().rev().take(20).collect();
    let last_lines: Vec<&str> = last_lines.into_iter().rev().collect();
    let json = format!(
        r#"{{"running":{},"nonces_found":{},"log_lines":[{}]}}"#,
        running,
        nonces,
        last_lines
            .iter()
            .map(|l| format!("\"{}\"", l.replace('\\', "\\\\").replace('"', "\\\"")))
            .collect::<Vec<_>>()
            .join(",")
    );
    CString::new(json).unwrap_or_default().into_raw()
}

#[no_mangle]
pub extern "C" fn keryx_miner_free_string(s: *mut std::ffi::c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)); }
    }
}
