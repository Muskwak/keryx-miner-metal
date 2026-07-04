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

// Stratum (pool) transport. Shares the wire codec with the desktop StratumHandler
// (crate::statum_codec) so the JSON-RPC format is identical. iOS speaks a trimmed
// subset: subscribe/authorize/declare, then set_difficulty/set_extranonce/notify →
// mine PoM → mining.submit (MiningSubmitWithPom).
use crate::statum_codec::{
    MiningNotify, MiningSubmit, MiningSubscribe, NewLineJsonCodec, SetExtranonce, StratumCommand,
    StratumError, StratumLine, StratumLinePayload, StratumResult,
};
use crate::target::Uint256;

/// Difficulty-1 target (mantissa, exponent) — 0xffff · 2^208. Identical to the
/// desktop StratumHandler's DIFFICULTY_1_TARGET so set_difficulty math matches.
const DIFFICULTY_1_TARGET: (u64, i16) = (0xffffu64, 208);
/// Capability string advertised in mining.subscribe so a Keryx pool sends
/// daa_score-carrying notifies (ShortV2/WithTask) — required for the PoM branch.
const KERYX_STRATUM_DAA_CAPABILITY: &str = "keryx-stratum-v2";

// Mutex<Option<String>>, not OnceLock<String> — the user can reconnect with a different address
// after a Stop, and OnceLock::set silently no-ops after the first call, which would leave every
// later reconnect stuck on whatever address was used first.
static GRPC_ADDRESS: Mutex<Option<String>> = Mutex::new(None);
static MINING_ADDRESS: Mutex<Option<String>> = Mutex::new(None);
static NONCES_FOUND: AtomicU64 = AtomicU64::new(0);
static LAST_LOG: OnceLock<Mutex<String>> = OnceLock::new();
static RUNNING: AtomicBool = AtomicBool::new(false);
static STOP_TX: OnceLock<watch::Sender<bool>> = OnceLock::new();
/// Set true once the heavy one-time PoM model load (index + Metal upload) has
/// succeeded, so we log it once and don't re-attempt on every template.
static INSTALLED_OK: AtomicBool = AtomicBool::new(false);
/// Total GPU mining batches dispatched — drives the heartbeat log.
static BATCH_COUNT: AtomicU64 = AtomicU64::new(0);
/// Most recent batch's MH/s, as `f64::to_bits` (no atomic f64 in std) — exposed in the status JSON
/// so the UI can show a live number instead of scanning log text for the heartbeat line.
static LAST_HASHRATE_MHS_BITS: AtomicU64 = AtomicU64::new(0);

const BATCH_SIZE: u64 = 1 << 20;
// 1, not 16: mobile GPU throughput can be far below desktop-class rates, so a 16-batch heartbeat
// interval could take minutes to produce its first hashrate log line — looking like mining had
// silently stalled. Report every batch.
const HEARTBEAT_BATCHES: u64 = 1;

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

/// Builds the coinbase `extra_data` for a solo (gRPC) GetBlockTemplateRequest.
///
/// The node validates that every block's coinbase carries a Phase-2 OPoI tag:
/// `/{nonce_hex}/ai:v1:{tag}` where `tag = keryx_inference::tag_fixed(nonce)`
/// (a bit-exact fixed-point MLP the node recomputes). Without it the node
/// rejects the submitted block as `BlockInvalid` — which is exactly what a plain
/// `extra_data` produced. This mirrors `GrpcClient::client_get_block_template`
/// (src/client/grpc.rs): a fresh random coinbase nonce (distinct from the PoW
/// nonce — it only keeps parallel coinbases unique), the tag over it, and the
/// loaded-model capability so the node's model-id enforcement passes.
fn build_template_extra_data() -> String {
    let nonce = rand::random::<u64>();
    let nonce_hex = format!("{:016x}", nonce);
    let opoi_tag = keryx_inference::tag_fixed(nonce);
    let cap_part = models::specs_for(VERY_LIGHT_ACTIVATION_DAA, Tier::VeryLight)
        .first()
        .map(|s| format!("/ai:cap:{}", hex::encode(s.model_id)))
        .unwrap_or_default();
    format!(
        "keryx-miner-ios/{}/{}/ai:v1:{}{}",
        env!("CARGO_PKG_VERSION"),
        nonce_hex,
        opoi_tag,
        cap_part
    )
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
    // Upstream made PoM per-GPU: set_mining_tier is keyed by device_id. iOS is
    // single-GPU (device 0).
    pom_gpu::set_mining_tier(0, spec.model_id, gguf_path.to_string_lossy().into_owned());
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
    *GRPC_ADDRESS.lock().unwrap() = Some(addr);
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
    *MINING_ADDRESS.lock().unwrap() = Some(addr);
    true
}

#[no_mangle]
pub extern "C" fn keryx_miner_start() -> bool {
    install_log_bridge();
    if RUNNING.swap(true, Ordering::SeqCst) {
        return false;
    }
    let address = match GRPC_ADDRESS.lock().unwrap().clone() {
        Some(a) => a,
        None => {
            RUNNING.store(false, Ordering::SeqCst);
            return false;
        }
    };
    let mining_addr = MINING_ADDRESS
        .lock()
        .unwrap()
        .clone()
        .unwrap_or_else(|| "keryx:ios:miner".into());
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

    // Transport is chosen by the address scheme, exactly like the desktop binary
    // (main.rs): `stratum+tcp://host:port` → pool stratum client, anything else
    // (bare host, `grpc://…`) → solo gRPC. A `stratum+tcp://` address fed to the
    // gRPC path previously failed inside tonic (the "metadataMap { headers: {} }"
    // transport error) because tonic only speaks HTTP/2 gRPC.
    let is_stratum = address.starts_with("stratum+tcp://") || address.starts_with("stratum://");

    thread::spawn(move || {
        let rt = match tokio::runtime::Runtime::new() {
            Ok(r) => r,
            Err(e) => {
                log_msg(&format!("ios: tokio runtime creation failed: {e}"));
                RUNNING.store(false, Ordering::SeqCst);
                return;
            }
        };
        if is_stratum {
            rt.block_on(stratum_mining_loop(address, mining_addr, stop_rx));
        } else {
            rt.block_on(mining_loop(address, mining_addr, stop_rx));
        }
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

    // Winning blocks flow back over a transport-agnostic BlockSeed channel; the
    // worker never knows whether it's feeding gRPC or stratum. Here (gRPC) a
    // forwarder converts each FullBlock seed into a SubmitBlockRequest.
    let (submit_tx, mut submit_rx) = tokio::sync::mpsc::channel::<crate::pow::BlockSeed>(16);
    let submit_req_tx = req_tx.clone();
    tokio::spawn(async move {
        while let Some(seed) = submit_rx.recv().await {
            if let crate::pow::BlockSeed::FullBlock(block) = seed {
                log_msg("ios: submitting block…");
                let msg = KaspadMessage {
                    payload: Some(Payload::SubmitBlockRequest(SubmitBlockRequestMessage {
                        block: Some(*block),
                        allow_non_daa_blocks: false,
                    })),
                };
                let _ = submit_req_tx.send(msg).await;
            }
        }
    });

    // The GPU miner runs on its own OS thread: pom_gpu::mine() is a *blocking*
    // call and must not run on the async executor — doing so previously starved
    // this receive loop, so it fell behind the stream and mined stale templates.
    let worker = std::thread::spawn(move || mining_worker(job_rx, submit_tx));

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
                extra_data: build_template_extra_data(),
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
                                            extra_data: build_template_extra_data(),
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

/// Compute a Uint256 pool target from a stratum `mining.set_difficulty` value.
/// Byte-for-byte identical to the desktop StratumHandler::set_difficulty math so
/// the miner and pool agree on which nonces clear the share threshold.
fn stratum_difficulty_target(difficulty: f32) -> Option<Uint256> {
    use num::Float;
    let mut buf = [0u64, 0u64, 0u64, 0u64];
    let (mantissa, exponent, _) = difficulty.recip().integer_decode();
    let new_mantissa = mantissa * DIFFICULTY_1_TARGET.0;
    let new_exponent = (DIFFICULTY_1_TARGET.1 + exponent) as u64;
    let start = (new_exponent / 64) as usize;
    let remainder = new_exponent % 64;
    buf[start] = new_mantissa << remainder;
    if start < 3 {
        buf[start + 1] = new_mantissa >> (64 - remainder);
    } else if new_mantissa.leading_zeros() < remainder as u32 {
        return None; // target too big
    }
    Some(Uint256::new(buf))
}

/// Apply a stratum extranonce assignment. `nonce_size` is the number of *low*
/// bytes the miner controls; the extranonce occupies the high bytes (same
/// convention as the desktop StratumHandler::set_extranonce). Guards the shifts
/// so a `nonce_size >= 8` (miner owns the whole nonce) can't overflow the u64.
fn apply_extranonce(nonce_fixed: &mut u64, nonce_mask: &mut u64, fixed: u64, nonce_size: u32) {
    if nonce_size >= 8 {
        *nonce_fixed = 0;
        *nonce_mask = u64::MAX;
    } else {
        let bits = nonce_size * 8;
        *nonce_fixed = fixed << bits;
        *nonce_mask = (1u64 << bits) - 1;
    }
}

/// Pool (stratum) mining loop — the transport counterpart of `mining_loop`.
///
/// Speaks the same JSON-RPC wire format as the desktop StratumHandler (via the
/// shared `crate::statum_codec`), but trimmed to what an iOS PoM miner needs:
/// subscribe/authorize/declare → receive set_difficulty/set_extranonce/notify →
/// build a PoM `State` from the `PartialBlock` → grind on the shared GPU worker →
/// submit `mining.submit` (MiningSubmitWithPom). The heavy Phase-2 OPoI inference
/// (challenges / AiRequest CIDs) is intentionally not implemented on iOS.
async fn stratum_mining_loop(address: String, mining_addr: String, mut stop_rx: watch::Receiver<bool>) {
    use futures::StreamExt as _;
    use futures_util::TryStreamExt as _;
    use std::sync::atomic::AtomicU32;
    use std::sync::Arc;

    // Strip the scheme → bare host:port for TcpStream::connect.
    let host_port = address
        .strip_prefix("stratum+tcp://")
        .or_else(|| address.strip_prefix("stratum://"))
        .unwrap_or(&address)
        .to_string();

    log_msg(&format!("ios: stratum connecting to {host_port} …"));
    let socket = match tokio::net::TcpStream::connect(&host_port).await {
        Ok(s) => s,
        Err(e) => {
            log_msg(&format!("ios: stratum connect failed: {e}"));
            return;
        }
    };

    let client = tokio_util::codec::Framed::new(socket, NewLineJsonCodec::new());
    let (send_channel, recv) = tokio::sync::mpsc::channel::<StratumLine>(16);
    let (sink, mut stream) = client.split();
    // Pump outbound lines from send_channel → the TCP sink.
    tokio::spawn(async move {
        let _ = tokio_stream::wrappers::ReceiverStream::new(recv).map(Ok).forward(sink).await;
    });

    let last_id = Arc::new(AtomicU32::new(1));

    // Coalescing job channel + transport-agnostic submit channel, same as gRPC.
    let (job_tx, job_rx) =
        crate::watch::channel::<Option<std::sync::Arc<crate::pow::State>>>(None);
    let (submit_tx, mut submit_rx) = tokio::sync::mpsc::channel::<crate::pow::BlockSeed>(16);
    let worker = std::thread::spawn(move || mining_worker(job_rx, submit_tx));

    // Submit forwarder: winning PartialBlock seed → mining.submit (WithPom).
    let submit_send = send_channel.clone();
    let submit_addr = mining_addr.clone();
    let submit_id = last_id.clone();
    tokio::spawn(async move {
        while let Some(seed) = submit_rx.recv().await {
            let crate::pow::BlockSeed::PartialBlock { id: job_id, nonce, pom_proof, .. } = seed else {
                continue; // gRPC FullBlock never reaches the stratum forwarder
            };
            let nonce_hex = format!("{:016x}", nonce);
            let opoi_tag = keryx_inference::tag_fixed(nonce);
            let msg_id = submit_id.fetch_add(1, Ordering::SeqCst);
            let submit = if !pom_proof.is_empty() {
                let proof_hex = hex::encode(&pom_proof);
                log_msg(&format!(
                    "ios: PoM submitting share ({} B proof) job={job_id} nonce={nonce_hex}",
                    pom_proof.len()
                ));
                // Fixed 6-slot PoM submit: CID stays empty at params[4], proof at params[5].
                MiningSubmit::MiningSubmitWithPom((
                    submit_addr.clone(),
                    job_id,
                    nonce_hex,
                    opoi_tag,
                    String::new(),
                    proof_hex,
                ))
            } else {
                MiningSubmit::MiningSubmitWithTag((submit_addr.clone(), job_id, nonce_hex, opoi_tag))
            };
            let line = StratumLine {
                id: Some(msg_id),
                payload: StratumLinePayload::StratumCommand(StratumCommand::MiningSubmit(submit)),
                jsonrpc: None,
                error: None,
            };
            if submit_send.send(line).await.is_err() {
                break;
            }
        }
    });

    // ── Handshake: subscribe (with DAA capability) → authorize → declare model ──
    let subscribe = StratumLine {
        id: Some(last_id.fetch_add(1, Ordering::SeqCst)),
        payload: StratumLinePayload::StratumCommand(StratumCommand::Subscribe(
            MiningSubscribe::MiningSubscribeOptions((
                format!("keryx-miner-ios/{}", env!("CARGO_PKG_VERSION")),
                KERYX_STRATUM_DAA_CAPABILITY.into(),
            )),
        )),
        jsonrpc: None,
        error: None,
    };
    if send_channel.send(subscribe).await.is_err() {
        log_msg("ios: stratum subscribe send failed");
        return;
    }

    // Authorize with the pay address (devfund rotation mirrors the gRPC path).
    let pay_address = next_pay_address(&mining_addr);
    let authorize = StratumLine {
        id: Some(last_id.fetch_add(1, Ordering::SeqCst)),
        payload: StratumLinePayload::StratumCommand(StratumCommand::Authorize((
            pay_address,
            "x".into(),
        ))),
        jsonrpc: None,
        error: None,
    };
    let _ = send_channel.send(authorize).await;

    // Declare the very-light model so the pool bridge knows which model we hold
    // (OPoI capability). iOS only ever mines the very-light tier.
    if let Some(spec) = models::specs_for(VERY_LIGHT_ACTIVATION_DAA, Tier::VeryLight).first() {
        let declare = StratumLine {
            id: None,
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningDeclareCapabilities(
                vec![hex::encode(spec.model_id)],
            )),
            jsonrpc: None,
            error: None,
        };
        let _ = send_channel.send(declare).await;
    }

    // ── Connection state (updated by set_difficulty / set_extranonce) ──
    let mut target_pool = Uint256::new([0, 0, 0, 0]); // impossible until set_difficulty
    let mut nonce_mask: u64 = u64::MAX; // full nonce space until set_extranonce
    let mut nonce_fixed: u64 = 0;

    // Build a PoM State from the current job + connection state and publish it.
    macro_rules! publish_job {
        ($id:expr, $header_hash:expr, $timestamp:expr, $daa:expr) => {{
            let seed = crate::pow::BlockSeed::PartialBlock {
                id: $id,
                header_hash: $header_hash,
                timestamp: $timestamp,
                daa_score: $daa,
                nonce: 0,
                target: target_pool,
                nonce_mask,
                nonce_fixed,
                hash: None,
                pom_proof: Vec::new(),
            };
            match crate::pow::State::new(0, seed) {
                Ok(s) => {
                    let _ = job_tx.send(Some(std::sync::Arc::new(s)));
                }
                Err(e) => log_msg(&format!("ios: stratum bad job: {e}")),
            }
        }};
    }

    log_msg("ios: stratum handshake sent — waiting for jobs");
    let mut last_tmpl_log: Option<std::time::Instant> = None;

    loop {
        tokio::select! {
            next = stream.try_next() => {
                let msg = match next {
                    Ok(Some(m)) => m,
                    Ok(None) => { log_msg("ios: stratum stream closed"); break; }
                    Err(_) => { log_msg("ios: stratum decode error"); break; }
                };
                match msg {
                    // Rejected/accepted share results carry an id + optional error.
                    StratumLine { id: Some(_), error: Some(StratumError(code, err, _)), .. } => {
                        log_msg(&format!("ios: share rejected ({code}): {err}"));
                    }
                    StratumLine { payload: StratumLinePayload::StratumResult { result }, error: None, id: Some(_), .. } => {
                        match result {
                            StratumResult::Plain(Some(true)) | StratumResult::Eth((true, _)) => {
                                log_msg("ios: share accepted");
                            }
                            StratumResult::Subscribe((_, ref extranonce, ref nonce_size)) => {
                                if let Ok(fixed) = u64::from_str_radix(extranonce, 16) {
                                    apply_extranonce(&mut nonce_fixed, &mut nonce_mask, fixed, *nonce_size);
                                    log_msg(&format!("ios: extranonce={extranonce} size={nonce_size}"));
                                }
                            }
                            _ => {}
                        }
                    }
                    StratumLine { payload: StratumLinePayload::StratumCommand(command), .. } => {
                        match command {
                            StratumCommand::MiningSetDifficulty((difficulty,)) => {
                                if let Some(t) = stratum_difficulty_target(difficulty) {
                                    target_pool = t;
                                    log_msg(&format!("ios: difficulty={difficulty} target=0x{}", hex::encode(target_pool.to_be_bytes())));
                                } else {
                                    log_msg("ios: set_difficulty target too big — ignored");
                                }
                            }
                            StratumCommand::SetExtranonce(SetExtranonce::SetExtranoncePlain((ref extranonce, ref nonce_size))) => {
                                if let Ok(fixed) = u64::from_str_radix(extranonce, 16) {
                                    apply_extranonce(&mut nonce_fixed, &mut nonce_mask, fixed, *nonce_size);
                                    log_msg(&format!("ios: extranonce={extranonce} size={nonce_size}"));
                                }
                            }
                            StratumCommand::MiningNotify(MiningNotify::MiningNotifyWithTask((id, header_hash, timestamp, daa_score, _task))) => {
                                // iOS does not run Phase-2 AiRequest inference; mine PoM on the block.
                                publish_job!(id, header_hash, timestamp, daa_score);
                                let stale = last_tmpl_log.map_or(true, |t| t.elapsed() >= std::time::Duration::from_secs(2));
                                if stale { log_msg(&format!("ios: stratum job DAA={daa_score}")); last_tmpl_log = Some(std::time::Instant::now()); }
                            }
                            StratumCommand::MiningNotify(MiningNotify::MiningNotifyShortV2((id, header_hash, timestamp, daa_score))) => {
                                publish_job!(id, header_hash, timestamp, daa_score);
                                let stale = last_tmpl_log.map_or(true, |t| t.elapsed() >= std::time::Duration::from_secs(2));
                                if stale { log_msg(&format!("ios: stratum job DAA={daa_score}")); last_tmpl_log = Some(std::time::Instant::now()); }
                            }
                            StratumCommand::MiningNotify(MiningNotify::MiningNotifyShort((id, header_hash, timestamp))) => {
                                // Short notify carries no daa_score — pin to the current salt era
                                // so the matrix generation matches (same as the desktop handler).
                                publish_job!(id, header_hash, timestamp, crate::pow::heavy_hash::POW_SALT_V4_ACTIVATION_DAA);
                            }
                            StratumCommand::MiningChallenge((model_id_hex, _nonce_hex)) => {
                                // Phase-2 OPoI challenge — not answered on iOS (no inference engine).
                                log_msg(&format!("ios: OPoI challenge for {:.8} ignored (iOS PoM-only)", model_id_hex));
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            _ = stop_rx.changed() => {
                if *stop_rx.borrow() { log_msg("ios: stop requested"); break; }
            }
        }
    }

    drop(job_tx);
    let _ = worker.join();
    log_msg("ios: stratum loop ended");
}

/// Blocking GPU mining thread — a faithful port of the desktop's
/// `launch_gpu_miner` PoM branch (src/miner.rs). Reads the latest template from
/// the coalescing `watch` channel, grinds one `pom_gpu::mine` batch at a time on
/// a persistent (never-reset) nonce cursor, and submits winning blocks via the
/// shared outbound channel. Runs on its own OS thread so the blocking `mine`
/// call never starves the async gRPC receiver.
fn mining_worker(
    mut job_rx: crate::watch::Receiver<Option<std::sync::Arc<crate::pow::State>>>,
    submit_tx: tokio::sync::mpsc::Sender<crate::pow::BlockSeed>,
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

        // Upstream made PoM per-tier: resolve this block's tier (device 0), then
        // fetch that tier's resident host index (an Arc). Mirrors the desktop
        // worker (miner.rs): current_tier → active_index_for_tier → walk.
        let tier = match pom_gpu::current_tier(0, daa) {
            Some(t) => t,
            None => {
                log_msg("ios: ERROR current_tier() None after install — retrying");
                std::thread::sleep(std::time::Duration::from_millis(200));
                continue;
            }
        };
        let index = match pom::active_index_for_tier(tier) {
            Some(x) => x,
            None => {
                log_msg("ios: ERROR active_index_for_tier() None after install — retrying");
                std::thread::sleep(std::time::Duration::from_millis(200));
                continue;
            }
        };

        let mut pph = [0u8; 32];
        pph.copy_from_slice(&s.pow_hash_header[..32]);
        let timestamp = u64::from_le_bytes(s.pow_hash_header[32..40].try_into().unwrap());
        let target_le = s.target.to_le_bytes();

        // Honor the pool's assigned nonce sub-range: on stratum, set_extranonce
        // pins the high bits (nonce_fixed) and leaves the low bits (nonce_mask)
        // for us to grind. Solo gRPC leaves mask=u64::MAX / fixed=0, so `start`
        // reduces to the raw cursor and behavior is unchanged. Submitting nonces
        // outside the assigned range would get every share rejected by the pool.
        let start = s.nonce_fixed | (pom_nonce & s.nonce_mask);

        let t0 = std::time::Instant::now();
        let found = pom_gpu::mine(0, &pph, timestamp, &target_le, start, BATCH_SIZE);
        // Advance within the masked window (wraps inside the sub-range, not the
        // full u64) so we keep the fixed bits intact batch after batch.
        pom_nonce = (pom_nonce.wrapping_add(BATCH_SIZE)) & s.nonce_mask;

        let batches = BATCH_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if batches % HEARTBEAT_BATCHES == 0 {
            let secs = t0.elapsed().as_secs_f64().max(1e-6);
            let mhs = (BATCH_SIZE as f64 / secs) / 1e6;
            LAST_HASHRATE_MHS_BITS.store(mhs.to_bits(), Ordering::Relaxed);
            log_msg(&format!("ios: mining… {batches} batches, {mhs:.2} MH/s"));
        }

        if let Some(winning_nonce) = found {
            log_msg(&format!("ios: PoM winner nonce={winning_nonce}"));
            // generate_block_if_pom re-validates on the CPU and, on success,
            // returns a seed of the *same variant* as this State's block:
            // FullBlock for gRPC, PartialBlock (with the borsh PoM proof) for
            // stratum. Either way the transport-specific forwarder handles it.
            if let Some(seed) = s.generate_block_if_pom(winning_nonce, index.as_ref(), tier) {
                NONCES_FOUND.fetch_add(1, Ordering::Relaxed);
                let _ = submit_tx.blocking_send(seed);
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
    let hashrate_mhs = f64::from_bits(LAST_HASHRATE_MHS_BITS.load(Ordering::Relaxed));
    let log = LAST_LOG.get_or_init(|| Mutex::new(String::new()));
    let log_content = log.lock().ok().map(|l| l.clone()).unwrap_or_default();
    let last_lines: Vec<&str> = log_content.lines().rev().take(20).collect();
    let last_lines: Vec<&str> = last_lines.into_iter().rev().collect();
    let json = format!(
        r#"{{"running":{},"nonces_found":{},"hashrate_mhs":{:.4},"log_lines":[{}]}}"#,
        running,
        nonces,
        hashrate_mhs,
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
