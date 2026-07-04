//! Android FFI + mining. Structurally a port of `ios.rs`: same stratum/gRPC dual-transport
//! client, same coalescing `watch` job channel + dedicated GPU-worker thread, same OPoI coinbase
//! tagging — only the GPU backend (`crate::pom_gpu` aliases to `pom_gpu_vulkan.rs` here instead of
//! `pom_gpu_metal.rs`) and the FFI boundary (JNI instead of a raw C ABI) differ. Like iOS, Android
//! only mines the `--very-light` tier (Qwen3-1.7B, the only tier that fits alongside the OS and
//! Vulkan's runtime overhead in a phone's RAM), and does not yet answer `mining.challenge` OPoI
//! challenges (logged and ignored — the same state iOS ships in today; challenge-answering is
//! planned for both platforms together).

use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

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

use crate::statum_codec::{
    MiningNotify, MiningSubmit, MiningSubscribe, NewLineJsonCodec, SetExtranonce, StratumCommand,
    StratumError, StratumLine, StratumLinePayload, StratumResult,
};
use crate::target::Uint256;

const DIFFICULTY_1_TARGET: (u64, i16) = (0xffffu64, 208);
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
/// Set true once the heavy one-time PoM model load (index + Vulkan upload) has
/// succeeded, so we log it once and don't re-attempt on every template.
static INSTALLED_OK: AtomicBool = AtomicBool::new(false);
static BATCH_COUNT: AtomicU64 = AtomicU64::new(0);
/// Most recent batch's MH/s, as `f64::to_bits` (no atomic f64 in std) — exposed in `status_impl`'s
/// JSON so the UI can show a live number instead of scanning log text for the heartbeat line.
static LAST_HASHRATE_MHS_BITS: AtomicU64 = AtomicU64::new(0);

const BATCH_SIZE: u64 = 1 << 20;
// 1, not 16: mobile GPU throughput (especially the shaderInt64-emulated Vulkan path on Adreno) can
// be far below desktop-class rates, so a 16-batch heartbeat interval could take minutes to produce
// its first hashrate log line — looking like mining had silently stalled. Report every batch.
const HEARTBEAT_BATCHES: u64 = 1;

/// Same mechanism as the desktop CLI's `--devfund-percent` and iOS's fixed 2% floor.
const DEVFUND_ADDRESS: &str = "keryx:qpcptntu45n0xtyq60apnwnhpkta0ujzt5sy3uk5v6nrjvxlqhamjyc882jj3";
const DEVFUND_PERCENT: u16 = 200;
static DEVFUND_CTR: AtomicU16 = AtomicU16::new(0);

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

/// Builds the coinbase `extra_data` for a solo (gRPC) GetBlockTemplateRequest. See `ios.rs`'s
/// `build_template_extra_data` for the full rationale — the node requires this OPoI tag or it
/// rejects the block as `BlockInvalid`.
fn build_template_extra_data() -> String {
    let nonce = rand::random::<u64>();
    let nonce_hex = format!("{:016x}", nonce);
    let opoi_tag = keryx_inference::tag_fixed(nonce);
    let cap_part = models::specs_for(VERY_LIGHT_ACTIVATION_DAA, Tier::VeryLight)
        .first()
        .map(|s| format!("/ai:cap:{}", hex::encode(s.model_id)))
        .unwrap_or_default();
    format!(
        "keryx-miner-android/{}/{}/ai:v1:{}{}",
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

/// Bridges the `log` crate into the on-screen status log — Android's `main()`-less native lib
/// never calls `env_logger::init()`, so without this every `log::info!/warn!/error!` inside
/// pom.rs / pom_gpu_vulkan.rs / slm.rs would be a silent no-op.
struct AndroidLogger;

impl log::Log for AndroidLogger {
    fn enabled(&self, meta: &log::Metadata) -> bool {
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

static ANDROID_LOGGER: AndroidLogger = AndroidLogger;
static LOGGER_SET: OnceLock<()> = OnceLock::new();

fn install_log_bridge() {
    LOGGER_SET.get_or_init(|| {
        if log::set_logger(&ANDROID_LOGGER).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
    });
}

static MODEL_BASE: OnceLock<std::path::PathBuf> = OnceLock::new();

/// The root that holds one sub-directory per tier-model. Defaults to `<exe_dir>/keryx-models`
/// until Kotlin overrides it via `nativeSetDocPath` with the app's `Context.filesDir` path.
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

/// Called once from Kotlin at app launch with `context.filesDir.absolutePath` (private,
/// always-writable internal storage — the Android analogue of iOS's sandboxed Documents URL). We
/// append "keryx-models/" so downloads are isolated from any other app data. Must be called
/// before the first `model_dir()` access (i.e. before `initialize`/`start`) or it has no effect.
fn set_doc_path_impl(path: &str) -> bool {
    let mut p = std::path::PathBuf::from(path);
    p.push("keryx-models");
    if std::fs::create_dir_all(&p).is_err() {
        return false;
    }
    let _ = MODEL_BASE.set(p);
    true
}

/// Android only ever mines the `--very-light` tier (Qwen3-1.7B) — the only tier that fits
/// alongside the OS, the JVM/app, and Vulkan's driver overhead in a phone's RAM budget.
fn ensure_mining_model_ready() -> bool {
    let specs: &[&ModelSpec] = models::specs_for(VERY_LIGHT_ACTIVATION_DAA, Tier::VeryLight);
    let Some(spec) = specs.first() else {
        return false;
    };
    let Some(gguf_path) = download_model(spec) else {
        return false;
    };
    // Android is single-GPU (device 0), same as iOS.
    pom_gpu::set_mining_tier(0, spec.model_id, gguf_path.to_string_lossy().into_owned());
    true
}

fn initialize_impl() -> bool {
    install_log_bridge();
    ensure_mining_model_ready()
}

/// Serializes model downloads across `initialize` (launch, background thread) and `start`
/// (defensive fallback) so a Start tap mid-launch download can't race two writers on one path.
static DOWNLOAD_LOCK: Mutex<()> = Mutex::new(());

fn download_model(spec: &ModelSpec) -> Option<std::path::PathBuf> {
    let _guard = DOWNLOAD_LOCK.lock();
    let dir = model_dir().join(spec.dir_name);
    let gguf_path = dir.join("model.gguf");
    let ok_file = dir.join(".ok");
    if ok_file.exists() && gguf_path.exists() {
        log_msg(&format!("android: model '{}' already downloaded", spec.name));
        return Some(gguf_path);
    }
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log_msg(&format!("android: model dir create error: {e}"));
        return None;
    }
    let _ = std::fs::remove_file(&ok_file);

    let url = crate::slm::ipfs_url(spec.weight_cids[0]);
    log_msg(&format!("android: downloading model '{}' from {} …", spec.name, url));
    if let Err(e) = crate::slm::download_file(&url, &gguf_path) {
        log_msg(&format!("android: model download failed: {e}"));
        return None;
    }
    let _ = std::fs::write(&ok_file, b"ok");
    log_msg(&format!("android: model '{}' downloaded", spec.name));
    Some(gguf_path)
}

fn connect_impl(address: &str) -> bool {
    *GRPC_ADDRESS.lock().unwrap() = Some(address.to_string());
    log_msg("android: gRPC/stratum address set");
    true
}

fn set_mining_address_impl(address: &str) -> bool {
    *MINING_ADDRESS.lock().unwrap() = Some(address.to_string());
    true
}

fn start_impl() -> bool {
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
        .unwrap_or_else(|| "keryx:android:miner".into());
    let (stop_tx, stop_rx) = watch::channel(false);
    let _ = STOP_TX.set(stop_tx);

    log_msg("android: starting mining runtime…");
    log_msg(&format!(
        "android: devfund enabled, mining {:.2}% of the time to {}",
        DEVFUND_PERCENT as f64 / 100.0,
        DEVFUND_ADDRESS
    ));

    if !ensure_mining_model_ready() {
        log_msg("android: FAILED to download model — cannot mine");
        RUNNING.store(false, Ordering::SeqCst);
        return false;
    }

    // Transport is chosen by the address scheme, exactly like iOS/the desktop binary.
    let is_stratum = address.starts_with("stratum+tcp://") || address.starts_with("stratum://");

    std::thread::spawn(move || {
        let rt = match tokio::runtime::Runtime::new() {
            Ok(r) => r,
            Err(e) => {
                log_msg(&format!("android: tokio runtime creation failed: {e}"));
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
        log_msg("android: mining loop exited");
    });

    true
}

async fn mining_loop(grpc_addr: String, mining_addr: String, mut stop_rx: watch::Receiver<bool>) {
    let endpoint_str = if grpc_addr.contains("://") { grpc_addr.clone() } else { format!("grpc://{}", grpc_addr) };

    let endpoint = match tonic::transport::Endpoint::new(endpoint_str) {
        Ok(e) => e,
        Err(e) => {
            log_msg(&format!("android: invalid gRPC endpoint: {e}"));
            return;
        }
    };

    let mut client = match RpcClient::connect(endpoint).await {
        Ok(c) => c,
        Err(e) => {
            log_msg(&format!("android: gRPC connect failed: {e}"));
            return;
        }
    };

    let (req_tx, req_rx) = tokio::sync::mpsc::channel::<KaspadMessage>(1024);
    let response = match client.message_stream(tokio_stream::wrappers::ReceiverStream::new(req_rx)).await {
        Ok(r) => r.into_inner(),
        Err(e) => {
            log_msg(&format!("android: gRPC stream open failed: {e}"));
            return;
        }
    };

    tokio::pin!(response);

    let (job_tx, job_rx) = crate::watch::channel::<Option<std::sync::Arc<crate::pow::State>>>(None);

    let (submit_tx, mut submit_rx) = tokio::sync::mpsc::channel::<crate::pow::BlockSeed>(16);
    let submit_req_tx = req_tx.clone();
    tokio::spawn(async move {
        while let Some(seed) = submit_rx.recv().await {
            if let crate::pow::BlockSeed::FullBlock(block) = seed {
                log_msg("android: submitting block…");
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

    let worker = std::thread::spawn(move || mining_worker(job_rx, submit_tx));

    let _ = req_tx
        .send(KaspadMessage { payload: Some(Payload::NotifyNewBlockTemplateRequest(NotifyNewBlockTemplateRequestMessage {})) })
        .await;

    let _ = req_tx
        .send(KaspadMessage {
            payload: Some(Payload::GetBlockTemplateRequest(GetBlockTemplateRequestMessage {
                pay_address: next_pay_address(&mining_addr),
                extra_data: build_template_extra_data(),
                inference_result: String::new(),
            })),
        })
        .await;

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
                                        match crate::pow::State::new(0, crate::pow::BlockSeed::FullBlock(Box::new(block))) {
                                            Ok(s) => {
                                                let _ = job_tx.send(Some(std::sync::Arc::new(s)));
                                                let stale = last_tmpl_log.map_or(true, |t| t.elapsed() >= std::time::Duration::from_secs(2));
                                                if stale {
                                                    log_msg(&format!("android: mining on latest template DAA={daa}"));
                                                    last_tmpl_log = Some(std::time::Instant::now());
                                                }
                                            }
                                            Err(e) => log_msg(&format!("android: bad template DAA={daa}: {e}")),
                                        }
                                    }
                                }
                                Payload::NewBlockTemplateNotification(_) => {
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
                                    log_msg(&format!("android: submit block response: reject={reject:?}"));
                                }
                                _ => {}
                            }
                        }
                    }
                    Ok(None) => { log_msg("android: gRPC stream closed"); break; }
                    Err(e) => { log_msg(&format!("android: gRPC recv error: {e}")); break; }
                }
            }
            _ = stop_rx.changed() => {
                if *stop_rx.borrow() { log_msg("android: stop requested"); break; }
            }
        }
    }

    drop(job_tx);
    let _ = worker.join();
    log_msg("android: mining loop ended");
}

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
        return None;
    }
    Some(Uint256::new(buf))
}

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

/// Pool (stratum) mining loop. Same trimmed subset as iOS: subscribe/authorize/declare, then
/// set_difficulty/set_extranonce/notify → mine PoM → mining.submit. `mining.challenge` (Phase-2
/// OPoI) is logged and ignored — Android has no inference engine wired up yet.
async fn stratum_mining_loop(address: String, mining_addr: String, mut stop_rx: watch::Receiver<bool>) {
    use futures::StreamExt as _;
    use futures_util::TryStreamExt as _;
    use std::sync::atomic::AtomicU32;
    use std::sync::Arc;

    let host_port = address
        .strip_prefix("stratum+tcp://")
        .or_else(|| address.strip_prefix("stratum://"))
        .unwrap_or(&address)
        .to_string();

    log_msg(&format!("android: stratum connecting to {host_port} …"));
    let socket = match tokio::net::TcpStream::connect(&host_port).await {
        Ok(s) => s,
        Err(e) => {
            log_msg(&format!("android: stratum connect failed: {e}"));
            return;
        }
    };

    let client = tokio_util::codec::Framed::new(socket, NewLineJsonCodec::new());
    let (send_channel, recv) = tokio::sync::mpsc::channel::<StratumLine>(16);
    let (sink, mut stream) = client.split();
    tokio::spawn(async move {
        let _ = tokio_stream::wrappers::ReceiverStream::new(recv).map(Ok).forward(sink).await;
    });

    let last_id = Arc::new(AtomicU32::new(1));

    let (job_tx, job_rx) = crate::watch::channel::<Option<std::sync::Arc<crate::pow::State>>>(None);
    let (submit_tx, mut submit_rx) = tokio::sync::mpsc::channel::<crate::pow::BlockSeed>(16);
    let worker = std::thread::spawn(move || mining_worker(job_rx, submit_tx));

    let submit_send = send_channel.clone();
    let submit_addr = mining_addr.clone();
    let submit_id = last_id.clone();
    tokio::spawn(async move {
        while let Some(seed) = submit_rx.recv().await {
            let crate::pow::BlockSeed::PartialBlock { id: job_id, nonce, pom_proof, .. } = seed else {
                continue;
            };
            let nonce_hex = format!("{:016x}", nonce);
            let opoi_tag = keryx_inference::tag_fixed(nonce);
            let msg_id = submit_id.fetch_add(1, Ordering::SeqCst);
            let submit = if !pom_proof.is_empty() {
                let proof_hex = hex::encode(&pom_proof);
                log_msg(&format!(
                    "android: PoM submitting share ({} B proof) job={job_id} nonce={nonce_hex}",
                    pom_proof.len()
                ));
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

    let subscribe = StratumLine {
        id: Some(last_id.fetch_add(1, Ordering::SeqCst)),
        payload: StratumLinePayload::StratumCommand(StratumCommand::Subscribe(MiningSubscribe::MiningSubscribeOptions((
            format!("keryx-miner-android/{}", env!("CARGO_PKG_VERSION")),
            KERYX_STRATUM_DAA_CAPABILITY.into(),
        )))),
        jsonrpc: None,
        error: None,
    };
    if send_channel.send(subscribe).await.is_err() {
        log_msg("android: stratum subscribe send failed");
        return;
    }

    let pay_address = next_pay_address(&mining_addr);
    let authorize = StratumLine {
        id: Some(last_id.fetch_add(1, Ordering::SeqCst)),
        payload: StratumLinePayload::StratumCommand(StratumCommand::Authorize((pay_address, "x".into()))),
        jsonrpc: None,
        error: None,
    };
    let _ = send_channel.send(authorize).await;

    if let Some(spec) = models::specs_for(VERY_LIGHT_ACTIVATION_DAA, Tier::VeryLight).first() {
        let declare = StratumLine {
            id: None,
            payload: StratumLinePayload::StratumCommand(StratumCommand::MiningDeclareCapabilities(vec![hex::encode(spec.model_id)])),
            jsonrpc: None,
            error: None,
        };
        let _ = send_channel.send(declare).await;
    }

    let mut target_pool = Uint256::new([0, 0, 0, 0]);
    let mut nonce_mask: u64 = u64::MAX;
    let mut nonce_fixed: u64 = 0;

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
                Err(e) => log_msg(&format!("android: stratum bad job: {e}")),
            }
        }};
    }

    log_msg("android: stratum handshake sent — waiting for jobs");
    let mut last_tmpl_log: Option<std::time::Instant> = None;

    loop {
        tokio::select! {
            next = stream.try_next() => {
                let msg = match next {
                    Ok(Some(m)) => m,
                    Ok(None) => { log_msg("android: stratum stream closed"); break; }
                    Err(_) => { log_msg("android: stratum decode error"); break; }
                };
                match msg {
                    StratumLine { id: Some(_), error: Some(StratumError(code, err, _)), .. } => {
                        log_msg(&format!("android: share rejected ({code}): {err}"));
                    }
                    StratumLine { payload: StratumLinePayload::StratumResult { result }, error: None, id: Some(_), .. } => {
                        match result {
                            StratumResult::Plain(Some(true)) | StratumResult::Eth((true, _)) => {
                                log_msg("android: share accepted");
                            }
                            StratumResult::Subscribe((_, ref extranonce, ref nonce_size)) => {
                                if let Ok(fixed) = u64::from_str_radix(extranonce, 16) {
                                    apply_extranonce(&mut nonce_fixed, &mut nonce_mask, fixed, *nonce_size);
                                    log_msg(&format!("android: extranonce={extranonce} size={nonce_size}"));
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
                                    log_msg(&format!("android: difficulty={difficulty} target=0x{}", hex::encode(target_pool.to_be_bytes())));
                                } else {
                                    log_msg("android: set_difficulty target too big — ignored");
                                }
                            }
                            StratumCommand::SetExtranonce(SetExtranonce::SetExtranoncePlain((ref extranonce, ref nonce_size))) => {
                                if let Ok(fixed) = u64::from_str_radix(extranonce, 16) {
                                    apply_extranonce(&mut nonce_fixed, &mut nonce_mask, fixed, *nonce_size);
                                    log_msg(&format!("android: extranonce={extranonce} size={nonce_size}"));
                                }
                            }
                            StratumCommand::MiningNotify(MiningNotify::MiningNotifyWithTask((id, header_hash, timestamp, daa_score, _task))) => {
                                publish_job!(id, header_hash, timestamp, daa_score);
                                let stale = last_tmpl_log.map_or(true, |t| t.elapsed() >= std::time::Duration::from_secs(2));
                                if stale { log_msg(&format!("android: stratum job DAA={daa_score}")); last_tmpl_log = Some(std::time::Instant::now()); }
                            }
                            StratumCommand::MiningNotify(MiningNotify::MiningNotifyShortV2((id, header_hash, timestamp, daa_score))) => {
                                publish_job!(id, header_hash, timestamp, daa_score);
                                let stale = last_tmpl_log.map_or(true, |t| t.elapsed() >= std::time::Duration::from_secs(2));
                                if stale { log_msg(&format!("android: stratum job DAA={daa_score}")); last_tmpl_log = Some(std::time::Instant::now()); }
                            }
                            StratumCommand::MiningNotify(MiningNotify::MiningNotifyShort((id, header_hash, timestamp))) => {
                                publish_job!(id, header_hash, timestamp, crate::pow::heavy_hash::POW_SALT_V4_ACTIVATION_DAA);
                            }
                            StratumCommand::MiningChallenge((model_id_hex, _nonce_hex)) => {
                                log_msg(&format!("android: OPoI challenge for {:.8} ignored (Android PoM-only)", model_id_hex));
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            _ = stop_rx.changed() => {
                if *stop_rx.borrow() { log_msg("android: stop requested"); break; }
            }
        }
    }

    drop(job_tx);
    let _ = worker.join();
    log_msg("android: stratum loop ended");
}

/// Blocking GPU mining thread — identical shape to iOS's `mining_worker`, backed by the Vulkan
/// PoM backend (`crate::pom_gpu` → `pom_gpu_vulkan.rs`) instead of Metal.
fn mining_worker(
    mut job_rx: crate::watch::Receiver<Option<std::sync::Arc<crate::pow::State>>>,
    submit_tx: tokio::sync::mpsc::Sender<crate::pow::BlockSeed>,
) {
    let mut pom_nonce: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    let mut state: Option<std::sync::Arc<crate::pow::State>> = None;

    loop {
        if !RUNNING.load(Ordering::Relaxed) {
            break;
        }

        if state.is_none() {
            match job_rx.wait_for_change() {
                Ok(s) => state = s,
                Err(_) => break,
            }
            continue;
        }
        let s = state.clone().unwrap();
        let daa = s.daa_score;

        if !INSTALLED_OK.load(Ordering::Relaxed) {
            log_msg(&format!("android: loading PoM model into GPU (one-time) at DAA={daa}…"));
            if pom_gpu::ensure_installed(0, daa) {
                INSTALLED_OK.store(true, Ordering::Relaxed);
                log_msg("android: PoM model installed — mining now active");
            } else {
                log_msg("android: ERROR ensure_installed returned false (see [ERROR]/[WARN] above) — retrying");
                std::thread::sleep(std::time::Duration::from_millis(500));
                if let Ok(Some(ns)) = job_rx.get_changed() {
                    if ns.is_some() {
                        state = ns;
                    }
                }
                continue;
            }
        }

        let tier = match pom_gpu::current_tier(0, daa) {
            Some(t) => t,
            None => {
                log_msg("android: ERROR current_tier() None after install — retrying");
                std::thread::sleep(std::time::Duration::from_millis(200));
                continue;
            }
        };
        let index = match pom::active_index_for_tier(tier) {
            Some(x) => x,
            None => {
                log_msg("android: ERROR active_index_for_tier() None after install — retrying");
                std::thread::sleep(std::time::Duration::from_millis(200));
                continue;
            }
        };

        let mut pph = [0u8; 32];
        pph.copy_from_slice(&s.pow_hash_header[..32]);
        let timestamp = u64::from_le_bytes(s.pow_hash_header[32..40].try_into().unwrap());
        let target_le = s.target.to_le_bytes();

        let start = s.nonce_fixed | (pom_nonce & s.nonce_mask);

        let t0 = std::time::Instant::now();
        let found = pom_gpu::mine(0, &pph, timestamp, &target_le, start, BATCH_SIZE);
        pom_nonce = (pom_nonce.wrapping_add(BATCH_SIZE)) & s.nonce_mask;

        let batches = BATCH_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if batches % HEARTBEAT_BATCHES == 0 {
            let secs = t0.elapsed().as_secs_f64().max(1e-6);
            let mhs = (BATCH_SIZE as f64 / secs) / 1e6;
            LAST_HASHRATE_MHS_BITS.store(mhs.to_bits(), Ordering::Relaxed);
            log_msg(&format!("android: mining… {batches} batches, {mhs:.2} MH/s"));
        }

        if let Some(winning_nonce) = found {
            log_msg(&format!("android: PoM winner nonce={winning_nonce}"));
            if let Some(seed) = s.generate_block_if_pom(winning_nonce, index.as_ref(), tier) {
                NONCES_FOUND.fetch_add(1, Ordering::Relaxed);
                let _ = submit_tx.blocking_send(seed);
            }
            state = None;
        } else {
            match job_rx.get_changed() {
                Ok(Some(ns)) => {
                    if ns.is_some() {
                        state = ns;
                    }
                }
                Ok(None) => {}
                Err(_) => break,
            }
        }
    }
    log_msg("android: mining worker exited");
}

fn stop_impl() {
    log_msg("android: stopping miner…");
    RUNNING.store(false, Ordering::SeqCst);
    if let Some(tx) = STOP_TX.get() {
        let _ = tx.send(true);
    }
}

fn status_impl() -> String {
    let running = RUNNING.load(Ordering::Relaxed);
    let nonces = NONCES_FOUND.load(Ordering::Relaxed);
    let hashrate_mhs = f64::from_bits(LAST_HASHRATE_MHS_BITS.load(Ordering::Relaxed));
    let log = LAST_LOG.get_or_init(|| Mutex::new(String::new()));
    let log_content = log.lock().ok().map(|l| l.clone()).unwrap_or_default();
    let last_lines: Vec<&str> = log_content.lines().rev().take(20).collect();
    let last_lines: Vec<&str> = last_lines.into_iter().rev().collect();
    format!(
        r#"{{"running":{},"nonces_found":{},"hashrate_mhs":{:.4},"log_lines":[{}]}}"#,
        running,
        nonces,
        hashrate_mhs,
        last_lines
            .iter()
            .map(|l| format!("\"{}\"", l.replace('\\', "\\\\").replace('"', "\\\"")))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// JNI boundary — thin type-marshalling shims over the `_impl` functions above. Function names
/// follow the `Java_<package>_<Class>_<method>` mangling JNI requires for
/// `com.keryx.miner.android.MinerBridge` (see `android-app/`).
mod jni_bridge {
    use super::*;
    use jni::objects::{JClass, JString};
    use jni::sys::{jboolean, jstring, JNI_FALSE, JNI_TRUE};
    use jni::JNIEnv;

    fn get_string(env: &mut JNIEnv, s: &JString) -> Option<String> {
        env.get_string(s).ok().map(|s| s.into())
    }

    #[no_mangle]
    pub extern "system" fn Java_com_keryx_miner_android_MinerBridge_nativeSetDocPath(
        mut env: JNIEnv,
        _class: JClass,
        path: JString,
    ) -> jboolean {
        match get_string(&mut env, &path) {
            Some(p) if set_doc_path_impl(&p) => JNI_TRUE,
            _ => JNI_FALSE,
        }
    }

    #[no_mangle]
    pub extern "system" fn Java_com_keryx_miner_android_MinerBridge_nativeInitialize(_env: JNIEnv, _class: JClass) -> jboolean {
        if initialize_impl() { JNI_TRUE } else { JNI_FALSE }
    }

    #[no_mangle]
    pub extern "system" fn Java_com_keryx_miner_android_MinerBridge_nativeConnect(
        mut env: JNIEnv,
        _class: JClass,
        address: JString,
    ) -> jboolean {
        match get_string(&mut env, &address) {
            Some(a) if connect_impl(&a) => JNI_TRUE,
            _ => JNI_FALSE,
        }
    }

    #[no_mangle]
    pub extern "system" fn Java_com_keryx_miner_android_MinerBridge_nativeSetMiningAddress(
        mut env: JNIEnv,
        _class: JClass,
        address: JString,
    ) -> jboolean {
        match get_string(&mut env, &address) {
            Some(a) if set_mining_address_impl(&a) => JNI_TRUE,
            _ => JNI_FALSE,
        }
    }

    #[no_mangle]
    pub extern "system" fn Java_com_keryx_miner_android_MinerBridge_nativeStart(_env: JNIEnv, _class: JClass) -> jboolean {
        if start_impl() { JNI_TRUE } else { JNI_FALSE }
    }

    #[no_mangle]
    pub extern "system" fn Java_com_keryx_miner_android_MinerBridge_nativeStop(_env: JNIEnv, _class: JClass) {
        stop_impl();
    }

    #[no_mangle]
    pub extern "system" fn Java_com_keryx_miner_android_MinerBridge_nativeStatus(env: JNIEnv, _class: JClass) -> jstring {
        match env.new_string(status_impl()) {
            Ok(s) => s.into_raw(),
            Err(_) => std::ptr::null_mut(),
        }
    }
}
