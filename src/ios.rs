use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
    RpcBlock, SubmitBlockRequestMessage,
};

static GRPC_ADDRESS: OnceLock<String> = OnceLock::new();
static MINING_ADDRESS: OnceLock<String> = OnceLock::new();
static NONCES_FOUND: AtomicU64 = AtomicU64::new(0);
static LAST_LOG: OnceLock<Mutex<String>> = OnceLock::new();
static RUNNING: AtomicBool = AtomicBool::new(false);
static STOP_TX: OnceLock<watch::Sender<bool>> = OnceLock::new();

const BATCH_SIZE: u64 = 1 << 20;

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

    let (req_tx, req_rx) = tokio::sync::mpsc::channel::<KaspadMessage>(64);
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
                pay_address: mining_addr.clone(),
                extra_data: format!("keryx-miner-ios/{}", env!("CARGO_PKG_VERSION")),
                inference_result: String::new(),
            })),
        })
        .await;

    let mut current_block: Option<RpcBlock> = None;
    let mut nonce: u64 = 0;

    loop {
        tokio::select! {
            msg = response.message() => {
                match msg {
                    Ok(Some(m)) => {
                        if let Some(payload) = m.payload {
                            match payload {
                                Payload::GetBlockTemplateResponse(r) => {
                                    if let Some(block) = r.block {
                                        log_msg(&format!("ios: got block template DAA={}", block.header.as_ref().map(|h| h.daa_score).unwrap_or(0)));
                                        // Reset nonce for the new block
                                        current_block = Some(block);
                                        nonce = 0;
                                    }
                                }
                                Payload::NewBlockTemplateNotification(_) => {
                                    log_msg("ios: new block template available, requesting…");
                                    let _ = req_tx.send(KaspadMessage {
                                        payload: Some(Payload::GetBlockTemplateRequest(GetBlockTemplateRequestMessage {
                                            pay_address: mining_addr.clone(),
                                            extra_data: format!("keryx-miner-ios/{}", env!("CARGO_PKG_VERSION")),
                                            inference_result: String::new(),
                                        })),
                                    }).await;
                                }
                                Payload::BlockAddedNotification(n) => {
                                    if let Some(block) = n.block {
                                        let daa = block.header.as_ref().map(|h| h.daa_score).unwrap_or(0);
                                        log_msg(&format!("ios: block added DAA={}", daa));
                                    }
                                }
                                Payload::SubmitBlockResponse(r) => {
                                    let reject = r.reject_reason();
                                    log_msg(&format!("ios: submit block response: reject={:?}", reject));
                                }
                                Payload::NotifyNewBlockTemplateResponse(_) => {}
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

        // Mine on the current block if available
        if let Some(ref block) = current_block {
            if let Some(ref header) = block.header {
                let daa_score = header.daa_score;

                // Create State for PoM mining
                let state = match crate::pow::State::new(0, crate::pow::BlockSeed::FullBlock(Box::new(block.clone()))) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                // Build/ensure PoM index
                pom_gpu::ensure_installed(0, daa_score);

                let (index, tier) = match pom::active_index() {
                    Some(x) => x,
                    None => {
                        // Index not ready yet — try again next iteration
                        continue;
                    }
                };

                let mut pph = [0u8; 32];
                pph.copy_from_slice(&state.pow_hash_header[..32]);
                let timestamp = u64::from_le_bytes(state.pow_hash_header[32..40].try_into().unwrap());
                let target_le = state.target.to_le_bytes();

                let batch_start = nonce;
                let found = pom_gpu::mine(0, &pph, timestamp, &target_le, batch_start, BATCH_SIZE);
                nonce = nonce.wrapping_add(BATCH_SIZE);

                if let Some(winning_nonce) = found {
                    log_msg(&format!("ios: PoM winning nonce found: {}", winning_nonce));
                    if let Some(block_seed) = state.generate_block_if_pom(winning_nonce, index, *tier) {
                        match block_seed {
                            crate::pow::BlockSeed::FullBlock(found_block) => {
                                NONCES_FOUND.fetch_add(1, Ordering::Relaxed);
                                log_msg("ios: submitting block…");
                                let _ = req_tx
                                    .send(KaspadMessage {
 payload: Some(Payload::SubmitBlockRequest(SubmitBlockRequestMessage {
 block: Some(*found_block.clone()),
 allow_non_daa_blocks: false,
 })),
                                    })
                                    .await;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    log_msg("ios: mining loop ended");
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
