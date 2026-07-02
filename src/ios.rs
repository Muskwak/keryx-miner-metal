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
    SubmitBlockRequestMessage, SubmitBlockResponseMessage, RpcBlock,
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

fn model_dir() -> std::path::PathBuf {
    let mut p = std::env::current_exe().unwrap_or_default();
    p.pop();
    p.push("models");
    p
}

fn download_model(spec: &ModelSpec) -> bool {
    let dir = model_dir();
    let ok_file = dir.join(".ok");
    if ok_file.exists() {
        log_msg(&format!("ios: model '{}' already downloaded", spec.name));
        return true;
    }
    let _ = std::fs::create_dir_all(&dir);

    let model_url = format!("https://keryx-labs.com/models/{}", spec.dir_name);
    log_msg(&format!("ios: downloading model '{}' from {} …", spec.name, model_url));
    let response = match ureq::get(&model_url).call() {
        Ok(r) => r,
        Err(e) => {
            log_msg(&format!("ios: model download failed: {e}"));
            return false;
        }
    };
    let len: usize = response.header("Content-Length").and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut body: Vec<u8> = Vec::with_capacity(len);
    let mut reader = response.into_reader();
    if let Err(e) = std::io::copy(&mut reader, &mut body) {
        log_msg(&format!("ios: model download read error: {e}"));
        return false;
    }

    let gguf_path = dir.join(format!("{}.gguf", spec.dir_name));
    if let Err(e) = std::fs::write(&gguf_path, &body) {
        log_msg(&format!("ios: model write error: {e}"));
        return false;
    }
    let _ = std::fs::write(&ok_file, b"ok");
    log_msg(&format!("ios: model '{}' downloaded ({:.1} MB)", spec.name, body.len() as f64 / 1e6));
    true
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

    // Pre-download the very-light model if needed
    let very_light_specs: &[&ModelSpec] = models::specs_for(VERY_LIGHT_ACTIVATION_DAA, Tier::VeryLight);
    if !very_light_specs.is_empty() {
        let spec = very_light_specs[0];
        if !download_model(spec) {
            log_msg("ios: FAILED to download model — cannot mine");
            RUNNING.store(false, Ordering::SeqCst);
            return false;
        }

        let gguf_path = model_dir().join(format!("{}.gguf", spec.dir_name));
        pom_gpu::set_mining_tier(spec.model_id, gguf_path.to_string_lossy().into_owned());
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
                    if let Some(block_seed) = state.generate_block_if_pom(winning_nonce, index, tier) {
                        match block_seed {
                            crate::pow::BlockSeed::FullBlock(found_block) => {
                                NONCES_FOUND.fetch_add(1, Ordering::Relaxed);
                                log_msg("ios: submitting block…");
                                let _ = req_tx
                                    .send(KaspadMessage {
                                        payload: Some(Payload::SubmitBlockRequest(SubmitBlockRequestMessage {
                                            block: *found_block.clone(),
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
