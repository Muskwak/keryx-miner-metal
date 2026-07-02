//! iOS C FFI bridge — exposes keryx-miner functions to the SwiftUI app.
//! The app links `libkeryx_miner.a` and calls these C-ABI functions.

use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

static GRPC_ADDRESS: OnceLock<String> = OnceLock::new();
static MINING_TIER: OnceLock<CString> = OnceLock::new();
static RUNNING: AtomicBool = AtomicBool::new(false);
static NONCES_FOUND: AtomicU64 = AtomicU64::new(0);
static LAST_LOG: OnceLock<std::sync::Mutex<String>> = OnceLock::new();

fn log_msg(msg: &str) {
    let log = LAST_LOG.get_or_init(|| std::sync::Mutex::new(String::new()));
    if let Ok(mut log) = log.lock() {
        log.push_str(msg);
        log.push('\n');
        if log.len() > 65536 {
            *log = log.split_off(log.len() - 32768);
        }
    }
}

// ── C FFI exports ──────────────────────────────────────────────────────────

/// Initialise the miner with a gRPC address (e.g. `192.168.1.100:22110`).
/// Call once before `keryx_miner_start`.
#[no_mangle]
pub extern "C" fn keryx_miner_connect(address: *const std::ffi::c_char) -> bool {
    let c_str = unsafe { CStr::from_ptr(address) };
    let addr = match c_str.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return false,
    };
    let _ = GRPC_ADDRESS.set(addr);
    log_msg(&format!("keryx-miner: connected to {address:?}"));
    true
}

/// Start mining in the background.
#[no_mangle]
pub extern "C" fn keryx_miner_start() -> bool {
    if RUNNING.swap(true, Ordering::SeqCst) {
        return false; // already running
    }
    let address = match GRPC_ADDRESS.get() {
        Some(a) => a.clone(),
        None => return false,
    };
    log_msg("keryx-miner: starting…");
    std::thread::spawn(move || {
        // TODO: spawn tokio runtime and launch the actual mining loop.
        // For now, a stub that simulates mining.
        log_msg(&format!("keryx-miner: mining against {address}"));
        let mut nonce: u64 = 0;
        while RUNNING.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_secs(1));
            nonce += 1;
            NONCES_FOUND.store(nonce, Ordering::Relaxed);
        }
        log_msg("keryx-miner: stopped");
    });
    true
}

/// Stop the miner.
#[no_mangle]
pub extern "C" fn keryx_miner_stop() {
    log_msg("keryx-miner: stopping…");
    RUNNING.store(false, Ordering::SeqCst);
}

/// Return current status as a JSON string (caller must free with `keryx_miner_free_string`).
#[no_mangle]
pub extern "C" fn keryx_miner_status() -> *mut std::ffi::c_char {
    let running = RUNNING.load(Ordering::Relaxed);
    let nonces = NONCES_FOUND.load(Ordering::Relaxed);
    let log = LAST_LOG.get_or_init(|| std::sync::Mutex::new(String::new()));
    let log = log.lock().ok().map(|l| l.clone()).unwrap_or_default();
    let last_lines: Vec<&str> = log.lines().rev().take(10).collect();
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

/// Free a string returned by `keryx_miner_status`.
#[no_mangle]
pub extern "C" fn keryx_miner_free_string(s: *mut std::ffi::c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)); }
    }
}
