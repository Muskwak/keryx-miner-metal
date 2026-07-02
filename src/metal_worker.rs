//! Built-in Metal GPU worker for macOS desktop mining.
//!
//! On CUDA/OpenCL platforms the GPU worker threads are provided by dynamically
//! loaded plugins (which enumerate devices and build `Worker`s). macOS has no
//! such plugin, so the desktop binary otherwise launches zero GPU workers and
//! aborts with "No workers specified".
//!
//! Mining is PoM-only: `MinerManager::launch_gpu_miner`'s loop takes the PoM
//! branch whenever `daa >= POM_ACTIVATION_DAA` and drives `pom_gpu::mine`
//! directly, using only `Worker::id()` to derive the device index. It never
//! calls the kHeavyHash methods below, so they are inert stubs (no-ops rather
//! than `unreachable!` so that even the legacy branch, if ever reached, simply
//! finds nothing instead of panicking).

use crate::{Error, Worker, WorkerSpec};

/// Device label consumed by `launch_gpu_miner`: it strips the leading `#` and
/// parses the first whitespace-separated token as the device id → `0`.
const METAL_DEVICE_ID: &str = "#0 Apple GPU (Metal)";

pub struct MetalWorkerSpec;

impl WorkerSpec for MetalWorkerSpec {
    fn id(&self) -> String {
        METAL_DEVICE_ID.to_string()
    }

    fn build(&self) -> Box<dyn Worker> {
        Box::new(MetalWorker)
    }
}

pub struct MetalWorker;

impl Worker for MetalWorker {
    fn id(&self) -> String {
        METAL_DEVICE_ID.to_string()
    }

    // ── Legacy kHeavyHash interface — unused under PoM, kept as safe no-ops ──
    fn load_block_constants(&mut self, _hash_header: &[u8; 72], _matrix: &[[u16; 64]; 64], _target: &[u64; 4]) {}

    fn calculate_hash(&mut self, _nonces: Option<&Vec<u64>>, _nonce_mask: u64, _nonce_fixed: u64) {}

    fn sync(&self) -> Result<(), Error> {
        Ok(())
    }

    fn get_workload(&self) -> usize {
        0
    }

    fn copy_output_to(&mut self, _nonces: &mut Vec<u64>) -> Result<(), Error> {
        Ok(())
    }
}
