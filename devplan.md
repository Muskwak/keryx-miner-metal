# keryx-miner Apple (macOS + iOS) Development Plan

## Goal
Build keryx-miner for Apple Silicon (M-series) with Metal backend for both OPoI inference and PoM GPU mining, targeting **macOS** (`aarch64-apple-darwin`) and **iOS** (`aarch64-apple-ios`) from the same Rust codebase.

## Progress Log

### 2026-07-01 — Session 1: Local Edits (Windows)
- Analyzed original Keryx-Labs/keryx-miner, shmutalov fork (metal inference gating), Muskwak RDNA3 fork (Vulkan PoM approach)
- Cloned shmutalov fork locally at `C:\Users\ADMIN\AppData\Local\Temp\opencode\keryx-metal`
- **build.rs**: guarded nvcc compilation behind `target_os != "macos"` check, added `cargo:rerun-if-changed=cuda/pom_mine.metal` for macOS
- **Cargo.toml**: added `metal = "0.27"` under `[target.'cfg(target_os = "macos")'.dependencies]`
- **Wrote `cuda/pom_mine.metal`** — complete MSL kernel with mix64, pom_seed_fold, pom_pow_fold, pom_le_leq, pom_mine kernel using all_data buffer + base_offsets + prefix + PomParams constant struct + atomic_uint64_t winner
- **Rewrote `pom_gpu.rs` macOS section** — replaced stubs with full Metal `PomGpuMiner` struct implementing load(), load_shared(), mine(), and all registry functions (install, uninstall, is_installed, mine, ensure_installed, ensure_installed_inner)
- Fixed API bugs: `get_function` missing None arg, raw `0` → `MTLResourceOptions::StorageModeShared`, `.ok_or_else` → `.map_err`, added `use candle_core::Device` import

### 2026-07-01 — Session 2: iOS Cross-Compile + CI + SwiftUI App
- Verified: cross-compiling macOS/iOS from Windows is impossible (requires Apple SDK + Metal framework)
- **Created `.github/workflows/build-apple.yml`** — macOS GitHub Actions workflow:
  - `build-macos`: cargo build --release --bin keryx-miner, strip, upload artifact
  - `build-ios-lib`: rustup target add aarch64-apple-ios, cargo build --release --target aarch64-apple-ios --lib, upload .a
  - `build-ios-app`: xcodebuild archive → unsigned .ipa, upload artifact
  - `test-macos`: cargo test --release
- **Updated `Cargo.toml`** for iOS:
  - Added `crate-type = ["bin", "staticlib"]`
  - Moved `libloading` and `nix` behind `cfg(not(target_os = "ios"))`
  - Changed `cfg(target_os = "macos")` → `cfg(any(target_os = "macos", target_os = "ios"))` for Metal deps
  - Changed `cfg(not(target_os = "macos"))` → `cfg(not(any(target_os = "macos", target_os = "ios")))` for CUDA deps
  - Added keccak dep for `aarch64` + `ios`
- **Updated `build.rs`**: extracted `is_apple` helper for macOS+iOS checks
- **Updated `src/pom_gpu.rs`**: replaced all `cfg(target_os = "macos")` with `cfg(any(target_os = "macos", target_os = "ios"))`, and `cfg(not(...))` accordingly
- **Updated `src/slm.rs`**: same cfg replacement for inference device creation
- **Created `src/ios.rs`** — C FFI bridge exposing `keryx_miner_connect()`, `keryx_miner_start()`, `keryx_miner_stop()`, `keryx_miner_status()`, callback-based log emission
- **Updated `src/main.rs`**: added `#[cfg(not(target_os = "ios"))]` guard on main function + platform-specific main gating so the staticlib compiles without a main()
  - **Created `ios-app/Sources/`** — SwiftUI source files (App.swift, ContentView.swift, Bridge.h)
  - **Created `ios-app/Info.plist`** — app metadata with `metal` capability
  - **Created `ios-app/project.yml`** — xcodegen project spec to generate Xcode project in CI
  - CI workflow updated to run `brew install xcodegen && xcodegen generate` before `xcodebuild`

### Source Changes Summary (Platform Gating)

| File | Change |
|---|---|
| `.github/workflows/build-apple.yml` | New — macOS binary, iOS staticlib, iOS app, macOS tests |
| `Cargo.toml` | `crate-type = ["bin", "staticlib"]`; gated `libloading` / `nix` behind `not(ios)`; CUDA → `not(any(macos, ios))`; Metal → `any(macos, ios)`; keccak for aarch64 apple |
| `build.rs` | Extracted `is_apple` helper for macOS+iOS checks |
| `src/lib.rs` | Gated `inference`, `quantized_llama_split`, `quantized_qwen3_split` behind `not(ios)`; added `ios` module for iOS |
| `src/ios.rs` | New — C FFI bridge: connect, start, stop, status, free_string |
| `src/pom_gpu.rs` | Replaced all `cfg(target_os = "macos")` with `cfg(any(target_os = "macos", target_os = "ios"))` and `cfg(not(...))` accordingly |
| `src/slm.rs` | Gated split loader imports, enum variants, if-blocks, and match arms behind `not(ios)` |
| `src/main.rs` | No changes needed (not compiled on iOS `--lib` build) |

## Remaining Work (Next Session)

### Critical
- **Test on Mac**: Pull repo on macOS, `cargo build --bin keryx-miner` — verify Metal PoM works
- **Test on iPhone 13**: Xcode archive + deploy (--very-light tier, Qwen3-1.7B ~1-2GB GGUF fits in 4GB RAM)
- **Inference metal backend**: candle-core with `features = ["metal"]` must work on iOS — verify metal-rs 0.27 compiles for `aarch64-apple-ios`
- **GGUF + model download**: The iOS app needs document-dir model storage. Currently the download functions in slm.rs write to `exe_dir/models/`. On iOS, need to redirect to `NSDocumentDirectory` (passed from Swift)

### Known Issues
- **`metal` crate deprecated**: replaced by `objc2-metal`, but candle-core 0.9 still uses `metal` 0.27 — ok until candle bumps
- **Stratum codec uses tokio TcpStream**: iOS supports BSD sockets via Network.framework but tokio's `net` feature uses POSIX sockets + kqueue. Should work on iOS (Darwin kernel). If not, need a feature-gated net provider.
- **`ureq` with TLS**: uses native-tls (Security.framework on iOS). Should work.
- **`nix` crate**: not available on iOS, gated away with `cfg(not(target_os = "ios"))`. Freeze handler is a no-op on iOS.
- **Workspace members**: `plugins/cuda` and `plugins/opencl` won't compile for iOS. Default-members excludes them when iOS target is active via cfg.

### iOS-Specific Edge Cases
- **App backgrounding**: If user locks phone or switches apps, Metal compute may be paused. Need `beginBackgroundTask` or keep phone awake with `UIApplication.isIdleTimerDisabled = true`
- **Thermal throttling**: iPhone 13 A15 may throttle after sustained Metal compute. Monitor thermal state, pause mining if `ProcessInfo.thermalState == .critical`
- **App Store compliance**: unsigned .ipa can be side-loaded via AltStore or Xcode. Not App Store compatible (no code signing in CI)

## Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| iOS UI framework | SwiftUI | Minimum boilerplate, native look |
| Rust→Swift bridge | C FFI via staticlib | Simple, reliable, no cbindgen needed |
| Metal shader compilation | Runtime (newLibraryWithSource) | Avoids xcrun metallib step, +100ms startup |
| Model storage on iOS | NSDocumentDirectory from Swift | slm.rs download must accept dynamic path |
| CI | GitHub Actions macos-latest | Free for public repos |
