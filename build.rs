use std::env;
use time::{format_description, OffsetDateTime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let format = format_description::parse_borrowed::<2>("[year repr:last_two][month][day][hour][minute]")?;
    let dt = OffsetDateTime::now_utc().format(&format)?;
    println!("cargo:rustc-env=PACKAGE_COMPILE_TIME={}", dt);

    println!("cargo:rerun-if-changed=proto");
    println!("cargo:rerun-if-changed=src/keccakf1600_x86-64.s");
    tonic_build::configure()
        .build_server(false)
        .compile(
            &["proto/rpc.proto", "proto/p2p.proto", "proto/messages.proto"],
            &["proto"],
        )?;

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    // PoM mining kernel → PTX (loaded at runtime into candle's CUDA context). nvcc 12.2.
    // On macOS & iOS the Metal kernel (cuda/pom_mine.metal) is compiled at runtime via
    // the Metal framework's in-process compiler; we only track changes for rebuild triggers.
    let is_apple = target_os == "macos" || target_os == "ios";
    if is_apple {
        println!("cargo:rerun-if-changed=cuda/pom_mine.metal");
    }
    if !is_apple {
        println!("cargo:rerun-if-changed=cuda/pom_mine.cu");
        let out_dir = env::var("OUT_DIR").unwrap();
        let nvcc = env::var("NVCC").ok().unwrap_or_else(|| {
            let pinned = "/home/slash/cuda-12.2/bin/nvcc";
            if std::path::Path::new(pinned).exists() { pinned.to_string() } else { "nvcc".to_string() }
        });
        let sm = env::var("SM_ARCH").unwrap_or_else(|_| "86".to_string());
        let ptx = format!("{out_dir}/pom_mine.ptx");
        let status = std::process::Command::new(&nvcc)
            .args(["-ptx", "-O3", &format!("-arch=sm_{sm}"), "cuda/pom_mine.cu", "-o", &ptx])
            .status()
            .unwrap_or_else(|e| panic!("nvcc ({nvcc}) failed to run: {e}"));
        assert!(status.success(), "nvcc failed to compile cuda/pom_mine.cu");
    }

    // Keccak-f1600 assembly: x86_64 only. On ARM64 (Apple Silicon) the Rust keccak crate
    // is used instead (activated as a dependency in Cargo.toml for aarch64 macOS).
    if target_arch == "x86_64" && target_os != "windows" && target_os != "macos" {
        cc::Build::new().flag("-c").file("src/keccakf1600_x86-64.s").compile("libkeccak.a");
    }
    if target_arch == "x86_64" && target_os == "macos" {
        cc::Build::new().flag("-c").file("src/keccakf1600_x86-64-osx.s").compile("libkeccak.a");
    }
    Ok(())
}
