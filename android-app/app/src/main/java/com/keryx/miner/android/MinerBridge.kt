package com.keryx.miner.android

/**
 * JNI bridge to the Rust miner core (`keryx-miner`, compiled as a cdylib — see
 * `../../../../../../Cargo.toml`'s `[lib]` section and `src/android.rs`'s `jni_bridge` module for
 * the corresponding `Java_com_keryx_miner_android_MinerBridge_*` exports). Mirrors the iOS app's
 * `@_silgen_name` C-ABI declarations in `ios-app/Sources/ContentView.swift` one-to-one, just over
 * JNI instead of a raw C ABI.
 */
object MinerBridge {
    init {
        System.loadLibrary("keryx_miner")
    }

    external fun nativeSetDocPath(path: String): Boolean
    external fun nativeInitialize(): Boolean
    external fun nativeConnect(address: String): Boolean
    external fun nativeSetMiningAddress(address: String): Boolean
    external fun nativeStart(): Boolean
    external fun nativeStop()
    external fun nativeStatus(): String
}
