plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.keryx.miner.android"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.keryx.miner.android"
        // Vulkan 1.2 (bufferDeviceAddress + shaderInt64) is broadly available from API 26
        // (Android 8.0) onward across mainstream Adreno/Mali drivers.
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.3.7"
    }

    // Android refuses to install any unsigned APK, even via sideloading (unlike iOS's unsigned
    // .ipa + on-device AltStore resign flow) — sign "release" with the auto-generated debug key
    // so CI output is installable without a dedicated release keystore. There's no Play Store
    // distribution here, so a stable release signature isn't needed.
    signingConfigs {
        getByName("debug") {}
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("debug")
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.14"
    }

    packaging {
        // Avoid duplicate-file build failures some native/Kotlin deps trigger.
        resources.excludes.add("META-INF/*")
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation(platform("androidx.compose:compose-bom:2024.06.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.material3:material3")
}

// ── Rust native library ─────────────────────────────────────────────────────
//
// The Rust cdylib (crate `keryx-miner`, lib name `keryx_miner` — see ../Cargo.toml's [lib]
// section) is built with `cargo ndk` (https://github.com/bbqsrc/cargo-ndk, `cargo install
// cargo-ndk`) rather than a plain `cargo build --target aarch64-linux-android`, because it
// handles the NDK's clang/linker paths and API-level flags for you. Requires the Android NDK
// installed (ANDROID_NDK_HOME set, or discoverable via the Android SDK) and `rustup target add
// aarch64-linux-android`.
//
// This task is NOT wired into `preBuild` automatically (a multi-minute native build shouldn't
// silently fire on every Gradle sync) — run it explicitly before a device/CI build:
//   ./gradlew :app:cargoNdkBuild
// then a normal assembleDebug/assembleRelease picks up the .so from jniLibs/.
tasks.register<Exec>("cargoNdkBuild") {
    workingDir = file("../..") // the keryx-miner crate root (one level above android-app/)
    commandLine(
        "cargo", "ndk",
        "-o", file("src/main/jniLibs").absolutePath,
        "-t", "arm64-v8a",
        "-P", "26", // match android.defaultConfig.minSdk above
        // --lib only: the workspace's [[bin]] (src/main.rs) calls desktop/CUDA-only functions
        // that are cfg'd out on Android.
        "build", "--release", "--lib",
    )
}
