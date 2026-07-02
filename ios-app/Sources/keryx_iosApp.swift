import SwiftUI

@main
struct KeryxMinerApp: App {
    init() {
        // Point the Rust side at our sandbox so model downloads land in
        // Documents/keryx-models instead of failing to resolve a bundle path.
        if let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            _ = docs.path.withCString { keryx_miner_set_doc_path($0) }
        }

        // Kick off the --very-light model download in the background so it's
        // ready (or well underway) by the time the user taps Start — this is
        // a synchronous, potentially multi-minute network call on the Rust
        // side, so it must not run on the main thread.
        DispatchQueue.global(qos: .utility).async {
            _ = keryx_miner_initialize()
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
