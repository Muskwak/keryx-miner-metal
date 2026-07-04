import SwiftUI

// Rust FFI declarations
@_silgen_name("keryx_miner_set_doc_path")
func keryx_miner_set_doc_path(_ path: UnsafePointer<CChar>) -> Bool

@_silgen_name("keryx_miner_initialize")
func keryx_miner_initialize() -> Bool

@_silgen_name("keryx_miner_connect")
func keryx_miner_connect(_ address: UnsafePointer<CChar>) -> Bool

@_silgen_name("keryx_miner_set_mining_address")
func keryx_miner_set_mining_address(_ address: UnsafePointer<CChar>) -> Bool

@_silgen_name("keryx_miner_start")
func keryx_miner_start() -> Bool

@_silgen_name("keryx_miner_stop")
func keryx_miner_stop()

@_silgen_name("keryx_miner_status")
func keryx_miner_status() -> UnsafeMutablePointer<CChar>?

@_silgen_name("keryx_miner_free_string")
func keryx_miner_free_string(_ s: UnsafeMutablePointer<CChar>?)

struct ContentView: View {
    @State private var grpcAddress: String = "127.0.0.1:22110"
    @State private var miningAddress: String = ""
    @State private var isMining: Bool = false
    @State private var hashrateMhs: Double = 0.0
    @State private var logLines: [String] = ["keryx-miner iOS — ready"]
    @State private var statusTimer: Timer?

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                // gRPC address input
                VStack(alignment: .leading, spacing: 4) {
                    Text("gRPC Address").font(.caption).foregroundColor(.secondary)
                    TextField("host:port", text: $grpcAddress)
                        .textFieldStyle(.roundedBorder)
                        .autocapitalization(.none)
                        .disableAutocorrection(true)
                        .disabled(isMining)
                }
                .padding(.horizontal)

                // Mining address input
                VStack(alignment: .leading, spacing: 4) {
                    Text("Mining Address (wallet)").font(.caption).foregroundColor(.secondary)
                    TextField("keryx:...", text: $miningAddress)
                        .textFieldStyle(.roundedBorder)
                        .autocapitalization(.none)
                        .disableAutocorrection(true)
                        .disabled(isMining)
                }
                .padding(.horizontal)

                // Start / Stop button
                Button(action: {
                    if isMining {
                        stopMining()
                    } else {
                        startMining()
                    }
                }) {
                    HStack {
                        Image(systemName: isMining ? "stop.circle.fill" : "play.circle.fill")
                        Text(isMining ? "Stop Mining" : "Start Mining")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(isMining ? Color.red : Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .padding(.horizontal)

                if isMining {
                    Text(String(format: "Hashrate: %.4f MH/s", hashrateMhs))
                        .font(.callout)
                        .padding(.horizontal)
                }

                // Log output
                VStack(alignment: .leading, spacing: 2) {
                    Text("Log").font(.caption).foregroundColor(.secondary)
                    ScrollViewReader { scrollView in
                        ScrollView {
                            VStack(alignment: .leading, spacing: 1) {
                                ForEach(Array(logLines.enumerated()), id: \.offset) { _, line in
                                    Text(line)
                                        .font(.system(.caption2, design: .monospaced))
                                        .foregroundColor(.green)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .background(Color.black.opacity(0.05))
                        .cornerRadius(8)
                    }
                }
                .padding(.horizontal)

                Spacer()
            }
            .padding(.vertical)
            .navigationTitle("Keryx Miner")
            .navigationBarTitleDisplayMode(.inline)
        }
        .onAppear {
            // Status polling (and the model download it reports on) runs for the
            // whole app lifetime, independent of Start/Stop — the --very-light
            // model is fetched once at launch by KeryxMinerApp.init().
            if statusTimer == nil {
                statusTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
                    pollStatus()
                }
            }
        }
    }

    func startMining() {
        let addr = grpcAddress.trimmingCharacters(in: .whitespaces)
        guard !addr.isEmpty else {
            logLines.append("ERROR: enter a gRPC address first")
            return
        }
        guard addr.withCString({ ptr in keryx_miner_connect(ptr) }) else {
            logLines.append("ERROR: keryx_miner_connect failed")
            return
        }

        let wallet = miningAddress.trimmingCharacters(in: .whitespaces)
        if !wallet.isEmpty {
            _ = wallet.withCString({ ptr in keryx_miner_set_mining_address(ptr) })
        }

        guard keryx_miner_start() else {
            logLines.append("ERROR: keryx_miner_start failed (already running?)")
            return
        }
        isMining = true
        logLines.append("Mining started — gRPC: \(addr)")
    }

    func stopMining() {
        keryx_miner_stop()
        isMining = false
        logLines.append("Mining stopped")
        pollStatus()
    }

    func pollStatus() {
        guard let ptr = keryx_miner_status() else { return }
        let json = String(cString: ptr)
        keryx_miner_free_string(ptr)
        if let data = json.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let running = obj["running"] as? Bool,
           let nonces = obj["nonces_found"] as? UInt64,
           let lines = obj["log_lines"] as? [String] {
            isMining = running
            hashrateMhs = obj["hashrate_mhs"] as? Double ?? 0.0
            logLines = Array(lines.suffix(20))
            if nonces > 0 {
                logLines.append("Nonces found: \(nonces)")
            }
        }
    }
}
