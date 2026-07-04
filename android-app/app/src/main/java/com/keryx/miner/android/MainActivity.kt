package com.keryx.miner.android

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * Single-screen miner UI — a Kotlin/Compose port of the iOS app's `ContentView.swift`: address +
 * mining-address inputs, a start/stop button, and a scrolling log fed by polling
 * [MinerBridge.nativeStatus] every 2 seconds. Accepts either a bare `host:port` (solo gRPC) or a
 * `stratum+tcp://host:port` pool address — the Rust side picks the transport from the scheme.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Point the Rust side at our app-private storage so model downloads land in
        // filesDir/keryx-models — the Android analogue of iOS's sandboxed Documents URL.
        MinerBridge.nativeSetDocPath(filesDir.absolutePath)

        // Kick off the --very-light model download in the background so it's ready (or well
        // underway) by the time the user taps Start. This is a synchronous, potentially
        // multi-minute network call on the Rust side, so it must not run on the main thread.
        Thread { MinerBridge.nativeInitialize() }.apply { isDaemon = true; start() }

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    MinerScreen()
                }
            }
        }
    }
}

@Composable
fun MinerScreen() {
    var address by remember { mutableStateOf("stratum+tcp://127.0.0.1:22110") }
    var miningAddress by remember { mutableStateOf("") }
    var isMining by remember { mutableStateOf(false) }
    var hashrateMhs by remember { mutableStateOf(0.0) }
    var logLines by remember { mutableStateOf(listOf("keryx-miner Android — ready")) }
    val scope = rememberCoroutineScope()
    val scrollState = rememberScrollState()

    // Status polling (and the model download it reports on) runs for the whole app lifetime,
    // independent of Start/Stop — the --very-light model is fetched once at launch.
    LaunchedEffect(Unit) {
        while (true) {
            delay(2000)
            val json = withContext(Dispatchers.IO) { MinerBridge.nativeStatus() }
            runCatching {
                val obj = JSONObject(json)
                isMining = obj.getBoolean("running")
                hashrateMhs = obj.optDouble("hashrate_mhs", 0.0)
                val lines = obj.getJSONArray("log_lines")
                val parsed = (0 until lines.length()).map { lines.getString(it) }
                val nonces = obj.optLong("nonces_found", 0)
                logLines = if (nonces > 0) parsed + "Nonces found: $nonces" else parsed
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("Keryx Miner", style = MaterialTheme.typography.titleLarge)

        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Pool / gRPC Address", style = MaterialTheme.typography.labelSmall)
            OutlinedTextField(
                value = address,
                onValueChange = { address = it },
                enabled = !isMining,
                singleLine = true,
                keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.None),
                placeholder = { Text("stratum+tcp://host:port or host:port") },
                modifier = Modifier.fillMaxWidth(),
            )
        }

        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Mining Address (wallet)", style = MaterialTheme.typography.labelSmall)
            OutlinedTextField(
                value = miningAddress,
                onValueChange = { miningAddress = it },
                enabled = !isMining,
                singleLine = true,
                keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.None),
                placeholder = { Text("keryx:...") },
                modifier = Modifier.fillMaxWidth(),
            )
        }

        Button(
            onClick = {
                if (isMining) {
                    MinerBridge.nativeStop()
                    isMining = false
                    logLines = logLines + "Mining stopped"
                } else {
                    val addr = address.trim()
                    if (addr.isEmpty()) {
                        logLines = logLines + "ERROR: enter an address first"
                        return@Button
                    }
                    scope.launch {
                        val ok = withContext(Dispatchers.IO) {
                            val connected = MinerBridge.nativeConnect(addr)
                            val wallet = miningAddress.trim()
                            if (wallet.isNotEmpty()) MinerBridge.nativeSetMiningAddress(wallet)
                            connected && MinerBridge.nativeStart()
                        }
                        if (ok) {
                            isMining = true
                            logLines = logLines + "Mining started — $addr"
                        } else {
                            logLines = logLines + "ERROR: start failed (already running, or bad address?)"
                        }
                    }
                }
            },
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isMining) Color(0xFFD32F2F) else Color(0xFF2E7D32),
            ),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (isMining) "Stop Mining" else "Start Mining")
        }

        if (isMining) {
            Text(
                "Hashrate: %.4f MH/s".format(hashrateMhs),
                style = MaterialTheme.typography.bodyMedium,
            )
        }

        Column(
            verticalArrangement = Arrangement.spacedBy(2.dp),
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
        ) {
            Text("Log", style = MaterialTheme.typography.labelSmall)
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.05f))
                    .verticalScroll(scrollState)
                    .padding(8.dp),
            ) {
                logLines.takeLast(20).forEach { line ->
                    Text(
                        text = line,
                        fontFamily = FontFamily.Monospace,
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFF2E7D32),
                    )
                }
            }
        }
    }
}
