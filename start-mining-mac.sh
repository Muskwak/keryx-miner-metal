#!/usr/bin/env bash
#
# start-mining-mac.sh — Keryx miner, Apple Silicon (Metal) fork, with debug diagnostics.
#
# Edit the values in the CONFIG block below, then run:  ./start-mining-mac.sh
# (or pass overrides:  ./start-mining-mac.sh <wallet-address> <pool> [extra keryx-miner flags...])
#
# Mines Keryx Proof-of-Model on the Apple GPU via candle's Metal backend, plus the mandatory
# OPoI inference (this fork hard-gates mining on it — "no inference, no mining"). OPoI results are
# uploaded to a local IPFS daemon (Kubo API, default http://127.0.0.1:5001; the miner starts one).
#
set -euo pipefail

# ============================== CONFIG (edit me) ==============================

# --- Your Keryx payout address (bare wallet, no worker suffix) ---
MINING_ADDRESS="${MINING_ADDRESS:-keryx:REPLACE_ME_WITH_YOUR_KERYX_ADDRESS}"

# --- Worker/rig name, sent to the pool as `address.worker` via --worker (set to "" for none) ---
WORKER="${WORKER-mac}"

# --- Pool (stratum). These are OPoI-capable Keryx pools; port picks the difficulty tier. ---
POOL="${POOL:-stratum+tcp://krx.suprnova.cc:4404}"
# POOL="stratum+tcp://krx.suprnova.cc:4401"
# POOL="stratum+tcp://krx.baikalmine.com:9020"
# POOL="stratum+tcp://eu.miningcrib.com:7212"
# POOL="stratum+tcp://pool.ddsolutions.ai:5555"
# POOL="stratum+tcp://sg.keryx.dongqn.com:5555"
# Solo instead of a pool? Point at a local keryxd node, e.g. POOL="grpc://127.0.0.1:22110"

# --- Model tier + any extra flags. Default (empty) = Gemma-3-4B + Dolphin-8B (~16GB unified memory).
#     Smaller Macs: TIER=(--light) (Gemma-3-4B, ~8GB) or TIER=(--very-light) (Qwen3-1.7B, ~4GB).
#     Bigger: TIER=(--high) / TIER=(--very-high). CPU threads: TIER=(--threads 4). ---
TIER=()

# --- Debug diagnostics (what you asked for) ---
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"   # symbolized backtrace on panic (use `full` for std frames)
export RUST_LOG="${RUST_LOG:-debug}"           # env_logger level; scope it e.g. info,keryx_miner=debug if noisy

# =============================================================================

# CLI overrides: $1 = wallet address, $2 = pool. Args starting with `-` are passed to the miner.
if [[ "${1:-}" != "" && "${1:-}" != -* ]]; then MINING_ADDRESS="$1"; shift; fi
if [[ "${1:-}" != "" && "${1:-}" != -* ]]; then POOL="$1"; shift; fi

if [[ -z "$MINING_ADDRESS" || "$MINING_ADDRESS" == *REPLACE_ME* ]]; then
  echo "error: set MINING_ADDRESS in the script (or pass it as the first argument)." >&2
  echo "  usage: $0 <wallet-address> [pool] [extra keryx-miner flags...]" >&2
  exit 1
fi

# ── Locate the keryx-miner binary ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [[ -z "${KERYX_MINER_BIN:-}" ]]; then
  for cand in \
    "$SCRIPT_DIR/keryx-miner" \
    "$SCRIPT_DIR/target/aarch64-apple-darwin/release/keryx-miner" \
    "$SCRIPT_DIR/target/release/keryx-miner" \
    "$(command -v keryx-miner 2>/dev/null || true)"
  do
    if [[ -n "$cand" && -x "$cand" ]]; then KERYX_MINER_BIN="$cand"; break; fi
  done
fi
if [[ -z "${KERYX_MINER_BIN:-}" ]]; then
  echo "error: keryx-miner binary not found." >&2
  echo "  build it:  cargo build --release --target aarch64-apple-darwin --bin keryx-miner" >&2
  echo "  or set:    KERYX_MINER_BIN=/path/to/keryx-miner $0 ..." >&2
  exit 1
fi

# macOS: drop the download-quarantine flag so Gatekeeper won't block launch.
xattr -d com.apple.quarantine "$KERYX_MINER_BIN" 2>/dev/null || true

# ── Assemble args ────────────────────────────────────────────────────────────
args=( --debug --mining-address "$MINING_ADDRESS" --worker "$WORKER" --keryxd-address "$POOL" )
if [[ ${#TIER[@]} -gt 0 ]]; then args+=( "${TIER[@]}" ); fi
args+=( "$@" )   # remaining pass-through flags: --light / --high / --threads N / --testnet / ...

echo "──────────────────────────────────────────────────────────────────────"
echo " keryx-miner : $KERYX_MINER_BIN"
echo " address     : $MINING_ADDRESS"
echo " worker      : ${WORKER:-<none>}"
echo " pool/node   : $POOL"
echo " RUST_LOG=$RUST_LOG  RUST_BACKTRACE=$RUST_BACKTRACE"
echo "──────────────────────────────────────────────────────────────────────"

# ── Run, auto-restarting on exit (Ctrl-C stops) ──────────────────────────────
trap 'echo; echo "Stopped."; exit 0' INT TERM
while true; do
  if "$KERYX_MINER_BIN" "${args[@]}"; then code=0; else code=$?; fi
  [[ $code -eq 130 ]] && { echo; echo "Interrupted."; exit 0; }
  echo
  echo "keryx-miner exited (code $code). Restarting in 5s — press Ctrl-C to stop."
  sleep 5
done
