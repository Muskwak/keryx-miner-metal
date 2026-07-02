// Keryx Proof-of-Model mining kernel — Metal Shading Language.
// Byte-identical semantics to pom_mine.cu and src/pom.rs build_proof.
// Per nonce: seed-fold + data-dependent gather walk over the resident weight
// blob (single all_data buffer, indexed via base_offsets) + pow-fold + target check.

#include <metal_stdlib>
using namespace metal;

// ── Primitives (identical to pom_mine.cu and src/pom.rs) ────────────────────

static inline ulong mix64(ulong x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline ulong pom_seed_fold(ulong nonce, ulong time_,
                                  ulong p0, ulong p1, ulong p2, ulong p3) {
    ulong s = mix64(nonce ^ 0x4B65727978531ULL);
    s = mix64(s ^ time_);
    s = mix64(s ^ p0); s = mix64(s ^ p1); s = mix64(s ^ p2); s = mix64(s ^ p3);
    return s;
}

static inline void pom_pow_fold(ulong fin,
                                ulong p0, ulong p1, ulong p2, ulong p3,
                                thread ulong out[4]) {
    out[0] = mix64(fin ^ p0 ^ 0x9E3779B97F4A7C15ULL);
    out[1] = mix64(out[0] ^ p1 ^ 0xC2B2AE3D27D4EB4FULL);
    out[2] = mix64(out[1] ^ p2 ^ 0x165667B19E3779F9ULL);
    out[3] = mix64(out[2] ^ p3 ^ 0xD6E8FEB86659FD93ULL);
}

static inline bool pom_le_leq(const thread ulong a[4],
                               ulong b0, ulong b1, ulong b2, ulong b3) {
    if (a[3] != b3) return a[3] < b3;
    if (a[2] != b2) return a[2] < b2;
    if (a[1] != b1) return a[1] < b1;
    return a[0] <= b0;
}

// ── Kernel parameters layout (PackedParams in Rust) ─────────────────────────

struct PomParams {
    uint  T;               // number of tensors
    uint  K;               // walk steps
    ulong n_total_chunks;  // total chunk count
    ulong p0, p1, p2, p3; // pre_pow_hash words
    ulong time_;           // timestamp
    ulong t0, t1, t2, t3; // target words (LE)
    ulong nonce_base;
    ulong n_nonces;
};

// ── Kernel ──────────────────────────────────────────────────────────────────

kernel void pom_mine(
    device const uchar*       all_data       [[buffer(0)]],
    device const ulong*       base_offsets   [[buffer(1)]],
    device const ulong*       prefix         [[buffer(2)]],
    constant PomParams&       params         [[buffer(3)]],
    device atomic_uint*       winner         [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n_nonces) return;
    ulong nonce = params.nonce_base + tid;

    ulong state = pom_seed_fold(nonce, params.time_, params.p0, params.p1, params.p2, params.p3);
    ulong off = state % params.n_total_chunks;

    for (uint i = 0; i < params.K; i++) {
        // Binary search for the owning tensor.
        uint lo = 0, hi = params.T;
        while (lo + 1 < hi) {
            uint mid = (lo + hi) >> 1;
            if (prefix[mid] <= off) lo = mid; else hi = mid;
        }
        ulong local = off - prefix[lo];
        device const uchar* tensor_start = all_data + base_offsets[lo];
        device const ulong* p = (device const ulong*)tensor_start;
        ulong base = local * 4ULL;
        ulong h = state;
        h ^= p[base]; h ^= p[base + 1]; h ^= p[base + 2]; h ^= p[base + 3];
        state = mix64(h);
        off = state % params.n_total_chunks;
    }

    ulong pv[4];
    pom_pow_fold(state, params.p0, params.p1, params.p2, params.p3, pv);
    if (pom_le_leq(pv, params.t0, params.t1, params.t2, params.t3)) {
        // MSL has no reliable 64-bit atomic min on A15-class GPUs, so we store the
        // batch-local thread index (tid, < n_nonces ≤ 2^20, fits in 32 bits) via a
        // universally-supported 32-bit atomic min. The host reconstructs the full
        // 64-bit nonce as nonce_base + winner. Determinism (min tid) still holds.
        atomic_fetch_min_explicit(winner, tid, memory_order_relaxed);
    }
}
