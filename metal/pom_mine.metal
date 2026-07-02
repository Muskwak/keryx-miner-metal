// Keryx Proof-of-Model mining kernel (Metal, Apple Silicon port).
//
// Port of cuda/pom_mine.cu. Per nonce: seed-fold + data-dependent gather walk over the
// resident (packed) weight blob + pow-fold + target check. The seed/pow folds are
// byte-identical to `pom_mine.cu::pom_seed_fold`/`pom_pow_fold` and the host
// `pom::pom_block_seed`/`pom::pom_pow_value`, so nonces mined here build proofs the node
// accepts.
//
// Two shape differences vs the CUDA kernel:
//
//   1. Weights are laid out as a single packed device buffer, in the canonical name-sorted
//      GGUF tensor order — the same order the CUDA gather is built over — so chunk `off`
//      lives at bytes [off*32..off*32+32]. No per-tensor (bases,prefix) binary search.
//   2. `winner` is an `atomic_uint` holding the winning tid (0..n_nonces); Metal does not
//      guarantee `atomic_ulong`. Host reconstructs the nonce as `nonce_base + tid`. Since
//      POM_BATCH is 1<<20 and always < 2^32, this is byte-identical to CUDA's
//      `atomicMin(winner, nonce)`.

#include <metal_stdlib>
using namespace metal;

struct PomUniforms {
    ulong  n_total_chunks;
    uint   k_steps;
    uint   _pad;
    ulong  p0; ulong p1; ulong p2; ulong p3;
    ulong  time_;
    ulong  t0; ulong t1; ulong t2; ulong t3;
    ulong  nonce_base;
    uint   n_nonces;
    uint   _pad2;
};

inline ulong mix64(ulong x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9UL;
    x ^= x >> 27; x *= 0x94d049bb133111ebUL;
    x ^= x >> 31;
    return x;
}

inline ulong pom_seed_fold(ulong nonce, ulong time_,
                           ulong p0, ulong p1, ulong p2, ulong p3) {
    ulong s = mix64(nonce ^ 0x4B65727978531UL);
    s = mix64(s ^ time_);
    s = mix64(s ^ p0); s = mix64(s ^ p1); s = mix64(s ^ p2); s = mix64(s ^ p3);
    return s;
}

inline void pom_pow_fold(ulong fin, ulong p0, ulong p1, ulong p2, ulong p3,
                         thread ulong* out) {
    out[0] = mix64(fin    ^ p0 ^ 0x9E3779B97F4A7C15UL);
    out[1] = mix64(out[0] ^ p1 ^ 0xC2B2AE3D27D4EB4FUL);
    out[2] = mix64(out[1] ^ p2 ^ 0x165667B19E3779F9UL);
    out[3] = mix64(out[2] ^ p3 ^ 0xD6E8FEB86659FD93UL);
}

inline bool pom_le_leq(thread const ulong* a,
                       ulong b0, ulong b1, ulong b2, ulong b3) {
    if (a[3] != b3) return a[3] < b3;
    if (a[2] != b2) return a[2] < b2;
    if (a[1] != b1) return a[1] < b1;
    return a[0] <= b0;
}

kernel void pom_mine(
    device   const ulong*   weights [[buffer(0)]],
    constant const PomUniforms& u   [[buffer(1)]],
    device   atomic_uint*   winner  [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= u.n_nonces) return;
    ulong nonce = u.nonce_base + (ulong)tid;

    ulong state = pom_seed_fold(nonce, u.time_, u.p0, u.p1, u.p2, u.p3);
    ulong off = state % u.n_total_chunks;
    for (uint i = 0; i < u.k_steps; i++) {
        ulong base = off * 4UL;
        ulong h = state;
        h ^= weights[base + 0];
        h ^= weights[base + 1];
        h ^= weights[base + 2];
        h ^= weights[base + 3];
        state = mix64(h);
        off = state % u.n_total_chunks;
    }
    ulong pv[4];
    pom_pow_fold(state, u.p0, u.p1, u.p2, u.p3, pv);
    if (pom_le_leq(pv, u.t0, u.t1, u.t2, u.t3)) {
        atomic_fetch_min_explicit(winner, tid, memory_order_relaxed);
    }
}
