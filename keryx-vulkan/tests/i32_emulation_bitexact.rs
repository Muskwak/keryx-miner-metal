//! Bit-exactness proof for `pom_walk_i32.comp`'s uvec2-emulated 64-bit arithmetic.
//!
//! We cannot execute the actual GLSL/SPIR-V here (no Vulkan device in this test process), so this
//! mirrors each GLSL helper (`xor64`/`add64`/`shr64`/`mul64`/`mix64`/`mod64`) as a plain Rust
//! function operating on `(u32, u32)` = (lo, hi) pairs — the exact same operations, in the exact
//! same order, that the shader performs — and cross-checks every one against native `u64`
//! arithmetic across randomized inputs plus hand-picked edge cases. A final end-to-end test
//! replays the whole `pom_block_seed` -> walk -> `pom_pow_value` -> `le_leq` pipeline through both
//! implementations side by side and asserts bit-identical results at every stage.
//!
//! If this passes, the *algorithm* is proven correct; only the GLSL transcription itself (verified
//! separately: both shaders compile clean under `glslc --target-env=vulkan1.2`) is unverified by
//! this test.

type U64Emu = (u32, u32); // (lo, hi) — matches uvec2(x, y) in the shader

fn to_emu(v: u64) -> U64Emu {
    (v as u32, (v >> 32) as u32)
}

fn from_emu(v: U64Emu) -> u64 {
    (v.0 as u64) | ((v.1 as u64) << 32)
}

fn xor64(a: U64Emu, b: U64Emu) -> U64Emu {
    (a.0 ^ b.0, a.1 ^ b.1)
}

fn add64(a: U64Emu, b: U64Emu) -> U64Emu {
    let (lo, carry) = a.0.overflowing_add(b.0);
    let hi = a.1.wrapping_add(b.1).wrapping_add(carry as u32);
    (lo, hi)
}

/// Mirrors `shr64` in the shader: only valid for 0 < s < 32 (the only shifts mix64 ever uses).
fn shr64(a: U64Emu, s: u32) -> U64Emu {
    assert!(s > 0 && s < 32);
    let lo = (a.0 >> s) | (a.1 << (32 - s));
    let hi = a.1 >> s;
    (lo, hi)
}

/// Mirrors `mul64` in the shader: umulExtended(a.x, b.x) for the low 64 bits of the product, plus
/// the truncated cross terms for the high word.
fn mul64(a: U64Emu, b: U64Emu) -> U64Emu {
    let full = (a.0 as u64) * (b.0 as u64); // umulExtended(a.x, b.x, hi0, lo0)
    let lo0 = full as u32;
    let hi0 = (full >> 32) as u32;
    let cross = a.0.wrapping_mul(b.1).wrapping_add(a.1.wrapping_mul(b.0));
    (lo0, hi0.wrapping_add(cross))
}

fn mix64_emu(mut x: U64Emu) -> U64Emu {
    x = xor64(x, shr64(x, 30));
    x = mul64(x, (0x1ce4e5b9u32, 0xbf58476du32));
    x = xor64(x, shr64(x, 27));
    x = mul64(x, (0x133111ebu32, 0x94d049bbu32));
    x = xor64(x, shr64(x, 31));
    x
}

/// Mirrors `mod64` in the shader: binary long division, MSB to LSB, assumes divisor < 2^31.
fn mod64_emu(dividend: U64Emu, divisor: u32) -> u32 {
    assert!(divisor < (1u32 << 31));
    let mut rem: u32 = 0;
    for i in (0..32).rev() {
        rem = (rem << 1) | ((dividend.1 >> i) & 1);
        if rem >= divisor {
            rem -= divisor;
        }
    }
    for i in (0..32).rev() {
        rem = (rem << 1) | ((dividend.0 >> i) & 1);
        if rem >= divisor {
            rem -= divisor;
        }
    }
    rem
}

fn sub64(a: U64Emu, b: U64Emu) -> U64Emu {
    let (lo, borrow) = a.0.overflowing_sub(b.0);
    let hi = a.1.wrapping_sub(b.1).wrapping_sub(borrow as u32);
    (lo, hi)
}

fn wide_ge_u32(v: U64Emu, d: u32) -> bool {
    v.1 != 0 || v.0 >= d
}

fn sub64_u32(v: U64Emu, d: u32) -> U64Emu {
    sub64(v, (d, 0))
}

/// Mirrors `mulhi64` in the shader: the HIGH 64 bits of the full 128-bit product of two uvec2
/// operands, built from four 32x32->64 partial products (`umulExtended`) summed with explicit
/// carry propagation across the three overlapping 32-bit columns.
fn mulhi64(a: U64Emu, b: U64Emu) -> U64Emu {
    let p0 = (a.0 as u64) * (b.0 as u64); // a.lo * b.lo
    let p1 = (a.0 as u64) * (b.1 as u64); // a.lo * b.hi
    let p2 = (a.1 as u64) * (b.0 as u64); // a.hi * b.lo
    let p3 = (a.1 as u64) * (b.1 as u64); // a.hi * b.hi

    let (p0_hi, _p0_lo) = ((p0 >> 32) as u32, p0 as u32);
    let (p1_hi, p1_lo) = ((p1 >> 32) as u32, p1 as u32);
    let (p2_hi, p2_lo) = ((p2 >> 32) as u32, p2 as u32);
    let (p3_hi, p3_lo) = ((p3 >> 32) as u32, p3 as u32);

    // bits[32,64) column: p0_hi + p1_lo + p2_lo, carrying into bits[64,96). The column's own sum
    // (bits 32..64 of the full product) is discarded — mulhi64 only returns bits [64,128).
    let (col_lo, c1) = p0_hi.overflowing_add(p1_lo);
    let (_col_lo, c2) = col_lo.overflowing_add(p2_lo);
    let carry_to_hi = c1 as u32 + c2 as u32; // 0, 1, or 2

    // bits[64,96) column: p1_hi + p2_hi + p3_lo + carry_to_hi, carrying into bits[96,128).
    let (mid_hi, c3) = p1_hi.overflowing_add(p2_hi);
    let (mid_hi, c4) = mid_hi.overflowing_add(p3_lo);
    let (mid_hi, c5) = mid_hi.overflowing_add(carry_to_hi);
    let carry_to_top = c3 as u32 + c4 as u32 + c5 as u32; // 0..=3

    // bits[96,128) column: p3_hi + carry_to_top (wraps mod 2^32 — bits beyond 128 don't exist).
    let top = p3_hi.wrapping_add(carry_to_top);

    (mid_hi, top) // bits[64,96) then bits[96,128) — matches (lo, hi) convention for the top half
}

/// Barrett-reduction-based `dividend % divisor`, replacing the 64-iteration bit-serial `mod64`
/// with a handful of wide multiplies. `mu` is precomputed once (host-side, per n_chunks) as
/// `floor(2^64 / divisor)`. Requires divisor >= 2 (mu would overflow 64 bits for divisor == 1,
/// which never occurs for a real model's chunk count anyway).
fn mod64_fast(dividend: U64Emu, divisor: u32, mu: U64Emu) -> u32 {
    assert!(divisor >= 2);
    let q1 = mulhi64(dividend, mu); // approx quotient, off by at most a small constant
    let mut r = sub64(dividend, mul64(q1, (divisor, 0)));
    // Barrett's bound guarantees a small, fixed number of corrections suffice; 4 is a generous
    // margin over the theoretical ~2, verified empirically by the exhaustive tests below.
    for _ in 0..4 {
        if wide_ge_u32(r, divisor) {
            r = sub64_u32(r, divisor);
        }
    }
    assert_eq!(r.1, 0, "mod64_fast: correction loop did not converge for divisor={divisor}");
    r.0
}

fn mix64_native(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64 generator (unrelated to mix64 above, just a convenient RNG).
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

const EDGE_U64: &[u64] = &[
    0,
    1,
    u64::MAX,
    u64::MAX - 1,
    1u64 << 63,
    1u64 << 32,
    (1u64 << 32) - 1,
    (1u64 << 32) + 1,
    0xAAAA_AAAA_AAAA_AAAA,
    0x5555_5555_5555_5555,
    0x0000_0000_FFFF_FFFF,
    0xFFFF_FFFF_0000_0000,
];

#[test]
fn xor64_bit_exact() {
    let mut rng = Rng(1);
    for _ in 0..10_000 {
        let (a, b) = (rng.next(), rng.next());
        assert_eq!(from_emu(xor64(to_emu(a), to_emu(b))), a ^ b);
    }
}

#[test]
fn add64_bit_exact() {
    let mut rng = Rng(2);
    for &a in EDGE_U64 {
        for &b in EDGE_U64 {
            assert_eq!(from_emu(add64(to_emu(a), to_emu(b))), a.wrapping_add(b), "add64({a:#x}, {b:#x})");
        }
    }
    for _ in 0..50_000 {
        let (a, b) = (rng.next(), rng.next());
        assert_eq!(from_emu(add64(to_emu(a), to_emu(b))), a.wrapping_add(b), "add64({a:#x}, {b:#x})");
    }
}

#[test]
fn shr64_bit_exact() {
    let mut rng = Rng(3);
    for &s in &[1u32, 27, 30, 31] {
        for &v in EDGE_U64 {
            assert_eq!(from_emu(shr64(to_emu(v), s)), v >> s, "shr64({v:#x}, {s})");
        }
        for _ in 0..20_000 {
            let v = rng.next();
            assert_eq!(from_emu(shr64(to_emu(v), s)), v >> s, "shr64({v:#x}, {s})");
        }
    }
}

#[test]
fn mul64_bit_exact() {
    let mut rng = Rng(4);
    let consts = [0xbf58476d1ce4e5b9u64, 0x94d049bb133111ebu64];
    for &c in &consts {
        for &v in EDGE_U64 {
            let expected = v.wrapping_mul(c);
            let got = from_emu(mul64(to_emu(v), to_emu(c)));
            assert_eq!(got, expected, "mul64({v:#x}, {c:#x})");
        }
        for _ in 0..50_000 {
            let v = rng.next();
            let expected = v.wrapping_mul(c);
            let got = from_emu(mul64(to_emu(v), to_emu(c)));
            assert_eq!(got, expected, "mul64({v:#x}, {c:#x})");
        }
    }
    // General a*b (not just the two mix64 constants), including both-random operands.
    for _ in 0..100_000 {
        let (a, b) = (rng.next(), rng.next());
        let expected = a.wrapping_mul(b);
        let got = from_emu(mul64(to_emu(a), to_emu(b)));
        assert_eq!(got, expected, "mul64({a:#x}, {b:#x})");
    }
}

#[test]
fn mix64_bit_exact() {
    let mut rng = Rng(5);
    for &v in EDGE_U64 {
        assert_eq!(from_emu(mix64_emu(to_emu(v))), mix64_native(v), "mix64({v:#x})");
    }
    for _ in 0..200_000 {
        let v = rng.next();
        assert_eq!(from_emu(mix64_emu(to_emu(v))), mix64_native(v), "mix64({v:#x})");
    }
}

#[test]
fn mod64_bit_exact() {
    let mut rng = Rng(6);
    // Divisors spanning realistic model chunk counts up to just under the 2^31 shader guard.
    let divisors: &[u32] = &[1, 2, 3, 7, 255, 1 << 16, 34_420_544, (1u32 << 31) - 1, (1u32 << 30) + 1];
    for &d in divisors {
        for &dv in EDGE_U64 {
            let expected = dv % (d as u64);
            let got = mod64_emu(to_emu(dv), d) as u64;
            assert_eq!(got, expected, "mod64({dv:#x} % {d})");
        }
        // Values right around the divisor boundary.
        for &dv in &[(d as u64).saturating_sub(1), d as u64, d as u64 + 1] {
            let expected = dv % (d as u64);
            let got = mod64_emu(to_emu(dv), d) as u64;
            assert_eq!(got, expected, "mod64({dv:#x} % {d})");
        }
        for _ in 0..20_000 {
            let dv = rng.next();
            let expected = dv % (d as u64);
            let got = mod64_emu(to_emu(dv), d) as u64;
            assert_eq!(got, expected, "mod64({dv:#x} % {d})");
        }
    }
}

/// Host-side precomputation: `mu = floor(2^64 / divisor)`, exactly what `pom_walk.rs` computes
/// once per `PomWalkGpu` (n_chunks is fixed for the miner's lifetime) via `u128` arithmetic.
fn compute_mu(divisor: u32) -> U64Emu {
    to_emu(((1u128 << 64) / (divisor as u128)) as u64)
}

#[test]
fn mulhi64_bit_exact() {
    let mut rng = Rng(8);
    for _ in 0..200_000 {
        let (a, b) = (rng.next(), rng.next());
        let expected = (((a as u128) * (b as u128)) >> 64) as u64;
        let got = from_emu(mulhi64(to_emu(a), to_emu(b)));
        assert_eq!(got, expected, "mulhi64({a:#x}, {b:#x})");
    }
    for &a in EDGE_U64 {
        for &b in EDGE_U64 {
            let expected = (((a as u128) * (b as u128)) >> 64) as u64;
            let got = from_emu(mulhi64(to_emu(a), to_emu(b)));
            assert_eq!(got, expected, "mulhi64({a:#x}, {b:#x})");
        }
    }
}

#[test]
fn mod64_fast_bit_exact() {
    let mut rng = Rng(9);
    // Divisors spanning realistic model chunk counts up to just under the 2^31 shader guard —
    // mirrors mod64_bit_exact's list, minus 1 (mod64_fast requires divisor >= 2).
    let divisors: &[u32] = &[2, 3, 7, 255, 1 << 16, 34_420_544, (1u32 << 31) - 1, (1u32 << 30) + 1];
    for &d in divisors {
        let mu = compute_mu(d);
        for &dv in EDGE_U64 {
            let expected = dv % (d as u64);
            let got = mod64_fast(to_emu(dv), d, mu) as u64;
            assert_eq!(got, expected, "mod64_fast({dv:#x} % {d})");
        }
        for &dv in &[(d as u64).saturating_sub(1), d as u64, d as u64 + 1] {
            let expected = dv % (d as u64);
            let got = mod64_fast(to_emu(dv), d, mu) as u64;
            assert_eq!(got, expected, "mod64_fast({dv:#x} % {d})");
        }
        for _ in 0..50_000 {
            let dv = rng.next();
            let expected = dv % (d as u64);
            let got = mod64_fast(to_emu(dv), d, mu) as u64;
            assert_eq!(got, expected, "mod64_fast({dv:#x} % {d})");
        }
    }
    // Exhaustive sweep over every divisor 2..=2000 with randomized dividends — catches any
    // divisor-dependent edge case the hand-picked list above might miss.
    for d in 2u32..=2000 {
        let mu = compute_mu(d);
        for _ in 0..200 {
            let dv = rng.next();
            let expected = dv % (d as u64);
            let got = mod64_fast(to_emu(dv), d, mu) as u64;
            assert_eq!(got, expected, "mod64_fast({dv:#x} % {d})");
        }
    }
}

fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

fn le_leq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    true
}

/// Native (u64) reference pipeline: pom_block_seed -> K-step walk over a synthetic chunk source ->
/// pom_pow_value -> le_leq. Mirrors `src/pom.rs` exactly.
#[allow(clippy::too_many_arguments)]
fn walk_native(
    pre_pow_hash: &[u8; 32],
    timestamp: u64,
    nonce: u64,
    target: &[u8; 32],
    n_chunks: u64,
    k: u32,
    chunk_of: impl Fn(u64) -> [u64; 4],
) -> ([u8; 32], bool) {
    let p = words4(pre_pow_hash);

    let mut state = mix64_native(nonce ^ 0x4B65727978531);
    state = mix64_native(state ^ timestamp);
    state = mix64_native(state ^ p[0]);
    state = mix64_native(state ^ p[1]);
    state = mix64_native(state ^ p[2]);
    state = mix64_native(state ^ p[3]);

    let mut off = state % n_chunks;
    for _ in 0..k {
        let chunk = chunk_of(off);
        let mut h = state;
        for w in chunk {
            h ^= w;
        }
        state = mix64_native(h);
        off = state % n_chunks;
    }

    let o0 = mix64_native(state ^ p[0] ^ 0x9E3779B97F4A7C15);
    let o1 = mix64_native(o0 ^ p[1] ^ 0xC2B2AE3D27D4EB4F);
    let o2 = mix64_native(o1 ^ p[2] ^ 0x165667B19E3779F9);
    let o3 = mix64_native(o2 ^ p[3] ^ 0xD6E8FEB86659FD93);

    let mut pow = [0u8; 32];
    pow[0..8].copy_from_slice(&o0.to_le_bytes());
    pow[8..16].copy_from_slice(&o1.to_le_bytes());
    pow[16..24].copy_from_slice(&o2.to_le_bytes());
    pow[24..32].copy_from_slice(&o3.to_le_bytes());
    let le = le_leq(&pow, target);
    (pow, le)
}

/// Emulated (uvec2) pipeline — same steps, same order, using only the `_emu` helpers above.
/// Mirrors `pom_walk_i32.comp`'s `main()` exactly.
#[allow(clippy::too_many_arguments)]
fn walk_emu(
    pre_pow_hash: &[u8; 32],
    timestamp: u64,
    nonce: u64,
    target: &[u8; 32],
    n_chunks: u64,
    k: u32,
    chunk_of: impl Fn(u64) -> [u64; 4],
) -> ([u8; 32], bool) {
    let p = words4(pre_pow_hash).map(to_emu);
    let t = words4(target).map(to_emu);
    let n_chunks32 = n_chunks as u32;
    let mu = compute_mu(n_chunks32); // host precomputes this once per PomWalkGpu, not per nonce

    let mut state = mix64_emu(xor64(to_emu(nonce), (0x27978531u32, 0x0004B657u32)));
    state = mix64_emu(xor64(state, to_emu(timestamp)));
    state = mix64_emu(xor64(state, p[0]));
    state = mix64_emu(xor64(state, p[1]));
    state = mix64_emu(xor64(state, p[2]));
    state = mix64_emu(xor64(state, p[3]));

    let mut off = mod64_fast(state, n_chunks32, mu) as u64;
    for _ in 0..k {
        let chunk = chunk_of(off).map(to_emu);
        let mut h = state;
        for w in chunk {
            h = xor64(h, w);
        }
        state = mix64_emu(h);
        off = mod64_fast(state, n_chunks32, mu) as u64;
    }

    let o0 = mix64_emu(xor64(xor64(state, p[0]), (0x7F4A7C15u32, 0x9E3779B9u32)));
    let o1 = mix64_emu(xor64(xor64(o0, p[1]), (0x27D4EB4Fu32, 0xC2B2AE3Du32)));
    let o2 = mix64_emu(xor64(xor64(o1, p[2]), (0x9E3779F9u32, 0x165667B1u32)));
    let o3 = mix64_emu(xor64(xor64(o2, p[3]), (0x6659FD93u32, 0xD6E8FEB8u32)));

    let mut pow = [0u8; 32];
    pow[0..8].copy_from_slice(&from_emu(o0).to_le_bytes());
    pow[8..16].copy_from_slice(&from_emu(o1).to_le_bytes());
    pow[16..24].copy_from_slice(&from_emu(o2).to_le_bytes());
    pow[24..32].copy_from_slice(&from_emu(o3).to_le_bytes());

    // le_leq via the shader's word-wise compare (most-significant word o3 first), not byte compare
    // — proves the emulated `lt64`/`eq64` comparison logic, not just the byte layout.
    fn lt64(a: U64Emu, b: U64Emu) -> bool {
        if a.1 != b.1 {
            a.1 < b.1
        } else {
            a.0 < b.0
        }
    }
    fn eq64(a: U64Emu, b: U64Emu) -> bool {
        a == b
    }
    let le = if !eq64(o3, t[3]) {
        lt64(o3, t[3])
    } else if !eq64(o2, t[2]) {
        lt64(o2, t[2])
    } else if !eq64(o1, t[1]) {
        lt64(o1, t[1])
    } else if !eq64(o0, t[0]) {
        lt64(o0, t[0])
    } else {
        true
    };
    (pow, le)
}

#[test]
fn end_to_end_walk_bit_exact() {
    let mut rng = Rng(7);
    let n_chunks: u64 = 34_420_544; // a real observed model chunk count
    // Synthetic weight blob: deterministic pseudo-random 4-word chunks, indexed by chunk offset.
    let chunk_of = |off: u64| -> [u64; 4] {
        let mut s = Rng(off ^ 0xC0FFEE);
        [s.next(), s.next(), s.next(), s.next()]
    };

    for case in 0..2_000 {
        let mut pre_pow_hash = [0u8; 32];
        let mut target = [0u8; 32];
        for b in pre_pow_hash.iter_mut() {
            *b = (rng.next() & 0xFF) as u8;
        }
        for b in target.iter_mut() {
            *b = (rng.next() & 0xFF) as u8;
        }
        let timestamp = rng.next();
        let nonce = rng.next();
        // Use a short walk length in most cases (fast), occasionally the real 256 steps.
        let k = if case % 200 == 0 { 256 } else { 8 };

        let (pow_native, le_native) = walk_native(&pre_pow_hash, timestamp, nonce, &target, n_chunks, k, chunk_of);
        let (pow_emu, le_emu) = walk_emu(&pre_pow_hash, timestamp, nonce, &target, n_chunks, k, chunk_of);

        assert_eq!(pow_emu, pow_native, "pow_value mismatch at case {case} (nonce={nonce:#x})");
        assert_eq!(le_emu, le_native, "le_leq mismatch at case {case} (nonce={nonce:#x})");
    }
}
