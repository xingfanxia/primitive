//! Counter-based RNG (Philox-2×32 style) — the determinism substrate for the GPU search loop.
//!
//! fogleman seeds a stateful PRNG from the wall clock → non-reproducible (plan §2/§6.6). A
//! counter-based RNG instead derives each draw from `(seed, counter)` with no carried state, so a
//! GPU thread can compute its own independent, **bit-reproducible** stream from its indices alone —
//! and the CPU reproduces it exactly for parity. Pure `u32` (no i64/f64), so it ports to WGSL/Metal
//! unchanged (`primitive-gpu-cubecl::kernels` mirrors this).
//!
//! This is its own permutation (CPU and GPU agree with each other); it is not bit-compatible with
//! Random123, and doesn't need to be.

const PHILOX_M: u32 = 0xD256_D193;
const PHILOX_W: u32 = 0x9E37_79B9;
const ROUNDS: u32 = 10;

/// 32×32→64-bit multiply, split into `(hi, lo)` — pure `u32`, WGSL-portable (no `u64` on device).
#[inline]
pub fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let a_lo = a & 0xffff;
    let a_hi = a >> 16;
    let b_lo = b & 0xffff;
    let b_hi = b >> 16;

    let ll = a_lo.wrapping_mul(b_lo);
    let lh = a_lo.wrapping_mul(b_hi);
    let hl = a_hi.wrapping_mul(b_lo);
    let hh = a_hi.wrapping_mul(b_hi);

    let mid = (ll >> 16)
        .wrapping_add(lh & 0xffff)
        .wrapping_add(hl & 0xffff);
    let lo = (ll & 0xffff) | (mid << 16);
    let hi = hh
        .wrapping_add(lh >> 16)
        .wrapping_add(hl >> 16)
        .wrapping_add(mid >> 16);
    (hi, lo)
}

/// One Philox-2×32 block: hash counter `(c0, c1)` under `key` to two `u32` words.
#[inline]
pub fn philox(c0: u32, c1: u32, key: u32) -> (u32, u32) {
    let mut x0 = c0;
    let mut x1 = c1;
    let mut k = key;
    for _ in 0..ROUNDS {
        let (hi, lo) = mulhilo(PHILOX_M, x0);
        x0 = hi ^ k ^ x1;
        x1 = lo;
        k = k.wrapping_add(PHILOX_W);
    }
    (x0, x1)
}

/// First word of `philox` — a single `u32` draw for counter `ctr` under `seed`.
#[inline]
pub fn rand_u32(seed: u32, ctr: u32) -> u32 {
    philox(ctr, 0, seed).0
}

/// Uniform integer in `[0, n)` via the standard high-bits multiply-shift (no modulo bias for the
/// ranges we use). `n` must be > 0.
#[inline]
pub fn rand_below(seed: u32, ctr: u32, n: u32) -> u32 {
    mulhilo(rand_u32(seed, ctr), n).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_is_deterministic() {
        assert_eq!(philox(1, 2, 3), philox(1, 2, 3));
        assert_eq!(rand_u32(42, 7), rand_u32(42, 7));
    }

    #[test]
    fn distinct_counters_decorrelate() {
        // Consecutive counters must not produce the same word (a stuck/identity permutation would).
        let a = rand_u32(99, 0);
        let b = rand_u32(99, 1);
        let c = rand_u32(99, 2);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn rand_below_in_range_and_spreads() {
        let mut counts = [0u32; 8];
        for ctr in 0..4000 {
            let v = rand_below(7, ctr, 8);
            assert!(v < 8);
            counts[v as usize] += 1;
        }
        // Every bucket hit — a roughly uniform spread over 4000 draws into 8 bins.
        assert!(counts.iter().all(|&c| c > 250), "uneven: {counts:?}");
    }

    #[test]
    fn mulhilo_matches_u64_reference() {
        for &(a, b) in &[
            (0xDEAD_BEEFu32, 0x1234_5678u32),
            (0xFFFF_FFFF, 0xFFFF_FFFF),
            (3, 5),
        ] {
            let (hi, lo) = mulhilo(a, b);
            let full = (a as u64) * (b as u64);
            assert_eq!(hi, (full >> 32) as u32);
            assert_eq!(lo, (full & 0xffff_ffff) as u32);
        }
    }
}
