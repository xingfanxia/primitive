//! Deterministic, injectable RNG for the optimizer.
//!
//! Per the engineering rule "inject the RNG", every stochastic core routine takes
//! `&mut Rng` rather than reaching for a global. A fixed `seed` ⇒ a fully reproducible
//! run, which is what makes the golden-image test (`tests/golden.rs`) a determinism oracle.
//!
//! This is *our* RNG (SplitMix64 core + Marsaglia-polar normal), not Go's `math/rand`. We
//! deliberately do not replicate Go's generator: fogleman seeds with wall-clock time and
//! runs across goroutines, so it is not reproducible run-to-run and cannot be matched
//! bit-for-bit anyway. Parity with fogleman is proven on the RNG-independent *math*
//! (see `tests/parity.rs`), not on the random stream.

/// A small, fast, fully deterministic PRNG (SplitMix64).
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    /// Cached second normal deviate from the Marsaglia polar method (None when empty).
    spare_normal: Option<f64>,
}

impl Rng {
    /// Seed the generator. The same seed always yields the same stream.
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero fixed point of SplitMix64.
        Rng {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
            spare_normal: None,
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        // SplitMix64 (Steele, Lea & Flood 2014).
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0, 1)` with 53 bits of entropy.
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `[0, n)`. Panics if `n == 0` (mirrors Go's `rand.Intn`).
    #[inline]
    pub fn intn(&mut self, n: i32) -> i32 {
        assert!(n > 0, "Rng::intn requires n > 0");
        // Modulo bias is ~n/2^64 (negligible for the small n here) and we deliberately do
        // not match Go's stream — do NOT "fix" this to rejection sampling, it would perturb
        // the committed golden for no real-world gain.
        (self.next_u64() % n as u64) as i32
    }

    /// Standard normal deviate (mean 0, variance 1) via the Marsaglia polar method.
    #[inline]
    pub fn norm(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        // Draw a point in the unit disc (Marsaglia polar form — no trig, deterministic).
        loop {
            let u = 2.0 * self.f64() - 1.0;
            let v = 2.0 * self.f64() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let mul = (-2.0 * s.ln() / s).sqrt();
                self.spare_normal = Some(v * mul);
                return u * mul;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_fixed_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn intn_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let x = r.intn(31);
            assert!((0..31).contains(&x));
        }
    }

    #[test]
    fn f64_in_unit_interval() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let x = r.f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn normal_is_reasonable() {
        let mut r = Rng::new(123);
        let n = 50_000;
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for _ in 0..n {
            let z = r.norm();
            sum += z;
            sumsq += z * z;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "mean {mean} not ~0");
        assert!((var - 1.0).abs() < 0.05, "var {var} not ~1");
    }
}
