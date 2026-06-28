//! Self-adaptive (1+1)-ES with Rechenberg's 1/5th success rule.
//!
//! Plan §4's "more elegant algorithm": hill-climbing is a degenerate (1+1)-ES with a fixed
//! step and no adaptation. This generic continuous optimizer adds the 1/5 rule — grow the
//! mutation step when more than 1/5 of mutations improve, shrink it otherwise — which gives
//! annealing-like behavior *for free*, deleting fogleman's hand-tuned schedule.
//!
//! It is parameterized over an objective `f: &[f64] -> f64` and an injected [`Rng`], so it is
//! pure and deterministically testable. At GPU-3 this same rule runs on-device over a
//! shape's geometry vector; here it is validated standalone (see tests) before that wiring.

use crate::rng::Rng;

/// Outcome of a (1+1)-ES run.
#[derive(Clone, Debug)]
pub struct EsResult {
    /// Best parameter vector found.
    pub x: Vec<f64>,
    /// Objective value at `x` (minimized).
    pub energy: f64,
    /// Final mutation step size (shrinks as the search converges).
    pub sigma: f64,
}

/// Configuration for [`one_plus_one_es`].
#[derive(Clone, Copy, Debug)]
pub struct EsConfig {
    /// Initial mutation step size.
    pub sigma0: f64,
    /// Total mutation steps.
    pub max_iters: usize,
    /// Step-adaptation multiplier `c ∈ (0, 1)`. Rechenberg's classic value ≈ 0.817.
    pub adapt: f64,
}

impl Default for EsConfig {
    fn default() -> Self {
        EsConfig {
            sigma0: 1.0,
            max_iters: 2000,
            adapt: 0.817,
        }
    }
}

/// Minimize `f` from `x0` with a self-adaptive (1+1)-ES.
///
/// The 1/5 rule is applied over windows of `dim` steps: a success rate above 1/5 grows
/// `sigma` (divide by `c`), below 1/5 shrinks it (multiply by `c`).
pub fn one_plus_one_es<F>(x0: Vec<f64>, mut f: F, cfg: EsConfig, rng: &mut Rng) -> EsResult
where
    F: FnMut(&[f64]) -> f64,
{
    let dim = x0.len().max(1);
    let mut x = x0;
    let mut energy = f(&x);
    let mut sigma = cfg.sigma0;

    let window = dim;
    let mut successes = 0usize;
    let mut trial = vec![0.0; x.len()];

    for i in 1..=cfg.max_iters {
        for k in 0..x.len() {
            trial[k] = x[k] + sigma * rng.norm();
        }
        let e = f(&trial);
        if e < energy {
            x.copy_from_slice(&trial);
            energy = e;
            successes += 1;
        }

        // 1/5 success rule, evaluated once per window.
        if i % window == 0 {
            let rate = successes as f64 / window as f64;
            if rate > 0.2 {
                sigma /= cfg.adapt; // improving often ⇒ take bigger steps
            } else if rate < 0.2 {
                sigma *= cfg.adapt; // rarely improving ⇒ refine
            }
            successes = 0;
        }
    }

    EsResult { x, energy, sigma }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    #[test]
    fn minimizes_sphere() {
        let mut rng = Rng::new(1);
        let res = one_plus_one_es(
            vec![5.0, -4.0, 3.0],
            sphere,
            EsConfig {
                sigma0: 1.0,
                max_iters: 4000,
                adapt: 0.817,
            },
            &mut rng,
        );
        assert!(res.energy < 1e-6, "energy {} not near 0", res.energy);
    }

    #[test]
    fn improves_on_initial() {
        let mut rng = Rng::new(2);
        let x0 = vec![10.0, 10.0];
        let e0 = sphere(&x0);
        let res = one_plus_one_es(x0, sphere, EsConfig::default(), &mut rng);
        assert!(res.energy < e0);
    }

    #[test]
    fn sigma_shrinks_on_convergence() {
        // As the search homes in on the optimum, the 1/5 rule must reduce the step size.
        let mut rng = Rng::new(3);
        let res = one_plus_one_es(
            vec![5.0, 5.0],
            sphere,
            EsConfig {
                sigma0: 1.0,
                max_iters: 4000,
                adapt: 0.817,
            },
            &mut rng,
        );
        assert!(res.sigma < 0.05, "sigma {} did not shrink", res.sigma);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let run = || {
            let mut rng = Rng::new(99);
            one_plus_one_es(vec![3.0, 3.0, 3.0], sphere, EsConfig::default(), &mut rng).energy
        };
        assert_eq!(run().to_bits(), run().to_bits());
    }
}
