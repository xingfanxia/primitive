//! # primitive-gpu-cpu — the CPU adapter
//!
//! Implements the [`ShapeSearch`] port by delegating to `primitive_core`'s reference
//! `Searcher` (single-threaded). Because it *is* the reference search, it is the permanent
//! **parity oracle** every GPU adapter is checked against, and the **graceful-degradation
//! fallback** the GUI uses when no GPU is found (plan §5A — the amber `CPU (no GPU found)`
//! chip is this backend running normally, never an error).
//!
//! Import direction: depends on `primitive-core` + `primitive-compute` only.

#![forbid(unsafe_code)]

use primitive_compute::{Backend, BestShape, SearchParams, ShapeSearch};
use primitive_core::{Canvas, Ctx, Rng, Searcher};

/// CPU-backed shape search. Holds a reusable [`Searcher`] scratch buffer.
pub struct CpuSearch {
    searcher: Searcher,
}

impl CpuSearch {
    /// Build a CPU search for a `w × h` canvas.
    pub fn new(w: i32, h: i32) -> CpuSearch {
        CpuSearch {
            searcher: Searcher::new(w, h),
        }
    }
}

impl ShapeSearch for CpuSearch {
    fn backend(&self) -> Backend {
        Backend::Cpu
    }

    fn find_best_shape(
        &mut self,
        target: &Canvas,
        current: &Canvas,
        score: f64,
        params: &SearchParams,
        rng: &mut Rng,
    ) -> BestShape {
        self.searcher.counter = 0;
        let ctx = Ctx {
            target,
            current,
            score,
        };
        let state = self.searcher.best_hill_climb_state(&ctx, params, rng);
        BestShape {
            shape: state.shape,
            alpha: state.alpha,
            evaluations: self.searcher.counter,
        }
    }
}
