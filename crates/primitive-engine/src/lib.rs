//! # primitive-engine — the application layer
//!
//! Orchestrates the run loop through the [`ShapeSearch`] port: per shape, ask the port for
//! the best next shape, then commit it to the canvas state (a `primitive_core::Model`). It
//! is generic over the backend and never imports a concrete adapter — the composition root
//! (GUI/CLI/tests) injects one.
//!
//! Import direction (plan §5): depends on `primitive-core` + `primitive-compute` only.
//! With the CPU adapter injected, the loop is byte-identical to `Model::run`, which is what
//! makes CORE-2 reproduce the CORE-1 golden exactly.

#![forbid(unsafe_code)]

use primitive_compute::{Backend, SearchParams, ShapeSearch};
use primitive_core::{Canvas, Color, Model, Rng};

/// Progress emitted after each committed shape (for the GUI ticker / CLI log).
#[derive(Clone, Copy, Debug)]
pub struct StepProgress {
    /// 1-based index of the shape just committed.
    pub shape_index: usize,
    /// Running normalized-RMSE score (lower is better).
    pub score: f64,
    /// Candidate evaluations performed this step.
    pub evaluations: u64,
}

/// Drives the reconstruction over a pluggable search backend.
pub struct Engine<S: ShapeSearch> {
    pub model: Model,
    search: S,
}

impl<S: ShapeSearch> Engine<S> {
    /// Build an engine over `target` with the given `background` and search backend.
    pub fn new(target: Canvas, background: Color, search: S) -> Engine<S> {
        Engine {
            model: Model::new(target, background),
            search,
        }
    }

    /// Build an engine using the target's average color as background (fogleman default).
    pub fn with_average_background(target: Canvas, search: S) -> Engine<S> {
        Engine {
            model: Model::with_average_background(target),
            search,
        }
    }

    /// The active backend (for the device chip).
    pub fn backend(&self) -> Backend {
        self.search.backend()
    }

    /// Search for and commit one shape; returns this step's progress.
    pub fn step(&mut self, params: &SearchParams, rng: &mut Rng) -> StepProgress {
        let best = self.search.find_best_shape(
            &self.model.target,
            &self.model.current,
            self.model.score,
            params,
            rng,
        );
        self.model.add(best.shape, best.alpha);
        StepProgress {
            shape_index: self.model.shapes.len(),
            score: self.model.score,
            evaluations: best.evaluations,
        }
    }

    /// Run `count` shapes, invoking `on_step` after each (for live progress).
    pub fn run<F: FnMut(StepProgress)>(
        &mut self,
        params: &SearchParams,
        count: i32,
        rng: &mut Rng,
        mut on_step: F,
    ) {
        for _ in 0..count {
            let p = self.step(params, rng);
            on_step(p);
        }
    }
}
