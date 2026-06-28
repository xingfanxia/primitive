//! The reference run loop — canvas state + per-shape commit, single-threaded.
//!
//! Port of fogleman's `model.go` (`NewModel` / `Add` / `Step`), minus the GUI/IO and the
//! goroutine fan-out (one deterministic `Searcher`). Operates at the target's native
//! resolution. This `Model` is the golden/quality reference; the engine crate (CORE-2)
//! reproduces its output through the `GpuCompute` port.

use crate::canvas::Canvas;
use crate::color::{compute_color, Color};
use crate::draw::draw_lines;
use crate::optimizer::{Ctx, SearchBudget, Searcher};
use crate::rng::Rng;
use crate::score::{difference_full, difference_partial};
use crate::shape::{Shape, ShapeType};

/// A committed shape: geometry, the solved fill color, and the model score after it landed.
#[derive(Clone, Debug)]
pub struct AddedShape {
    pub shape: Shape,
    pub color: Color,
    pub score: f64,
}

/// The reconstruction in progress.
pub struct Model {
    pub w: i32,
    pub h: i32,
    pub background: Color,
    pub target: Canvas,
    pub current: Canvas,
    /// Running normalized-RMSE score in `[0, 1]` (lower is better).
    pub score: f64,
    pub shapes: Vec<AddedShape>,
    searcher: Searcher,
}

impl Model {
    /// Build a model for `target` on a solid `background`.
    pub fn new(target: Canvas, background: Color) -> Model {
        let w = target.w as i32;
        let h = target.h as i32;
        let current = Canvas::filled(
            target.w,
            target.h,
            background.r as u8,
            background.g as u8,
            background.b as u8,
        );
        let score = difference_full(&target, &current);
        let searcher = Searcher::new(w, h);
        Model {
            w,
            h,
            background,
            target,
            current,
            score,
            shapes: Vec::new(),
            searcher,
        }
    }

    /// Background = the target's average color (fogleman's default when `-bg` is unset).
    pub fn with_average_background(target: Canvas) -> Model {
        let (r, g, b) = target.average_color();
        Model::new(
            target,
            Color {
                r: r as i32,
                g: g as i32,
                b: b as i32,
                a: 255,
            },
        )
    }

    /// Commit a shape: solve its color, composite it, and update the running score — `Add`.
    pub fn add(&mut self, shape: Shape, alpha: i32) {
        let before = self.current.clone();
        let lines = shape.rasterize(self.w, self.h);
        let color = compute_color(&self.target, &self.current, &lines, alpha);
        draw_lines(&mut self.current, color, &lines);
        let score = difference_partial(&self.target, &before, &self.current, self.score, &lines);
        self.score = score;
        self.shapes.push(AddedShape {
            shape,
            color,
            score,
        });
    }

    /// Search for the best next shape and commit it; returns candidate evaluations this step.
    ///
    /// fogleman's `repeat` (reduced-search extra shapes) is not yet ported — v1 commits one
    /// shape per step (the default `-rep 0`). The golden/quality gates run with `repeat = 0`.
    pub fn step(&mut self, budget: &SearchBudget, rng: &mut Rng) -> u64 {
        self.searcher.counter = 0;
        let ctx = Ctx {
            target: &self.target,
            current: &self.current,
            score: self.score,
        };
        let state = self.searcher.best_hill_climb_state(&ctx, budget, rng);
        self.add(state.shape, state.alpha);
        self.searcher.counter
    }

    /// Convenience: run `count` shapes of type `t` at `alpha` with the default `(n, age, m)`.
    pub fn run(&mut self, t: ShapeType, alpha: i32, count: i32, rng: &mut Rng) -> u64 {
        let budget = SearchBudget {
            shape_type: t,
            alpha,
            ..SearchBudget::triangles_default()
        };
        let mut total = 0;
        for _ in 0..count {
            total += self.step(&budget, rng);
        }
        total
    }

    /// Peak signal-to-noise ratio (dB) of the current reconstruction vs the target.
    pub fn psnr(&self) -> f64 {
        psnr_from_score(self.score)
    }

    /// Native-resolution SVG of the committed shapes over the background.
    pub fn svg(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"{}\" height=\"{}\">\n",
            self.w, self.h
        ));
        out.push_str(&format!(
            "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#{:02x}{:02x}{:02x}\" />\n",
            self.w, self.h, self.background.r, self.background.g, self.background.b
        ));
        for a in &self.shapes {
            let attrs = format!(
                "fill=\"#{:02x}{:02x}{:02x}\" fill-opacity=\"{:.3}\"",
                a.color.r,
                a.color.g,
                a.color.b,
                a.color.a as f64 / 255.0
            );
            out.push_str(&a.shape.svg(&attrs));
            out.push('\n');
        }
        out.push_str("</svg>");
        out
    }
}

/// PSNR (dB) from a normalized-RMSE score in `(0, 1]`. `score == 0` ⇒ +∞.
pub fn psnr_from_score(score: f64) -> f64 {
    if score <= 0.0 {
        f64::INFINITY
    } else {
        -20.0 * score.log10()
    }
}
