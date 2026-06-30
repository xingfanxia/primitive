//! Per-shape search — the fogleman reference optimizer (random-restart + hill-climb).
//!
//! Faithful port of `worker.go` + `optimize.go` + `state.go`, single-threaded so a fixed
//! seed is fully reproducible. This is the *reference* search the golden + quality gates
//! are measured against; the plan's (1+1)-ES improvement lives in [`crate::es`] and is held
//! to "≥ this quality". The CPU adapter (CORE-2) and every GPU milestone re-prove parity
//! against the output of this module.

use crate::canvas::Canvas;
use crate::color::compute_color;
use crate::draw::{copy_lines, draw_lines};
use crate::rng::Rng;
use crate::score::difference_partial;
use crate::shape::{Shape, ShapeType};

#[inline]
fn clamp_i32(x: i32, lo: i32, hi: i32) -> i32 {
    x.max(lo).min(hi)
}

/// Per-shape search budget — fogleman's `(n, age, m)` plus shape type and fill alpha.
#[derive(Clone, Copy, Debug)]
pub struct SearchBudget {
    pub shape_type: ShapeType,
    /// Fill alpha; `0` selects fogleman's "mutate the alpha too" mode.
    pub alpha: i32,
    /// Random restarts per attempt.
    pub n: i32,
    /// Hill-climb age (non-improving steps tolerated).
    pub age: i32,
    /// Independent attempts; the best wins.
    pub m: i32,
}

impl SearchBudget {
    /// fogleman's defaults: triangle, alpha 128, `n=1000, age=100, m=16`.
    pub fn triangles_default() -> SearchBudget {
        SearchBudget {
            shape_type: ShapeType::Triangle,
            alpha: 128,
            n: 1000,
            age: 100,
            m: 16,
        }
    }
}

/// Immutable scoring context for one search: what the candidate is fitted against.
#[derive(Clone, Copy)]
pub struct Ctx<'a> {
    pub target: &'a Canvas,
    pub current: &'a Canvas,
    /// The running model score of `current`.
    pub score: f64,
}

/// A candidate being optimized: geometry + alpha + a lazily-cached score (`< 0` = stale).
#[derive(Clone, Debug)]
pub struct State {
    pub shape: Shape,
    pub alpha: i32,
    pub mutate_alpha: bool,
    pub score: f64,
}

impl State {
    /// `alpha == 0` selects fogleman's "mutate the alpha too" mode, starting at 128.
    pub fn new(shape: Shape, alpha: i32) -> State {
        let (alpha, mutate_alpha) = if alpha == 0 {
            (128, true)
        } else {
            (alpha, false)
        };
        State {
            shape,
            alpha,
            mutate_alpha,
            score: -1.0,
        }
    }
}

/// Reusable per-search scratch: a compositing buffer + an evaluation counter (for n/s).
pub struct Searcher {
    pub w: i32,
    pub h: i32,
    buffer: Canvas,
    pub counter: u64,
}

impl Searcher {
    pub fn new(w: i32, h: i32) -> Searcher {
        Searcher {
            w,
            h,
            buffer: Canvas::new(w as usize, h as usize),
            counter: 0,
        }
    }

    /// Energy of placing `shape@alpha` over `ctx.current` — `Worker.Energy`:
    /// rasterize → closed-form color → composite into buffer → bbox delta-SSE.
    pub fn energy(&mut self, ctx: &Ctx, shape: &Shape, alpha: i32) -> f64 {
        self.counter += 1;
        let lines = shape.rasterize(self.w, self.h);
        let color = compute_color(ctx.target, ctx.current, &lines, alpha);
        copy_lines(&mut self.buffer, ctx.current, &lines);
        draw_lines(&mut self.buffer, color, &lines);
        difference_partial(ctx.target, ctx.current, &self.buffer, ctx.score, &lines)
    }

    fn energy_of(&mut self, ctx: &Ctx, st: &mut State) -> f64 {
        if st.score < 0.0 {
            st.score = self.energy(ctx, &st.shape, st.alpha);
        }
        st.score
    }

    fn random_state(&self, t: ShapeType, alpha: i32, rng: &mut Rng) -> State {
        State::new(Shape::random(t, self.w, self.h, rng), alpha)
    }

    fn best_random_state(
        &mut self,
        ctx: &Ctx,
        t: ShapeType,
        alpha: i32,
        n: i32,
        rng: &mut Rng,
    ) -> State {
        let mut best_energy = 0.0;
        let mut best: Option<State> = None;
        for i in 0..n {
            let mut st = self.random_state(t, alpha, rng);
            let e = self.energy_of(ctx, &mut st);
            if i == 0 || e < best_energy {
                best_energy = e;
                best = Some(st);
            }
        }
        best.expect("n >= 1")
    }

    /// Hill-climb from `st` for `max_age` non-improving steps — `optimize.go::HillClimb`.
    fn hill_climb(&mut self, ctx: &Ctx, mut st: State, max_age: i32, rng: &mut Rng) -> State {
        let mut best_energy = self.energy_of(ctx, &mut st);
        let mut best = st.clone();
        let mut age: i32 = 0;
        while age < max_age {
            // DoMove: copy for undo, mutate geometry (+ alpha), invalidate score.
            let undo = st.clone();
            st.shape.mutate(self.w, self.h, rng);
            if st.mutate_alpha {
                st.alpha = clamp_i32(st.alpha + rng.intn(21) - 10, 1, 255);
            }
            st.score = -1.0;

            let energy = self.energy_of(ctx, &mut st);
            if energy >= best_energy {
                st = undo; // UndoMove
            } else {
                best_energy = energy;
                best = st.clone();
                age = -1;
            }
            age += 1;
        }
        best
    }

    /// Best of `m` random-restart + hill-climb attempts — `Worker.BestHillClimbState`.
    pub fn best_hill_climb_state(&mut self, ctx: &Ctx, b: &SearchBudget, rng: &mut Rng) -> State {
        let mut best_energy = 0.0;
        let mut best: Option<State> = None;
        for i in 0..b.m {
            let st = self.best_random_state(ctx, b.shape_type, b.alpha, b.n, rng);
            let st = self.hill_climb(ctx, st, b.age, rng);
            let energy = st.score; // valid after hill_climb
            if i == 0 || energy < best_energy {
                best_energy = energy;
                best = Some(st);
            }
        }
        best.expect("m >= 1")
    }
}
