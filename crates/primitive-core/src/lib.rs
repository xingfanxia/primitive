//! # primitive-core
//!
//! The pure domain core for the GPU-native rebuild of fogleman/primitive. Reproduces a
//! target image with geometric primitives via closed-form color + bounding-box delta-SSE
//! scoring + a random-restart hill-climb / (1+1)-ES search.
//!
//! **Purity contract (plan §5, enforced by `tools/verify/check-boundaries.sh`):** this crate
//! imports nothing GPU, no IO, no `wgpu`, no clock, no global RNG. Every stochastic routine
//! takes an injected [`rng::Rng`]. It is the deterministic repair target and the permanent
//! parity oracle every GPU adapter is measured against.
//!
//! Module map mirrors the architecture plan:
//! - [`canvas`] — RGBA pixel buffer (`image.RGBA` analogue)
//! - [`shape`] — primitives + geometry mutation (`raster_ref`/shape)
//! - [`raster`] — scanline rasterization (the golden rasterizer)
//! - [`color`] — closed-form optimal fill color (`color_solve`)
//! - [`draw`] — alpha compositing of covered scanlines
//! - [`score`] — full + bbox-local delta-SSE metric (`score`)
//! - [`optimizer`] — fogleman-reference hill-climb search
//! - [`es`] — self-adaptive (1+1)-ES (the §4 upgrade)
//! - [`energy_map`] — residual→PDF restart sampling
//! - [`model`] — the reference run loop

#![forbid(unsafe_code)]

pub mod canvas;
pub mod color;
pub mod draw;
pub mod energy_map;
pub mod es;
pub mod eval;
pub mod model;
pub mod optimizer;
pub mod philox;
pub mod raster;
pub mod raster_int;
pub mod rng;
pub mod score;
pub mod shape;

pub use canvas::Canvas;
pub use color::{compute_color, Color};
pub use eval::candidate_color_and_delta;
pub use model::{psnr_from_score, AddedShape, Model};
pub use optimizer::{Ctx, SearchBudget, Searcher, State};
pub use philox::{mulhilo, philox, rand_below, rand_u32};
pub use raster::Scanline;
pub use raster_int::{
    ellipse_inside, rasterize_ellipse_int, rasterize_rectangle_int, rasterize_triangle_int,
    triangle_inside,
};
pub use rng::Rng;
pub use score::{delta_sse_partial, difference_full, difference_partial, sse_full};
pub use shape::{Ellipse, Rectangle, Shape, ShapeType, Triangle};
