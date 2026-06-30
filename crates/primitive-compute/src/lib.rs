//! # primitive-compute — the ports layer
//!
//! Traits + DTOs the engine orchestrates through, with **no** concrete backend. The CPU
//! adapter (`primitive-gpu-cpu`) and the future CubeCL adapter (`primitive-gpu-cubecl`)
//! both implement [`ShapeSearch`]; the engine is generic over it and never names a backend.
//!
//! Import direction (plan §5): `core` ← **`compute`** ← `engine` ← adapters / `app`.
//! This crate may depend only on `primitive-core`.

#![forbid(unsafe_code)]

use primitive_core::{Canvas, Rng, Shape};

/// Per-shape search budget. One source of truth: defined in `primitive-core` (the search owns
/// it) and surfaced here as the port's request type.
pub use primitive_core::SearchBudget as SearchParams;

/// Which compute backend is servicing the search — drives the GUI device chip (plan §5A).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// CPU/Rayon adapter — the permanent parity oracle and graceful-degradation fallback.
    Cpu,
    /// Apple GPU via CubeCL/wgpu (Metal).
    Metal,
    /// NVIDIA GPU via CubeCL (CUDA).
    Cuda,
}

impl Backend {
    /// Short backend identity for the device chip (`Metal` / `CUDA` / `CPU`).
    ///
    /// This is the backend *identity*, not the reason for it: the CPU adapter is also a
    /// deliberate choice (the parity oracle, or a future `--cpu` flag), so the GUI appends
    /// the "(no GPU found)" qualifier (plan §5A) only when it actually fell back — see
    /// [`Backend::cpu_fallback_chip`].
    pub fn chip_label(self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            Backend::Metal => "Metal",
            Backend::Cuda => "CUDA",
        }
    }

    /// The amber-chip label the GUI shows when it fell back to CPU because no GPU was found.
    pub fn cpu_fallback_chip() -> &'static str {
        "CPU (no GPU found)"
    }
}

/// The winning shape from a search, plus how many candidates were evaluated (for n/s).
#[derive(Clone, Debug)]
pub struct BestShape {
    pub shape: Shape,
    pub alpha: i32,
    pub evaluations: u64,
}

/// The port: given the current reconstruction state, find the best next shape to commit.
///
/// The engine owns canvas state (a `primitive_core::Model`) and calls this once per shape,
/// then commits the result. A GPU adapter does the same search massively in parallel; the
/// CPU adapter delegates to `primitive_core`'s reference `Searcher`, making it the oracle.
pub trait ShapeSearch {
    /// Which backend this implementation runs on.
    fn backend(&self) -> Backend;

    /// Find the best next shape for `current` (over `target`, given the running `score`).
    fn find_best_shape(
        &mut self,
        target: &Canvas,
        current: &Canvas,
        score: f64,
        params: &SearchParams,
        rng: &mut Rng,
    ) -> BestShape;
}
