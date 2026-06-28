//! CORE-2 gate: the CPU adapter reproduces the CORE-1 reference, and the CPU baseline.
//!
//! This test file is the composition root (dev-dep on `primitive-gpu-cpu`): it wires the
//! `Engine` to the concrete `CpuSearch` adapter — the only place that names a backend.
//!
//! 1. **Parity** — `Engine<CpuSearch>` and `primitive_core::Model` produce a **byte-identical**
//!    reconstruction for the same seed, because the CPU adapter *is* the reference search.
//!    This makes the CPU adapter the permanent parity oracle (plan §7).
//! 2. **Baseline** — prints the single-threaded CPU shapes/sec and candidates/sec that every
//!    GPU throughput claim (GPU-2: ≥ 20×) is measured against. Run under `--release` for the
//!    real number.

use primitive_compute::{Backend, SearchParams};
use primitive_core::{Canvas, Model, Rng, ShapeType};
use primitive_engine::Engine;
use primitive_gpu_cpu::CpuSearch;
use std::time::Instant;

const SIZE: usize = 128;
const COUNT: i32 = 100;
const SEED: u64 = 1;

/// Deterministic synthetic target (same gradient as the parity fixture) — asset-independent.
fn synthetic_target(w: usize, h: usize) -> Canvas {
    let mut c = Canvas::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = c.pix_offset(x, y);
            c.pix[i] = ((x * 7 + y * 13) & 255) as u8;
            c.pix[i + 1] = ((x * 3 + y * 5) & 255) as u8;
            c.pix[i + 2] = ((x + y * 2) & 255) as u8;
            c.pix[i + 3] = 255;
        }
    }
    c
}

#[test]
fn engine_cpu_adapter_matches_core_reference_byte_for_byte() {
    let params = SearchParams::triangles_default();

    // Reference: core Model driven directly.
    let mut model = Model::with_average_background(synthetic_target(SIZE, SIZE));
    let mut rng_a = Rng::new(SEED);
    model.run(ShapeType::Triangle, params.alpha, COUNT, &mut rng_a);

    // Through the hexagonal seam: Engine + CpuSearch adapter.
    let mut engine = Engine::with_average_background(
        synthetic_target(SIZE, SIZE),
        CpuSearch::new(SIZE as i32, SIZE as i32),
    );
    let mut rng_b = Rng::new(SEED);
    engine.run(&params, COUNT, &mut rng_b, |_| {});

    assert_eq!(engine.backend(), Backend::Cpu);
    assert_eq!(engine.model.shapes.len(), model.shapes.len());
    assert_eq!(
        engine.model.score.to_bits(),
        model.score.to_bits(),
        "engine score must match core reference exactly"
    );
    assert_eq!(
        engine.model.current.pix, model.current.pix,
        "engine reconstruction must be byte-identical to the core reference"
    );
}

#[test]
fn cpu_baseline_throughput() {
    let params = SearchParams::triangles_default();
    let mut engine = Engine::with_average_background(
        synthetic_target(SIZE, SIZE),
        CpuSearch::new(SIZE as i32, SIZE as i32),
    );
    let mut rng = Rng::new(SEED);

    let mut evals: u64 = 0;
    let start = Instant::now();
    engine.run(&params, COUNT, &mut rng, |p| evals += p.evaluations);
    let secs = start.elapsed().as_secs_f64();

    let shapes_per_sec = COUNT as f64 / secs;
    let candidates_per_sec = evals as f64 / secs;
    println!(
        "CPU baseline ({}x{}, {} triangles, single-thread): {:.1} shapes/sec, {:.0} candidates/sec ({} evals in {:.3}s) backend={:?}",
        SIZE, SIZE, COUNT, shapes_per_sec, candidates_per_sec, evals, secs, engine.backend()
    );
    assert!(shapes_per_sec > 0.0 && candidates_per_sec > 0.0);
}
