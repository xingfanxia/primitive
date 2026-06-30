//! GPU-3 gate (plan §7 / `.agent/EVIDENCE.md`): the whole search loop runs on-device and is both
//! **fast** and **good**:
//!   - throughput ≥ **460 shapes/sec** — the documented "≥ 20× the CORE-2 CPU baseline" target
//!     (the established baseline is ~22–24 shapes/sec at 128×128; the GPU is i32-capped to 64×64,
//!     so the absolute 460 sps figure is the stable cross-resolution target, not a same-run ratio
//!     against the unusually-fast 64×64 CPU). This is **hardware-dependent**: only enforced under
//!     `PRIMITIVE_PERF_GATE` (`make perf`) on representative hardware; `make verify` / CI measures +
//!     prints only (see `tests/common/mod.rs`);
//!   - final PSNR within **0.5 dB** of the CPU reference run on the same target — hardware-
//!     independent, so it stays an unconditional gate everywhere.
//!
//! The CPU side is the fogleman-reference hill-climb (`primitive_core::Model`); the GPU side is the
//! on-device Philox + energy-targeted + annealed parallel hill-climb + commit loop. Independent
//! optimizers, so quality is compared by PSNR, not pixels.

mod common;

use std::time::Instant;

use primitive_core::{difference_full, psnr_from_score, Canvas, Model, Rng, ShapeType};
use primitive_gpu_cubecl::{gpu_optimize, OptConfig};

const W: usize = 64;
const H: usize = 64;
const ALPHA: i32 = 128;
const SHAPES: usize = 100;
/// Documented end-to-end target: ≥ 20× the CORE-2 CPU baseline (~23 shapes/s) ⇒ ≳ 460 shapes/s.
const TARGET_SPS: f64 = 460.0;

/// A non-trivial but triangle-fittable target: a two-axis color gradient with a contrasting block.
fn synthetic_target() -> Canvas {
    let mut c = Canvas::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) * 4;
            let (mut r, mut g, mut b) = (
                (x * 255 / W) as u8,
                (y * 255 / H) as u8,
                ((x + y) * 255 / (W + H)) as u8,
            );
            if (16..40).contains(&x) && (20..44).contains(&y) {
                r = 230;
                g = 40;
                b = 90;
            }
            c.pix[i] = r;
            c.pix[i + 1] = g;
            c.pix[i + 2] = b;
            c.pix[i + 3] = 255;
        }
    }
    c
}

#[test]
fn gpu3_end_to_end_meets_throughput_and_psnr_gates() {
    let target = synthetic_target();

    // CPU reference: fogleman hill-climb, SHAPES shapes (also prints the same-run CPU rate).
    let mut model = Model::with_average_background(target.clone());
    let mut rng = Rng::new(1);
    let t_cpu = Instant::now();
    model.run(ShapeType::Triangle, ALPHA, SHAPES as i32, &mut rng);
    let cpu_secs = t_cpu.elapsed().as_secs_f64();
    let cpu_psnr = model.psnr();
    let cpu_sps = SHAPES as f64 / cpu_secs;

    // GPU: warm up (kernel compile + device init) on a tiny run, then the timed run.
    let warm = OptConfig {
        workers: 64,
        age: 8,
        shapes: 2,
        alpha: ALPHA,
        seed: 9,
        shape_type: ShapeType::Triangle,
    };
    let _ = gpu_optimize(&target, &warm);

    // Self-adaptive (1/5-rule) step lets a shallower, more-parallel config hold quality: 10240×9
    // clears the gate at higher throughput than the old fixed-anneal 6144×14 (see examples/sweep).
    let cfg = OptConfig {
        workers: 10240,
        age: 9,
        shapes: SHAPES,
        alpha: ALPHA,
        seed: 1,
        shape_type: ShapeType::Triangle,
    };
    let t_gpu = Instant::now();
    let recon = gpu_optimize(&target, &cfg);
    let gpu_secs = t_gpu.elapsed().as_secs_f64();
    let gpu_psnr = psnr_from_score(difference_full(&target, &recon));
    let gpu_sps = SHAPES as f64 / gpu_secs;

    println!(
        "GPU-3 end-to-end ({SHAPES} shapes, {W}×{H}, workers={} age={}):\n  CPU {cpu_psnr:.2} dB @ {cpu_sps:.1} shapes/s\n  GPU {gpu_psnr:.2} dB @ {gpu_sps:.1} shapes/s  (target ≥ {TARGET_SPS:.0})\n  PSNR gap {:.2} dB  |  GPU is {:.1}× the same-run 64×64 CPU",
        cfg.workers,
        cfg.age,
        gpu_psnr - cpu_psnr,
        gpu_sps / cpu_sps
    );

    // Hardware-dependent throughput gate — opt-in via PRIMITIVE_PERF_GATE / `make perf`.
    common::perf_gate_min(gpu_sps, TARGET_SPS, "GPU-3 throughput (shapes/sec)");
    // Quality gate — hardware-independent, always enforced.
    assert!(
        gpu_psnr >= cpu_psnr - 0.5,
        "GPU PSNR {gpu_psnr:.2} dB is >0.5 dB below CPU {cpu_psnr:.2} dB"
    );
}
