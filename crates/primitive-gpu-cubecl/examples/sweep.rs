//! Optimization sweep harness (not a gate — a measurement tool). Runs the CPU reference once, then
//! `gpu_optimize` across a matrix of (workers, age) configs at a chosen canvas size, printing
//! shapes/sec + PSNR for each. Sequential by construction so timings stay clean.
//!
//!   cargo run -p primitive-gpu-cubecl --release --example sweep -- [size] [shapes]
//!
//! Size defaults to 64 (the i32-safe cap); 128+ needs the hi-lo i64 path.

use std::time::Instant;

use primitive_core::{difference_full, psnr_from_score, Canvas, Model, Rng, ShapeType};
use primitive_gpu_cubecl::{gpu_optimize, OptConfig};

/// Same gradient+contrasting-block target as the GPU-3 gate, scaled to `w×h` (block kept at the
/// same relative region so PSNR is comparable across sizes).
fn synthetic_target(w: usize, h: usize) -> Canvas {
    let mut c = Canvas::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let (mut r, mut g, mut b) = (
                (x * 255 / w) as u8,
                (y * 255 / h) as u8,
                ((x + y) * 255 / (w + h)) as u8,
            );
            // x ∈ [w/4, 5w/8), y ∈ [5h/16, 11h/16) — == [16,40)×[20,44) at 64.
            if x * 8 >= w * 2 && x * 8 < w * 5 && y * 16 >= h * 5 && y * 16 < h * 11 {
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let size: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let shapes: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let alpha = 128i32;
    let target = synthetic_target(size, size);

    // CPU reference (fogleman hill-climb, single-thread) for the PSNR baseline.
    let mut model = Model::with_average_background(target.clone());
    let mut rng = Rng::new(1);
    let t = Instant::now();
    model.run(ShapeType::Triangle, alpha, shapes as i32, &mut rng);
    let cpu_secs = t.elapsed().as_secs_f64();
    let cpu_psnr = model.psnr();
    println!(
        "== {size}×{size}, {shapes} shapes, triangle a={alpha} ==\nCPU(1-thread)        {cpu_psnr:.2} dB @ {:.1} sps",
        shapes as f64 / cpu_secs
    );

    // Warm up kernel compile + device init off the clock.
    let _ = gpu_optimize(
        &target,
        &OptConfig {
            workers: 64,
            age: 8,
            shapes: 2,
            alpha,
            seed: 9,
            shape_type: ShapeType::Triangle,
        },
    );

    // Constant-ish work (workers×age ≈ 86k) plus a couple of wider/deeper points.
    // Fast→deep spread; the fast end is the 64×64 frontier, the deep end holds quality at 128×128.
    let configs: [(usize, u32); 6] = [
        (10240, 9),
        (8192, 11),
        (6144, 14),
        (4096, 21),
        (4096, 42),
        (2048, 84),
    ];
    for (workers, age) in configs {
        let cfg = OptConfig {
            workers,
            age,
            shapes,
            alpha,
            seed: 1,
            shape_type: ShapeType::Triangle,
        };
        let t = Instant::now();
        let recon = gpu_optimize(&target, &cfg);
        let secs = t.elapsed().as_secs_f64();
        let psnr = psnr_from_score(difference_full(&target, &recon));
        println!(
            "GPU w={workers:5} age={age:3}  {psnr:.2} dB @ {:.1} sps   (Δ{:+.2} dB vs CPU)",
            shapes as f64 / secs,
            psnr - cpu_psnr
        );
    }
}
