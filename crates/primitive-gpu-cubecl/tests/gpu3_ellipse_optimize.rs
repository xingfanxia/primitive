//! CORE-3b.3 gate: the GPU **ellipse** search loop (`evolve_ellipse` → `commit_ellipse`, fully
//! on-device) reconstructs as well as the CPU ellipse reference — final PSNR within tolerance. Like
//! the triangle GPU-3 gate, the two are independent optimizers (the GPU is a native energy-anchored
//! self-adaptive hill-climb, not a replay of the CPU search), so quality is compared by PSNR, not
//! pixels. Hardware-independent → unconditional in `make verify`.

use primitive_core::{Canvas, Model, Rng, ShapeType};
use primitive_gpu_cubecl::{gpu_optimize, OptConfig};

const W: usize = 64;
const H: usize = 64;
const ALPHA: i32 = 128;
const SHAPES: usize = 80;
/// The GPU native search must land within this many dB of the CPU ellipse reference. Both sides are
/// deterministic (fixed seeds), so the gap is reproducible — empirically **−0.32 dB** (GPU 33.99 vs
/// CPU 34.31 on this target); 1.0 dB gives a comfortable margin while still catching a real
/// quality regression in `evolve_ellipse`/`commit_ellipse`.
const PSNR_TOL_DB: f64 = 1.0;

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
fn gpu3_ellipse_search_matches_cpu_psnr() {
    let target = synthetic_target();

    // CPU reference: the fogleman hill-climb fitting ellipses.
    let mut model = Model::with_average_background(target.clone());
    let mut rng = Rng::new(1);
    model.run(ShapeType::Ellipse, ALPHA, SHAPES as i32, &mut rng);
    let cpu_psnr = model.psnr();

    // Warm up (kernel compile + device init), then the real GPU run.
    let warm = OptConfig {
        workers: 64,
        age: 8,
        shapes: 2,
        alpha: ALPHA,
        seed: 9,
        shape_type: ShapeType::Ellipse,
    };
    let _ = gpu_optimize(&target, &warm);

    let cfg = OptConfig {
        workers: 10240,
        age: 9,
        shapes: SHAPES,
        alpha: ALPHA,
        seed: 1,
        shape_type: ShapeType::Ellipse,
    };
    let recon = gpu_optimize(&target, &cfg);
    let gpu_psnr =
        primitive_core::psnr_from_score(primitive_core::difference_full(&target, &recon));

    println!(
        "CORE-3b.3 ellipse search: CPU {cpu_psnr:.2} dB | GPU {gpu_psnr:.2} dB | gap {:.2} dB",
        gpu_psnr - cpu_psnr
    );
    assert!(
        gpu_psnr >= cpu_psnr - PSNR_TOL_DB,
        "GPU ellipse search {gpu_psnr:.2} dB is > {PSNR_TOL_DB} dB below CPU {cpu_psnr:.2} dB"
    );
}
