//! GPU-2 throughput gate (plan §7): on-device candidates/sec ≥ 20× the single-core CPU.
//!
//! Both sides do the *same work* — integer-edge rasterize + closed-form color + integer
//! delta-SSE per candidate — so the ratio is apples-to-apples. The CPU side is the exact oracle
//! the GPU is checked against in `gpu2_triangles.rs`, just timed.

use std::time::Instant;

use primitive_core::{
    candidate_color_and_delta, rasterize_triangle_int, Canvas, Rng, Shape, ShapeType,
};
use primitive_gpu_cubecl::{GpuSession, TriangleBatch};

const W: usize = 64;
const H: usize = 64;
const ALPHA: i32 = 128;

fn target_canvas() -> Canvas {
    let mut target = Canvas::new(W, H);
    for (k, px) in target.pix.chunks_exact_mut(4).enumerate() {
        px[0] = (k * 7 % 256) as u8;
        px[1] = (k * 13 % 256) as u8;
        px[2] = (k % 256) as u8;
        px[3] = 255;
    }
    target
}

#[test]
fn gpu2_throughput_at_least_20x_single_core_cpu() {
    const BATCH: usize = 262_144;
    const GPU_REPS: usize = 8;
    const CPU_N: usize = 20_000;

    let target = target_canvas();
    let current = Canvas::filled(W, H, 120, 110, 100);

    // One batch of candidates; the CPU loop reuses the same triangles.
    let mut rng = Rng::new(11);
    let mut coords: Vec<[i32; 6]> = Vec::with_capacity(BATCH);
    let mut tris: Vec<i32> = Vec::with_capacity(BATCH * 6);
    for _ in 0..BATCH {
        let t = Shape::random(ShapeType::Triangle, W as i32, H as i32, &mut rng).triangle_coords();
        tris.extend_from_slice(&t);
        coords.push(t);
    }
    let batch = TriangleBatch {
        tris,
        alphas: vec![ALPHA; BATCH],
    };

    // GPU: target/current resident in the session; each step uploads only candidates and
    // rasterizes+scores them in-kernel. This is the candidates-scored-per-second rate (and it
    // still pays a full delta readback, which the on-device argmin path avoids — so it under-
    // states the resident rate). Warm up (kernel compile + device init), then time.
    let session = GpuSession::new(&target, &current);
    std::hint::black_box(session.score_triangles(&batch));
    let t0 = Instant::now();
    for _ in 0..GPU_REPS {
        std::hint::black_box(session.score_triangles(&batch));
    }
    let gpu_cps = (GPU_REPS * BATCH) as f64 / t0.elapsed().as_secs_f64();

    // CPU: same integer path, single core.
    let mut scratch = Canvas::new(W, H);
    let mut buf = Vec::new();
    let t1 = Instant::now();
    for i in 0..CPU_N {
        let t = coords[i % BATCH];
        buf.clear();
        rasterize_triangle_int(t, W as i32, H as i32, &mut buf);
        let r = candidate_color_and_delta(&target, &current, &buf, ALPHA, &mut scratch);
        std::hint::black_box(&r);
    }
    let cpu_cps = CPU_N as f64 / t1.elapsed().as_secs_f64();

    let speedup = gpu_cps / cpu_cps;
    println!(
        "GPU-2 throughput: GPU {:.2} M cand/s | CPU {:.2} M cand/s (integer path, 1 core) | speedup {:.1}×",
        gpu_cps / 1e6,
        cpu_cps / 1e6,
        speedup
    );
    assert!(speedup >= 20.0, "GPU throughput {speedup:.1}× < 20× gate");
}
