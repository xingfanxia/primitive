//! GPU-2 gate (plan §7): fully on-device rasterize + score + reduce_argmin.
//!
//! 1. **Parity** — the GPU `score_triangles` kernel (integer edge-function raster + closed-form
//!    color + integer delta-SSE, all in-kernel) == the CPU integer path
//!    (`rasterize_triangle_int` + `candidate_color_and_delta`) **exactly** for 1000 candidates.
//! 2. **Argmin** — the on-device `argmin` reduction picks the same winning index as CPU brute
//!    force (first-minimum tie-break).
//!
//! Throughput (≥20× CPU) is measured separately in `gpu2_throughput.rs`.

use primitive_core::{
    candidate_color_and_delta, rasterize_triangle_int, Canvas, Rng, Shape, ShapeType,
};
use primitive_gpu_cubecl::{gpu_best_triangle, gpu_score_triangles, TriangleBatch};

const W: usize = 64;
const H: usize = 64;
const ALPHA: i32 = 128;
const N: usize = 1000;

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

/// First-minimum argmin — the CPU brute-force winner the GPU reduction must match.
fn argmin_first(v: &[i64]) -> usize {
    let mut best = v[0];
    let mut best_i = 0;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x < best {
            best = x;
            best_i = i;
        }
    }
    best_i
}

fn build_batch() -> (Canvas, Canvas, TriangleBatch, Vec<i64>) {
    let target = target_canvas();
    let current = Canvas::filled(W, H, 120, 110, 100);
    let mut rng = Rng::new(7);
    let mut scratch = Canvas::new(W, H);
    let mut buf = Vec::new();
    let mut tris = Vec::with_capacity(N * 6);
    let mut alphas = Vec::with_capacity(N);
    let mut cpu = Vec::with_capacity(N);

    for _ in 0..N {
        let shape = Shape::random(ShapeType::Triangle, W as i32, H as i32, &mut rng);
        let t = shape.triangle_coords();
        buf.clear();
        rasterize_triangle_int(t, W as i32, H as i32, &mut buf);
        let (_color, delta) =
            candidate_color_and_delta(&target, &current, &buf, ALPHA, &mut scratch);
        cpu.push(delta);
        tris.extend_from_slice(&t);
        alphas.push(ALPHA);
    }
    (target, current, TriangleBatch { tris, alphas }, cpu)
}

#[test]
fn gpu2_on_device_raster_score_matches_cpu_exactly() {
    let (target, current, batch, cpu) = build_batch();
    let gpu = gpu_score_triangles(&target, &current, &batch);

    assert_eq!(gpu.len(), N);
    let mut mismatches = 0;
    for i in 0..N {
        if gpu[i] as i64 != cpu[i] {
            if mismatches < 5 {
                eprintln!("candidate {i}: gpu={} cpu={}", gpu[i], cpu[i]);
            }
            mismatches += 1;
        }
    }
    println!(
        "GPU-2 on-device raster+score: {}/{} candidates bit-identical to CPU integer path",
        N - mismatches,
        N
    );
    assert_eq!(
        mismatches, 0,
        "{mismatches}/{N} mismatched (on-device integer path must be exact)"
    );
}

#[test]
fn gpu2_argmin_picks_same_winner_as_cpu() {
    let (target, current, batch, cpu) = build_batch();
    let cpu_winner = argmin_first(&cpu);
    let gpu_winner = gpu_best_triangle(&target, &current, &batch) as usize;
    println!(
        "GPU-2 reduce_argmin: winner={gpu_winner} (delta {}), CPU brute-force winner={cpu_winner} (delta {})",
        cpu[gpu_winner], cpu[cpu_winner]
    );
    assert_eq!(
        gpu_winner, cpu_winner,
        "on-device argmin must match CPU brute force"
    );
}
