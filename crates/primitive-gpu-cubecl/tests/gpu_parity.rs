//! GPU-1 gate: the GPU `score_candidates` kernel == the CPU integer delta-SSE, **exactly**,
//! for 1000 individual candidates (plan §7 GPU-1: "integer path: exact").
//!
//! Coverage is rasterized on the CPU (the deterministic oracle) and fed to the GPU; the GPU
//! does the closed-form color + composite + integer delta-SSE. A 64×64 target keeps every
//! accumulator within i32/u32 range with no overflow.

use primitive_core::{candidate_color_and_delta, Canvas, Rng, Shape, ShapeType};
use primitive_gpu_cubecl::{gpu_score_candidates, CandidateSpans};

#[test]
fn gpu_delta_sse_matches_cpu_exactly_for_1000_candidates() {
    const W: usize = 64;
    const H: usize = 64;
    const ALPHA: i32 = 128;
    const N: usize = 1000;

    // Deterministic target + a solid current canvas.
    let mut target = Canvas::new(W, H);
    for (k, px) in target.pix.chunks_exact_mut(4).enumerate() {
        px[0] = (k * 7 % 256) as u8;
        px[1] = (k * 13 % 256) as u8;
        px[2] = (k % 256) as u8;
        px[3] = 255;
    }
    let current = Canvas::filled(W, H, 120, 110, 100);

    // Generate N random triangle candidates; rasterize each on the CPU and score on the CPU.
    let mut rng = Rng::new(42);
    let mut scratch = Canvas::new(W, H);
    let mut spans: Vec<i32> = Vec::new();
    let mut offsets: Vec<i32> = vec![0];
    let mut alphas: Vec<i32> = Vec::new();
    let mut cpu: Vec<i64> = Vec::new();
    let mut covered_total = 0usize;

    for _ in 0..N {
        let shape = Shape::random(ShapeType::Triangle, W as i32, H as i32, &mut rng);
        let lines = shape.rasterize(W as i32, H as i32);
        let (_color, delta) =
            candidate_color_and_delta(&target, &current, &lines, ALPHA, &mut scratch);
        cpu.push(delta);
        for ln in &lines {
            spans.push(ln.y);
            spans.push(ln.x1);
            spans.push(ln.x2);
            covered_total += (ln.x2 - ln.x1 + 1) as usize;
        }
        offsets.push((spans.len() / 3) as i32);
        alphas.push(ALPHA);
    }

    let gpu = gpu_score_candidates(
        &target,
        &current,
        &CandidateSpans {
            spans,
            offsets,
            alphas,
        },
    );

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
        "GPU vs CPU integer delta-SSE: {}/{} exact; {} covered pixels scored on Metal; sample deltas {:?}",
        N - mismatches,
        N,
        covered_total,
        &cpu[..5]
    );
    assert_eq!(
        mismatches, 0,
        "{mismatches}/{N} candidates mismatched (integer path must be exact)"
    );
}
