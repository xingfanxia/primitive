//! CORE-3b.2 gate: the GPU `score_rectangles` kernel (in-kernel integer rect raster + closed-form
//! color + integer delta-SSE) == the CPU integer path (`rasterize_rectangle_int` +
//! `candidate_color_and_delta`) **exactly** for 1000 random rectangle candidates. The same bit-exact
//! parity the triangle scorer holds (`gpu2_triangles.rs`), now for rectangles.

use primitive_core::{
    candidate_color_and_delta, rasterize_rectangle_int, Canvas, Rng, Shape, ShapeType,
};
use primitive_gpu_cubecl::{gpu_score_rectangles, RectBatch};

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

#[test]
fn gpu2_rect_raster_score_matches_cpu_exactly() {
    let target = target_canvas();
    let current = Canvas::filled(W, H, 120, 110, 100);
    let mut rng = Rng::new(7);
    let mut scratch = Canvas::new(W, H);
    let mut buf = Vec::new();
    let mut rects = Vec::with_capacity(N * 4);
    let mut alphas = Vec::with_capacity(N);
    let mut cpu = Vec::with_capacity(N);

    for _ in 0..N {
        let shape = Shape::random(ShapeType::Rectangle, W as i32, H as i32, &mut rng);
        let [x1, y1, x2, y2] = shape.rectangle_coords();
        buf.clear();
        rasterize_rectangle_int(x1, y1, x2, y2, W as i32, H as i32, &mut buf);
        let (_color, delta) =
            candidate_color_and_delta(&target, &current, &buf, ALPHA, &mut scratch);
        cpu.push(delta);
        rects.extend_from_slice(&[x1, y1, x2, y2]);
        alphas.push(ALPHA);
    }

    let gpu = gpu_score_rectangles(&target, &current, &RectBatch { rects, alphas });
    assert_eq!(gpu.len(), N);

    let mut mismatches = 0;
    for i in 0..N {
        if gpu[i] as i64 != cpu[i] {
            if mismatches < 5 {
                eprintln!("rect {i}: gpu={} cpu={}", gpu[i], cpu[i]);
            }
            mismatches += 1;
        }
    }
    println!(
        "CORE-3b.2 on-device rect raster+score: {}/{} candidates bit-identical to CPU integer path",
        N - mismatches,
        N
    );
    assert_eq!(
        mismatches, 0,
        "{mismatches}/{N} mismatched (on-device integer rect path must be exact)"
    );
}
