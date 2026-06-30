//! CORE-3a gate: Ellipse + Rectangle shape support (plan §10 Q5 — "triangle + ellipse + rect
//! first, expand after PKG-1").
//!
//! The fogleman bit-exact parity fixture (`parity.rs` / `parity_fogleman.json`) is triangle-only;
//! extending its Go dumper to ellipse/rect is the CORE-3a.2 follow-up. Until then, correctness of
//! the two new shapes is pinned three ways here:
//!   1. **rasterizer geometry** — hand-verified scanline spans for a known circle + rectangle
//!      (these are the exact spans fogleman's `Ellipse.Rasterize` / `Rectangle.Rasterize` emit);
//!   2. **determinism** — same seed + target ⇒ byte-identical reconstruction (the search is pure);
//!   3. **effectiveness** — an N-shape run with each type strictly and substantially reduces the
//!      score (the optimizer drives reconstruction end-to-end) and exports well-formed SVG.

use std::collections::BTreeMap;

use primitive_core::raster::{rasterize_ellipse, rasterize_rectangle, Scanline};
use primitive_core::{Canvas, Model, Rng, ShapeType};

/// Collect scanlines into a `y -> (x1, x2)` map (one span per row for these convex shapes).
fn spans(lines: &[Scanline]) -> BTreeMap<i32, (i32, i32)> {
    lines.iter().map(|l| (l.y, (l.x1, l.x2))).collect()
}

#[test]
fn rectangle_rasterizes_one_span_per_row_sorted_corners() {
    // Unsorted corners (x1>x2 here would still sort); inclusive [2,5] × [3,7] ⇒ rows 3..=7.
    let mut buf = Vec::new();
    rasterize_rectangle(2, 3, 5, 7, &mut buf);
    let s = spans(&buf);
    assert_eq!(s.len(), 5, "rows 3..=7 inclusive");
    for y in 3..=7 {
        assert_eq!(s[&y], (2, 5), "row {y} spans the full inclusive x range");
    }

    // Corner order must not matter (bounds() sorts): same rectangle, corners swapped.
    let mut buf2 = Vec::new();
    rasterize_rectangle(5, 7, 2, 3, &mut buf2);
    assert_eq!(spans(&buf2), s, "swapped corners ⇒ identical raster");
}

#[test]
fn ellipse_rasterizes_analytic_circle_spans() {
    // A radius-3 circle centred at (5,5) on a 20×20 canvas. fogleman's per-row half-width is
    // int(sqrt(9 - dy²)): dy0→3, dy1→int(2.82)=2, dy2→int(2.23)=2. So the centre row is widest.
    let mut buf = Vec::new();
    rasterize_ellipse(5, 5, 3, 3, 20, 20, &mut buf);
    let s = spans(&buf);
    assert_eq!(s.len(), 5, "rows 3..=7 (centre + two mirrored pairs)");
    assert_eq!(s[&5], (2, 8), "centre row: half-width 3");
    for y in [3, 4, 6, 7] {
        assert_eq!(s[&y], (3, 7), "row {y}: half-width 2");
    }
}

#[test]
fn ellipse_rasterizes_aspect_ratio_spans() {
    // rx ≠ ry pins the `* aspect` term (a circle's aspect == 1 hides bugs in it). Centre (8,8),
    // rx=6, ry=3 ⇒ aspect=2, half-width = int(sqrt(9−dy²) · 2):
    //   dy0 → int(3·2)=6 → row 8 = [2,14];  dy1 → int(2.828·2)=5 → rows 7,9 = [3,13];
    //   dy2 → int(2.236·2)=4 → rows 6,10 = [4,12].
    let mut buf = Vec::new();
    rasterize_ellipse(8, 8, 6, 3, 20, 20, &mut buf);
    let s = spans(&buf);
    assert_eq!(s.len(), 5, "rows 6..=10");
    assert_eq!(s[&8], (2, 14), "centre row: half-width int(3·2)=6");
    assert_eq!(s[&7], (3, 13), "dy=1: half-width int(sqrt(8)·2)=5");
    assert_eq!(s[&9], (3, 13));
    assert_eq!(s[&6], (4, 12), "dy=2: half-width int(sqrt(5)·2)=4");
    assert_eq!(s[&10], (4, 12));
}

#[test]
fn ellipse_clamps_spans_to_canvas() {
    // Centre near the left/top edge: spans must clamp into [0, w) / drop out-of-frame rows.
    let mut buf = Vec::new();
    rasterize_ellipse(1, 1, 5, 5, 16, 16, &mut buf);
    for l in &buf {
        assert!(l.x1 >= 0 && l.x2 < 16, "x clamped into the canvas: {l:?}");
        assert!(l.y >= 0 && l.y < 16, "y inside the canvas: {l:?}");
        assert!(l.x1 <= l.x2, "non-empty span: {l:?}");
    }
}

/// A synthetic, easily-reconstructable target: a two-axis gradient with a contrasting block.
fn synthetic_target(w: usize, h: usize) -> Canvas {
    let mut c = Canvas::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            c.pix[i] = (x * 255 / w) as u8;
            c.pix[i + 1] = (y * 255 / h) as u8;
            c.pix[i + 2] = ((x + y) * 255 / (w + h)) as u8;
            c.pix[i + 3] = 255;
            if (w / 4..w / 2).contains(&x) && (h / 3..2 * h / 3).contains(&y) {
                c.pix[i] = 230;
                c.pix[i + 1] = 40;
                c.pix[i + 2] = 90;
            }
        }
    }
    c
}

fn run(t: ShapeType, seed: u64, count: i32) -> Model {
    let mut model = Model::with_average_background(synthetic_target(48, 48));
    let mut rng = Rng::new(seed);
    model.run(t, 128, count, &mut rng);
    model
}

#[test]
fn ellipse_and_rectangle_runs_are_deterministic() {
    for t in [ShapeType::Ellipse, ShapeType::Rectangle] {
        let a = run(t, 7, 40);
        let b = run(t, 7, 40);
        assert_eq!(a.score, b.score, "{t:?}: same seed ⇒ identical score");
        assert_eq!(
            a.current.pix, b.current.pix,
            "{t:?}: same seed ⇒ byte-identical reconstruction"
        );
        assert_eq!(a.shapes.len(), b.shapes.len());
    }
}

#[test]
fn ellipse_and_rectangle_reconstruct_the_target() {
    const COUNT: i32 = 60;
    for t in [ShapeType::Ellipse, ShapeType::Rectangle] {
        let initial = Model::with_average_background(synthetic_target(48, 48)).score;
        let model = run(t, 1, COUNT);
        println!(
            "{t:?}: score {initial:.5} -> {:.5} ({:.0}% of initial) over {COUNT} shapes, PSNR {:.2} dB",
            model.score,
            model.score / initial * 100.0,
            model.psnr()
        );
        // The hill-climb only ever accepts a strictly-improving shape, so the score is monotone;
        // a real reconstruction cuts it well below the flat-background baseline.
        assert_eq!(
            model.shapes.len(),
            COUNT as usize,
            "{t:?}: every shape added"
        );
        assert!(
            model.score < initial * 0.6,
            "{t:?}: {COUNT} shapes should cut the score to <60% of the flat baseline (got {:.1}%)",
            model.score / initial * 100.0
        );
    }
}

#[test]
fn ellipse_and_rectangle_export_valid_svg() {
    let e = run(ShapeType::Ellipse, 3, 10).svg();
    assert!(
        e.contains("<ellipse "),
        "ellipse run emits <ellipse> elements"
    );
    assert!(
        e.contains("<svg") && e.contains("</svg>"),
        "well-formed SVG root"
    );

    let r = run(ShapeType::Rectangle, 3, 10).svg();
    assert!(r.contains("<rect "), "rectangle run emits <rect> elements");
    assert!(
        r.contains("<svg") && r.contains("</svg>"),
        "well-formed SVG root"
    );
}
