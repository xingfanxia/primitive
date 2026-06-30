//! Single-candidate evaluation: the exact integer color + delta-SSE a shape would score.
//!
//! This ties together closed-form color → composite → integer delta-SSE for one candidate's
//! coverage. It is the precise computation the GPU `raster_score` kernel reproduces, so it is
//! the parity oracle for every GPU scoring milestone (the GPU is checked against this, exactly).

use crate::canvas::Canvas;
use crate::color::{compute_color, Color};
use crate::draw::{copy_lines, draw_lines};
use crate::raster::Scanline;
use crate::score::delta_sse_partial;

/// The closed-form fill color and the signed integer delta-SSE for placing a shape (given its
/// covered `lines` and `alpha`) over `current`, scored against `target`.
///
/// Equivalent to one `Searcher::energy` call but returning the *integer* delta (pre-float),
/// plus the color. `scratch` is a reusable `w×h` buffer (avoids per-call allocation).
pub fn candidate_color_and_delta(
    target: &Canvas,
    current: &Canvas,
    lines: &[Scanline],
    alpha: i32,
    scratch: &mut Canvas,
) -> (Color, i64) {
    let color = compute_color(target, current, lines, alpha);
    copy_lines(scratch, current, lines);
    draw_lines(scratch, color, lines);
    let delta = delta_sse_partial(target, current, scratch, lines);
    (color, delta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::{crop_scanlines, rasterize_triangle};

    #[test]
    fn delta_is_zero_for_empty_coverage() {
        let target = Canvas::filled(8, 8, 10, 20, 30);
        let current = Canvas::filled(8, 8, 0, 0, 0);
        let mut scratch = Canvas::new(8, 8);
        let (_c, delta) = candidate_color_and_delta(&target, &current, &[], 128, &mut scratch);
        assert_eq!(delta, 0);
    }

    #[test]
    fn delta_matches_recomputed_sse_difference() {
        // The incremental delta must equal a full before/after SSE recompute over the bbox.
        let mut target = Canvas::new(16, 16);
        for (k, px) in target.pix.chunks_exact_mut(4).enumerate() {
            px[0] = (k * 3 % 256) as u8;
            px[1] = (k * 5 % 256) as u8;
            px[2] = (k * 7 % 256) as u8;
            px[3] = 255;
        }
        let current = Canvas::filled(16, 16, 100, 110, 120);
        let mut lines = Vec::new();
        rasterize_triangle(2, 2, 13, 4, 6, 12, &mut lines);
        crop_scanlines(&mut lines, 16, 16);

        let mut scratch = Canvas::new(16, 16);
        let (_c, delta) = candidate_color_and_delta(&target, &current, &lines, 128, &mut scratch);

        // Recompute: build the same `after` and diff the full-image SSE.
        let color = compute_color(&target, &current, &lines, 128);
        let mut after = current.clone();
        copy_lines(&mut after, &current, &lines);
        draw_lines(&mut after, color, &lines);
        let expected = crate::score::sse_full(&target, &after) as i64
            - crate::score::sse_full(&target, &current) as i64;
        assert_eq!(delta, expected);
    }
}
