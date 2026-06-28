//! Error metric — full and bounding-box-local (delta) scoring.
//!
//! Faithful port of fogleman's `differenceFull` / `differencePartial` (`core.go`). The
//! score is normalized RMSE over all four RGBA channels: `sqrt(SSE / (w·h·4)) / 255`.
//!
//! `difference_partial` is the load-bearing trick the Python/MPS port discarded (plan §2):
//! it reconstructs the absolute SSE from the running score and updates only the candidate's
//! bounding-box pixels, so full-image error is never recomputed. The SSE accumulator is
//! integer (`u64`) — associative, therefore order/backend-independent (plan §6.6).

use crate::canvas::Canvas;
use crate::raster::Scanline;

/// Raw integer SSE between two equal-size canvases over all four channels.
pub fn sse_full(a: &Canvas, b: &Canvas) -> u64 {
    debug_assert_eq!(
        a.pix.len(),
        b.pix.len(),
        "sse_full requires equal-size canvases"
    );
    let mut total: u64 = 0;
    for (pa, pb) in a.pix.chunks_exact(4).zip(b.pix.chunks_exact(4)) {
        let dr = pa[0] as i32 - pb[0] as i32;
        let dg = pa[1] as i32 - pb[1] as i32;
        let db = pa[2] as i32 - pb[2] as i32;
        let da = pa[3] as i32 - pb[3] as i32;
        total += (dr * dr + dg * dg + db * db + da * da) as u64;
    }
    total
}

/// Normalized RMSE score in `[0, 1]` — `differenceFull`.
pub fn difference_full(a: &Canvas, b: &Canvas) -> f64 {
    let n = (a.w * a.h * 4) as f64;
    (sse_full(a, b) as f64 / n).sqrt() / 255.0
}

/// Incremental score after compositing a shape — `differencePartial`.
///
/// `score` is the score of `before`; `before`/`after` differ only inside `lines`.
pub fn difference_partial(
    target: &Canvas,
    before: &Canvas,
    after: &Canvas,
    score: f64,
    lines: &[Scanline],
) -> f64 {
    let w = target.w;
    let h = target.h;
    let n = (w * h * 4) as f64;
    // Reconstruct absolute SSE from the running score (float round-trip, matched to Go).
    let mut total: u64 = ((score * 255.0).powi(2) * n) as u64;
    for line in lines {
        let mut i = target.pix_offset(line.x1 as usize, line.y as usize);
        for _x in line.x1..=line.x2 {
            let tr = target.pix[i] as i32;
            let tg = target.pix[i + 1] as i32;
            let tb = target.pix[i + 2] as i32;
            let ta = target.pix[i + 3] as i32;
            let br = before.pix[i] as i32;
            let bg = before.pix[i + 1] as i32;
            let bb = before.pix[i + 2] as i32;
            let ba = before.pix[i + 3] as i32;
            let ar = after.pix[i] as i32;
            let ag = after.pix[i + 1] as i32;
            let ab = after.pix[i + 2] as i32;
            let aa = after.pix[i + 3] as i32;
            i += 4;
            let dr1 = tr - br;
            let dg1 = tg - bg;
            let db1 = tb - bb;
            let da1 = ta - ba;
            let dr2 = tr - ar;
            let dg2 = tg - ag;
            let db2 = tb - ab;
            let da2 = ta - aa;
            total -= (dr1 * dr1 + dg1 * dg1 + db1 * db1 + da1 * da1) as u64;
            total += (dr2 * dr2 + dg2 * dg2 + db2 * db2 + da2 * da2) as u64;
        }
    }
    (total as f64 / n).sqrt() / 255.0
}
