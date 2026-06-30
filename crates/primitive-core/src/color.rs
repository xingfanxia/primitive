//! Closed-form optimal fill color over covered pixels.
//!
//! Faithful port of fogleman's `computeColor` (`core.go`): given the target, the current
//! canvas, the shape's covered scanlines, and an alpha, returns the per-channel
//! alpha-aware least-squares color. This is *the* reason color is removed from the search
//! space (plan §4) — geometry is all that's optimized.

use crate::canvas::Canvas;
use crate::raster::Scanline;

/// An RGBA color with integer channels in `[0, 255]` (alpha in `[1, 255]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Color {
    pub r: i32,
    pub g: i32,
    pub b: i32,
    pub a: i32,
}

#[inline]
fn clamp_i32(x: i32, lo: i32, hi: i32) -> i32 {
    x.max(lo).min(hi)
}

/// Optimal color for `lines` over `target` given `current` and `alpha` — `computeColor`.
pub fn compute_color(target: &Canvas, current: &Canvas, lines: &[Scanline], alpha: i32) -> Color {
    let (mut rsum, mut gsum, mut bsum, mut count): (i64, i64, i64, i64) = (0, 0, 0, 0);
    let a = 0x101 * 255 / alpha;
    for line in lines {
        let mut i = target.pix_offset(line.x1 as usize, line.y as usize);
        for _x in line.x1..=line.x2 {
            let tr = target.pix[i] as i32;
            let tg = target.pix[i + 1] as i32;
            let tb = target.pix[i + 2] as i32;
            let cr = current.pix[i] as i32;
            let cg = current.pix[i + 1] as i32;
            let cb = current.pix[i + 2] as i32;
            i += 4;
            rsum += ((tr - cr) * a + cr * 0x101) as i64;
            gsum += ((tg - cg) * a + cg * 0x101) as i64;
            bsum += ((tb - cb) * a + cb * 0x101) as i64;
            count += 1;
        }
    }
    if count == 0 {
        return Color::default();
    }
    let r = clamp_i32(((rsum / count) >> 8) as i32, 0, 255);
    let g = clamp_i32(((gsum / count) >> 8) as i32, 0, 255);
    let b = clamp_i32(((bsum / count) >> 8) as i32, 0, 255);
    Color { r, g, b, a: alpha }
}
