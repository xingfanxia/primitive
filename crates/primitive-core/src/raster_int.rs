//! Deterministic integer rasterizers — the GPU-shared path (plan §6.6).
//!
//! [`crate::raster`] holds fogleman's f64 edge-walk rasterizers (the CORE/golden *reference*).
//! Metal has no f64, so the GPU path uses the integer rasterizers here instead: pure integer
//! arithmetic, trivially identical on CPU and GPU (no float rounding to diverge on), so the GPU and
//! its CPU oracle cover the exact same pixels. Coverage differs from the f64 reference by sub-pixel
//! edges — expected and fine; this is the deterministic production path, the f64 one stays the
//! historical reference. Each kernel in `primitive-gpu-cubecl` mirrors one function here.

use crate::raster::{clamp_i32, Scanline};

/// Twice the signed area of triangle `(a, b, p)` — the integer edge function. Inside a triangle,
/// all three edge functions share one sign (positive for CCW, negative for CW).
#[inline]
pub fn edge(ax: i32, ay: i32, bx: i32, by: i32, px: i32, py: i32) -> i32 {
    (bx - ax) * (py - ay) - (by - ay) * (px - ax)
}

/// Is pixel `(px, py)` inside (or on the boundary of) the triangle? Integer, winding-agnostic.
#[inline]
pub fn triangle_inside(t: [i32; 6], px: i32, py: i32) -> bool {
    let e0 = edge(t[0], t[1], t[2], t[3], px, py);
    let e1 = edge(t[2], t[3], t[4], t[5], px, py);
    let e2 = edge(t[4], t[5], t[0], t[1], px, py);
    (e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0)
}

/// Deterministic integer rasterization of a triangle into cropped scanlines.
///
/// Scans the clamped bounding box and emits, per row, the contiguous covered run (valid
/// triangles are convex, so inside pixels on a row are contiguous). This is the rasterizer the
/// GPU kernel reproduces pixel-for-pixel.
pub fn rasterize_triangle_int(t: [i32; 6], w: i32, h: i32, buf: &mut Vec<Scanline>) {
    let xmin = clamp_i32(t[0].min(t[2]).min(t[4]), 0, w - 1);
    let xmax = clamp_i32(t[0].max(t[2]).max(t[4]), 0, w - 1);
    let ymin = clamp_i32(t[1].min(t[3]).min(t[5]), 0, h - 1);
    let ymax = clamp_i32(t[1].max(t[3]).max(t[5]), 0, h - 1);
    for py in ymin..=ymax {
        let mut lo = -1;
        let mut hi = -1;
        for px in xmin..=xmax {
            if triangle_inside(t, px, py) {
                if lo < 0 {
                    lo = px;
                }
                hi = px;
            }
        }
        if lo >= 0 {
            buf.push(Scanline {
                y: py,
                x1: lo,
                x2: hi,
                alpha: 0xffff,
            });
        }
    }
}

/// Is pixel `(px, py)` inside (or on the boundary of) the axis-aligned ellipse centred `(cx, cy)`
/// with radii `(rx, ry)`? The integer implicit test `ry²·dx² + rx²·dy² ≤ rx²·ry²` — no `sqrt`, so
/// it's trivially identical on CPU and GPU (the integer analogue of [`triangle_inside`]). Squaring
/// makes the sign of `rx`/`ry` irrelevant. Computed in `i64`: `ry²·dx²` overflows `i32` past ~46k px,
/// so the GPU mirror (i32) is valid only within the documented canvas bound (§6.6), where the two
/// agree bit-for-bit because no overflow occurs.
#[inline]
pub fn ellipse_inside(cx: i32, cy: i32, rx: i32, ry: i32, px: i32, py: i32) -> bool {
    let dx = (px - cx) as i64;
    let dy = (py - cy) as i64;
    let (rx, ry) = (rx as i64, ry as i64);
    ry * ry * dx * dx + rx * rx * dy * dy <= rx * rx * ry * ry
}

/// Deterministic integer rasterization of an axis-aligned ellipse into cropped scanlines. Scans the
/// clamped bounding box and emits, per row, the contiguous covered run (ellipses are convex, so
/// inside pixels on a row are contiguous). The integer counterpart of [`crate::raster::rasterize_ellipse`]
/// and the rasterizer the GPU kernel will reproduce pixel-for-pixel (CORE-3b).
pub fn rasterize_ellipse_int(
    cx: i32,
    cy: i32,
    rx: i32,
    ry: i32,
    w: i32,
    h: i32,
    buf: &mut Vec<Scanline>,
) {
    let (rx, ry) = (rx.abs(), ry.abs());
    let xmin = clamp_i32(cx - rx, 0, w - 1);
    let xmax = clamp_i32(cx + rx, 0, w - 1);
    let ymin = clamp_i32(cy - ry, 0, h - 1);
    let ymax = clamp_i32(cy + ry, 0, h - 1);
    for py in ymin..=ymax {
        let mut lo = -1;
        let mut hi = -1;
        for px in xmin..=xmax {
            if ellipse_inside(cx, cy, rx, ry, px, py) {
                if lo < 0 {
                    lo = px;
                }
                hi = px;
            }
        }
        if lo >= 0 {
            buf.push(Scanline {
                y: py,
                x1: lo,
                x2: hi,
                alpha: 0xffff,
            });
        }
    }
}

/// Deterministic integer rasterization of an axis-aligned rectangle with inclusive opposite corners
/// `(x1, y1)`–`(x2, y2)` (any order) into cropped scanlines: one full-width span per row, clamped to
/// the image. Already all-integer (no f64 to diverge), so the GPU mirror is trivial; this is the
/// self-cropping counterpart of [`crate::raster::rasterize_rectangle`] (which leaves cropping to the caller).
pub fn rasterize_rectangle_int(
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    w: i32,
    h: i32,
    buf: &mut Vec<Scanline>,
) {
    let xlo = clamp_i32(x1.min(x2), 0, w - 1);
    let xhi = clamp_i32(x1.max(x2), 0, w - 1);
    let ylo = clamp_i32(y1.min(y2), 0, h - 1);
    let yhi = clamp_i32(y1.max(y2), 0, h - 1);
    for py in ylo..=yhi {
        buf.push(Scanline {
            y: py,
            x1: xlo,
            x2: xhi,
            alpha: 0xffff,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::rasterize_ellipse;

    #[test]
    fn int_raster_coverage_is_contiguous_and_inside() {
        let t = [2, 2, 30, 6, 10, 28];
        let mut buf = Vec::new();
        rasterize_triangle_int(t, 32, 32, &mut buf);
        assert!(!buf.is_empty());
        for s in &buf {
            assert!(s.x1 <= s.x2);
            // every pixel in the emitted run is actually inside
            for x in s.x1..=s.x2 {
                assert!(triangle_inside(t, x, s.y), "({x},{}) not inside", s.y);
            }
            // the pixel just left of the run is outside (run is the maximal inside span)
            if s.x1 > 0 {
                assert!(!triangle_inside(t, s.x1 - 1, s.y));
            }
        }
    }

    #[test]
    fn int_raster_is_deterministic() {
        let t = [1, 0, 20, 5, 7, 19];
        let mut a = Vec::new();
        let mut b = Vec::new();
        rasterize_triangle_int(t, 24, 24, &mut a);
        rasterize_triangle_int(t, 24, 24, &mut b);
        assert_eq!(a, b);
    }

    // ── CORE-3b.1: integer ellipse / rectangle rasterizers (the GPU-shared path) ──────────

    #[test]
    fn ellipse_inside_matches_implicit_test() {
        // centre + the four axis extrema are on/inside the boundary; one step past each axis is out.
        let (cx, cy, rx, ry) = (10, 10, 6, 3);
        assert!(ellipse_inside(cx, cy, rx, ry, cx, cy), "centre is inside");
        assert!(
            ellipse_inside(cx, cy, rx, ry, cx + rx, cy),
            "+x vertex on boundary"
        );
        assert!(
            ellipse_inside(cx, cy, rx, ry, cx - rx, cy),
            "-x vertex on boundary"
        );
        assert!(
            ellipse_inside(cx, cy, rx, ry, cx, cy + ry),
            "+y vertex on boundary"
        );
        assert!(
            ellipse_inside(cx, cy, rx, ry, cx, cy - ry),
            "-y vertex on boundary"
        );
        assert!(
            !ellipse_inside(cx, cy, rx, ry, cx + rx + 1, cy),
            "one past +x is outside"
        );
        assert!(
            !ellipse_inside(cx, cy, rx, ry, cx, cy + ry + 1),
            "one past +y is outside"
        );
        assert!(
            !ellipse_inside(cx, cy, rx, ry, cx + rx, cy + ry),
            "the bbox corner is outside"
        );
    }

    #[test]
    fn int_ellipse_raster_is_contiguous_inside_and_symmetric() {
        let (cx, cy, rx, ry) = (16, 16, 10, 6);
        let mut buf = Vec::new();
        rasterize_ellipse_int(cx, cy, rx, ry, 40, 40, &mut buf);
        assert!(!buf.is_empty());
        for s in &buf {
            assert!(s.x1 <= s.x2);
            // every emitted pixel is inside, and the run is maximal (pixel left of it is outside).
            for x in s.x1..=s.x2 {
                assert!(
                    ellipse_inside(cx, cy, rx, ry, x, s.y),
                    "({x},{}) not inside",
                    s.y
                );
            }
            if s.x1 > 0 {
                assert!(!ellipse_inside(cx, cy, rx, ry, s.x1 - 1, s.y));
            }
            // each row's covered span is symmetric about the centre column.
            assert_eq!(s.x1 + s.x2, 2 * cx, "row {} span not centred on cx", s.y);
        }
        // vertical symmetry: row cy-k and cy+k cover identical spans.
        let span = |y: i32| buf.iter().find(|s| s.y == y).map(|s| (s.x1, s.x2));
        for k in 1..=ry {
            assert_eq!(
                span(cy - k),
                span(cy + k),
                "rows {} / {} differ",
                cy - k,
                cy + k
            );
        }
    }

    #[test]
    fn int_ellipse_raster_is_deterministic() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        rasterize_ellipse_int(12, 9, 7, 5, 30, 30, &mut a);
        rasterize_ellipse_int(12, 9, 7, 5, 30, 30, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn int_rect_raster_clamps_and_spans_corner_order_invariant() {
        // a rect fully inside the image: one full-width span per row, inclusive corners.
        let mut buf = Vec::new();
        rasterize_rectangle_int(4, 5, 9, 8, 20, 20, &mut buf);
        assert_eq!(buf.len(), 4, "rows 5..=8");
        for (i, s) in buf.iter().enumerate() {
            assert_eq!((s.y, s.x1, s.x2, s.alpha), (5 + i as i32, 4, 9, 0xffff));
        }
        // corner order doesn't matter (min/max normalised).
        let mut swapped = Vec::new();
        rasterize_rectangle_int(9, 8, 4, 5, 20, 20, &mut swapped);
        assert_eq!(buf, swapped);
        // a rect straddling the edges is cropped to the image, never out of bounds.
        let mut clipped = Vec::new();
        rasterize_rectangle_int(-3, -2, 5, 100, 10, 10, &mut clipped);
        assert_eq!(clipped.len(), 10, "rows 0..=9 (y2 clamped to h-1)");
        for s in &clipped {
            assert_eq!((s.x1, s.x2), (0, 5), "x clamped to [0, 5]");
            assert!(s.y >= 0 && s.y < 10);
        }
    }

    #[test]
    fn int_ellipse_roughly_matches_f64_reference() {
        // the integer implicit test and fogleman's f64 scanline raster cover nearly the same pixels
        // (sub-pixel edges differ, as for triangles); assert the covered-pixel counts agree to ≤ 8%.
        let (cx, cy, rx, ry) = (24, 24, 14, 9);
        let mut iref = Vec::new();
        rasterize_ellipse(cx, cy, rx, ry, 50, 50, &mut iref);
        let mut iint = Vec::new();
        rasterize_ellipse_int(cx, cy, rx, ry, 50, 50, &mut iint);
        let area = |b: &[Scanline]| b.iter().map(|s| (s.x2 - s.x1 + 1) as i64).sum::<i64>();
        let (fa, ia) = (area(&iref), area(&iint));
        let diff = (fa - ia).abs() as f64 / fa as f64;
        assert!(
            diff <= 0.08,
            "int ellipse area {ia} vs f64 {fa} differ {:.1}%",
            diff * 100.0
        );
    }
}
