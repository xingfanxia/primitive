//! Scanline rasterization — the reference (golden) rasterizer.
//!
//! Faithful port of fogleman's `scanline.go` + the triangle rasterizer in `triangle.go`.
//! Triangles use the dedicated edge-walking rasterizer (not the freetype path), so this is
//! pure integer/float arithmetic with no external rasterizer dependency. Every `as i32`
//! truncates toward zero, matching Go's `int(...)` conversion exactly.

/// A horizontal run of covered pixels on row `y`, inclusive `[x1, x2]`, with 16-bit coverage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scanline {
    pub y: i32,
    pub x1: i32,
    pub x2: i32,
    /// Coverage alpha in `[0, 0xffff]`. Hard-edged triangles use `0xffff`.
    pub alpha: u32,
}

/// Clamp to `[lo, hi]` with fogleman's `clampInt` semantics (lower bound wins when `lo > hi`).
/// Identical to `max(lo).min(hi)` for the `lo <= hi` inputs `crop_scanlines` always passes.
/// `pub(crate)` so the integer rasterizers in [`crate::raster_int`] share this one clamp.
#[inline]
pub(crate) fn clamp_i32(x: i32, lo: i32, hi: i32) -> i32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Clip scanlines to the image rectangle, dropping fully-outside runs — port of
/// `cropScanlines`. Mutates in place and returns the retained prefix length.
pub fn crop_scanlines(lines: &mut Vec<Scanline>, w: i32, h: i32) {
    let mut i = 0;
    for k in 0..lines.len() {
        let mut line = lines[k];
        if line.y < 0 || line.y >= h {
            continue;
        }
        if line.x1 >= w {
            continue;
        }
        if line.x2 < 0 {
            continue;
        }
        line.x1 = clamp_i32(line.x1, 0, w - 1);
        line.x2 = clamp_i32(line.x2, 0, w - 1);
        if line.x1 > line.x2 {
            continue;
        }
        lines[i] = line;
        i += 1;
    }
    lines.truncate(i);
}

/// Rasterize a triangle into scanlines (uncropped) — port of `rasterizeTriangle`.
pub fn rasterize_triangle(
    mut x1: i32,
    mut y1: i32,
    mut x2: i32,
    mut y2: i32,
    mut x3: i32,
    mut y3: i32,
    buf: &mut Vec<Scanline>,
) {
    // Sort vertices by y ascending: y1 <= y2 <= y3.
    if y1 > y3 {
        core::mem::swap(&mut x1, &mut x3);
        core::mem::swap(&mut y1, &mut y3);
    }
    if y1 > y2 {
        core::mem::swap(&mut x1, &mut x2);
        core::mem::swap(&mut y1, &mut y2);
    }
    if y2 > y3 {
        core::mem::swap(&mut x2, &mut x3);
        core::mem::swap(&mut y2, &mut y3);
    }

    if y2 == y3 {
        rasterize_triangle_bottom(x1, y1, x2, y2, x3, y3, buf);
    } else if y1 == y2 {
        rasterize_triangle_top(x1, y1, x2, y2, x3, y3, buf);
    } else {
        let x4 = x1 + ((y2 - y1) as f64 / (y3 - y1) as f64 * (x3 - x1) as f64) as i32;
        let y4 = y2;
        rasterize_triangle_bottom(x1, y1, x2, y2, x4, y4, buf);
        rasterize_triangle_top(x2, y2, x4, y4, x3, y3, buf);
    }
}

fn rasterize_triangle_bottom(
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    x3: i32,
    y3: i32,
    buf: &mut Vec<Scanline>,
) {
    let s1 = (x2 - x1) as f64 / (y2 - y1) as f64;
    let s2 = (x3 - x1) as f64 / (y3 - y1) as f64;
    let mut ax = x1 as f64;
    let mut bx = x1 as f64;
    let mut y = y1;
    while y <= y2 {
        let mut a = ax as i32;
        let mut b = bx as i32;
        ax += s1;
        bx += s2;
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        buf.push(Scanline {
            y,
            x1: a,
            x2: b,
            alpha: 0xffff,
        });
        y += 1;
    }
}

fn rasterize_triangle_top(
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    x3: i32,
    y3: i32,
    buf: &mut Vec<Scanline>,
) {
    let s1 = (x3 - x1) as f64 / (y3 - y1) as f64;
    let s2 = (x3 - x2) as f64 / (y3 - y2) as f64;
    let mut ax = x3 as f64;
    let mut bx = x3 as f64;
    let mut y = y3;
    while y > y1 {
        ax -= s1;
        bx -= s2;
        let mut a = ax as i32;
        let mut b = bx as i32;
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        buf.push(Scanline {
            y,
            x1: a,
            x2: b,
            alpha: 0xffff,
        });
        y -= 1;
    }
}

/// Rasterize an axis-aligned ellipse centred `(cx, cy)` with radii `(rx, ry)` into scanlines —
/// faithful port of fogleman's `Ellipse.Rasterize`. Each row's half-width is the analytic
/// `sqrt(ry² − dy²) · (rx/ry)`, truncated toward zero (Go's `int(...)`); x is clamped to the image
/// and the centre row is emitted once (`dy > 0` guards the mirrored row). The caller crops for the
/// final contract. Requires `ry ≥ 1` (guaranteed by the shape's random/mutate bounds).
pub fn rasterize_ellipse(
    cx: i32,
    cy: i32,
    rx: i32,
    ry: i32,
    w: i32,
    h: i32,
    buf: &mut Vec<Scanline>,
) {
    let aspect = rx as f64 / ry as f64;
    for dy in 0..ry {
        let y1 = cy - dy;
        let y2 = cy + dy;
        if (y1 < 0 || y1 >= h) && (y2 < 0 || y2 >= h) {
            continue;
        }
        // i64 like Go's `int`: ry can reach a canvas dimension, and ry² overflows i32 past ~46k px.
        let s = (((ry as i64 * ry as i64 - dy as i64 * dy as i64) as f64).sqrt() * aspect) as i32;
        let mut x1 = cx - s;
        let mut x2 = cx + s;
        if x1 < 0 {
            x1 = 0;
        }
        if x2 >= w {
            x2 = w - 1;
        }
        if y1 >= 0 && y1 < h {
            buf.push(Scanline {
                y: y1,
                x1,
                x2,
                alpha: 0xffff,
            });
        }
        if y2 >= 0 && y2 < h && dy > 0 {
            buf.push(Scanline {
                y: y2,
                x1,
                x2,
                alpha: 0xffff,
            });
        }
    }
}

/// Rasterize an axis-aligned rectangle with inclusive opposite corners `(x1, y1)`–`(x2, y2)` (any
/// winding; sorted internally to fogleman's `bounds()`) into one span per row — port of
/// fogleman's `Rectangle.Rasterize`. The caller crops to the image.
pub fn rasterize_rectangle(x1: i32, y1: i32, x2: i32, y2: i32, buf: &mut Vec<Scanline>) {
    let (xa, xb) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
    let (ya, yb) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
    for y in ya..=yb {
        buf.push(Scanline {
            y,
            x1: xa,
            x2: xb,
            alpha: 0xffff,
        });
    }
}
