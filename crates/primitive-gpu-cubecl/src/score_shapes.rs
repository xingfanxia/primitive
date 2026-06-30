//! CORE-3b.2 — on-device score kernels for **ellipse** & **rectangle** candidates.
//!
//! Mirror the triangle path in [`crate::kernels`] (`score_one` / `score_triangles`): in-kernel
//! integer raster + closed-form color + integer delta-SSE, held to **exact** parity against the CPU
//! oracle (`primitive_core::rasterize_ellipse_int` / `rasterize_rectangle_int` +
//! `candidate_color_and_delta`). Only the bounding box + inside test differ per shape; the
//! color/delta block is the same proven arithmetic as `score_one`, so the parity that holds for
//! triangles carries to these by construction.

use cubecl::prelude::*;

use crate::kernels::{channel_delta, clamp_0_255, clampi};

/// Integer point-in-ellipse test — mirrors `primitive_core::raster_int::ellipse_inside`
/// (`ry²·dx² + rx²·dy² ≤ rx²·ry²`, no `sqrt`). i32; overflow-free within the ≤128-px canvas bound
/// (the degree-4 product peaks at `2·128⁴ ≈ 5.4e8`, ~4× under i32::MAX — §6.6).
#[cube]
fn inside_ellipse(cx: i32, cy: i32, rx: i32, ry: i32, px: i32, py: i32) -> bool {
    let dx = px - cx;
    let dy = py - cy;
    ry * ry * dx * dx + rx * rx * dy * dy <= rx * rx * ry * ry
}

/// Integer delta-SSE of compositing one ellipse (centre `cx,cy`, radii `rx,ry`, fill `alpha`) over
/// `current` toward `target`. Mirrors `score_one` with the ellipse bbox + `inside_ellipse`.
#[cube]
pub fn score_one_ellipse(
    target: &Array<i32>,
    current: &Array<i32>,
    cx: i32,
    cy: i32,
    rx: i32,
    ry: i32,
    alpha: i32,
    width: i32,
    height: i32,
) -> i32 {
    let a_coef = 65535 / alpha;
    let xmin = clampi(cx - rx, 0, width - 1);
    let xmax = clampi(cx + rx, 0, width - 1);
    let ymin = clampi(cy - ry, 0, height - 1);
    let ymax = clampi(cy + ry, 0, height - 1);

    let mut rsum = 0i32;
    let mut gsum = 0i32;
    let mut bsum = 0i32;
    let mut count = 0i32;
    for py in ymin..ymax + 1 {
        for px in xmin..xmax + 1 {
            if inside_ellipse(cx, cy, rx, ry, px, py) {
                let idx = ((py * width + px) * 4) as usize;
                rsum += (target[idx] - current[idx]) * a_coef + current[idx] * 257;
                gsum += (target[idx + 1] - current[idx + 1]) * a_coef + current[idx + 1] * 257;
                bsum += (target[idx + 2] - current[idx + 2]) * a_coef + current[idx + 2] * 257;
                count += 1;
            }
        }
    }

    let mut delta = 0i32;
    if count > 0 {
        let r = clamp_0_255((rsum / count) >> 8);
        let g = clamp_0_255((gsum / count) >> 8);
        let b = clamp_0_255((bsum / count) >> 8);
        let av = alpha as u32;
        let sr = ((r as u32) | ((r as u32) << 8)) * av / 255;
        let sg = ((g as u32) | ((g as u32) << 8)) * av / 255;
        let sb = ((b as u32) | ((b as u32) << 8)) * av / 255;
        let sa = av | (av << 8);
        let aa = (65535 - sa * 65535 / 65535) * 257;
        for py in ymin..ymax + 1 {
            for px in xmin..xmax + 1 {
                if inside_ellipse(cx, cy, rx, ry, px, py) {
                    let idx = ((py * width + px) * 4) as usize;
                    delta += channel_delta(target[idx], current[idx], sr, aa);
                    delta += channel_delta(target[idx + 1], current[idx + 1], sg, aa);
                    delta += channel_delta(target[idx + 2], current[idx + 2], sb, aa);
                    delta += channel_delta(target[idx + 3], current[idx + 3], sa, aa);
                }
            }
        }
    }
    delta
}

/// Integer delta-SSE of compositing one axis-aligned rectangle (opposite corners `(x1,y1)`–`(x2,y2)`,
/// fill `alpha`). Mirrors `score_one` with the rect bbox; **every pixel in the clamped bbox is inside
/// the rectangle**, so there is no per-pixel inside test. Matches `rasterize_rectangle_int` for an
/// in-bounds rect (the production domain — `Rectangle` clamps its corners).
#[cube]
pub fn score_one_rect(
    target: &Array<i32>,
    current: &Array<i32>,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    alpha: i32,
    width: i32,
    height: i32,
) -> i32 {
    let a_coef = 65535 / alpha;
    let xa = if x1 < x2 { x1 } else { x2 };
    let xb = if x1 < x2 { x2 } else { x1 };
    let ya = if y1 < y2 { y1 } else { y2 };
    let yb = if y1 < y2 { y2 } else { y1 };
    let xmin = clampi(xa, 0, width - 1);
    let xmax = clampi(xb, 0, width - 1);
    let ymin = clampi(ya, 0, height - 1);
    let ymax = clampi(yb, 0, height - 1);

    let mut rsum = 0i32;
    let mut gsum = 0i32;
    let mut bsum = 0i32;
    let mut count = 0i32;
    for py in ymin..ymax + 1 {
        for px in xmin..xmax + 1 {
            let idx = ((py * width + px) * 4) as usize;
            rsum += (target[idx] - current[idx]) * a_coef + current[idx] * 257;
            gsum += (target[idx + 1] - current[idx + 1]) * a_coef + current[idx + 1] * 257;
            bsum += (target[idx + 2] - current[idx + 2]) * a_coef + current[idx + 2] * 257;
            count += 1;
        }
    }

    let mut delta = 0i32;
    if count > 0 {
        let r = clamp_0_255((rsum / count) >> 8);
        let g = clamp_0_255((gsum / count) >> 8);
        let b = clamp_0_255((bsum / count) >> 8);
        let av = alpha as u32;
        let sr = ((r as u32) | ((r as u32) << 8)) * av / 255;
        let sg = ((g as u32) | ((g as u32) << 8)) * av / 255;
        let sb = ((b as u32) | ((b as u32) << 8)) * av / 255;
        let sa = av | (av << 8);
        let aa = (65535 - sa * 65535 / 65535) * 257;
        for py in ymin..ymax + 1 {
            for px in xmin..xmax + 1 {
                let idx = ((py * width + px) * 4) as usize;
                delta += channel_delta(target[idx], current[idx], sr, aa);
                delta += channel_delta(target[idx + 1], current[idx + 1], sg, aa);
                delta += channel_delta(target[idx + 2], current[idx + 2], sb, aa);
                delta += channel_delta(target[idx + 3], current[idx + 3], sa, aa);
            }
        }
    }
    delta
}

/// GPU-2 batch scorer for ellipses. `ell` is 4 ints per candidate `[cx, cy, rx, ry]`; one thread
/// per candidate. Output `out[c]` is the integer delta-SSE.
#[cube(launch_unchecked)]
pub fn score_ellipses(
    target: &Array<i32>,
    current: &Array<i32>,
    ell: &Array<i32>,
    alphas: &Array<i32>,
    width: i32,
    height: i32,
    out: &mut Array<i32>,
) {
    let c = ABSOLUTE_POS;
    if c < out.len() {
        let base = c * 4;
        out[c] = score_one_ellipse(
            target,
            current,
            ell[base],
            ell[base + 1],
            ell[base + 2],
            ell[base + 3],
            alphas[c],
            width,
            height,
        );
    }
}

/// GPU-2 batch scorer for rectangles. `rects` is 4 ints per candidate `[x1, y1, x2, y2]`; one thread
/// per candidate. Output `out[c]` is the integer delta-SSE.
#[cube(launch_unchecked)]
pub fn score_rectangles(
    target: &Array<i32>,
    current: &Array<i32>,
    rects: &Array<i32>,
    alphas: &Array<i32>,
    width: i32,
    height: i32,
    out: &mut Array<i32>,
) {
    let c = ABSOLUTE_POS;
    if c < out.len() {
        let base = c * 4;
        out[c] = score_one_rect(
            target,
            current,
            rects[base],
            rects[base + 1],
            rects[base + 2],
            rects[base + 3],
            alphas[c],
            width,
            height,
        );
    }
}
