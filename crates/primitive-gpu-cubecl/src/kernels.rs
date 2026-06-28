//! CubeCL `#[cube]` kernels (Metal/CUDA from one source). All integer/u32 — the determinism
//! contract (plan §6.6). Each is held to exact parity against `primitive_core` on the CPU.

use cubecl::prelude::*;

/// 16-bit premultiplied over-composite of one channel — Go's `image/draw` integer math.
#[cube]
pub fn composite(before: u32, s: u32, aa: u32) -> u32 {
    ((before * aa + s * 65535) / 65535) >> 8
}

/// Signed per-channel SSE delta `(t-after)² − (t-before)²` for one composited channel.
#[cube]
pub fn channel_delta(t: i32, before: i32, s: u32, aa: u32) -> i32 {
    let after = composite(before as u32, s, aa) as i32;
    let da = t - after;
    let db = t - before;
    da * da - db * db
}

#[cube]
#[allow(clippy::manual_clamp)] // std `i32::clamp` isn't available inside a #[cube] kernel
pub fn clamp_0_255(x: i32) -> i32 {
    let mut r = x;
    if r < 0 {
        r = 0;
    }
    if r > 255 {
        r = 255;
    }
    r
}

#[cube]
fn imin3(a: i32, b: i32, c: i32) -> i32 {
    let mut m = a;
    if b < m {
        m = b;
    }
    if c < m {
        m = c;
    }
    m
}

#[cube]
fn imax3(a: i32, b: i32, c: i32) -> i32 {
    let mut m = a;
    if b > m {
        m = b;
    }
    if c > m {
        m = c;
    }
    m
}

#[cube]
#[allow(clippy::manual_clamp)]
fn clampi(x: i32, lo: i32, hi: i32) -> i32 {
    let mut r = x;
    if r < lo {
        r = lo;
    }
    if r > hi {
        r = hi;
    }
    r
}

/// Integer edge-function inside test — mirrors `primitive_core::raster::triangle_inside`.
#[cube]
fn inside(t0: i32, t1: i32, t2: i32, t3: i32, t4: i32, t5: i32, px: i32, py: i32) -> bool {
    let e0 = (t2 - t0) * (py - t1) - (t3 - t1) * (px - t0);
    let e1 = (t4 - t2) * (py - t3) - (t5 - t3) * (px - t2);
    let e2 = (t0 - t4) * (py - t5) - (t1 - t5) * (px - t4);
    let pos = e0 >= 0 && e1 >= 0 && e2 >= 0;
    let neg = e0 <= 0 && e1 <= 0 && e2 <= 0;
    pos || neg
}

/// GPU-1: score candidates from CPU-supplied coverage. `spans` is `[y,x1,x2]` triples;
/// `offsets[c]..offsets[c+1]` are candidate `c`'s spans. Mirrors `candidate_color_and_delta`.
#[cube(launch_unchecked)]
pub fn score_candidates(
    target: &Array<i32>,
    current: &Array<i32>,
    spans: &Array<i32>,
    offsets: &Array<i32>,
    alphas: &Array<i32>,
    width: i32,
    out: &mut Array<i32>,
) {
    let c = ABSOLUTE_POS;
    if c < out.len() {
        let alpha = alphas[c];
        let a_coef = 65535 / alpha;
        let s_start = offsets[c];
        let s_end = offsets[c + 1];

        let mut rsum = 0i32;
        let mut gsum = 0i32;
        let mut bsum = 0i32;
        let mut count = 0i32;
        for s in s_start..s_end {
            let su = (s * 3) as usize;
            let y = spans[su];
            let x1 = spans[su + 1];
            let x2 = spans[su + 2];
            for x in x1..x2 + 1 {
                let idx = ((y * width + x) * 4) as usize;
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
            for s in s_start..s_end {
                let su = (s * 3) as usize;
                let y = spans[su];
                let x1 = spans[su + 1];
                let x2 = spans[su + 2];
                for x in x1..x2 + 1 {
                    let idx = ((y * width + x) * 4) as usize;
                    delta += channel_delta(target[idx], current[idx], sr, aa);
                    delta += channel_delta(target[idx + 1], current[idx + 1], sg, aa);
                    delta += channel_delta(target[idx + 2], current[idx + 2], sb, aa);
                    delta += channel_delta(target[idx + 3], current[idx + 3], sa, aa);
                }
            }
        }
        out[c] = delta;
    }
}

/// GPU-2: reduce a batch of delta-SSE scores to the winning index (minimum, first-min tie-break)
/// — a single-thread on-device reduction. Matches CPU brute-force argmin. In the search loop this
/// runs once per *step* (over the step's candidates), so single-thread is not the hot path;
/// scoring (the parallel `score_*` kernels) is.
#[cube(launch_unchecked)]
pub fn argmin(deltas: &Array<i32>, n: i32, out_idx: &mut Array<i32>) {
    if ABSOLUTE_POS == 0 {
        let mut best = deltas[0];
        let mut best_i = 0i32;
        for i in 1..n {
            let v = deltas[i as usize];
            if v < best {
                best = v;
                best_i = i;
            }
        }
        out_idx[0] = best_i;
    }
}

/// GPU-2: fully on-device — rasterize each triangle (integer edge test) over its bounding box,
/// then closed-form color + composite + integer delta-SSE. `tris` is 6 ints per candidate.
/// Mirrors `rasterize_triangle_int` + `candidate_color_and_delta` exactly.
#[cube(launch_unchecked)]
pub fn score_triangles(
    target: &Array<i32>,
    current: &Array<i32>,
    tris: &Array<i32>,
    alphas: &Array<i32>,
    width: i32,
    height: i32,
    out: &mut Array<i32>,
) {
    let c = ABSOLUTE_POS;
    if c < out.len() {
        let base = c * 6;
        let t0 = tris[base];
        let t1 = tris[base + 1];
        let t2 = tris[base + 2];
        let t3 = tris[base + 3];
        let t4 = tris[base + 4];
        let t5 = tris[base + 5];
        let alpha = alphas[c];
        let a_coef = 65535 / alpha;

        let xmin = clampi(imin3(t0, t2, t4), 0, width - 1);
        let xmax = clampi(imax3(t0, t2, t4), 0, width - 1);
        let ymin = clampi(imin3(t1, t3, t5), 0, height - 1);
        let ymax = clampi(imax3(t1, t3, t5), 0, height - 1);

        let mut rsum = 0i32;
        let mut gsum = 0i32;
        let mut bsum = 0i32;
        let mut count = 0i32;
        for py in ymin..ymax + 1 {
            for px in xmin..xmax + 1 {
                if inside(t0, t1, t2, t3, t4, t5, px, py) {
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
                    if inside(t0, t1, t2, t3, t4, t5, px, py) {
                        let idx = ((py * width + px) * 4) as usize;
                        delta += channel_delta(target[idx], current[idx], sr, aa);
                        delta += channel_delta(target[idx + 1], current[idx + 1], sg, aa);
                        delta += channel_delta(target[idx + 2], current[idx + 2], sb, aa);
                        delta += channel_delta(target[idx + 3], current[idx + 3], sa, aa);
                    }
                }
            }
        }
        out[c] = delta;
    }
}
