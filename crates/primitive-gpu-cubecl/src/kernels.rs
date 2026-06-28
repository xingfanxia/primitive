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
pub(crate) fn imin3(a: i32, b: i32, c: i32) -> i32 {
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
pub(crate) fn imax3(a: i32, b: i32, c: i32) -> i32 {
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
pub(crate) fn clampi(x: i32, lo: i32, hi: i32) -> i32 {
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

// ── Counter-based RNG (mirrors primitive_core::philox, pure u32 → WGSL-portable) ─────────────

/// One `u32` Philox draw for counter `ctr` under `seed` — bit-identical to
/// `primitive_core::rand_u32`.
#[cube]
pub fn rand_u32(seed: u32, ctr: u32) -> u32 {
    let mut x0 = ctr;
    let mut x1 = 0u32;
    let mut k = seed;
    for _ in 0..10u32 {
        let a_lo = 0xD256D193u32 & 0xffff;
        let a_hi = 0xD256D193u32 >> 16;
        let b_lo = x0 & 0xffff;
        let b_hi = x0 >> 16;
        let ll = a_lo * b_lo;
        let lh = a_lo * b_hi;
        let hl = a_hi * b_lo;
        let hh = a_hi * b_hi;
        let mid = (ll >> 16) + (lh & 0xffff) + (hl & 0xffff);
        let lo = (ll & 0xffff) | (mid << 16);
        let hi = hh + (lh >> 16) + (hl >> 16) + (mid >> 16);
        x0 = hi ^ k ^ x1;
        x1 = lo;
        k += 0x9E3779B9u32;
    }
    x0
}

/// Uniform integer in `[0, n)` — bit-identical to `primitive_core::rand_below`.
#[cube]
pub fn rand_below(seed: u32, ctr: u32, n: u32) -> u32 {
    let r = rand_u32(seed, ctr);
    let a_lo = r & 0xffff;
    let a_hi = r >> 16;
    let b_lo = n & 0xffff;
    let b_hi = n >> 16;
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;
    let mid = (ll >> 16) + (lh & 0xffff) + (hl & 0xffff);
    hh + (lh >> 16) + (hl >> 16) + (mid >> 16)
}

/// Parity probe: `out[i] = rand_below(seed, i, range)` — proves the kernel RNG == CPU RNG.
#[cube(launch_unchecked)]
pub fn philox_fill(seed: u32, range: u32, out: &mut Array<u32>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = rand_below(seed, i as u32, range);
    }
}

/// Integer x where edge `(ax,ay)-(bx,by)` crosses row `py`, or `none` if the edge is horizontal or
/// doesn't span the row. Returning a sentinel (vs a tuple) lets `imin3`/`imax3` combine the three
/// edges into the row span without bool plumbing.
#[cube]
pub(crate) fn edge_cross(ax: i32, ay: i32, bx: i32, by: i32, py: i32, none: i32) -> i32 {
    let mut r = none;
    if ay != by && ((ay <= py && by >= py) || (by <= py && ay >= py)) {
        r = ax + (bx - ax) * (py - ay) / (by - ay);
    }
    r
}

/// Faster integer delta-SSE of compositing one triangle — **scanline** raster (analytic per-row
/// span via edge crossings, O(height)) instead of a full bounding-box inside-test. Used by the
/// GPU-3 search loop (`evolve` + `commit`), which is PSNR-compared to the CPU, not byte-parity —
/// so this GPU-only raster need only be self-consistent between scoring and committing. The
/// edge-function [`score_one`] stays the parity-exact path for the GPU-2 batch scorer.
#[cube]
pub fn score_one_scanline(
    target: &Array<i32>,
    current: &Array<i32>,
    t0: i32,
    t1: i32,
    t2: i32,
    t3: i32,
    t4: i32,
    t5: i32,
    alpha: i32,
    width: i32,
    height: i32,
) -> i32 {
    let a_coef = 65535 / alpha;
    let ymin = clampi(imin3(t1, t3, t5), 0, height - 1);
    let ymax = clampi(imax3(t1, t3, t5), 0, height - 1);

    let mut rsum = 0i32;
    let mut gsum = 0i32;
    let mut bsum = 0i32;
    let mut count = 0i32;
    for py in ymin..ymax + 1 {
        let lo = imin3(
            edge_cross(t0, t1, t2, t3, py, width),
            edge_cross(t2, t3, t4, t5, py, width),
            edge_cross(t4, t5, t0, t1, py, width),
        );
        let hi = imax3(
            edge_cross(t0, t1, t2, t3, py, -1),
            edge_cross(t2, t3, t4, t5, py, -1),
            edge_cross(t4, t5, t0, t1, py, -1),
        );
        let xlo = clampi(lo, 0, width - 1);
        let xhi = clampi(hi, 0, width - 1);
        for px in xlo..xhi + 1 {
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
            let lo = imin3(
                edge_cross(t0, t1, t2, t3, py, width),
                edge_cross(t2, t3, t4, t5, py, width),
                edge_cross(t4, t5, t0, t1, py, width),
            );
            let hi = imax3(
                edge_cross(t0, t1, t2, t3, py, -1),
                edge_cross(t2, t3, t4, t5, py, -1),
                edge_cross(t4, t5, t0, t1, py, -1),
            );
            let xlo = clampi(lo, 0, width - 1);
            let xhi = clampi(hi, 0, width - 1);
            for px in xlo..xhi + 1 {
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

/// Integer delta-SSE of compositing one triangle (vertices `t0..t5`, fill `alpha`) over `current`
/// toward `target`. In-kernel integer edge-function raster + closed-form color + delta-SSE —
/// mirrors `rasterize_triangle_int` + `candidate_color_and_delta` exactly. Shared by the GPU-2
/// batch scorer and the GPU-3 hill-climb.
#[cube]
pub fn score_one(
    target: &Array<i32>,
    current: &Array<i32>,
    t0: i32,
    t1: i32,
    t2: i32,
    t3: i32,
    t4: i32,
    t5: i32,
    alpha: i32,
    width: i32,
    height: i32,
) -> i32 {
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
    delta
}

/// GPU-2: fully on-device batch scorer. `tris` is 6 ints per candidate; one thread per candidate.
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
        out[c] = score_one(
            target,
            current,
            tris[base],
            tris[base + 1],
            tris[base + 2],
            tris[base + 3],
            tris[base + 4],
            tris[base + 5],
            alphas[c],
            width,
            height,
        );
    }
}
