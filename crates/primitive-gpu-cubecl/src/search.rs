//! GPU-3 on-device search kernels — the optimizer loop (`evolve` hill-climb + `commit`), split
//! out of `kernels` to keep each file within the architecture size gate. Shares the scoring +
//! raster primitives from [`crate::kernels`].

use crate::kernels::{
    clamp_0_255, clampi, composite, edge_cross, imax3, imin3, rand_below, rand_u32,
    score_one_scanline,
};
use cubecl::prelude::*;

/// Absolute RGB residual `|target-current|` at pixel `(px, py)` — the per-pixel error the
/// energy-targeted restart steers toward (fogleman's high-error-region heuristic, plan §2).
#[cube]
fn residual(target: &Array<i32>, current: &Array<i32>, px: i32, py: i32, width: i32) -> i32 {
    let idx = ((py * width + px) * 4) as usize;
    let mut s = 0i32;
    let dr = target[idx] - current[idx];
    if dr < 0 {
        s -= dr;
    } else {
        s += dr;
    }
    let dg = target[idx + 1] - current[idx + 1];
    if dg < 0 {
        s -= dg;
    } else {
        s += dg;
    }
    let db = target[idx + 2] - current[idx + 2];
    if db < 0 {
        s -= db;
    } else {
        s += db;
    }
    s
}

/// GPU-3: composite the step-winning triangle into `current` in place, on-device. Reads the
/// winning worker from `winner[0]` (the argmin result), recomputes its closed-form color over the
/// current canvas, and over-composites it — the same 16-bit premultiplied math as `draw_lines`,
/// so `current`'s SSE drops by exactly the winner's delta. Single thread (one triangle/step; not
/// the hot path). Keeps the canvas resident across all shapes — no per-step host sync.
#[cube(launch_unchecked)]
pub fn commit(
    target: &Array<i32>,
    current: &mut Array<i32>,
    best_tri: &Array<i32>,
    best_delta: &Array<i32>,
    n_workers: i32,
    alpha: i32,
    width: i32,
    height: i32,
) {
    if ABSOLUTE_POS == 0 {
        // Argmin fused in (first-min tie-break) — saves a kernel launch + sync per step.
        let mut bd = best_delta[0];
        let mut wi = 0i32;
        for i in 1..n_workers {
            let v = best_delta[i as usize];
            if v < bd {
                bd = v;
                wi = i;
            }
        }
        // Only commit shapes that actually reduce the error; a non-improving winner would raise
        // the SSE, so skip it (matches "never add a worse shape").
        if bd < 0 {
            let b = (wi * 6) as usize;
            let t0 = best_tri[b];
            let t1 = best_tri[b + 1];
            let t2 = best_tri[b + 2];
            let t3 = best_tri[b + 3];
            let t4 = best_tri[b + 4];
            let t5 = best_tri[b + 5];
            let a_coef = 65535 / alpha;
            let ymin = clampi(imin3(t1, t3, t5), 0, height - 1);
            let ymax = clampi(imax3(t1, t3, t5), 0, height - 1);

            // Closed-form color over the same scanline coverage `evolve` scored.
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
            if count > 0 {
                let r = clamp_0_255((rsum / count) >> 8);
                let g = clamp_0_255((gsum / count) >> 8);
                let bb = clamp_0_255((bsum / count) >> 8);
                let av = alpha as u32;
                let sr = ((r as u32) | ((r as u32) << 8)) * av / 255;
                let sg = ((g as u32) | ((g as u32) << 8)) * av / 255;
                let sb = ((bb as u32) | ((bb as u32) << 8)) * av / 255;
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
                        current[idx] = composite(current[idx] as u32, sr, aa) as i32;
                        current[idx + 1] = composite(current[idx + 1] as u32, sg, aa) as i32;
                        current[idx + 2] = composite(current[idx + 2] as u32, sb, aa) as i32;
                        current[idx + 3] = composite(current[idx + 3] as u32, sa, aa) as i32;
                    }
                }
            }
        }
    }
}

/// GPU-3: one independent hill-climb per worker, fully on-device.
///
/// Worker `w` seeds its own Philox stream from `(seed, w)`, generates a random triangle, then for
/// `age` iterations perturbs a random vertex and keeps the change only if it lowers the delta-SSE.
/// Writes its best delta + best triangle. A following `argmin` over `best_delta` selects the
/// step winner across workers (the random-restart + hill-climb structure of fogleman, parallelized
/// over independent restarts). `step` rotates the per-step seed so successive shapes differ.
#[cube(launch_unchecked)]
pub fn evolve(
    target: &Array<i32>,
    current: &Array<i32>,
    seed: u32,
    step: u32,
    alpha: i32,
    width: i32,
    height: i32,
    age: u32,
    best_delta: &mut Array<i32>,
    best_tri: &mut Array<i32>,
) {
    let w = ABSOLUTE_POS;
    if w < best_delta.len() {
        // Per-worker independent stream: hash (seed+step, worker id) through Philox itself, so
        // each (step, worker) gets a well-mixed distinct key with no overflow-prone multiply.
        let s = rand_u32(seed + step, w as u32);
        let mut ctr = 1u32;

        // Energy-targeted anchor: the highest-residual of 4 random pixels (cheap restart bias
        // toward error-heavy regions — fogleman's energy-map trick, sampled).
        let mut x0 = rand_below(s, ctr, width as u32) as i32;
        ctr += 1;
        let mut y0 = rand_below(s, ctr, height as u32) as i32;
        ctr += 1;
        let mut e_best = residual(target, current, x0, y0, width);
        for _ in 0..7 {
            let qx = rand_below(s, ctr, width as u32) as i32;
            ctr += 1;
            let qy = rand_below(s, ctr, height as u32) as i32;
            ctr += 1;
            let qe = residual(target, current, qx, qy, width);
            if qe > e_best {
                e_best = qe;
                x0 = qx;
                y0 = qy;
            }
        }
        let mut x1 = x0 + rand_below(s, ctr, 31) as i32 - 15;
        ctr += 1;
        let mut y1 = y0 + rand_below(s, ctr, 31) as i32 - 15;
        ctr += 1;
        let mut x2 = x0 + rand_below(s, ctr, 31) as i32 - 15;
        ctr += 1;
        let mut y2 = y0 + rand_below(s, ctr, 31) as i32 - 15;
        ctr += 1;

        let mut best = score_one_scanline(
            target, current, x0, y0, x1, y1, x2, y2, alpha, width, height,
        );

        for it in 0..age {
            // Annealed perturbation: the step span shrinks from ~15 (explore) to 2 (refine) over
            // the climb, so each evaluation is spent more efficiently — fewer evals reach the same
            // quality (the eval-efficiency the parallel search needs to clear the throughput gate).
            let span = 2u32 + (13u32 * (age - 1 - it)) / age;
            let v = rand_below(s, ctr, 3);
            ctr += 1;
            let dx = rand_below(s, ctr, 2 * span + 1) as i32 - span as i32;
            ctr += 1;
            let dy = rand_below(s, ctr, 2 * span + 1) as i32 - span as i32;
            ctr += 1;

            let mut cx0 = x0;
            let mut cy0 = y0;
            let mut cx1 = x1;
            let mut cy1 = y1;
            let mut cx2 = x2;
            let mut cy2 = y2;
            if v == 0 {
                cx0 += dx;
                cy0 += dy;
            } else if v == 1 {
                cx1 += dx;
                cy1 += dy;
            } else {
                cx2 += dx;
                cy2 += dy;
            }

            let d = score_one_scanline(
                target, current, cx0, cy0, cx1, cy1, cx2, cy2, alpha, width, height,
            );
            if d < best {
                best = d;
                x0 = cx0;
                y0 = cy0;
                x1 = cx1;
                y1 = cy1;
                x2 = cx2;
                y2 = cy2;
            }
        }

        best_delta[w] = best;
        let b = w * 6;
        best_tri[b] = x0;
        best_tri[b + 1] = y0;
        best_tri[b + 2] = x1;
        best_tri[b + 3] = y1;
        best_tri[b + 4] = x2;
        best_tri[b + 5] = y2;
    }
}
