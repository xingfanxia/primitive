//! CORE-3b.3 — on-device search loop (`evolve` + `commit`) for **ellipse** & **rectangle**.
//!
//! Generalizes the triangle GPU-3 loop in [`crate::search`] to the new shapes. Each is a GPU-*native*
//! optimizer (energy-targeted restart + Rechenberg 1/5 self-adaptive hill-climb), **not** a bit-exact
//! replay of the CPU search — the gate is PSNR-within-tolerance vs the CPU run (`gpu3_*_optimize`
//! tests), exactly as for triangles. Shapes are scored with the parity-exact `score_one_ellipse`/
//! `score_one_rect` (CORE-3b.2) and committed over the same bbox+inside coverage. All params clamp to
//! the canvas (radii to `[1, w-1]`), keeping the i32 ellipse test overflow-safe (§6.6) and the search
//! in fogleman's domain.

use cubecl::prelude::*;

use crate::kernels::{clamp_0_255, clampi, composite, rand_below, rand_u32};
use crate::score_shapes::{inside_ellipse, score_one_ellipse, score_one_rect};
use crate::search::residual;

/// Best-of-8 energy-targeted anchor: returns the highest-residual of 8 random pixels (the restart
/// bias toward error-heavy regions). Advances `ctr` by 16 (2 draws × 8 samples).
#[cube]
fn anchor(
    target: &Array<i32>,
    current: &Array<i32>,
    s: u32,
    ctr: &mut u32,
    width: i32,
    height: i32,
    ax: &mut i32,
    ay: &mut i32,
) {
    let mut x0 = rand_below(s, *ctr, width as u32) as i32;
    *ctr += 1;
    let mut y0 = rand_below(s, *ctr, height as u32) as i32;
    *ctr += 1;
    let mut e_best = residual(target, current, x0, y0, width);
    for _ in 0..7 {
        let qx = rand_below(s, *ctr, width as u32) as i32;
        *ctr += 1;
        let qy = rand_below(s, *ctr, height as u32) as i32;
        *ctr += 1;
        let qe = residual(target, current, qx, qy, width);
        if qe > e_best {
            e_best = qe;
            x0 = qx;
            y0 = qy;
        }
    }
    *ax = x0;
    *ay = y0;
}

/// CORE-3b.3: one independent ellipse hill-climb per worker. Energy-anchored centre + small random
/// radii, then `age` Rechenberg-adaptive perturbations of one of `{cx, cy, rx, ry}` (clamped to the
/// canvas; radii ≥ 1), keeping a change only if it lowers the delta-SSE. Writes best delta + best
/// `[cx, cy, rx, ry]`; a following `commit_ellipse` composites the step winner.
#[cube(launch_unchecked)]
pub fn evolve_ellipse(
    target: &Array<i32>,
    current: &Array<i32>,
    seed: u32,
    step: u32,
    alpha: i32,
    width: i32,
    height: i32,
    age: u32,
    best_delta: &mut Array<i32>,
    best_ell: &mut Array<i32>,
) {
    let w = ABSOLUTE_POS;
    if w < best_delta.len() {
        let s = rand_u32(seed + step, w as u32);
        let mut ctr = 1u32;
        let mut cx = 0i32;
        let mut cy = 0i32;
        anchor(
            target, current, s, &mut ctr, width, height, &mut cx, &mut cy,
        );
        let mut rx = rand_below(s, ctr, 16) as i32 + 1;
        ctr += 1;
        let mut ry = rand_below(s, ctr, 16) as i32 + 1;
        ctr += 1;

        let mut best = score_one_ellipse(target, current, cx, cy, rx, ry, alpha, width, height);
        let mut span = 10u32;
        for _it in 0..age {
            let v = rand_below(s, ctr, 4);
            ctr += 1;
            let d = rand_below(s, ctr, 2 * span + 1) as i32 - span as i32;
            ctr += 1;
            let mut ncx = cx;
            let mut ncy = cy;
            let mut nrx = rx;
            let mut nry = ry;
            if v == 0 {
                ncx = clampi(ncx + d, 0, width - 1);
            } else if v == 1 {
                ncy = clampi(ncy + d, 0, height - 1);
            } else if v == 2 {
                nrx = clampi(nrx + d, 1, width - 1);
            } else {
                nry = clampi(nry + d, 1, height - 1);
            }
            let dl = score_one_ellipse(target, current, ncx, ncy, nrx, nry, alpha, width, height);
            if dl < best {
                best = dl;
                cx = ncx;
                cy = ncy;
                rx = nrx;
                ry = nry;
                span += 4u32;
                if span > 20u32 {
                    span = 20u32;
                }
            } else if span > 2u32 {
                span -= 1u32;
            }
        }

        best_delta[w] = best;
        let b = w * 4;
        best_ell[b] = cx;
        best_ell[b + 1] = cy;
        best_ell[b + 2] = rx;
        best_ell[b + 3] = ry;
    }
}

/// CORE-3b.3: one independent rectangle hill-climb per worker. Energy-anchored first corner + a small
/// random opposite corner, then `age` Rechenberg-adaptive perturbations of one of the four corner
/// coordinates (clamped to the canvas). Writes best delta + best `[x1, y1, x2, y2]`.
#[cube(launch_unchecked)]
pub fn evolve_rect(
    target: &Array<i32>,
    current: &Array<i32>,
    seed: u32,
    step: u32,
    alpha: i32,
    width: i32,
    height: i32,
    age: u32,
    best_delta: &mut Array<i32>,
    best_rect: &mut Array<i32>,
) {
    let w = ABSOLUTE_POS;
    if w < best_delta.len() {
        let s = rand_u32(seed + step, w as u32);
        let mut ctr = 1u32;
        let mut x1 = 0i32;
        let mut y1 = 0i32;
        anchor(
            target, current, s, &mut ctr, width, height, &mut x1, &mut y1,
        );
        let mut x2 = clampi(x1 + rand_below(s, ctr, 16) as i32 + 1, 0, width - 1);
        ctr += 1;
        let mut y2 = clampi(y1 + rand_below(s, ctr, 16) as i32 + 1, 0, height - 1);
        ctr += 1;

        let mut best = score_one_rect(target, current, x1, y1, x2, y2, alpha, width, height);
        let mut span = 10u32;
        for _it in 0..age {
            let v = rand_below(s, ctr, 4);
            ctr += 1;
            let d = rand_below(s, ctr, 2 * span + 1) as i32 - span as i32;
            ctr += 1;
            let mut nx1 = x1;
            let mut ny1 = y1;
            let mut nx2 = x2;
            let mut ny2 = y2;
            if v == 0 {
                nx1 = clampi(nx1 + d, 0, width - 1);
            } else if v == 1 {
                ny1 = clampi(ny1 + d, 0, height - 1);
            } else if v == 2 {
                nx2 = clampi(nx2 + d, 0, width - 1);
            } else {
                ny2 = clampi(ny2 + d, 0, height - 1);
            }
            let dl = score_one_rect(target, current, nx1, ny1, nx2, ny2, alpha, width, height);
            if dl < best {
                best = dl;
                x1 = nx1;
                y1 = ny1;
                x2 = nx2;
                y2 = ny2;
                span += 4u32;
                if span > 20u32 {
                    span = 20u32;
                }
            } else if span > 2u32 {
                span -= 1u32;
            }
        }

        best_delta[w] = best;
        let b = w * 4;
        best_rect[b] = x1;
        best_rect[b + 1] = y1;
        best_rect[b + 2] = x2;
        best_rect[b + 3] = y2;
    }
}

/// CORE-3b.3: composite the step-winning ellipse into `current` in place (argmin over `best_delta`
/// fused in, first-min tie-break; only commits an improving winner). Closed-form color + 16-bit
/// premultiplied over-composite over the same bbox+`inside_ellipse` coverage `evolve_ellipse` scored.
#[cube(launch_unchecked)]
pub fn commit_ellipse(
    target: &Array<i32>,
    current: &mut Array<i32>,
    best_ell: &Array<i32>,
    best_delta: &Array<i32>,
    n_workers: i32,
    alpha: i32,
    width: i32,
    height: i32,
) {
    if ABSOLUTE_POS == 0 {
        let mut bd = best_delta[0];
        let mut wi = 0i32;
        for i in 1..n_workers {
            let v = best_delta[i as usize];
            if v < bd {
                bd = v;
                wi = i;
            }
        }
        if bd < 0 {
            let b = (wi * 4) as usize;
            let cx = best_ell[b];
            let cy = best_ell[b + 1];
            let rx = best_ell[b + 2];
            let ry = best_ell[b + 3];
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
                        gsum +=
                            (target[idx + 1] - current[idx + 1]) * a_coef + current[idx + 1] * 257;
                        bsum +=
                            (target[idx + 2] - current[idx + 2]) * a_coef + current[idx + 2] * 257;
                        count += 1;
                    }
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
                    for px in xmin..xmax + 1 {
                        if inside_ellipse(cx, cy, rx, ry, px, py) {
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
}

/// CORE-3b.3: composite the step-winning rectangle into `current` in place (argmin fused in; only an
/// improving winner). Color + over-composite over the sorted, clamped bbox (every bbox pixel inside).
#[cube(launch_unchecked)]
pub fn commit_rect(
    target: &Array<i32>,
    current: &mut Array<i32>,
    best_rect: &Array<i32>,
    best_delta: &Array<i32>,
    n_workers: i32,
    alpha: i32,
    width: i32,
    height: i32,
) {
    if ABSOLUTE_POS == 0 {
        let mut bd = best_delta[0];
        let mut wi = 0i32;
        for i in 1..n_workers {
            let v = best_delta[i as usize];
            if v < bd {
                bd = v;
                wi = i;
            }
        }
        if bd < 0 {
            let b = (wi * 4) as usize;
            let rx1 = best_rect[b];
            let ry1 = best_rect[b + 1];
            let rx2 = best_rect[b + 2];
            let ry2 = best_rect[b + 3];
            let xa = if rx1 < rx2 { rx1 } else { rx2 };
            let xb = if rx1 < rx2 { rx2 } else { rx1 };
            let ya = if ry1 < ry2 { ry1 } else { ry2 };
            let yb = if ry1 < ry2 { ry2 } else { ry1 };
            let a_coef = 65535 / alpha;
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
                    for px in xmin..xmax + 1 {
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
