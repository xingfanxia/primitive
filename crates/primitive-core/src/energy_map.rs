//! Residual energy map → restart sampling PDF.
//!
//! Plan §4's highest-value, lowest-effort upgrade: bias where new shapes are *tried* toward
//! pixels that still carry error. We build a per-pixel residual (SSE vs target), turn it into
//! a cumulative distribution, and draw pixels in proportion to their residual — so candidate
//! restarts are spent where error actually lives (ProHC, *Biomimetics* 2023).
//!
//! Pure and RNG-injected. Wired into the search at GPU-3; validated standalone here.

use crate::canvas::Canvas;
use crate::rng::Rng;

/// Per-pixel residual SSE (RGB) between `target` and `current`, row-major `w*h`.
pub fn residual_map(target: &Canvas, current: &Canvas) -> Vec<u32> {
    debug_assert_eq!(
        target.pix.len(),
        current.pix.len(),
        "residual_map requires equal-size canvases"
    );
    target
        .pix
        .chunks_exact(4)
        .zip(current.pix.chunks_exact(4))
        .map(|(t, c)| {
            let dr = t[0] as i32 - c[0] as i32;
            let dg = t[1] as i32 - c[1] as i32;
            let db = t[2] as i32 - c[2] as i32;
            (dr * dr + dg * dg + db * db) as u32
        })
        .collect()
}

/// Inclusive-prefix-sum CDF over a residual map (the last entry is the total weight).
pub fn build_cdf(residual: &[u32]) -> Vec<u64> {
    let mut cdf = Vec::with_capacity(residual.len());
    let mut acc: u64 = 0;
    for &r in residual {
        acc += r as u64;
        cdf.push(acc);
    }
    cdf
}

/// Draw a pixel index in proportion to residual weight via binary search on the CDF.
///
/// Returns `None` only when total weight is zero (a perfect reconstruction) — callers should
/// fall back to uniform sampling in that case.
pub fn sample_pixel(cdf: &[u64], rng: &mut Rng) -> Option<usize> {
    let total = *cdf.last()?;
    if total == 0 {
        return None;
    }
    // Target a weight in [0, total); find the first CDF entry strictly greater than it.
    let target = (rng.f64() * total as f64) as u64;
    let idx = cdf.partition_point(|&c| c <= target);
    Some(idx.min(cdf.len() - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_zero_for_identical() {
        let a = Canvas::filled(4, 4, 10, 20, 30);
        let r = residual_map(&a, &a);
        assert!(r.iter().all(|&v| v == 0));
    }

    #[test]
    fn sampling_favors_hot_pixel() {
        // One pixel carries all the error; sampling should land there almost always.
        let mut target = Canvas::filled(8, 8, 0, 0, 0);
        let current = Canvas::filled(8, 8, 0, 0, 0);
        let hot = target.pix_offset(5, 3);
        target.pix[hot] = 255;
        target.pix[hot + 1] = 255;
        target.pix[hot + 2] = 255;

        let cdf = build_cdf(&residual_map(&target, &current));
        let mut rng = Rng::new(7);
        let hot_index = 3 * 8 + 5;
        let mut hits = 0;
        for _ in 0..1000 {
            if sample_pixel(&cdf, &mut rng) == Some(hot_index) {
                hits += 1;
            }
        }
        assert_eq!(hits, 1000, "all weight is on the hot pixel");
    }

    #[test]
    fn zero_total_returns_none() {
        let a = Canvas::filled(4, 4, 1, 2, 3);
        let cdf = build_cdf(&residual_map(&a, &a));
        let mut rng = Rng::new(1);
        assert_eq!(sample_pixel(&cdf, &mut rng), None);
    }
}
