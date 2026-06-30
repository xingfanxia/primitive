//! Shared test helpers: deterministic target loading + SSIM.
//!
//! Kept out of the pure `primitive-core` library (which does no image IO) — this lives in
//! test code only.

use image::{GenericImageView, RgbaImage};
use primitive_core::Canvas;

/// Resize `path` to fit within `size × size` preserving aspect (fogleman's Thumbnail), with
/// a fixed Triangle filter, and return it as an opaque RGBA [`Canvas`]. Fully deterministic.
pub fn resize_target_canvas(path: &str, size: u32) -> Canvas {
    let img = image::open(path).expect("open target image");
    let (w, h) = img.dimensions();
    let aspect = w as f64 / h as f64;
    let (nw, nh) = if w <= size && h <= size {
        (w, h)
    } else if aspect >= 1.0 {
        (size, (size as f64 / aspect).round() as u32)
    } else {
        ((size as f64 * aspect).round() as u32, size)
    };
    let resized = image::imageops::resize(
        &img.to_rgba8(),
        nw,
        nh,
        image::imageops::FilterType::Triangle,
    );
    let mut canvas = Canvas::new(nw as usize, nh as usize);
    canvas.pix.copy_from_slice(resized.as_raw());
    for px in canvas.pix.chunks_exact_mut(4) {
        px[3] = 255;
    }
    canvas
}

/// Render an RGBA [`Canvas`] to an `image::RgbaImage` (for saving / SSIM).
pub fn canvas_to_image(c: &Canvas) -> RgbaImage {
    RgbaImage::from_raw(c.w as u32, c.h as u32, c.pix.clone()).expect("valid rgba buffer")
}

fn to_luma(img: &RgbaImage) -> (usize, usize, Vec<f64>) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let luma = img
        .pixels()
        .map(|p| 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64)
        .collect();
    (w, h, luma)
}

/// Mean SSIM over 8×8 non-overlapping luma windows. Identical images ⇒ exactly 1.0.
pub fn ssim(a: &RgbaImage, b: &RgbaImage) -> f64 {
    assert_eq!(
        a.dimensions(),
        b.dimensions(),
        "SSIM needs equal dimensions"
    );
    let (w, h, la) = to_luma(a);
    let (_, _, lb) = to_luma(b);
    const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
    const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0);
    const WIN: usize = 8;

    let mut total = 0.0;
    let mut windows = 0;
    let mut y0 = 0;
    while y0 < h {
        let y1 = (y0 + WIN).min(h);
        let mut x0 = 0;
        while x0 < w {
            let x1 = (x0 + WIN).min(w);
            let n = ((y1 - y0) * (x1 - x0)) as f64;
            let (mut sa, mut sb) = (0.0, 0.0);
            for y in y0..y1 {
                for x in x0..x1 {
                    sa += la[y * w + x];
                    sb += lb[y * w + x];
                }
            }
            let (ma, mb) = (sa / n, sb / n);
            let (mut va, mut vb, mut cov) = (0.0, 0.0, 0.0);
            for y in y0..y1 {
                for x in x0..x1 {
                    let da = la[y * w + x] - ma;
                    let db = lb[y * w + x] - mb;
                    va += da * da;
                    vb += db * db;
                    cov += da * db;
                }
            }
            va /= n;
            vb /= n;
            cov /= n;
            let s = ((2.0 * ma * mb + C1) * (2.0 * cov + C2))
                / ((ma * ma + mb * mb + C1) * (va + vb + C2));
            total += s;
            windows += 1;
            x0 += WIN;
        }
        y0 += WIN;
    }
    total / windows as f64
}
