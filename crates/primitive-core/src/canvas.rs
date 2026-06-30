//! RGBA pixel buffer — the in-memory image the optimizer rasterizes into.
//!
//! Mirrors Go's `image.RGBA`: row-major, 4 bytes/pixel (R,G,B,A), `pix_offset` identical
//! to `image.RGBA.PixOffset`. Pure data + indexing; no IO (decode/encode lives in adapters
//! and test harnesses).

/// An 8-bit RGBA image buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Canvas {
    pub w: usize,
    pub h: usize,
    /// `w * h * 4` bytes, row-major, [R,G,B,A] per pixel.
    pub pix: Vec<u8>,
}

impl Canvas {
    /// Allocate a transparent (all-zero) canvas.
    pub fn new(w: usize, h: usize) -> Self {
        Canvas {
            w,
            h,
            pix: vec![0; w * h * 4],
        }
    }

    /// Allocate a canvas filled with a solid opaque color (alpha forced to 255).
    pub fn filled(w: usize, h: usize, r: u8, g: u8, b: u8) -> Self {
        let mut c = Canvas::new(w, h);
        for px in c.pix.chunks_exact_mut(4) {
            px[0] = r;
            px[1] = g;
            px[2] = b;
            px[3] = 255;
        }
        c
    }

    /// Byte offset of pixel `(x, y)` — identical to `image.RGBA.PixOffset`.
    #[inline]
    pub fn pix_offset(&self, x: usize, y: usize) -> usize {
        (y * self.w + x) * 4
    }

    /// Average opaque color of the image (per-channel integer mean), as fogleman's
    /// `AverageImageColor` computes the default background.
    pub fn average_color(&self) -> (u8, u8, u8) {
        let (mut r, mut g, mut b) = (0u64, 0u64, 0u64);
        for px in self.pix.chunks_exact(4) {
            r += px[0] as u64;
            g += px[1] as u64;
            b += px[2] as u64;
        }
        let n = (self.w * self.h) as u64;
        ((r / n) as u8, (g / n) as u8, (b / n) as u8)
    }
}
