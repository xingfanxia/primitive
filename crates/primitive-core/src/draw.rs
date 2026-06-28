//! Compositing covered scanlines into a canvas — ports of `copyLines` / `drawLines`.
//!
//! `draw_lines` alpha-composites a solid `Color` over the destination using the exact
//! 16-bit premultiplied arithmetic of Go's `image/draw` + `color.NRGBA.RGBA()`. The
//! intermediate products are provably ≤ `u32::MAX` for any `alpha ∈ [1,255]` and coverage
//! `≤ 0xffff`, so `u32` matches Go's `uint32` bit-for-bit without wrapping.

use crate::canvas::Canvas;
use crate::color::Color;
use crate::raster::Scanline;

/// Copy `lines`' pixel runs from `src` into `dst` — `copyLines`.
pub fn copy_lines(dst: &mut Canvas, src: &Canvas, lines: &[Scanline]) {
    for line in lines {
        let a = dst.pix_offset(line.x1 as usize, line.y as usize);
        let b = a + (line.x2 - line.x1 + 1) as usize * 4;
        dst.pix[a..b].copy_from_slice(&src.pix[a..b]);
    }
}

/// Expand an NRGBA color to 16-bit premultiplied (sr,sg,sb,sa) — Go's `color.NRGBA.RGBA()`.
#[inline]
fn nrgba_to_premul(c: Color) -> (u32, u32, u32, u32) {
    let av = c.a as u32;
    let mut r = c.r as u32;
    r |= r << 8;
    r = r * av / 0xff;
    let mut g = c.g as u32;
    g |= g << 8;
    g = g * av / 0xff;
    let mut b = c.b as u32;
    b |= b << 8;
    b = b * av / 0xff;
    let mut a = av;
    a |= a << 8;
    (r, g, b, a)
}

/// Alpha-composite solid `c` over `im` along `lines` — `drawLines`.
pub fn draw_lines(im: &mut Canvas, c: Color, lines: &[Scanline]) {
    const M: u32 = 0xffff;
    let (sr, sg, sb, sa) = nrgba_to_premul(c);
    for line in lines {
        let ma = line.alpha;
        let a = (M - sa * ma / M) * 0x101;
        let mut i = im.pix_offset(line.x1 as usize, line.y as usize);
        for _x in line.x1..=line.x2 {
            let dr = im.pix[i] as u32;
            let dg = im.pix[i + 1] as u32;
            let db = im.pix[i + 2] as u32;
            let da = im.pix[i + 3] as u32;
            im.pix[i] = (((dr * a + sr * ma) / M) >> 8) as u8;
            im.pix[i + 1] = (((dg * a + sg * ma) / M) >> 8) as u8;
            im.pix[i + 2] = (((db * a + sb * ma) / M) >> 8) as u8;
            im.pix[i + 3] = (((da * a + sa * ma) / M) >> 8) as u8;
            i += 4;
        }
    }
}
