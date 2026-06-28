//! Shapes and their geometry mutations.
//!
//! Port of fogleman's `Shape` family. v1 ships the `Triangle` (default mode `m=1`); the
//! enum is the extension seam for ellipse / rect / rotated-rect (plan §10 Q5 recommends
//! triangle + ellipse + rect by GPU-3). Geometry only — color is solved closed-form
//! (`color.rs`), never mutated here.

use crate::raster::{crop_scanlines, rasterize_triangle, Scanline};
use crate::rng::Rng;

/// Which primitive a search is fitting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeType {
    Triangle,
}

/// A geometric primitive instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Triangle(Triangle),
}

impl Shape {
    /// A fresh random shape of `t`, already mutated once (matches fogleman's constructors).
    pub fn random(t: ShapeType, w: i32, h: i32, rng: &mut Rng) -> Shape {
        match t {
            ShapeType::Triangle => Shape::Triangle(Triangle::random(w, h, rng)),
        }
    }

    /// Perturb the geometry in place, re-rolling until the shape is valid.
    pub fn mutate(&mut self, w: i32, h: i32, rng: &mut Rng) {
        match self {
            Shape::Triangle(t) => t.mutate(w, h, rng),
        }
    }

    /// Cropped scanlines covering this shape on a `w × h` canvas.
    pub fn rasterize(&self, w: i32, h: i32) -> Vec<Scanline> {
        match self {
            Shape::Triangle(t) => t.rasterize(w, h),
        }
    }

    /// SVG element for this shape, with the given `fill`/`fill-opacity` attribute string.
    pub fn svg(&self, attrs: &str) -> String {
        match self {
            Shape::Triangle(t) => t.svg(attrs),
        }
    }
}

#[inline]
fn clamp_i32(x: i32, lo: i32, hi: i32) -> i32 {
    x.max(lo).min(hi)
}

#[inline]
fn degrees(radians: f64) -> f64 {
    radians * 180.0 / core::f64::consts::PI
}

/// A triangle defined by three integer vertices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Triangle {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
    pub x3: i32,
    pub y3: i32,
}

impl Triangle {
    fn random(w: i32, h: i32, rng: &mut Rng) -> Triangle {
        let x1 = rng.intn(w);
        let y1 = rng.intn(h);
        let x2 = x1 + rng.intn(31) - 15;
        let y2 = y1 + rng.intn(31) - 15;
        let x3 = x1 + rng.intn(31) - 15;
        let y3 = y1 + rng.intn(31) - 15;
        let mut t = Triangle {
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
        };
        t.mutate(w, h, rng);
        t
    }

    fn mutate(&mut self, w: i32, h: i32, rng: &mut Rng) {
        const M: i32 = 16;
        loop {
            match rng.intn(3) {
                0 => {
                    self.x1 = clamp_i32(self.x1 + (rng.norm() * 16.0) as i32, -M, w - 1 + M);
                    self.y1 = clamp_i32(self.y1 + (rng.norm() * 16.0) as i32, -M, h - 1 + M);
                }
                1 => {
                    self.x2 = clamp_i32(self.x2 + (rng.norm() * 16.0) as i32, -M, w - 1 + M);
                    self.y2 = clamp_i32(self.y2 + (rng.norm() * 16.0) as i32, -M, h - 1 + M);
                }
                _ => {
                    self.x3 = clamp_i32(self.x3 + (rng.norm() * 16.0) as i32, -M, w - 1 + M);
                    self.y3 = clamp_i32(self.y3 + (rng.norm() * 16.0) as i32, -M, h - 1 + M);
                }
            }
            if self.valid() {
                break;
            }
        }
    }

    /// Reject slivers — every interior angle must exceed 15° (fogleman's `Valid`).
    fn valid(&self) -> bool {
        const MIN_DEGREES: f64 = 15.0;
        let angle = |ax: i32, ay: i32, bx: i32, by: i32| -> f64 {
            let (mut x1, mut y1) = (ax as f64, ay as f64);
            let (mut x2, mut y2) = (bx as f64, by as f64);
            let d1 = (x1 * x1 + y1 * y1).sqrt();
            let d2 = (x2 * x2 + y2 * y2).sqrt();
            x1 /= d1;
            y1 /= d1;
            x2 /= d2;
            y2 /= d2;
            degrees((x1 * x2 + y1 * y2).acos())
        };
        let a1 = angle(
            self.x2 - self.x1,
            self.y2 - self.y1,
            self.x3 - self.x1,
            self.y3 - self.y1,
        );
        let a2 = angle(
            self.x1 - self.x2,
            self.y1 - self.y2,
            self.x3 - self.x2,
            self.y3 - self.y2,
        );
        let a3 = 180.0 - a1 - a2;
        a1 > MIN_DEGREES && a2 > MIN_DEGREES && a3 > MIN_DEGREES
    }

    fn rasterize(&self, w: i32, h: i32) -> Vec<Scanline> {
        let mut buf = Vec::new();
        rasterize_triangle(
            self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, &mut buf,
        );
        crop_scanlines(&mut buf, w, h);
        buf
    }

    fn svg(&self, attrs: &str) -> String {
        format!(
            "<polygon {} points=\"{},{} {},{} {},{}\" />",
            attrs, self.x1, self.y1, self.x2, self.y2, self.x3, self.y3
        )
    }
}
