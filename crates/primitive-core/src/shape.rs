//! Shapes and their geometry mutations.
//!
//! Port of fogleman's `Shape` family. Ships `Triangle` (default mode `m=1`), `Ellipse`, and
//! axis-aligned `Rectangle` (plan §10 Q5: triangle + ellipse + rect first, expand after PKG-1);
//! the enum is the extension seam for the rotated/curved variants. Geometry only — color is solved
//! closed-form (`color.rs`), never mutated here.

use crate::raster::{
    clamp_i32, crop_scanlines, rasterize_ellipse, rasterize_rectangle, rasterize_triangle, Scanline,
};
use crate::rng::Rng;

/// Which primitive a search is fitting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ShapeType {
    #[default]
    Triangle,
    Ellipse,
    Rectangle,
}

/// A geometric primitive instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Triangle(Triangle),
    Ellipse(Ellipse),
    Rectangle(Rectangle),
}

impl Shape {
    /// A fresh random shape of `t` (Triangle is mutated once on construction, matching fogleman's
    /// `NewRandomTriangle`; Ellipse/Rectangle are returned as-rolled, matching theirs).
    pub fn random(t: ShapeType, w: i32, h: i32, rng: &mut Rng) -> Shape {
        match t {
            ShapeType::Triangle => Shape::Triangle(Triangle::random(w, h, rng)),
            ShapeType::Ellipse => Shape::Ellipse(Ellipse::random(w, h, rng)),
            ShapeType::Rectangle => Shape::Rectangle(Rectangle::random(w, h, rng)),
        }
    }

    /// Perturb the geometry in place (Triangle re-rolls until valid; Ellipse/Rectangle have no
    /// validity constraint, matching fogleman).
    pub fn mutate(&mut self, w: i32, h: i32, rng: &mut Rng) {
        match self {
            Shape::Triangle(t) => t.mutate(w, h, rng),
            Shape::Ellipse(e) => e.mutate(w, h, rng),
            Shape::Rectangle(r) => r.mutate(w, h, rng),
        }
    }

    /// Cropped scanlines covering this shape on a `w × h` canvas.
    pub fn rasterize(&self, w: i32, h: i32) -> Vec<Scanline> {
        match self {
            Shape::Triangle(t) => t.rasterize(w, h),
            Shape::Ellipse(e) => e.rasterize(w, h),
            Shape::Rectangle(r) => r.rasterize(w, h),
        }
    }

    /// SVG element for this shape, with the given `fill`/`fill-opacity` attribute string.
    pub fn svg(&self, attrs: &str) -> String {
        match self {
            Shape::Triangle(t) => t.svg(attrs),
            Shape::Ellipse(e) => e.svg(attrs),
            Shape::Rectangle(r) => r.svg(attrs),
        }
    }

    /// Flat `[x1, y1, x2, y2, x3, y3]` — the layout the GPU `score_triangles` kernel consumes.
    pub fn triangle_coords(&self) -> [i32; 6] {
        match self {
            Shape::Triangle(t) => [t.x1, t.y1, t.x2, t.y2, t.x3, t.y3],
            Shape::Ellipse(_) | Shape::Rectangle(_) => {
                panic!("triangle_coords called on a non-triangle shape (use ellipse_coords / rectangle_coords)")
            }
        }
    }

    /// Flat `[cx, cy, rx, ry]` — the layout the GPU `score_ellipses` kernel consumes (CORE-3b.2).
    pub fn ellipse_coords(&self) -> [i32; 4] {
        match self {
            Shape::Ellipse(e) => [e.x, e.y, e.rx, e.ry],
            _ => panic!("ellipse_coords called on a non-ellipse shape"),
        }
    }

    /// Flat `[x1, y1, x2, y2]` (opposite corners) — the layout the GPU `score_rectangles` kernel
    /// consumes (CORE-3b.2).
    pub fn rectangle_coords(&self) -> [i32; 4] {
        match self {
            Shape::Rectangle(r) => [r.x1, r.y1, r.x2, r.y2],
            _ => panic!("rectangle_coords called on a non-rectangle shape"),
        }
    }
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

/// An axis-aligned ellipse: centre `(x, y)`, radii `(rx, ry)`. Port of fogleman's `Ellipse`
/// (the non-rotated, non-circle variant). Always valid — no sliver rejection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ellipse {
    pub x: i32,
    pub y: i32,
    pub rx: i32,
    pub ry: i32,
}

impl Ellipse {
    fn random(w: i32, h: i32, rng: &mut Rng) -> Ellipse {
        Ellipse {
            x: rng.intn(w),
            y: rng.intn(h),
            rx: rng.intn(32) + 1,
            ry: rng.intn(32) + 1,
        }
    }

    fn mutate(&mut self, w: i32, h: i32, rng: &mut Rng) {
        match rng.intn(3) {
            0 => {
                self.x = clamp_i32(self.x + (rng.norm() * 16.0) as i32, 0, w - 1);
                self.y = clamp_i32(self.y + (rng.norm() * 16.0) as i32, 0, h - 1);
            }
            1 => {
                self.rx = clamp_i32(self.rx + (rng.norm() * 16.0) as i32, 1, w - 1);
            }
            _ => {
                self.ry = clamp_i32(self.ry + (rng.norm() * 16.0) as i32, 1, h - 1);
            }
        }
    }

    fn rasterize(&self, w: i32, h: i32) -> Vec<Scanline> {
        let mut buf = Vec::new();
        rasterize_ellipse(self.x, self.y, self.rx, self.ry, w, h, &mut buf);
        // Defensive only: fogleman's Ellipse.Rasterize self-clips x and skips out-of-frame rows, so
        // with cx/cy in-bounds (random/mutate enforce it) this crop is a no-op — kept for safety.
        crop_scanlines(&mut buf, w, h);
        buf
    }

    fn svg(&self, attrs: &str) -> String {
        format!(
            "<ellipse {} cx=\"{}\" cy=\"{}\" rx=\"{}\" ry=\"{}\" />",
            attrs, self.x, self.y, self.rx, self.ry
        )
    }
}

/// An axis-aligned rectangle with inclusive opposite corners `(x1, y1)`–`(x2, y2)`. Port of
/// fogleman's `Rectangle`. Corners may be unsorted; `bounds`/raster/SVG normalize. Always valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rectangle {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

impl Rectangle {
    fn random(w: i32, h: i32, rng: &mut Rng) -> Rectangle {
        let x1 = rng.intn(w);
        let y1 = rng.intn(h);
        let x2 = clamp_i32(x1 + rng.intn(32) + 1, 0, w - 1);
        let y2 = clamp_i32(y1 + rng.intn(32) + 1, 0, h - 1);
        Rectangle { x1, y1, x2, y2 }
    }

    /// Sorted inclusive corners `(x1, y1, x2, y2)` with `x1 ≤ x2`, `y1 ≤ y2` — fogleman's `bounds`.
    fn bounds(&self) -> (i32, i32, i32, i32) {
        let (xa, xb) = if self.x1 <= self.x2 {
            (self.x1, self.x2)
        } else {
            (self.x2, self.x1)
        };
        let (ya, yb) = if self.y1 <= self.y2 {
            (self.y1, self.y2)
        } else {
            (self.y2, self.y1)
        };
        (xa, ya, xb, yb)
    }

    fn mutate(&mut self, w: i32, h: i32, rng: &mut Rng) {
        match rng.intn(2) {
            0 => {
                self.x1 = clamp_i32(self.x1 + (rng.norm() * 16.0) as i32, 0, w - 1);
                self.y1 = clamp_i32(self.y1 + (rng.norm() * 16.0) as i32, 0, h - 1);
            }
            _ => {
                self.x2 = clamp_i32(self.x2 + (rng.norm() * 16.0) as i32, 0, w - 1);
                self.y2 = clamp_i32(self.y2 + (rng.norm() * 16.0) as i32, 0, h - 1);
            }
        }
    }

    fn rasterize(&self, w: i32, h: i32) -> Vec<Scanline> {
        let mut buf = Vec::new();
        rasterize_rectangle(self.x1, self.y1, self.x2, self.y2, &mut buf);
        // Defensive only: corners are always in-bounds (random/mutate clamp), so crop is a no-op
        // here — fogleman's Rectangle.Rasterize doesn't crop; kept for safety/consistency.
        crop_scanlines(&mut buf, w, h);
        buf
    }

    fn svg(&self, attrs: &str) -> String {
        let (xa, ya, xb, yb) = self.bounds();
        format!(
            "<rect {} x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" />",
            attrs,
            xa,
            ya,
            xb - xa + 1,
            yb - ya + 1
        )
    }
}
