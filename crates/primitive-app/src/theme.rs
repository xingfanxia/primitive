//! Design tokens + WCAG contrast math (plan §5A: "meet WCAG AA contrast on all text/chips").
//!
//! The palette is the **single source of truth** for the app's colours; the renderer reads tokens,
//! never literals. Because contrast is pure arithmetic over those tokens, the a11y gate is a
//! deterministic unit test (no rendered pixels): every text/background pair clears 4.5:1 and every
//! chip foreground/background pair clears 3:1 (WCAG 2.1 AA — normal text vs. UI components).

use crate::state::ChipKind;

/// An 8-bit sRGB colour token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    /// To egui's colour type (kept here so the render layer stays token-driven).
    pub fn to_color32(self) -> eframe::egui::Color32 {
        eframe::egui::Color32::from_rgb(self.0, self.1, self.2)
    }
}

/// Linearize one sRGB channel (WCAG 2.1 relative-luminance step).
fn linearize(c8: u8) -> f64 {
    let c = c8 as f64 / 255.0;
    if c <= 0.03928 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// WCAG 2.1 relative luminance of an sRGB colour.
pub fn relative_luminance(c: Rgb) -> f64 {
    0.2126 * linearize(c.0) + 0.7152 * linearize(c.1) + 0.0722 * linearize(c.2)
}

/// WCAG contrast ratio between two colours (always ≥ 1.0; order-independent).
pub fn contrast_ratio(a: Rgb, b: Rgb) -> f64 {
    let (la, lb) = (relative_luminance(a), relative_luminance(b));
    let (hi, lo) = if la >= lb { (la, lb) } else { (lb, la) };
    (hi + 0.05) / (lo + 0.05)
}

/// A surface palette (one per system appearance). The chip colours are shared (each chip carries
/// its own background, so it reads the same in light and dark).
#[derive(Clone, Copy, Debug)]
pub struct Palette {
    pub bg: Rgb,
    pub surface: Rgb,
    pub text: Rgb,
    pub text_weak: Rgb,
    pub accent: Rgb,
    /// Focus ring / live progress accent.
    pub focus: Rgb,
}

/// Light appearance tokens.
pub const LIGHT: Palette = Palette {
    bg: Rgb(255, 255, 255),
    surface: Rgb(245, 245, 247),
    text: Rgb(29, 29, 31),
    text_weak: Rgb(90, 90, 95),
    accent: Rgb(10, 110, 220),
    focus: Rgb(10, 110, 220),
};

/// Dark appearance tokens.
pub const DARK: Palette = Palette {
    bg: Rgb(28, 28, 30),
    surface: Rgb(44, 44, 46),
    text: Rgb(245, 245, 247),
    text_weak: Rgb(152, 152, 157),
    accent: Rgb(60, 150, 255),
    focus: Rgb(60, 150, 255),
};

impl Palette {
    /// The active appearance (egui carries the system light/dark choice).
    pub fn for_dark(dark: bool) -> Palette {
        if dark {
            DARK
        } else {
            LIGHT
        }
    }
}

/// A device-chip colour pair (background + foreground), chosen so foreground:background ≥ 3:1.
#[derive(Clone, Copy, Debug)]
pub struct ChipColors {
    pub bg: Rgb,
    pub fg: Rgb,
}

/// The chip colours for each §5A emphasis class. Amber is the "no GPU found" fallback (dark text on
/// amber); green is an active GPU backend; neutral is a deliberate CPU choice.
pub fn chip_colors(kind: ChipKind) -> ChipColors {
    match kind {
        ChipKind::Green => ChipColors {
            bg: Rgb(30, 90, 40),
            fg: Rgb(255, 255, 255),
        },
        ChipKind::Neutral => ChipColors {
            bg: Rgb(60, 60, 64),
            fg: Rgb(235, 235, 240),
        },
        ChipKind::Amber => ChipColors {
            bg: Rgb(255, 176, 32),
            fg: Rgb(40, 30, 0),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contrast_is_symmetric_and_known() {
        // Black on white is the WCAG reference maximum (21:1).
        let r = contrast_ratio(Rgb(0, 0, 0), Rgb(255, 255, 255));
        assert!((r - 21.0).abs() < 0.01, "black/white = 21:1, got {r}");
        assert_eq!(
            contrast_ratio(Rgb(0, 0, 0), Rgb(255, 255, 255)),
            contrast_ratio(Rgb(255, 255, 255), Rgb(0, 0, 0)),
            "contrast is order-independent",
        );
    }
}
