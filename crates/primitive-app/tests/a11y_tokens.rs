//! GUI-2 gate 5 (plan architecture.md line 278): deterministic a11y math over the design tokens —
//! no rendered pixels. Two contracts:
//!   (1) WCAG AA contrast: every text/background pair ≥ 4.5:1, every chip fg/bg pair ≥ 3:1.
//!   (2) Reduce Motion turns the live/done repaint pulse off.

use primitive_app::state::{ChipKind, Device, Phase, Ui, UiFacts};
use primitive_app::theme::{self, Palette};

fn assert_text_aa(p: Palette, name: &str) {
    for (label, fg) in [("text", p.text), ("text_weak", p.text_weak)] {
        let r = theme::contrast_ratio(fg, p.bg);
        assert!(
            r >= 4.5,
            "{name}: {label} on bg must be ≥ 4.5:1 (WCAG AA normal text), got {r:.2}:1",
        );
    }
    // Accent (focus ring / live progress) is a UI component → the 3:1 graphical-object threshold.
    let r = theme::contrast_ratio(p.accent, p.bg);
    assert!(r >= 3.0, "{name}: accent on bg must be ≥ 3:1, got {r:.2}:1");
}

#[test]
fn light_palette_text_meets_wcag_aa() {
    assert_text_aa(theme::LIGHT, "light");
}

#[test]
fn dark_palette_text_meets_wcag_aa() {
    assert_text_aa(theme::DARK, "dark");
}

#[test]
fn every_chip_foreground_meets_3to1() {
    for kind in [ChipKind::Green, ChipKind::Neutral, ChipKind::Amber] {
        let c = theme::chip_colors(kind);
        let r = theme::contrast_ratio(c.fg, c.bg);
        assert!(
            r >= 3.0,
            "{kind:?} chip fg/bg must be ≥ 3:1 (WCAG UI component), got {r:.2}:1",
        );
    }
}

#[test]
fn amber_fallback_chip_is_legible() {
    // The §5A "CPU (no GPU found)" chip is the most-likely-to-fail amber pair — guard it explicitly.
    let c = theme::chip_colors(Device::CpuFallback.chip_kind());
    assert!(theme::contrast_ratio(c.fg, c.bg) >= 3.0);
}

#[test]
fn reduce_motion_disables_repaint_pulse() {
    let base = UiFacts {
        phase: Phase::Done,
        has_image: true,
        has_error: false,
        shapes_committed: 100,
        svg_ready: true,
        device: Device::Metal,
        reduce_motion: false,
    };
    assert!(Ui::derive(base).motion_pulse, "pulse on by default");
    let reduced = UiFacts {
        reduce_motion: true,
        ..base
    };
    assert!(
        !Ui::derive(reduced).motion_pulse,
        "Reduce Motion must turn the repaint pulse off",
    );
}
