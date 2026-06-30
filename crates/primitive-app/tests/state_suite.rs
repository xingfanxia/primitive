//! GUI-2 gate 1 (plan architecture.md lines 246-259): the §5A interaction-state matrix, one
//! `#[test]` per surface×state cell, over the **pure** [`primitive_app::state`] model (no egui, no
//! window). The harness "N passed" line is the gate's "pass count covering every §5A row".
//!
//! Rows covered: Canvas · Source panel · Controls · Start/Export · Device/perf.

use primitive_app::state::{CanvasView, ChipKind, Device, Phase, Primary, SourceView, Ui, UiFacts};

/// A neutral baseline (empty/first-run, GPU present). Each test perturbs only the cell it asserts.
fn facts() -> UiFacts {
    UiFacts {
        phase: Phase::Empty,
        has_image: false,
        has_error: false,
        shapes_committed: 0,
        svg_ready: false,
        device: Device::Metal,
        reduce_motion: false,
    }
}

fn loaded_idle() -> UiFacts {
    UiFacts {
        phase: Phase::Ready,
        has_image: true,
        ..facts()
    }
}

fn running() -> UiFacts {
    UiFacts {
        phase: Phase::Running,
        has_image: true,
        shapes_committed: 12,
        ..facts()
    }
}

fn done() -> UiFacts {
    UiFacts {
        phase: Phase::Done,
        has_image: true,
        shapes_committed: 250,
        svg_ready: true,
        ..facts()
    }
}

// ── Canvas row ───────────────────────────────────────────────────────────────
#[test]
fn canvas_empty_first_run_is_dropzone() {
    assert_eq!(Ui::derive(facts()).canvas, CanvasView::DropZone);
}

#[test]
fn canvas_error_on_decode_fail() {
    let f = UiFacts {
        has_error: true,
        ..facts()
    };
    assert_eq!(Ui::derive(f).canvas, CanvasView::Error);
}

#[test]
fn canvas_partial_in_progress_is_live() {
    assert_eq!(Ui::derive(running()).canvas, CanvasView::Live);
}

#[test]
fn canvas_loaded_idle_is_staged() {
    assert_eq!(Ui::derive(loaded_idle()).canvas, CanvasView::Staged);
}

#[test]
fn canvas_success_is_done() {
    assert_eq!(Ui::derive(done()).canvas, CanvasView::Done);
}

// ── Source panel row ─────────────────────────────────────────────────────────
#[test]
fn source_empty_is_dropzone() {
    assert_eq!(Ui::derive(facts()).source, SourceView::DropZone);
}

#[test]
fn source_success_is_loaded() {
    assert_eq!(Ui::derive(loaded_idle()).source, SourceView::Loaded);
}

#[test]
fn source_partial_is_dimmed_mid_run() {
    assert_eq!(Ui::derive(running()).source, SourceView::Dimmed);
}

// ── Controls row ─────────────────────────────────────────────────────────────
#[test]
fn controls_empty_are_editable() {
    assert!(Ui::derive(loaded_idle()).controls_editable);
}

#[test]
fn controls_disabled_mid_run_except_pause_and_reset() {
    let ui = Ui::derive(running());
    assert!(!ui.controls_editable, "core controls disabled mid-run");
    assert_eq!(ui.actions.primary, Primary::Pause, "Pause still actionable");
    assert!(ui.actions.primary_enabled);
    assert!(ui.actions.reset_enabled, "Reset still actionable");
}

// ── Start / Export row ───────────────────────────────────────────────────────
#[test]
fn start_gated_on_image() {
    assert!(
        !Ui::derive(facts()).actions.primary_enabled,
        "no image → Start off"
    );
    assert!(
        Ui::derive(loaded_idle()).actions.primary_enabled,
        "image → Start on"
    );
}

#[test]
fn export_disabled_until_one_shape() {
    let ui = Ui::derive(loaded_idle());
    assert!(!ui.actions.export_png_enabled, "0 shapes → PNG off");
    assert!(!ui.actions.export_gif_enabled, "0 shapes → GIF off");
    let one = UiFacts {
        phase: Phase::Done,
        shapes_committed: 1,
        ..loaded_idle()
    };
    assert!(
        Ui::derive(one).actions.export_png_enabled,
        "≥1 shape → PNG on"
    );
}

#[test]
fn start_pause_toggle_mid_run() {
    assert_eq!(Ui::derive(running()).actions.primary, Primary::Pause);
    let paused = UiFacts {
        phase: Phase::Paused,
        ..running()
    };
    assert_eq!(Ui::derive(paused).actions.primary, Primary::Resume);
}

#[test]
fn export_disabled_mid_run_until_paused_or_done() {
    // Actively painting → raster export off even with shapes committed…
    assert!(!Ui::derive(running()).actions.export_png_enabled);
    // …paused → PNG/GIF on (a stable frame), SVG still waits for done.
    let paused = UiFacts {
        phase: Phase::Paused,
        ..running()
    };
    let ui = Ui::derive(paused);
    assert!(ui.actions.export_png_enabled);
    assert!(!ui.actions.export_svg_enabled);
}

#[test]
fn export_svg_enabled_when_done() {
    assert!(Ui::derive(done()).actions.export_svg_enabled);
}

// ── Device / perf row ────────────────────────────────────────────────────────
#[test]
fn device_backend_chip_metal() {
    let ui = Ui::derive(loaded_idle());
    assert_eq!(ui.device_label, "Metal");
    assert_eq!(ui.device_kind, ChipKind::Green);
}

#[test]
fn device_amber_chip_no_gpu_found() {
    let f = UiFacts {
        device: Device::CpuFallback,
        ..facts()
    };
    let ui = Ui::derive(f);
    assert_eq!(ui.device_label, "CPU (no GPU found)");
    assert_eq!(
        ui.device_kind,
        ChipKind::Amber,
        "first-class fallback is amber"
    );
}

#[test]
fn device_neutral_chip_deliberate_cpu() {
    let f = UiFacts {
        device: Device::Cpu,
        ..facts()
    };
    assert_eq!(Ui::derive(f).device_kind, ChipKind::Neutral);
}
