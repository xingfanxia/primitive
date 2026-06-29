//! # primitive — the GUI library / composition root (eframe)
//!
//! GUI-1/GUI-2 (plan §5/§5A): a one-window, single-document tool whose hero is the **live
//! reconstruction canvas**. The composition root wires the compute adapters (CPU oracle + GPU
//! "instant" mode) into the engine and drives them off the UI thread, streaming canvas frames
//! to the UI (plan §5A: the spectacle).
//!
//! Split lib + thin bin so the §5A interaction logic, the design-token a11y math, the device
//! probe, and the end-to-end run are **unit-testable headless** (the GUI-2 gates) — the binary
//! (`primitive`) is just [`run`].
//!
//! Layering inside the app:
//! - [`state`] — the **pure** §5A interaction-state model (no egui/IO). The render layer reads it.
//! - [`theme`] — design tokens + WCAG contrast math (a11y gate).
//! - [`device`] — backend probe (env override + GPU detection) → the device chip.
//! - [`i18n`] — the string table (no hard-coded UI text).
//! - [`app`]/[`sidebar`]/[`hero`] — egui rendering, derived entirely from `state`.
//! - [`runner`] — the background optimizer thread (CPU streaming + GPU instant).
//! - [`image_io`] — load/export adapters (PNG/SVG/GIF).

pub mod app;
pub mod device;
pub mod hero;
pub mod i18n;
pub mod image_io;
pub mod runner;
pub mod sidebar;
pub mod state;
pub mod theme;

use eframe::egui;

/// Launch the native window (the `primitive` binary's whole body).
pub fn run() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1040.0, 700.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title(i18n::Strings::en().app_title),
        // Remember last window size/position across launches (§5A window behavior).
        persist_window: true,
        ..Default::default()
    };
    eframe::run_native(
        "primitive",
        options,
        Box::new(|cc| Ok(Box::new(app::PrimitiveApp::new(cc)))),
    )
}
