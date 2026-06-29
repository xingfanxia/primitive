//! # primitive — the GUI shell (eframe composition root)
//!
//! GUI-1/GUI-2 (plan §5/§5A): a one-window, single-document tool whose hero is the **live
//! reconstruction canvas**. The composition root wires the CPU search adapter into the engine and
//! drives it on a background thread, streaming canvas frames to the UI (plan §5A: the spectacle).
//!
//! The compute device chip, controls, and interaction states follow the §5A design spec — which is
//! the project's DESIGN source of truth for this surface.

mod app;
mod image_io;
mod runner;
mod sidebar;

use eframe::egui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1040.0, 700.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("primitive"),
        ..Default::default()
    };
    eframe::run_native(
        "primitive",
        options,
        Box::new(|cc| Ok(Box::new(app::PrimitiveApp::new(cc)))),
    )
}
