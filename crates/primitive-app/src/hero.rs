//! The hero region renderer (plan §5A): the live reconstruction canvas, the progress strip, and the
//! device chip. Pure rendering — every visibility/emphasis decision is read from the [`Ui`] state
//! ([`crate::app`] derives it); this module never re-decides interaction logic.

use eframe::egui;

use crate::app::PrimitiveApp;
use crate::state::{CanvasView, Ui};
use crate::theme;

/// The §5A device chip — token-coloured by emphasis (green GPU / neutral CPU / amber fallback).
pub fn device_chip(app: &PrimitiveApp, ui: &mut egui::Ui, state: &Ui) {
    let c = theme::chip_colors(state.device_kind);
    egui::Frame::new()
        .fill(c.bg.to_color32())
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::vec2(8.0, 2.0))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(state.device_label)
                    .monospace()
                    .color(c.fg.to_color32()),
            )
            .on_hover_text(app.strings.chip_tooltip);
        });
}

/// The hero region: the live reconstruction, aspect-fit, with the progress strip beneath.
pub fn canvas(app: &PrimitiveApp, ui: &mut egui::Ui, state: &Ui) {
    if let Some((msg, _)) = &app.toast {
        let color = if app.error_toast {
            ui.visuals().error_fg_color
        } else {
            ui.visuals().weak_text_color()
        };
        ui.colored_label(color, msg);
    }
    if matches!(state.canvas, CanvasView::DropZone | CanvasView::Error) && app.texture.is_none() {
        ui.centered_and_justified(|ui| {
            ui.label(
                egui::RichText::new(app.strings.drop_to_begin)
                    .size(18.0)
                    .weak(),
            );
        });
        return;
    }
    let Some(tex) = &app.texture else { return };
    let (cw, ch) = app
        .target
        .as_ref()
        .map(|c| (c.w as f32, c.h as f32))
        .unwrap_or((1.0, 1.0));
    let avail = ui.available_size();
    let scale = ((avail.x) / cw).min((avail.y - 46.0) / ch).max(0.01);
    let size = egui::vec2(cw * scale, ch * scale);
    // §5A "done" emphasis pulse — suppressed under Reduce Motion (a11y gate).
    if state.canvas == CanvasView::Done && state.motion_pulse {
        ui.ctx().request_repaint();
    }
    ui.vertical_centered(|ui| {
        ui.add(egui::Image::new(egui::load::SizedTexture::new(
            tex.id(),
            size,
        )));
        progress_strip(app, ui, size.x);
    });
}

/// The progress bar + `done/total · sps · dB` readout. Always renders a labelled progress node once
/// an image is loaded (so VoiceOver / AccessKit always exposes a live-progress element).
fn progress_strip(app: &PrimitiveApp, ui: &mut egui::Ui, width: f32) {
    if app.target.is_none() {
        return;
    }
    let (idx, total, sps, score) = match &app.last {
        Some(p) => (p.shape_index, p.total, p.sps, p.score),
        None => (0, app.params.count.max(1) as usize, 0.0, f64::NAN),
    };
    let frac = idx as f32 / total.max(1) as f32;
    ui.add(
        egui::ProgressBar::new(frac)
            .desired_width(width)
            .show_percentage(),
    );
    let readout = if score.is_finite() {
        format!(
            "{idx}/{total}  ·  {sps:.0} shapes/sec  ·  {:.1} dB",
            primitive_core::psnr_from_score(score)
        )
    } else {
        format!("{idx}/{total}")
    };
    // Labelled so it lands in the AccessKit tree as the live-progress node.
    ui.label(egui::RichText::new(readout).monospace().weak());
}
