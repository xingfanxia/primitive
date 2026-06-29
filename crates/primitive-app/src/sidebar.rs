//! The quiet right-hand sidebar (plan §5A): Source · Shapes · actions. Returns an [`Action`] for the
//! app to apply, so widget borrows stay local. Enable/disable follows the §5A interaction table:
//! controls are live only when not mid-run; export needs ≥1 shape and a paused/done run.

use eframe::egui;

use crate::app::{Action, Phase, PrimitiveApp};
use crate::image_io;

fn header(ui: &mut egui::Ui, text: &str) {
    ui.add_space(10.0);
    ui.label(egui::RichText::new(text).strong().small());
    ui.add_space(3.0);
}

pub fn show(app: &mut PrimitiveApp, ui: &mut egui::Ui) -> Option<Action> {
    let mut action = None;
    let running = matches!(app.phase, Phase::Running | Phase::Paused);
    let has_image = app.target.is_some();

    // ── SOURCE ──────────────────────────────────────────────
    header(ui, "SOURCE");
    ui.add_enabled_ui(!running, |ui| {
        if ui.button("Drop image · or Browse…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Image", &["png", "jpg", "jpeg", "webp", "bmp", "gif"])
                .pick_file()
            {
                action = Some(Action::LoadPath(path));
            }
        }
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("try:").weak().small());
            for (i, (name, _)) in image_io::SAMPLES.iter().enumerate() {
                if ui.small_button(*name).clicked() {
                    action = Some(Action::LoadSample(i));
                }
            }
        });
    });
    if let Some((w, h)) = app.source_dims {
        ui.label(
            egui::RichText::new(format!("{}  ·  {}×{}", app.source_name, w, h))
                .weak()
                .small(),
        );
    }

    // ── SHAPES ──────────────────────────────────────────────
    header(ui, "SHAPES");
    ui.add_enabled_ui(!running, |ui| {
        ui.horizontal(|ui| {
            ui.label("type");
            let _ = ui.selectable_label(true, "△ triangle");
            ui.label(egui::RichText::new("(more soon)").weak().small());
        });
        ui.add(egui::Slider::new(&mut app.count, 1..=2000).text("count"));
        ui.add(egui::Slider::new(&mut app.alpha, 1..=255).text("alpha"));
    });

    ui.add_space(14.0);
    ui.separator();
    ui.add_space(6.0);

    // ── ACTIONS ─────────────────────────────────────────────
    ui.horizontal(|ui| match app.phase {
        Phase::Running => {
            if ui.button("⏸ Pause").clicked() {
                action = Some(Action::Pause);
            }
            if ui.button("↺ Reset").clicked() {
                action = Some(Action::Reset);
            }
        }
        Phase::Paused => {
            if ui.button("▶ Resume").clicked() {
                action = Some(Action::Resume);
            }
            if ui.button("↺ Reset").clicked() {
                action = Some(Action::Reset);
            }
        }
        _ => {
            if ui
                .add_enabled(has_image, egui::Button::new("▶ Start"))
                .clicked()
            {
                action = Some(Action::Start);
            }
            if ui
                .add_enabled(app.phase == Phase::Done, egui::Button::new("↺ Reset"))
                .clicked()
            {
                action = Some(Action::Reset);
            }
        }
    });

    // Export: PNG once a frame exists and the run isn't actively painting; SVG when the run is done.
    let can_png = app.current_canvas.is_some() && app.phase != Phase::Running;
    let can_svg = app.final_svg.is_some();
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if ui
            .add_enabled(can_png, egui::Button::new("⤓ PNG"))
            .clicked()
        {
            action = Some(Action::ExportPng);
        }
        if ui
            .add_enabled(can_svg, egui::Button::new("⤓ SVG"))
            .clicked()
        {
            action = Some(Action::ExportSvg);
        }
    });

    action
}
