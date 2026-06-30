//! The quiet right-hand sidebar (plan §5A): Source · Shapes · Advanced · actions. Every enable
//! decision is read from the pure [`Ui`] state — this module only renders and returns an [`Action`]
//! (so widget borrows stay local). It never re-derives interaction logic.

use eframe::egui;

use crate::app::{Action, PrimitiveApp};
use crate::image_io;
use crate::state::{Primary, Ui};

fn header(ui: &mut egui::Ui, text: &str) {
    ui.add_space(10.0);
    ui.label(egui::RichText::new(text).strong().small());
    ui.add_space(3.0);
}

pub fn show(app: &mut PrimitiveApp, state: &Ui, ui: &mut egui::Ui) -> Option<Action> {
    let mut action = None;
    let s = app.strings;

    // ── SOURCE ──────────────────────────────────────────────
    header(ui, s.source_header);
    ui.add_enabled_ui(state.controls_editable, |ui| {
        if ui.button(s.browse).clicked() {
            action = Some(Action::OpenFile);
        }
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(s.try_label).weak().small());
            for (i, (name, _)) in image_io::SAMPLES.iter().enumerate() {
                if ui.small_button(*name).clicked() {
                    action = Some(Action::LoadSample(i));
                }
            }
        });
    });
    if let Some((w, h)) = app.source_dims {
        // Dimmed while a run owns the source (§5A: can't swap mid-run; Reset to change).
        let mut text = egui::RichText::new(format!("{}  ·  {}×{}", app.source_name, w, h)).small();
        if !state.controls_editable {
            text = text.weak();
        }
        ui.label(text);
    }

    // ── SHAPES ──────────────────────────────────────────────
    header(ui, s.shapes_header);
    ui.add_enabled_ui(state.controls_editable, |ui| {
        ui.horizontal(|ui| {
            ui.label(s.type_label);
            let _ = ui.selectable_label(true, s.triangle);
            ui.label(egui::RichText::new(s.more_soon).weak().small());
        });
        ui.add(egui::Slider::new(&mut app.params.count, 1..=2000).text(s.count));
        ui.add(egui::Slider::new(&mut app.params.alpha, 1..=255).text(s.alpha));
    });

    // ── ADVANCED (collapsed power-user knobs) ───────────────
    // Controlled by `app.show_advanced` (so ⌘, can open it); clicking the header toggles it too.
    let resp = egui::CollapsingHeader::new(s.advanced_header)
        .open(Some(app.show_advanced))
        .show(ui, |ui| {
            ui.add_enabled_ui(state.controls_editable, |ui| {
                let mut seed = app.params.seed as i64;
                if ui
                    .add(egui::Slider::new(&mut seed, 0..=9999).text(s.seed))
                    .changed()
                {
                    app.params.seed = seed as u64;
                }
                ui.add(egui::Slider::new(&mut app.params.n, 1..=4000).text(s.restarts_n));
                ui.add(egui::Slider::new(&mut app.params.age, 1..=400).text(s.age_m));
                ui.add(egui::Slider::new(&mut app.params.m, 1..=64).text(s.attempts));
            });
            ui.checkbox(&mut app.params.reduce_motion, s.reduce_motion);
        });
    if resp.header_response.clicked() {
        app.show_advanced = !app.show_advanced;
    }

    ui.add_space(14.0);
    ui.separator();
    ui.add_space(6.0);

    // ── ACTIONS ─────────────────────────────────────────────
    ui.horizontal(|ui| {
        let primary_label = match state.actions.primary {
            Primary::Start => s.start,
            Primary::Pause => s.pause,
            Primary::Resume => s.resume,
        };
        if ui
            .add_enabled(
                state.actions.primary_enabled,
                egui::Button::new(primary_label),
            )
            .clicked()
        {
            action = Some(match state.actions.primary {
                Primary::Start => Action::Start,
                Primary::Pause => Action::Pause,
                Primary::Resume => Action::Resume,
            });
        }
        if ui
            .add_enabled(state.actions.reset_enabled, egui::Button::new(s.reset))
            .clicked()
        {
            action = Some(Action::Reset);
        }
    });

    // ── EXPORT ▾ (PNG · SVG · GIF) ──────────────────────────
    ui.add_space(6.0);
    let any_export = state.actions.export_png_enabled
        || state.actions.export_svg_enabled
        || state.actions.export_gif_enabled;
    ui.add_enabled_ui(any_export, |ui| {
        ui.menu_button(s.export, |ui| {
            if ui
                .add_enabled(
                    state.actions.export_png_enabled,
                    egui::Button::new(s.export_png),
                )
                .clicked()
            {
                action = Some(Action::ExportPng);
                ui.close();
            }
            if ui
                .add_enabled(
                    state.actions.export_svg_enabled,
                    egui::Button::new(s.export_svg),
                )
                .clicked()
            {
                action = Some(Action::ExportSvg);
                ui.close();
            }
            if ui
                .add_enabled(
                    state.actions.export_gif_enabled,
                    egui::Button::new(s.export_gif),
                )
                .clicked()
            {
                action = Some(Action::ExportGif);
                ui.close();
            }
        });
    });

    action
}
