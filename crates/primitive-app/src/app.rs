//! App state machine + frame polling + the hero canvas (plan §5A). The right-hand controls live in
//! [`crate::sidebar`]; IO in [`crate::image_io`]; the optimizer thread in [`crate::runner`].

use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;
use primitive_core::Canvas;

use crate::image_io;
use crate::runner::{self, RunConfig, RunHandle, PAUSE, RUN, STOP};
use crate::sidebar;

/// Where the single-document flow is (drives which controls are live — §5A interaction table).
#[derive(PartialEq, Clone, Copy)]
pub enum Phase {
    Empty,
    Ready,
    Running,
    Paused,
    Done,
}

/// A button action raised by the sidebar, applied by [`PrimitiveApp::apply`] (keeps borrows simple).
pub enum Action {
    LoadPath(PathBuf),
    LoadSample(usize),
    Start,
    Pause,
    Resume,
    Reset,
    ExportPng,
    ExportSvg,
}

pub struct Progress {
    pub shape_index: usize,
    pub total: usize,
    pub score: f64,
    pub sps: f64,
}

pub struct PrimitiveApp {
    pub target: Option<Canvas>,
    pub source_dims: Option<(u32, u32)>,
    pub source_name: String,
    pub texture: Option<egui::TextureHandle>,
    pub current_canvas: Option<Canvas>,
    pub final_svg: Option<String>,
    pub phase: Phase,
    pub run: Option<RunHandle>,
    pub last: Option<Progress>,
    pub toast: Option<(String, Instant)>,
    pub count: i32,
    pub alpha: i32,
}

impl PrimitiveApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            target: None,
            source_dims: None,
            source_name: String::new(),
            texture: None,
            current_canvas: None,
            final_svg: None,
            phase: Phase::Empty,
            run: None,
            last: None,
            toast: None,
            count: 250,
            alpha: 128,
        }
    }

    fn toast(&mut self, msg: impl Into<String>) {
        self.toast = Some((msg.into(), Instant::now()));
    }

    fn set_texture(&mut self, ctx: &egui::Context, canvas: &Canvas) {
        let img = egui::ColorImage::from_rgba_unmultiplied([canvas.w, canvas.h], &canvas.pix);
        match &mut self.texture {
            Some(t) => t.set(img, egui::TextureOptions::LINEAR),
            None => {
                self.texture = Some(ctx.load_texture("canvas", img, egui::TextureOptions::LINEAR))
            }
        }
    }

    fn load(
        &mut self,
        ctx: &egui::Context,
        res: Result<((u32, u32), Canvas), String>,
        name: String,
    ) {
        match res {
            Ok((dims, canvas)) => {
                self.set_texture(ctx, &canvas);
                self.current_canvas = Some(canvas.clone());
                self.target = Some(canvas);
                self.source_dims = Some(dims);
                self.source_name = name;
                self.final_svg = None;
                self.last = None;
                self.run = None;
                self.phase = Phase::Ready;
            }
            Err(e) => self.toast(e),
        }
    }

    fn apply(&mut self, ctx: &egui::Context, action: Action) {
        match action {
            Action::LoadPath(p) => {
                let name = p
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                self.load(ctx, image_io::load_path(&p), name);
            }
            Action::LoadSample(i) => {
                let (name, bytes) = image_io::SAMPLES[i];
                self.load(ctx, image_io::load_bytes(bytes), name.to_string());
            }
            Action::Start => {
                if let Some(target) = &self.target {
                    self.final_svg = None;
                    self.run = Some(runner::start(RunConfig {
                        target: target.clone(),
                        count: self.count as usize,
                        alpha: self.alpha,
                        seed: 1,
                    }));
                    self.phase = Phase::Running;
                }
            }
            Action::Pause => {
                if let Some(r) = &self.run {
                    r.set(PAUSE);
                }
                self.phase = Phase::Paused;
            }
            Action::Resume => {
                if let Some(r) = &self.run {
                    r.set(RUN);
                }
                self.phase = Phase::Running;
            }
            Action::Reset => {
                if let Some(r) = &self.run {
                    r.set(STOP);
                }
                self.run = None;
                self.last = None;
                self.final_svg = None;
                if let Some(t) = self.target.clone() {
                    self.set_texture(ctx, &t);
                    self.current_canvas = Some(t);
                    self.phase = Phase::Ready;
                } else {
                    self.phase = Phase::Empty;
                }
            }
            Action::ExportPng => self.export_png(),
            Action::ExportSvg => self.export_svg(),
        }
    }

    fn export_png(&mut self) {
        let Some(canvas) = &self.current_canvas else {
            return;
        };
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name(format!("{}.png", self.stem()))
            .add_filter("PNG", &["png"])
            .save_file()
        {
            match image_io::export_png(canvas, &path) {
                Ok(()) => self.toast(format!(
                    "Saved {}",
                    path.file_name().unwrap_or_default().to_string_lossy()
                )),
                Err(e) => self.toast(e),
            }
        }
    }

    fn export_svg(&mut self) {
        let Some(svg) = self.final_svg.clone() else {
            return;
        };
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name(format!("{}.svg", self.stem()))
            .add_filter("SVG", &["svg"])
            .save_file()
        {
            match image_io::export_svg(&svg, &path) {
                Ok(()) => self.toast(format!(
                    "Saved {}",
                    path.file_name().unwrap_or_default().to_string_lossy()
                )),
                Err(e) => self.toast(e),
            }
        }
    }

    fn stem(&self) -> String {
        let s = self
            .source_name
            .rsplit_once('.')
            .map(|(a, _)| a)
            .unwrap_or(&self.source_name);
        if s.is_empty() {
            "primitive".into()
        } else {
            s.into()
        }
    }

    /// Drain the run channel to the latest frame; refresh the canvas + progress.
    fn poll_run(&mut self, ctx: &egui::Context) {
        let mut latest = None;
        let mut finished = false;
        if let Some(run) = &self.run {
            while let Ok(frame) = run.rx.try_recv() {
                if frame.done {
                    finished = true;
                    self.final_svg = frame.svg.clone();
                }
                latest = Some(frame);
            }
        }
        if let Some(frame) = latest {
            self.set_texture(ctx, &frame.canvas);
            self.current_canvas = Some(frame.canvas);
            self.last = Some(Progress {
                shape_index: frame.shape_index,
                total: frame.total,
                score: frame.score,
                sps: frame.shapes_per_sec,
            });
        }
        if finished && self.phase == Phase::Running {
            self.phase = Phase::Done;
        }
        if self.phase == Phase::Running {
            ctx.request_repaint();
        }
    }

    fn dropped_path(&self, ctx: &egui::Context) -> Option<PathBuf> {
        ctx.input(|i| i.raw.dropped_files.first().and_then(|f| f.path.clone()))
    }
}

impl eframe::App for PrimitiveApp {
    // eframe 0.34 made `ui(&mut self, ui, frame)` the required method; full-window panels are shown
    // *inside* the given root `ui` via `show_inside`. The Context (repaint/input/textures) is `ui.ctx()`.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_run(&ctx);
        if let Some(path) = self.dropped_path(&ctx) {
            self.apply(&ctx, Action::LoadPath(path));
        }
        if let Some((_, t)) = &self.toast {
            if t.elapsed().as_secs_f32() > 4.0 {
                self.toast = None;
            }
        }

        let chip = self
            .run
            .as_ref()
            .map(|r| r.backend.chip_label())
            .unwrap_or("CPU");
        egui::Panel::top("title").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("primitive");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(chip).monospace().weak())
                        .on_hover_text("Live preview runs on the CPU adapter (the parity oracle). GPU 'instant' mode is a later increment.");
                });
            });
        });

        let mut action = None;
        egui::Panel::right("controls")
            .exact_size(320.0)
            .resizable(false)
            .show_inside(ui, |ui| action = sidebar::show(self, ui));
        if let Some(a) = action {
            self.apply(&ctx, a);
        }

        egui::CentralPanel::default().show_inside(ui, |ui| self.canvas(ui));
    }
}

impl PrimitiveApp {
    /// The hero region: the live reconstruction, aspect-fit, with the progress strip beneath.
    fn canvas(&mut self, ui: &mut egui::Ui) {
        if let Some((msg, _)) = &self.toast {
            ui.colored_label(ui.visuals().warn_fg_color, msg);
        }
        let Some(tex) = &self.texture else {
            ui.centered_and_justified(|ui| {
                ui.label(
                    egui::RichText::new("Drop an image to begin")
                        .size(18.0)
                        .weak(),
                );
            });
            return;
        };
        let (cw, ch) = self
            .target
            .as_ref()
            .map(|c| (c.w as f32, c.h as f32))
            .unwrap_or((1.0, 1.0));
        let avail = ui.available_size();
        let scale = ((avail.x) / cw).min((avail.y - 46.0) / ch).max(0.01);
        let size = egui::vec2(cw * scale, ch * scale);
        ui.vertical_centered(|ui| {
            ui.add(egui::Image::new(egui::load::SizedTexture::new(
                tex.id(),
                size,
            )));
            if let Some(p) = &self.last {
                let frac = p.shape_index as f32 / p.total.max(1) as f32;
                ui.add(
                    egui::ProgressBar::new(frac)
                        .desired_width(size.x)
                        .show_percentage(),
                );
                let psnr = primitive_core::psnr_from_score(p.score);
                ui.label(
                    egui::RichText::new(format!(
                        "{}/{}  ·  {:.0} shapes/sec  ·  {:.1} dB",
                        p.shape_index, p.total, p.sps, psnr
                    ))
                    .monospace()
                    .weak(),
                );
            }
        });
    }
}
