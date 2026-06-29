//! App state + frame polling + the hero canvas (plan §5A). Every enable/disable/visibility decision
//! comes from the **pure** [`crate::state`] model — this module only *renders* it and turns input
//! (clicks, drops, keys) into [`Action`]s. Controls live in [`crate::sidebar`], IO in
//! [`crate::image_io`], the optimizer thread in [`crate::runner`], colours in [`crate::theme`].

use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;
use primitive_core::Canvas;
use serde::{Deserialize, Serialize};

use crate::device;
use crate::hero;
use crate::i18n::Strings;
use crate::image_io;
use crate::runner::{self, RunConfig, RunHandle, PAUSE, RUN, STOP};
use crate::sidebar;
use crate::state::{Device, Phase, Primary, Ui, UiFacts};

/// User-tunable run parameters — persisted across launches (§5A "remember last-used params").
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Params {
    pub count: i32,
    pub alpha: i32,
    pub seed: u64,
    pub n: i32,
    pub age: i32,
    pub m: i32,
    pub reduce_motion: bool,
}

impl Default for Params {
    fn default() -> Self {
        // fogleman's defaults (triangle / 250 / alpha 128 / n=1000 age=100 m=16).
        Self {
            count: 250,
            alpha: 128,
            seed: 1,
            n: 1000,
            age: 100,
            m: 16,
            reduce_motion: false,
        }
    }
}

/// Latest streamed progress (drives the canvas readout + Export gating via `shape_index`).
pub struct Progress {
    pub shape_index: usize,
    pub total: usize,
    pub score: f64,
    pub sps: f64,
}

/// A UI intent raised by the sidebar/keyboard, applied by [`PrimitiveApp::apply`].
pub enum Action {
    OpenFile,
    LoadPath(PathBuf),
    LoadSample(usize),
    Start,
    Pause,
    Resume,
    Reset,
    ExportPng,
    ExportSvg,
    ExportGif,
}

const STORAGE_KEY: &str = "primitive_params";

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
    /// The active toast is an *error* (drives the §5A error column vs a neutral "Saved" toast).
    pub error_toast: bool,
    /// Downscaled keyframes captured during a run, for the GIF export (§5A Export ▾).
    pub gif_frames: Vec<Canvas>,
    pub params: Params,
    pub show_advanced: bool,
    /// The backend the launch probe found (drives the device chip — §5A device row).
    pub device: Device,
    pub strings: Strings,
}

impl PrimitiveApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let params = cc
            .storage
            .and_then(|s| eframe::get_value::<Params>(s, STORAGE_KEY))
            .unwrap_or_default();
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
            error_toast: false,
            gif_frames: Vec::new(),
            params,
            show_advanced: false,
            device: device::detect(),
            strings: Strings::en(),
        }
    }

    /// The observable facts the pure [`Ui`] model derives every surface from.
    pub fn facts(&self) -> UiFacts {
        UiFacts {
            phase: self.phase,
            has_image: self.target.is_some(),
            has_error: self.error_toast && self.toast.is_some(),
            shapes_committed: self.last.as_ref().map(|p| p.shape_index).unwrap_or(0),
            svg_ready: self.final_svg.is_some(),
            device: self.device,
            reduce_motion: self.params.reduce_motion,
        }
    }

    fn toast(&mut self, msg: impl Into<String>, is_error: bool) {
        self.toast = Some((msg.into(), Instant::now()));
        self.error_toast = is_error;
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
                self.gif_frames.clear();
                self.toast = None;
                self.error_toast = false;
                self.phase = Phase::Ready;
            }
            Err(e) => self.toast(e, true),
        }
    }

    fn apply(&mut self, ctx: &egui::Context, action: Action) {
        match action {
            Action::OpenFile => {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Image", &["png", "jpg", "jpeg", "webp", "bmp", "gif"])
                    .pick_file()
                {
                    self.apply(ctx, Action::LoadPath(path));
                }
            }
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
                    self.gif_frames.clear();
                    self.run = Some(runner::start(RunConfig {
                        target: target.clone(),
                        count: self.params.count as usize,
                        alpha: self.params.alpha,
                        seed: self.params.seed,
                        n: self.params.n,
                        age: self.params.age,
                        m: self.params.m,
                        device: self.device,
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
                self.gif_frames.clear();
                if let Some(t) = self.target.clone() {
                    self.set_texture(ctx, &t);
                    self.current_canvas = Some(t);
                    self.phase = Phase::Ready;
                } else {
                    self.phase = Phase::Empty;
                }
            }
            Action::ExportPng => self.export_raster(false),
            Action::ExportGif => self.export_raster(true),
            Action::ExportSvg => self.export_svg(),
        }
    }

    fn export_raster(&mut self, gif: bool) {
        let ext = if gif { "gif" } else { "png" };
        let Some(path) = rfd::FileDialog::new()
            .set_file_name(format!("{}.{ext}", self.stem()))
            .add_filter(ext.to_uppercase(), &[ext])
            .save_file()
        else {
            return;
        };
        let res = if gif {
            image_io::export_gif(&self.gif_frames, 60, &path)
        } else if let Some(c) = &self.current_canvas {
            image_io::export_png(c, &path)
        } else {
            return;
        };
        self.report_save(res, &path);
    }

    fn export_svg(&mut self) {
        let Some(svg) = self.final_svg.clone() else {
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .set_file_name(format!("{}.svg", self.stem()))
            .add_filter("SVG", &["svg"])
            .save_file()
        else {
            return;
        };
        let res = image_io::export_svg(&svg, &path);
        self.report_save(res, &path);
    }

    fn report_save(&mut self, res: Result<(), String>, path: &std::path::Path) {
        match res {
            Ok(()) => {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                self.toast(format!("{} {name}", self.strings.saved), false);
            }
            Err(e) => self.toast(e, true),
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

    /// Drain the run channel to the latest frame; refresh the canvas + progress; capture GIF keyframes.
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
            self.capture_gif_frame(&frame.canvas, frame.shape_index, frame.total, finished);
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
        // Keep draining while a run is live; honor Reduce Motion by not adding a decorative pulse.
        if self.phase == Phase::Running {
            ctx.request_repaint();
        }
    }

    /// Keep ~80 evenly-spaced keyframes (plus the final) for a smooth, bounded GIF.
    fn capture_gif_frame(&mut self, canvas: &Canvas, idx: usize, total: usize, done: bool) {
        let stride = (total / 80).max(1);
        if done || idx % stride == 0 {
            self.gif_frames.push(image_io::downscale(canvas, 240));
        }
    }

    fn dropped_path(&self, ctx: &egui::Context) -> Option<PathBuf> {
        ctx.input(|i| i.raw.dropped_files.first().and_then(|f| f.path.clone()))
    }

    /// Map the §5A keyboard shortcuts (⌘O / Space / ⌘E / ⌘R / ⌘,) to actions.
    fn keyboard(&mut self, ctx: &egui::Context, ui: &Ui) -> Option<Action> {
        use egui::{Key, Modifiers};
        let cmd = Modifiers::COMMAND;
        let mut out = None;
        ctx.input_mut(|i| {
            if i.consume_key(cmd, Key::O) {
                out = Some(Action::OpenFile);
            } else if i.consume_key(cmd, Key::E) && ui.actions.export_png_enabled {
                out = Some(Action::ExportPng);
            } else if i.consume_key(cmd, Key::R) && ui.actions.reset_enabled {
                out = Some(Action::Reset);
            } else if i.consume_key(cmd, Key::Comma) {
                self.show_advanced = !self.show_advanced;
            } else if i.consume_key(Modifiers::NONE, Key::Space) && ui.actions.primary_enabled {
                out = Some(match ui.actions.primary {
                    Primary::Start => Action::Start,
                    Primary::Pause => Action::Pause,
                    Primary::Resume => Action::Resume,
                });
            }
        });
        out
    }
}

impl eframe::App for PrimitiveApp {
    // eframe 0.34 made `ui(&mut self, ui, frame)` the required method; full-window panels are shown
    // *inside* the given root `ui` via `show_inside`. The Context is `ui.ctx()`.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_run(&ctx);
        if let Some(path) = self.dropped_path(&ctx) {
            self.apply(&ctx, Action::LoadPath(path));
        }
        if let Some((_, t)) = &self.toast {
            if t.elapsed().as_secs_f32() > 4.0 {
                self.toast = None;
                self.error_toast = false;
            }
        }

        let state = Ui::derive(self.facts());
        if let Some(a) = self.keyboard(&ctx, &state) {
            self.apply(&ctx, a);
        }

        egui::Panel::top("title").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading(self.strings.app_title);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    hero::device_chip(self, ui, &state);
                });
            });
        });

        let mut action = None;
        egui::Panel::right("controls")
            .exact_size(320.0)
            .resizable(false)
            .show_inside(ui, |ui| action = sidebar::show(self, &state, ui));
        if let Some(a) = action {
            self.apply(&ctx, a);
        }

        egui::CentralPanel::default().show_inside(ui, |ui| hero::canvas(self, ui, &state));
    }

    /// Persist the run params (window geometry is persisted by eframe's `persist_window`).
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, STORAGE_KEY, &self.params);
    }
}
