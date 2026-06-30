//! Background optimizer thread. The engine runs off the UI thread, streaming a canvas snapshot +
//! progress after each committed shape; the UI drains to the latest frame each repaint (plan §5A:
//! live ≥ 30 fps spectacle). Control (run/pause/stop) is a shared atomic the loop polls per shape.
//!
//! Two backends behind one handle (plan §5A device row):
//! - **CPU streaming** (the live, watchable parity oracle) — every shape streams a frame.
//! - **GPU instant** (`gpu_optimize`, the whole loop on-device) — finishes in one shot and streams a
//!   single final raster frame (no per-shape spectacle, no SVG; the GPU path is raster-only).

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use primitive_core::{Canvas, Rng, SearchBudget, ShapeType};
use primitive_engine::Engine;
use primitive_gpu_cpu::CpuSearch;

use crate::image_io;
use crate::state::Device;

pub const RUN: u8 = 0;
pub const PAUSE: u8 = 1;
pub const STOP: u8 = 2;

/// GPU "instant" mode caps the working canvas to this side length — the i32-accumulator bound the
/// kernels are proven correct within (`.agent/EVIDENCE.md`).
const GPU_INSTANT_MAX: u32 = 128;

/// One streamed step: the current reconstruction + progress. `svg`/`done` are set on the last frame.
pub struct Frame {
    pub canvas: Canvas,
    pub shape_index: usize,
    pub total: usize,
    pub score: f64,
    pub shapes_per_sec: f64,
    pub done: bool,
    /// Vector output, only attached to the final (`done`) frame of a **CPU** run (the GPU path is
    /// raster-only, so its done frame carries `None`).
    pub svg: Option<String>,
}

/// Live config for a run (core controls + the Advanced power-user knobs + the chosen backend).
pub struct RunConfig {
    pub target: Canvas,
    pub shape_type: ShapeType,
    pub count: usize,
    pub alpha: i32,
    pub seed: u64,
    /// fogleman's `(n, age, m)` budget — exposed in the Advanced disclosure.
    pub n: i32,
    pub age: i32,
    pub m: i32,
    pub device: Device,
}

/// Handle the UI holds while a run is in flight.
pub struct RunHandle {
    pub rx: Receiver<Frame>,
    pub control: Arc<AtomicU8>,
}

impl RunHandle {
    pub fn set(&self, state: u8) {
        self.control.store(state, Ordering::Relaxed);
    }
}

/// Whether a run takes the GPU "instant" path: a GPU device **and** a triangle. GPU instant mode is
/// triangle-only until CORE-3b generalizes the kernels, so ellipse/rect always stream on the CPU
/// adapter (which supports every shape type). Pure + hardware-independent so the routing guard is
/// unit-testable on any runner — not via a GPU-dependent observable a CPU fallback could mask.
fn should_use_gpu(device: Device, shape: ShapeType) -> bool {
    matches!(device, Device::Metal | Device::Cuda) && shape == ShapeType::Triangle
}

/// Spawn the optimizer thread (CPU streaming or GPU instant, per `cfg.device`) and return a handle.
pub fn start(cfg: RunConfig) -> RunHandle {
    let (tx, rx) = channel::<Frame>();
    let control = Arc::new(AtomicU8::new(RUN));
    let ctrl = control.clone();
    let gpu = should_use_gpu(cfg.device, cfg.shape_type);
    thread::spawn(move || {
        if gpu {
            gpu_instant(cfg, &tx);
        } else {
            cpu_stream(cfg, &tx, &ctrl);
        }
    });
    RunHandle { rx, control }
}

/// The CPU adapter loop: one committed shape → one streamed frame (the spectacle + parity oracle).
fn cpu_stream(cfg: RunConfig, tx: &Sender<Frame>, control: &Arc<AtomicU8>) {
    let (w, h) = (cfg.target.w as i32, cfg.target.h as i32);
    let mut engine = Engine::with_average_background(cfg.target, CpuSearch::new(w, h));
    let mut budget = SearchBudget::triangles_default();
    budget.alpha = cfg.alpha;
    budget.shape_type = cfg.shape_type;
    budget.n = cfg.n;
    budget.age = cfg.age;
    budget.m = cfg.m;
    let mut rng = Rng::new(cfg.seed);
    let start = Instant::now();

    for i in 0..cfg.count {
        loop {
            match control.load(Ordering::Relaxed) {
                STOP => return,
                PAUSE => thread::sleep(Duration::from_millis(40)),
                _ => break,
            }
        }
        let p = engine.step(&budget, &mut rng);
        let elapsed = start.elapsed().as_secs_f64().max(1e-6);
        if tx
            .send(Frame {
                canvas: engine.model.current.clone(),
                shape_index: p.shape_index,
                total: cfg.count,
                score: p.score,
                shapes_per_sec: (i + 1) as f64 / elapsed,
                done: false,
                svg: None,
            })
            .is_err()
        {
            return; // UI dropped the receiver (window closed / reset) — stop quietly.
        }
    }

    let _ = tx.send(Frame {
        canvas: engine.model.current.clone(),
        shape_index: cfg.count,
        total: cfg.count,
        score: engine.model.score,
        shapes_per_sec: 0.0,
        done: true,
        svg: Some(engine.model.svg()),
    });
}

/// The GPU adapter "instant" path: `gpu_optimize` runs the whole loop on-device, then we stream one
/// final raster frame. Defensive — any GPU fault falls back to a CPU stream so the app never dies.
fn gpu_instant(cfg: RunConfig, tx: &Sender<Frame>) {
    use primitive_gpu_cubecl::{gpu_optimize, OptConfig};

    let small = image_io::downscale(&cfg.target, GPU_INSTANT_MAX);
    let opt = OptConfig {
        workers: 8192,
        age: cfg.age.max(1) as u32,
        shapes: cfg.count,
        alpha: cfg.alpha,
        seed: cfg.seed as u32,
    };
    let start = Instant::now();
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu_optimize(&small, &opt)));
    match result {
        Ok(canvas) => {
            let elapsed = start.elapsed().as_secs_f64().max(1e-6);
            let _ = tx.send(Frame {
                canvas,
                shape_index: cfg.count,
                total: cfg.count,
                score: f64::NAN, // PSNR readout is a CPU-path detail; the GPU path is a fast preview.
                shapes_per_sec: cfg.count as f64 / elapsed,
                done: true,
                svg: None, // raster-only path.
            });
        }
        // GPU faulted mid-run — degrade to the CPU oracle rather than leave a dead run.
        Err(_) => {
            let ctrl = Arc::new(AtomicU8::new(RUN));
            cpu_stream(cfg, tx, &ctrl);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_io;

    /// Co-located fast smoke: a tiny CPU run streams to a `done` frame carrying an SVG. The full
    /// load→100-shapes→export→xmllint gate lives in tests/e2e.rs.
    #[test]
    fn cpu_stream_reaches_done_with_svg() {
        let (_, target) = image_io::load_bytes(image_io::SAMPLES[0].1).expect("sample decodes");
        let target = image_io::downscale(&target, 48); // tiny → fast in debug
        let handle = start(RunConfig {
            target,
            shape_type: ShapeType::Triangle,
            count: 8,
            alpha: 128,
            seed: 1,
            n: 100,
            age: 20,
            m: 8,
            device: Device::CpuFallback,
        });
        let svg = loop {
            let f = handle.rx.recv().expect("frames stream until done");
            if f.done {
                break f.svg;
            }
        };
        let svg = svg.expect("CPU done frame carries the SVG");
        assert!(svg.contains("<svg") && svg.contains("polygon"));
    }

    /// CORE-3c guard: GPU instant mode runs only for a triangle on a GPU device; every other
    /// combination streams on the CPU adapter. Tests the pure routing decision directly, so deleting
    /// the guard fails this on ANY runner (an observable like `<ellipse>` in the output can be masked
    /// by the `catch_unwind → cpu_stream` fallback on a GPU-less machine). The full ellipse plumbing
    /// is covered end-to-end by `tests/e2e.rs::ellipse_run_exports_ellipse_svg`.
    #[test]
    fn gpu_routing_is_triangle_and_gpu_device_only() {
        assert!(should_use_gpu(Device::Metal, ShapeType::Triangle));
        assert!(should_use_gpu(Device::Cuda, ShapeType::Triangle));
        assert!(!should_use_gpu(Device::Metal, ShapeType::Ellipse));
        assert!(!should_use_gpu(Device::Metal, ShapeType::Rectangle));
        assert!(!should_use_gpu(Device::Cuda, ShapeType::Ellipse));
        assert!(!should_use_gpu(Device::CpuFallback, ShapeType::Triangle));
    }
}
