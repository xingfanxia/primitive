//! Background optimizer thread. The engine runs off the UI thread, streaming a canvas snapshot +
//! progress after each committed shape; the UI drains to the latest frame each repaint (plan §5A:
//! live ≥30 fps spectacle). Control (run/pause/stop) is a shared atomic the loop polls per shape.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use primitive_compute::Backend;
use primitive_core::{Canvas, Rng, SearchBudget, ShapeType};
use primitive_engine::Engine;
use primitive_gpu_cpu::CpuSearch;

pub const RUN: u8 = 0;
pub const PAUSE: u8 = 1;
pub const STOP: u8 = 2;

/// One streamed step: the current reconstruction + progress. `svg`/`done` are set on the last frame.
pub struct Frame {
    pub canvas: Canvas,
    pub shape_index: usize,
    pub total: usize,
    pub score: f64,
    pub shapes_per_sec: f64,
    pub done: bool,
    /// Vector output, only attached to the final (`done`) frame (SVG export is enabled when done).
    pub svg: Option<String>,
}

/// Live config for a run.
pub struct RunConfig {
    pub target: Canvas,
    pub count: usize,
    pub alpha: i32,
    pub seed: u64,
}

/// Handle the UI holds while a run is in flight.
pub struct RunHandle {
    pub rx: Receiver<Frame>,
    pub control: Arc<AtomicU8>,
    pub backend: Backend,
}

impl RunHandle {
    pub fn set(&self, state: u8) {
        self.control.store(state, Ordering::Relaxed);
    }
}

/// Spawn the optimizer thread and return a handle the UI polls.
pub fn start(cfg: RunConfig) -> RunHandle {
    let (tx, rx) = channel::<Frame>();
    let control = Arc::new(AtomicU8::new(RUN));
    let ctrl = control.clone();
    thread::spawn(move || run_loop(cfg, &tx, &ctrl));
    // GUI-1 drives the CPU adapter (the live, watchable backend + parity oracle). A GPU "instant"
    // mode (gpu_optimize) is a later increment — the GPU finishes 250 shapes in <1 s, no spectacle.
    RunHandle {
        rx,
        control,
        backend: Backend::Cpu,
    }
}

fn run_loop(cfg: RunConfig, tx: &Sender<Frame>, control: &Arc<AtomicU8>) {
    let (w, h) = (cfg.target.w as i32, cfg.target.h as i32);
    let mut engine = Engine::with_average_background(cfg.target, CpuSearch::new(w, h));
    let mut budget = SearchBudget::triangles_default();
    budget.alpha = cfg.alpha;
    budget.shape_type = ShapeType::Triangle;
    let mut rng = Rng::new(cfg.seed);
    let start = Instant::now();

    for i in 0..cfg.count {
        // Poll control: park while paused, bail on stop.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_io;

    /// The machine-checkable GUI gate (plan §5A): decode a bundled image → run shapes → export SVG.
    /// Exercises image_io + runner + engine end-to-end, headless (no window).
    #[test]
    fn smoke_load_run_export_svg() {
        let (_, target) = image_io::load_bytes(image_io::SAMPLES[0].1).expect("sample decodes");
        let handle = start(RunConfig {
            target,
            count: 25,
            alpha: 128,
            seed: 1,
        });

        let (svg, last_index) = loop {
            let f = handle.rx.recv().expect("frames stream until done");
            if f.done {
                break (f.svg, f.shape_index);
            }
        };

        assert_eq!(last_index, 25, "ran all requested shapes");
        let svg = svg.expect("the done frame carries the SVG");
        assert!(svg.contains("<svg"), "valid SVG root element");
        assert!(
            svg.contains("polygon"),
            "SVG contains the committed triangles"
        );
    }
}
