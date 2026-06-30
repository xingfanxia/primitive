//! Pure §5A interaction-state model (plan architecture.md lines 246-259). **No egui, no IO** — it
//! maps the app's observable facts ([`UiFacts`]) to what each surface should show/enable. The render
//! layer ([`crate::app`] / [`crate::sidebar`]) reads these decisions; it never re-derives them. Keeping
//! it pure is what makes the whole interaction table unit-testable headless (the GUI-2 gate).

/// Where the single-document flow is (drives which controls are live — §5A interaction table).
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Phase {
    /// No image loaded yet — the canvas is the drop-zone.
    Empty,
    /// An image is loaded; idle, ready to Start.
    Ready,
    /// A run is in flight, painting live.
    Running,
    /// A run is paused (resumable).
    Paused,
    /// A run finished; the final reconstruction holds on the canvas.
    Done,
}

/// Which compute backend the device probe found (drives the §5A device chip). The GPU path runs the
/// whole loop "instantly" via `gpu_optimize`; the CPU path is the live, watchable parity oracle.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Device {
    /// Apple GPU (Metal) — green chip.
    Metal,
    /// NVIDIA GPU (CUDA) — green chip.
    Cuda,
    /// CPU chosen deliberately (e.g. the live oracle), a GPU *was* available — neutral chip.
    Cpu,
    /// No compatible GPU found → graceful CPU fallback — amber chip (§5A first-class state).
    CpuFallback,
}

impl Device {
    /// The chip text (`Metal` / `CUDA` / `CPU` / `CPU (no GPU found)`).
    pub fn chip_label(self) -> &'static str {
        match self {
            Device::Metal => "Metal",
            Device::Cuda => "CUDA",
            Device::Cpu => "CPU",
            Device::CpuFallback => "CPU (no GPU found)",
        }
    }

    /// Visual emphasis of the chip — the renderer maps this to a token colour.
    pub fn chip_kind(self) -> ChipKind {
        match self {
            Device::Metal | Device::Cuda => ChipKind::Green,
            Device::Cpu => ChipKind::Neutral,
            Device::CpuFallback => ChipKind::Amber,
        }
    }
}

/// Emphasis class for the device chip (the renderer picks the token colour for each).
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum ChipKind {
    /// Active GPU backend.
    Green,
    /// Deliberate CPU (a GPU exists).
    Neutral,
    /// CPU because no GPU was found — the amber "first-class fallback" state.
    Amber,
}

/// The observable facts the app feeds the state model each frame.
#[derive(Clone, Copy, Debug)]
pub struct UiFacts {
    pub phase: Phase,
    /// A target image is loaded.
    pub has_image: bool,
    /// A load/decode/export error toast is currently active (§5A error column).
    pub has_error: bool,
    /// How many shapes have been committed so far (Export needs ≥ 1).
    pub shapes_committed: usize,
    /// The vector (SVG) output is ready (only on a finished CPU run).
    pub svg_ready: bool,
    /// The backend the device probe found.
    pub device: Device,
    /// System / user "Reduce Motion" is on (§5A a11y: cap the live repaint pulse).
    pub reduce_motion: bool,
}

/// What the hero canvas shows (§5A "Canvas" row).
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum CanvasView {
    /// Soft dashed drop-zone + sample strip ("Drop an image to begin").
    DropZone,
    /// Decode/format failure → inline toast, canvas stays in the drop state.
    Error,
    /// Image loaded, idle — staged on the canvas, ready to Start (the "Ready" between empty & live).
    Staged,
    /// Repaints live with the progress strip beneath.
    Live,
    /// Final frame holds, "done" emphasis, shape count + PSNR.
    Done,
}

/// What the source panel shows (§5A "Source panel" row).
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum SourceView {
    /// The drop-zone (no image yet).
    DropZone,
    /// Thumbnail + dimensions (idle/done).
    Loaded,
    /// Loaded but dimmed — can't swap mid-run (Reset to change).
    Dimmed,
}

/// The primary action button's identity (§5A Start ⇄ Pause toggle).
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Primary {
    Start,
    Pause,
    Resume,
}

/// The fully-derived action-area state (Start/Pause, Reset, Export menu).
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct Actions {
    pub primary: Primary,
    /// Start is enabled only once an image is loaded; Pause/Resume always actionable mid-run.
    pub primary_enabled: bool,
    pub reset_enabled: bool,
    /// PNG/GIF need ≥ 1 committed shape and a not-actively-painting run.
    pub export_png_enabled: bool,
    pub export_gif_enabled: bool,
    /// SVG needs a finished run (vector output only exists when done).
    pub export_svg_enabled: bool,
}

/// The whole §5A interaction state, derived once per frame from [`UiFacts`].
#[derive(Clone, Copy, Debug)]
pub struct Ui {
    pub canvas: CanvasView,
    pub source: SourceView,
    /// Core controls (type/count/alpha) are editable only when NOT mid-run (§5A "Controls" row).
    pub controls_editable: bool,
    pub actions: Actions,
    pub device_label: &'static str,
    pub device_kind: ChipKind,
    /// The decorative live/done repaint pulse — off under Reduce Motion (§5A a11y).
    pub motion_pulse: bool,
}

impl Ui {
    /// Derive the full interaction state. This is the single source of truth for every §5A cell.
    pub fn derive(f: UiFacts) -> Ui {
        let running = matches!(f.phase, Phase::Running | Phase::Paused);
        let has_shape = f.shapes_committed >= 1;

        let canvas = if !f.has_image {
            if f.has_error {
                CanvasView::Error
            } else {
                CanvasView::DropZone
            }
        } else {
            match f.phase {
                Phase::Done => CanvasView::Done,
                Phase::Running | Phase::Paused => CanvasView::Live,
                // Idle with an error toast still surfaces the error overlay over the source.
                _ if f.has_error => CanvasView::Error,
                _ => CanvasView::Staged, // image loaded, idle — staged and ready to Start
            }
        };

        let source = if !f.has_image {
            SourceView::DropZone
        } else if running {
            SourceView::Dimmed
        } else {
            SourceView::Loaded
        };

        let primary = match f.phase {
            Phase::Running => Primary::Pause,
            Phase::Paused => Primary::Resume,
            _ => Primary::Start,
        };
        let primary_enabled = match primary {
            Primary::Start => f.has_image,
            Primary::Pause | Primary::Resume => true,
        };
        // Reset is meaningful whenever a run exists or has finished (something to clear).
        let reset_enabled = running || f.phase == Phase::Done;
        // Raster exports need a committed shape and a canvas that isn't mid-repaint.
        let raster_ready = has_shape && f.phase != Phase::Running;

        let actions = Actions {
            primary,
            primary_enabled,
            reset_enabled,
            export_png_enabled: raster_ready,
            export_gif_enabled: raster_ready,
            export_svg_enabled: f.svg_ready,
        };

        Ui {
            canvas,
            source,
            controls_editable: !running,
            actions,
            device_label: f.device.chip_label(),
            device_kind: f.device.chip_kind(),
            motion_pulse: !f.reduce_motion,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // Co-located fast sanity; the full §5A surface×state matrix lives in tests/state_suite.rs
    // (one #[test] per cell, so the gate's "pass count covers every row" is legible in the output).
    #[test]
    fn empty_first_run_is_dropzone_and_start_gated() {
        let ui = Ui::derive(facts());
        assert_eq!(ui.canvas, CanvasView::DropZone);
        assert_eq!(ui.source, SourceView::DropZone);
        assert!(
            !ui.actions.primary_enabled,
            "Start gated until an image loads"
        );
        assert!(ui.controls_editable);
    }
}
