//! GUI-2 gate 3 (plan §5A device row): with `PRIMITIVE_FORCE_CPU=1` the app degrades gracefully —
//! the device probe reports the amber `CPU (no GPU found)` chip and a full 100-shape run completes
//! on the CPU adapter. Run with `--nocapture` so the printed backend label lands in the transcript.
//!
//! Invoke: `PRIMITIVE_FORCE_CPU=1 cargo test -p primitive-app --release --test forced_cpu -- --nocapture`

use primitive_app::device;
use primitive_app::image_io;
use primitive_app::runner::{start, RunConfig};
use primitive_core::ShapeType;

#[test]
fn forced_cpu_probe_labels_and_runs_100_shapes() {
    // Hermetic: set the override in-process so the gate is the same whether or not the caller
    // exported it (this is the exact code path `PRIMITIVE_FORCE_CPU=1` triggers — simulate a
    // no-GPU machine). SAFETY: this integration binary has a single test, no concurrent env reads.
    unsafe { std::env::set_var(device::FORCE_CPU_ENV, "1") };

    let device = device::detect();
    let label = device.chip_label();
    println!("BACKEND_LABEL={label}");
    assert_eq!(
        label, "CPU (no GPU found)",
        "forced-CPU shows the amber fallback chip"
    );

    // …and the app still works: a full 100-shape run completes on the CPU adapter.
    let (_, target) = image_io::load_bytes(image_io::SAMPLES[0].1).expect("sample decodes");
    let target = image_io::downscale(&target, 64);
    let handle = start(RunConfig {
        target,
        shape_type: ShapeType::Triangle,
        count: 100,
        alpha: 128,
        seed: 1,
        n: 300,
        age: 40,
        m: 8,
        device,
    });
    let last = loop {
        let f = handle.rx.recv().expect("frames stream until done");
        if f.done {
            break f.shape_index;
        }
    };
    assert_eq!(last, 100, "forced-CPU run completes all 100 shapes");
    println!("FORCED_CPU_RUN=ok 100/100 shapes on {label}");
}
