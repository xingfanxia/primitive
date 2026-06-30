//! GUI-2 gate 2 (plan §5A): the scripted end-to-end — load a bundled sample → run 100 shapes →
//! export an SVG — exercised headless through the real adapters (`image_io` + `runner` + engine).
//! The test writes the SVG to disk and self-validates; the gate then runs `xmllint --noout` on the
//! same file. Override the output path with `PRIMITIVE_E2E_OUT` (defaults to the system temp dir).

use std::path::PathBuf;

use primitive_app::image_io;
use primitive_app::runner::{start, RunConfig};
use primitive_app::state::Device;
use primitive_core::ShapeType;

fn out_path() -> PathBuf {
    std::env::var_os("PRIMITIVE_E2E_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("primitive_e2e.svg"))
}

#[test]
fn load_sample_run_100_export_svg() {
    // Load a bundled sample and downscale for a fast, deterministic CPU run.
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
        device: Device::CpuFallback, // CPU path → streamed shapes + an SVG on the done frame
    });

    let (svg, last_index) = loop {
        let f = handle.rx.recv().expect("frames stream until done");
        if f.done {
            break (f.svg, f.shape_index);
        }
    };

    assert_eq!(last_index, 100, "ran all 100 requested shapes");
    let svg = svg.expect("the CPU done frame carries the SVG");
    assert!(svg.contains("<svg"), "valid SVG root element");
    assert!(
        svg.contains("polygon"),
        "SVG carries the committed triangles"
    );

    let path = out_path();
    image_io::export_svg(&svg, &path).expect("SVG writes to disk");
    // Print the absolute path so the gate's `xmllint --noout <path>` can find it.
    println!("E2E_SVG_PATH={}", path.display());
}

/// CORE-3c: a non-triangle shape selected in the UI runs end-to-end through `runner` and exports
/// the matching SVG primitive (proves `params.shape_type` is plumbed Sidebar → RunConfig → engine).
#[test]
fn ellipse_run_exports_ellipse_svg() {
    let (_, target) = image_io::load_bytes(image_io::SAMPLES[0].1).expect("sample decodes");
    let target = image_io::downscale(&target, 64);

    let handle = start(RunConfig {
        target,
        shape_type: ShapeType::Ellipse,
        count: 40,
        alpha: 128,
        seed: 1,
        n: 200,
        age: 30,
        m: 8,
        device: Device::CpuFallback,
    });

    let (svg, last_index) = loop {
        let f = handle.rx.recv().expect("frames stream until done");
        if f.done {
            break (f.svg, f.shape_index);
        }
    };

    assert_eq!(last_index, 40, "ran all 40 requested ellipses");
    let svg = svg.expect("the CPU done frame carries the SVG");
    assert!(
        svg.contains("<ellipse"),
        "SVG carries the committed ellipses"
    );
    assert!(
        !svg.contains("polygon"),
        "an ellipse run emits no triangle polygons"
    );
}
