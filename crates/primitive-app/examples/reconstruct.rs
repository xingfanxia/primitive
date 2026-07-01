//! Headless reconstruction demo (and manual smoke test): rebuild an image from geometric primitives
//! for all three shape types and write the results as PNGs — no window, no screen capture.
//!
//! It exercises the same search code the app is built on: the GPU "instant" path (`gpu_optimize`,
//! incl. the CORE-3b.3 ellipse/rect on-device search) when a Metal/CUDA device is present, else the
//! CPU reference search (`Model` — the parity oracle the GUI's `Engine<CpuSearch>` wraps). NOTE: the
//! GUI itself currently routes only triangles to the GPU (`runner::should_use_gpu`), so this demo —
//! which runs ALL three shapes through `gpu_optimize` — is the only end-to-end driver of the GPU
//! ellipse/rect search outside the test suite.
//!
//!   cargo run -p primitive-app --release --example reconstruct -- [out_dir] [sample|image_path]
//!
//! `out_dir` (default `.`, created if missing) gets `recon-{target,triangle,ellipse,rect}.png` (fixed
//! names — overwrites). Source: the bundled `cat` (default) or `mona lisa`, or a path to any PNG/JPEG.
//! Prints each shape's PSNR vs the downscaled target — this is **per-backend** (GPU and CPU use
//! different search effort), not a cross-backend benchmark.

use std::path::{Path, PathBuf};

use primitive_app::image_io;
use primitive_core::{difference_full, psnr_from_score, Canvas, Model, Rng, ShapeType};
use primitive_gpu_cubecl::{gpu_available, gpu_optimize, OptConfig};

/// Working resolution: the GPU "instant" cap, i32-safe for the integer ellipse test (§6.6 — the
/// `w <= 182` bound in `gpu_optimize`; the app pins the same 128 in `runner::GPU_INSTANT_MAX`). Both
/// backends run at this size so a run's per-shape PSNRs are comparable to each other.
const WORK: u32 = 128;
const SHAPES: usize = 300;
const ALPHA: i32 = 128;
const SEED: u32 = 1;

/// Resolve the source arg to a working-resolution target: an existing file path, else a bundled
/// sample matched by name. Errors loudly on an unresolvable arg (rather than silently substituting a
/// default) so a typo'd path or sample name can't masquerade as a bad reconstruction.
fn load_target(src: &str) -> Canvas {
    let (_dims, canvas) = if Path::new(src).is_file() {
        image_io::load_path(Path::new(src)).expect("read image")
    } else if let Some((_, bytes)) = image_io::SAMPLES.iter().find(|(name, _)| *name == src) {
        image_io::load_bytes(bytes).expect("decode sample")
    } else if src.contains(['/', '.']) {
        panic!("no such image file: {src:?}");
    } else {
        let names: Vec<&str> = image_io::SAMPLES.iter().map(|(n, _)| *n).collect();
        panic!("unknown source {src:?}; pass an image path or a sample name: {names:?}");
    };
    image_io::downscale(&canvas, WORK)
}

/// One reconstruction: GPU instant path when a device is present, else the CPU reference search.
fn reconstruct(target: &Canvas, shape: ShapeType, gpu: bool) -> Canvas {
    if gpu {
        gpu_optimize(
            target,
            &OptConfig {
                workers: 8192,
                age: 12,
                shapes: SHAPES,
                alpha: ALPHA,
                seed: SEED,
                shape_type: shape,
            },
        )
    } else {
        let mut model = Model::with_average_background(target.clone());
        let mut rng = Rng::new(SEED as u64);
        model.run(shape, ALPHA, SHAPES as i32, &mut rng);
        model.current.clone()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(args.get(1).map(String::as_str).unwrap_or("."));
    // Default source only when NO arg is given; a supplied-but-unresolvable arg errors in load_target.
    let src = args.get(2).map(String::as_str).unwrap_or("cat");

    let target = load_target(src);
    let gpu = gpu_available();
    println!(
        "source '{src}' -> {}x{} | backend: {} | {SHAPES} shapes/shape-type",
        target.w,
        target.h,
        if gpu {
            "GPU (gpu_optimize)"
        } else {
            "CPU (Model)"
        },
    );

    std::fs::create_dir_all(&out).expect("create out_dir");
    image_io::export_png(&target, &out.join("recon-target.png")).expect("write target");

    for (shape, name) in [
        (ShapeType::Triangle, "triangle"),
        (ShapeType::Ellipse, "ellipse"),
        (ShapeType::Rectangle, "rect"),
    ] {
        let recon = reconstruct(&target, shape, gpu);
        let psnr = psnr_from_score(difference_full(&target, &recon));
        let path = out.join(format!("recon-{name}.png"));
        image_io::export_png(&recon, &path).expect("write recon");
        println!("  {name:<9} -> PSNR {psnr:5.2} dB  ({})", path.display());
    }
}
