//! Headless reconstruction demo (and manual smoke test): rebuild an image from geometric primitives
//! for all three shape types and write the results as PNGs — no window, no screen capture. This is
//! the same engine the GUI drives: the GPU "instant" path (`gpu_optimize`, incl. the CORE-3b.3
//! ellipse/rect on-device search) when a Metal/CUDA device is present, else the CPU oracle (`Model`).
//!
//!   cargo run -p primitive-app --release --example reconstruct -- [out_dir] [sample|image_path]
//!
//! `out_dir` defaults to `.`; the source defaults to the bundled `cat` sample (or pass `mona lisa`,
//! or a path to any PNG/JPEG). Writes `<out_dir>/recon-{target,triangle,ellipse,rect}.png` and prints
//! the PSNR of each reconstruction against the (downscaled) target.

use std::path::{Path, PathBuf};

use primitive_app::image_io;
use primitive_core::{difference_full, psnr_from_score, Canvas, Model, Rng, ShapeType};
use primitive_gpu_cubecl::{gpu_available, gpu_optimize, OptConfig};

/// GPU instant-mode canvas cap (i32-safe for the integer rasterizers, §6.6); the CPU path uses the
/// same size so both backends produce comparable output.
const WORK: u32 = 128;
const SHAPES: usize = 300;
const ALPHA: i32 = 128;
const SEED: u32 = 1;

/// Resolve the source arg to a working-resolution target: an existing file path, else a bundled
/// sample matched by name (default `cat`).
fn load_target(src: &str) -> Canvas {
    let (_dims, canvas) = if Path::new(src).is_file() {
        image_io::load_path(Path::new(src)).expect("read image")
    } else {
        let bytes = image_io::SAMPLES
            .iter()
            .find(|(name, _)| *name == src)
            .map(|(_, b)| *b)
            .unwrap_or(image_io::SAMPLES[0].1);
        image_io::load_bytes(bytes).expect("decode sample")
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
