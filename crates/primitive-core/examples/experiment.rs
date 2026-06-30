//! Quality/throughput experiment harness (not shipped).
//!
//! Resizes an input image once, saves it as the shared optimization target (so fogleman and
//! this port optimize byte-identical pixels), runs the reference `Model`, and prints the
//! final normalized-RMSE score, PSNR, and shapes/sec.
//!
//! Usage: `cargo run --release -p primitive-core --example experiment -- <img> <size> <count> <seed> <target_out.png>`

use image::GenericImageView;
use primitive_core::{Model, Rng, ShapeType};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let size: u32 = args[2].parse().unwrap();
    let count: i32 = args[3].parse().unwrap();
    let seed: u64 = args[4].parse().unwrap();
    let target_out = &args[5];

    // Resize preserving aspect to fit within size×size (fogleman's Thumbnail behavior).
    let img = image::open(path).expect("open image");
    let (w, h) = img.dimensions();
    let aspect = w as f64 / h as f64;
    let (nw, nh) = if w <= size && h <= size {
        (w, h)
    } else if aspect >= 1.0 {
        (size, (size as f64 / aspect).round() as u32)
    } else {
        ((size as f64 * aspect).round() as u32, size)
    };
    let resized = image::imageops::resize(
        &img.to_rgba8(),
        nw,
        nh,
        image::imageops::FilterType::Triangle,
    );
    resized.save(target_out).expect("save target");

    let mut canvas = primitive_core::Canvas::new(nw as usize, nh as usize);
    canvas.pix.copy_from_slice(resized.as_raw());
    // Force opaque alpha so scoring matches fogleman's opaque target.
    for px in canvas.pix.chunks_exact_mut(4) {
        px[3] = 255;
    }

    let mut model = Model::with_average_background(canvas);
    let mut rng = Rng::new(seed);

    let start = Instant::now();
    let evals = model.run(ShapeType::Triangle, 128, count, &mut rng);
    let secs = start.elapsed().as_secs_f64();

    println!(
        "size={}x{} count={} seed={} final_score={:.6} psnr={:.3} shapes_per_sec={:.1} candidates_per_sec={:.0} evals={}",
        nw,
        nh,
        count,
        seed,
        model.score,
        model.psnr(),
        count as f64 / secs,
        evals as f64 / secs,
        evals,
    );
}
