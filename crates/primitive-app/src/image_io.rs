//! Image load (→ working-resolution `Canvas`) + export (PNG/SVG). Kept apart from UI/run so the
//! boundary with `image`/filesystem is a thin adapter (plan: browser/IO behind typed adapters).

use std::path::Path;

use primitive_core::Canvas;

/// Search working resolution cap (fogleman's `-r 256` default): the longest side is resized to this,
/// preserving aspect. The display upscales; the search runs at this size.
pub const WORK_MAX: u32 = 256;

/// Bundled sample images (always available regardless of CWD — one-click "just try it", §5A).
pub const SAMPLES: &[(&str, &[u8])] = &[
    ("cat", include_bytes!("../../../docs/images/cat.png")),
    (
        "mona lisa",
        include_bytes!("../../../docs/images/monalisa.png"),
    ),
];

fn to_canvas(img: image::DynamicImage) -> Canvas {
    let img = img.to_rgba8();
    let (w, h) = img.dimensions();
    let scale = (WORK_MAX as f32 / w.max(h) as f32).min(1.0);
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    let resized = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
    Canvas {
        w: nw as usize,
        h: nh as usize,
        pix: resized.into_raw(),
    }
}

/// Original dimensions (for the source-panel readout) + the working `Canvas`.
pub fn load_path(path: &Path) -> Result<((u32, u32), Canvas), String> {
    let img = image::open(path).map_err(|e| format!("Couldn't read that image — {e}"))?;
    Ok((
        image::image_dimensions(path).unwrap_or((img.width(), img.height())),
        to_canvas(img),
    ))
}

/// Decode a bundled sample from memory.
pub fn load_bytes(bytes: &[u8]) -> Result<((u32, u32), Canvas), String> {
    let img =
        image::load_from_memory(bytes).map_err(|e| format!("Couldn't decode sample — {e}"))?;
    Ok(((img.width(), img.height()), to_canvas(img)))
}

/// Write the current canvas as a PNG.
pub fn export_png(canvas: &Canvas, path: &Path) -> Result<(), String> {
    image::RgbaImage::from_raw(canvas.w as u32, canvas.h as u32, canvas.pix.clone())
        .ok_or_else(|| "internal: canvas buffer size mismatch".to_string())?
        .save(path)
        .map_err(|e| format!("Couldn't save — {e}"))
}

/// Write the reconstruction's vector SVG (from `Model::svg`).
pub fn export_svg(svg: &str, path: &Path) -> Result<(), String> {
    std::fs::write(path, svg).map_err(|e| format!("Couldn't save — {e}"))
}

/// Box-downscale a canvas so its longest side is ≤ `max` (used by the GPU "instant" path, which is
/// i32-capped to ≤ 128 px per `.agent/EVIDENCE.md`). A no-op when already within bound.
pub fn downscale(canvas: &Canvas, max: u32) -> Canvas {
    let (w, h) = (canvas.w as u32, canvas.h as u32);
    if w.max(h) <= max {
        return canvas.clone();
    }
    let scale = max as f32 / w.max(h) as f32;
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    let img = image::RgbaImage::from_raw(w, h, canvas.pix.clone()).expect("canvas buffer matches");
    let resized = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
    Canvas {
        w: nw as usize,
        h: nh as usize,
        pix: resized.into_raw(),
    }
}

/// Encode the run's progression as an animated GIF (plan §5A Export ▾ third format). Each captured
/// keyframe becomes one GIF frame at `delay_ms`; the last frame holds longer so the result lingers.
pub fn export_gif(frames: &[Canvas], delay_ms: u16, path: &Path) -> Result<(), String> {
    use image::codecs::gif::{GifEncoder, Repeat};
    use image::{Delay, Frame};

    if frames.is_empty() {
        return Err("Nothing to export yet — run a few shapes first".to_string());
    }
    let file = std::fs::File::create(path).map_err(|e| format!("Couldn't save — {e}"))?;
    let mut enc = GifEncoder::new(std::io::BufWriter::new(file));
    enc.set_repeat(Repeat::Infinite)
        .map_err(|e| format!("Couldn't save — {e}"))?;
    let last = frames.len() - 1;
    for (i, c) in frames.iter().enumerate() {
        let buf = image::RgbaImage::from_raw(c.w as u32, c.h as u32, c.pix.clone())
            .ok_or_else(|| "internal: canvas buffer size mismatch".to_string())?;
        let ms = if i == last {
            delay_ms.max(1) * 8
        } else {
            delay_ms
        };
        let frame = Frame::from_parts(buf, 0, 0, Delay::from_numer_denom_ms(ms as u32, 1));
        enc.encode_frame(frame)
            .map_err(|e| format!("Couldn't save — {e}"))?;
    }
    Ok(())
}
