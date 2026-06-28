//! # primitive-gpu-cubecl — GPU adapter (Metal now, CUDA at GPU-4)
//!
//! CubeCL `#[cube]` kernels compiled to Metal (via wgpu) from one source, held to **exact
//! integer-SSE parity** against the CPU oracle (`primitive_core::eval`).
//!
//! GPU-1: the fused `score_candidates` kernel — closed-form color → 16-bit premultiplied
//! composite → integer delta-SSE, one thread per candidate. It reproduces
//! `primitive_core::candidate_color_and_delta` bit-for-bit (plan §6.6 determinism contract).
//!
//! Coverage (scanlines) is rasterized on the CPU and fed in: fogleman's rasterizer uses f64
//! and Metal has no f64, so on-device rasterization (with a deterministic integer rasterizer)
//! is deferred to GPU-2/3. Accumulators are sized for ≤ 64×64 targets so every value fits in
//! i32/u32 with no overflow (the same bound the CPU integer math satisfies); larger targets
//! get the i64 / hi-lo split at GPU-2.

use cubecl::prelude::*;
use primitive_core::Canvas;

/// 16-bit premultiplied over-composite of one channel — Go's `image/draw` integer math.
#[cube]
fn composite(before: u32, s: u32, aa: u32) -> u32 {
    ((before * aa + s * 65535) / 65535) >> 8
}

/// Signed per-channel SSE delta `(t-after)² − (t-before)²` for one composited channel.
#[cube]
fn channel_delta(t: i32, before: i32, s: u32, aa: u32) -> i32 {
    let after = composite(before as u32, s, aa) as i32;
    let da = t - after;
    let db = t - before;
    da * da - db * db
}

#[cube]
#[allow(clippy::manual_clamp)] // std `i32::clamp` isn't available inside a #[cube] kernel
fn clamp_0_255(x: i32) -> i32 {
    let mut r = x;
    if r < 0 {
        r = 0;
    }
    if r > 255 {
        r = 255;
    }
    r
}

/// One thread per candidate: closed-form color over its coverage, then integer delta-SSE.
///
/// `spans` is a flat `[y, x1, x2]` triple per scanline; `offsets[c]..offsets[c+1]` are
/// candidate `c`'s spans (in triple units). Mirrors `compute_color` + `draw_lines` +
/// `delta_sse_partial` exactly.
#[cube(launch_unchecked)]
fn score_candidates(
    target: &Array<i32>,
    current: &Array<i32>,
    spans: &Array<i32>,
    offsets: &Array<i32>,
    alphas: &Array<i32>,
    width: i32,
    out: &mut Array<i32>,
) {
    let c = ABSOLUTE_POS;
    if c < out.len() {
        let alpha = alphas[c];
        let a_coef = 65535 / alpha;
        let s_start = offsets[c];
        let s_end = offsets[c + 1];

        // Pass 1 — closed-form color sums over covered pixels.
        let mut rsum = 0i32;
        let mut gsum = 0i32;
        let mut bsum = 0i32;
        let mut count = 0i32;
        for s in s_start..s_end {
            let su = (s * 3) as usize;
            let y = spans[su];
            let x1 = spans[su + 1];
            let x2 = spans[su + 2];
            for x in x1..x2 + 1 {
                let idx = ((y * width + x) * 4) as usize;
                rsum += (target[idx] - current[idx]) * a_coef + current[idx] * 257;
                gsum += (target[idx + 1] - current[idx + 1]) * a_coef + current[idx + 1] * 257;
                bsum += (target[idx + 2] - current[idx + 2]) * a_coef + current[idx + 2] * 257;
                count += 1;
            }
        }

        let mut delta = 0i32;
        if count > 0 {
            let r = clamp_0_255((rsum / count) >> 8);
            let g = clamp_0_255((gsum / count) >> 8);
            let b = clamp_0_255((bsum / count) >> 8);

            // NRGBA → 16-bit premultiplied source channels + dest coefficient.
            let av = alpha as u32;
            let sr = ((r as u32) | ((r as u32) << 8)) * av / 255;
            let sg = ((g as u32) | ((g as u32) << 8)) * av / 255;
            let sb = ((b as u32) | ((b as u32) << 8)) * av / 255;
            let sa = av | (av << 8);
            let aa = (65535 - sa * 65535 / 65535) * 257;

            // Pass 2 — composite + integer delta-SSE over all four channels.
            for s in s_start..s_end {
                let su = (s * 3) as usize;
                let y = spans[su];
                let x1 = spans[su + 1];
                let x2 = spans[su + 2];
                for x in x1..x2 + 1 {
                    let idx = ((y * width + x) * 4) as usize;
                    delta += channel_delta(target[idx], current[idx], sr, aa);
                    delta += channel_delta(target[idx + 1], current[idx + 1], sg, aa);
                    delta += channel_delta(target[idx + 2], current[idx + 2], sb, aa);
                    delta += channel_delta(target[idx + 3], current[idx + 3], sa, aa);
                }
            }
        }
        out[c] = delta;
    }
}

/// A candidate's covered scanlines flattened for the kernel.
pub struct CandidateSpans {
    /// `[y, x1, x2]` triples for every scanline, all candidates concatenated.
    pub spans: Vec<i32>,
    /// Prefix offsets (in triple units): candidate `c` owns `spans[offsets[c]..offsets[c+1]]`.
    pub offsets: Vec<i32>,
    /// Per-candidate fill alpha.
    pub alphas: Vec<i32>,
}

/// Score every candidate on the GPU (Metal); returns the integer delta-SSE per candidate.
pub fn gpu_score_candidates(target: &Canvas, current: &Canvas, c: &CandidateSpans) -> Vec<i32> {
    use cubecl::wgpu::WgpuRuntime;
    let n_cand = c.alphas.len();
    let target_i32: Vec<i32> = target.pix.iter().map(|&b| b as i32).collect();
    let current_i32: Vec<i32> = current.pix.iter().map(|&b| b as i32).collect();

    let device = Default::default();
    let client = WgpuRuntime::client(&device);

    let t_h = client.create_from_slice(bytemuck::cast_slice(&target_i32));
    let c_h = client.create_from_slice(bytemuck::cast_slice(&current_i32));
    let spans_h = client.create_from_slice(bytemuck::cast_slice(&c.spans));
    let off_h = client.create_from_slice(bytemuck::cast_slice(&c.offsets));
    let alpha_h = client.create_from_slice(bytemuck::cast_slice(&c.alphas));
    let out_h = client.empty(n_cand * core::mem::size_of::<i32>());

    let threads = 64u32;
    let groups = (n_cand as u32).div_ceil(threads);
    unsafe {
        score_candidates::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(threads),
            ArrayArg::from_raw_parts(t_h.clone(), target_i32.len()),
            ArrayArg::from_raw_parts(c_h.clone(), current_i32.len()),
            ArrayArg::from_raw_parts(spans_h.clone(), c.spans.len()),
            ArrayArg::from_raw_parts(off_h.clone(), c.offsets.len()),
            ArrayArg::from_raw_parts(alpha_h.clone(), c.alphas.len()),
            target.w as i32, // runtime scalar arg (i32: LaunchArg)
            ArrayArg::from_raw_parts(out_h.clone(), n_cand),
        );
    }

    let bytes = client.read_one_unchecked(out_h);
    bytemuck::cast_slice(&bytes).to_vec()
}
