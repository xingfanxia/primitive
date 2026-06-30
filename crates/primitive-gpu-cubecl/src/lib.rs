//! # primitive-gpu-cubecl — GPU adapter (Metal now, CUDA at GPU-4)
//!
//! CubeCL `#[cube]` kernels compiled to Metal (via wgpu) from one source, held to **exact
//! integer-SSE parity** against the CPU oracle (`primitive_core`). All kernels live in
//! [`kernels`]; this module is the host side — buffer upload, launch, readback.
//!
//! - **GPU-1** [`gpu_score_candidates`]: scoring with CPU-supplied coverage (isolates the scoring
//!   math). Proves the closed-form color + composite + integer delta-SSE is exact on Metal.
//! - **GPU-2** [`GpuSession`]: target/current uploaded **once** and kept resident; each step uploads
//!   only the candidate triangles (6 ints each), rasterizes + scores them in-kernel (the integer
//!   edge-function rasterizer `primitive_core::rasterize_triangle_int` runs *on device*), and —
//!   for [`GpuSession::best_triangle`] — reduces to the winning index with the on-device `argmin`,
//!   so only 4 bytes cross back. No per-candidate host sync.
//!
//! Accumulators are sized for ≤ 64×64 targets so every value fits in i32/u32 (the same bound the
//! CPU integer math satisfies). Larger targets need i64 / a hi-lo split — WGSL lacks i64, so that
//! is GPU-2 hardening tracked in `.agent/PROGRESS.md`.

mod kernels;
mod score_shapes;
mod search;

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use primitive_core::Canvas;

const THREADS: u32 = 64;

fn pix_i32(c: &Canvas) -> Vec<i32> {
    c.pix.iter().map(|&b| b as i32).collect()
}

/// A candidate's covered scanlines flattened for [`gpu_score_candidates`] (GPU-1 path).
pub struct CandidateSpans {
    /// `[y, x1, x2]` triples for every scanline, all candidates concatenated.
    pub spans: Vec<i32>,
    /// Prefix offsets (in triple units): candidate `c` owns `spans[offsets[c]..offsets[c+1]]`.
    pub offsets: Vec<i32>,
    /// Per-candidate fill alpha.
    pub alphas: Vec<i32>,
}

/// GPU-1: score candidates from CPU-supplied coverage; returns integer delta-SSE per candidate.
pub fn gpu_score_candidates(target: &Canvas, current: &Canvas, c: &CandidateSpans) -> Vec<i32> {
    let n = c.alphas.len();
    let target_i32 = pix_i32(target);
    let current_i32 = pix_i32(current);
    let client = WgpuRuntime::client(&Default::default());

    let t_h = client.create_from_slice(bytemuck::cast_slice(&target_i32));
    let c_h = client.create_from_slice(bytemuck::cast_slice(&current_i32));
    let spans_h = client.create_from_slice(bytemuck::cast_slice(&c.spans));
    let off_h = client.create_from_slice(bytemuck::cast_slice(&c.offsets));
    let alpha_h = client.create_from_slice(bytemuck::cast_slice(&c.alphas));
    let out_h = client.empty(n * core::mem::size_of::<i32>());

    unsafe {
        kernels::score_candidates::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static((n as u32).div_ceil(THREADS), 1, 1),
            CubeDim::new_1d(THREADS),
            ArrayArg::from_raw_parts(t_h.clone(), target_i32.len()),
            ArrayArg::from_raw_parts(c_h.clone(), current_i32.len()),
            ArrayArg::from_raw_parts(spans_h.clone(), c.spans.len()),
            ArrayArg::from_raw_parts(off_h.clone(), c.offsets.len()),
            ArrayArg::from_raw_parts(alpha_h.clone(), c.alphas.len()),
            target.w as i32,
            ArrayArg::from_raw_parts(out_h.clone(), n),
        );
    }
    bytemuck::cast_slice(&client.read_one_unchecked(out_h)).to_vec()
}

/// A batch of triangle candidates for the GPU-2 on-device path.
pub struct TriangleBatch {
    /// 6 ints per candidate: `[x1, y1, x2, y2, x3, y3]`.
    pub tris: Vec<i32>,
    /// Per-candidate fill alpha.
    pub alphas: Vec<i32>,
}

/// A batch of ellipse candidates for the on-device scorer (CORE-3b.2).
pub struct EllipseBatch {
    /// 4 ints per candidate: `[cx, cy, rx, ry]`.
    pub ells: Vec<i32>,
    /// Per-candidate fill alpha.
    pub alphas: Vec<i32>,
}

/// A batch of axis-aligned rectangle candidates for the on-device scorer (CORE-3b.2).
pub struct RectBatch {
    /// 4 ints per candidate: `[x1, y1, x2, y2]` (opposite corners).
    pub rects: Vec<i32>,
    /// Per-candidate fill alpha.
    pub alphas: Vec<i32>,
}

/// GPU-2 resident session: the target and current canvases are uploaded once and stay on the
/// device across many candidate batches (the steady state of a search step / the GPU-3 loop).
pub struct GpuSession {
    client: ComputeClient<WgpuRuntime>,
    target_h: Handle,
    current_h: Handle,
    tlen: usize,
    clen: usize,
    w: i32,
    h: i32,
}

impl GpuSession {
    /// Upload `target` and `current` to the device once.
    pub fn new(target: &Canvas, current: &Canvas) -> Self {
        let target_i32 = pix_i32(target);
        let current_i32 = pix_i32(current);
        let client = WgpuRuntime::client(&Default::default());
        let target_h = client.create_from_slice(bytemuck::cast_slice(&target_i32));
        let current_h = client.create_from_slice(bytemuck::cast_slice(&current_i32));
        Self {
            client,
            target_h,
            current_h,
            tlen: target_i32.len(),
            clen: current_i32.len(),
            w: target.w as i32,
            h: target.h as i32,
        }
    }

    /// Dispatch the on-device rasterize+score kernel for `b`; returns the resident output handle
    /// and candidate count (so callers can reduce on-device or read back).
    fn dispatch_scores(&self, b: &TriangleBatch) -> (Handle, usize) {
        let n = b.alphas.len();
        let tris_h = self.client.create_from_slice(bytemuck::cast_slice(&b.tris));
        let alpha_h = self
            .client
            .create_from_slice(bytemuck::cast_slice(&b.alphas));
        let out_h = self.client.empty(n * core::mem::size_of::<i32>());
        unsafe {
            kernels::score_triangles::launch_unchecked::<WgpuRuntime>(
                &self.client,
                CubeCount::Static((n as u32).div_ceil(THREADS), 1, 1),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(self.target_h.clone(), self.tlen),
                ArrayArg::from_raw_parts(self.current_h.clone(), self.clen),
                ArrayArg::from_raw_parts(tris_h.clone(), b.tris.len()),
                ArrayArg::from_raw_parts(alpha_h.clone(), n),
                self.w,
                self.h,
                ArrayArg::from_raw_parts(out_h.clone(), n),
            );
        }
        (out_h, n)
    }

    /// Rasterize + score every candidate on the GPU; returns integer delta-SSE per candidate.
    /// Matches `rasterize_triangle_int` + `candidate_color_and_delta` on the CPU exactly.
    pub fn score_triangles(&self, b: &TriangleBatch) -> Vec<i32> {
        let (out_h, _n) = self.dispatch_scores(b);
        bytemuck::cast_slice(&self.client.read_one_unchecked(out_h)).to_vec()
    }

    /// Rasterize + score, then reduce to the winning index **on-device** (minimum delta-SSE,
    /// first-min tie-break — the same winner CPU brute force picks). Only 4 bytes sync back.
    pub fn best_triangle(&self, b: &TriangleBatch) -> i32 {
        let (out_h, n) = self.dispatch_scores(b);
        let idx_h = self.client.empty(core::mem::size_of::<i32>());
        unsafe {
            kernels::argmin::launch_unchecked::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(out_h.clone(), n),
                n as i32,
                ArrayArg::from_raw_parts(idx_h.clone(), 1),
            );
        }
        let idx: Vec<i32> = bytemuck::cast_slice(&self.client.read_one_unchecked(idx_h)).to_vec();
        idx[0]
    }

    /// Rasterize + score every **ellipse** candidate on the GPU (CORE-3b.2); integer delta-SSE per
    /// candidate, exact vs `rasterize_ellipse_int` + `candidate_color_and_delta` on the CPU.
    pub fn score_ellipses(&self, b: &EllipseBatch) -> Vec<i32> {
        let n = b.alphas.len();
        let ells_h = self.client.create_from_slice(bytemuck::cast_slice(&b.ells));
        let alpha_h = self
            .client
            .create_from_slice(bytemuck::cast_slice(&b.alphas));
        let out_h = self.client.empty(n * core::mem::size_of::<i32>());
        unsafe {
            score_shapes::score_ellipses::launch_unchecked::<WgpuRuntime>(
                &self.client,
                CubeCount::Static((n as u32).div_ceil(THREADS), 1, 1),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(self.target_h.clone(), self.tlen),
                ArrayArg::from_raw_parts(self.current_h.clone(), self.clen),
                ArrayArg::from_raw_parts(ells_h.clone(), b.ells.len()),
                ArrayArg::from_raw_parts(alpha_h.clone(), n),
                self.w,
                self.h,
                ArrayArg::from_raw_parts(out_h.clone(), n),
            );
        }
        bytemuck::cast_slice(&self.client.read_one_unchecked(out_h)).to_vec()
    }

    /// Rasterize + score every **rectangle** candidate on the GPU (CORE-3b.2); exact vs
    /// `rasterize_rectangle_int` + `candidate_color_and_delta` on the CPU.
    pub fn score_rects(&self, b: &RectBatch) -> Vec<i32> {
        let n = b.alphas.len();
        let rects_h = self
            .client
            .create_from_slice(bytemuck::cast_slice(&b.rects));
        let alpha_h = self
            .client
            .create_from_slice(bytemuck::cast_slice(&b.alphas));
        let out_h = self.client.empty(n * core::mem::size_of::<i32>());
        unsafe {
            score_shapes::score_rectangles::launch_unchecked::<WgpuRuntime>(
                &self.client,
                CubeCount::Static((n as u32).div_ceil(THREADS), 1, 1),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(self.target_h.clone(), self.tlen),
                ArrayArg::from_raw_parts(self.current_h.clone(), self.clen),
                ArrayArg::from_raw_parts(rects_h.clone(), b.rects.len()),
                ArrayArg::from_raw_parts(alpha_h.clone(), n),
                self.w,
                self.h,
                ArrayArg::from_raw_parts(out_h.clone(), n),
            );
        }
        bytemuck::cast_slice(&self.client.read_one_unchecked(out_h)).to_vec()
    }
}

/// GPU-2 one-shot: rasterize + score a batch (uploads target/current per call). For repeated
/// dispatches against the same canvases, build a [`GpuSession`] instead.
pub fn gpu_score_triangles(target: &Canvas, current: &Canvas, b: &TriangleBatch) -> Vec<i32> {
    GpuSession::new(target, current).score_triangles(b)
}

/// GPU-2 one-shot: the winning candidate index via the on-device argmin.
pub fn gpu_best_triangle(target: &Canvas, current: &Canvas, b: &TriangleBatch) -> i32 {
    GpuSession::new(target, current).best_triangle(b)
}

/// CORE-3b.2 one-shot: rasterize + score an ellipse batch (uploads target/current per call).
pub fn gpu_score_ellipses(target: &Canvas, current: &Canvas, b: &EllipseBatch) -> Vec<i32> {
    GpuSession::new(target, current).score_ellipses(b)
}

/// CORE-3b.2 one-shot: rasterize + score a rectangle batch (uploads target/current per call).
pub fn gpu_score_rectangles(target: &Canvas, current: &Canvas, b: &RectBatch) -> Vec<i32> {
    GpuSession::new(target, current).score_rects(b)
}

/// GPU-3 search configuration. Per shape, `workers` independent hill-climbs each run `age`
/// mutate-and-keep-better iterations; the best across workers is committed.
pub struct OptConfig {
    pub workers: usize,
    pub age: u32,
    pub shapes: usize,
    pub alpha: i32,
    pub seed: u32,
}

/// GPU-3: run the whole search loop on-device and return the final reconstruction.
///
/// `target` and a `current` canvas (seeded to the target's average color) stay resident; each
/// shape step launches `evolve` (parallel hill-climbs) → `argmin` (winner) → `commit` (composite
/// the winner into `current`) with **no host sync** until the final readback. The reconstruction
/// is the GPU's own optimizer output — compared to the CPU run by PSNR, not byte-parity.
pub fn gpu_optimize(target: &Canvas, cfg: &OptConfig) -> Canvas {
    let (ar, ag, ab) = target.average_color();
    let current = Canvas::filled(target.w, target.h, ar, ag, ab);
    let target_i32 = pix_i32(target);
    let current_i32 = pix_i32(&current);
    let n_pix = target_i32.len();
    let w = target.w as i32;
    let h = target.h as i32;

    let client = WgpuRuntime::client(&Default::default());
    let target_h = client.create_from_slice(bytemuck::cast_slice(&target_i32));
    let current_h = client.create_from_slice(bytemuck::cast_slice(&current_i32));
    let best_delta_h = client.empty(cfg.workers * core::mem::size_of::<i32>());
    let best_tri_h = client.empty(cfg.workers * 6 * core::mem::size_of::<i32>());

    for step in 0..cfg.shapes {
        unsafe {
            search::evolve::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static((cfg.workers as u32).div_ceil(THREADS), 1, 1),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(target_h.clone(), n_pix),
                ArrayArg::from_raw_parts(current_h.clone(), n_pix),
                cfg.seed,
                step as u32,
                cfg.alpha,
                w,
                h,
                cfg.age,
                ArrayArg::from_raw_parts(best_delta_h.clone(), cfg.workers),
                ArrayArg::from_raw_parts(best_tri_h.clone(), cfg.workers * 6),
            );
            // commit fuses the argmin (over best_delta) + composite — one launch, not two.
            search::commit::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(target_h.clone(), n_pix),
                ArrayArg::from_raw_parts(current_h.clone(), n_pix),
                ArrayArg::from_raw_parts(best_tri_h.clone(), cfg.workers * 6),
                ArrayArg::from_raw_parts(best_delta_h.clone(), cfg.workers),
                cfg.workers as i32,
                cfg.alpha,
                w,
                h,
            );
        }
    }

    let bytes = client.read_one_unchecked(current_h);
    let out: &[i32] = bytemuck::cast_slice(&bytes);
    let pix: Vec<u8> = out.iter().map(|&v| v.clamp(0, 255) as u8).collect();
    Canvas {
        w: target.w,
        h: target.h,
        pix,
    }
}

/// Is a compatible GPU present? Probes by spinning up the wgpu/Metal client and allocating one
/// tiny buffer; any failure (no adapter, driver fault) is caught and reported as "unavailable" so
/// the GUI can degrade to the CPU oracle instead of crashing (plan §5A: the GPU-unavailable state
/// is first-class, never a fatal error). This is the app's "one-time probe on launch".
pub fn gpu_available() -> bool {
    std::panic::catch_unwind(|| {
        let client = WgpuRuntime::client(&Default::default());
        let _ = client.empty(core::mem::size_of::<i32>());
    })
    .is_ok()
}

/// GPU-3 probe: `rand_below(seed, i, range)` for `i in 0..n` on the GPU — proves the kernel's
/// counter-based RNG is bit-identical to `primitive_core::rand_below` (the determinism substrate).
pub fn gpu_philox_fill(seed: u32, range: u32, n: usize) -> Vec<u32> {
    let client = WgpuRuntime::client(&Default::default());
    let out_h = client.empty(n * core::mem::size_of::<u32>());
    let threads = THREADS;
    unsafe {
        kernels::philox_fill::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static((n as u32).div_ceil(threads), 1, 1),
            CubeDim::new_1d(threads),
            seed,
            range,
            ArrayArg::from_raw_parts(out_h.clone(), n),
        );
    }
    bytemuck::cast_slice(&client.read_one_unchecked(out_h)).to_vec()
}
