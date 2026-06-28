# EVIDENCE — primitive 2026 rebuild

Committed gate evidence. Each entry = the command + its printed proof, so a fresh agent can
re-derive "done" without trusting chat memory. Numbers are this Mac (Apple Silicon).

## CORE-1 / CORE-2 (2026-06-27)

### `cargo test --workspace --release` → exit 0
16 tests across the workspace pass (unit + parity + golden + engine). Run: `make test` / `make verify`.

### Math parity vs fogleman — `crates/primitive-core/tests/parity.rs`
`fogleman_math_parity_bit_exact ... ok` — scanlines, closed-form color, composited pixels, and
**integer SSE all bit-identical** to fogleman's Go on the committed fixture; normalized scores
within 1e-9 (cross-libm ULP).

### CORE-1 golden — `crates/primitive-core/tests/golden.rs`
```
golden SSIM = 1.000000                                   # ≥ 0.999 determinism gate
final_score=0.062417 fogleman_ref=0.062658 relative_diff=0.3848% psnr=24.094dB   # < 1% quality gate
```
fogleman reference reproducible via `scripts/ops/regen-fogleman-ref.sh` (committed target fixture).

### Architecture boundaries — `tools/verify/check-boundaries.sh` → exit 0
Negative-tested: catches inline dep, `[dependencies.X]` table-header dep, and fully-qualified
adapter paths; ignores comment-only GPU mentions.

### CPU baseline (the GPU-2 ≥20× anchor) — `crates/primitive-engine/tests/cpu_baseline.rs`
Single-thread, synthetic 128×128, 100 triangles, `n=1000 age=100 m=16`:

| Metric | Value (M-series, single-thread) |
|---|---|
| shapes/sec | ~22–24 |
| **candidates/sec** | **~435,000–465,000** |
| evals / 100 shapes | 1,980,304 |

**GPU-2 target: ≥ 20× candidates/sec ⇒ ≳ 9.0 M candidates/sec** on the same Mac. GPU-3
end-to-end target: ≥ 20× shapes/sec ⇒ ≳ 460 shapes/sec, PSNR within 0.5 dB of the CPU run.

> The CPU adapter is single-threaded by choice (exact parity oracle). A multi-core Rayon
> baseline (later perf pass) would raise this anchor; re-measure and update here if adopted.

## GPU-1 (2026-06-27)

### CubeCL→Metal pipeline + fused scoring kernel — exact integer parity
`crates/primitive-gpu-cubecl/tests/gpu_parity.rs`:
```
GPU vs CPU integer delta-SSE: 1000/1000 exact; 185979 covered pixels scored on Metal
```
The GPU `score_candidates` kernel (color + composite + integer delta-SSE) reproduces
`primitive_core::candidate_color_and_delta` bit-for-bit for 1000 random triangle candidates on a
64×64 target. Coverage rasterized on CPU (on-device raster → GPU-2). Green in `make verify`
(19 tests, stable rustc 1.96).

## GPU-2 (2026-06-27)

### On-device rasterize + score parity — `crates/primitive-gpu-cubecl/tests/gpu2_triangles.rs`
```
GPU-2 on-device raster+score: 1000/1000 candidates bit-identical to CPU integer path
GPU-2 reduce_argmin: winner=509 (delta -2875356), CPU brute-force winner=509 (delta -2875356)
```
The `score_triangles` kernel rasterizes in-kernel (integer edge-function test ==
`primitive_core::rasterize_triangle_int`) then color+composite+delta-SSE; exact vs the CPU integer
path. On-device `argmin` matches CPU brute force.

### Throughput ≥20× — `crates/primitive-gpu-cubecl/tests/gpu2_throughput.rs`
```
GPU-2 throughput: GPU 25.66 M cand/s | CPU 0.58 M cand/s (integer path, 1 core) | speedup 43.9×
```
Same integer work both sides (rasterize + color + delta-SSE per candidate); 64×64, alpha 128,
262k-candidate batches, target/current resident in a `GpuSession`. Comfortably clears the ≥20×
(≳9 M cand/s) gate. Green in `make verify` (24 tests).
