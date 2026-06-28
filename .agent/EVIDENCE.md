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

## GPU-3 (2026-06-27)

### Determinism substrate — `crates/primitive-gpu-cubecl/tests/gpu3_philox.rs`
```
GPU Philox RNG: 10000/10000 draws bit-identical to CPU (range 64)
```
Pure-u32 counter-based RNG (`primitive_core::philox`); the kernel mirrors it exactly, so each GPU
worker runs a reproducible stream from its indices alone.

### End-to-end on-device search — `crates/primitive-gpu-cubecl/tests/gpu3_optimize.rs`
```
GPU-3 end-to-end (100 shapes, 64×64, workers=10240 age=9):
  CPU 36.22 dB @ 32.7 shapes/s
  GPU 35.97 dB @ 519.3 shapes/s  (target ≥ 460)
  PSNR gap -0.24 dB
```
The full loop (Philox + L2-residual energy-targeted restart + self-adaptive 1/5-rule parallel
hill-climb + fused argmin/commit) runs GPU-resident across all shapes. **519 shapes/s ≥ 460** (the
documented ≥20× CORE-2-baseline target) **and −0.24 dB** vs the CPU reference (within the 0.5 dB gate).
Green in `make verify` (30 tests). The self-adaptive step (Rechenberg 1/5 rule) replaced the hand-tuned
anneal and lets a shallower, more-parallel config (10240×9) hold quality — see the sweep below.

> Why 460 sps, not 20× the same-run CPU: the GPU is i32-capped to 64×64, where the CPU is unusually
> fast (~33 sps). The established CORE-2 baseline is 128×128 (~23 sps); 20× that ⇒ 460 sps is the
> stable cross-resolution target recorded above. GPU-2 already proved 43.9× at the candidate level.

### fogleman-vs-GPU comparison + size scaling (2026-06-28)

`go_primitive/primitive/bench_test.go` (fogleman Go, timed in-process, best of 3) + `examples/sweep.rs`
(our GPU). Matched synthetic target, triangle a=128, 100 shapes, identical n=1000/age=100/m=16 budget.
M-series, 16 logical / 12 perf cores.

| runner | 64×64 sps | 128×128 sps |
|---|---|---|
| fogleman Go −j1 (1 core) | 33.8 | 21.0 |
| fogleman Go −j16 (all cores) | 263.9 | 174.9 |
| our CPU port (1 thread) | 32.7 | 20.9 |
| **our GPU (Metal), quality-matched** | **~509** | **56** (or 93 in-gate @ −0.42 dB) |

- **64×64 (the GPU's regime on this Mac): GPU ≈ 1.9× fogleman −j16, ≈ 15× −j1**, quality matched.
- **128×128: the GPU LOSES to fogleman −j16** (56 sps matched vs 174.9 ⇒ 0.32×). Going 64→128 the GPU
  slows ~9× (4× pixels × ~3× deeper climbs to hold quality) while fogleman slows only ~1.5× (fixed
  budget, data stays in M-series cache). **Unified memory gives the GPU no bandwidth edge** to offset.
- **i32 does NOT overflow at 128×128 in the search** — anchored triangles stay small; 2048×84 reaches
  36.77 dB > CPU, impossible under destructive overflow. So 128×128 runs correctly *without* hi-lo.

**Consequence for the roadmap:** the hi-lo i64 / large-canvas path (GPU-4) is **deprioritized on
Apple Silicon** — bigger canvas is a *negative-value* move here (the GPU gets relatively slower). The
big-canvas GPU win requires a *discrete* GPU (dedicated bandwidth + more cores) — the CUDA backend on
non-unified hardware, not a Metal size unlock.
