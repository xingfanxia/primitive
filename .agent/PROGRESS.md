# PROGRESS — primitive 2026 rebuild

Milestone naming per plan §7. "Done" = the §7 gate passes via `make verify`. Evidence is the
command + its printed numbers, not prose confidence.

## CORE-1 — pure-Rust scalar port — ✅ DONE (2026-06-27)

Ported fogleman's math into `primitive-core` (pure, RNG injected): `canvas`, `raster`,
`color`, `draw`, `score`, `shape` (Triangle), `optimizer` (hill-climb), `es` ((1+1)-ES +
1/5 rule), `energy_map`, `model`.

Gate evidence:
- **Math parity (exact)** — `tests/parity.rs` vs committed `parity_fogleman.json`: scanlines,
  closed-form color, composited pixels, and **integer SSE all bit-identical** to fogleman;
  normalized scores within 1e-9 (cross-libm ULP). Stronger than the §7 SSIM-image ask.
- **Golden** — `tests/golden.rs`: fixed seed+target → **SSIM = 1.000000** (≥ 0.999) vs committed
  golden; **final score 0.062417 vs fogleman 0.062680 = 0.42%** (< 1%), PSNR 24.09 dB.

Notes: a true fogleman *image* match is impossible (it seeds with wall-clock time, non-reproducible),
so the golden is this port's own fixed-seed output as a determinism/regression lock; fogleman
equivalence is proven on the *math* exactly. `repeat` (-rep) not yet ported (default 0; gates use 0).

## CORE-2 — hexagonal split + CPU adapter — ✅ DONE (2026-06-27)

`primitive-compute` (ports: `ShapeSearch`), `primitive-gpu-cpu` (`CpuSearch`),
`primitive-engine` (`Engine<S>`). `tools/verify/check-boundaries.sh` enforces import direction.

Gate evidence:
- **Boundaries** — `check-boundaries.sh` exit 0 (negative-tested: catches an injected violation).
- **Parity** — `tests/cpu_baseline.rs`: `Engine<CpuSearch>` reconstruction is **byte-identical**
  to `core::Model` for the same seed (score bit-equal, pixels equal).
- **CPU baseline (M-series, single-thread, 128×128)** — **~22–24 shapes/sec, ~435–465k candidates/sec.**
  This is the number GPU throughput (GPU-2: ≥ 20×) is measured against.

## Open risks / carry-forward

- **Cross-arch float determinism**: the golden SSIM gate is exact only on the dev/CI arch (macOS
  arm64). True cross-platform determinism arrives with the Philox + integer-only path at GPU-3
  (plan §6.6 / Risk #1). CI pinned to `macos-latest` to keep the golden green meanwhile.
- **Versions** (re-checked 2026-06-27): rayon 1.12, image 0.25. For later milestones the ecosystem
  moved past the plan's pins — eframe is **0.35** (plan said 0.34.2), CubeCL **0.10** is now stable
  (plan said pin 0.9.x). Re-confirm with `cargo search` at GPU-1/GUI-1 before pinning.
- CPU adapter is single-threaded (exact oracle). Rayon multi-core is a deliberate later perf pass.

## GPU-1 — CubeCL → Metal fused scoring kernel — ✅ DONE (2026-06-27)

`primitive-gpu-cubecl` crate (cubecl 0.10, wgpu → Metal). Toolchain bumped to stable rustc 1.96
(cubecl 0.10 needs ≥1.92), pinned via `rust-toolchain.toml`.

Gate evidence:
- **CubeCL→Metal pipeline proven** (device init / upload / `#[cube]` launch / readback). De-risks
  plan Risk #1 (cross-backend toolchain) + #2 (Metal-from-Rust), the highest-risk items.
- **Fused `score_candidates` kernel — exact integer parity** (§7 GPU-1 gate): closed-form color +
  16-bit premultiplied composite + integer delta-SSE, one thread/candidate. Test
  `gpu_delta_sse_matches_cpu_exactly_for_1000_candidates`: **1000/1000 candidates bit-identical**
  to `primitive_core::candidate_color_and_delta` (185,979 covered pixels scored on Metal). Green in
  `make verify` (19 tests). Core exposes `delta_sse_partial` + `candidate_color_and_delta` as the
  integer oracle (plan §6.6); `eval.rs` is the parity target.

Scope notes (carry-forward to GPU-2/3, by design, not a gap):
- **Coverage is rasterized on the CPU and fed to the kernel.** fogleman's rasterizer uses f64 and
  Metal has no f64, so on-device rasterization needs a deterministic integer rasterizer (§6.6) —
  folded into GPU-2's on-device batch. GPU-1 isolates+proves the *scoring* math is exact on Metal.
- **64×64 cap keeps every accumulator in i32/u32** (no overflow). Larger targets need i64 or a
  hi/lo split — WGSL lacks i64, so this is a GPU-2 hardening item (validate on Metal first).

## GPU-2 — on-device rasterize + score + reduce_argmin — ✅ DONE (2026-06-27)

The §6.6 determinism work that unblocks the GPU: a **deterministic integer rasterizer** shared
CPU↔GPU. `primitive_core::rasterize_triangle_int` / `triangle_inside` use an integer edge-function
test (no f64 → no Metal divergence); the GPU kernel runs the identical test in-kernel. The f64
`rasterize_triangle` stays the CORE/golden reference (unchanged, CORE-1 intact); the integer one is
the portable production path.

Gate evidence (all green in `make verify`, 24 tests):
- **On-device parity** (§7 GPU-2): `gpu2_triangles.rs` — the `score_triangles` kernel (in-kernel
  raster + color + integer delta-SSE, candidate = 6 ints) is **1000/1000 bit-identical** to the CPU
  integer path (`rasterize_triangle_int` + `candidate_color_and_delta`).
- **reduce_argmin**: on-device `argmin` picks the **same winning index as CPU brute force**
  (winner 509, first-min tie-break). Only the 4-byte index syncs back.
- **Throughput** (§7 GPU-2): `gpu2_throughput.rs` — **43.9× single-core CPU** (GPU 25.7 M cand/s
  vs CPU 0.58 M cand/s, same integer work), well over the ≥20× gate.

Architecture: `GpuSession` holds target/current resident on-device; a step uploads only candidate
triangles (no per-candidate host sync) — the foundation the GPU-3 loop builds on.

Carry-forward: parallel (tree) argmin deferred — in the real loop argmin runs once per *step*
(~20k candidates), so single-thread is not the hot path; scoring is, and that's parallel. 64×64
i32/u32 cap still applies (i64/large-target = later hardening; WGSL lacks i64).

## Next: GPU-3

On-device search loop: Philox RNG + (1+1)-ES triangle mutation + energy-map restart sampling +
accept/reject + composite-winner, all GPU-resident across 1000 shapes. Gate: end-to-end ≥ 20× the
CPU shapes/sec baseline AND final PSNR within 0.5 dB of the CPU run. Re-run the CPU-oracle parity
test on every GPU change.
