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

## Next: GPU-2 → GPU-3

GPU-2: batched B-candidate dispatch + `reduce_argmin` (winning index == CPU brute force), on-device
deterministic rasterization, **≥ 20× the CORE-2 CPU candidates/sec baseline**. GPU-3: on-device
(1+1)-ES + Philox + energy-map restarts; 1000-shape run ≥ 20× CPU, PSNR within 0.5 dB.
Re-run the CPU-oracle parity test on every GPU change.
