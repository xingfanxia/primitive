# AGENTS.md — primitive 2026 rebuild (routing map)

GPU-native rebuild of `fogleman/primitive` in Rust. **Spec is the source of truth:**
[`docs/plan/primitive-2026-architecture.md`](docs/plan/primitive-2026-architecture.md) (§5 layout, §6 kernel,
§7 milestones+gates) and provenance [`docs/research/primitive-2026-research-findings.md`](docs/research/primitive-2026-research-findings.md).
Milestone state + evidence: [`.agent/PROGRESS.md`](.agent/PROGRESS.md).

## Verify (correctness gate) + perf (hardware gate)

```
make verify        # fmt + clippy -D warnings + boundaries + giant-file + cargo test (release)
make perf          # hardware-dependent GPU perf gates (≥20× / ≥460 sps) — representative HW only
```
"Done" = `make verify` exits 0. CI (`.github/workflows/ci.yml`) runs `make verify` on macOS arm64.
**`make verify` is correctness only** — the GPU throughput thresholds are hardware-dependent, so they
measure-and-print under `make verify`/CI (a shared CI GPU is far slower) and are hard-asserted only
under `make perf` (`PRIMITIVE_PERF_GATE=1`) on representative hardware (Apple Silicon / discrete
NVIDIA). See `crates/primitive-gpu-cubecl/tests/common/mod.rs`.
Useful sub-targets: `make baseline` (CPU shapes/sec), `make golden` (SSIM + quality margin),
`make bundle` (PKG-1: build+validate `primitive.app`, halt before codesign), `make icon` (regenerate the app icon).

Run the desktop app: `cargo run -p primitive-app --release` (binary `primitive`).

## Crate map & import direction (hexagonal — enforced by `tools/verify/check-boundaries.sh`)

```
core (pure) ← compute (ports) ← engine (orchestration) ← adapters / app (composition root)
```

| Crate | Layer | May depend on | Notes |
|---|---|---|---|
| `primitive-core` | domain (pure) | — | shapes, raster, color, score, ES, model. No GPU/IO/clock/global-RNG. RNG is injected. The parity oracle + golden reference. |
| `primitive-compute` | ports | core | `ShapeSearch` trait + DTOs (`SearchParams`=`core::SearchBudget`, `BestShape`, `Backend`). Backend-agnostic. |
| `primitive-gpu-cpu` | adapter | core, compute | `CpuSearch`: single-threaded reference search → permanent parity oracle + GPU-absent fallback. |
| `primitive-gpu-cubecl` | adapter | core, compute | GPU adapter (CubeCL→Metal): fused score kernel, on-device search (`gpu_optimize`), `gpu_available()` probe. No float/64-bit atomics; integer-SSE parity with the CPU oracle. |
| `primitive-engine` | application | core, compute | `Engine<S: ShapeSearch>`: per-shape loop. Never names a concrete adapter (composition root is in `tests/`). |
| `primitive-app` | composition root (lib+bin `primitive`) | core, compute, engine, both adapters | eframe GUI (§5/§5A). Pure `state` module drives every interaction cell; `theme`/`device`/`i18n`/`hero`/`sidebar`/`runner`/`image_io` around it. The ONLY crate that imports adapters. |

## Where things are proven (gates per plan §7)

- **Math parity vs fogleman (exact)** — `crates/primitive-core/tests/parity.rs` against committed
  `tests/fixtures/parity_fogleman.json` (regenerate: `cd go_primitive && go test -run TestDumpParityFixture ./primitive`).
- **CORE-1 golden** (SSIM ≥ 0.999 determinism + final score within 1% of fogleman) — `tests/golden.rs`.
- **CORE-2 parity + CPU baseline** — `crates/primitive-engine/tests/cpu_baseline.rs` (byte-identical to core; prints shapes/sec).
- **GPU-1/2/3** — integer-SSE parity + on-device search in `crates/primitive-gpu-cubecl/tests/*` (Metal; `make verify` runs them). The **throughput** thresholds (GPU-2 ≥20×, GPU-3 ≥460 sps) are hardware-dependent → enforced by `make perf`, not `make verify` (see `tests/common/mod.rs`); the PSNR + integer-parity gates stay in `make verify`.
- **CORE-3 Part A** (Ellipse + Rectangle, core + CPU) — `crates/primitive-core/tests/shapes.rs` (rasterizer geometry, determinism, effectiveness). `Shape::triangle_coords()` panics for non-triangles until the GPU kernels generalize (CORE-3b).
- **CORE-3 Part B.1** (integer ellipse/rect rasterizers, the GPU-shared §6.6 path) — `crates/primitive-core/src/raster_int.rs` (new module; f64 golden reference stays in `raster.rs`). `ellipse_inside` = integer implicit test `ry²·dx²+rx²·dy² ≤ rx²·ry²`; `rasterize_ellipse_int`/`rasterize_rectangle_int`.
- **CORE-3 Part B.2** (on-device ellipse/rect score kernels) — `crates/primitive-gpu-cubecl/src/score_shapes.rs` (`score_ellipses`/`score_rectangles`), tests `gpu2_ellipses.rs`/`gpu2_rects.rs` (1000/1000 bit-identical to the CPU integer path on Metal).
- **CORE-3 Part B.3** (on-device ellipse/rect search loop) — `crates/primitive-gpu-cubecl/src/search_shapes.rs` (`evolve_ellipse`/`evolve_rect` + `commit_*`), `OptConfig.shape_type` dispatched in `gpu_optimize`; gates `gpu3_ellipse_optimize.rs`/`gpu3_rect_optimize.rs` (GPU PSNR within 0.4 dB of CPU). **CORE-3b complete** — GPU fits all 3 shapes. App routing stays CPU for ellipse/rect (keeps SVG + streaming); GPU instant for them is a one-line `should_use_gpu` change away.
- **CORE-3 Part C** (GUI shape selector) — `crates/primitive-app/tests/e2e.rs` (ellipse run → `<ellipse>` SVG), `runner::tests` (pure `should_use_gpu` routing guard), `a11y_tree.rs` (all three options in the AccessKit tree + click mutates selection). `ShapeType` is selectable + persisted; GPU instant mode stays triangle-only (CORE-3b).
- **GUI-2** — the §5A interaction gates in `crates/primitive-app/tests/*` (`state_suite` pure-state matrix,
  `e2e` load→100→SVG, `forced_cpu` device chip, `a11y_tree` AccessKit, `a11y_tokens` WCAG/Reduce-Motion).

Full milestone state (CORE-1/2 · GPU-1/2/3 · GUI-1/2 · PKG-1 Part A · CORE-3 complete: A+B.1+B.2+B.3+C — all ✅) lives in `.agent/PROGRESS.md` + `.agent/EVIDENCE.md`.

## Rules for changes here

- New code is TDD-first against the §7 gate. The CPU adapter is the permanent oracle — never delete it.
- Keep kernels (later) free of float/64-bit atomics; integer SSE is the determinism contract (§6.6).
- Re-run `make verify` before declaring done. Don't add a crate dependency that crosses the import direction.
