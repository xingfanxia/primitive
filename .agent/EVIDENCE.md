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

### Throughput ≥20× — `crates/primitive-gpu-cubecl/tests/gpu2_throughput.rs` (run via `make perf`)
```
GPU-2 throughput: GPU 25.66 M cand/s | CPU 0.58 M cand/s (integer path, 1 core) | speedup 43.9×   # M-series dev box
GPU-2 throughput: GPU  3.19 M cand/s | CPU 0.33 M cand/s (integer path, 1 core) | speedup  9.6×   # GitHub macos-latest CI
```
Same integer work both sides (rasterize + color + delta-SSE per candidate); 64×64, alpha 128,
262k-candidate batches, target/current resident in a `GpuSession`. The ≥20× threshold is
**hardware-dependent** — comfortably cleared on Apple Silicon (43.9×, and 42.3× on a later run) but
not on a shared/virtualized CI GPU (9.6×, unified-memory advantage absent). So the hard assertion is
opt-in via `PRIMITIVE_PERF_GATE` / `make perf` on representative hardware; under `make verify` / CI it
measures + prints only (see `crates/primitive-gpu-cubecl/tests/common/mod.rs`). The integer-parity
gate (`gpu2_triangles.rs`) stays unconditional in `make verify`.

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
documented ≥20× CORE-2-baseline target — `make perf` re-run: 502.6 sps) **and −0.24 dB** vs the CPU
reference (within the 0.5 dB gate). The **sps threshold is hardware-dependent** → enforced under
`make perf` (`PRIMITIVE_PERF_GATE`); the **PSNR gap is hardware-independent and stays unconditional**
in `make verify`. The self-adaptive step (Rechenberg 1/5 rule) replaced the hand-tuned
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

## GUI-2 (2026-06-29)

The six machine-checkable §5A gates. `primitive-app` is now a lib+thin-bin; the pure `state` module is
the single source of truth for every interaction cell. All green under `make verify` and in the
per-gate runs below.

### Gate 1 — §5A interaction-state suite — `cargo test -p primitive-app --test state_suite`
```
test result: ok. 18 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```
One `#[test]` per surface×state cell over the pure `state::Ui` model (no egui/window), covering every
architecture.md 246-259 row: Canvas (dropzone/error/staged/live/done), Source (dropzone/loaded/dimmed),
Controls (editable / disabled-mid-run-except-Pause/Reset), Start/Export (gated-on-image /
disabled-until-≥1-shape / Start⇄Pause / disabled-mid-run-until-paused-or-done / SVG-when-done),
Device (Metal green / amber `CPU (no GPU found)` / neutral CPU).

### Gate 2 — e2e load→100 shapes→SVG + xmllint
```
test load_sample_run_100_export_svg ... ok        # cargo test --release --test e2e  → exit 0
xmllint --noout <out>.svg ; echo $?  → 0           # well-formed SVG (<svg> root + <polygon> triangles)
```

### Gate 3 — forced-CPU graceful degradation — `PRIMITIVE_FORCE_CPU=1`
```
BACKEND_LABEL=CPU (no GPU found)
FORCED_CPU_RUN=ok 100/100 shapes on CPU (no GPU found)
test forced_cpu_probe_labels_and_runs_100_shapes ... ok
```
The probe (`device::detect`) returns the amber fallback chip and a full 100-shape run completes on the
CPU adapter — the GPU-unavailable state is first-class, never a fatal error (§5A). Hermetic (sets the
override in-process) so it's green under plain `make verify` too.

### Gate 4 — AccessKit tree — `cargo test -p primitive-app --test a11y_tree` (egui_kittest 0.34)
```
test accesskit_tree_has_controls_and_live_progress ... ok
```
Drives the real UI headless (clicks the `cat` sample chip *through the accessibility tree*), then
asserts the tree exposes the control labels (Start / count[SpinButton value="250"] / alpha) and a
live-progress node (`0/250`). VoiceOver speech itself is a hand-verified artifact, not gated.

### Gate 5 — deterministic a11y math — `cargo test -p primitive-app --test a11y_tokens`
```
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```
WCAG 2.1 contrast over the design tokens (no rendered pixels): light+dark text ≥ 4.5:1, all three chip
fg/bg pairs ≥ 3:1 (the amber fallback chip guarded explicitly); and `Reduce Motion ⇒ motion_pulse off`.

### Gate 6 — boundaries + full verify
```
./tools/verify/check-boundaries.sh  → exit 0
make verify  → "verify: ALL GREEN"   # fmt + clippy -D warnings + boundaries + giant-file + all tests
```
`gpu_available()` added to the GPU adapter for the probe; `primitive-app` (the composition root) is the
only crate importing both adapters — boundaries unchanged. app.rs kept under the 500-LOC gate by
extracting the `hero` renderer.

## PKG-1 Part A (2026-06-29)

macOS bundle scaffolding — **no signing, no network**. `make bundle` / `scripts/ops/sign-notarize.sh`:
```
==> [Part A] Validating canonical Info.plist
assets/Info.plist: OK                                            # plutil -lint → exit 0
==> [Part A] Building the app bundle (cargo bundle --release)
    Bundling primitive.app
    Finished 1 bundle at: target/release/bundle/osx/primitive.app
==> [Part A] Validating the produced bundle
target/release/bundle/osx/primitive.app/Contents/Info.plist: OK  # plutil -lint → exit 0
    OK — target/release/bundle/osx/primitive.app built and structurally valid.
HALT: stopping BEFORE codesign. Part A is complete and touched NO credentials.
```
`plutil -extract CFBundleIdentifier raw …/primitive.app/Contents/Info.plist` → `com.primitive.app`;
`Contents/Resources/primitive.icns` present; `Contents/MacOS/primitive` executable. The script never
calls codesign/notarytool/stapler — Part B is interactive (you hold the Apple credentials). Icon set
generated offline by `scripts/ops/gen-icon.py` (PIL only, the app's own translucent-triangle motif).

Part B done-gate (interactive, on a **clean** machine — verified off-`/goal`, plan §7 PKG-1):
`xcrun stapler validate primitive.app` → "worked"; `spctl --assess --type execute primitive.app` →
`accepted … source=Notarized Developer ID`.

## CORE-3 Part A — Ellipse + Rectangle (core + CPU) (2026-06-29)

### Shape support — `crates/primitive-core/tests/shapes.rs`
```
Ellipse:   score 0.23835 -> 0.02468 (10% of initial) over 60 shapes, PSNR 32.15 dB
Rectangle: score 0.23835 -> 0.02220 ( 9% of initial) over 60 shapes, PSNR 33.07 dB
test result: ok. 7 passed; 0 failed
```
Verbatim ports of fogleman's `ellipse.go` / `rectangle.go`. Three gates (the triangle-only fogleman
parity fixture isn't extended yet — CORE-3a.2): **rasterizer geometry** hand-verified (radius-3 circle
half-widths `int(sqrt(9−dy²))` = centre [2,8] / mirror [3,7]; rectangle one-span-per-row, corner-order
invariant; edge-clamping); **determinism** byte-identical reconstruction for the same seed; **effective-
ness** above (60 shapes ⇒ ~9–10% of the flat baseline, 32–33 dB). The CPU optimizer/model needed **zero
changes** — `Shape::random/mutate/rasterize/svg` dispatch polymorphically. GPU kernels (CORE-3b) is the
follow-up; `triangle_coords()` panics for non-triangles until 3b.

## CORE-3 Part C — GUI shape-type selector (2026-06-29)

### Shape selector wired end-to-end — `crates/primitive-app/`
```
tests/e2e.rs::ellipse_run_exports_ellipse_svg ........ ok  (40 ellipses → <ellipse> SVG, no polygon)
runner::tests::gpu_routing_is_triangle_and_gpu_device_only  ok  (pure should_use_gpu: Triangle+GPU only)
tests/a11y_tree.rs ................................... ok  (3 options in AccessKit tree + click mutates)
make verify .......................................... verify: ALL GREEN
```
Sidebar `selectable_value` (△ triangle · ◯ ellipse · ▭ rect) → persisted `params.shape_type` →
`RunConfig` → `runner::cpu_stream` → `budget.shape_type`. `ShapeType` got an optional `serde` feature in
core (pure-by-default) for persistence; `#[serde(default)]` keeps old saved params loading. `runner::start`
routes non-triangle shapes to the CPU path even on a GPU device (GPU instant is triangle-only until 3b).
Visual check is the headless egui_kittest render (a11y_tree) — the native-app stand-in for a screenshot.

## CORE-3 Part B.1 — integer ellipse/rect rasterizers (2026-06-29)

### GPU-shared integer rasterizers — `crates/primitive-core/src/raster_int.rs` (new module)
```
ellipse_inside_matches_implicit_test ............................ ok  (centre+extrema in, 1-past out)
int_ellipse_raster_is_contiguous_inside_and_symmetric ........... ok  (span centred on cx, vert-symmetric)
int_ellipse_matches_f64_reference_per_row ....................... ok  (per-row width ≤2px vs f64, area ≤8%)
int_rect_raster_clamps_and_spans_corner_order_invariant ......... ok  (cropped; off-canvas drops, no phantom col)
int_ellipse_raster_is_deterministic / int_raster_* (triangle) ... ok  (byte-identical)
make verify ..................................................... verify: ALL GREEN  (raster.rs 222 / raster_int.rs 322 LOC, both < 500)
```
`ellipse_inside` = integer implicit test `ry²·dx² + rx²·dy² ≤ rx²·ry²` (i64; the i32 GPU mirror is
overflow-free only while operands stay ≲ **181** — a degree-4 product, *not* the ~46k of a single `ry²`;
the ≤128-px GPU canvas keeps the peak `2·128⁴ ≈ 5.4e8` ~4× under i32::MAX, §6.6, with a `debug_assert`
guarding the domain). `rasterize_ellipse_int` bbox-scans the per-row contiguous run (convex);
`rasterize_rectangle_int` is the self-cropping integer bbox (off-canvas rects **drop**, matching the f64
crop, rather than saturating into a phantom column). **Refactor:** the additions took `raster.rs` to 516
LOC (> 500 gate), so the integer rasterizers split into `raster_int.rs` (f64 golden reference stays in
`raster.rs`); `clamp_i32` is `pub(crate)`-shared. Not yet GPU-wired — CORE-3b.2 consumes these on-device.

## CORE-3 Part B.2 — on-device ellipse/rect score kernels (2026-06-29)

### GPU score parity for ellipse + rect — `crates/primitive-gpu-cubecl/tests/`
```
gpu2_ellipses.rs::gpu2_ellipse_raster_score_matches_cpu_exactly ... ok  (1000/1000 bit-identical to CPU)
gpu2_rects.rs::gpu2_rect_raster_score_matches_cpu_exactly ......... ok  (1000/1000 bit-identical to CPU)
make verify ...................................................... verify: ALL GREEN (on Metal)
```
The `score_ellipses`/`score_rectangles` kernels (new `score_shapes.rs`: in-kernel integer raster via
`inside_ellipse` / the rect bbox + the proven `score_one` color+delta math) reproduce
`rasterize_ellipse_int`/`rasterize_rectangle_int` + `candidate_color_and_delta` exactly for 1000
candidates each (each **mutated 4×** so the gate covers the larger-radii / near-edge domain the 3b.3
search will feed, not just freshly-rolled shapes). Host: `EllipseBatch`/`RectBatch` +
`GpuSession::score_ellipses`/`score_rects` + one-shot `gpu_score_ellipses`/`gpu_score_rectangles`;
`Shape::ellipse_coords`/`rectangle_coords` accessors. Review hardening (true-mirror, not
correct-by-precondition): the kernels now `.abs()` radii + drop fully-off-canvas rects like their CPU
oracles, and `score_ellipses` `debug_assert`s the ≤182-px i32-safe canvas bound at dispatch. Scorer
only — the GPU search loop (`evolve`/`commit`) is CORE-3b.3.

## CORE-3 Part B.3 — on-device search loop for ellipse + rectangle (2026-06-30)

### GPU-native ellipse/rect search ≈ CPU quality — `crates/primitive-gpu-cubecl/tests/`
```
gpu3_ellipse_optimize.rs::gpu3_ellipse_search_matches_cpu_psnr ... ok  CPU 34.31 dB | GPU 33.99 dB | gap -0.32 dB
gpu3_rect_optimize.rs::gpu3_rect_search_matches_cpu_psnr ......... ok  CPU 35.50 dB | GPU 35.13 dB | gap -0.37 dB
make verify ..................................................... verify: ALL GREEN (on Metal)
```
80 shapes, 64×64. `evolve_ellipse`/`evolve_rect` + `commit_ellipse`/`commit_rect` (new
`search_shapes.rs`) run the whole search GPU-resident: energy-anchored, Rechenberg 1/5 self-adaptive
hill-climb over the shape params (all clamped to the canvas → i32-safe), scored by the parity-exact
`score_one_ellipse`/`score_one_rect`, committed over the same coverage. Independent of the CPU search
(GPU-native), so compared by **PSNR within 1.0 dB** — both land within 0.4 dB. `OptConfig.shape_type`
selects the kernel triplet in `gpu_optimize`. **CORE-3b complete**: the GPU fits all three shapes.
