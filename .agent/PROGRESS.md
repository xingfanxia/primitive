# PROGRESS — primitive 2026 rebuild

Milestone naming per plan §7. "Done" = the §7 gate passes via `make verify` (correctness: parity,
golden, PSNR, a11y, boundaries — what CI runs). **Hardware-dependent throughput gates** (GPU-2 ≥20×,
GPU-3 ≥460 sps) run separately via `make perf` on representative hardware — they measure-and-print
under `make verify`/CI but only hard-assert under `PRIMITIVE_PERF_GATE` (a shared CI GPU is too slow
for a portable numeric gate; see `crates/primitive-gpu-cubecl/tests/common/mod.rs`). Evidence is the
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

## GPU-3 — on-device search loop — ✅ DONE (2026-06-27)

The whole optimizer runs **GPU-resident** across all shapes: per step `evolve` (parallel
independent hill-climbs) → `commit` (fused argmin + composite winner into `current`, in place), no
per-step host sync; only the final canvas reads back. New modules: `primitive-core::philox`
(counter-based RNG) and `primitive-gpu-cubecl::search` (`evolve`/`commit`, split from `kernels` for
the size gate).

Each worker seeds its own Philox stream and runs: **energy-targeted restart** (best of 8
high-residual pixels by **L2/SSE** residual — `core::energy_map`'s heuristic, sampled) →
**self-adaptive hill-climb**
(Rechenberg 1/5-rule step: a hit widens the mutation span, a stall narrows it — replaces the
hand-tuned anneal, plan §4) → keep-better, scored by an in-kernel **scanline** raster (analytic
per-row span, O(height) — the GPU-3-only fast path; GPU-2's edge-function scorer stays the
parity-exact one).

Gate evidence (all green in `make verify`, 30 tests):
- **Determinism substrate**: `gpu3_philox.rs` — the kernel RNG is **10000/10000 bit-identical** to
  `primitive_core::rand_below` (pure-u32 Philox; mulhilo matches a u64 reference).
- **End-to-end** (§7 / EVIDENCE target): `gpu3_optimize.rs`, 100 shapes 64×64, workers=10240 age=9:
  **519 shapes/s ≥ 460 target** (≥20× the CORE-2 128×128 baseline) **and PSNR 35.97 dB, −0.24 dB
  vs the CPU run** (within the 0.5 dB gate). (GPU is also ~15× the same-run 64×64 CPU, which is
  unusually fast at that small size — hence the absolute 460 sps cross-resolution target.)

Carry-forward (measured 2026-06-28, see EVIDENCE.md "fogleman-vs-GPU comparison"): the **hi-lo i64 /
large-canvas path is deprioritized on Apple Silicon.** 128×128 already runs correctly on i32 (the
anchored search keeps triangles small, no destructive overflow), but the GPU there *loses* to a
16-core fogleman (unified memory → no bandwidth edge; the parallel search needs ~3× deeper climbs to
hold quality at 4× the pixels). The big-canvas GPU win needs a *discrete* GPU — folded into GPU-4
(CUDA), not a Metal size unlock. Remaining 64×64 polish (energy-map CDF restart, incremental scoring)
is ~1.1–1.3× and optional; a parallel tree-argmin is not the hot path.

## GUI-1 — eframe shell + live canvas — ✅ DONE (2026-06-28)

`primitive-app` crate (the composition root; the binary is `primitive`). One-window single-document
tool per §5A: hero **live canvas** + quiet 320 px sidebar (Source · Shapes · actions), drop/Browse +
bundled samples, count/alpha sliders, Start/Pause/Resume/Reset, PNG/SVG export, device chip.

The optimizer runs on a **background thread** (`runner.rs`) streaming a canvas snapshot + progress
per committed shape; the UI drains to the latest frame each repaint (the §5A spectacle). GUI-1 drives
the **CPU adapter** — the live/watchable backend (the GPU's 519 sps finishes 250 shapes in <1 s, no
spectacle; wiring a GPU "instant" mode is a later increment). Modules: `main`/`app`/`sidebar`/
`runner`/`image_io`, all ≤ size gate.

- **eframe pinned to 0.34** (not 0.35): 0.35 reworked the `App` trait to `ui()`-only and dropped
  `SidePanel`/`TopBottomPanel`. 0.34.3 already made `ui(&mut self, ui, frame)` the required method —
  panels are shown *inside* the root `ui` via `show_inside`, ctx via `ui.ctx()`.
- **Gate evidence** (machine-checkable, plan §5A — fps/zero-copy need a screenshot, not gated):
  `runner::tests::smoke_load_run_export_svg` decodes a bundled image → runs 25 shapes → exports SVG
  (`<svg` root + `polygon` elements). Green in `make verify`. Launch: `cargo run -p primitive-app --release`.

Carry-forward (GUI-2): full §5A interaction-state polish (toasts/Reveal-in-Finder, mid-run SVG, sample
thumbnails, keyboard shortcuts, AccessKit a11y, Reduce-Motion); optional GPU "instant" run mode.

## GUI-2 — §5A interaction states, a11y, export, GPU chip — ✅ DONE (2026-06-29)

`primitive-app` split into a **lib + thin bin** so the interaction logic, a11y math, device probe, and
end-to-end run are unit-testable headless. New modules: `state` (the **pure**, render-free §5A model —
no egui/IO), `theme` (design tokens + WCAG contrast math), `device` (backend probe), `i18n` (string
table), `hero` (canvas/chip/progress renderer, split from `app` for the size gate). The CPU adapter is
still the live spectacle; the GPU adapter is now wired for **detection** (the device chip) + an
**"instant" mode** (`gpu_optimize`, raster-only, ≤128 px) — `primitive-gpu-cubecl::gpu_available()`
added as the probe. New §5A features: keyboard shortcuts (⌘O/Space/⌘E/⌘R/⌘,), Advanced disclosure
(seed/n/age/m + Reduce-Motion), Export ▾ (PNG·SVG·GIF), window+param persistence (eframe `persistence`).

Gate evidence (all 6 machine-checkable gates green; full detail in EVIDENCE.md → GUI-2):
- **§5A state suite** (`tests/state_suite.rs`): **18/18** — one `#[test]` per surface×state cell over
  the pure `state::Ui` model (Canvas · Source · Controls · Start/Export · Device), every architecture.md
  246-259 row covered.
- **e2e** (`tests/e2e.rs`): load sample → 100 shapes → export SVG, then `xmllint --noout` — **both exit 0**.
- **forced-CPU** (`tests/forced_cpu.rs`): `PRIMITIVE_FORCE_CPU=1` → prints `CPU (no GPU found)` + 100/100.
- **AccessKit tree** (`tests/a11y_tree.rs`, egui_kittest): control labels + a live-progress node exist.
- **a11y tokens** (`tests/a11y_tokens.rs`): WCAG AA — text ≥ 4.5:1, chips ≥ 3:1; Reduce-Motion pulse off.
- **boundaries** (`check-boundaries.sh`) exit 0; **`make verify` ALL GREEN** (fmt/clippy/giant-file/tests).

Genuinely-visual items (live ≥30 fps, VoiceOver speech, amber-chip colour, GIF playback) are
hand-verified artifacts, never in the predicate (the headless harness can't see a window).

Carry-forward: the GPU "instant" mode is wired + defensive (catch_unwind → CPU fallback) but is an
ungated artifact (needs a GPU window to watch). Out of scope by design: the multi-shape-type selector
(core implements only `Triangle` — that's a CORE milestone, not GUI).

## Next: GPU-4 (different machine) / PKG

- **GPU-4: CUDA on a discrete NVIDIA GPU — see `docs/gpu4-cuda/RUNBOOK.md`** (runs on another machine;
  settles the big-canvas GPU-vs-fogleman number the Mac can't produce). i64 is native on CUDA.
- **PKG-1 Part B** (codesign → notarytool → staple) is now **scripted** — `make sign` / `scripts/ops/
  sign-notarize.sh --sign`, runbook `docs/pkg/RUNBOOK.md`. Still credential-gated (your Apple creds via
  `SIGN_IDENTITY` + `NOTARY_PROFILE`) and never runs without `--sign`, so it stays off auto-mode / `/goal`;
  the *authoring* is done. **PKG-2** (Windows): still interactive/other-machine.

## PKG-1 Part A — macOS bundle scaffolding (no signing, no network) — ✅ DONE (2026-06-29)

`assets/Info.plist` (canonical, plutil-lintable: name `primitive`, id `com.primitive.app`, version
0.1.0, LSMinimumSystemVersion 11.0, category `public.app-category.graphics-design`), a flat-geometric
icon set `assets/icons/icon_{512..16}.png` + `primitive.icns` (the app's own translucent-triangle
motif, generated offline by `scripts/ops/gen-icon.py` — no gpt-image/network), `[package.metadata.bundle]`
in `crates/primitive-app/Cargo.toml`, and `scripts/ops/sign-notarize.sh` (also `make bundle`) that
builds + validates the `.app` then **HALTS before the first `codesign` step** (default / no-arg mode
touches no credentials and no network).

Gate evidence (no credentials touched): `plutil -lint assets/Info.plist` → `OK`; `cargo bundle --format
osx` → `target/release/bundle/osx/primitive.app` (generated Info.plist id `com.primitive.app`,
`primitive.icns` in Resources, executable present); the script halts at the codesign boundary.

**PKG-1 Part B — sign/notarize/staple is now scripted (2026-07-01):** the same
`scripts/ops/sign-notarize.sh`, invoked with `--sign` (`make sign`), runs the full Developer ID chain —
codesign **inside-out** (hardened runtime + secure timestamp, **no `--deep`**, no entitlements), notarize
via `xcrun notarytool` (keychain profile or Apple-ID trio), staple **both** the `.app` and the `.dmg`,
then the done-gate `stapler validate` + `spctl --assess … = Notarized Developer ID`. Credential-gated
(`SIGN_IDENTITY` + `NOTARY_PROFILE`/`APPLE_ID`…), so it stays off auto-mode / `/goal`. One-time setup +
troubleshooting: `docs/pkg/RUNBOOK.md`. (Commands verified against current Apple docs — TN3147 notarytool,
TN2206 `--deep` deprecation, hardened-runtime entitlements guidance.)

## CORE-3 Part A — Ellipse + Rectangle (core + CPU) — ✅ DONE (2026-06-29)

The first slice of the architecture's shape-set expansion (§10 Q5: "triangle + ellipse + rect first,
expand after PKG-1"). `core::shape` gains `Ellipse` (x,y,rx,ry) and axis-aligned `Rectangle`
(x1,y1,x2,y2) variants — verbatim ports of fogleman's `ellipse.go` / `rectangle.go` (random/mutate/
rasterize/svg), with `rasterize_ellipse` / `rasterize_rectangle` reference rasterizers added to
`core::raster`. The `Shape`/`ShapeType` enums + dispatch are extended; because the optimizer/model
call `Shape::random(t, …)` / `shape.mutate/rasterize/svg` polymorphically, **the entire CPU search +
SVG export works for the new shapes with zero changes to `optimizer.rs` / `model.rs`**.

Gate evidence (all green in `make verify`; `crates/primitive-core/tests/shapes.rs`, 7 tests):
- **Rasterizer geometry** — hand-verified scanline spans for a radius-3 circle (`int(sqrt(9−dy²))`
  half-widths: centre row [2,8], mirrored rows [3,7]), a non-circular ellipse (rx=6,ry=3 ⇒ aspect=2,
  pins the `* aspect` term a circle would hide), and a sorted-corner rectangle (one span/row);
  edge-clamping checked.
- **Determinism** — same seed + target ⇒ **byte-identical reconstruction** (score + `current.pix`)
  for both Ellipse and Rectangle.
- **Effectiveness** — 60 shapes on a 48×48 synthetic target cut the score to **~10%/9% of the
  flat-background baseline** (PSNR 32.15 / 33.07 dB); SVG export emits valid `<ellipse>` / `<rect>`.

Scope notes (carry-forward, by design):
- **CORE-3b (GPU) ✅ DONE** (all three): **3b.1** integer rasterizers, **3b.2** on-device score kernels,
  **3b.3** on-device search loop (`evolve_ellipse`/`evolve_rect` + `commit_*`) — see Part B.1/B.2/B.3.
  The GPU now fits triangles, ellipses, and rects fully on-device (gated by PSNR within tolerance of the
  CPU search). The app keeps routing ellipse/rect to the CPU streaming path by default (it preserves the
  SVG export + live spectacle the raster-only GPU instant mode can't) — flipping `runner::should_use_gpu`
  to offer a GPU fast-preview for the new shapes is now a one-line guard change, deferred as a UX call.
- **CORE-3c (GUI)**: ✅ DONE — see the Part C section below.
- **CORE-3a.2 (rigour)**: fogleman **bit-exact** parity for ellipse/rect needs the Go dumper
  (`go_primitive/primitive/parity_dump_test.go`, schema is triangle-only) extended; until then the
  three gates above pin correctness (the verbatim port + hand-checked rasters make a hidden math
  divergence unlikely, but the fixture is the gold standard the triangles already meet).

## CORE-3 Part C — GUI shape-type selector — ✅ DONE (2026-06-29)

The sidebar's placeholder "△ triangle (more soon)" row becomes a real 3-way selector
(`sidebar.rs`, egui `selectable_value` over `△ triangle · ◯ ellipse · ▭ rect`) bound to a new
persisted `params.shape_type`. The choice is plumbed Sidebar → `Params` → `RunConfig` →
`runner::cpu_stream` → `budget.shape_type`, so ellipse/rect run live end-to-end on the CPU adapter
(the watchable backend). `ShapeType` gained an **optional `serde` feature** in core (off by default —
keeps core pure) so the app can persist it; `#[serde(default)]` keeps pre-CORE-3c saved params loading.

**GPU guard**: `runner::start` routes any non-triangle shape to the CPU path even on a Metal/CUDA
device (`gpu = is_gpu_device && shape_type == Triangle`) — GPU instant mode is triangle-only until
CORE-3b. The CPU path is the live spectacle anyway, so ellipse/rect are fully usable now.

Gate evidence (all green in `make verify`):
- **e2e** (`tests/e2e.rs::ellipse_run_exports_ellipse_svg`): Ellipse selected → 40-shape run through
  the real `runner` → SVG carries `<ellipse>` and **no** `polygon` (proves the type is plumbed).
- **GPU-guard unit** (`runner::tests::gpu_routing_is_triangle_and_gpu_device_only`): the pure
  `should_use_gpu(device, shape)` decision — GPU instant only for Triangle on a GPU device; ellipse/rect
  always CPU. Hardware-independent, so deleting the guard fails on any runner (a code-review fix:
  the prior `<ellipse>`-observable test could be masked by the `catch_unwind → cpu_stream` fallback).
- **a11y render + mutation** (`tests/a11y_tree.rs`): the headless egui_kittest render exposes the
  `△ triangle` / `◯ ellipse` / `▭ rect` options in the AccessKit tree, **and** clicking `◯ ellipse`
  flips `params.shape_type` — the native-app equivalent of a screenshot + interaction gate.
- Existing §5A state suite (18) + forced-CPU + a11y-tokens unchanged & green.

Carry-forward: GPU instant mode for ellipse/rect is CORE-3b. The device chip still reads `Metal` when a
non-triangle run is silently on CPU — cosmetic (the chip is GPU *availability*, not active backend);
revisit if it confuses.

## CORE-3 Part B.1 — integer ellipse/rect rasterizers (GPU-shared path) — ✅ DONE (2026-06-29)

The foundation for the GPU shape-set expansion: the §6.6 integer rasterizers ellipse/rect need so the
GPU kernel and its CPU oracle cover the **exact same pixels** (Metal has no f64). Mirrors the existing
triangle path (`rasterize_triangle_int` / `triangle_inside`):

- **`ellipse_inside(cx, cy, rx, ry, px, py)`** — the integer implicit test `ry²·dx² + rx²·dy² ≤ rx²·ry²`
  (no `sqrt`; the integer analogue of `triangle_inside`). Computed in `i64`: `ry²·dx²` overflows `i32`
  past ~46k px, so the GPU mirror (i32) is valid within the documented canvas bound (≤128 px, §6.6),
  where the two agree bit-for-bit (no overflow).
- **`rasterize_ellipse_int`** — bbox-scan emitting the per-row contiguous run (ellipses are convex), the
  integer counterpart of fogleman's f64 `rasterize_ellipse`.
- **`rasterize_rectangle_int`** — self-cropping integer bbox (already all-integer; trivial GPU mirror).

**Refactor (size gate, not allowlist):** adding these pushed `raster.rs` to 516 LOC (> 500 hard gate),
so the integer rasterizers split into a new **`raster_int.rs`** module — the seam the file's own comment
already drew (f64 golden reference in `raster.rs` ← → integer GPU-shared path in `raster_int.rs`). Both
files now ~220–290 LOC. `clamp_i32` is `pub(crate)` so both share one clamp.

Gate evidence (all green in `make verify`; `crates/primitive-core/src/raster_int.rs` tests, 7 total):
- **`ellipse_inside_matches_implicit_test`** — centre + 4 axis extrema inside/on-boundary; one step past
  each axis and the bbox corner are outside.
- **`int_ellipse_raster_is_contiguous_inside_and_symmetric`** — every emitted pixel inside, maximal run,
  each row's span centred on `cx` (`x1+x2 == 2·cx`), vertical symmetry (`cy−k`/`cy+k` identical).
- **`int_ellipse_matches_f64_reference_per_row`** — per-row span widths agree with fogleman's f64 raster
  to ≤ 2px (a real shape oracle: total-area alone can't catch a shifted/distorted ellipse), ≤ 2 extreme
  rows of coverage mismatch, total area within 8%.
- **`int_rect_raster_clamps_and_spans_corner_order_invariant`** — full-width span/row, corner-order
  invariant, cropped to the image (never out of bounds).
- determinism (ellipse + the carried-over triangle int tests) byte-identical.

Carry-forward: these are **not yet GPU-wired** — CORE-3b.2 generalizes the score kernel to consume them
on-device with GPU↔CPU bit-identical parity (like GPU-2 for triangles), then 3b.3 the `evolve`/`commit`
search loop. Minor debt (review NIT, deferred to avoid a rename-only change): `shape.rs` has a private
`clamp_i32` byte-identical to `raster.rs`'s now-`pub(crate)` one — fold the duplicate into the next
`shape.rs` touch.

## CORE-3 Part B.2 — on-device score kernels for ellipse + rectangle — ✅ DONE (2026-06-29)

The GPU can now **rasterize + score ellipse & rectangle candidates on-device**, bit-exact against the
CPU oracle — the GPU-2 triangle path generalized to the new shapes. New module
`crates/primitive-gpu-cubecl/src/score_shapes.rs` (keeps `kernels.rs` under the 500-LOC gate):

- **`inside_ellipse`** (`#[cube]`) mirrors `raster_int::ellipse_inside` in i32 (overflow-free ≤128 px).
- **`score_one_ellipse`** / **`score_one_rect`** mirror `kernels::score_one` (closed-form color +
  16-bit premultiplied composite + integer delta-SSE) with the ellipse bbox + `inside_ellipse` / the
  rect bbox (every bbox pixel inside, no per-pixel test). Reuse the proven `channel_delta`/`clamp_0_255`
  helpers, so the parity that holds for triangles carries by construction.
- **`score_ellipses`** / **`score_rectangles`** batch kernels (4 ints/candidate), one thread each.
- Host: `EllipseBatch`/`RectBatch` + `GpuSession::score_ellipses`/`score_rects` + one-shot
  `gpu_score_ellipses`/`gpu_score_rectangles`. New `Shape::ellipse_coords()`/`rectangle_coords()`
  accessors (the `[cx,cy,rx,ry]`/`[x1,y1,x2,y2]` flat layouts the kernels consume).

Gate evidence (green in `make verify`, on Metal):
- **`gpu2_ellipses.rs`** — `score_ellipses` == `rasterize_ellipse_int` + `candidate_color_and_delta`,
  **1000/1000 candidates bit-identical** to the CPU integer path.
- **`gpu2_rects.rs`** — `score_rectangles` == `rasterize_rectangle_int` + `candidate_color_and_delta`,
  **1000/1000 candidates bit-identical**.

Cleanup folded in (3b.1 review NIT): `shape.rs`'s private `clamp_i32` removed — now uses the shared
`raster::clamp_i32` (`pub(crate)`).

Carry-forward: this is the **scorer** only. CORE-3b.3 generalizes the GPU *search loop*
(`search::evolve`/`commit` — in-kernel random/mutate per shape type) so the app's GPU instant mode can
drop the triangle-only guard and run ellipse/rect fully on-device.

## CORE-3 Part B.3 — on-device search loop for ellipse + rectangle — ✅ DONE (2026-06-30)

The **whole search runs GPU-resident** for ellipses and rects now, not just triangles — the GPU-3
triangle loop generalized. New module `crates/primitive-gpu-cubecl/src/search_shapes.rs`:

- **`evolve_ellipse`** / **`evolve_rect`** — one independent hill-climb per worker: a shared
  energy-targeted `anchor` (best-of-8 high-residual pixels) seeds the centre/first-corner; small random
  radii / opposite corner; then `age` **Rechenberg 1/5 self-adaptive** perturbations of one param
  (`{cx,cy,rx,ry}` / the four corners), each clamped to the canvas (radii to `[1,w-1]` — keeps the i32
  ellipse test overflow-safe, §6.6), keeping a change only if it lowers the delta-SSE (scored by the
  parity-exact `score_one_ellipse`/`score_one_rect`). Writes best delta + best 4-int params.
- **`commit_ellipse`** / **`commit_rect`** — fused argmin + composite the improving step winner over the
  bbox+`inside_ellipse` / bbox coverage, in place, no host sync. Mirrors triangle `commit`.
- Host: `OptConfig.shape_type` selects the kernel triplet; `gpu_optimize` matches on it (best-param
  buffer 6 ints for triangle, 4 for ellipse/rect). Like GPU-3, the GPU is a **native** optimizer, not a
  replay of the CPU search — compared by **PSNR**, not pixels.

Gate evidence (hardware-independent → unconditional in `make verify`):
- **`gpu3_ellipse_optimize.rs`** — 80 shapes, 64×64: **GPU 33.99 dB vs CPU 34.31 dB, gap −0.32 dB**
  (≤ 1.0 dB tolerance).
- **`gpu3_rect_optimize.rs`** — 80 shapes, 64×64: **GPU 35.13 dB vs CPU 35.50 dB, gap −0.37 dB**.

App: `runner::gpu_instant` plumbs `cfg.shape_type` into `OptConfig`; the `should_use_gpu` guard stays
triangle-only by choice (ellipse/rect keep the CPU path's SVG + streaming). **CORE-3b is complete** —
the GPU fits all three shapes.
