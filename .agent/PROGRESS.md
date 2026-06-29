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
- **PKG-1 Part A** (macOS bundle scaffolding — Info.plist, icons, `[package.metadata.bundle]`,
  `scripts/ops/sign-notarize.sh` that HALTS before codesign): autonomous, no credentials.
- **PKG-1 Part B** (codesign → create-dmg → notarytool → staple) + **PKG-2** (Windows): **interactive
  only** (Apple/Windows credentials), never under auto-mode / `/goal`.
