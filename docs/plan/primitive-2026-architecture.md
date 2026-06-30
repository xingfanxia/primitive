# Primitive 2026 — GPU-Native Rebuild Architecture Plan

> Status: **proposed** · Date: 2026-06-27 · Author: research synthesis (2 verified multi-agent workflows)
> Scope: rebuild `fogleman/primitive`-style geometric image reproduction in a performant language,
> with real CUDA + Apple GPU acceleration and a packaged native macOS GUI app (Windows later).

---

## 0. The ask, restated

Reproduce a target image with N geometric primitives (triangles / ellipses / rects / …), the way
[`fogleman/primitive`](https://github.com/fogleman/primitive) does — but:

1. **Performant language** — Go or Rust (pick the better, justify it).
2. **Real GPU acceleration** — CUDA (NVIDIA) **and** the Apple stack (Metal / MPS; the brief also said
   "CoreML" — this plan resolves whether that's the right tool).
3. **A modern, simple GUI**, packaged as a native, signed macOS `.app`. Mac first, Windows later.

And two follow-ups the brief added:

4. **Is there a better / more elegant algorithm** than the original hill-climb? (→ §4)
5. **Is there a better way to implement** the algorithm? (→ §6)

---

## 1. Executive summary

| Axis | Decision |
|---|---|
| **Language** | **Rust** |
| **GPU compute** | **CubeCL** (`#[cube]` kernels → one source compiles to **CUDA** *and* **Metal** + Vulkan/DX12/WGSL/CPU-SIMD) |
| **Apple path** | **Metal compute shaders** (reached via CubeCL/wgpu). **Not CoreML** — that's a category error (see §3). |
| **GUI** | **egui / eframe** (wgpu renderer), sharing one `wgpu::Device` with the compute core → **zero-copy GPU→display** |
| **Algorithm** | Keep the **massively-parallel random-restart** skeleton; replace hand-tuned hill-climb+SA with **self-adaptive (1+1)-ES** + **energy-map restart sampling** |
| **Implementation** | One **fused, fully GPU-resident kernel** (mutate→raster→color→delta-score→argmin); **integer delta-SSE** for bitwise cross-backend determinism |
| **Packaging** | `cargo-bundle` + `codesign` + `xcrun notarytool` + `create-dmg` (Mac); Windows installer later |

**Why now:** an exhaustive search confirms **no GPU-accelerated reimplementation of the primitive/geometrize
hill-climb exists** in any language. And as of 2026 the Rust toolchain finally makes the hard part tractable
in *one* language: CubeCL gives single-source CUDA+Metal compute, eframe defaults to wgpu so the GUI and the
compute core can share a device, and notarization is fully scriptable. The Python/MPS port that exists today
is *slower than CPU* (§2) — this design exists specifically to make "GPU" mean "much faster," not "experimental."

---

## 2. Diagnosis — why the current Python/MPS port is slower than CPU

The existing PyTorch + MPS port (`py_primitive/`) is documented in its own README as *"experimental… currently
performs slower than CPU for most workloads."* This is **structural, not a tuning problem.** Confirmed by reading
the code (`optimizer.py`, `shapes.py`, `model.py`):

1. **Per-candidate rasterization in a Python loop** — `masks[i] = shape.to_tensor()` allocates a *full H×W* GPU
   tensor per candidate and launches its own kernel. Thousands of candidates per shape ⇒ thousands of tiny dispatches.
2. **Color & score computed per-shape, per-channel in Python loops** (`for i in range(batch): … for c in range(3): …`)
   — more micro-kernels, each doing trivial work.
3. **Full-image MSE recomputed every step** (`_compute_score` over the whole image after each shape) — O(H·W) when
   only the shape's bounding box changed.
4. **Host↔device sync per candidate** — `.item()` / `.cpu().numpy()` / Python control flow stalls the pipeline.

Net effect: **kernel-launch + sync overhead dominates** the (tiny) useful work, so the GPU sits idle between
launches and loses to a cache-friendly CPU loop. The rebuild fixes all four at the root: **one fused dispatch over
the whole candidate batch, bounding-box-local delta scoring, everything resident on-device, and a single tiny
readback per committed shape** (§5–§6).

---

## 3. Decision matrix

| Axis | Chosen | Runner-up | Why |
|---|---|---|---|
| **Language** | **Rust** | Go | Go's only GPU compute ecosystem, **GoGPU** (launched Dec 18 2025), is v0.x with **no CUDA backend** — a hard fail on the explicit CUDA requirement — and CGO cross-compilation friction blocks clean signed Mac→Windows builds. Rust has a mature multi-backend GPU stack (CubeCL/wgpu/cudarc) *and* best-in-class native packaging. |
| **GPU compute** | **CubeCL** (`#[cube]`) | raw wgpu + WGSL | CubeCL is the *only* path that gives a **native CUDA runtime** (`cubecl_cuda` + `cudarc`) **and** native Metal from **one kernel source**. Raw wgpu on NVIDIA runs Vulkan/DX12, *not* CUDA — which fails the literal CUDA ask. Pin **0.9.x stable** (0.10 is pre-release as of this writing). |
| **Apple path** | **Metal compute** (via CubeCL→wgpu `MSL_SHADER_PASSTHROUGH`) | `objc2-metal` direct | Metal is wgpu's **default, first-class native backend** on macOS (MoltenVK is opt-in, not the default). Apple-Silicon **unified memory** is a real zero-copy advantage for keeping target+canvas+candidates resident. We never touch Metal directly — wgpu insulates us from the `metal-rs`→`objc2-metal` churn. |
| **CUDA path** | **CubeCL `cubecl_cuda` + cudarc** | hand-written PTX | `cudarc` is actively maintained (0.19.x through mid-2026). The same `#[cube]` kernel emits PTX/NVVM — no second kernel codebase. |
| **GUI** | **egui / eframe** (≥0.34, wgpu default) | **Slint** (≥1.17) | eframe defaults to wgpu; `WgpuSetupExisting` shares our `Device`, and `CallbackTrait::{prepare,paint}` samples the compute-output texture inside the egui render pass — true zero-copy. Slint is a *genuine* runner-up now (it shipped wgpu device injection, `slint::wgpu_28` / PR #8278), but egui's immediate-mode model is simpler for a live-updating canvas. **Rejected:** Tauri/Wails (WebView ⇒ CPU roundtrip), Iced (Elm message-queue awkward for live GPU), Fyne (own GL stack, can't share `wgpu::Device`). |
| **Algorithm** | **(1+1)-ES + energy-map restarts** over a parallel candidate batch | differentiable rendering (DiffVG / Bézier Splatting) | Discrete, embarrassingly parallel across candidates ⇒ perfect for batch dispatch, and more *elegant* than the original (deletes the annealing schedule). The gradient methods are genuinely smarter but **CUDA-only** (§4) — they fail the Metal requirement. |
| **Packaging** | `cargo-bundle` + `codesign` + `notarytool` + `create-dmg` | `cargo-codesign` (one-shot) | Scriptable, well-established native chain. Requires a paid Apple Developer account ($99/yr). |

### Resolving the "CoreML / MPS" ask honestly

**CoreML is the wrong tool — a category error, not an option to weigh.** CoreML is an *ML-model inference runtime*
(it runs pre-trained models on the ANE/GPU, layered over MPSGraph). It exposes **no rasterization primitives and no
arbitrary custom compute kernels** — you cannot author your own node types. **MPS / MPSGraph** similarly expose only
fixed primitives (conv, FFT, sort, linear algebra) and cannot express scanline rasterization of arbitrary geometry.
The correct Apple API for *this* workload (parallel rasterize + reduce) is **Metal compute shaders (MSL kernels)**,
which we reach via wgpu/CubeCL. **We will not use CoreML or the Neural Engine anywhere in this project.** This is the
honest answer to the brief's "MPS/CoreML" phrasing: MPS-the-concept (Metal GPU compute) yes; the CoreML/MPSGraph
*frameworks* no.

---

## 4. Algorithm strategy — *is there a more elegant algorithm?* (answer to follow-up #4)

**Verdict: keep the massively-parallel random-restart structure — it is the right skeleton — but replace the
hand-tuned hill-climb + simulated-annealing inner loop with self-adaptive (1+1)-ES, and seed restarts from a
residual energy map.** This is *more elegant than the original, not a different paradigm*: it **deletes magic
numbers** (annealing schedule, fixed mutation step) rather than adding machinery.

First-principles reasoning:

- **The search is low-dimensional.** Closed-form fill color removes color from the search entirely, so each
  candidate optimizes only geometry: triangle ≈ 6 params, ellipse ≈ 5, rotated-rect ≈ 5, circle ≈ 3. For black-box
  optimization in <10 dims with a cheap delta-scored objective, **evolution strategies are the canonical tool.**
- **Hill-climbing *is* a degenerate (1+1)-ES** with a fixed step and no adaptation. Upgrading to (1+1)-ES with the
  **1/5th success rule** (grow the mutation step when >1/5 of mutations improve, shrink otherwise) gives
  annealing-like behavior *for free* and removes Fogleman's hand-tuned temperature schedule. ~20 lines, not a framework.
- **Per-shape ES is independently validated.** *CamoPatch* (Williams & Li, NeurIPS 2023) optimizes per-shape
  RGB + circle geometry with exactly **(1+1)-ES** — direct precedent for ES on opaque primitive+geometry fitting.
- **Energy-map initialization is validated and cheap.** *ProHC* (Zhang/Wei/Wang, *Biomimetics* 2023,
  doi:10.3390/biomimetics8020174) biases polygon placement from an energy/saliency map and adds polygons
  progressively, beating vanilla hill-climbing. Concretely: **bias restart sampling toward high-residual pixels**
  (use the current per-pixel SSE as a sampling PDF) so candidates are spent where error actually lives.

### Decision table

| Option | Cross-platform (CUDA+Metal) | Elegance / LOC | Per-shape convergence | Evidence | Verdict |
|---|---|---|---|---|---|
| Hill-climb + SA (Fogleman baseline) | ✅ trivial | medium (tuned schedule, magic numbers) | baseline | original `primitive` | Baseline — beat it |
| **(1+1)-ES + 1/5-rule + energy-map init** | ✅ trivial (no autodiff) | **highest** (deletes the schedule) | better than HC, fewer wasted restarts | CamoPatch, ProHC | **CHOSEN** |
| CMA-ES per worker | ✅ (batchable on GPU) | medium-high | best sample-efficiency, higher per-step cost | standard ES literature | **Optional upgrade** |
| DiffVG / Bézier Splatting (gradient) | ❌ **CUDA-only** | low (heavy autodiff) | best | DiffVG (SIGGRAPH Asia 2020); Bézier Splatting (NeurIPS 2025) | **Rejected** — fails Metal req |
| Paint Transformer (learned feed-forward) | ⚠️ needs trained net + runtime | low to infer, high to train/port | fast but not error-minimizing | ICCV 2021 | Out of scope (learned, not optimization) |

**Why the strongest alternative is rejected (not on preference, on a hard constraint):** differentiable
rasterization (gradient descent *through* the rasterizer) is the genuinely "smarter" optimizer and converges in
far fewer iterations per shape. But **DiffVG is CUDA-only** — its entire issue tracker is CUDA/Torch build problems
and no Metal backend exists anywhere in the ecosystem as of mid-2026. Its successor **Bézier Splatting** (NeurIPS
2025; reports ~31× forward / ~149× backward speedups vs DiffVG — *ratios verified, absolute timings not*) is a
research prototype benchmarked on a single RTX 4090 and inherits the same gradient-through-rasterizer Metal gap.
A hand-rolled differentiable rasterizer in CubeCL is a multi-month research project; the ES path delivers
cross-platform parity *now*.

**Optional high-value upgrades** (priority order): (1) energy-map restart sampling — biggest win for least effort,
folds into the existing loop; (2) CMA-ES per worker when a primitive type plateaus (GPU-batchable: one covariance
per worker); (3) progressive primitive scheduling (ProHC) — large coarse shapes first, shrink the size prior as
global error drops so late shapes refine detail.

> **Evidence caveat (be honest in the build):** ES-CLIP (Tian & Ha, EvoMUSART 2022) reports ES "comparable to
> gradient-based methods" *on its own CLIP metric with transparent alpha-blended triangles* — **not** a controlled
> head-to-head vs DiffVG on identical images with opaque solid fills. So treat "ES ≈ gradient quality" as
> *suggestive, not proven* for this exact objective. The CORE-1 golden-image gate (§7) is what actually proves our
> quality, not these papers.

---

## 5. Target architecture (hexagonal)

```
primitive/                       # cargo workspace
├─ crates/
│  ├─ primitive-core/            # PURE domain — no GPU, no IO, no wgpu
│  │   ├─ shape.rs               # Shape enum (Triangle, Ellipse, Rect, RotRect, Circle, Line, …)
│  │   ├─ raster_ref.rs          # scalar CPU reference rasterizer (golden oracle)
│  │   ├─ color_solve.rs         # closed-form optimal-color math (pure fn)
│  │   ├─ score.rs               # bbox delta-SSE math (pure fn)
│  │   ├─ es.rs                  # (1+1)-ES + 1/5-success-rule state machine (pure, RNG injected)
│  │   └─ energy_map.rs          # residual→sampling-PDF (pure)
│  │
│  ├─ primitive-compute/         # PORTS — traits the engine calls (no concrete backend)
│  │   └─ lib.rs                 # trait GpuCompute { upload_target, eval_batch, commit_shape,
│  │                             #   readback_best, preview_texture } + DTOs
│  │
│  ├─ primitive-gpu-cubecl/      # ADAPTER — CubeCL #[cube] kernels (CUDA + Metal + WGSL)
│  │   ├─ kernels/raster_score.rs    # fused mutate+raster+color+delta-score #[cube]
│  │   ├─ kernels/reduce_argmin.rs   # parallel best-candidate reduction
│  │   ├─ kernels/commit.rs          # blend winner into canvas + update total_sse (on-device)
│  │   └─ runtime.rs             # device/queue mgmt, buffer residency, autotune
│  │
│  ├─ primitive-gpu-cpu/         # ADAPTER — CPU/Rayon impl of GpuCompute (test oracle + fallback)
│  │
│  ├─ primitive-engine/          # APPLICATION — orchestrates the loop through ports
│  │   └─ lib.rs                 # add_n_shapes(), per-shape ES driver, progress events
│  │
│  └─ primitive-app/             # GUI SHELL (eframe) — the composition root
│      ├─ main.rs                # builds ONE wgpu Device, injects into BOTH egui + compute adapter
│      ├─ canvas_view.rs         # CallbackTrait paint of compute texture (zero-copy)
│      └─ controls.rs            # shape type / count / alpha / params / load / export
│
├─ assets/Info.plist, icons/
├─ scripts/ops/sign-notarize.sh
└─ tools/verify/check-boundaries.sh   # CI lint: app/engine must not import adapters directly
```

**Import direction (enforced by `check-boundaries.sh` in CI):**
`core` (pure) ← `compute` (ports) ← `engine` (orchestration) ← adapters / `app` (composition root).
`core` imports nothing GPU; concrete adapters are imported *only* at `primitive-app/main.rs`.

**One device, shared.** `primitive-app/main.rs` creates a single `wgpu::Instance/Adapter/Device/Queue` and hands the
same `Device`+`Queue` to (a) eframe via `WgpuSetupExisting` and (b) the CubeCL adapter. On Metal both share the wgpu
device, so display is genuinely zero-copy. On the **CUDA** path, `cubecl_cuda` owns its own context; the GUI still
renders via wgpu, so the *small* preview frame is copied across (heavy buffers stay on the CUDA device) — a conscious,
bounded tradeoff (Open Q #6).

**GPU-residency contract.** Target image, current canvas, the current per-pixel error buffer, and the precomputed
integral tables are uploaded **once** and never leave the GPU. Candidate params, scores, and best-index live
on-device. Host reads back only (1) the winning shape's params per committed shape (a dozen values, for SVG/log),
and (2) a **throttled** preview texture for the GUI (~16 ms cadence, not per candidate).

**Zero-copy display.** The compute kernel writes the canvas into a `wgpu::Texture` with `TEXTURE_BINDING`; egui's
`CallbackTrait::paint` binds that exact texture in the egui render pass. The documented `prepare → finish_prepare →
paint` cycle means the pixels are never copied to host on the Metal path.

---

## 5A. GUI / UX design (the "simple Mac app")

The brief asks for a *modern, simple* GUI — "simple" here means **subtraction by default** (Rams), not "unfinished."
A primitive run is a *spectacle*: the whole appeal is watching an image assemble itself from shapes. The single most
important design decision is therefore that **the live reconstruction canvas is the hero** and everything else is a
quiet sidebar. The app is a one-window, single-document tool — no tabs, no navigation, no dashboard.

### Information architecture (one window, two regions)

```
┌──────────────────────────────────────────────────────────────┐
│  primitive                                            ⓘ  ⚙︎    │  title bar (native traffic lights)
├───────────────────────────────┬──────────────────────────────┤
│                               │  SOURCE                       │
│                               │   [ drop image · or Browse ]  │
│                               │   thumbnail · 1920×1080       │
│        LIVE CANVAS            │                               │
│        (the hero —           │  SHAPES                       │
│         current reconstruction│   type  ◉△ ○◯ ○▭ ○rot        │
│         renders here,         │   count ▕━━━━●━━▏ 250         │
│         updating live)        │   alpha ▕━━●━━━━▏ 128         │
│                               │                               │
│        ▕███████░░░░▏ 142/250  │  ADVANCED  ▸ (collapsed)      │
│        12,400 shapes/sec      │   candidates · ES steps · seed│
│                               │                               │
│                               │  ───────────────────────────  │
│                               │   ▶ Start    ⏸ Pause   ↺ Reset│
│                               │   ⤓ Export ▾ (PNG·SVG·GIF)    │
└───────────────────────────────┴──────────────────────────────┘
```

**Visual hierarchy (what the eye hits 1-2-3):** (1) the live canvas; (2) the primary action button (`Start`, which
becomes `Pause` mid-run); (3) the three core controls (shape type, count, alpha). Everything else — advanced params,
export format menu, device/perf readout — is tertiary and visually recessed. Power-user knobs (candidate batch size,
ES steps, seed, deterministic mode) live behind a collapsed **Advanced** disclosure so the default surface stays at
three sliders.

**Not a 3-column card grid, not a hero-with-features page** (the AI-slop traps): it's a focused single-document editor
whose layout is dictated by the one thing that matters — the image becoming itself. The 60/40 canvas/sidebar split and
the live shapes/sec ticker are the signature, intentional details.

### Interaction-state table (what the user *sees*, per surface)

| Surface | Loading | Empty / first-run | Error | Success | Partial / in-progress |
|---|---|---|---|---|---|
| **Canvas** | n/a (instant) | Soft dashed drop-zone: *"Drop an image to begin"* + a faint sample thumbnail strip (cat / monalisa) the user can click to try instantly | Decode/format fail → inline toast on the canvas: *"Couldn't read that image — try PNG, JPG, or WebP"*; canvas stays in drop state | Final frame holds on canvas; subtle "done" pulse; shape count + final PSNR shown | Canvas repaints live at ≥30 fps; progress bar + `shapes done / total` + live shapes/sec under it |
| **Source panel** | thumbnail decode spinner (sub-100 ms) | the drop-zone above | unsupported file → red helper text under Browse | thumbnail + dimensions shown | dimmed during a run (can't swap mid-run; Reset to change) |
| **Controls** | — | sensible defaults pre-filled (triangle / 250 / alpha 128) | out-of-range guarded by slider bounds (can't enter invalid) | — | **disabled** during a run except Pause/Reset (changing params mid-run is meaningless) |
| **Start/Export** | — | Start enabled only after an image loads; Export disabled until ≥1 shape exists | export write failure → toast: *"Couldn't save — check folder permissions"* | Export → native save panel → toast *"Saved cat.svg"* with **Reveal in Finder** | Start ⇄ Pause toggle; Export disabled mid-run until paused/done |
| **Device/perf** | one-time probe on launch | shows the active backend chip: `Metal` / `CUDA` / `CPU` | **No compatible GPU → amber chip `CPU (no GPU found)`** + tooltip explaining it'll be slower; app still fully works via the CPU adapter | green chip + live shapes/sec | live shapes/sec ticker |

**The GPU-unavailable state is a first-class UX, not an error.** Because the CPU adapter (`primitive-gpu-cpu`) is a
real, shipping backend, an unsupported GPU must **degrade gracefully**: the app opens, shows an amber `CPU (no GPU
found)` chip with a one-line tooltip, and runs correctly (just slower). It must **never** show a fatal error or refuse
to launch — the worst failure mode for a GPU app is a blank window on a machine without the "right" GPU.

### Emotional arc (time-horizon design)

| Step | User does | Should feel | Plan provides |
|---|---|---|---|
| First 5 s | Opens app, sees drop-zone + sample thumbnails | "Oh, I can just try it" | one-click samples; no signup, no config wall |
| 5–60 s | Drops a photo, hits Start | delight as shapes accrete | live ≥30 fps canvas + shapes/sec ticker — the spectacle |
| Repeat use | Tweaks count/type, re-runs, exports | in control, fast | deterministic seed for reproducible runs; instant Reset; one-key export |

### Responsive, window & accessibility

- **Window behavior (desktop app, not a web page):** resizable; minimum 900×600; the canvas is the flex region and
  keeps the source image's aspect ratio (letterboxed), the sidebar is a fixed ~320 px. Remember last window size and
  last-used params across launches.
- **Keyboard:** `⌘O` open, `Space` start/pause, `⌘E` export, `⌘R` reset, `⌘,` settings; full tab order through controls;
  visible focus rings. (egui supports keyboard focus + shortcuts natively.)
- **Accessibility:** egui exposes an **AccessKit** tree → VoiceOver reads control labels and the live progress value;
  honor the system light/dark appearance; respect "Reduce Motion" (cap/disable the live repaint pulse); slider hit
  targets ≥ 24 px; meet WCAG AA contrast on all text/chips. These are gates in GUI-2, not afterthoughts.
- **Localization:** all strings through a string table from day one (en first); no hard-coded UI text.

### What's deliberately NOT in v1 (subtraction)

No project/file management, no undo history of individual shapes, no in-app gallery, no theming beyond system
light/dark, no batch/CLI-in-GUI. (A headless CLI ships separately from `primitive-engine` for scripting — the GUI
stays a single-image tool.)

---

## 6. Core GPU kernel & hot-loop implementation — *is there a better way to implement it?* (answer to follow-up #5)

Target: **Rust + CubeCL, one fused compute kernel, fully on-device, bitwise-deterministic across CUDA and Metal.**
Unifying principle: **the candidate never leaves the GPU between `mutate → rasterize → color-fit → delta-score → argmin`.**

### 6.1 Delta scoring (running error, bounding-box local) — *the* fix vs the Python port
Persist two device buffers for the whole run: the committed `current` image and a scalar `total_sse`. Score **only
each candidate's bounding box**:

```
delta = Σ_{covered px in bbox} [ (fill − target)²  −  (current − target)² ]
```

Uncovered pixels in the bbox contribute 0 (`current == current`). At commit time the winner's `delta` is *added* to
the running `total_sse` — **full-image SSE is never recomputed.** This is Fogleman's core trick, preserved verbatim,
and is exactly what the Python port threw away (it recomputed whole-image MSE every candidate).

### 6.2 Closed-form color over covered pixels (inside the same kernel)
- **Opaque fill:** optimal color = mean of `target` over the covered pixels (per channel).
- **Alpha-blended fill:** per-channel weighted least-squares against `current` and `target` — closed form, O(1) after
  the sums. Must be computed *inside* the scoring kernel so the color is known before the SSE term is formed.
- **Axis-aligned rectangles:** precompute **Summed-Area Tables / integral images** of `target`, `target²` (and
  `target·current` for alpha) → color *and* SSE become O(1) per rect via four corner lookups (Hensley et al. 2005;
  Nehab/Hoppe SIGGRAPH Asia 2011 for the recursive-filter formulation).
- **Triangles / ellipses / non-AABB:** SAT doesn't apply — accumulate sums via **scanline spans** inside the kernel
  (Fogleman's scanline rasterization), with the bbox bounding the loop.
- **Overflow guard (verified):** at 1024×1024 a single-channel SAT corner reaches ~267M, far past fp32's exact-integer
  ceiling of 2²⁴ ≈ 16.7M. **SATs and SSE accumulators must be i32/i64, never fp32** (also see §6.6).

### 6.3 Fused single-dispatch kernel (one workgroup per candidate)
Each candidate is owned by one workgroup; in one launch over the whole batch:
`Philox mutate → rasterize into bbox → reduce coverage sums → closed-form color → integer delta-SSE reduction → write {delta, candidate_id}`.

Use a **graduated workgroup strategy by shape area** — tiny shapes → one thread, medium → one warp/subgroup, large →
full workgroup — exactly as **CuRast** (arXiv 2604.21749, 2026; `m-schuetz/CuRast`) does for rasterization. This keeps
occupancy high across the wildly varying primitive sizes a real run produces. Fusing the passes (vs separate
rasterize/score kernels) is the documented portability-safe win — it eliminates the intermediate-buffer round-trips
that otherwise dominate at these tiny per-candidate workloads.

### 6.4 On-device RNG (counter-based, stateless, reproducible)
Use **Philox4x32-10** (Salmon et al., SC 2011 Best Paper, *"Parallel random numbers: as easy as 1, 2, 3"*; library =
"Random123"). A counter-based PRNG carries no state — the stream is a pure function of a counter:

```
key     = global_seed
counter = (shape_index, restart_id, es_iteration, draw_index)
```

Every draw is reproducible and independent across threads with **zero synchronization** — so the whole optimizer is
replayable from a single seed (critical for the determinism contract and CI parity). Mutations and restarts are
generated on-device, so the inner loop needs no per-iteration host upload.

### 6.5 Parallel argmin (select the winner)
Reduce `{delta, candidate_id}` to the single minimum with a **shared-memory tree reduction** (Harris, *GPU Gems 3*
ch. 39). Reduce on `delta`, carry `candidate_id` alongside, and **break ties by lowest `candidate_id`** so the result
is independent of thread scheduling. Then a tiny `commit` kernel blends the winner into `current` and updates
`total_sse` *on-device* — the only mandatory host transfer per shape is the winning struct out (and the seed in).

### 6.6 Numerical determinism (bitwise across CUDA + Metal)
- **Integer SSE is the headline decision.** `target`/`current` are u8; the squared per-pixel diff fits exactly in
  i32 and bbox sums in i64. **Integer addition is associative**, so the SSE reduction is order-independent and
  **bitwise-identical regardless of reduction-tree shape or backend.** Prefer integer accumulation everywhere it's
  possible.
- **If a float reduction is unavoidable** (alpha-blend least squares): fix the reduction-tree topology and use an
  accurate parallel reduction (Xu et al., IEEE HPEC 2020, *Work-Efficient Parallel Algorithms for Accurate
  Floating-Point Prefix Sums*) or Kahan compensation, so CUDA and Metal agree despite non-associative fp.
- **Anti-aliasing**, if used for sub-pixel quality, should be **analytic** (frost.kiwi, *Analytical Anti-Aliasing*,
  Nov 2024), not stochastic supersampling — deterministic and cheap. Keep AA out of the scoring loop unless it
  measurably improves output; integer hard coverage is the deterministic default for opaque fills.

### 6.7 Cross-backend kernel abstraction
**Write the kernel once in CubeCL.** It compiles to CUDA (via `cubecl_cuda`) and Metal (via wgpu's
`MSL_SHADER_PASSTHROUGH`), with autotuning over launch configs — directly serving the CUDA+Metal requirement and
letting the graduated-workgroup tuning (§6.3) be data-driven. Keep *all* platform-specific code behind the CubeCL
kernel boundary so adding a backend never touches the optimizer — DiffVG's CUDA lock-in is precisely the trap this
boundary exists to avoid. (Fallback if CubeCL's CUDA path proves unstable: `rust-gpu` SPIR-V, or a hand-written
cudarc kernel behind the same `GpuCompute` port — see Open Q #1.)

**End-to-end hot-loop shape:**
`[precompute target SAT + target² (i64, once)] → for each of N shapes: { generate K candidates (energy-map-biased
restart) → fused kernel (Philox mutate → graduated raster → closed-form color → integer delta-SSE) → (1+1)-ES inner
loop of M mutations with 1/5-rule step adaptation → tree argmin → on-device commit + update total_sse }`.
After the one-time SAT precompute, everything stays GPU-resident; per-shape host traffic = seed in, winning struct out.

---

## 7. Phased roadmap (Mac first, Windows later)

Naming: `<MILESTONE>-<N>`. Each milestone has a **verification gate** — "done" means the gate passes, not that it
looks plausible. The CPU adapter (`primitive-gpu-cpu`) is a permanent parity oracle for every GPU milestone.

| Milestone | Goal | Verification gate (proof) |
|---|---|---|
| **CORE-1** | Pure-Rust scalar port: shape / raster_ref / color_solve / score / ES in `primitive-core`. | Golden-image test: fixed seed + fixed target → output matches a `fogleman/primitive` reference at **SSIM ≥ 0.999** and final SSE within **1%**; golden fixtures committed. |
| **CORE-2** | Hexagonal split: define `GpuCompute` port; implement CPU/Rayon adapter. | `check-boundaries.sh` green (core imports no adapters); CPU adapter reproduces CORE-1 golden. **Records the CPU baseline shapes/sec on M-series** — every GPU claim is measured against this. |
| **GPU-1** | CubeCL device init on Metal; single-candidate fused `raster_score` kernel. | Parity: GPU score for 1k *individual* candidates == CPU score within fp/int epsilon (integer path: exact). |
| **GPU-2** | Batched B-candidate fused dispatch + `reduce_argmin`; fully GPU-resident; bbox-local delta. | Parity (same winning index as CPU brute force on a fixed batch) **and** throughput **≥ 20× CORE-2 CPU baseline** (candidates/sec, M-series); **zero per-candidate host sync** (verified via Metal capture / dispatch count). |
| **GPU-3** | Full on-device (1+1)-ES loop + Philox mutate + energy-map restarts; throttled preview readback. | N-shape result within **0.5 dB PSNR** of CPU ES on the same seed; **end-to-end ≥ 20–50× faster** than CPU for a 1000-shape run → **proves the "slower than CPU" regression is gone.** |
| **GUI-1** | eframe shell sharing the wgpu `Device`; live zero-copy canvas via `CallbackTrait`. | App renders live progress at **≥ 30 fps** during a run; Metal capture confirms the compute texture is bound directly (no host pixel copy). |
| **GUI-2** | Controls + all interaction states from §5A: first-run drop-zone, GPU-fallback chip, per-state behavior, PNG/SVG/GIF export, keyboard shortcuts, a11y. | e2e smoke (load → run 100 shapes → export SVG) exits 0 with valid SVG; **interaction-state checklist (§5A table) all covered**; **launches with `CPU (no GPU found)` chip and still completes a run** when GPU is forced off; VoiceOver reads controls + progress (AccessKit tree present); WCAG-AA contrast + Reduce-Motion honored. |
| **PKG-1** | macOS `.app` bundle → codesign (Developer ID) → notarize → staple → DMG. | `xcrun stapler validate` passes and `spctl --assess --type execute` returns **accepted** on a **clean machine** (no dev tools); Gatekeeper opens it without warning. |
| **GPU-4** | CUDA backend via the **same** `#[cube]` source on Windows/NVIDIA. | Same parity test as GPU-2 on NVIDIA; throughput ≥ 20× a Windows CPU baseline; **golden output cross-checked bit-for-bit (integer path) against the Metal run.** |
| **PKG-2** | Windows packaging (MSI/exe); code-signing later. | Installer runs on clean Windows; app launches and completes a 100-shape run. |

> **Throughput gates are hardware-dependent.** The ≥20× / ≥460 sps thresholds (GPU-2, GPU-3, GPU-4)
> hold on representative target hardware (Apple Silicon / discrete NVIDIA) and are enforced by
> `make perf` (`PRIMITIVE_PERF_GATE=1`). The always-run `make verify` / CI gate is **correctness only**
> (parity, golden, PSNR, a11y, boundaries) — it measures + prints throughput but does not assert a
> fixed number, because a shared/virtualized CI GPU is far slower (observed 9.6× vs 43.9×). See
> `crates/primitive-gpu-cubecl/tests/common/mod.rs`.

---

## 8. Risks & mitigations (top 5)

1. **Cross-backend kernel portability (highest risk).** CubeCL's Metal path rides wgpu `MSL_SHADER_PASSTHROUGH` and
   is young (0.9.x stable; 0.10 pre-release). WGSL has **no float atomics** and MoltenVK has **no 64-bit atomics** —
   capabilities differ across Metal/CUDA. **Mitigation:** the kernels are designed to need **no float/64-bit atomics**
   (per-slot writes + a separate reduction pass, §6.5; integer accumulation, §6.6); the CPU adapter is a permanent
   parity oracle so any divergence trips the GPU-2 gate; pin CubeCL 0.9.x and gate upgrades behind the parity suite.
2. **Metal-from-Rust churn.** `metal-rs` is deprecated (→ `objc2-metal`) and wgpu's own migration (PR #5641) is in
   progress. **Mitigation:** we never touch Metal directly — wgpu/CubeCL fully insulate us. Also note **wgpu v25 gave
   +20% on M3 but −20% on NVIDIA H100** — so pin a known-good wgpu line and benchmark *both* platforms before any bump.
3. **Notarization friction.** Needs a paid Apple Developer account; `notarytool` can be slow/flaky and rejects
   unsigned nested binaries. **Mitigation:** script the whole chain (`scripts/ops/sign-notarize.sh`), make PKG-1's
   gate the **clean-machine** `spctl`/`stapler` check (not "it ran on my Mac"), and run it **early**, not at the end.
4. **GUI/compute device interop on CUDA.** eframe renders via wgpu; `cubecl_cuda` is a separate context, so the
   Windows live preview needs a device→host→wgpu copy. **Mitigation:** keep heavy buffers on the CUDA device, transfer
   only the throttled preview frame (~16 ms). On Metal both share the wgpu device, so this is moot.
5. **CubeCL pre-release churn / API drift.** Fast-moving ecosystem (Tracel raised $3M Aug 2025; Burn ships CubeCL as
   its primary GPU backend). **Mitigation:** depend on **stable 0.9.x**, lock via `Cargo.lock`, and isolate all CubeCL
   types behind the `GpuCompute` port so a future migration (or a drop to raw wgpu WGSL for Metal-only) touches one crate.

---

## 9. Prior art — study vs fork

**Verdict: build from scratch in Rust. Do not fork.** No GPU implementation of this algorithm exists anywhere, and
the CPU ports are simple enough that a clean hexagonal rebuild beats retrofitting one.

**Algorithm reference (read, don't fork):**
- **`fogleman/primitive`** (Go, ~13k★, MIT, unmaintained ~2020) — the canonical algorithm (scanline raster, closed-form
  color, bbox SSE, hill-climb). Port its math into `primitive-core` as the golden oracle.
- **Rust CPU ports to mine for idioms:** `chux0519/purr` (crate `purrmitive`), `larryng/primg` (crate `primg`),
  `samgoldman/primitive_image`. None are GPU; none worth forking — read for Rust shape/raster structuring.
- **Geometrize** (C++/Qt, unmaintained ~2021) + `swift-geometrize` (recently maintained, currently quiescent) — for
  alternate shape-set ideas only.

**GPU-residency pipeline patterns (study the *architecture*, not the algorithm):**
- **SAD — Soft Anisotropic Diagrams** (arXiv 2604.21984, SIGGRAPH 2026; **code exists** — HuggingFace Kernels
  `blanchon/soft-anisotropic-diagrams`, CUDA+Metal ops, jump-flooding + on-device optimization). The best living
  example of a GPU-resident, no-host-roundtrip Metal+CUDA pipeline — exactly our target shape.
- **CuRast** (arXiv 2604.21749, 2026; `m-schuetz/CuRast`) — the graduated-workgroup rasterization strategy (§6.3).
- **DiffBMP** (CVPR 2026, `pip install pydiffbmp`) and **Bézier Splatting** (NeurIPS 2025) — *differentiable* (different
  algorithm); mine for buffer layout and dispatch structure only.
- **Graphite editor** GSoC-2025 GPU raster (discussion #2658) — a real rust-gpu→naga→wgpu compute pipeline to study
  for Rust-GPU toolchain ergonomics.

---

## 10. Open questions (decide before / during the build)

1. **CubeCL vs raw-wgpu fallback trigger.** If CubeCL's CUDA path is unstable at GPU-4, do we (a) accept
   wgpu-Vulkan/DX12 on NVIDIA (drops literal "CUDA" but keeps one kernel source) or (b) hand-write a cudarc kernel
   behind the same port? Decide at GPU-4 from the parity-suite results. *(Recommend (a) only if "GPU-accelerated on
   NVIDIA" is acceptable as ≠ "CUDA"; else (b).)*
2. **Batch size B and ES K/M** — tune empirically per GPU; needs a sweep harness in GPU-2/GPU-3.
3. **Deterministic mode for CI.** On-device RNG (chosen) eliminates sync but complicates golden tests. Ship a
   `--deterministic` path that seeds the device Philox identically to the CPU reference. *(Recommend: yes.)*
4. **Preview cadence vs throughput** — start 16 ms, expose as a setting.
5. **Shape set for v1** — match Fogleman fully (triangle, ellipse, rect, rotated-rect, line, bezier, polygon) or ship
   a subset first? *(Recommend: triangle + ellipse + rect for GPU-3, expand after PKG-1.)*
6. **Windows GUI device-copy cost** — accept the per-frame CUDA→wgpu preview copy, or run a wgpu-Vulkan compute path on
   Windows so GUI+compute share a device there too? Decide at GPU-4 / PKG-2.
7. **License** — keep MIT (matches the Fogleman lineage)?

---

## 11. References & confidence notes

Citations corrected and graded by an adversarial verification pass. **Substance is solid; treat the flagged numbers
as approximate.**

**Verified-solid:** Rust GPU stack (wgpu native-Metal default; CubeCL 0.9.x stable → CUDA+Metal; cudarc 0.19.x
active); eframe wgpu default + `WgpuSetupExisting` + `CallbackTrait`; Slint wgpu injection (`slint::wgpu_28`, PR #8278);
CoreML/MPSGraph category limits; Go GoGPU has no CUDA backend; CamoPatch (1+1)-ES (Williams & Li, NeurIPS 2023); ProHC
energy-map (doi:10.3390/biomimetics8020174, *Biomimetics* 2023); DiffVG (SIGGRAPH Asia 2020) CUDA-only; Bézier
Splatting speedup *ratios* (NeurIPS 2025); CuRast graduated workgroups (arXiv 2604.21749, 2026); Philox / Salmon et al.
SC 2011 (*"Parallel random numbers: as easy as 1, 2, 3"*); Harris parallel reduction (GPU Gems 3 ch.39); Hensley 2005 /
Nehab-Hoppe 2011 SAT; Xu et al. accurate fp prefix sums (HPEC 2020); fp32 SAT-overflow guard; SAD code on HuggingFace
Kernels.

**Flagged — do not treat as hard numbers:** "wgpu ≈ 85% native-Metal throughput" (unverified); RWTH "1M candidates/sec
on GTX 1080" (domain analog, unverified); Bézier-Splatting *absolute* PSNR/timing (only ratios confirmed); "ES ≈
gradient quality" (suggestive, not a controlled head-to-head). The CORE-1/GPU-3 gates (§7) are the real proof of
quality and speed, not any cited figure.
