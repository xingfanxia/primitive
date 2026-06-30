# Primitive 2026 — Research Findings & Provenance

> Companion to [`../plan/primitive-2026-architecture.md`](../plan/primitive-2026-architecture.md).
> This file records *how the decisions were verified* — what an adversarial pass confirmed vs corrected —
> so the implementer knows which claims are load-bearing and which are flagged-approximate.
> Method: two multi-agent research workflows (18 agents total). Every research claim was re-checked by a
> second agent tasked to refute it. Trust ratings below are that second agent's verdict.

---

## How to read this

- **Verified-solid** = independently re-confirmed; build on it.
- **Corrected** = the first-pass research was wrong; the corrected fact is what the plan uses.
- **Flagged-approximate** = plausible but unverifiable; do **not** treat as a hard number — the roadmap's
  verification gates (golden-image SSIM, measured throughput) are the real proof.

---

## Workflow 1 — Architecture (6 dimensions)

### Language: Rust vs Go — trust: mostly-solid
**Verified-solid**
- Go's only GPU compute ecosystem **GoGPU launched 2025-12-18, is v0.x, and has NO CUDA backend** (WebGPU semantics, 5 backends: Vulkan/Metal/DX12/GLES/Software). This alone disqualifies Go against the explicit CUDA requirement.
- Go CGO cross-compilation friction (C toolchain on host, static-linking complications) is a real, well-known limitation for signed cross-platform builds.
- `cudarc` is **actively maintained at 0.19.x** (mid-2026; repo moved to `chelsea0x3b/cudarc`).
- CoreML is an ML-inference runtime, not a general compute-dispatch API — correctly a category error for rasterization.

**Corrected**
- ❌ "cudarc last updated Feb 2025" → actually 0.19.x through mid-2026 (stale by a year; directionally *more* active than claimed).
- ❌ "CubeCL 0.10.0 stable (May 2026)" → **stable is 0.9.0**; 0.10.0 is **pre-release** (0.10.0-pre.2, Mar 2026). **Pin 0.9.x.**
- ❌ "wgpu uses MoltenVK on macOS by default" → **false; Metal is wgpu's native default backend.** MoltenVK is opt-in via `vulkan-portability`.

### Cross-platform GPU compute — trust: mostly-solid
**Verified-solid**
- WGSL atomics are **i32/u32 only — no float atomics** (W3C spec). Design kernels to avoid them (the plan uses per-slot writes + a reduction pass).
- MoltenVK has documented gaps: **no 64-bit atomics**, no ray tracing, no sparse resources (LunarG Jan 2026).
- CubeCL targets CUDA + Metal + ROCm/HIP + WGSL/SPIR-V + CPU-SIMD via `#[cube]`; Burn 0.20 (Jan 2026) ships CubeCL as its primary GPU backend; Tracel raised $3M (Aug 2025).
- Metal-dispatch overhead ≈ **71 µs** via wgpu-native (arXiv:2604.02344, Feb 2026) — the quantitative basis for "kernel-launch overhead dominates the Python port."

**Corrected**
- ❌ "CubeCL Metal is a standalone native runtime" → it's delivered via **wgpu's `MSL_SHADER_PASSTHROUGH`** (PR #7326) — generates native MSL but is wgpu-hosted, inside `cubecl_wgpu`. (`cubecl_cuda` is the separate CUDA runtime.)
- ⚠️ **Flagged-approximate:** "wgpu ≈ 85% of native Metal throughput" — **unverified**; no Dawn benchmark found stating this. Do not plan around it.

### Apple acceleration — trust: mostly-solid
**Verified-solid**
- CoreML / MPSGraph expose only fixed primitives (conv/FFT/sort/linalg) and **cannot author custom rasterization kernels** ("you cannot create your own node types"). **Metal compute shaders (MSL) are the correct API.**
- Apple-Silicon unified memory (`MTLResourceStorageModeShared`) is a genuine zero-copy advantage (arXiv:2502.05317).
- `metal-rs` is **formally deprecated → objc2-metal**; wgpu's own migration (PR #5641) is in progress. We stay insulated via wgpu/CubeCL.
- WWDC25 sessions 205 (Discover Metal 4), 262 (Metal 4 ML+graphics), 236 (WebGPU on Apple) confirmed; Metal 4's Shader-ML additions are irrelevant to this geometric workload.

**Corrected**
- ❌ "wgpu v25 improved Metal/M3 by 20%" (stated with no caveat) → **true, but the same release regressed NVIDIA H100 by ~20%.** Pin a known-good wgpu line and benchmark both platforms before bumping. *(This is now Risk #2.)*

### GUI + macOS packaging — trust: shaky (most corrections here)
**Verified-solid**
- eframe made **wgpu the default renderer in 0.34.0** (~Mar 2026); `WgpuSetupExisting` shares an external Device; `CallbackTrait` (prepare/finish_prepare/paint) samples a compute texture in the egui render pass.
- Tauri/Wails put a WebView (CPU roundtrip) between Rust compute and display; Fyne uses its own GL stack and can't share a `wgpu::Device`.
- macOS `codesign` + `xcrun notarytool` + `create-dmg` is a well-established scriptable chain.

**Corrected**
- ❌ "eframe 0.35.0 (June 2026)" → **does not exist; latest stable is 0.34.2.** (Confusion with `egui_plot` 0.35.0.)
- ❌ "Tauri current is v2.7.0" → actually **~v2.11.x**.
- ❌ **"Slint lacks wgpu device injection (open FR #4499)"** → **materially wrong.** Slint **shipped** it: `slint::wgpu_28`, `BackendSelector::require_wgpu_28()`, `WGPUConfiguration::Manual` (inject Instance/Adapter/Device/Queue), `Image::try_from<wgpu::Texture>()` (PR #8278). **Slint is a real runner-up**, not "out" — the plan reflects this.

### Algorithm modernization — trust: mostly-solid
**Verified-solid**
- DiffVG (SIGGRAPH Asia 2020) and Bézier Splatting (NeurIPS 2025, arXiv:2503.16424, ~31×/149× vs DiffVG) are real; **both are CUDA-only / no Metal** → rejected on the cross-platform constraint.
- SAD (arXiv:2604.21984, SIGGRAPH 2026) is real and **its code exists** (HuggingFace Kernels `blanchon/soft-anisotropic-diagrams`, CUDA+Metal) — the best living reference for a GPU-resident no-host-roundtrip pipeline.

**Corrected**
- ❌ "SAD has no public code" → **code exists** (see above).
- ❌ "arXiv:2601.07258 supports GPU-parallel SA for shape search" → **category mismatch** (it's Bayesian optimization for materials science). Dropped from the plan's reasoning.
- ⚠️ **Flagged-approximate:** RWTH Aachen "1M candidate evals/sec on GTX 1080" — domain analog (image-filter params, not shapes); throughput figure unverified.

### Prior art — trust: mostly-solid
**Verified-solid**
- `fogleman/primitive` (Go, ~13k★, MIT, unmaintained ~2020) is canonical. **No GPU reimplementation of the hill-climb exists in any language** (exhaustive search) — confirming this is greenfield.
- Rust CPU ports exist to mine for idioms: `chux0519/purr` (`purrmitive`), `larryng/primg`, `samgoldman/primitive_image`.

**Corrected**
- ❌ "A Rust `geometrize` crate v0.1.0 (Delaunay, May 2026)" → **hallucinated; no such crate.** Only `geometrize-sys` (C++ FFI) exists.
- ❌ "samgoldman is the only Rust port" → at least `purr`/`purrmitive` and `primg` also exist.

---

## Workflow 2 — Algorithm & implementation deep dive (2 dimensions)

### Algorithm elegance — trust: mostly-solid
**Verified-solid**
- **CamoPatch (NeurIPS 2023, Williams & Li)** uses exactly **(1+1)-ES** for per-shape geometry+color — direct precedent for the chosen optimizer.
- **ProHC (Biomimetics 2023)** validates energy/saliency-map initialization + progressive polygon addition beating vanilla hill-climbing.
- DiffVG is CUDA-only (issue tracker is all CUDA/Torch builds; no Metal anywhere) — confirms the rejection.

**Corrected**
- ❌ ProHC DOI was wrong → correct is **doi:10.3390/biomimetics8020174** (PMC10204576).
- ❌ CamoPatch citation was a literal `XXXXX` placeholder → real paper Williams & Li, NeurIPS 2023.
- ⚠️ **Flagged-approximate:** Bézier-Splatting absolute PSNR/timings, and "ES ≈ gradient quality" (ES-CLIP compares to *gradient methods generally on its own CLIP metric*, not a controlled DiffVG head-to-head on opaque fills). Treat ES quality as **suggestive, proven only by our own CORE-1 golden gate.**

### Implementation technique — trust: shaky (citations), solid (techniques)
**Verified-solid (techniques)**
- **Integer (i32/i64) delta-SSE** is order-independent → bitwise-identical across CUDA/Metal. *The* determinism decision.
- **fp32 SAT overflow is real:** 1024² single-channel corner ≈ 267M ≫ fp32 exact-int ceiling 2²⁴ ≈ 16.7M → SATs/accumulators must be integer.
- **CuRast** (arXiv:2604.21749) graduated-workgroup-by-area strategy is real (1 thread / 1 warp / full workgroup by shape size).
- SAT generation (Hensley 2005; Nehab/Hoppe SIGGRAPH Asia 2011), tree reduction (Harris, GPU Gems 3 ch.39), Philox counter-based RNG (Salmon et al. SC 2011), accurate fp prefix sums (Xu et al. HPEC 2020), analytic AA (frost.kiwi 2024) — all confirmed real.

**Corrected (citation hygiene — techniques stand)**
- ❌ CuRast dated "April 2025" → arXiv 2604 decodes to **April 2026**.
- ❌ Philox paper title "Random123: A Library…" → actual title **"Parallel random numbers: as easy as 1, 2, 3"** (SC 2011); Random123 is the library.
- ❌ Xu et al. "HPEC 2019" → **HPEC 2020**.
- ❌ "Burn uses CubeCL as its *sole* backend" → CubeCL is the *primary* GPU backend; Burn also has LibTorch/NdArray/etc.
- ⚠️ **Flagged-approximate:** PhiloxRNG.jl benchmark ns/value figures — unverified.

---

## Net effect on the plan

The corrections that actually changed decisions: **pin CubeCL 0.9.x** (not 0.10), **eframe 0.34.2** (not 0.35),
**Slint is a viable runner-up** (it shipped wgpu injection), **wgpu version bumps must be benchmarked on both
Mac and NVIDIA** (the v25 regression), and **don't cite the parallel-SA paper or SAD-has-no-code**. Everything
load-bearing for the architecture (Rust + CubeCL single-source CUDA+Metal, Metal-not-CoreML, integer delta-SSE,
(1+1)-ES, no-GPU-prior-art-so-build-greenfield) survived adversarial verification.
