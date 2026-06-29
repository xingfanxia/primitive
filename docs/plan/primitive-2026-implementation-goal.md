# Primitive 2026 — Implementation Kickoff & Goal Prompts

> Spec: [`primitive-2026-architecture.md`](./primitive-2026-architecture.md) · Provenance: [`../research/primitive-2026-research-findings.md`](../research/primitive-2026-research-findings.md)
> Paste the **Kickoff** block below to start the next session. The full build (CORE → GPU → GUI → PKG) spans
> several sessions; each `/goal` block targets one coherent, single-session milestone slice with a verifiable
> exit gate and a turn cap. Run them in order.

---

## Why milestone-by-milestone, not one "fully implement" goal

`/goal` guardrails (workflow.md): goals are **single-session**, must have a **turn cap**, and the Haiku evaluator
only sees **evidence pasted into the transcript** (so gates must be `cargo test` exit-0 / printed metric numbers,
not "it looks done"). A 10-milestone build can't be one goal. The sequence below is the decomposition; each slice
ends on machine-checkable proof. **Codesign/notarize/push milestones (PKG-*) must NOT run under auto-mode** — they
touch credentials/distribution and need you in the loop.

---

## KICKOFF (paste this first, next session)

```
Read docs/plan/primitive-2026-architecture.md and docs/research/primitive-2026-research-findings.md in full —
they are the spec and the verified provenance. We are implementing the rebuild, Tier 4. Use /big-task.

Start with the first vertical slice that proves the entire thesis ("GPU makes this much faster than CPU, with
identical output"): milestones CORE-1 → CORE-2 → GPU-1 → GPU-2 → GPU-3 from §7 of the plan. Set up the cargo
workspace with the hexagonal crate layout from §5, pin CubeCL 0.9.x and eframe 0.34.2, and enforce the import
boundaries with tools/verify/check-boundaries.sh in CI.

Then immediately:

/goal CORE-1 + CORE-2 are implemented and `cargo test` exits 0 with the golden-image test passing (fixed-seed
output vs a committed fogleman reference at SSIM >= 0.999 and final SSE within 1%), check-boundaries.sh exits 0,
and the CPU-adapter baseline shapes/sec is printed to the transcript; or stop after 30 turns.

Skill(skill="autonomous-grind", args="start CORE-1+CORE-2 done: cargo test exit 0, golden SSIM>=0.999, boundaries green, CPU baseline printed, or 30 turns")
```

**Why this slice first:** CORE-1/CORE-2 build the pure-Rust reference + the parity oracle the *entire* GPU effort
is measured against. Nothing downstream is trustworthy without the golden image and the CPU baseline number.

---

## Subsequent goal blocks (run each in its own session, in order)

> **Status (2026-06-29):** ✅ CORE-1/2 · ✅ GPU-1/2/3 · ✅ GPU-3 perf pass (self-adaptive step, 519 sps @ 64×64,
> ~1.9× fogleman) · ✅ GUI-1 (eframe shell + live canvas) · ✅ GUI-2 (all 6 §5A gates green; `make verify` ALL
> GREEN). **Next → PKG-1 Part A (autonomous), then PKG-1 Part B (interactive credentials).** GPU-4 (CUDA) is
> a different-machine job (`docs/gpu4-cuda/RUNBOOK.md`). Full ledger: `.agent/PROGRESS.md` + `.agent/EVIDENCE.md`.
>
> **PKG-1 Part A ✅ DONE (2026-06-29):** `make bundle` builds + validates `primitive.app` (`com.primitive.app`,
> v0.1.0, flat-triangle icon) and HALTS before codesign — no credentials. **Remaining = interactive/other-machine
> only:** PKG-1 Part B (codesign → notarize → staple, you hold the Apple credentials) and GPU-4/PKG-2 (NVIDIA/Windows).

### GPU slice — prove the speedup (the headline result) — ✅ DONE
```
Continue the plan (docs/plan/primitive-2026-architecture.md §6, §7). Implement GPU-1 → GPU-2 → GPU-3:
the fused single-dispatch CubeCL kernel (Philox mutate -> graduated raster -> closed-form color ->
integer delta-SSE), parallel argmin, on-device (1+1)-ES loop with energy-map restarts, all GPU-resident.

/goal GPU-1/GPU-2/GPU-3 done and proven in transcript: (1) `cargo test` exits 0 with the GPU-vs-CPU parity
test passing (integer path: exact match of winning candidate index on a fixed batch), (2) a printed benchmark
shows GPU candidates/sec >= 20x the CORE-2 CPU baseline on this Mac, and (3) a 1000-shape run prints end-to-end
speedup >= 20x CPU AND PSNR within 0.5 dB of the CPU ES run on the same seed; or stop after 40 turns.

Skill(skill="autonomous-grind", args="start GPU-1/2/3 done: cargo test exit 0, parity exact, >=20x CPU throughput printed, 1000-shape >=20x + PSNR within 0.5dB, or 40 turns")
```
> This is the milestone that **proves the old Python/MPS regression is gone.** If parity fails, the integer-SSE
> determinism design (§6.6) is the first thing to check.

### GUI-2 slice — interaction states, a11y, export, GPU chip — ✅ DONE (2026-06-29)

> GUI-1 shipped (`primitive-app`, binary `primitive`): hero live canvas + sidebar, drop/Browse/samples,
> count+alpha, Start/Pause/Resume/Reset, PNG/SVG export, background-thread frame streaming, CPU adapter — see
> `.agent/PROGRESS.md` → GUI-1. GUI-2 closes the remaining §5A gaps. **Out of scope:** the multi-shape-type
> selector — core implements only `Triangle`; adding ellipse/rect/… is a CORE milestone, not GUI.
```
Continue the plan (architecture.md §5A, §7). GUI-1 is done. Implement GUI-2 in `primitive-app`: the remaining
§5A interaction-state cells; GPU backend detection + the amber `CPU (no GPU found)` fallback chip (the GPU runs
the WHOLE loop via `gpu_optimize` — an "instant" mode; it is NOT a per-shape `ShapeSearch` port); keyboard
shortcuts (⌘O/Space/⌘E/⌘R/⌘,); an Advanced disclosure (seed + n/age/m); Export ▾ (PNG·SVG·GIF); AccessKit a11y;
Reduce-Motion; an i18n string table; window+param persistence. Build the §5A interaction state as a PURE,
render-free module so it is unit-testable headless.

/goal GUI-2 done and proven in the transcript:
(1) `cargo test -p primitive-app` exits 0 with the §5A interaction-state suite passing — one test per
    surface×state cell over the pure state module (canvas drop/error/live; source dimmed-mid-run; controls
    disabled-mid-run-except-Pause/Reset; Start gated-on-image; Export disabled-until->=1-shape; device amber
    chip) — paste exit code + the printed pass count covering every §5A row (architecture.md lines 246-259).
(2) the scripted e2e (load sample -> 100 shapes -> export SVG) exits 0 AND `xmllint --noout out.svg` exits 0 —
    paste BOTH exit codes.
(3) a forced-CPU run (`PRIMITIVE_FORCE_CPU=1`) completes a 100-shape run and prints the backend label
    `CPU (no GPU found)` to the transcript — paste the line.
(4) the AccessKit-tree test (egui_kittest, architecture.md line 276) exits 0 asserting control labels + a
    live-progress node exist — paste exit code.
(5) the deterministic a11y test exits 0 — WCAG contrast from the design tokens (text >=4.5:1, chips >=3:1) and
    repaint-pulse-off-when-Reduce-Motion (architecture.md line 278) — paste exit code.
(6) `tools/verify/check-boundaries.sh` exits 0 (adapters imported only at primitive-app/main.rs) — paste exit code.
Live >=30 fps, VoiceOver speech, the amber-chip render, and GIF visual progression are ARTIFACTS, not gates (the
Haiku evaluator can't see a window). Or stop after 35 turns.

Skill(skill="autonomous-grind", args="start GUI-2 done: cargo test -p primitive-app §5A state suite exit 0, e2e exit 0 + xmllint valid SVG, forced-CPU run prints 'CPU (no GPU found)', AccessKit-tree test exit 0, WCAG/Reduce-Motion token test exit 0, check-boundaries.sh exit 0, or 35 turns")
```
> Machine-checkable = the 6 gates above (pasteable exit codes / printed labels). Genuinely-visual items (fps,
> VoiceOver speech, chip color, GIF playback) are hand-verified artifacts, never in the predicate. The AccessKit
> gate asserts the tree *data* (not speech); the WCAG gate is *math over design tokens* (not rendered pixels).

### PKG-1 slice — signed Mac app (Part A autonomous · Part B interactive-credentials)

**Part A — bundle scaffolding (autonomous, NO credentials, NO /goal): ✅ DONE (2026-06-29) — `make bundle`**
```
Continue the plan (architecture.md §7 PKG-1). Scaffold the macOS bundle — NO signing, NO network:
- assets/Info.plist (name primitive, id com.primitive.app, version from the workspace, LSMinimumSystemVersion, category)
- assets/icons/ — a flat geometric "primitive" mark as a PNG set (512..16) and/or .icns (gpt-image is fine)
- [package.metadata.bundle] in crates/primitive-app/Cargo.toml
- scripts/ops/sign-notarize.sh that runs cargo-bundle -> validation -> create-dmg, then HALTS with printed
  instructions immediately BEFORE the first codesign step (it must never call codesign/notarytool itself).
Done (pasteable): `plutil -lint assets/Info.plist` exits 0, `cargo bundle` produces .../primitive.app, and the
script halts at the codesign step. No credentials touched.
```

**Part B — sign / notarize / staple (interactive; confirm before EACH step; NO /goal, NO auto-mode):**
```
Continue scripts/ops/sign-notarize.sh from where Part A halted. Run step-by-step, pausing for my explicit
confirmation before EACH credential/network step (I hold the Apple credentials):
  1. codesign --sign "Developer ID Application: <name>" --timestamp --options runtime --deep --force primitive.app
  2. create-dmg ... primitive.dmg
  3. xcrun notarytool submit primitive.dmg --apple-id <email> --team-id <team> --password <app-specific-pw>
  4. xcrun stapler staple primitive.app
No /goal, no auto-mode for any of the above.
```
> Done-gate (architecture.md §7 PKG-1): on a **clean machine** (no dev tools), `xcrun stapler validate
> primitive.app` exits 0 AND `spctl --assess --type execute primitive.app` says `accepted`, and Gatekeeper opens
> it with no warning. This gate is verified on a *second* machine — exactly why PKG-1 stays out of /goal.

### GPU-4 + PKG-2 — Windows / CUDA (different machine)

> **Step-by-step runbook: `docs/gpu4-cuda/RUNBOOK.md`** — runs on a machine with a discrete NVIDIA GPU (the
> Mac can't produce the headline number). Measured rationale: on Apple Silicon's unified memory the GPU loses
> to a 16-core fogleman at 128×128 (`.agent/EVIDENCE.md`); a discrete card with dedicated bandwidth is the real
> big-canvas regime, and CUDA's native i64 removes the 64×64 i32 cap (no hi-lo split needed there).
```
Continue the plan (§7 GPU-4, PKG-2). Bring up the CUDA backend from the SAME #[cube] source on NVIDIA/Windows.

/goal GPU-4 done: the GPU-2 parity test passes on NVIDIA hardware AND the integer-path golden output matches the
Metal run bit-for-bit (paste a diff/hash showing identical bytes), with throughput >= 20x a Windows CPU baseline
printed; or stop after 40 turns. (PKG-2 Windows installer + signing: run interactively, not under /goal.)

Skill(skill="autonomous-grind", args="start GPU-4 done: NVIDIA parity passes, integer golden bit-identical to Metal (hash pasted), >=20x Win CPU baseline, or 40 turns")
```

---

## Standing rules for every implementation session

- **TDD per §7 gates:** write the golden/parity test first, then make it pass (the gates ARE the acceptance tests).
- **Pin versions** (CubeCL 0.9.x, eframe 0.34.2, cudarc 0.19.x) and lock via `Cargo.lock`; re-check latest with
  `cargo search` before pinning — research is from 2026-06 and the ecosystem moves fast (see Risk #5).
- **The CPU adapter is the permanent parity oracle** — never delete it; every GPU change re-runs parity.
- **No float/64-bit atomics** in kernels (§6.6) — per-slot writes + reduction pass keep it portable + deterministic.
- **Benchmark wgpu bumps on BOTH Mac and NVIDIA** before accepting (the v25 +20%/−20% split, Risk #2).
- Pair every multi-turn `/goal` with `autonomous-grind`; never put codesign/notarize/push under auto-mode.
