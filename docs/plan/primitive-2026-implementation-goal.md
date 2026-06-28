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

### GPU slice — prove the speedup (the headline result)
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

### GUI slice — the spectacle
```
Continue the plan (§5, §5A, §7). Implement GUI-1 → GUI-2: eframe shell sharing ONE wgpu::Device with the compute
core; live zero-copy canvas via CallbackTrait; the controls + all interaction states from §5A including the
GPU-unavailable amber chip; PNG/SVG/GIF export; keyboard shortcuts; AccessKit a11y.

/goal GUI-1/GUI-2 done: an e2e smoke (load image -> run 100 shapes -> export SVG) exits 0 and produces a valid
SVG (paste the exit code + `xmllint --noout` result), the app launches with a forced-CPU build and still
completes a run showing the `CPU (no GPU found)` chip, and the §5A interaction-state checklist is all covered;
or stop after 35 turns.

Skill(skill="autonomous-grind", args="start GUI-1/2 done: e2e smoke exit 0 + valid SVG, forced-CPU run completes with chip, §5A states covered, or 35 turns")
```
> Note: live-fps and zero-copy verification need a screenshot/Metal-capture — surface those as artifacts; the
> Haiku evaluator can't see a running window, so the *machine-checkable* gate is the e2e smoke + forced-CPU run.

### PKG-1 slice — signed Mac app (DO NOT use auto-mode here)
```
Continue the plan (§7 PKG-1). Bundle the macOS .app (cargo-bundle + Info.plist + icons), then codesign with my
Developer ID, notarize via xcrun notarytool, staple, and build the DMG via scripts/ops/sign-notarize.sh.
Pause and confirm with me before any codesign/notarize/network step — these use my Apple credentials.

(No /goal here — PKG-1 touches credentials/distribution; run it interactively, not autonomously. Done = on a
clean machine, `spctl --assess --type execute` returns "accepted" and `xcrun stapler validate` passes.)
```

### GPU-4 + PKG-2 — Windows / CUDA (later)
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
