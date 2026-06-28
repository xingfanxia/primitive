# AGENTS.md — primitive 2026 rebuild (routing map)

GPU-native rebuild of `fogleman/primitive` in Rust. **Spec is the source of truth:**
[`docs/plan/primitive-2026-architecture.md`](docs/plan/primitive-2026-architecture.md) (§5 layout, §6 kernel,
§7 milestones+gates) and provenance [`docs/research/primitive-2026-research-findings.md`](docs/research/primitive-2026-research-findings.md).
Milestone state + evidence: [`.agent/PROGRESS.md`](.agent/PROGRESS.md).

## Verify (the one gate)

```
make verify        # fmt + clippy -D warnings + boundaries + giant-file + cargo test (release)
```
"Done" = `make verify` exits 0. CI (`.github/workflows/ci.yml`) runs the same on macOS arm64.
Useful sub-targets: `make baseline` (CPU shapes/sec), `make golden` (SSIM + quality margin).

## Crate map & import direction (hexagonal — enforced by `tools/verify/check-boundaries.sh`)

```
core (pure) ← compute (ports) ← engine (orchestration) ← adapters / app (composition root)
```

| Crate | Layer | May depend on | Notes |
|---|---|---|---|
| `primitive-core` | domain (pure) | — | shapes, raster, color, score, ES, model. No GPU/IO/clock/global-RNG. RNG is injected. The parity oracle + golden reference. |
| `primitive-compute` | ports | core | `ShapeSearch` trait + DTOs (`SearchParams`=`core::SearchBudget`, `BestShape`, `Backend`). Backend-agnostic. |
| `primitive-gpu-cpu` | adapter | core, compute | `CpuSearch`: single-threaded reference search → permanent parity oracle + GPU-absent fallback. |
| `primitive-engine` | application | core, compute | `Engine<S: ShapeSearch>`: per-shape loop. Never names a concrete adapter (composition root is in `tests/`). |

GPU (`primitive-gpu-cubecl`) and GUI (`primitive-app`) crates are added at GPU-1 / GUI-1 — not present yet.

## Where things are proven (gates per plan §7)

- **Math parity vs fogleman (exact)** — `crates/primitive-core/tests/parity.rs` against committed
  `tests/fixtures/parity_fogleman.json` (regenerate: `cd go_primitive && go test -run TestDumpParityFixture ./primitive`).
- **CORE-1 golden** (SSIM ≥ 0.999 determinism + final score within 1% of fogleman) — `tests/golden.rs`.
- **CORE-2 parity + CPU baseline** — `crates/primitive-engine/tests/cpu_baseline.rs` (byte-identical to core; prints shapes/sec).

## Rules for changes here

- New code is TDD-first against the §7 gate. The CPU adapter is the permanent oracle — never delete it.
- Keep kernels (later) free of float/64-bit atomics; integer SSE is the determinism contract (§6.6).
- Re-run `make verify` before declaring done. Don't add a crate dependency that crosses the import direction.
