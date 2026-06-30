//! Shared test helpers for the GPU performance gates.
//!
//! Throughput is hardware-dependent. The GPU-2 ≥20× and GPU-3 ≥460 shapes/sec claims hold on
//! representative target hardware (Apple Silicon / discrete NVIDIA), but a shared or virtualized CI
//! runner's GPU is far slower — and because Apple Silicon's unified memory also widens the GPU/CPU
//! ratio, even a *relative* ratio gate isn't portable (observed 9.6× on a GitHub `macos-latest`
//! runner vs 43.9× on an M-series dev machine). So the perf tests always measure and print (the
//! number stays visible in every CI log), but the hard threshold is **opt-in**: it asserts only when
//! `PRIMITIVE_PERF_GATE` is set, which `make perf` does on representative hardware. This keeps
//! `make verify` / CI enforcing correctness everywhere while the perf claim is gated where it is
//! actually valid. Correctness and quality gates (integer parity, golden SSIM, the GPU-3 PSNR gap)
//! are hardware-independent and stay unconditional in the suite.

/// Assert a "higher is better" performance threshold, but only when the perf gate is enabled via the
/// `PRIMITIVE_PERF_GATE` env var (set by `make perf`). When it is unset — the default, including CI —
/// this measures and reports without failing, so a slow non-representative runner can't break the
/// correctness build. `what` names the metric (e.g. `"GPU-2 throughput speedup (×)"`).
#[track_caller]
pub fn perf_gate_min(measured: f64, threshold: f64, what: &str) {
    let met = measured >= threshold;
    if std::env::var_os("PRIMITIVE_PERF_GATE").is_some() {
        assert!(met, "{what}: {measured:.2} < gate {threshold:.2}");
    } else if !met {
        println!(
            "  [perf] SOFT (PRIMITIVE_PERF_GATE unset — non-representative runner): {what}: \
             {measured:.2} < gate {threshold:.2}. Measuring only; run `make perf` on target \
             hardware to enforce."
        );
    }
}
