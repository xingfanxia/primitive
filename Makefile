# Primitive 2026 — developer entry points. `make verify` is the canonical CORRECTNESS gate (CI runs
# it); `make perf` is the hardware-dependent PERFORMANCE gate, run on representative hardware only.
.PHONY: verify perf test fmt clippy boundaries build baseline golden bundle icon

verify:
	./tools/verify/check.sh

# Performance gates (hardware-dependent): enforce the headline GPU-2 ≥20× and GPU-3 ≥460 shapes/sec
# claims. NOT part of `make verify` / CI — a CI runner's shared/virtualized GPU is far slower, so a
# fixed numeric gate isn't portable (see crates/primitive-gpu-cubecl/tests/common/mod.rs). Run this
# on the dev machine (Apple Silicon) or a discrete-NVIDIA box; PRIMITIVE_PERF_GATE flips the soft
# measurements into hard assertions.
perf:
	PRIMITIVE_PERF_GATE=1 cargo test -p primitive-gpu-cubecl --release \
		--test gpu2_throughput --test gpu3_optimize -- --nocapture

build:
	cargo build --workspace

test:
	cargo test --workspace

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

boundaries:
	./tools/verify/check-boundaries.sh

# Print the single-threaded CPU baseline (shapes/sec, candidates/sec).
baseline:
	cargo test -p primitive-engine --test cpu_baseline --release -- --nocapture cpu_baseline_throughput

# Print the CORE-1 golden SSIM + quality margin.
golden:
	cargo test -p primitive-core --test golden --release -- --nocapture

# PKG-1 Part A: build + validate the macOS .app and HALT before codesign (no credentials, no network).
bundle:
	./scripts/ops/sign-notarize.sh

# Regenerate the app icon set (flat-geometric triangle mark) + primitive.icns.
icon:
	python3 scripts/ops/gen-icon.py
	iconutil -c icns assets/icons/primitive.iconset -o assets/icons/primitive.icns
