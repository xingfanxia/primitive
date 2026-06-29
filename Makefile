# Primitive 2026 — developer entry points. `make verify` is the canonical gate (CI runs it).
.PHONY: verify test fmt clippy boundaries build baseline golden bundle icon

verify:
	./tools/verify/check.sh

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
