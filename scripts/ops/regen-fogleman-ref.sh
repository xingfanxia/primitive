#!/usr/bin/env bash
# Provenance for the CORE-1 quality gate's FOGLEMAN_REF_SCORE constant
# (crates/primitive-core/tests/golden.rs).
#
# Regenerates the committed resized target fixture via this port's *canonical* resize path,
# then runs the real fogleman/primitive binary on that BYTE-IDENTICAL fixture (`-r 0` skips
# fogleman's own resize) and prints each run's final normalized-RMSE score + the mean. The
# mean is what FOGLEMAN_REF_SCORE records. fogleman seeds with wall-clock time, so the mean
# drifts slightly run-to-run — the 1% gate tolerance accounts for it.
#
# Requires: the fogleman binary. Build it once with:
#   (cd go_primitive && GOFLAGS=-mod=mod go build -o /tmp/fogleman-primitive .)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
FOG="${FOGLEMAN_BIN:-/tmp/fogleman-primitive}"
FIXTURE="crates/primitive-core/tests/fixtures/target_picasso_128.png"
RUNS="${RUNS:-5}"

[ -x "$FOG" ] || { echo "fogleman binary not found at $FOG (set FOGLEMAN_BIN or build it — see header)"; exit 1; }

echo "==> regenerating committed target fixture: $FIXTURE (resize via experiment example)"
cargo run --release -q -p primitive-core --example experiment -- assets/picasso.jpg 128 100 1 "$FIXTURE" >/dev/null

echo "==> fogleman on the committed fixture (-r 0 -s 128 -n 100 -m 1 -j 1), $RUNS runs"
sum=0
for s in $(seq 1 "$RUNS"); do
  sc=$("$FOG" -i "$FIXTURE" -r 0 -s 128 -n 100 -m 1 -j 1 -v -o "/tmp/fog_ref_$s.png" 2>&1 \
        | awk -F'score=' '/^100:/{print $2}' | tr -d ',' | awk '{print $1}')
  echo "  run $s: score=$sc"
  sum=$(awk "BEGIN{print $sum + $sc}")
done
mean=$(awk "BEGIN{printf \"%.6f\", $sum / $RUNS}")
echo "==> FOGLEMAN_REF_SCORE (mean over $RUNS runs) = $mean"
echo "    Update the constant in crates/primitive-core/tests/golden.rs if it has drifted materially."
