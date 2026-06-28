#!/usr/bin/env bash
# Canonical verification harness — the single gate (`make verify` runs this; CI runs it verbatim).
# "Done" = this exits 0. Runs static + architecture + giant-file + the full test suite.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
rc=0
step() { echo; echo "==> $*"; }

step "rustfmt (check)"
cargo fmt --all --check || { echo "FAIL: run 'cargo fmt --all'"; rc=1; }

step "clippy (-D warnings)"
cargo clippy --workspace --all-targets -- -D warnings || rc=1

step "architecture boundaries"
"$ROOT/tools/verify/check-boundaries.sh" || rc=1

step "giant-file gate (non-generated src > 500 LOC)"
big=0
while IFS= read -r f; do
  n=$(wc -l < "$f")
  if [ "$n" -gt 500 ]; then echo "  TOO BIG ($n LOC): $f"; big=1; fi
done < <(find "$ROOT/crates" -path '*/src/*.rs' -not -path '*/target/*')
if [ "$big" -ne 0 ]; then echo "FAIL: file(s) over the 500 LOC architecture gate"; rc=1; else echo "  ok — all src files within budget"; fi

step "cargo test (workspace, release)"
cargo test --workspace --release || rc=1

echo
if [ "$rc" -eq 0 ]; then echo "verify: ALL GREEN"; else echo "verify: FAILED"; fi
exit "$rc"
