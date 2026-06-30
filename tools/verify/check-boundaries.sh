#!/usr/bin/env bash
# Architecture boundary check (plan §5). Enforces the hexagonal import direction:
#
#   core (pure) ← compute (ports) ← engine (orchestration) ← adapters / app (composition root)
#
# Two layers of enforcement:
#   1. Manifest: each crate's runtime dependencies (any `[dependencies]`, `[dependencies.X]`,
#      `[build-dependencies]`, `[target.*.dependencies]` form — but NOT dev-dependencies, which
#      ARE the composition root for tests/benches) may only name allowed crates.
#   2. Source: no GPU types (wgpu/cubecl/cudarc) and no cross-boundary `use` in src/ (comment-
#      only lines are ignored so a doc comment mentioning a backend never false-fails).
#
# Exit 0 = clean. Non-zero = a violation (printed). Run from anywhere; CI runs this verbatim.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fail=0

# Print every runtime (non-dev) dependency name in a Cargo.toml, one per line. Section-aware
# so the table-header form `[dependencies.foo]` and `[target.'cfg(...)'.dependencies]` are
# covered, not just inline `foo = { ... }`. BSD-awk compatible (no gawk-only 3-arg match).
runtime_dep_names() {
  awk '
    /^\[/ {
      s = tolower($0)
      dev = (s ~ /\[dev-dependencies/)
      rt  = (s ~ /\[dependencies/ || s ~ /dependencies\]/ || s ~ /\.dependencies/ || s ~ /build-dependencies/)
      if ($0 ~ /dependencies\.[A-Za-z0-9_-]+\]/) {       # [..dependencies.NAME] table header
        name = $0; sub(/.*dependencies\./, "", name); sub(/\].*/, "", name)
        if (!dev) print name
      }
      insect = (rt && !dev)
      next
    }
    insect && /^[[:space:]]*[A-Za-z0-9_-]+[[:space:]]*=/ {
      name = $0; sub(/[[:space:]]*=.*/, "", name); gsub(/[[:space:]]/, "", name)
      print name
    }
  ' "$1"
}

# manifest_allows <crate> <space-separated allowed primitive-* deps>
manifest_allows() {
  local crate="$1"; shift
  local allowed=" $* "
  local toml="$ROOT/crates/$crate/Cargo.toml"
  [ -f "$toml" ] || { echo "MISSING: $toml"; fail=1; return; }
  while IFS= read -r dep; do
    [ -z "$dep" ] && continue
    case "$dep" in
      primitive-*)
        if [[ "$allowed" != *" $dep "* ]]; then
          echo "BOUNDARY VIOLATION: crate '$crate' depends on '$dep' (not allowed at this layer)"
          fail=1
        fi
        ;;
    esac
  done < <(runtime_dep_names "$toml")
}

# gpu_free <crate> — the layer must carry no GPU/GUI dependency (adapters are exempt).
gpu_free() {
  local crate="$1"
  local toml="$ROOT/crates/$crate/Cargo.toml"
  [ -f "$toml" ] || return 0
  while IFS= read -r dep; do
    case "$dep" in
      wgpu*|cubecl*|cudarc*|eframe*|egui*)
        echo "BOUNDARY VIOLATION: crate '$crate' has GPU/GUI dep '$dep' (must be GPU-free)"
        fail=1
        ;;
    esac
  done < <(runtime_dep_names "$toml")
}

# src_forbids <crate> <ERE> <message> — scans src/, ignoring comment-only lines.
src_forbids() {
  local crate="$1" pat="$2" msg="$3"
  local dir="$ROOT/crates/$crate/src"
  [ -d "$dir" ] || return 0
  # Drop matches whose content (after `path:line:`) begins with a comment marker.
  if grep -rEn "$pat" "$dir" 2>/dev/null | grep -vE ':[0-9]+:[[:space:]]*(//|/\*|\*)' >/tmp/boundary_hits; then
    echo "BOUNDARY VIOLATION: crate '$crate' src — $msg"
    cat /tmp/boundary_hits
    fail=1
  fi
}

# --- Manifest contract: allowed runtime dependencies per layer ---
manifest_allows primitive-core              # core: zero primitive-* deps (pure)
manifest_allows primitive-compute   primitive-core
manifest_allows primitive-gpu-cpu   primitive-core primitive-compute
manifest_allows primitive-gpu-cubecl primitive-core primitive-compute  # GPU adapter, same layer as CPU
manifest_allows primitive-engine    primitive-core primitive-compute   # NOT any adapter

# These layers must carry no GPU/GUI dependency (the GPU adapter legitimately does).
gpu_free primitive-core
gpu_free primitive-compute
gpu_free primitive-gpu-cpu
gpu_free primitive-engine

# --- Source contract ---
# core is pure: no GPU types, no cross-boundary use, no IO/clock in src.
src_forbids primitive-core '(wgpu|cubecl|cudarc)::' 'core must be GPU-free'
src_forbids primitive-core '^[[:space:]]*use[[:space:]]+primitive_(compute|engine|gpu)' 'core must not import compute/engine/adapters'
src_forbids primitive-core '^[[:space:]]*use[[:space:]]+std::(fs|net|process|env|time)' 'core must do no IO/clock/env'
# compute (ports) must not reach engine/adapters or GPU types.
src_forbids primitive-compute '^[[:space:]]*use[[:space:]]+primitive_(engine|gpu)' 'ports must not import engine/adapters'
src_forbids primitive-compute '(wgpu|cubecl|cudarc)::' 'ports must be backend-agnostic'
# engine must not import a concrete adapter (only the ports), via `use` OR fully-qualified path.
src_forbids primitive-engine 'primitive_gpu_[a-z]+::' 'engine must not reference a concrete adapter'
# adapters implement the ports; they must not depend on the engine (one-way: adapter ← engine).
src_forbids primitive-gpu-cpu '^[[:space:]]*use[[:space:]]+primitive_engine' 'adapter must not import the engine'
src_forbids primitive-gpu-cubecl '^[[:space:]]*use[[:space:]]+primitive_engine' 'adapter must not import the engine'

if [ "$fail" -eq 0 ]; then
  echo "check-boundaries: OK — hexagonal import direction holds (core ← compute ← engine ← adapters)"
fi
exit "$fail"
