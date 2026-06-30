# GPU-4 Runbook — CUDA backend on a discrete NVIDIA GPU (different machine)

**Run this on a machine with a discrete NVIDIA GPU (Linux or Windows).** It does NOT run on the
Apple Silicon dev machine — that's the whole point.

## Why this milestone exists (measured, not assumed)

On the M-series dev Mac the GPU adapter is **only ~1.9× a 16-core fogleman at 64×64, and *loses*
(0.3–0.5×) at 128×128** — see `.agent/EVIDENCE.md` → "fogleman-vs-GPU comparison + size scaling".
Root cause is **unified memory**: the M-series GPU shares one memory controller/bandwidth with the
CPU, so a bigger canvas (where a GPU should pull ahead) gives no bandwidth edge, while the parallel
search needs ~3× deeper climbs to hold quality → the GPU gets *relatively slower* as the canvas grows.

A **discrete** GPU is the opposite regime: dedicated high-bandwidth VRAM (≈500–3000 GB/s vs CPU
DDR ≈50–100 GB/s) + thousands of cores. The hypothesis this milestone *settles with data*: at
256×256 / 512×512 on a discrete card, the GPU should beat a multi-core fogleman by a wide margin.
**If it doesn't, the "GPU-native primitive" thesis is bounded to small canvases — record that result
either way.**

Second win: CUDA has **native i64**, so the i32 overflow that caps the Metal path at 64×64 (and the
WGSL-has-no-i64 hi-lo workaround we deliberately skipped) simply goes away — large canvases compile
straight through.

## Acceptance gates (from plan §7 GPU-4 — these ARE the tests)

1. **Parity on NVIDIA**: `gpu2_triangles` + `gpu_parity` pass bit-exact on the CUDA backend
   (winning-candidate index + integer delta-SSE identical to the CPU oracle).
2. **Cross-backend golden**: the integer-path output is **bit-for-bit identical to the Metal run**
   on the same fixture (paste a `sha256` of the score/canvas bytes from both machines).
3. **Throughput ≥ 20× a same-machine single-core CPU baseline** — hard-asserted only under
   `PRIMITIVE_PERF_GATE=1` (run `make perf`, or `PRIMITIVE_PERF_GATE=1 cargo test -p
   primitive-gpu-cubecl --release --features cuda --no-default-features --test gpu2_throughput
   --test gpu3_optimize -- --nocapture`). Plain `cargo test` / `make verify` only prints the number
   (the threshold is hardware-dependent — see `crates/primitive-gpu-cubecl/tests/common/mod.rs`).
   On a discrete NVIDIA card this is representative hardware, so the gate is meaningful here.
4. **The headline measurement**: GPU vs fogleman (`-j1` and all-cores) at 64 / 128 / 256 / 512,
   recorded in `.agent/EVIDENCE.md`. This is the number the Mac can't produce.

## Prereqs

- NVIDIA GPU + driver; **CUDA Toolkit 12.x** (`nvcc --version` works).
- Rust via rustup, **stable ≥ 1.92** (CubeCL 0.10 needs it): `rustup default stable`.
  - On this repo the toolchain is pinned in `rust-toolchain.toml`; confirm it resolves ≥1.92.
- Go 1.22+ (only to re-run the fogleman baseline `go_primitive/primitive/bench_test.go`).

## Repo state you're starting from

- Branch: **`feat/primitive-core-cpu-slice`** (has CORE-1/2 + GPU-1/2/3 + the perf pass).
- The Metal path is `primitive-gpu-cubecl` using `cubecl::wgpu::WgpuRuntime`. The `#[cube]` kernels
  in `src/kernels.rs` + `src/search.rs` are **backend-agnostic source** — CUDA reuses them verbatim.
- The whole numeric contract is integer/u32 (Philox RNG is pure-u32, already portable). **Keep it
  that way** — no float, no 64-bit atomics (plan §6.6).

## Steps

1. **Clone + checkout**
   ```bash
   git clone https://github.com/xingfanxia/primitive && cd primitive
   git checkout feat/primitive-core-cpu-slice
   export PATH="$HOME/.cargo/bin:$PATH"
   ```

2. **Add the CUDA runtime** to `crates/primitive-gpu-cubecl/Cargo.toml`:
   ```toml
   [features]
   default = ["wgpu"]
   wgpu = ["cubecl/wgpu"]
   cuda = ["cubecl/cuda"]
   ```
   (Re-check the exact feature name with `cargo search cubecl` / the cubecl 0.10 docs — it may be
   `cubecl-cuda`. `cudarc` 0.19.x is the transitive driver crate.)

3. **Abstract the runtime.** Today `lib.rs` hardcodes `WgpuRuntime`. Make the host fns generic over
   `R: Runtime` (or gate with `#[cfg(feature="cuda")] type Rt = CudaRuntime;` /
   `#[cfg(not)] type Rt = WgpuRuntime;`). The `launch_unchecked::<R>` calls already take the runtime
   as a type param, so this is mechanical. Composition root picks the backend.

4. **i64 accumulators for >64×64.** On CUDA the SSE/color accumulators can be plain `i64` (no hi-lo).
   Either (a) bump the accumulator type to i64 unconditionally on the CUDA path, or (b) keep i32 for
   the 64×64 parity test and add an i64 variant for large canvases. The kernels that overflow past
   64×64 are the `rsum/gsum/bsum` and `delta` accumulators in `score_one` / `score_one_scanline`
   (see `.agent/PROGRESS.md` carry-forward). **Add a 128×128 parity golden** before trusting i64.

5. **Parity on NVIDIA** (gate 1):
   ```bash
   cargo test -p primitive-gpu-cubecl --release --features cuda --no-default-features \
     --test gpu2_triangles --test gpu_parity -- --nocapture
   ```
   Both must report `1000/1000 ... bit-identical`. If they diverge, the integer-determinism design
   (§6.6) is broken on CUDA — debug there first, it's the load-bearing contract.

6. **Cross-backend golden** (gate 2): hash the integer scores from a fixed batch on CUDA and compare
   to the Metal hash. Easiest: add a tiny test/example that prints `sha256(bytemuck::cast_slice(&scores))`
   for the committed `parity_fogleman.json` batch; run it on both machines; the hashes must match.

7. **Throughput + the headline measurement** (gates 3–4): run the sweep at each size and re-run the
   fogleman baseline on the same box:
   ```bash
   cargo run -p primitive-gpu-cubecl --release --features cuda --no-default-features \
     --example sweep -- 64  100
   cargo run ... --example sweep -- 128 100
   cargo run ... --example sweep -- 256 100      # needs the i64 path from step 4
   cargo run ... --example sweep -- 512 100
   cd go_primitive && go test ./primitive/ -run TestBenchFogleman -v   # add 256/512 to the size loop
   ```
   The `sweep` example and the size-parameterized Go bench already exist on this branch.

8. **Record results** in `.agent/EVIDENCE.md` under a new "GPU-4 (CUDA, <gpu name>)" section: the
   parity confirmations, the golden hashes (both machines), and the GPU-vs-fogleman table at
   64/128/256/512. Update `.agent/PROGRESS.md` GPU-4 status. Commit on a `feat/gpu4-cuda` branch.

## Decision to make from the results (plan Open Q #1)

If CubeCL's CUDA codegen is unstable, the fallback is **wgpu-Vulkan/DX12 on NVIDIA** (keeps one
kernel source, drops the literal "CUDA" runtime) vs **hand-written cudarc kernel** behind the same
port. Decide from the parity-suite results — prefer wgpu-Vulkan only if "GPU-accelerated on NVIDIA"
is acceptable in place of "CUDA specifically."

## What success looks like

A printed line like `GPU 256×256: 380 shapes/s vs fogleman -j16: 41 shapes/s = 9.3×`, parity green,
golden hashes equal across Metal+CUDA. That single number is the thing the Mac fundamentally cannot
produce — it's the real headline of the GPU-native rebuild.
