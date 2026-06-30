//! GPU-3 foundation: the kernel's counter-based RNG is bit-identical to the CPU reference, so a
//! GPU thread can run a reproducible search from its indices alone (plan §6.6). Without this the
//! on-device hill-climb would be non-deterministic and unverifiable.

use primitive_core::rand_below;
use primitive_gpu_cubecl::gpu_philox_fill;

#[test]
fn gpu_philox_matches_cpu_exactly() {
    const N: usize = 10_000;
    const SEED: u32 = 0xC0FFEE;
    const RANGE: u32 = 64;

    let gpu = gpu_philox_fill(SEED, RANGE, N);
    assert_eq!(gpu.len(), N);

    let mut mismatches = 0;
    for (i, &g) in gpu.iter().enumerate() {
        let cpu = rand_below(SEED, i as u32, RANGE);
        if g != cpu {
            if mismatches < 5 {
                eprintln!("ctr {i}: gpu={g} cpu={cpu}");
            }
            mismatches += 1;
        }
        assert!(g < RANGE);
    }
    println!(
        "GPU Philox RNG: {}/{} draws bit-identical to CPU (range {RANGE})",
        N - mismatches,
        N
    );
    assert_eq!(
        mismatches, 0,
        "{mismatches}/{N} RNG draws diverged — determinism substrate broken"
    );
}
