//! # primitive-gpu-cubecl — GPU adapter (Metal now, CUDA at GPU-4)
//!
//! CubeCL `#[cube]` kernels compiled to Metal (via wgpu) from one source. Implements the
//! `ShapeSearch` port and is held to **exact integer parity** against the CPU oracle.
//!
//! GPU-1 (this file) is a toolchain spike: a trivial i32 vector-add kernel that proves the
//! CubeCL→Metal device init, buffer upload, launch, and readback path works on this Mac.
//! The fused `raster_score` kernel (§6.3) replaces it next.

use cubecl::prelude::*;

/// Trivial element-wise add — the spike kernel that validates the Metal pipeline.
#[cube(launch_unchecked)]
pub fn vadd_i32(a: &Array<i32>, b: &Array<i32>, out: &mut Array<i32>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

/// Run the spike kernel on the default wgpu (Metal) device; returns `a + b` element-wise.
pub fn gpu_vadd_i32(a: &[i32], b: &[i32]) -> Vec<i32> {
    use cubecl::wgpu::WgpuRuntime;
    assert_eq!(a.len(), b.len());
    let n = a.len();

    let device = Default::default();
    let client = WgpuRuntime::client(&device);

    let a_h = client.create_from_slice(bytemuck::cast_slice(a));
    let b_h = client.create_from_slice(bytemuck::cast_slice(b));
    let out_h = client.empty(core::mem::size_of_val(a)); // n * size_of::<i32>()

    let threads = 64u32;
    let groups = n.div_ceil(threads as usize) as u32;
    unsafe {
        vadd_i32::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(groups, 1, 1),
            CubeDim::new_1d(threads),
            ArrayArg::from_raw_parts(a_h.clone(), n),
            ArrayArg::from_raw_parts(b_h.clone(), n),
            ArrayArg::from_raw_parts(out_h.clone(), n),
        );
    }

    let bytes = client.read_one_unchecked(out_h);
    bytemuck::cast_slice(&bytes).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vadd_matches_cpu_on_metal() {
        let a: Vec<i32> = (0..1000).collect();
        let b: Vec<i32> = (0..1000).map(|x| x * 3 - 7).collect();
        let gpu = gpu_vadd_i32(&a, &b);
        let cpu: Vec<i32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert_eq!(gpu, cpu);
    }
}
