//! Backend probe (plan §5A "Device/perf" row). Maps the machine + an env override to the [`Device`]
//! the device chip displays. This is the *only* place the app reaches the GPU adapter for detection;
//! the run itself wires the backend in [`crate::runner`].

use crate::state::Device;

/// Env var that forces the CPU path and the amber "no GPU found" chip — the supported way to
/// exercise the graceful-degradation state on a machine that *does* have a GPU (the GUI-2 gate).
pub const FORCE_CPU_ENV: &str = "PRIMITIVE_FORCE_CPU";

/// Probe the active backend once (the launch-time device check).
///
/// `PRIMITIVE_FORCE_CPU` set → [`Device::CpuFallback`] (simulate a no-GPU machine). Otherwise probe
/// the GPU adapter: present → [`Device::Metal`] (this build's wgpu→Metal target; CUDA arrives at
/// GPU-4 on NVIDIA), absent/faulted → [`Device::CpuFallback`].
pub fn detect() -> Device {
    if std::env::var_os(FORCE_CPU_ENV).is_some() {
        return Device::CpuFallback;
    }
    if primitive_gpu_cubecl::gpu_available() {
        Device::Metal
    } else {
        Device::CpuFallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_cpu_env_yields_amber_fallback() {
        // SAFETY: single-threaded test; we set then clear the override. This is the contract the
        // forced-CPU gate relies on (env set ⇒ "CPU (no GPU found)").
        unsafe { std::env::set_var(FORCE_CPU_ENV, "1") };
        let d = detect();
        unsafe { std::env::remove_var(FORCE_CPU_ENV) };
        assert_eq!(d, Device::CpuFallback);
        assert_eq!(d.chip_label(), "CPU (no GPU found)");
    }
}
