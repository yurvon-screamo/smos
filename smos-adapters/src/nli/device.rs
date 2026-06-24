//! Runtime device detection + ort [`Session`] construction for the native
//! NLI backend.
//!
//! One binary supports every device. The GPU class is detected at runtime
//! via filesystem probes (no compile-time features), the matching ONNX
//! Runtime DLL is downloaded into the local cache (see
//! [`super::ort_cache`]), and the ort session is built against it. There
//! are no `cfg!(feature = …)` branches — the same `smos` binary runs on
//! CPU, Intel Arc (DirectML), NVIDIA (CUDA), and Apple Silicon (Metal)
//! without rebuilds.
//!
//! [`detect_device`] honours an explicit `[nli_backend].device` override
//! before probing the host, so an operator can force a slower EP for
//! debugging without editing code.

use std::path::Path;

use ort::ep::{self, ExecutionProviderDispatch};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

/// Hardware target the NLI session runs on. Detected at runtime via
/// [`detect_device`]; the chosen variant is logged at startup so the
/// operator can verify which EP ort actually committed.
///
/// Variant order is **not** significant — the runtime priority lives in
/// [`auto_detect`] (CUDA on Windows/Linux > DirectML on Windows >
/// Metal on macOS > CPU), not in the enum declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    DirectML,
    Cuda,
    Metal,
}

impl DeviceKind {
    /// Lowercase canonical token used in logs, diagnostics, and the
    /// `ort_cache` subdir name. Stable string contract — never rename
    /// without coordinating downstream consumers (log dashboards,
    /// alerts, the on-disk cache layout).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::DirectML => "directml",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
}

/// Resolve the device the NLI backend should target.
///
/// `config_device` is the `[nli_backend].device` value from `smos.toml`.
/// Any explicit value (`"cpu"`, `"directml"`, `"cuda"`, `"metal"`)
/// bypasses runtime probing and forces that device — useful for
/// debugging a flaky GPU EP. `"auto"` and any unrecognised string run
/// [`auto_detect`], which probes the host for the best available EP.
pub fn detect_device(config_device: &str) -> DeviceKind {
    match config_device {
        "cpu" => DeviceKind::Cpu,
        "directml" => DeviceKind::DirectML,
        "cuda" => DeviceKind::Cuda,
        "metal" => DeviceKind::Metal,
        _ => auto_detect(),
    }
}

/// Best-effort host GPU probe. Order is platform-aware and **preserves the
/// pre-dynamic-loading priority** so existing NVIDIA-equipped deployments
/// do not silently regress onto a slower EP:
///
/// - Windows: CUDA first (NVIDIA-only, fastest EP for ort on NVIDIA),
///   then DirectML (covers Intel Arc, AMD, NVIDIA via DX12 as a
///   fallback when the CUDA driver is absent).
/// - Linux: CUDA.
/// - macOS: Metal (CoreML) on Apple Silicon.
/// - Anything else or no GPU detected: CPU.
///
/// The probes are deliberately cheap filesystem stats, not driver-init
/// round-trips — the latter would add hundreds of milliseconds to every
/// startup, even on CPU-only CI hosts. A path-stat false positive
/// (DLL present, GPU broken) is caught by ort's own EP-init and surfaces
/// as a clean fallthrough to the next provider in [`provider_chain`].
fn auto_detect() -> DeviceKind {
    #[cfg(target_os = "windows")]
    {
        if has_cuda_support() {
            return DeviceKind::Cuda;
        }
        if has_directml_support() {
            return DeviceKind::DirectML;
        }
    }
    #[cfg(target_os = "linux")]
    {
        if has_cuda_support() {
            return DeviceKind::Cuda;
        }
    }
    #[cfg(target_os = "macos")]
    {
        if is_apple_silicon() {
            return DeviceKind::Metal;
        }
    }
    DeviceKind::Cpu
}

#[cfg(target_os = "windows")]
fn has_directml_support() -> bool {
    // DirectML rides on D3D12; if `d3d12.dll` is missing the host has no
    // DX12-capable OS (Server Core without Desktop Experience, legacy
    // Windows). A bare path stat is enough — actually creating a D3D12
    // device here would multiply startup time on every machine, even
    // CPU-only ones, just to confirm what the DLL's presence already
    // implies.
    Path::new("C:/Windows/System32/d3d12.dll").exists()
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn has_cuda_support() -> bool {
    // NVIDIA's user-mode driver. Its presence is a strong signal that an
    // NVIDIA GPU + driver are installed; absence rules CUDA out without
    // spinning up a CUcontext (slow and racy with other processes
    // initialising CUDA first).
    #[cfg(target_os = "windows")]
    {
        Path::new("C:/Windows/System32/nvcuda.dll").exists()
    }
    #[cfg(target_os = "linux")]
    {
        Path::new("/usr/lib/x86_64-linux-gnu/libcuda.so").exists()
            || Path::new("/usr/lib64/libcuda.so").exists()
            || Path::new("/usr/lib/libcuda.so").exists()
    }
}

#[cfg(target_os = "macos")]
fn is_apple_silicon() -> bool {
    // ort's macOS build bundles CoreML only into the arm64 slice; Intel
    // Macs fall through to CPU. The check is a compile-time constant per
    // target so the dead arm64/intel branch is eliminated by the
    // optimiser.
    std::env::consts::ARCH == "aarch64"
}

/// Build the ordered execution-provider chain for `device`.
///
/// The CPU EP is always listed last so an unsupported operator on the
/// specialised EP degrades to CPU instead of failing the whole session.
/// With `load-dynamic` the EP availability depends on the loaded DLL;
/// ort logs a warning and continues down the chain if a provider cannot
/// initialise.
fn provider_chain(device: DeviceKind) -> Vec<ExecutionProviderDispatch> {
    match device {
        DeviceKind::DirectML => vec![ep::DirectML::default().build(), ep::CPU::default().build()],
        DeviceKind::Cuda => vec![ep::CUDA::default().build(), ep::CPU::default().build()],
        DeviceKind::Metal => vec![ep::CoreML::default().build(), ep::CPU::default().build()],
        DeviceKind::Cpu => vec![ep::CPU::default().build()],
    }
}

/// Normalize a filesystem path to forward slashes for ort's
/// `commit_from_file`.
///
/// ort-rs hands the path string to the native `onnxruntime.dll`'s
/// `CreateSession` after a wide-char encoding step (see
/// `ort::util::path_to_os_char`) — without separator normalisation. The
/// native layer rejects mixed-separator paths like
/// `./data/nli_cache\model_quantized.onnx` with a misleading
/// "system error 13 (permission denied)". The cache layer joins a
/// forward-slash `cache_dir` (passed verbatim from `smos.toml`) with an
/// OS-native file name via `PathBuf::join`, which on Windows produces
/// exactly that mixed shape — so the separator is flattened here, at
/// the ort boundary, instead of polluting the cache layer with platform
/// branches.
fn normalize_model_path(path: &str) -> String {
    path.replace('\\', "/")
}

/// Commit an ort [`Session`] for `model_path` configured for `device`.
///
/// `ort_dll_path` is the location of the dynamically-loaded ONNX Runtime
/// library (see [`super::ort_cache::ensure_ort_binary`]). When `Some`,
/// the path is handed to [`ort::init_from`] BEFORE the session is built.
/// `init_from` is ort's documented, safe API for `load-dynamic` builds:
/// it `LoadLibrary`s/`dlopen`s the DLL, builds the global ort
/// environment against it, and returns a builder — no `unsafe` env-var
/// mutation needed. SMOS calls `build_session` exactly once at startup
/// (single-threaded init in [`super::runtime::build_classifier`]), so
/// the environment is committed before any other ort API is touched.
///
/// When `None`, ort falls back to the `ORT_DYLIB_PATH` env var (if set
/// by the operator) or returns a load error.
///
/// Single intra-op thread because the NLI graph is small enough that the
/// coordination overhead of multi-threaded execution outweighs the
/// speedup for a single (premise, hypothesis) pair. Level3 graph
/// optimisation is the most aggressive preset; the cost is paid once at
/// session build.
pub fn build_session(
    model_path: &str,
    device: DeviceKind,
    ort_dll_path: Option<&Path>,
) -> Result<Session, ort::Error> {
    if let Some(path) = ort_dll_path {
        tracing::info!(dll = %path.display(), "loading dynamic ONNX Runtime");
        // `init_from` is ort's safe API for `load-dynamic`: it loads the
        // dylib and returns an environment builder whose `commit()`
        // finalises the global ort environment. In ort 2.0.0-rc.12
        // `commit()` returns `bool` — `false` means an environment was
        // already committed (by an earlier call) and this configuration
        // is ignored. SMOS builds exactly one ort session at startup,
        // so the false branch would indicate unexpected double init.
        let committed = ort::init_from(path)?.commit();
        if !committed {
            tracing::warn!(
                "ort environment was already committed before DLL load; \
                 the dynamic ONNX Runtime path may not have taken effect"
            );
        }
    } else {
        tracing::warn!(
            "no dynamic ORT DLL provided; ort will honour a manually-set \
             ORT_DYLIB_PATH or fail to load"
        );
    }

    let normalized = normalize_model_path(model_path);
    if normalized != model_path {
        tracing::debug!(
            original = model_path,
            normalized = %normalized,
            "normalized ort model path (forward slashes for ort Windows compat)"
        );
    }

    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_execution_providers(provider_chain(device))?
        .commit_from_file(&normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_str_matches_canonical_token() {
        assert_eq!(DeviceKind::Cpu.as_str(), "cpu");
        assert_eq!(DeviceKind::DirectML.as_str(), "directml");
        assert_eq!(DeviceKind::Cuda.as_str(), "cuda");
        assert_eq!(DeviceKind::Metal.as_str(), "metal");
    }

    #[test]
    fn detect_device_explicit_override_short_circuits_probe() {
        // Any canonical token bypasses auto_detect; this is how an
        // operator forces a slower EP for debugging without editing
        // code. The mapping must stay 1:1 with the documented
        // `[nli_backend].device` enum.
        assert_eq!(detect_device("cpu"), DeviceKind::Cpu);
        assert_eq!(detect_device("directml"), DeviceKind::DirectML);
        assert_eq!(detect_device("cuda"), DeviceKind::Cuda);
        assert_eq!(detect_device("metal"), DeviceKind::Metal);
    }

    #[test]
    fn detect_device_treats_unknown_as_auto() {
        // An unrecognised token falls through to the runtime probe
        // rather than panicking. Whatever auto_detect() returns on this
        // host, the unknown-token and "auto" paths must agree.
        let auto = detect_device("auto");
        let unknown = detect_device("quantum-computing");
        assert_eq!(auto, unknown);
        assert!(matches!(
            auto,
            DeviceKind::Cpu | DeviceKind::DirectML | DeviceKind::Cuda | DeviceKind::Metal
        ));
    }

    #[test]
    fn provider_chain_shape_matches_fallback_contract() {
        // Every specialised device emits a 2-entry chain (specialised
        // EP first, CPU fallback second); pure CPU emits a 1-entry
        // chain. ort::ep::ExecutionProviderDispatch does not expose the
        // wrapped EP's identifier, so we assert the shape rather than
        // read back the EP name.
        let expectations: [(DeviceKind, usize); 4] = [
            (DeviceKind::DirectML, 2),
            (DeviceKind::Cuda, 2),
            (DeviceKind::Metal, 2),
            (DeviceKind::Cpu, 1),
        ];
        for (device, expected_len) in expectations {
            let chain = provider_chain(device);
            assert_eq!(
                chain.len(),
                expected_len,
                "provider chain for {device:?} has wrong length"
            );
            assert!(!chain.is_empty(), "provider chain must not be empty");
        }
    }

    #[test]
    fn normalize_model_path_flattens_mixed_separators() {
        // Reproduces the exact shape produced on Windows when `cache_dir`
        // arrives from `smos.toml` with forward slashes and `PathBuf::join`
        // appends a file name with a backslash — the case that surfaced
        // as ort's misleading "system error 13 (permission denied)".
        let input = "./data/nli_cache\\model_quantized.onnx";
        assert_eq!(
            normalize_model_path(input),
            "./data/nli_cache/model_quantized.onnx"
        );
    }

    #[test]
    fn normalize_model_path_replaces_all_backslashes() {
        // A fully Windows-native path (e.g. when `cache_dir` is configured
        // as an absolute Windows path) — every separator must flip.
        let input = ".\\data\\nli_cache\\model_quantized.onnx";
        assert_eq!(
            normalize_model_path(input),
            "./data/nli_cache/model_quantized.onnx"
        );
    }

    #[test]
    fn normalize_model_path_is_idempotent_on_forward_slashes() {
        // A path that is already ort-safe must pass through untouched so
        // the debug log guard in `build_session` does not spam on
        // Linux/macOS or on a `cache_dir` that was already
        // forward-slashed.
        let input = "./data/nli_cache/model_quantized.onnx";
        assert_eq!(normalize_model_path(input), input);
    }

    #[test]
    fn normalize_model_path_flips_drive_letter_prefixed_path() {
        // A backslash inside a Windows drive-letter path is still a
        // separator and must flip — the drive token (`C:`) is left
        // intact because only the separator character is replaced. ort
        // accepts forward-slash drive paths on Windows.
        assert_eq!(
            normalize_model_path("C:\\Users\\me\\.cache\\model_quantized.onnx"),
            "C:/Users/me/.cache/model_quantized.onnx"
        );
    }
}
