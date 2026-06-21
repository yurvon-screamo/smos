//! Device detection + ort [`Session`] construction for the native NLI backend.
//!
//! The classifier targets the hardware the binary was compiled for, selected at
//! session-build time. Each cargo feature below pulls in the matching ort
//! execution provider (EP); an EP that was not compiled in cannot be selected,
//! so detection is purely compile-time.
//!
//! # GPU support
//!
//! - `nli-cuda`       — NVIDIA GPU via ort's CUDA EP. Prebuilt binaries.
//! - `nli-directml`   — Intel Arc, AMD, NVIDIA via DirectX 12 on Windows.
//!   **Recommended for Intel Arc on Windows** — works out of the box with
//!   the ort prebuilt binary.
//! - `nli-metal`      — Apple Silicon via ort's CoreML EP.
//! - `nli-webgpu`     — Cross-platform via ort's WebGPU EP (Vulkan / DX12 / Metal).
//!   Note: ort cannot combine WebGPU with CUDA in a single prebuilt binary,
//!   so do not enable both at once.
//!
//! [`detect_device`] picks the best target the binary was compiled for in a
//! fixed platform-aware priority order (see the function's docs).

use ort::ep::{self, ExecutionProviderDispatch};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

/// Hardware target the NLI session runs on.
///
/// Variant order is **not** significant — the runtime priority lives in
/// [`detect_device`] (CUDA > DirectML on Windows > Metal on macOS > WebGPU >
/// CPU), not in the enum declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cuda,
    DirectML,
    Metal,
    WebGpu,
    Cpu,
}

impl DeviceKind {
    /// Lowercase canonical token used in logs and diagnostics. Named after the
    /// **hardware class** (cuda, metal, …), not the underlying ort EP, so
    /// `directml` and `webgpu` are the lone exceptions (they have no separate
    /// hardware class — DirectML covers any DX12 GPU, WebGPU covers any
    /// GPU backend). Stable string contract — never rename without
    /// coordinating downstream consumers (log dashboards, alerts).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::DirectML => "directml",
            Self::Metal => "metal",
            Self::WebGpu => "webgpu",
            Self::Cpu => "cpu",
        }
    }
}

/// Pick the best device the binary was compiled for.
///
/// Priority order (platform-aware):
///
/// 1. **CUDA** — top priority where compiled in and the platform supports it
///    (Windows + Linux). Skipped on macOS because ort's CUDA EP is not built
///    there.
/// 2. **DirectML** — Windows-only. Works out of the box for Intel Arc, AMD and
///    NVIDIA via DirectX 12.
/// 3. **Metal** — macOS-only via CoreML.
/// 4. **WebGPU** — cross-platform fallback GPU path (Vulkan / DX12 / Metal).
/// 5. **CPU** — pure fallback. Always available.
///
/// Only the devices whose cargo feature (`nli-cuda` / `nli-directml` /
/// `nli-metal` / `nli-webgpu`) is enabled are even considered — without the
/// EP shared libraries, a runtime probe would always come back negative and
/// just cost startup time.
pub fn detect_device() -> DeviceKind {
    // `cfg!()` evaluates to a compile-time constant so dead branches are
    // eliminated without triggering the `unreachable_code` lint that a chain
    // of `#[cfg]` + early `return` would produce.
    if cfg!(all(feature = "nli-cuda", not(target_os = "macos"))) {
        DeviceKind::Cuda
    } else if cfg!(all(feature = "nli-directml", target_os = "windows")) {
        DeviceKind::DirectML
    } else if cfg!(all(feature = "nli-metal", target_os = "macos")) {
        DeviceKind::Metal
    } else if cfg!(feature = "nli-webgpu") {
        DeviceKind::WebGpu
    } else {
        DeviceKind::Cpu
    }
}

/// Build the ordered execution-provider chain for `device`.
///
/// The CPU EP is always listed last so an unsupported operator on the
/// specialised EP degrades to CPU instead of failing the whole session.
fn provider_chain(device: DeviceKind) -> Vec<ExecutionProviderDispatch> {
    match device {
        DeviceKind::Cuda => vec![ep::CUDA::default().build(), ep::CPU::default().build()],
        DeviceKind::DirectML => vec![ep::DirectML::default().build(), ep::CPU::default().build()],
        DeviceKind::Metal => vec![ep::CoreML::default().build(), ep::CPU::default().build()],
        DeviceKind::WebGpu => vec![ep::WebGPU::default().build(), ep::CPU::default().build()],
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
/// OS-native file name via `PathBuf::join`, which on Windows produces exactly
/// that mixed shape — so the separator is flattened here, at the ort
/// boundary, instead of polluting the cache layer with platform branches.
///
/// Caveat: a Windows verbatim path (`\\?\C:\…`, where the `\\?\` prefix
/// opts out of Win32 normalisation and `/` is not a separator) would be
/// corrupted by the slash flip. Irrelevant in practice because `cache_dir`
/// arrives as a relative path from `smos.toml` and `PathBuf::join` never
/// synthesises a verbatim prefix; noted here so a future change that starts
/// piping verbatim paths through this helper knows to add a guard.
///
/// Pure helper (no IO) so every branch is coverable by unit tests without
/// building a real ort session.
fn normalize_model_path(path: &str) -> String {
    path.replace('\\', "/")
}

/// Commit an ort [`Session`] for `model_path` configured for `device`.
///
/// Single intra-op thread because the NLI graph is small enough that the
/// coordination overhead of multi-threaded execution outweighs the speedup
/// for a single (premise, hypothesis) pair. Level3 graph optimisation is the
/// most aggressive preset; the cost is paid once at session build.
///
/// `model_path` is run through [`normalize_model_path`] before being handed
/// to ort — see that function's docs for the Windows mixed-separator
/// rationale.
pub fn build_session(model_path: &str, device: DeviceKind) -> Result<Session, ort::Error> {
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
        assert_eq!(DeviceKind::Cuda.as_str(), "cuda");
        assert_eq!(DeviceKind::DirectML.as_str(), "directml");
        assert_eq!(DeviceKind::Metal.as_str(), "metal");
        assert_eq!(DeviceKind::WebGpu.as_str(), "webgpu");
        assert_eq!(DeviceKind::Cpu.as_str(), "cpu");
    }

    #[test]
    fn detect_device_picks_a_compiled_target() {
        // The detection is feature-driven + platform-aware; the assertions
        // below mirror the exact priority order documented on
        // `detect_device`. Exactly one branch fires per build configuration.
        let detected = detect_device();

        #[cfg(all(feature = "nli-cuda", not(target_os = "macos")))]
        assert_eq!(detected, DeviceKind::Cuda);

        #[cfg(all(
            not(all(feature = "nli-cuda", not(target_os = "macos"))),
            all(feature = "nli-directml", target_os = "windows")
        ))]
        assert_eq!(detected, DeviceKind::DirectML);

        #[cfg(all(
            not(all(feature = "nli-cuda", not(target_os = "macos"))),
            not(all(feature = "nli-directml", target_os = "windows")),
            all(feature = "nli-metal", target_os = "macos")
        ))]
        assert_eq!(detected, DeviceKind::Metal);

        #[cfg(all(
            not(all(feature = "nli-cuda", not(target_os = "macos"))),
            not(all(feature = "nli-directml", target_os = "windows")),
            not(all(feature = "nli-metal", target_os = "macos")),
            feature = "nli-webgpu"
        ))]
        assert_eq!(detected, DeviceKind::WebGpu);

        #[cfg(not(any(
            all(feature = "nli-cuda", not(target_os = "macos")),
            all(feature = "nli-directml", target_os = "windows"),
            all(feature = "nli-metal", target_os = "macos"),
            feature = "nli-webgpu"
        )))]
        assert_eq!(detected, DeviceKind::Cpu);
    }

    #[test]
    fn provider_chain_shape_matches_fallback_contract() {
        // Documents the contract of `provider_chain`: every specialised
        // device appends a CPU EP as the last entry so an unsupported
        // operator degrades to CPU instead of failing the session. The pure
        // CPU device emits a single-entry chain.
        //
        // `ort::ep::ExecutionProviderDispatch` does not expose the wrapped
        // EP's identifier, so we assert the *shape* of the chain rather
        // than read back the EP name: a 2-entry chain for every
        // non-`Cpu` device (specialised EP first, CPU fallback second) and
        // a 1-entry chain for `Cpu`.
        let expectations: [(DeviceKind, usize); 5] = [
            (DeviceKind::Cuda, 2),
            (DeviceKind::DirectML, 2),
            (DeviceKind::Metal, 2),
            (DeviceKind::WebGpu, 2),
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
        // appends a file name with a backslash — the case that surfaced as
        // ort's misleading "system error 13 (permission denied)".
        let input = "./data/nli_cache\\model_quantized.onnx";
        assert_eq!(
            normalize_model_path(input),
            "./data/nli_cache/model_quantized.onnx"
        );
    }

    #[test]
    fn normalize_model_path_replaces_all_backslashes() {
        // A fully Windows-native path (e.g. when `cache_dir` is configured as
        // an absolute Windows path) — every separator must flip.
        let input = ".\\data\\nli_cache\\model_quantized.onnx";
        assert_eq!(
            normalize_model_path(input),
            "./data/nli_cache/model_quantized.onnx"
        );
    }

    #[test]
    fn normalize_model_path_is_idempotent_on_forward_slashes() {
        // A path that is already ort-safe must pass through untouched so
        // the debug log guard in `build_session` does not spam on Linux/macOS
        // or on a `cache_dir` that was already forward-slashed.
        let input = "./data/nli_cache/model_quantized.onnx";
        assert_eq!(normalize_model_path(input), input);
    }

    #[test]
    fn normalize_model_path_flips_drive_letter_prefixed_path() {
        // A backslash inside a Windows drive-letter path is still a
        // separator and must flip — the drive token (`C:`) is left intact
        // because only the separator character is replaced. ort accepts
        // forward-slash drive paths on Windows.
        assert_eq!(
            normalize_model_path("C:\\Users\\me\\.cache\\model_quantized.onnx"),
            "C:/Users/me/.cache/model_quantized.onnx"
        );
    }
}
