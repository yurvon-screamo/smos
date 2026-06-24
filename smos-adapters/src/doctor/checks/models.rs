//! Local model-cache existence checks for the doctor.
//!
//! Pure filesystem probes — NO downloads, NO network. The doctor must never
//! block on a model fetch (that is `smos init` / `smos serve`'s job); it
//! only reports whether each model the configured services reference is
//! already on disk at the expected path. Missing models surface as FAIL
//! rows with a remediation hint that points the operator at the right
//! subcommand (`smos init` for the GGUF trio, `smos serve` for the
//! lazily-fetched DeBERTa ONNX + ORT DLL).
//!
//! Path resolution honours `~` expansion via [`crate::paths::expand_tilde`]
//! so the same config string works regardless of the operator's OS. Device
//! detection reuses [`crate::nli::device::detect_device`] — no probes of
//! its own, the doctor just reports what `auto`/`cpu`/… resolved to.

use super::super::types::CheckResult;
use crate::config::SmosConfig;
use crate::nli::device::{DeviceKind, detect_device};
use crate::paths::expand_tilde;

/// Check existence of every model the configured services reference:
/// - three GGUF weights (`[llama_cpp].{extraction,embedding,reranker}`),
/// - the lazily-fetched DeBERTa ONNX file under `[nli_backend].cache_dir`,
/// - the device-specific ONNX Runtime shared library under
///   `[nli_backend].ort_cache_dir`.
///
/// Each entry resolves to one PASS or FAIL row; FAIL rows carry a
/// remediation hint. Returns the concatenation so the orchestrator can
/// `extend` the report without re-classifying.
pub fn check_models(config: &SmosConfig) -> Vec<CheckResult> {
    let mut results = Vec::new();
    results.extend(check_gguf_models(config));
    results.extend(check_nli_onnx_model(config));
    results.extend(check_ort_runtime(config));
    results
}

/// Probe each of the three GGUF model paths declared under `[llama_cpp]`.
fn check_gguf_models(config: &SmosConfig) -> Vec<CheckResult> {
    let services: [(&str, &crate::llama_server::LlamaCppServiceConfig); 3] = [
        ("extraction", &config.llama_cpp.extraction),
        ("embedding", &config.llama_cpp.embedding),
        ("reranker", &config.llama_cpp.reranker),
    ];

    services
        .iter()
        .map(|(name, service)| check_gguf_service(name, service))
        .collect()
}

fn check_gguf_service(
    name: &str,
    service: &crate::llama_server::LlamaCppServiceConfig,
) -> CheckResult {
    let path = expand_tilde(&service.model_path);
    if path.exists() {
        CheckResult::pass(format!("GGUF model ({name})"), path.display().to_string())
    } else {
        CheckResult::fail(
            format!("GGUF model ({name})"),
            format!("not found: {}", path.display()),
        )
        .with_recommendation("run `smos init` to download")
    }
}

/// Probe the DeBERTa ONNX cache. The native backend writes
/// `model_quantized.onnx` next to the tokenizer under `cache_dir`.
fn check_nli_onnx_model(config: &SmosConfig) -> Vec<CheckResult> {
    let model_path = expand_tilde(&config.nli_backend.cache_dir).join("model_quantized.onnx");
    if model_path.exists() {
        vec![CheckResult::pass(
            "NLI model (DeBERTa ONNX)",
            model_path.display().to_string(),
        )]
    } else {
        vec![
            CheckResult::fail("NLI model (DeBERTa ONNX)", "not cached".to_string())
                .with_recommendation("run `smos serve` to download (~643MB)"),
        ]
    }
}

/// Probe the device-specific ONNX Runtime shared library cache.
///
/// Reuses the same `device + cache_subdir + dll leaf name` derivation the
/// ORT downloader uses (`nli::ort_cache`). A device is "ready" iff BOTH
/// the shared library AND the `.{}-complete` marker file exist — the
/// marker is the very last write the downloader performs, so its presence
/// is the proof the directory reached its final state.
fn check_ort_runtime(config: &SmosConfig) -> Vec<CheckResult> {
    let device = detect_device(&config.nli_backend.device);
    let ort_dir = expand_tilde(&config.nli_backend.ort_cache_dir).join(device.as_str());

    let Some(dll_name) = ort_dll_filename(device) else {
        return vec![
            CheckResult::warn(
                "ORT runtime",
                format!("no canonical DLL name for device {}", device.as_str()),
            )
            .with_recommendation(
                "set [nli_backend].device to one of cpu / directml / cuda / metal",
            ),
        ];
    };

    let dll_path = ort_dir.join(dll_name);
    let marker = ort_dir.join(format!(".{dll_name}-complete"));
    if dll_path.exists() && marker.exists() {
        vec![CheckResult::pass(
            "ORT runtime",
            format!("{} ({})", dll_path.display(), device.as_str()),
        )]
    } else {
        vec![
            CheckResult::fail("ORT runtime", format!("not cached for {}", device.as_str()))
                .with_recommendation("will auto-download on first `smos serve`"),
        ]
    }
}

/// Canonical shared-library leaf name ort expects for `device` on the
/// current platform. Mirrors [`crate::nli::ort_cache`]'s table so the
/// doctor probes the same file the downloader fetches. Returns `None` for
/// device/platform pairs upstream ORT does not publish artifacts for.
fn ort_dll_filename(device: DeviceKind) -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH, device) {
        ("windows", "x86_64", DeviceKind::Cpu)
        | ("windows", "x86_64", DeviceKind::DirectML)
        | ("windows", "x86_64", DeviceKind::Cuda) => Some("onnxruntime.dll"),
        ("linux", "x86_64", DeviceKind::Cpu) | ("linux", "x86_64", DeviceKind::Cuda) => {
            Some("libonnxruntime.so")
        }
        ("macos", "aarch64", DeviceKind::Cpu) | ("macos", "aarch64", DeviceKind::Metal) => {
            Some("libonnxruntime.dylib")
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ort_dll_filename_returns_canonical_leaf_on_supported_pairs() {
        // The dev/CI host MUST be a supported pair; assert that SOME leaf
        // name comes back so a future platform addition does not silently
        // regress the doctor probe.
        let device = detect_device("cpu");
        let leaf = ort_dll_filename(device);
        assert!(
            leaf.is_some(),
            "expected a canonical ORT DLL leaf name for cpu on {}, got None",
            std::env::consts::OS
        );
        let leaf = leaf.unwrap();
        assert!(
            leaf.ends_with(".dll") || leaf.ends_with(".so") || leaf.ends_with(".dylib"),
            "unexpected DLL leaf: {leaf}"
        );
    }

    #[test]
    fn ort_dll_filename_rejects_cuda_on_macos() {
        // Cross-platform check independent of the dev host: the lookup must
        // refuse CUDA on macOS (upstream ORT does not publish it).
        assert!(ort_dll_filename(DeviceKind::Cuda).is_some() || cfg!(not(target_os = "macos")));
        if cfg!(target_os = "macos") {
            assert_eq!(ort_dll_filename(DeviceKind::Cuda), None);
        }
    }

    #[test]
    fn check_models_with_empty_config_emits_fail_rows_with_hints() {
        // A blank LlamaCppConfig defaults model_path to "" — expand_tilde
        // leaves it relative, the file does not exist, the doctor must
        // surface FAIL with the `smos init` recommendation.
        let cfg = SmosConfig {
            llama_cpp: crate::llama_server::LlamaCppConfig {
                extraction: crate::llama_server::LlamaCppServiceConfig {
                    model_path: "/nonexistent/extraction.gguf".into(),
                    ..Default::default()
                },
                embedding: crate::llama_server::LlamaCppServiceConfig {
                    model_path: "/nonexistent/embedding.gguf".into(),
                    ..Default::default()
                },
                reranker: crate::llama_server::LlamaCppServiceConfig {
                    model_path: "/nonexistent/reranker.gguf".into(),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..SmosConfig::default()
        };
        let rows = check_models(&cfg);
        // 3 GGUF + 1 NLI ONNX + 1 ORT runtime.
        assert!(
            rows.len() >= 5,
            "expected >= 5 model rows, got {}",
            rows.len()
        );
        let any_fail_with_hint = rows
            .iter()
            .any(|r| r.status.is_fail() && r.recommendation.is_some());
        assert!(
            any_fail_with_hint,
            "at least one FAIL row must carry a remediation hint"
        );
    }
}
