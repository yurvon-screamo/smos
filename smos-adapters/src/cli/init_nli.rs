//! DeBERTa-v3 NLI model download for `smos init`.
//!
//! Mirrors [`super::init_models`] but for the native NLI backend: ensures
//! the quantised DeBERTa-v3 ONNX graph (`model_quantized.onnx`, ~643 MB)
//! and its `tokenizer.json` are BOTH present under
//! `config.nli_backend.cache_dir`. Pre-downloading on `init` keeps the
//! first `smos serve` / Windows service start from paying the 643 MB HF
//! Hub fetch on the hot path (and, on Windows, from Session 0 where the
//! service account has no interactive download path).
//!
//! Re-uses the same atomic `.part`-claim download path
//! ([`crate::nli::model_cache`]) that the lazy first-use code in
//! `NativeNliClassifier::build_blocking` runs, so the on-disk layout and
//! the race-resolution semantics are identical — `init` simply triggers
//! the download earlier.

use std::path::Path;

use crate::config::SmosConfig;
use crate::nli::model_cache::{
    MODEL_FILENAME, TOKENIZER_FILENAME, ensure_model_cached, ensure_tokenizer_cached,
};

/// Ensure BOTH the DeBERTa-v3 ONNX graph AND its tokenizer are cached
/// under `config.nli_backend.cache_dir`. Prints one row, mirroring
/// [`super::init_models::download_gguf_models`]; never aborts the wizard
/// — the lazy first-use path in `NativeNliClassifier::build_blocking`
/// retries on the next `smos serve` / `smos audit`.
pub(super) fn download_nli_model(config: &SmosConfig) {
    let model_id = &config.nli_backend.model;
    let cache_dir = Path::new(&config.nli_backend.cache_dir);

    if both_artifacts_present(cache_dir) {
        println!("  ✓ DeBERTa-v3 NLI ({model_id}) — already present");
        return;
    }

    println!("  ⬇ Downloading DeBERTa-v3 NLI model (~643 MB)...");
    if let Err(e) = ensure_both_artifacts(model_id, cache_dir) {
        println!("  ✗ DeBERTa-v3 NLI ({model_id}) — {e}");
        println!("    The model will be re-fetched on first `smos serve` / `smos audit`.");
        return;
    }
    println!("  ✓ DeBERTa-v3 NLI ({model_id}) — downloaded");
}

/// Both the model graph and its tokenizer are on disk. The AND (not just
/// MODEL) matters: a partial cache (model present, tokenizer missing
/// after a mid-download crash) MUST fall through to the download path so
/// re-running `init` actually repairs it, keeping the wizard's
/// "re-running init retries the failed downloads" guarantee intact.
fn both_artifacts_present(cache_dir: &Path) -> bool {
    cache_dir.join(MODEL_FILENAME).exists() && cache_dir.join(TOKENIZER_FILENAME).exists()
}

fn ensure_both_artifacts(model_id: &str, cache_dir: &Path) -> Result<(), String> {
    ensure_model_cached(model_id, cache_dir).map_err(|e| e.to_string())?;
    ensure_tokenizer_cached(model_id, cache_dir).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Acquire the workspace-wide env-test lock —
    /// `SmosConfig::default()` resolves NLI paths through `SmosPaths`,
    /// which reads `SMOS_HOME`. Sibling `init_models` takes the same
    /// lock for the same reason.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_env_lock::lock()
    }

    /// Both artifacts present → cached short-circuit. No network.
    #[test]
    fn both_artifacts_present_returns_true_when_model_and_tokenizer_exist() {
        let _g = lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let cache_dir = tmp.path();
        std::fs::write(cache_dir.join(MODEL_FILENAME), b"fake-onnx").expect("seed model");
        std::fs::write(cache_dir.join(TOKENIZER_FILENAME), b"fake-tok").expect("seed tokenizer");
        assert!(both_artifacts_present(cache_dir));
    }

    /// Partial cache (model only) → NOT cached, must fall through to
    /// the download path. Pinning the AND guards against a regression
    /// that silently reports success on a missing tokenizer.
    #[test]
    fn both_artifacts_present_returns_false_when_only_model_exists() {
        let _g = lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let cache_dir = tmp.path();
        std::fs::write(cache_dir.join(MODEL_FILENAME), b"fake-onnx").expect("seed model only");
        assert!(!both_artifacts_present(cache_dir));
    }

    #[test]
    fn both_artifacts_present_returns_false_when_neither_exists() {
        let _g = lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        assert!(!both_artifacts_present(tmp.path()));
    }
}
