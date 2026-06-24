//! Shared runtime wiring for the NLI backend.
//!
//! Both the `serve` and `finalize` subcommands need to translate the
//! configuration into a concrete classifier. Centralising the build here
//! keeps the two callers symmetric and avoids divergent error handling.

use std::path::PathBuf;

use anyhow::Result;

use crate::config::SmosConfig;
use crate::nli::NativeNliClassifier;

/// Build a [`NativeNliClassifier`] from `config.nli_backend`.
///
/// Used by `smos finalize` and `smos serve`. The classifier owns the ort
/// session + tokenizer; constructing it once at startup avoids paying the
/// model-load cost per request.
///
/// The classifier's [`NativeNliClassifier::new`] is itself async: it
/// awaits the ORT DLL download, then internally dispatches the heavy
/// sync work (HF Hub fetch, ort session commit, tokenizer load) to a
/// blocking-pool thread. The caller therefore does not need to wrap the
/// build in `spawn_blocking`.
pub async fn build_classifier(config: &SmosConfig) -> Result<NativeNliClassifier> {
    let model = config.nli_backend.model.clone();
    let cache_dir = PathBuf::from(&config.nli_backend.cache_dir);
    let ort_cache_dir = PathBuf::from(&config.nli_backend.ort_cache_dir);
    let device_config = config.nli_backend.device.clone();
    NativeNliClassifier::new(&model, cache_dir, device_config, ort_cache_dir)
        .await
        .map_err(anyhow::Error::from)
}
