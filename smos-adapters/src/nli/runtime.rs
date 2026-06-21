//! Shared runtime wiring for the NLI backend.
//!
//! Both the `serve` and `finalize` subcommands need to translate the
//! configuration into a concrete classifier. Centralising the build here
//! keeps the two callers symmetric and avoids divergent error handling.

use std::path::PathBuf;

use anyhow::Result;

use crate::config::SmosConfig;
use crate::nli::NativeNliClassifier;

/// Build a [`NativeNliClassifier`] from `config.nli`.
///
/// Used by `smos finalize` and `smos serve`. The classifier owns the ort
/// session + tokenizer; constructing it once at startup avoids paying the
/// model-load cost per request.
///
/// The ort session build + tokenizer load are CPU-bound, blocking operations
/// (HF Hub download, file IO, graph optimisation, EP init — 1–5 s on a warm
/// cache, much longer on a cold one). Running them on the tokio runtime's
/// async worker thread would block every other future sharing that worker
/// for the whole build window. `spawn_blocking` lifts the work onto the
/// dedicated blocking-pool where it belongs; the async caller still gets a
/// `NativeNliClassifier` back via `.await`.
pub async fn build_classifier(config: &SmosConfig) -> Result<NativeNliClassifier> {
    let model = config.nli_backend.model.clone();
    let cache_dir = PathBuf::from(&config.nli_backend.cache_dir);
    let classifier =
        tokio::task::spawn_blocking(move || NativeNliClassifier::new(&model, cache_dir))
            .await
            .map_err(|e| {
                anyhow::anyhow!("spawn_blocking join error during classifier build: {e}")
            })??;
    Ok(classifier)
}
