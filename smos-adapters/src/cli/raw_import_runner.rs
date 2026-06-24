//! `smos import raw <text>` — extract facts from arbitrary free-form text.
//!
//! Mirrors [`super::dir_import_runner`] but for a single text input instead
//! of a directory tree. The same [`ExtractFactsFromResponse`] pipeline runs,
//! so dedup, embedding, cross-session confirmation, and the
//! `MIN_INPUT_CHARS` floor all apply identically. No finalize drain is
//! triggered (the operator can run `smos finalize` by hand for that).

use std::sync::Arc;

use anyhow::Result;

use crate::SurrealStore;
use crate::cli::import_helpers::{derive_session_id, parse_memory_key};
use crate::cli::tracing_setup::init_tracing_default;
use crate::config::SmosConfig;
use crate::{OllamaEmbedding, OllamaExtractor, SystemClock, TokioDelay};
use smos_application::ports::SessionRepository;
use smos_application::use_cases::ExtractFactsFromResponse;

/// Parsed `smos import raw` invocation. The `smos` binary's clap parser
/// constructs this struct so the runner stays clap-free.
pub struct RawImportArgs {
    pub text: String,
    pub memory_key: String,
}

/// Entry point: install tracing, load config, connect to the store, run
/// the extraction pipeline once on `args.text`, and print the new-fact
/// count.
pub async fn run_raw_import(config_path: &str, args: RawImportArgs) -> Result<()> {
    init_tracing_default();
    let config = SmosConfig::load(config_path)?;

    let store = SurrealStore::connect(
        &config.surreal.path,
        &config.surreal.namespace,
        &config.surreal.database,
    )
    .await?;
    store.run_migrations().await?;

    let memory_key = parse_memory_key(&args.memory_key)?;
    let session_id = derive_session_id("raw-import");
    store.get_or_create(&session_id, &memory_key).await?;

    let embedder = OllamaEmbedding::new(Arc::new(config.embedding.clone()))?;
    let extractor = OllamaExtractor::new(Arc::new(config.llm_extraction.clone()))?;
    let clock = SystemClock;
    let delay = TokioDelay;

    let use_case = ExtractFactsFromResponse {
        facts: &store,
        sessions: &store,
        embedder: &embedder,
        extractor: &extractor,
        clock: &clock,
        delay: &delay,
        confidence_cfg: &config.confidence,
        extraction_cfg: &config.extraction,
        // Same rationale as `dir_import_runner`: this command's entire
        // purpose is to run extraction, so the live-proxy kill-switch
        // does not apply.
        enable_response_extraction: true,
    };

    let count = use_case
        .execute(&args.text, &[], &memory_key, &session_id)
        .await?;

    println!("\n=== Raw import complete ===");
    println!("Memory key: {}", memory_key);
    println!("Session:    {}", session_id);
    println!("New facts:  {}", count);
    Ok(())
}
