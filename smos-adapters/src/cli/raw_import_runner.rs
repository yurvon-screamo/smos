//! `smos import raw <text>` — extract facts from arbitrary free-form text.
//!
//! Mirrors [`super::dir_import_runner`] but for a single text input instead
//! of a directory tree. The same [`ExtractFactsFromResponse`] pipeline runs,
//! so dedup, embedding, cross-session confirmation, and the
//! `MIN_INPUT_CHARS` floor all apply identically.
//!
//! After extraction, the runner drains this chunk's pending facts through
//! a single `FinalizeSession` run by default (NLI promotes them to
//! Accepted and detects conflicts against the accumulated Accepted pool
//! from prior chunks). Pass `--no-finalize` to skip the drain.
//!
//! The finalize is built on the SAME `SurrealStore` that ran the
//! extraction (see [`finalize_inline`]) rather than delegating to
//! [`crate::cli::finalize_runner::run_finalize`], which opens its own
//! connection. A second `SurrealStore::connect` to the same RocksDB path
//! would deadlock on RocksDB's single-writer LOCK while this runner's
//! first connection is still held. The standalone `smos finalize` command
//! keeps using `run_finalize` — it is a separate subprocess, so the lock
//! is released before it connects.

use std::sync::Arc;

use anyhow::Result;

use crate::SurrealStore;
use crate::cli::import_helpers::{derive_session_id, parse_memory_key};
use crate::cli::tracing_setup::init_tracing_default;
use crate::config::SmosConfig;
use crate::{OllamaEmbedding, OllamaExtractor, SystemClock, TokioDelay};
use smos_application::ports::SessionRepository;
use smos_application::use_cases::{ExtractFactsFromResponse, FinalizeSession, FinalizeStats};
use smos_domain::{MemoryKey, SessionId};

/// Parsed `smos import raw` invocation. The `smos` binary's clap parser
/// constructs this struct so the runner stays clap-free.
pub struct RawImportArgs {
    pub text: String,
    pub memory_key: String,
    pub no_finalize: bool,
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

    if !args.no_finalize && count > 0 {
        println!("\n=== Finalizing ===");
        finalize_inline(&store, &config, &session_id, &memory_key).await?;
    } else if args.no_finalize {
        println!("(finalize skipped via --no-finalize)");
    }
    Ok(())
}

/// Drain the just-extracted pending facts through `FinalizeSession` using
/// the SAME `store` that ran the extraction. Building the use case here
/// (instead of calling [`crate::cli::finalize_runner::run_finalize`])
/// avoids a second `SurrealStore::connect` to the same RocksDB path,
/// which would deadlock on RocksDB's single-writer LOCK.
async fn finalize_inline(
    store: &SurrealStore,
    config: &SmosConfig,
    session_id: &SessionId,
    memory_key: &MemoryKey,
) -> Result<()> {
    let classifier = crate::nli::build_classifier(config).await?;

    let finalize = FinalizeSession {
        facts: store,
        sessions: store,
        classifier: &classifier,
        confidence_cfg: &config.confidence,
        nli_cfg: &config.nli,
        merge_cfg: &config.merge,
    };

    let stats = finalize.execute(session_id, memory_key).await?;
    print_finalize_stats(&stats);
    Ok(())
}

/// Pretty-printed JSON mirroring
/// [`crate::cli::finalize_runner::print_finalize_report`] so operators can
/// grep the same fields across `smos import raw` and `smos finalize` output.
fn print_finalize_stats(stats: &FinalizeStats) {
    let payload = serde_json::json!({
        "session_id": stats.session_id,
        "processed": stats.processed,
        "finalized": stats.finalized,
        "merged": stats.merged,
        "conflicts": stats.conflicts,
        "rejected": stats.rejected,
    });
    let json = serde_json::to_string_pretty(&payload).unwrap_or_else(|_| format!("{payload:?}"));
    println!("{json}");
}
