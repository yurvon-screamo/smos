//! `smos import` — import an opencode session transcript into SMOS memory.

use std::sync::Arc;

use anyhow::{Context, Result};

use crate::cli::import_helpers::{
    apply_offset_limit, derive_session_id, map_discovery_error, parse_memory_key, print_dry_run,
};
use crate::cli::tracing_setup::init_tracing_default;
use crate::config::SmosConfig;
use crate::opencode;
use crate::{OllamaEmbedding, OllamaExtractor, SurrealStore, SystemClock, TokioDelay};
use smos_application::use_cases::ImportOpencodeSession;
use smos_application::use_cases::extract_facts_from_response::MIN_INPUT_CHARS;
use smos_domain::MemoryKey;

/// Parsed `smos import` invocation. The `smos` binary's clap parser
/// constructs this struct so the runner does not depend on clap.
pub struct ImportArgs {
    pub session_id: Option<String>,
    pub from_file: Option<String>,
    pub memory_key: String,
    pub port: Option<u16>,
    pub agents: Vec<String>,
    pub limit: Option<usize>,
    pub offset: usize,
    pub dry_run: bool,
    pub list: bool,
}

/// Entry point: install tracing, load config, dispatch to list/dry-run/import.
pub async fn run_import(config_path: &str, args: ImportArgs) -> Result<()> {
    init_tracing_default();
    let config = SmosConfig::load(config_path)?;

    if args.list {
        return run_list(args.port).await;
    }

    let (session_id_str, transcript) = resolve_transcript(&args).await?;

    let turns = opencode::parse_transcript(&transcript);
    println!("Parsed {} assistant turns", turns.len());

    let windowed = apply_offset_limit(turns, args.offset, args.limit);
    println!("After offset/limit: {} turns to process", windowed.len());

    if args.dry_run {
        print_dry_run(&windowed);
        return Ok(());
    }

    run_import_pipeline(&config, &args, &session_id_str, windowed).await
}

/// Resolve the transcript either from `--from-file` or via discovery.
async fn resolve_transcript(args: &ImportArgs) -> Result<(String, serde_json::Value)> {
    if let Some(path) = &args.from_file {
        let content =
            std::fs::read_to_string(path).with_context(|| format!("read --from-file {}", path))?;
        let value: serde_json::Value =
            serde_json::from_str(&content).with_context(|| format!("parse JSON {}", path))?;
        let id = value
            .get("info")
            .and_then(|i| i.get("id"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("imported")
            .to_string();
        return Ok((id, value));
    }

    let session_id = args
        .session_id
        .as_ref()
        .context("session_id required (or pass --from-file / --list)")?;
    let client = reqwest::Client::new();
    let source = opencode::resolve_source(&client, args.port).await;
    println!("Source: {}", source.kind_str());
    let transcript = opencode::fetch_session_export(&source, &client, session_id)
        .await
        .map_err(map_discovery_error)?;
    Ok((session_id.clone(), transcript))
}

/// Discover sessions via the chosen source and print their ids + titles.
async fn run_list(port: Option<u16>) -> Result<()> {
    let client = reqwest::Client::new();
    let source = opencode::resolve_source(&client, port).await;
    println!("Source: {}", source.kind_str());
    let sessions = opencode::list_sessions(&source, &client)
        .await
        .map_err(map_discovery_error)?;
    if sessions.is_empty() {
        println!("(no sessions found)");
        return Ok(());
    }
    for s in &sessions {
        let id = s
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let title = s
            .get("title")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        println!("{id}\t{title}");
    }
    Ok(())
}

/// Wire the concrete adapters, build the use case, run it, print stats.
async fn run_import_pipeline(
    config: &SmosConfig,
    args: &ImportArgs,
    session_id_str: &str,
    turns: Vec<smos_application::use_cases::import_opencode_session::AssistantTurn>,
) -> Result<()> {
    let store = SurrealStore::connect(
        &config.surreal.path,
        &config.surreal.namespace,
        &config.surreal.database,
    )
    .await?;
    store.run_migrations().await?;

    let embedder = OllamaEmbedding::new(Arc::new(config.embedding.clone()))?;
    let extractor = OllamaExtractor::new(Arc::new(config.llm_extraction.clone()))?;
    let clock = SystemClock;
    let delay = TokioDelay;

    let memory_key: MemoryKey = parse_memory_key(&args.memory_key)?;
    let session_id = derive_session_id(session_id_str);

    let import = ImportOpencodeSession {
        facts: store.clone(),
        sessions: store.clone(),
        embedder,
        extractor,
        clock,
        delay,
        confidence_cfg: Arc::new(config.confidence.clone()),
        extraction_cfg: Arc::new(config.extraction.clone()),
        enable_response_extraction: config.server.enable_response_extraction,
        // Wire from the SAME const as the live response pipeline so the two
        // paths cannot drift apart — single source of truth for the
        // min-input floor.
        min_chars: MIN_INPUT_CHARS,
    };

    let agent_filter = if args.agents.is_empty() {
        None
    } else {
        Some(args.agents.as_slice())
    };

    let stats = import
        .execute(turns, &memory_key, &session_id, agent_filter)
        .await?;

    println!("\n=== Import complete ===");
    println!("Session:      {}", stats.session_id);
    println!("Memory key:   {}", memory_key);
    println!("Processed:    {} turns", stats.turns_processed);
    println!("Skipped:      {} turns", stats.turns_skipped);
    println!("New facts:    {}", stats.facts_extracted);
    Ok(())
}
