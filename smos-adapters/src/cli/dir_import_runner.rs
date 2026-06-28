//! `smos import-dir <path>` — bulk import facts from a directory tree.
//!
//! Recursively scans `path` for supported documents (`*.md`, `*.txt`,
//! `*.json`, `*.jsonl`, `*.yaml`, `*.yml`, `*.toml`), lifts the textual
//! content of each file (raw for prose formats; JSON-string extraction for
//! JSON / JSONL), and re-runs the same `ExtractFactsFromResponse`
//! pipeline the live proxy runs after each chat completion. After every
//! file has been processed, the runner optionally triggers a single
//! `FinalizeSession` drain via [`run_finalize`] so the operator does
//! not have to invoke it manually.
//!
//! Re-using the live extraction path keeps this command DRY with
//! `smos import` and `smos serve`: dedup, embedding, cross-session
//! confirmation, and the `MIN_INPUT_CHARS` floor all apply identically.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::SurrealStore;
use crate::cli::dir_scanner::{read_file_content, scan_directory};
use crate::cli::finalize_runner::run_finalize;
use crate::cli::import_helpers::{derive_session_id, parse_memory_key};
use crate::cli::tracing_setup::init_tracing_default;
use crate::config::SmosConfig;
use crate::{OllamaEmbedding, OllamaExtractor, SystemClock, TokioDelay};
use smos_application::ports::SessionRepository;
use smos_application::use_cases::ExtractFactsFromResponse;
use smos_domain::{MemoryKey, SessionId};

/// Parsed `smos import-dir` invocation. The `smos` binary's clap parser
/// constructs this struct so the runner stays clap-free.
pub struct ImportDirArgs {
    pub path: String,
    pub memory_key: String,
    pub limit: Option<usize>,
    pub no_finalize: bool,
}

/// Per-run aggregate surfaced to the operator at the end of the import.
#[derive(Debug, Default, Clone)]
struct DirImportStats {
    files_processed: usize,
    files_skipped: usize,
    total_facts: usize,
}

/// Entry point: install tracing, load config, scan + process files,
/// optionally finalize.
pub async fn run_dir_import(config_path: &str, args: ImportDirArgs) -> Result<()> {
    init_tracing_default();
    let config = SmosConfig::load(config_path)?;

    let dir = Path::new(&args.path);
    anyhow::ensure!(dir.is_dir(), "not a directory: {}", args.path);

    let mut files = scan_directory(dir);
    println!("Found {} supported files in {}", files.len(), args.path);
    if files.is_empty() {
        return Ok(());
    }
    if let Some(limit) = args.limit {
        files.truncate(limit);
    }

    let store = SurrealStore::connect(
        &config.surreal.path,
        &config.surreal.namespace,
        &config.surreal.database,
    )
    .await?;
    store.run_migrations().await?;

    let memory_key = parse_memory_key(&args.memory_key)?;
    let session_id = derive_session_id("dir-import");
    store.get_or_create(&session_id, &memory_key).await?;

    let stats = process_files(&files, &store, &config, &memory_key, &session_id).await?;

    println!("\n=== Import complete ===");
    println!("Directory:     {}", args.path);
    println!("Memory key:    {}", memory_key);
    println!("Session:       {}", session_id);
    println!("Files processed: {}", stats.files_processed);
    println!("Files skipped:   {}", stats.files_skipped);
    println!("New facts:       {}", stats.total_facts);

    if !args.no_finalize && stats.total_facts > 0 {
        println!("\n=== Finalizing ===");
        run_finalize(config_path, session_id.as_str(), Some(memory_key.as_str())).await?;
    }
    Ok(())
}

/// Iterate the scanned files, build the extraction use case per file, and
/// feed each file's content through it. Per-file failures abort the run
/// with full context — the underlying extractor already retries 3× with
/// exponential backoff, so a hard failure here means the model is
/// genuinely unreachable and continuing would log N copies of the same
/// error.
async fn process_files(
    files: &[std::path::PathBuf],
    store: &SurrealStore,
    config: &SmosConfig,
    memory_key: &MemoryKey,
    session_id: &SessionId,
) -> Result<DirImportStats> {
    let embedder = OllamaEmbedding::new(Arc::new(config.embedding.clone()))?;
    let extractor = OllamaExtractor::new(Arc::new(config.llm_extraction.clone()))?;
    let clock = SystemClock;
    let delay = TokioDelay;

    let mut stats = DirImportStats::default();
    let total = files.len();

    for (idx, file_path) in files.iter().enumerate() {
        // Progress indicator: flush BEFORE the long extraction call so the
        // operator sees "[i/N] file ... " while the model is working,
        // not only after it returns. `print!` (no newline) leaves the
        // line-buffer pending until the next `\n`, so an explicit flush
        // is required to surface the prefix in real time.
        print!("[{}/{}] {} ... ", idx + 1, total, file_path.display());
        let _ = std::io::stdout().flush();

        match read_file_content(file_path) {
            Ok(Some(content)) => {
                let use_case = ExtractFactsFromResponse {
                    facts: store,
                    sessions: store,
                    embedder: &embedder,
                    extractor: &extractor,
                    clock: &clock,
                    delay: &delay,
                    confidence_cfg: &config.confidence,
                    extraction_cfg: &config.extraction,
                    // Hardcoded `true` (NOT read from
                    // `config.server.enable_response_extraction`) on
                    // purpose: that kill-switch gates the BACKGROUND
                    // extraction in the live proxy so operators can
                    // disable it without redeploying. `smos import-dir`
                    // is an explicit, operator-initiated import — the
                    // whole point of the command is to run extraction.
                    enable_response_extraction: true,
                };
                let count = use_case
                    .execute("", &content, &[], memory_key, session_id)
                    .await
                    .with_context(|| format!("extraction failed for {}", file_path.display()))?;
                stats.files_processed += 1;
                stats.total_facts += count;
                println!("{count} facts");
            }
            Ok(None) => {
                println!("SKIP (no extractable content)");
                stats.files_skipped += 1;
            }
            Err(e) => {
                println!("SKIP (read error: {e})");
                stats.files_skipped += 1;
            }
        }
    }
    Ok(stats)
}
