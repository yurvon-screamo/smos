//! `smos search "<query>"` — read-only retrieval (BEAM benchmark contract).
//!
//! Returns the reranked accepted facts for a query as a single JSON array on
//! stdout, with no upstream LLM call and no state mutation: no session dedup,
//! no heat boost on write, no message injection. The ranking is identical to
//! the live [`EnrichRequest`] pipeline's pre-dedup survivors because both
//! share [`smos_application::helpers::retrieval_pipeline`].
//!
//! # Preconditions (documented in `--help`)
//!
//! - a `llama-server` embedding endpoint (`[embedding]`) AND a reranker
//!   endpoint (`[reranker]`) are reachable,
//! - the DB already contains **Accepted** facts (run `smos import raw` then
//!   `smos finalize` to populate it).
//!
//! # Concurrency
//!
//! [`SurrealStore::connect`] takes a process-exclusive RocksDB lock. Two
//! concurrent `smos` subprocesses pointing at the same `[surreal].path` will
//! contend; the BEAM harness `run.py` is a sequential for-loop, so call this
//! command strictly one invocation at a time per database path.
//!
//! # Failure modes
//!
//! Unlike [`EnrichRequest`], search is NOT on the hot proxy path, so it is
//! fail-CLOSED for every provider: an embedder `None`/`Err`, a vector-search
//! `Err`, or a reranker `Err`/empty result aborts with a non-zero exit. Silent
//! empty output would mislead the benchmark into scoring SMOS as "remembers
//! nothing". Only the genuine "nothing matched" outcomes (short/empty query,
//! zero vector hits, zero post-filter survivors) print `[]` and exit 0.

use std::sync::Arc;

use anyhow::Result;
use serde_json::{Value, json};

use crate::SurrealStore;
use crate::cli::import_helpers::parse_memory_key;
use crate::cli::tracing_setup::init_tracing_to_stderr;
use crate::config::SmosConfig;
use crate::{LlamaCppReranker, OllamaEmbedding, SystemClock};
use smos_application::use_cases::{RetrieveFacts, ScoredSearchHit};

/// Output format for `smos search`. Only JSON is supported today; the flag is
/// reserved so future formats (table, plain text) can be added without a CLI
/// contract change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Json,
}

impl OutputFormat {
    /// Parse the `--format` value. Anything other than `json` is rejected so a
    /// typo surfaces immediately rather than silently emitting the default.
    pub fn parse(raw: &str) -> Result<Self> {
        match raw {
            "json" => Ok(Self::Json),
            other => Err(anyhow::anyhow!(
                "unsupported --format {other:?}: only 'json' is supported"
            )),
        }
    }
}

/// Parsed `smos search` invocation. The `smos` binary's clap parser constructs
/// this struct so the runner stays clap-free (mirrors [`RawImportArgs`]).
///
/// [`RawImportArgs`]: crate::cli::raw_import_runner::RawImportArgs
pub struct SearchArgs {
    /// The already-resolved query text (positional arg or stdin). May be empty
    /// / whitespace-only — the use case returns an empty array in that case.
    pub query: String,
    /// Raw `--person` value; validated into a [`smos_domain::MemoryKey`] by
    /// [`parse_memory_key`] inside the runner.
    pub memory_key: String,
    /// `--top-k` override for `retrieval.top_k_final`. `None` keeps the
    /// configured default.
    pub top_k: Option<usize>,
    /// `--format` (reserved). Only [`OutputFormat::Json`] today.
    pub format: OutputFormat,
}

/// Entry point: install tracing (to stderr — stdout is reserved for the JSON
/// document), load config, connect to the store, run the read-only retrieval
/// pipeline once, and print the JSON array to stdout.
pub async fn run_search(config_path: &str, args: SearchArgs) -> Result<()> {
    init_tracing_to_stderr();
    let config = SmosConfig::load(config_path)?;

    let store = SurrealStore::connect(
        &config.surreal.path,
        &config.surreal.namespace,
        &config.surreal.database,
    )
    .await?;
    store.run_migrations().await?;

    let memory_key = parse_memory_key(&args.memory_key)?;
    let embedder = OllamaEmbedding::new(Arc::new(config.embedding.clone()))?;
    let reranker = LlamaCppReranker::new(Arc::new(config.reranker.clone()))?;
    let clock = SystemClock;

    let use_case = RetrieveFacts {
        facts: &store,
        embedder: &embedder,
        reranker: &reranker,
        clock: &clock,
        retrieval_cfg: &config.retrieval,
        heat_cfg: &config.heat,
    };

    let scored = use_case
        .execute(&args.query, &memory_key, args.top_k)
        .await?;

    match args.format {
        OutputFormat::Json => println!("{}", render_json(&scored)),
    }
    Ok(())
}

/// Render the scored hits as the BEAM-compatible JSON array. `score` is the
/// cross-encoder relevance score (higher = more relevant) — it is NOT the
/// vector cosine `distance` (lower = more similar) and must not be conflated
/// with it. Field names follow the mem0 `search` result shape so the BEAM
/// harness can consume the array verbatim.
fn render_json(scored: &[ScoredSearchHit]) -> String {
    let array: Vec<Value> = scored
        .iter()
        .map(|s| {
            json!({
                "id": s.hit.id.as_str(),
                "memory": s.hit.document,
                "score": s.score,
                "created_at": s.hit.metadata.created_at,
                "confidence": s.hit.metadata.confidence,
                "status": s.hit.metadata.status,
                "valid_until": s.hit.metadata.valid_until,
                "conflicts_with": s.hit.metadata.conflicts_with,
                "memory_key": s.hit.memory_key.as_str(),
            })
        })
        .collect();
    serde_json::to_string(&Value::Array(array)).unwrap_or_else(|_| "[]".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use smos_application::types::{SearchHit, SearchHitMetadata};
    use smos_domain::{FactId, MemoryKey};

    fn scored(document: &str, score: f32, created_at: Option<&str>) -> ScoredSearchHit {
        ScoredSearchHit {
            hit: SearchHit {
                id: FactId::from_raw("fact_0123456789abcdef").unwrap(),
                document: document.into(),
                memory_key: MemoryKey::from_raw("origa").unwrap(),
                metadata: SearchHitMetadata {
                    status: "accepted".into(),
                    confidence: 0.85,
                    valid_until: None,
                    heat_base: 1.0,
                    last_access_at: 1_700_000_000.0,
                    distance: Some(0.1),
                    created_at: created_at.map(str::to_string),
                    conflicts_with: vec!["fact_deadbeefdeadbee".into()],
                },
            },
            score,
        }
    }

    #[test]
    fn render_json_empty_array_when_no_hits() {
        let out = render_json(&[]);
        assert_eq!(out, "[]");
    }

    #[test]
    fn render_json_emits_score_memory_and_created_at() {
        let scored = vec![
            scored("Rust is memory-safe", 0.95, Some("2025-06-18T12:00:00Z")),
            scored("Ownership prevents double-free", 0.80, None),
        ];
        let v: Value = serde_json::from_str(&render_json(&scored)).unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["memory"], "Rust is memory-safe");
        // f32 → f64 widening on serialisation drifts the literal, so compare
        // with tolerance rather than strict equality.
        let score0 = arr[0]["score"].as_f64().unwrap_or(f64::NAN);
        assert!((score0 - 0.95).abs() < 1e-5, "got {score0}");
        assert_eq!(arr[0]["created_at"], "2025-06-18T12:00:00Z");
        assert_eq!(arr[0]["status"], "accepted");
        assert_eq!(arr[0]["memory_key"], "origa");
        // Higher score first (rerank order preserved).
        let score1 = arr[1]["score"].as_f64().unwrap_or(f64::NAN);
        assert!(
            score0 > score1,
            "rerank order (descending score) is preserved"
        );
        // Missing created_at serialises as null, not omitted.
        assert_eq!(arr[1]["created_at"], Value::Null);
    }

    #[test]
    fn output_format_parses_json_only() {
        assert_eq!(OutputFormat::parse("json").unwrap(), OutputFormat::Json);
        assert!(OutputFormat::parse("yaml").is_err());
        assert!(OutputFormat::parse("").is_err());
    }
}
