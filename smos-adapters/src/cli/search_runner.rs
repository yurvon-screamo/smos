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
//! # Execution modes (forwarding)
//!
//! When `smos serve` is running and holds the RocksDB lock, the runner can
//! forward the request to `/v1/cli/search` on the service's HTTP API. The
//! server-side handler invokes the SAME [`RetrieveFacts`] use case the local
//! branch invokes, and renders the result through the SAME [`render_json`]
//! function. The CLI's remote branch transports the response body verbatim
//! to stdout — no deserialisation, no re-render — so the BEAM harness cannot
//! observe any difference between the two paths.
//!
//! # Failure modes
//!
//! Unlike [`EnrichRequest`], search is NOT on the hot proxy path, so it is
//! fail-CLOSED for every provider: an embedder `None`/`Err`, a vector-search
//! `Err`, or a reranker `Err`/empty result aborts with a non-zero exit.

use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Value, json};

use crate::cli::forwarding::{
    ExecMode, announce_forward, emit_lock_recovery_message, is_lock_error,
};
use crate::cli::import_helpers::parse_memory_key;
use crate::cli::tracing_setup::init_tracing_to_stderr;
use crate::config::SmosConfig;
use crate::{LlamaCppReranker, OllamaEmbedding, SurrealStore, SystemClock};
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
/// this struct so the runner stays clap-free.
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

/// Wire shape for `POST /v1/cli/search`. The CLI remote branch serialises
/// this and the server handler deserialises it; the server's response body
/// is the already-rendered stdout text (see [`render_json`]).
#[derive(Debug, Serialize)]
pub struct SearchRequest {
    pub query: String,
    pub memory_key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
}

impl SearchRequest {
    /// Build the wire body from the parsed CLI args. The `memory_key` is
    /// forwarded verbatim (raw string); the server validates it through
    /// [`parse_memory_key`] — same helper the local branch uses — so both
    /// paths surface a malformed key identically.
    pub fn from_args(args: &SearchArgs) -> Self {
        Self {
            query: args.query.clone(),
            memory_key: args.memory_key.clone(),
            top_k: args.top_k,
        }
    }
}

/// Entry point: install tracing (stderr), load config, resolve the execution
/// mode, and either run [`RetrieveFacts`] locally or forward to
/// `/v1/cli/search`. The render function is invoked ONCE per request — on
/// the side that owns the typed result (local-branch here, server handler
/// for remote). The remote branch streams the response body verbatim.
pub async fn run_search(config_path: &str, args: SearchArgs, mode: ExecMode) -> Result<()> {
    init_tracing_to_stderr();
    let config = SmosConfig::load(config_path)?;

    match mode {
        ExecMode::Local => match execute_local(&config, &args).await {
            Ok(scored) => {
                match args.format {
                    OutputFormat::Json => println!("{}", render_json(&scored)),
                }
                Ok(())
            }
            Err(error) => {
                if is_lock_error(&error) {
                    emit_lock_recovery_message();
                }
                Err(error)
            }
        },
        ExecMode::Remote { client, base_url } => {
            announce_forward("search", &base_url);
            match execute_remote(&client, &base_url, &args).await? {
                RemoteOutcome::Body(body) => {
                    std::io::Write::write_all(&mut std::io::stdout(), &body)
                        .context("write search result to stdout")?;
                    // The server-rendered body has no trailing newline; the
                    // local branch's `println!` always appends one. Add it
                    // here so local and remote paths produce byte-equal stdout.
                    println!();
                    Ok(())
                }
                RemoteOutcome::EndpointNotFound => {
                    eprintln!(
                        "smos: server does not expose /v1/cli/search (older version?); \
                         falling back to local execution."
                    );
                    let scored = execute_local(&config, &args).await?;
                    match args.format {
                        OutputFormat::Json => println!("{}", render_json(&scored)),
                    }
                    Ok(())
                }
            }
        }
    }
}

/// Local execution: load adapters, build the use case, run it, return the
/// typed result. The caller renders.
async fn execute_local(config: &SmosConfig, args: &SearchArgs) -> Result<Vec<ScoredSearchHit>> {
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

    use_case
        .execute(&args.query, &memory_key, args.top_k)
        .await
        .map_err(anyhow::Error::from)
}

/// Remote execution: POST the request body to `/v1/cli/search` and return
/// the raw response bytes (already rendered as the final stdout document by
/// the server). The CLI does NOT parse the JSON; the body is piped verbatim
/// to stdout.
///
/// Returns:
/// - `Ok(RemoteOutcome::Body(bytes))` on HTTP 200.
/// - `Ok(RemoteOutcome::EndpointNotFound)` on HTTP 404 — the running server
///   predates the `/v1/cli/*` namespace; the caller falls back to local
///   execution with a stderr notice.
/// - `Err(...)` on transport failure or any non-2xx/non-404 response — the
///   caller surfaces the error, since retrying locally against a server that
///   has already begun mutating state is unsafe.
async fn execute_remote(
    client: &reqwest::Client,
    base_url: &str,
    args: &SearchArgs,
) -> Result<RemoteOutcome> {
    let url = format!("{base_url}/v1/cli/search");
    let body =
        serde_json::to_vec(&SearchRequest::from_args(args)).context("serialise search request")?;
    let response = client
        .post(&url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .with_context(|| format!("forward search to {url}"))?;
    let status = response.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Ok(RemoteOutcome::EndpointNotFound);
    }
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("search forwarding failed: HTTP {status}: {text}");
    }
    let bytes = response
        .bytes()
        .await
        .context("read forwarded search response body")?;
    Ok(RemoteOutcome::Body(bytes))
}

/// Distinguish the `RemoteOutcome` so the caller can react to a missing
/// endpoint (old server version) without conflating it with a transport
/// failure.
enum RemoteOutcome {
    /// Server-rendered stdout document.
    Body(bytes::Bytes),
    /// HTTP 404 — the server is up but does not expose `/v1/cli/search`.
    EndpointNotFound,
}

/// Render the scored hits as the BEAM-compatible JSON array. `score` is the
/// cross-encoder relevance score (higher = more relevant) — it is NOT the
/// vector cosine `distance` (lower = more similar) and must not be conflated
/// with it. Field names follow the mem0 `search` result shape so the BEAM
/// harness can consume the array verbatim.
///
/// Shared between the CLI local branch, the `/v1/cli/search` server handler,
/// and the integration test that pins their byte-parity. `pub` because the
/// integration test lives outside the crate.
pub fn render_json(scored: &[ScoredSearchHit]) -> String {
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
        let score0 = arr[0]["score"].as_f64().unwrap_or(f64::NAN);
        assert!((score0 - 0.95).abs() < 1e-5, "got {score0}");
        assert_eq!(arr[0]["created_at"], "2025-06-18T12:00:00Z");
        assert_eq!(arr[0]["status"], "accepted");
        assert_eq!(arr[0]["memory_key"], "origa");
        let score1 = arr[1]["score"].as_f64().unwrap_or(f64::NAN);
        assert!(
            score0 > score1,
            "rerank order (descending score) is preserved"
        );
        assert_eq!(arr[1]["created_at"], Value::Null);
    }

    #[test]
    fn output_format_parses_json_only() {
        assert_eq!(OutputFormat::parse("json").unwrap(), OutputFormat::Json);
        assert!(OutputFormat::parse("yaml").is_err());
        assert!(OutputFormat::parse("").is_err());
    }

    #[test]
    fn search_request_from_args_preserves_fields() {
        let args = SearchArgs {
            query: "explain rust".into(),
            memory_key: "bob".into(),
            top_k: Some(7),
            format: OutputFormat::Json,
        };
        let req = SearchRequest::from_args(&args);
        assert_eq!(req.query, "explain rust");
        assert_eq!(req.memory_key, "bob");
        assert_eq!(req.top_k, Some(7));
    }

    #[test]
    fn search_request_from_args_omits_top_k_when_none() {
        let args = SearchArgs {
            query: "q".into(),
            memory_key: "bob".into(),
            top_k: None,
            format: OutputFormat::Json,
        };
        let req = SearchRequest::from_args(&args);
        let v = serde_json::to_value(&req).unwrap();
        assert!(v.get("top_k").is_none() || v["top_k"].is_null());
    }

    #[tokio::test]
    async fn execute_remote_maps_404_to_endpoint_not_found() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/cli/search"))
            .respond_with(wiremock::ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let args = SearchArgs {
            query: "x".into(),
            memory_key: "origa".into(),
            top_k: None,
            format: OutputFormat::Json,
        };
        let outcome = execute_remote(&client, &server.uri(), &args)
            .await
            .expect("execute_remote should not hard-fail on 404");
        assert!(
            matches!(outcome, RemoteOutcome::EndpointNotFound),
            "404 must map to EndpointNotFound"
        );
    }
}
