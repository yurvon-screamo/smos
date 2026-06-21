//! Shared helpers for the SMOS E2E suites.
//!
//! Each test spins up wiremock upstreams (mock OpenAI server, optionally mock
//! Ollama embeddings + mock reranker), builds an SMOS router pointing at them,
//! and serves SMOS on an ephemeral port inside a spawned task. Tests then hit
//! SMOS with a plain `reqwest` client exactly the way a real OpenAI client
//! would.
//!
//! Passthrough tests don't exercise enrichment, so they reuse [`spawn_smos`]
//! which wires stub providers that short-circuit enrichment (unreachable
//! Ollama/reranker URLs fail-open). Enrichment tests use [`build_state`] /
//! [`serve_state`] to wire real providers against the supplied wiremock URLs
//! and seed facts through the returned `SurrealStore`.

#![allow(dead_code)]

use std::sync::Arc;

use axum::Router;
use serde_json::{Value, json};
use smos_adapters::SystemClock;
use smos_adapters::SystemIdGenerator;
use smos_adapters::config::{ServerConfig, SmosConfig, UpstreamProvider};
use smos_adapters::http::axum_server::{AppState, build_router};
use smos_adapters::upstream::ReqwestUpstreamPool;
use smos_adapters::{LlamaCppReranker, OllamaEmbedding, OllamaExtractor, SurrealStore};
use smos_application::ports::{Clock, FactRepository, IdGenerator};
use smos_domain::{
    Confidence, Embedding, Fact, FactId, FactStatus, MemoryKey, SessionId, Timestamp,
};
use surrealdb::Surreal;
use surrealdb::engine::local::RocksDb;
use tempfile::TempDir;
use wiremock::MockServer;

/// The canonical two-chunk stream the OpenAI shape produces:
/// `Hello` → ` world` (stop) → `[DONE]`. Reused across several streaming tests.
pub const SSE_HELLO_WORLD: &str = "\
data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
\n\
data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\
\n\
data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\
\n\
data: [DONE]\n\n";

/// Build an SMOS config whose upstream `[[upstream.providers]]` pool points
/// at `upstream_base`.
pub fn config_pointing_at(upstream_base: &str) -> SmosConfig {
    let mut config = SmosConfig::default();
    config.upstream.providers = vec![UpstreamProvider {
        name: "test-upstream".into(),
        url: format!("{upstream_base}/v1/chat/completions"),
        api_key: String::new(),
        auth_header: "Authorization".into(),
        timeout_seconds: 5,
    }];
    config.server = ServerConfig::default();
    config
}

/// Spawn SMOS on an ephemeral port against a wiremock `upstream_base` with
/// stub providers (empty SurrealStore, unreachable Ollama/reranker URLs that
/// short-circuit enrichment via fail-open). Used by passthrough tests.
pub async fn spawn_smos(upstream_base: &str) -> String {
    let mut config = config_pointing_at(upstream_base);
    config.llm_extraction.url = "http://127.0.0.1:1".into();
    config.llm_extraction.timeout_seconds = 1;
    config.embedding.url = "http://127.0.0.1:1".into();
    config.embedding.timeout_seconds = 1;
    config.reranker.url = "http://127.0.0.1:1".into();
    config.reranker.timeout_seconds = 1;
    // Passthrough tests do not assert on extraction; disable the pipeline so
    // an unreachable extractor never adds the §12 retry backoff (1 s + 2 s)
    // to every request.
    config.server.enable_response_extraction = false;
    let state = build_state(config).await;
    serve_state(state).await
}

/// Build a full `AppState` from a config. The SurrealDB files live in a
/// tempdir whose ownership is leaked (`std::mem::forget`) so the helper can
/// return just the `Arc<AppState>`.
///
/// # Why the leak is acceptable here
///
/// Each `cargo test` binary runs hundreds of short-lived tests in a single
/// process; the OS reclaims every leaked tempdir when the process exits. The
/// alternative — returning an `Arc<AppState>` together with an `Arc<TempDir>`
/// guard — would force every test to thread the guard through its spawn chain
/// (`tokio::spawn(async move { let _guard = guard; axum::serve(...) })`), which
/// is brittle and produces large amounts of boilerplate for ephemeral test
/// fixtures. The total leaked footprint is bounded by the test count times the
/// empty RocksDB size (~1 MB), so a full suite leaks on the order of tens of
/// MB. CI mitigations: run tests in a fresh process per binary, and use
/// `--test-threads` to cap concurrency.
pub async fn build_state(mut config: SmosConfig) -> Arc<AppState> {
    let tmp = TempDir::new().expect("tempdir");
    let db_path = tmp.path().join("smos.db");
    config.surreal.path = db_path.to_string_lossy().to_string();
    let db = Surreal::new::<RocksDb>(&config.surreal.path)
        .await
        .expect("rocksdb");
    db.use_ns(&config.surreal.namespace)
        .use_db(&config.surreal.database)
        .await
        .expect("use ns/db");
    let store = SurrealStore::from_client(db);
    store.run_migrations().await.expect("migrations");

    let upstream = ReqwestUpstreamPool::new(&config.upstream).expect("upstream pool");
    let embedder = OllamaEmbedding::new(Arc::new(config.embedding.clone())).expect("embedder");
    let reranker = LlamaCppReranker::new(Arc::new(config.reranker.clone())).expect("reranker");
    let extractor =
        OllamaExtractor::new(Arc::new(config.llm_extraction.clone())).expect("extractor");
    let clock: Arc<dyn Clock + Send + Sync> = Arc::new(SystemClock);
    let id_generator: Arc<dyn IdGenerator + Send + Sync> = Arc::new(SystemIdGenerator);
    let retrieval_cfg = Arc::new(config.retrieval.clone());
    let heat_cfg = Arc::new(config.heat.clone());
    let confidence_cfg = Arc::new(config.confidence.clone());
    let extraction_cfg = Arc::new(config.extraction.clone());
    let extraction_supervisor = smos_adapters::runtime::ExtractionSupervisor::new();

    let state = Arc::new(AppState {
        config: Arc::new(config),
        store,
        embedder,
        reranker,
        extractor,
        upstream,
        clock,
        id_generator,
        retrieval_cfg,
        heat_cfg,
        confidence_cfg,
        extraction_cfg,
        extraction_supervisor,
    });
    std::mem::forget(tmp);
    state
}

/// Spawn SMOS with the supplied state on an ephemeral port; return its URL.
pub async fn serve_state(state: Arc<AppState>) -> String {
    let router: Router = build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{addr}")
}

/// A minimal chat-completion request body with `model` and `messages` plus the
/// given extras (e.g. `stream: true`).
pub fn chat_body(model: &str, extras: Vec<(&str, Value)>) -> Value {
    let mut body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
    });
    let obj = body.as_object_mut().expect("object");
    for (k, v) in extras {
        obj.insert(k.into(), v);
    }
    body
}

/// Split a raw SSE byte stream into the `data:` payloads (without the `data: `
/// prefix), preserving order. Used to assert on the frames the client sees.
pub fn sse_payloads(raw: &str) -> Vec<String> {
    raw.split("\n\n")
        .filter_map(|frame| {
            frame
                .lines()
                .find_map(|line| line.strip_prefix("data:"))
                .map(|d| d.trim().to_string())
        })
        .collect()
}

/// Extract the `sess_<hex>` id from a session marker present in `text`.
pub fn session_id_in(text: &str) -> Option<String> {
    let marker = text.split("<!-- smos:").nth(1)?;
    let id = marker.split("-->").next()?.trim();
    Some(id.to_string())
}

// ---------------------------------------------------------------------------
// Enrichment-suite helpers
// ---------------------------------------------------------------------------

/// Canonical memory_key used by enrichment tests (matches the value embedded
/// in fixture facts / session rows).
pub fn enrichment_memory_key() -> MemoryKey {
    MemoryKey::from_raw("origa").expect("memory key")
}

/// Deterministic session id used by enrichment tests so dedup state is
/// predictable across calls.
pub fn fixed_session_id(tag: u8) -> SessionId {
    SessionId::from_raw(&format!("sess_{:012x}", tag as u64)).expect("session id")
}

/// Reference timestamp used as `now` in fixture facts. Uses the wall-clock so
/// the heat post-filter (which compares `last_access_at` against the runtime
/// clock) does not instantly decay seeded facts to zero.
///
/// Routed through `SystemClock` rather than `Timestamp::now_utc()` because
/// the latter is `pub(crate)` in the domain — production callers reach the
/// wall clock through the `Clock` port (the domain itself is IO-free).
pub fn fixed_now() -> Timestamp {
    use smos_application::ports::Clock;
    SystemClock.now()
}

/// Reference embedding dimensionality — pinned to 1024 to match the HNSW
/// index declared in `surreal_schema::FACT_DDL`. Tests that exercise vector
/// search must seed embeddings of this dimensionality.
pub const EMBEDDING_DIM: usize = 1024;

/// Build a unit-norm embedding of `dim` dimensions with `1.0` at `axis`.
pub fn unit_embedding(dim: usize, axis: usize) -> Embedding {
    let mut v = vec![0.0_f32; dim];
    v[axis] = 1.0;
    Embedding::new(v).expect("embedding")
}

/// Build a constant embedding (every dimension set to `value`); used so
/// every seeded fact scores identically against the query embedding.
pub fn constant_embedding(dim: usize, value: f32) -> Embedding {
    Embedding::new(vec![value; dim]).expect("embedding")
}

/// Convenience: a 1024-dim unit embedding at `axis` (matches HNSW schema).
pub fn unit_embedding_1024(axis: usize) -> Embedding {
    unit_embedding(EMBEDDING_DIM, axis)
}

/// Convenience: a 1024-dim constant embedding (every dim set to `value`).
pub fn constant_embedding_1024(value: f32) -> Embedding {
    constant_embedding(EMBEDDING_DIM, value)
}

/// Seed a single accepted fact into `store` under the canonical memory_key.
pub async fn seed_accepted_fact(
    store: &SurrealStore,
    content: &str,
    embedding: Embedding,
    confidence: f32,
    session: SessionId,
    extracted_at: Timestamp,
) -> FactId {
    seed_accepted_fact_with_threshold(
        store,
        content,
        embedding,
        confidence,
        session,
        extracted_at,
        0.7,
    )
    .await
}

/// Same as [`seed_accepted_fact`] but lets the caller lower the
/// `ConfidenceConfig::accept_threshold` so a below-0.7 confidence can still be
/// persisted as `Accepted`. Used by tests that exercise the retrieval
/// pre-filter's `min_confidence` gate against facts that the domain would
/// otherwise refuse to accept.
pub async fn seed_accepted_fact_with_threshold(
    store: &SurrealStore,
    content: &str,
    embedding: Embedding,
    confidence: f32,
    session: SessionId,
    extracted_at: Timestamp,
    accept_threshold: f32,
) -> FactId {
    let mut fact = Fact::new_pending(
        content,
        enrichment_memory_key(),
        session,
        embedding,
        extracted_at,
        smos_domain::config::ConfidenceConfig::default().base,
    )
    .expect("pending fact");
    let cfg = smos_domain::config::ConfidenceConfig {
        accept_threshold,
        ..smos_domain::config::ConfidenceConfig::default()
    };
    fact.set_status_and_confidence(
        FactStatus::Accepted,
        Confidence::new(confidence).expect("confidence"),
        &cfg,
    )
    .expect("accept");
    let id = fact.id().clone();
    FactRepository::save(store, &fact).await.expect("save fact");
    id
}

/// Seed a single pending fact (the pre-filter must drop it).
pub async fn seed_pending_fact(
    store: &SurrealStore,
    content: &str,
    embedding: Embedding,
    session: SessionId,
    extracted_at: Timestamp,
) -> FactId {
    let fact = Fact::new_pending(
        content,
        enrichment_memory_key(),
        session,
        embedding,
        extracted_at,
        smos_domain::config::ConfidenceConfig::default().base,
    )
    .expect("pending fact");
    let id = fact.id().clone();
    FactRepository::save(store, &fact).await.expect("save fact");
    id
}

/// Seed an accepted fact and tombstone it (`valid_until = Some`) so the
/// pre-filter must drop it.
pub async fn seed_expired_fact(
    store: &SurrealStore,
    content: &str,
    embedding: Embedding,
    session: SessionId,
    extracted_at: Timestamp,
) -> FactId {
    let mut fact = Fact::new_pending(
        content,
        enrichment_memory_key(),
        session,
        embedding,
        extracted_at,
        smos_domain::config::ConfidenceConfig::default().base,
    )
    .expect("pending fact");
    fact.set_status_and_confidence(
        FactStatus::Accepted,
        Confidence::new(0.9).expect("confidence"),
        &smos_domain::config::ConfidenceConfig::default(),
    )
    .expect("accept");
    let valid_from = fact.valid_from();
    let later = Timestamp::from_unix_secs(valid_from.as_unix_secs() + 3600).expect("later");
    fact.set_valid_until(Some(later)).expect("tombstone");
    let id = fact.id().clone();
    FactRepository::save(store, &fact).await.expect("save fact");
    id
}

/// Build a SmosConfig whose adapter URLs point at the supplied wiremock
/// servers. The default embedding dimensionality is 8 for fast fixture
/// construction; tests that exercise vector search use the same dimension.
///
/// Extraction is DISABLED by default: enrichment-focused tests do not mount
/// `/api/chat`, and an extraction attempt against the embeddings-only mock
/// would otherwise retry (1 s + 2 s) on every request. Extraction tests build
/// their own config with extraction enabled.
pub fn config_with_mocks(
    upstream_server: &MockServer,
    ollama_server: &MockServer,
    reranker_server: &MockServer,
) -> SmosConfig {
    let mut config = SmosConfig::default();
    config.upstream.providers = vec![UpstreamProvider {
        name: "test-upstream".into(),
        url: format!("{}/v1/chat/completions", upstream_server.uri()),
        api_key: String::new(),
        auth_header: "Authorization".into(),
        timeout_seconds: 5,
    }];
    config.llm_extraction.url = ollama_server.uri();
    config.llm_extraction.timeout_seconds = 5;
    config.embedding.url = ollama_server.uri();
    config.embedding.timeout_seconds = 5;
    config.reranker.url = reranker_server.uri();
    config.reranker.timeout_seconds = 5;
    config.server = ServerConfig::default();
    config.server.enable_response_extraction = false;
    config
}
