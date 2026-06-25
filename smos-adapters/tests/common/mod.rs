//! Shared helpers for the SMOS E2E suites.
//!
//! Each test spins up wiremock upstreams (mock OpenAI chat server, optionally
//! a mock OpenAI-compatible embedding+extraction server and a mock
//! reranker), builds an SMOS router pointing at them, and serves SMOS on an
//! ephemeral port inside a spawned task. Tests then hit SMOS with a plain
//! `reqwest` client exactly the way a real OpenAI client would.
//!
//! Passthrough tests don't exercise enrichment, so they reuse [`spawn_smos`]
//! which wires stub providers that short-circuit enrichment (unreachable
//! llama-server/reranker URLs fail-open). Enrichment tests use [`build_state`]
//! / [`serve_state`] to wire real providers against the supplied wiremock URLs
//! and seed facts through the returned `SurrealStore`.

#![allow(dead_code)]

use std::sync::Arc;

use axum::Router;
use serde_json::{Value, json};
use smos::SystemClock;
use smos::SystemIdGenerator;
use smos::config::{ProviderConfig, ServerConfig, SmosConfig};
use smos::http::axum_server::{AppState, build_router};
use smos::upstream::ReqwestUpstreamRouter;
use smos::{LlamaCppReranker, OllamaEmbedding, OllamaExtractor, SurrealStore};
use smos_application::ports::{Clock, FactRepository, IdGenerator};
use smos_domain::{
    Confidence, Embedding, Fact, FactId, FactStatus, MemoryKey, NewPendingRequest, SessionId,
    Timestamp,
};
use surrealdb::Surreal;
use surrealdb::engine::local::RocksDb;
use tempfile::TempDir;
use wiremock::MockServer;

/// Poll `predicate` every `interval` until it returns `true` or `timeout`
/// elapses. A final check after the loop keeps the caller's next assertion
/// message aligned with the post-timeout state.
///
/// Replaces a fixed `tokio::time::sleep` + state-check with a bounded poll:
/// the test proceeds as soon as the condition holds (instead of always
/// paying the full sleep) and a slow host gets the whole `timeout` headroom
/// rather than failing on a fixed deadline. The predicate is async because
/// the e2e state-checks are async SurrealDB / wiremock reads; sync checks
/// (`AtomicUsize` counters) are wrapped in an `async {}` block.
pub async fn wait_for<F, Fut>(
    mut predicate: F,
    timeout: std::time::Duration,
    interval: std::time::Duration,
) where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if predicate().await {
            return;
        }
        tokio::time::sleep(interval).await;
    }
    let _ = predicate().await;
}

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

/// The canonical person name used by every passthrough / enrichment test.
/// Tests that mount a person-specific mock MUST configure `[persons.origa]`
/// via [`config_pointing_at`] / [`config_with_mocks`] so the routing layer
/// can resolve `request.model = "origa"` to the test upstream.
pub const TEST_PERSON: &str = "origa";
/// Upstream model id the mock expects on the wire (after routing rewrites
/// `request.model` from `"origa"` to this value).
pub const TEST_UPSTREAM_MODEL: &str = "gpt-4o";

/// Build an SMOS config whose `[[providers]]` array + `[persons.origa]` map
/// route `request.model = "origa"` to the supplied `upstream_base`.
#[allow(clippy::field_reassign_with_default)]
pub fn config_pointing_at(upstream_base: &str) -> SmosConfig {
    let mut config = SmosConfig::default();
    config.providers = vec![ProviderConfig {
        name: "test-upstream".into(),
        url: format!("{upstream_base}/v1/chat/completions"),
        api_key_env: String::new(),
        auth_header: "Authorization".into(),
        timeout_seconds: 5,
    }];
    // Route person `origa` to the test provider, expecting the upstream
    // model `gpt-4o` on the wire. The legacy `parse_model("origa:gpt-4o")`
    // shape is gone — the routing now happens via the persons map.
    config.persons.insert(
        TEST_PERSON.into(),
        smos::config::PersonConfig {
            provider: "test-upstream".into(),
            model: TEST_UPSTREAM_MODEL.into(),
            persona: String::new(),
        },
    );
    config.server = ServerConfig::default();
    config
}

#[allow(clippy::field_reassign_with_default)]
/// Build a SmosConfig whose adapter URLs point at the supplied wiremock
/// servers. The default embedding dimensionality is 8 for fast fixture
/// construction; tests that exercise vector search use the same dimension.
///
/// Extraction is DISABLED by default: enrichment-focused tests do not mount
/// `/v1/chat/completions`, and an extraction attempt against the
/// embeddings-only mock would otherwise retry (1 s + 2 s) on every request.
/// Extraction tests build their own config with extraction enabled.
pub fn config_with_mocks(
    upstream_server: &MockServer,
    llama_server: &MockServer,
    reranker_server: &MockServer,
) -> SmosConfig {
    let mut config = SmosConfig::default();
    config.providers = vec![ProviderConfig {
        name: "test-upstream".into(),
        url: format!("{}/v1/chat/completions", upstream_server.uri()),
        api_key_env: String::new(),
        auth_header: "Authorization".into(),
        timeout_seconds: 5,
    }];
    config.persons.insert(
        TEST_PERSON.into(),
        smos::config::PersonConfig {
            provider: "test-upstream".into(),
            model: TEST_UPSTREAM_MODEL.into(),
            persona: String::new(),
        },
    );
    config.llm_extraction.url = llama_server.uri();
    config.llm_extraction.timeout_seconds = 5;
    config.embedding.url = llama_server.uri();
    config.embedding.timeout_seconds = 5;
    config.reranker.url = reranker_server.uri();
    config.reranker.timeout_seconds = 5;
    config.server = ServerConfig::default();
    config.server.enable_response_extraction = false;
    config
}

/// Spawn SMOS on an ephemeral port against a wiremock `upstream_base` with
/// stub providers (empty SurrealStore, unreachable llama-server / reranker
/// URLs that short-circuit enrichment via fail-open). Used by passthrough
/// tests.
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

    let upstream = ReqwestUpstreamRouter::from_config(&config.providers).expect("upstream router");
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
    let extraction_supervisor = smos::runtime::ExtractionSupervisor::new();

    // Pre-build the IO-free routing views the handler reads. Tests use the
    // same projection helper the production runner uses so the path stays
    // covered.
    let persons_view = Arc::new(build_person_view(&config.persons));
    let providers_view = Arc::new(build_provider_view(&config.providers));

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
        persons_view,
        providers_view,
    });
    std::mem::forget(tmp);
    state
}

/// Project `smos::config::PersonConfig` into the IO-free
/// [`PersonEntry`] view consumed by the routing layer. Mirrors the
/// production helper in `cli::server_runner::build_person_view` so the
/// test fixtures exercise the same code path.
fn build_person_view(
    persons: &std::collections::HashMap<String, smos::config::PersonConfig>,
) -> std::collections::HashMap<String, smos_application::helpers::person_router::PersonEntry> {
    persons
        .iter()
        .map(|(name, p)| {
            (
                name.clone(),
                smos_application::helpers::person_router::PersonEntry {
                    provider: p.provider.clone(),
                    model: p.model.clone(),
                    persona: p.persona.clone(),
                },
            )
        })
        .collect()
}

/// Project `smos::config::ProviderConfig` into the IO-free
/// [`ProviderEntry`] view.
fn build_provider_view(
    providers: &[smos::config::ProviderConfig],
) -> Vec<smos_application::helpers::person_router::ProviderEntry> {
    providers
        .iter()
        .map(
            |p| smos_application::helpers::person_router::ProviderEntry {
                name: p.name.clone(),
            },
        )
        .collect()
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

/// Reference embedding dimensionality — re-exported from
/// [`smos::storage::surreal_schema::EMBEDDING_DIM`] (itself an alias for the
/// domain's canonical `Embedding::EXPECTED_DIM`), so tests stay in lockstep
/// with the HNSW index declared in `surreal_schema::FACT_DDL`. Tests that
/// exercise vector search must seed embeddings of this dimensionality.
pub use smos::storage::surreal_schema::EMBEDDING_DIM;

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
    let mut fact = Fact::new_pending(NewPendingRequest {
        content,
        memory_key: enrichment_memory_key(),
        session,
        embedding,
        extracted_at,
        base_confidence: smos_domain::config::ConfidenceConfig::default().base,
    })
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
    let fact = Fact::new_pending(NewPendingRequest {
        content,
        memory_key: enrichment_memory_key(),
        session,
        embedding,
        extracted_at,
        base_confidence: smos_domain::config::ConfidenceConfig::default().base,
    })
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
    let mut fact = Fact::new_pending(NewPendingRequest {
        content,
        memory_key: enrichment_memory_key(),
        session,
        embedding,
        extracted_at,
        base_confidence: smos_domain::config::ConfidenceConfig::default().base,
    })
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
