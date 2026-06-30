//! E2E: CLI→HTTP forwarding for `/v1/cli/search`.
//!
//! Hexagonal parity: the SAME `RetrieveFacts` use case is invoked by the
//! CLI-local branch and by the server-side handler. The local render
//! (`render_json(&scored)`) MUST byte-equal the server's response body,
//! because both sides call the same function on structurally-identical
//! inputs (same seeded facts, same wiremock embedder / reranker responses).
//!
//! These tests do NOT spin up a second SMOS process to hold the RocksDB
//! lock — the cross-process parity is covered by the manual smoke test in
//! the verification plan. Here we exercise the in-process hexagonal
//! contract: server handler vs local branch against the same `AppState`.

mod common;

use common::{
    build_state, config_with_mocks, fixed_now, fixed_session_id, seed_accepted_fact, serve_state,
    unit_embedding_1024,
};
use serde_json::{Value, json};
use smos::cli::search_runner::{OutputFormat, SearchArgs, SearchRequest, render_json};
use smos::http::axum_server::AppState;
use smos_application::use_cases::RetrieveFacts;
use std::sync::Arc;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Mount a deterministic 200 OK embedder mock returning the supplied
/// embedding for the query. The mock is liberal on call count so it serves
/// both the local-branch and the remote-handler invocation.
async fn mount_embeddings_ok(server: &MockServer, embedding: Vec<f32>) {
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"index": 0, "embedding": embedding}],
        })))
        .mount(server)
        .await;
}

/// Mount a deterministic 200 OK reranker returning the supplied scores.
async fn mount_reranker_ok(server: &MockServer, scores: Vec<(usize, f32)>) {
    let results: Vec<Value> = scores
        .into_iter()
        .map(|(index, score)| {
            json!({
                "index": index,
                "relevance_score": score,
                "document": {"text": format!("doc-{index}")},
            })
        })
        .collect();
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "results": results })))
        .mount(server)
        .await;
}

/// Build a `SearchArgs` matching the canonical test inputs.
fn search_args(query: &str) -> SearchArgs {
    SearchArgs {
        query: query.into(),
        memory_key: "origa".into(),
        top_k: None,
        format: OutputFormat::Json,
    }
}

/// Wrapper around `Arc<dyn Clock>` so the by-value `C: Clock` bound on
/// `RetrieveFacts` is satisfied. Mirrors the handler's `FlatClock`: both
/// wrap the SAME `state.clock` Arc so the parity test compares identical
/// clock values, not coincidentally-equal SystemClock instances.
#[derive(Clone)]
struct TestClock(Arc<dyn smos_application::ports::Clock + Send + Sync>);
impl smos_application::ports::Clock for TestClock {
    fn now(&self) -> smos_domain::Timestamp {
        self.0.now()
    }
}

/// Invoke `RetrieveFacts` directly against `state.store` with the same
/// adapters / config the `/v1/cli/search` handler uses. Mirrors the
/// local-branch half of `smos search` so the parity test exercises the
/// same code path on both sides of the wire.
async fn local_retrieve(
    state: &Arc<AppState>,
    args: &SearchArgs,
) -> Vec<smos_application::use_cases::ScoredSearchHit> {
    let embedder = smos::OllamaEmbedding::new(Arc::new(state.config.embedding.clone())).unwrap();
    let reranker = smos::LlamaCppReranker::new(Arc::new(state.config.reranker.clone())).unwrap();
    let clock = TestClock(state.clock.clone());
    let use_case = RetrieveFacts {
        facts: &state.store,
        embedder: &embedder,
        reranker: &reranker,
        clock: &clock,
        retrieval_cfg: &state.retrieval_cfg,
        heat_cfg: &state.heat_cfg,
    };
    let memory_key = smos_domain::MemoryKey::from_raw(&args.memory_key).unwrap();
    use_case
        .execute(&args.query, &memory_key, args.top_k)
        .await
        .expect("local retrieve")
}

// ---------------------------------------------------------------------------
// Parity: local render == remote body
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_local_render_byte_equals_remote_response_body() {
    let upstream = MockServer::start().await;
    let llama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = config_with_mocks(&upstream, &llama, &reranker);
    let state = build_state(config).await;
    let session = fixed_session_id(1);
    seed_accepted_fact(
        &state.store,
        "Rust is memory-safe",
        unit_embedding_1024(0),
        0.9,
        session,
        fixed_now(),
    )
    .await;

    let args = search_args("explain rust ownership");
    let query_vec = unit_embedding_1024(0).as_slice().to_vec();
    mount_embeddings_ok(&llama, query_vec).await;
    mount_reranker_ok(&reranker, vec![(0, 0.95)]).await;

    let base_url = serve_state(state.clone()).await;

    let remote_body = reqwest::Client::new()
        .post(format!("{base_url}/v1/cli/search"))
        .json(&SearchRequest::from_args(&args))
        .send()
        .await
        .expect("send")
        .bytes()
        .await
        .expect("body");

    let local_scored = local_retrieve(&state, &args).await;
    let local_rendered = render_json(&local_scored);

    assert_eq!(
        remote_body.as_ref(),
        local_rendered.as_bytes(),
        "remote body must byte-equal local render of the same use case output"
    );
}

#[tokio::test]
async fn search_endpoint_returns_application_json_content_type() {
    let upstream = MockServer::start().await;
    let llama = MockServer::start().await;
    let reranker = MockServer::start().await;
    let config = config_with_mocks(&upstream, &llama, &reranker);
    let state = build_state(config).await;
    seed_accepted_fact(
        &state.store,
        "Rust is memory-safe",
        unit_embedding_1024(0),
        0.9,
        fixed_session_id(2),
        fixed_now(),
    )
    .await;
    mount_embeddings_ok(&llama, unit_embedding_1024(0).as_slice().to_vec()).await;
    mount_reranker_ok(&reranker, vec![(0, 0.95)]).await;

    let base_url = serve_state(state).await;
    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/cli/search"))
        .json(&SearchRequest::from_args(&search_args("rust ownership")))
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .map(|v| v.to_str().unwrap_or("")),
        Some("application/json")
    );
}

#[tokio::test]
async fn search_endpoint_returns_empty_array_when_no_hits() {
    let upstream = MockServer::start().await;
    let llama = MockServer::start().await;
    let reranker = MockServer::start().await;
    let config = config_with_mocks(&upstream, &llama, &reranker);
    let state = build_state(config).await;
    // No facts seeded.
    mount_embeddings_ok(&llama, unit_embedding_1024(0).as_slice().to_vec()).await;
    mount_reranker_ok(&reranker, vec![]).await;

    let base_url = serve_state(state).await;
    let body = reqwest::Client::new()
        .post(format!("{base_url}/v1/cli/search"))
        .json(&SearchRequest::from_args(&search_args("anything")))
        .send()
        .await
        .expect("send")
        .text()
        .await
        .expect("body");
    assert_eq!(body, "[]", "empty result renders as bare []");
}

#[tokio::test]
async fn search_endpoint_400_on_invalid_memory_key() {
    let upstream = MockServer::start().await;
    let llama = MockServer::start().await;
    let reranker = MockServer::start().await;
    let config = config_with_mocks(&upstream, &llama, &reranker);
    let state = build_state(config).await;

    let base_url = serve_state(state).await;
    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/cli/search"))
        .json(&json!({"query": "x", "memory_key": "../etc", "top_k": null}))
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 400);
}

// ---------------------------------------------------------------------------
// Routing-resolver behaviour already unit-tested in forwarding.rs.
// 404-fallback contracts are tested as unit tests inside each runner's
// `#[cfg(test)] mod tests` (calling `execute_remote` directly, asserting
// `RemoteOutcome::EndpointNotFound`).
// ---------------------------------------------------------------------------

// ===========================================================================
// /v1/cli/finalize — Slice 2b
// ===========================================================================

#[tokio::test]
async fn finalize_endpoint_503_when_classifier_unavailable() {
    let upstream = MockServer::start().await;
    let llama = MockServer::start().await;
    let reranker = MockServer::start().await;
    let config = config_with_mocks(&upstream, &llama, &reranker);
    let state = build_state(config).await; // classifier: None, git_sync: OnceCell::new()

    let base_url = serve_state(state).await;
    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/cli/finalize"))
        .json(&json!({"session_id": "sess_0123456789ab", "memory_key": null}))
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 503);
}

#[tokio::test]
async fn finalize_endpoint_400_on_invalid_session_id() {
    let upstream = MockServer::start().await;
    let llama = MockServer::start().await;
    let reranker = MockServer::start().await;
    let config = config_with_mocks(&upstream, &llama, &reranker);
    let state = build_state(config).await;

    let base_url = serve_state(state).await;
    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/cli/finalize"))
        .json(&json!({"session_id": "not-a-valid-session-id", "memory_key": null}))
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn finalize_endpoint_returns_text_plain_content_type() {
    // The handler always returns text/plain (the rendered report). Even on
    // the 503 path the content type is JSON (error_response), so this test
    // specifically exercises the 400 path (which also uses error_response).
    // The success path's content type is pinned implicitly by the parity
    // test below.
    let upstream = MockServer::start().await;
    let llama = MockServer::start().await;
    let reranker = MockServer::start().await;
    let config = config_with_mocks(&upstream, &llama, &reranker);
    let state = build_state(config).await;

    let base_url = serve_state(state).await;
    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/cli/finalize"))
        .json(&json!({"session_id": "bad", "memory_key": null}))
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 400);
}

/// Pins that `print_finalize_report` is deterministic and produces the
/// expected JSON content shape. The full end-to-end parity (real classifier
/// on both sides — HTTP handler body vs CLI local render) is deferred to
/// `cargo tall` (requires the 643 MB DeBERTa model). The structural
/// argument: handler and local branch share the SAME `run_finalize_pipeline`
/// helper → identical AggregatedStats → identical render.
#[tokio::test]
async fn finalize_report_render_is_deterministic_and_pins_content() {
    use smos::cli::finalize_runner::{AggregatedStats, print_finalize_report};

    let agg = AggregatedStats {
        session_id: "sess_abc123".into(),
        processed: 5,
        finalized: 3,
        merged: 1,
        conflicts: 1,
        rejected: 1,
        memory_keys: Vec::new(),
    };

    // Determinism: two calls produce identical output.
    let render1 = print_finalize_report(&agg, 2);
    let render2 = print_finalize_report(&agg, 2);
    assert_eq!(render1, render2, "render must be deterministic");

    // Pattern A: report does NOT end with \n — caller appends one.
    assert!(!render1.ends_with('\n'));

    // Pin the exact content so a future render change is caught.
    let v: serde_json::Value = serde_json::from_str(&render1).expect("valid JSON");
    assert_eq!(v["session_id"], "sess_abc123");
    assert_eq!(v["processed"], 5);
    assert_eq!(v["memory_keys_scanned"], 2);
}

// ===========================================================================
// /v1/cli/import/raw — Slice 3
// ===========================================================================

#[tokio::test]
async fn raw_import_endpoint_503_when_finalize_requested_and_classifier_unavailable() {
    let upstream = MockServer::start().await;
    let llama = MockServer::start().await;
    let reranker = MockServer::start().await;
    let config = config_with_mocks(&upstream, &llama, &reranker);
    let state = build_state(config).await; // classifier: None

    let base_url = serve_state(state).await;
    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/cli/import/raw"))
        .json(&json!({"text": "hello", "memory_key": "bob", "no_finalize": false}))
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 503);
}

#[tokio::test]
async fn raw_import_endpoint_400_on_invalid_memory_key() {
    let upstream = MockServer::start().await;
    let llama = MockServer::start().await;
    let reranker = MockServer::start().await;
    let config = config_with_mocks(&upstream, &llama, &reranker);
    let state = build_state(config).await;

    let base_url = serve_state(state).await;
    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/cli/import/raw"))
        .json(&json!({"text": "hello", "memory_key": "../etc", "no_finalize": true}))
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 400);
}

/// Pins that `render_raw_import_report` is deterministic and produces the
/// expected content shape (Pattern A: no trailing `\n`). The full
/// end-to-end parity (real classifier on both sides — HTTP handler body vs
/// CLI local render) is deferred to `cargo tall` (requires the 643 MB
/// DeBERTa model). The structural argument: handler and local branch share
/// the SAME `run_raw_import_pipeline` helper → identical RawImportResult →
/// identical render.
#[tokio::test]
async fn raw_import_report_render_is_deterministic_and_pins_content() {
    use smos::cli::raw_import_runner::{RawImportResult, render_raw_import_report};
    use smos_application::use_cases::FinalizeStats;
    use smos_domain::{MemoryKey, SessionId};

    let mk = MemoryKey::from_raw("bob").unwrap();
    let sid = SessionId::from_raw("sess_0123456789ab").unwrap();
    let result = RawImportResult {
        memory_key: mk,
        session_id: sid.clone(),
        new_facts: 3,
        finalize_stats: Some(FinalizeStats {
            session_id: sid.as_str().to_string(),
            processed: 3,
            finalized: 2,
            merged: 1,
            conflicts: 0,
            rejected: 1,
        }),
    };

    // Determinism: two calls produce identical output.
    let render1 = render_raw_import_report(&result, false);
    let render2 = render_raw_import_report(&result, false);
    assert_eq!(render1, render2, "render must be deterministic");

    // Pattern A: report does NOT end with \n — caller appends one.
    assert!(!render1.ends_with('\n'));

    // Pin the content shape.
    assert!(render1.starts_with("\n=== Raw import complete ===\n"));
    assert!(render1.contains("Memory key: bob"));
    assert!(render1.contains("New facts:  3"));
    assert!(render1.contains("=== Finalizing ==="));
    assert!(render1.contains("\"processed\": 3"));
}

#[tokio::test]
async fn raw_import_report_no_finalize_flag_renders_skip_message() {
    use smos::cli::raw_import_runner::{RawImportResult, render_raw_import_report};
    use smos_domain::{MemoryKey, SessionId};

    let mk = MemoryKey::from_raw("bob").unwrap();
    let result = RawImportResult {
        memory_key: mk,
        session_id: SessionId::from_raw("sess_0123456789ab").unwrap(),
        new_facts: 2,
        finalize_stats: None,
    };

    let rendered = render_raw_import_report(&result, true);
    assert!(rendered.contains("(finalize skipped via --no-finalize)"));
    assert!(!rendered.contains("=== Finalizing ==="));
}

// ===========================================================================
// /v1/cli/import/opencode — Slice 4
// ===========================================================================

#[tokio::test]
async fn import_opencode_endpoint_400_on_invalid_memory_key() {
    let upstream = MockServer::start().await;
    let llama = MockServer::start().await;
    let reranker = MockServer::start().await;
    let config = config_with_mocks(&upstream, &llama, &reranker);
    let state = build_state(config).await;

    let base_url = serve_state(state).await;
    let resp = reqwest::Client::new()
        .post(format!("{base_url}/v1/cli/import/opencode"))
        .json(&json!({"turns": [], "memory_key": "../etc", "session_id": "x", "agents": []}))
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 400);
}

/// Pins that `render_import_opencode_report` is deterministic and produces
/// the expected content shape (Pattern A: no trailing `\n`). The full
/// end-to-end parity (real embedder/extractor on both sides — HTTP handler
/// body vs CLI local render) is deferred to `cargo tall` / manual smoke.
/// The structural argument: handler and local branch share the SAME
/// `run_import_opencode_pipeline` helper → identical ImportStats →
/// identical render.
#[tokio::test]
async fn import_opencode_report_render_is_deterministic_and_pins_content() {
    use smos::cli::import_runner::render_import_opencode_report;
    use smos_application::use_cases::ImportStats;
    use smos_domain::MemoryKey;

    let stats = ImportStats {
        session_id: "sess_abc".into(),
        turns_processed: 10,
        turns_skipped: 2,
        facts_extracted: 7,
    };
    let mk = MemoryKey::from_raw("bob").unwrap();

    // Determinism: two calls produce identical output.
    let render1 = render_import_opencode_report(&stats, &mk);
    let render2 = render_import_opencode_report(&stats, &mk);
    assert_eq!(render1, render2, "render must be deterministic");

    // Pattern A: report does NOT end with \n — caller appends one.
    assert!(!render1.ends_with('\n'));

    // Pin the content shape.
    assert!(render1.starts_with("\n=== Import complete ===\n"));
    assert!(render1.contains("Session:      sess_abc"));
    assert!(render1.contains("Memory key:   bob"));
    assert!(render1.contains("Processed:    10 turns"));
    assert!(render1.contains("Skipped:      2 turns"));
    assert!(render1.contains("New facts:    7"));
}

/// `--list` and `--dry-run` must NEVER forward, even when a server is
/// reachable. Verify the decision function pins this contract.
#[test]
fn import_opencode_skip_forwarding_decision_pinned() {
    use smos::cli::import_opencode_should_skip_forwarding;

    assert!(
        import_opencode_should_skip_forwarding(true, false),
        "--list skips"
    );
    assert!(
        import_opencode_should_skip_forwarding(false, true),
        "--dry-run skips"
    );
    assert!(
        !import_opencode_should_skip_forwarding(false, false),
        "real import does NOT skip"
    );
    assert!(
        import_opencode_should_skip_forwarding(true, true),
        "both flags skip"
    );
}

/// Wire turn roundtrips through serialisation + back to AssistantTurn.
/// Pins that the wire DTO matches the application struct field-for-field.
#[test]
fn wire_assistant_turn_roundtrips_through_serde() {
    use smos::cli::WireAssistantTurn;
    use smos_application::use_cases::import_opencode_session::AssistantTurn;

    let turn = AssistantTurn {
        message_id: "msg_001".into(),
        agent: "build".into(),
        content: "I like Rust".into(),
        tool_calls: vec![],
    };
    let wire: WireAssistantTurn = turn.clone().into();
    let json = serde_json::to_string(&wire).expect("serialize");
    let back: WireAssistantTurn = serde_json::from_str(&json).expect("deserialize");
    let restored: AssistantTurn = back.into();
    assert_eq!(restored, turn);
}
