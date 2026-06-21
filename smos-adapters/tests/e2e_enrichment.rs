//! E2E: enrichment pipeline.
//!
//! Each test spins up three wiremock servers (Ollama embeddings, llama.cpp
//! reranker, OpenAI-compatible upstream) and an isolated in-process SurrealDB
//! seeded with fixture facts. The full `HandleChatCompletion` pipeline runs
//! through the spawned SMOS HTTP server.
//!
//! Assertions target the externally-observable behaviour: the body the
//! upstream mock received (memory-block presence, fact id lines) and the
//! persisted state (heat_base / last_access_at after retrieval). Fail-open
//! semantics are covered by tests that mount a 500 on the embedder / reranker
//! and verify the request still forwards.

mod common;

use common::{
    build_state, chat_body, enrichment_memory_key, fixed_now, fixed_session_id, seed_accepted_fact,
    seed_accepted_fact_with_threshold, seed_expired_fact, seed_pending_fact, serve_state,
    unit_embedding_1024,
};
use serde_json::{Value, json};
use smos_application::ports::FactRepository;
use smos_domain::Timestamp;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------
// Mock builders — keep the per-test `Mock::given(...)` chains short.
// ---------------------------------------------------------------------------

/// Mount a 200 OK mock on Ollama `/api/embeddings` returning a constant
/// 8-dimensional embedding. The mock expects exactly one call (the pipeline
/// embeds the topic once per request).
async fn mount_ollama_ok(server: &MockServer, embedding: Vec<f32>) {
    mount_ollama_ok_n(server, embedding, 1).await;
}

/// Same as [`mount_ollama_ok`] but lets the caller choose the expected number
/// of calls (used by tests that issue several enrichment requests against the
/// same Ollama mock).
async fn mount_ollama_ok_n(server: &MockServer, embedding: Vec<f32>, expected_calls: u64) {
    Mock::given(method("POST"))
        .and(path("/api/embeddings"))
        .and(wiremock::matchers::header(
            "content-type",
            "application/json",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embedding": embedding,
        })))
        .expect(expected_calls)
        .mount(server)
        .await;
}

/// Mount a 500 on Ollama to drive the fail-open path. `expect(0..=1)` is used
/// (rather than `expect(0)`) so the assertion is robust to short-topic
/// short-circuits in tests that also exercise that path.
async fn mount_ollama_500(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/api/embeddings"))
        .respond_with(ResponseTemplate::new(500))
        .mount(server)
        .await;
}

/// Mount a 200 OK reranker mock returning the supplied `(index, score)` pairs.
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

/// Mount a 500 reranker to drive the top-N-survivors fallback.
async fn mount_reranker_500(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(500))
        .mount(server)
        .await;
}

/// Mount a non-streaming 200 upstream mock returning a minimal OpenAI reply.
async fn mount_upstream_ok(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-x",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
        })))
        .mount(server)
        .await;
}

/// Fetch the recorded upstream request body (the last one).
async fn last_upstream_request(upstream: &MockServer) -> Value {
    let requests = upstream.received_requests().await.unwrap_or_default();
    let last = requests
        .into_iter()
        .filter(|r| r.url.path().contains("/v1/chat/completions"))
        .last()
        .expect("at least one upstream request");
    serde_json::from_slice(&last.body).expect("upstream body is JSON")
}

/// Reference embedding used by most tests: a 1024-dim unit vector at axis 0.
/// Seeded facts use the same vector so cosine distance is ~0 and the HNSW /
/// brute-force search surfaces them.
fn query_embedding() -> Vec<f32> {
    unit_embedding_1024(0).as_slice().to_vec()
}

/// Common chat body that exercises the canonical memory_key + a topic long
/// enough to pass `min_topic_chars`.
fn enrichment_chat_body() -> Value {
    chat_body("origa:gpt-4o", vec![]).replace_topic("hello world from rust")
}

trait ReplaceTopic {
    fn replace_topic(self, topic: &str) -> Self;
}

impl ReplaceTopic for Value {
    fn replace_topic(self, topic: &str) -> Self {
        let mut v = self;
        if let Some(obj) = v.as_object_mut()
            && let Some(messages) = obj.get_mut("messages").and_then(Value::as_array_mut)
            && let Some(first) = messages.first_mut()
        {
            first["content"] = json!(topic);
        }
        v
    }
}

// ---------------------------------------------------------------------------
// 1. Happy path: memory block injected
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enrichment_injects_memory_block_with_fact_ids() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    let session = fixed_session_id(1);
    let id = seed_accepted_fact(
        &state.store,
        "Rust is memory-safe",
        unit_embedding_1024(0),
        0.9,
        session.clone(),
        fixed_now(),
    )
    .await;

    mount_ollama_ok(&ollama, query_embedding()).await;
    mount_reranker_ok(&reranker, vec![(0, 0.95)]).await;
    mount_upstream_ok(&upstream).await;

    let smos = serve_state(state.clone()).await;
    let _ = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&enrichment_chat_body())
        .send()
        .await
        .expect("send");

    let upstream_body = last_upstream_request(&upstream).await;
    let first_content = upstream_body["messages"][0]["content"]
        .as_str()
        .expect("content");
    assert!(
        first_content.contains("<smos-memory session=\""),
        "expected memory block in messages[0]; got: {first_content}"
    );
    assert!(
        first_content.contains(&format!("[{}] Rust is memory-safe", id.as_str())),
        "expected fact id line; got: {first_content}"
    );
}

// ---------------------------------------------------------------------------
// 2. Fail-open: embedder error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enrichment_fail_open_on_embedder_error_forwards_without_block() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    seed_accepted_fact(
        &state.store,
        "Rust is memory-safe",
        unit_embedding_1024(0),
        0.9,
        fixed_session_id(1),
        fixed_now(),
    )
    .await;

    mount_ollama_500(&ollama).await;
    // Reranker should not be called (enrichment short-circuits at embed step).
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"results": []})))
        .expect(0)
        .mount(&reranker)
        .await;
    mount_upstream_ok(&upstream).await;

    let smos = serve_state(state).await;
    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&enrichment_chat_body())
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 200);

    let upstream_body = last_upstream_request(&upstream).await;
    let first_content = upstream_body["messages"][0]["content"]
        .as_str()
        .expect("content");
    assert!(
        !first_content.contains("<smos-memory"),
        "no memory block expected on embedder failure; got: {first_content}"
    );
}

/// Regression guard for the silent-data-loss bug fixed by making
/// `EnrichRequest::execute` infallible at the type level. Before the fix,
/// `HandleChatCompletion` did `std::mem::take(&mut request.messages)` and, on
/// the (now-deleted) `Err` arm, assigned `Vec::new()` — the user's original
/// messages vanished from the request forwarded to the upstream. This test
/// makes the failure mode externally observable: send a chat request whose
/// first message carries a unique sentinel topic, force the embedder to 500,
/// and assert the upstream received the original content verbatim. The
/// sentinel survives only if the pipeline never reaches the `Vec::new()`
/// replacement path.
///
/// Stronger than `enrichment_fail_open_on_embedder_error_forwards_without_block`
/// (which only checks no `<smos-memory>` block is added): this one pins the
/// exact bytes the upstream saw, so a regression that drops messages — even
/// without inventing a memory block — fails immediately.
#[tokio::test]
async fn enrichment_failure_preserves_original_user_message_verbatim() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    seed_accepted_fact(
        &state.store,
        "Rust is memory-safe",
        unit_embedding_1024(0),
        0.9,
        fixed_session_id(1),
        fixed_now(),
    )
    .await;

    // Force every enrichment stage that could conceivably fail to fail:
    // embedder 500 ⇒ short-circuit at step 3. Reranker is mounted 500 too so
    // the test stays correct even if a future refactor changes which step
    // the fail-open triggers at (reranker-error fallback path).
    mount_ollama_500(&ollama).await;
    mount_reranker_500(&reranker).await;
    mount_upstream_ok(&upstream).await;

    // Unique sentinel — never reused by any other test, never derived from a
    // fact document — so any replacement path (Vec::new(), placeholder text,
    // memory-block-only payload) cannot accidentally match.
    const SENTINEL_TOPIC: &str = "regression-sentinel-7c4a8d1e-fail-open";
    let body = enrichment_chat_body().replace_topic(SENTINEL_TOPIC);

    let smos = serve_state(state).await;
    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 200);

    let upstream_body = last_upstream_request(&upstream).await;
    let messages = upstream_body["messages"]
        .as_array()
        .expect("messages must be a non-empty array forwarded to upstream");
    assert!(
        !messages.is_empty(),
        "DATA-LOSS REGRESSION: upstream received zero messages after \
         enrichment failure. The `mem::take` + `Vec::new()` bug is back."
    );

    let first_content = messages[0]["content"]
        .as_str()
        .expect("first message content is a string");
    assert_eq!(
        first_content, SENTINEL_TOPIC,
        "DATA-LOSS REGRESSION: upstream received `{first_content}` instead \
         of the original user message `{SENTINEL_TOPIC}`. The \
         `EnrichRequest::execute` fail-open contract has been broken — the \
         original messages MUST be forwarded unchanged when any enrichment \
         port fails."
    );

    // Defence-in-depth: also verify no spurious extra messages were invented
    // by the fail-open path. The original request had exactly one user
    // message; the upstream must see exactly one user message.
    assert_eq!(
        messages.len(),
        1,
        "fail-open must not synthesise additional messages; got {messages:?}"
    );
}

// ---------------------------------------------------------------------------
// 3. Fail-open: reranker error → fallback to top-N survivors
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enrichment_fail_open_on_reranker_error_uses_survivor_fallback() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    let session = fixed_session_id(1);
    let id = seed_accepted_fact(
        &state.store,
        "Rust is memory-safe",
        unit_embedding_1024(0),
        0.9,
        session,
        fixed_now(),
    )
    .await;

    mount_ollama_ok(&ollama, query_embedding()).await;
    mount_reranker_500(&reranker).await;
    mount_upstream_ok(&upstream).await;

    let smos = serve_state(state).await;
    let _ = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&enrichment_chat_body())
        .send()
        .await
        .expect("send");

    let upstream_body = last_upstream_request(&upstream).await;
    let first_content = upstream_body["messages"][0]["content"]
        .as_str()
        .expect("content");
    assert!(
        first_content.contains(&format!("[{}]", id.as_str())),
        "expected survivor fallback to inject the fact; got: {first_content}"
    );
}

// ---------------------------------------------------------------------------
// 4. Fail-open: no hits
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enrichment_forwards_without_block_when_store_empty() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
    let state = build_state(config).await;

    mount_ollama_ok(&ollama, query_embedding()).await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"results": []})))
        .expect(0)
        .mount(&reranker)
        .await;
    mount_upstream_ok(&upstream).await;

    let smos = serve_state(state).await;
    let _ = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&enrichment_chat_body())
        .send()
        .await
        .expect("send");

    let upstream_body = last_upstream_request(&upstream).await;
    let first_content = upstream_body["messages"][0]["content"]
        .as_str()
        .expect("content");
    assert!(
        !first_content.contains("<smos-memory"),
        "no memory block expected on empty store; got: {first_content}"
    );
}

// ---------------------------------------------------------------------------
// 5. Topic too short — embedder must not be called
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enrichment_skips_when_topic_below_min_chars_and_skips_embedder() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
    let state = build_state(config).await;

    // `expect(0)` — the pipeline must short-circuit before embedding.
    Mock::given(method("POST"))
        .and(path("/api/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"embedding": [0.0]})))
        .expect(0)
        .mount(&ollama)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"results": []})))
        .expect(0)
        .mount(&reranker)
        .await;
    mount_upstream_ok(&upstream).await;

    let body = chat_body("origa:gpt-4o", vec![]).replace_topic("ab"); // 2 chars < min_topic_chars=3

    let smos = serve_state(state).await;
    let _ = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send");

    let upstream_body = last_upstream_request(&upstream).await;
    let first_content = upstream_body["messages"][0]["content"]
        .as_str()
        .expect("content");
    assert_eq!(
        first_content, "ab",
        "topic-too-short short-circuit must forward the original messages unchanged"
    );
}

// ---------------------------------------------------------------------------
// 6. Session dedup — second call does not re-inject
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enrichment_dedup_per_session_on_repeated_request() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    let session = fixed_session_id(7);
    seed_accepted_fact(
        &state.store,
        "Rust is memory-safe",
        unit_embedding_1024(0),
        0.9,
        session.clone(),
        fixed_now(),
    )
    .await;

    // The pipeline orders rerank (step 8) BEFORE session dedup (step 11), so
    // both requests hit the reranker even though the second call's dedup will
    // drop every survivor before injection. The Ollama embedder is also hit
    // twice (once per request). `mount_reranker_ok` is set up without an
    // `.expect(...)` count so it tolerates the second rerank call; the dedup
    // outcome is what we assert on below.
    mount_ollama_ok_n(&ollama, query_embedding(), 2).await;
    mount_reranker_ok(&reranker, vec![(0, 0.9)]).await;
    // Two upstream requests expected (first call + second call).
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-x",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
        })))
        .up_to_n_times(2)
        .mount(&upstream)
        .await;

    let smos = serve_state(state.clone()).await;
    // First request: history carries the session marker so detection resolves
    // to `session` and the first dedup_and_mark tags the fact as injected.
    let first_request_body = json!({
        "model": "origa:gpt-4o",
        "messages": [
            {"role": "assistant", "content": format!("prior answer\n<!-- smos:{} -->", session.as_str())},
            {"role": "user", "content": "hello world from rust"},
        ],
    });
    let _ = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&first_request_body)
        .send()
        .await
        .expect("first send");

    // Second request: same session marker in history. Dedup must skip the fact.
    let second_request_body = json!({
        "model": "origa:gpt-4o",
        "messages": [
            {"role": "assistant", "content": format!("prior answer\n<!-- smos:{} -->", session.as_str())},
            {"role": "user", "content": "hello world from rust again"},
        ],
    });
    let _ = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&second_request_body)
        .send()
        .await
        .expect("second send");

    let requests = upstream.received_requests().await.unwrap_or_default();
    assert!(
        requests.len() >= 2,
        "expected at least two upstream requests, got {}",
        requests.len()
    );
    let first_body: Value = serde_json::from_slice(&requests[0].body).expect("first body json");
    let first_first_msg = first_body["messages"][0]["content"]
        .as_str()
        .expect("first content");
    assert!(
        first_first_msg.contains("<smos-memory session=\""),
        "first call MUST inject the fact; got: {first_first_msg}"
    );
    assert!(
        first_first_msg.contains("[fact_"),
        "first call MUST include a fact id line; got: {first_first_msg}"
    );

    let second_body: Value = serde_json::from_slice(&requests.last().expect("at least two").body)
        .expect("second body json");
    let second_first_msg = second_body["messages"][0]["content"]
        .as_str()
        .expect("content");
    assert!(
        !second_first_msg.contains("[fact_"),
        "second call must NOT re-inject the fact; got: {second_first_msg}"
    );
}

// ---------------------------------------------------------------------------
// 7. Heat post-filter — stale fact filtered out
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enrichment_heat_post_filter_drops_stale_facts() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    let now = fixed_now();
    // 1000 hours ago — decay rate 0.03/hr → heat_live ≈ exp(-30) ≈ 0.
    let stale_at = Timestamp::from_unix_secs(now.as_unix_secs() - 1000 * 3600).expect("stale");
    seed_accepted_fact(
        &state.store,
        "stale fact",
        unit_embedding_1024(0),
        0.9,
        fixed_session_id(1),
        stale_at,
    )
    .await;

    mount_ollama_ok(&ollama, query_embedding()).await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"results": []})))
        .expect(0)
        .mount(&reranker)
        .await;
    mount_upstream_ok(&upstream).await;

    let smos = serve_state(state).await;
    let _ = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&enrichment_chat_body())
        .send()
        .await
        .expect("send");

    let upstream_body = last_upstream_request(&upstream).await;
    let first_content = upstream_body["messages"][0]["content"]
        .as_str()
        .expect("content");
    assert!(
        !first_content.contains("stale fact"),
        "stale fact must be filtered out; got: {first_content}"
    );
}

// ---------------------------------------------------------------------------
// 8. Min-confidence pre-filter
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enrichment_min_confidence_filter_drops_low_confidence_facts() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    // Seed an Accepted fact with confidence=0.5 (below min_confidence=0.7).
    // We lower the accept_threshold to 0.4 so the domain invariant permits
    // persisting it as Accepted — the retrieval pre-filter must still drop it.
    seed_accepted_fact_with_threshold(
        &state.store,
        "low-confidence fact",
        unit_embedding_1024(0),
        0.5,
        fixed_session_id(1),
        fixed_now(),
        0.4,
    )
    .await;

    mount_ollama_ok(&ollama, query_embedding()).await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"results": []})))
        .expect(0)
        .mount(&reranker)
        .await;
    mount_upstream_ok(&upstream).await;

    let smos = serve_state(state).await;
    let _ = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&enrichment_chat_body())
        .send()
        .await
        .expect("send");

    let upstream_body = last_upstream_request(&upstream).await;
    let first_content = upstream_body["messages"][0]["content"]
        .as_str()
        .expect("content");
    assert!(
        !first_content.contains("low-confidence fact"),
        "low-confidence fact must be filtered out; got: {first_content}"
    );
}

// ---------------------------------------------------------------------------
// 9. Heat boost — heat_base=1.0 and last_access_at=now persisted after hit
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enrichment_heat_boost_persists_after_retrieval_hit() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    let session = fixed_session_id(1);
    let stale_at = Timestamp::from_unix_secs(fixed_now().as_unix_secs() - 3600).expect("1h ago");
    let id = seed_accepted_fact(
        &state.store,
        "boostable fact",
        unit_embedding_1024(0),
        0.9,
        session,
        stale_at,
    )
    .await;

    // Sanity: pre-request heat_base < 1.0 after the setter? `new_pending`
    // defaults heat to 1.0; we instead set last_access_at to 1h ago and rely
    // on heat_live decay still being above the threshold (exp(-0.03) ≈ 0.97).
    mount_ollama_ok(&ollama, query_embedding()).await;
    mount_reranker_ok(&reranker, vec![(0, 0.9)]).await;
    mount_upstream_ok(&upstream).await;

    let smos = serve_state(state.clone()).await;
    let _ = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&enrichment_chat_body())
        .send()
        .await
        .expect("send");

    let refreshed = state
        .store
        .get(&id, &enrichment_memory_key())
        .await
        .expect("get")
        .expect("fact exists");
    assert!(
        (refreshed.heat_base().value() - 1.0).abs() < 1e-6,
        "heat_base should be 1.0 after boost, got {}",
        refreshed.heat_base().value()
    );
    assert!(
        refreshed.last_access_at().as_unix_secs() > stale_at.as_unix_secs(),
        "last_access_at should advance past the seeded value"
    );
}

// ---------------------------------------------------------------------------
// 10. Reranker empty response → top-N survivor fallback
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enrichment_reranker_empty_results_falls_back_to_survivors() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    let session = fixed_session_id(1);
    let id = seed_accepted_fact(
        &state.store,
        "fallback survivor",
        unit_embedding_1024(0),
        0.9,
        session,
        fixed_now(),
    )
    .await;

    mount_ollama_ok(&ollama, query_embedding()).await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"results": []})))
        .mount(&reranker)
        .await;
    mount_upstream_ok(&upstream).await;

    let smos = serve_state(state).await;
    let _ = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&enrichment_chat_body())
        .send()
        .await
        .expect("send");

    let upstream_body = last_upstream_request(&upstream).await;
    let first_content = upstream_body["messages"][0]["content"]
        .as_str()
        .expect("content");
    assert!(
        first_content.contains(&format!("[{}]", id.as_str())),
        "expected survivor fallback injection; got: {first_content}"
    );
}

// ---------------------------------------------------------------------------
// 11. Top-5 limit after rerank
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enrichment_top5_limit_caps_injected_facts() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    let session = fixed_session_id(1);
    // Seed 7 accepted facts that all match the query embedding.
    for i in 0..7 {
        seed_accepted_fact(
            &state.store,
            &format!("fact number {i}"),
            unit_embedding_1024(0),
            0.9,
            session.clone(),
            fixed_now(),
        )
        .await;
    }

    mount_ollama_ok(&ollama, query_embedding()).await;
    // Return all 7 from the reranker; the pipeline must truncate to top_k_final=5.
    mount_reranker_ok(
        &reranker,
        (0..7).map(|i| (i, 1.0 - i as f32 * 0.01)).collect(),
    )
    .await;
    mount_upstream_ok(&upstream).await;

    let smos = serve_state(state).await;
    let _ = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&enrichment_chat_body())
        .send()
        .await
        .expect("send");

    let upstream_body = last_upstream_request(&upstream).await;
    let first_content = upstream_body["messages"][0]["content"]
        .as_str()
        .expect("content");
    let fact_line_count = first_content
        .lines()
        .filter(|l| l.starts_with("[fact_"))
        .count();
    assert_eq!(
        fact_line_count, 5,
        "expected exactly 5 fact lines after top-5 cap; got: {first_content}"
    );
}

// ---------------------------------------------------------------------------
// 12. Pending facts filtered out
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enrichment_filters_out_pending_facts() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    let session = fixed_session_id(1);
    let accepted_id = seed_accepted_fact(
        &state.store,
        "accepted fact",
        unit_embedding_1024(0),
        0.9,
        session.clone(),
        fixed_now(),
    )
    .await;
    let pending_id = seed_pending_fact(
        &state.store,
        "pending fact",
        unit_embedding_1024(0),
        session,
        fixed_now(),
    )
    .await;
    let _ = pending_id; // sanity that we constructed it

    // SurrealStore's search_similar already filters by status=accepted, so the
    // pending fact never reaches the reranker. The reranker mock only sees the
    // accepted fact (index 0) — we mirror that in the response.
    mount_ollama_ok(&ollama, query_embedding()).await;
    mount_reranker_ok(&reranker, vec![(0, 0.9)]).await;
    mount_upstream_ok(&upstream).await;

    let smos = serve_state(state).await;
    let _ = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&enrichment_chat_body())
        .send()
        .await
        .expect("send");

    let upstream_body = last_upstream_request(&upstream).await;
    let first_content = upstream_body["messages"][0]["content"]
        .as_str()
        .expect("content");
    assert!(
        first_content.contains(&format!("[{}]", accepted_id.as_str())),
        "expected accepted fact injected; got: {first_content}"
    );
    assert!(
        !first_content.contains("pending fact"),
        "pending fact must NOT be injected; got: {first_content}"
    );
}

// ---------------------------------------------------------------------------
// 13. Expired facts (tombstoned via valid_until) filtered out
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enrichment_filters_out_expired_facts() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    let session = fixed_session_id(1);
    let _expired_id = seed_expired_fact(
        &state.store,
        "expired fact",
        unit_embedding_1024(0),
        session.clone(),
        fixed_now(),
    )
    .await;
    let accepted_id = seed_accepted_fact(
        &state.store,
        "live fact",
        unit_embedding_1024(0),
        0.9,
        session,
        fixed_now(),
    )
    .await;

    mount_ollama_ok(&ollama, query_embedding()).await;
    mount_reranker_ok(&reranker, vec![(0, 0.9)]).await;
    mount_upstream_ok(&upstream).await;

    let smos = serve_state(state).await;
    let _ = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&enrichment_chat_body())
        .send()
        .await
        .expect("send");

    let upstream_body = last_upstream_request(&upstream).await;
    let first_content = upstream_body["messages"][0]["content"]
        .as_str()
        .expect("content");
    assert!(
        first_content.contains(&format!("[{}]", accepted_id.as_str())),
        "expected live fact injected; got: {first_content}"
    );
    assert!(
        !first_content.contains("expired fact"),
        "expired fact must NOT be injected; got: {first_content}"
    );
}

// ---------------------------------------------------------------------------
// 14. Full pipeline smoke: enriched request round-trips end-to-end
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_pipeline_passthrough_with_enrichment_returns_marker() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = common::config_with_mocks(&upstream, &ollama, &reranker);
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

    mount_ollama_ok(&ollama, query_embedding()).await;
    mount_reranker_ok(&reranker, vec![(0, 0.95)]).await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-x",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
        })))
        .mount(&upstream)
        .await;

    let smos = serve_state(state).await;
    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&enrichment_chat_body())
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("body json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    // The handler injects the session marker into the assistant response; the
    // presence of "<!-- smos:" proves the full pipeline completed.
    assert!(
        content.contains("<!-- smos:"),
        "expected marker in response; got: {content}"
    );

    // Verify the upstream saw the memory block (i.e. enrichment actually ran).
    let upstream_body = last_upstream_request(&upstream).await;
    let upstream_first = upstream_body["messages"][0]["content"]
        .as_str()
        .expect("upstream first content");
    assert!(
        upstream_first.contains("<smos-memory session=\""),
        "expected memory block to reach upstream; got: {upstream_first}"
    );
}
