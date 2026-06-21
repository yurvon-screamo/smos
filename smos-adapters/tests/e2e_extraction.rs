//! E2E: response extraction pipeline.
//!
//! Each test spins up wiremock upstreams (OpenAI-compatible upstream, Ollama
//! `/api/chat` extractor, Ollama `/api/embeddings` embedder) and an isolated
//! in-process SurrealDB. The full `HandleChatCompletion` pipeline runs through
//! the spawned SMOS HTTP server; extraction is a non-blocking background task,
//! so tests poll the store until the expected facts land (or time out).
//!
//! Extraction is enabled by default in this suite (each test builds its own
//! config); the enrichment/passthrough suites keep it disabled to stay fast.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use common::{build_state, enrichment_memory_key, serve_state, unit_embedding_1024};
use serde_json::{Value, json};
use smos_adapters::SystemClock;
use smos_application::ports::{Clock, FactRepository};
use smos_domain::{MemoryKey, SessionId};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

// ---------------------------------------------------------------------------
// Mock builders
// ---------------------------------------------------------------------------

/// Build a SmosConfig pointing at the supplied wiremock servers WITH
/// extraction enabled. Reuses `config_with_mocks` then flips the kill-switch.
fn config_with_extraction(
    upstream: &MockServer,
    ollama: &MockServer,
    reranker: &MockServer,
) -> smos_adapters::config::SmosConfig {
    let mut config = common::config_with_mocks(upstream, ollama, reranker);
    config.server.enable_response_extraction = true;
    config
}

/// Mount a non-streaming upstream reply whose assistant content is `content`.
async fn mount_upstream_content(upstream: &MockServer, content: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-x",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
        })))
        .mount(upstream)
        .await;
}

/// Mount a non-streaming upstream reply carrying tool calls.
async fn mount_upstream_with_tool_call(upstream: &MockServer, content: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-x",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\":\"auth.rs\"}"}
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        })))
        .mount(upstream)
        .await;
}

/// Mount a streaming upstream reply (SSE) carrying `content` across two
/// deltas plus a terminal stop chunk and `[DONE]`.
async fn mount_upstream_streaming(upstream: &MockServer, content: &str) {
    // Split at a char boundary so multibyte UTF-8 inputs do not panic.
    let (first, second) = split_at_char_boundary(content, content.len() / 2);
    let body = format!(
        "data: {chunk1}\n\ndata: {chunk2}\n\ndata: {stop}\n\ndata: [DONE]\n\n",
        chunk1 = json!({"choices":[{"index":0,"delta":{"content":first},"finish_reason":null}]}),
        chunk2 = json!({"choices":[{"index":0,"delta":{"content":second},"finish_reason":null}]}),
        stop = json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}),
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .append_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(upstream)
        .await;
}

/// Split `s` at byte index `mid`, advancing `mid` to the next char boundary so
/// the halves are both valid UTF-8.
fn split_at_char_boundary(s: &str, mut mid: usize) -> (&str, &str) {
    if mid >= s.len() {
        return (s, "");
    }
    while !s.is_char_boundary(mid) {
        mid += 1;
    }
    s.split_at(mid)
}

/// Mount Ollama `/api/chat` returning the supplied bullet-list facts. The mock
/// echoes whatever facts the operator wants the (mock) extractor to "produce".
async fn mount_ollama_chat_facts(ollama: &MockServer, facts: Vec<String>) {
    let body = facts
        .into_iter()
        .map(|f| format!("- {f}"))
        .collect::<Vec<_>>()
        .join("\n");
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "message": {"content": body}
        })))
        .mount(ollama)
        .await;
}

/// Mount Ollama `/api/chat` with a fixed delay (used to prove extraction does
/// not block the client response).
async fn mount_ollama_chat_facts_delayed(ollama: &MockServer, facts: Vec<String>, delay: Duration) {
    let body = facts
        .into_iter()
        .map(|f| format!("- {f}"))
        .collect::<Vec<_>>()
        .join("\n");
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(delay)
                .set_body_json(json!({"message": {"content": body}})),
        )
        .mount(ollama)
        .await;
}

/// Mount Ollama `/api/embeddings` returning a deterministic 1024-dim unit
/// embedding derived from the request prompt. Two calls with the same prompt
/// yield the same vector; two calls with different prompts yield vectors at
/// different axes → cosine similarity 0. This mirrors real embedder behaviour
/// so the semantic-dedup Layer 2 does not collapse unrelated facts (which
/// would otherwise happen if every prompt returned the same constant vector).
async fn mount_ollama_embeddings(ollama: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/api/embeddings"))
        .respond_with(UniqueEmbeddings)
        .mount(ollama)
        .await;
}

/// Embeddings responder that maps each request prompt to a deterministic
/// 1024-dim unit vector. The axis is derived from a non-commutative FNV-1a
/// hash of the prompt so identical prompts always share an embedding
/// (allowing Layer 1 exact match + Layer 2 semantic match to work as in
/// production) while different prompts land on orthogonal axes (cosine 0
/// → no false dedup). FNV-1a is used instead of byte-sum so anagrammatic
/// prompts ("ab" vs "ba") do not collide on the same axis.
struct UniqueEmbeddings;

impl Respond for UniqueEmbeddings {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let prompt = serde_json::from_slice::<serde_json::Value>(&request.body)
            .ok()
            .and_then(|v| v.get("prompt").and_then(|p| p.as_str()).map(str::to_string))
            .unwrap_or_default();
        let axis = (fnv1a_32(prompt.as_bytes()) as usize) % common::EMBEDDING_DIM;
        ResponseTemplate::new(200).set_body_json(json!({
            "embedding": unit_embedding_1024(axis).as_slice().to_vec()
        }))
    }
}

/// 32-bit FNV-1a hash — non-commutative so byte order matters. Cheap and
/// stable; good enough distribution for a test mock.
fn fnv1a_32(bytes: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c9dc5;
    for b in bytes {
        hash ^= *b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// Mount Ollama `/api/chat` as a 500 — drives the retry / give-up paths.
async fn mount_ollama_chat_always_500(ollama: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(500))
        .mount(ollama)
        .await;
}

/// Respond with 500 for the first `fail_times` calls, then 200 with the
/// bullet facts — exercises the §12 retry-then-succeed path end-to-end.
struct FailNTimesThenSucceed {
    fail_times: usize,
    body: String,
    count: AtomicUsize,
}

impl Respond for FailNTimesThenSucceed {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        let n = self.count.fetch_add(1, Ordering::SeqCst);
        if n < self.fail_times {
            ResponseTemplate::new(500)
        } else {
            ResponseTemplate::new(200).set_body_json(json!({
                "message": {"content": self.body.clone()}
            }))
        }
    }
}

async fn mount_ollama_chat_fail_then_succeed(ollama: &MockServer, fail_times: usize, fact: &str) {
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(FailNTimesThenSucceed {
            fail_times,
            body: format!("- {fact}"),
            count: AtomicUsize::new(0),
        })
        .mount(ollama)
        .await;
}

/// A minimal chat-completion request whose user message is too short to
/// trigger enrichment (`"hi"` < `min_topic_chars=3`), so only extraction runs.
fn extraction_request() -> Value {
    json!({
        "model": "origa:gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    })
}

/// A chat-completion request whose history carries a session marker so the
/// detected session is deterministic (for cross-session confirmation tests).
fn extraction_request_with_session(session: &SessionId) -> Value {
    json!({
        "model": "origa:gpt-4o",
        "messages": [
            {"role": "assistant", "content": format!("prior\n<!-- smos:{} -->", session.as_str())},
            {"role": "user", "content": "hi"},
        ],
    })
}

/// Poll the store until at least `expected` pending facts are visible (or the
/// timeout elapses). Extraction is async + spawned, so the test must wait for
/// the background task to land its writes.
async fn wait_for_pending(
    store: &smos_adapters::SurrealStore,
    memory_key: &MemoryKey,
    expected: usize,
    timeout: Duration,
) -> Vec<smos_domain::Fact> {
    let start = Instant::now();
    loop {
        let pending = FactRepository::list_pending(store, memory_key)
            .await
            .expect("list_pending");
        if pending.len() >= expected {
            return pending;
        }
        if start.elapsed() > timeout {
            panic!(
                "timed out after {:?} waiting for {expected} pending facts, got {}",
                start.elapsed(),
                pending.len()
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn send(smos: &str, body: &Value) {
    reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(body)
        .send()
        .await
        .expect("send");
}

fn fixed_session(tag: u8) -> SessionId {
    SessionId::from_raw(&format!("sess_{:012x}", tag as u64)).expect("session id")
}

// ---------------------------------------------------------------------------
// 1. Happy path: pending fact saved
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extraction_saves_pending_facts() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = config_with_extraction(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    mount_upstream_content(&upstream, "TTL=10 prevents the token refresh loop").await;
    mount_ollama_chat_facts(
        &ollama,
        vec!["TTL=10 prevents the token refresh loop".into()],
    )
    .await;
    mount_ollama_embeddings(&ollama).await;

    let smos = serve_state(state.clone()).await;
    send(&smos, &extraction_request()).await;

    let pending = wait_for_pending(
        &state.store,
        &enrichment_memory_key(),
        1,
        Duration::from_secs(8),
    )
    .await;
    assert_eq!(pending.len(), 1);
    assert_eq!(
        pending[0].content(),
        "TTL=10 prevents the token refresh loop"
    );
    assert_eq!(pending[0].status(), smos_domain::FactStatus::Pending);
}

// ---------------------------------------------------------------------------
// 2. Short input skips extraction
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extraction_filters_short_input() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = config_with_extraction(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    // Upstream content "ok" → cleaned "ok" (2 chars < MIN_INPUT_CHARS=15).
    mount_upstream_content(&upstream, "ok").await;
    // The extractor MUST NOT be called.
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"message":{"content":"- should not happen"}})),
        )
        .expect(0)
        .mount(&ollama)
        .await;
    mount_ollama_embeddings(&ollama).await;

    let smos = serve_state(state.clone()).await;
    send(&smos, &extraction_request()).await;
    // Give the background task a moment to (not) run.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let pending = FactRepository::list_pending(&state.store, &enrichment_memory_key())
        .await
        .unwrap();
    assert!(pending.is_empty(), "no facts expected for short input");
}

// ---------------------------------------------------------------------------
// 3. Retries on failure, then succeeds
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extraction_retries_on_failure_then_succeeds() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = config_with_extraction(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    mount_upstream_content(&upstream, "auth.rs uses JWT for token validation").await;
    // Fail twice, then succeed on the 3rd attempt.
    mount_ollama_chat_fail_then_succeed(&ollama, 2, "auth.rs uses JWT for token validation").await;
    mount_ollama_embeddings(&ollama).await;

    let smos = serve_state(state.clone()).await;
    send(&smos, &extraction_request()).await;

    // Backoff is 1 s + 2 s between the 3 attempts → allow generous headroom.
    let pending = wait_for_pending(
        &state.store,
        &enrichment_memory_key(),
        1,
        Duration::from_secs(10),
    )
    .await;
    assert_eq!(pending.len(), 1);
    assert_eq!(
        pending[0].content(),
        "auth.rs uses JWT for token validation"
    );
}

// ---------------------------------------------------------------------------
// 4. Gives up after all attempts fail
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extraction_gives_up_after_all_attempts_fail() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = config_with_extraction(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    mount_upstream_content(&upstream, "a long enough response content here").await;
    mount_ollama_chat_always_500(&ollama).await;
    mount_ollama_embeddings(&ollama).await;

    let smos = serve_state(state.clone()).await;
    send(&smos, &extraction_request()).await;

    // Wait beyond the 3 s retry backoff so the give-up has surely completed.
    tokio::time::sleep(Duration::from_secs(5)).await;
    let pending = FactRepository::list_pending(&state.store, &enrichment_memory_key())
        .await
        .unwrap();
    assert!(
        pending.is_empty(),
        "no facts when every attempt fails (graceful)"
    );
}

// ---------------------------------------------------------------------------
// 5. Cross-session confirmation: provenance grows, no duplicate
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extraction_cross_session_confirmation_unions_provenance() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = config_with_extraction(&upstream, &ollama, &reranker);
    let state = build_state(config).await;

    // Seed a pending fact observed from session 1.
    let fact = smos_domain::Fact::new_pending(
        "TTL=10 prevents the token refresh loop",
        enrichment_memory_key(),
        fixed_session(1),
        unit_embedding_1024(0),
        SystemClock.now(),
        smos_domain::config::ConfidenceConfig::default().base,
    )
    .expect("pending fact");
    let fact_id = fact.id().clone();
    FactRepository::save(&state.store, &fact)
        .await
        .expect("seed");

    mount_upstream_content(&upstream, "TTL=10 prevents the token refresh loop").await;
    mount_ollama_chat_facts(
        &ollama,
        vec!["TTL=10 prevents the token refresh loop".into()],
    )
    .await;
    mount_ollama_embeddings(&ollama).await;

    let smos = serve_state(state.clone()).await;
    let session_b = fixed_session(2);
    send(&smos, &extraction_request_with_session(&session_b)).await;

    // Confirmation is synchronous within the extraction task; poll until the
    // second session lands on the fact's provenance.
    let start = Instant::now();
    let confirmed = loop {
        let fact = FactRepository::get(&state.store, &fact_id, &enrichment_memory_key())
            .await
            .unwrap()
            .expect("fact still present");
        if fact.source_sessions().distinct_count() == 2 {
            break fact;
        }
        if start.elapsed() > Duration::from_secs(8) {
            panic!("confirmation did not complete in time");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    };

    assert_eq!(confirmed.source_sessions().distinct_count(), 2);
    assert!(
        confirmed.source_sessions().contains(&session_b),
        "new session recorded in provenance"
    );
    // Confirmation promotes the fact to Accepted (0.5 base + 0.2 multi-source
    // bonus = 0.7 ≥ accept_threshold) — that is the whole point of
    // cross-session confirmation (§4 `_confirm_existing_fact`). It must NOT,
    // however, create a second fact for the same content.
    assert_eq!(
        confirmed.status(),
        smos_domain::FactStatus::Accepted,
        "two-session confirmation lifts the fact above the accept threshold"
    );
    let accepted: Vec<_> = FactRepository::list_accepted(&state.store, &enrichment_memory_key())
        .await
        .unwrap()
        .into_iter()
        .filter(|f| f.content() == "TTL=10 prevents the token refresh loop")
        .collect();
    assert_eq!(
        accepted.len(),
        1,
        "confirmation must not create a duplicate accepted fact"
    );
}

// ---------------------------------------------------------------------------
// 6. Noise filter strips SMOS markers before extraction
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extraction_noise_filter_strips_markers() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = config_with_extraction(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    // Response content carries SMOS control noise around the real signal.
    let noisy = "auth.rs uses JWT\n<!-- smos:sess_abcdef012345 -->\n<smos-memory session=\"s\">junk</smos-memory>";
    mount_upstream_content(&upstream, noisy).await;
    mount_ollama_chat_facts(&ollama, vec!["auth.rs uses JWT".into()]).await;
    mount_ollama_embeddings(&ollama).await;

    let smos = serve_state(state.clone()).await;
    send(&smos, &extraction_request()).await;

    // Wait for the fact to land — this proves the background extraction task
    // completed (it must have called /api/chat before saving).
    let pending = wait_for_pending(
        &state.store,
        &enrichment_memory_key(),
        1,
        Duration::from_secs(8),
    )
    .await;
    assert_eq!(pending[0].content(), "auth.rs uses JWT");

    // The extraction input sent to Ollama must NOT contain the markers.
    let chat_body = ollama
        .received_requests()
        .await
        .unwrap_or_default()
        .into_iter()
        .find(|r| r.url.path().ends_with("/api/chat"))
        .map(|r| serde_json::from_slice::<Value>(&r.body).expect("chat body json"))
        .expect("extraction /api/chat request recorded");
    let user_content = chat_body["messages"][1]["content"].as_str().unwrap_or("");
    assert!(
        !user_content.contains("smos:"),
        "session marker leaked into extraction input: {user_content}"
    );
    assert!(
        !user_content.contains("smos-memory"),
        "memory block leaked into extraction input: {user_content}"
    );
}

// ---------------------------------------------------------------------------
// 7. Extraction does not block the client response
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extraction_does_not_block_response() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = config_with_extraction(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    mount_upstream_content(&upstream, "Docker image split into three services").await;
    // Extractor takes 2 s — the client response must return well before that.
    mount_ollama_chat_facts_delayed(
        &ollama,
        vec!["Docker image split into three services".into()],
        Duration::from_secs(2),
    )
    .await;
    mount_ollama_embeddings(&ollama).await;

    let smos = serve_state(state).await;
    let start = Instant::now();
    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&extraction_request())
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 200);
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_millis(900),
        "response returned in {elapsed:?} — extraction must not block the client"
    );
}

// ---------------------------------------------------------------------------
// 8. Kill-switch: extraction disabled via config
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extraction_disabled_via_config_skips_pipeline() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let mut config = config_with_extraction(&upstream, &ollama, &reranker);
    config.server.enable_response_extraction = false;
    let state = build_state(config).await;
    mount_upstream_content(&upstream, "a sufficiently long response content").await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"message":{"content":"- x"}})),
        )
        .expect(0)
        .mount(&ollama)
        .await;
    mount_ollama_embeddings(&ollama).await;

    let smos = serve_state(state.clone()).await;
    send(&smos, &extraction_request()).await;
    tokio::time::sleep(Duration::from_millis(200)).await;

    let pending = FactRepository::list_pending(&state.store, &enrichment_memory_key())
        .await
        .unwrap();
    assert!(pending.is_empty(), "no extraction when disabled");
}

// ---------------------------------------------------------------------------
// 9. Tool calls included in the extraction input
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extraction_includes_tool_calls_in_input() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = config_with_extraction(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    mount_upstream_with_tool_call(&upstream, "ran the tool").await;
    mount_ollama_chat_facts(&ollama, vec!["read_file returned auth.rs".into()]).await;
    mount_ollama_embeddings(&ollama).await;

    let smos = serve_state(state.clone()).await;
    send(&smos, &extraction_request()).await;
    // Wait for the fact to land — proves extraction ran (and called /api/chat).
    let _pending = wait_for_pending(
        &state.store,
        &enrichment_memory_key(),
        1,
        Duration::from_secs(8),
    )
    .await;

    let chat_body = ollama
        .received_requests()
        .await
        .unwrap_or_default()
        .into_iter()
        .find(|r| r.url.path().ends_with("/api/chat"))
        .map(|r| serde_json::from_slice::<Value>(&r.body).expect("chat body json"))
        .expect("extraction /api/chat request recorded");
    let user_content = chat_body["messages"][1]["content"].as_str().unwrap_or("");
    assert!(
        user_content.contains("read_file"),
        "tool-call name must reach the extraction input; got: {user_content}"
    );
    assert!(
        user_content.contains("auth.rs"),
        "tool-call arguments must reach the extraction input; got: {user_content}"
    );
}

// ---------------------------------------------------------------------------
// 10. Multiple facts in one response
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extraction_persists_multiple_facts_and_registers_pending() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = config_with_extraction(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    mount_upstream_content(
        &upstream,
        "Several facts about the system architecture follow",
    )
    .await;
    let facts = vec![
        "The API gateway runs on port 8080".to_string(),
        "Postgres is the primary datastore".to_string(),
        "Redis caches session state".to_string(),
        "Nginx terminates TLS".to_string(),
        "The worker pool has four processes".to_string(),
    ];
    mount_ollama_chat_facts(&ollama, facts.clone()).await;
    mount_ollama_embeddings(&ollama).await;

    let smos = serve_state(state.clone()).await;
    send(&smos, &extraction_request()).await;

    let pending = wait_for_pending(
        &state.store,
        &enrichment_memory_key(),
        5,
        Duration::from_secs(8),
    )
    .await;
    let contents: Vec<&str> = pending.iter().map(|f| f.content()).collect();
    for fact in &facts {
        assert!(contents.contains(&fact.as_str()), "missing fact: {fact}");
    }
}

// ---------------------------------------------------------------------------
// 11. Prompt echoes are filtered out of saved facts
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extraction_filters_prompt_echoes() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = config_with_extraction(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    mount_upstream_content(&upstream, "Some real technical content here").await;
    // The mock "extractor" echoes prompt noise AND one real fact.
    let echoed = "Thinking Process: analyze input\n- Do not extract trivial actions\n- real knowledge fact about the system";
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "message": {"content": echoed}
        })))
        .mount(&ollama)
        .await;
    mount_ollama_embeddings(&ollama).await;

    let smos = serve_state(state.clone()).await;
    send(&smos, &extraction_request()).await;

    let pending = wait_for_pending(
        &state.store,
        &enrichment_memory_key(),
        1,
        Duration::from_secs(8),
    )
    .await;
    assert_eq!(
        pending.len(),
        1,
        "only the real fact survives echo filtering"
    );
    assert_eq!(pending[0].content(), "real knowledge fact about the system");
}

// ---------------------------------------------------------------------------
// 12. Streaming: pending fact saved after [DONE]
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extraction_streaming_saves_pending_fact() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = config_with_extraction(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    mount_upstream_streaming(&upstream, "TTL=10 prevents the token refresh loop").await;
    mount_ollama_chat_facts(
        &ollama,
        vec!["TTL=10 prevents the token refresh loop".into()],
    )
    .await;
    mount_ollama_embeddings(&ollama).await;

    let smos = serve_state(state.clone()).await;
    let body = json!({
        "model": "origa:gpt-4o",
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}],
    });
    send(&smos, &body).await;

    let pending = wait_for_pending(
        &state.store,
        &enrichment_memory_key(),
        1,
        Duration::from_secs(8),
    )
    .await;
    assert_eq!(pending.len(), 1);
    assert_eq!(
        pending[0].content(),
        "TTL=10 prevents the token refresh loop"
    );
}

// ---------------------------------------------------------------------------
// 13. Streaming buffer concatenates content deltas
// ---------------------------------------------------------------------------

#[tokio::test]
async fn extraction_streaming_concatenates_content_deltas() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = config_with_extraction(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    // The full content split across chunks; extraction input must be whole.
    mount_upstream_streaming(&upstream, "auth.rs uses JWT for token validation").await;
    mount_ollama_chat_facts(
        &ollama,
        vec!["auth.rs uses JWT for token validation".into()],
    )
    .await;
    mount_ollama_embeddings(&ollama).await;

    let smos = serve_state(state.clone()).await;
    let body = json!({
        "model": "origa:gpt-4o",
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}],
    });
    send(&smos, &body).await;
    // Wait for the fact to land — proves the concatenated input passed
    // MIN_INPUT_CHARS and extraction completed.
    let _pending = wait_for_pending(
        &state.store,
        &enrichment_memory_key(),
        1,
        Duration::from_secs(8),
    )
    .await;

    let chat_body = ollama
        .received_requests()
        .await
        .unwrap_or_default()
        .into_iter()
        .find(|r| r.url.path().ends_with("/api/chat"))
        .map(|r| serde_json::from_slice::<Value>(&r.body).expect("chat body json"))
        .expect("extraction /api/chat request recorded");
    let user_content = chat_body["messages"][1]["content"].as_str().unwrap_or("");
    assert!(
        user_content.contains("auth.rs uses JWT for token validation"),
        "streaming content must be reassembled before extraction; got: {user_content}"
    );
}

// ---------------------------------------------------------------------------
// 14. Full pipeline smoke: enrich (empty) + extract + save
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_pipeline_smoke_extraction_runs_after_passthrough() {
    let upstream = MockServer::start().await;
    let ollama = MockServer::start().await;
    let reranker = MockServer::start().await;

    let config = config_with_extraction(&upstream, &ollama, &reranker);
    let state = build_state(config).await;
    mount_upstream_content(&upstream, "Prometheus scrapes metrics on port 9090").await;
    mount_ollama_chat_facts(
        &ollama,
        vec!["Prometheus scrapes metrics on port 9090".into()],
    )
    .await;
    mount_ollama_embeddings(&ollama).await;

    let smos = serve_state(state.clone()).await;
    let resp = reqwest::Client::new()
        .post(format!("{smos}/v1/chat/completions"))
        .json(&extraction_request())
        .send()
        .await
        .expect("send");
    assert_eq!(resp.status(), 200);
    // Response carries the session marker (full passthrough still works).
    let body: Value = resp.json().await.expect("body json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        content.contains("<!-- smos:"),
        "session marker present in response"
    );

    let pending = wait_for_pending(
        &state.store,
        &enrichment_memory_key(),
        1,
        Duration::from_secs(8),
    )
    .await;
    assert_eq!(pending.len(), 1);
    assert_eq!(
        pending[0].content(),
        "Prometheus scrapes metrics on port 9090"
    );
}
