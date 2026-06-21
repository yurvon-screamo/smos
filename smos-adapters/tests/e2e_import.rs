//! E2E: opencode import pipeline.
//!
//! Each test wires a fresh `SurrealStore` (RocksDB tempdir) with mock Ollama
//! HTTP servers for the embedder + extractor, builds an
//! `ImportOpencodeSession` use case, and asserts on the resulting store
//! state. The opencode discovery HTTP probe is exercised against wiremock
//! endpoints returning synthetic opencode-shape responses; the CLI fallback
//! path is covered by unit-level helpers (it requires the `opencode` binary
//! on PATH, so an integration test would be flaky).

mod common;

use std::sync::Arc;

use serde_json::{Value, json};
use surrealdb::Surreal;
use surrealdb::engine::local::RocksDb;
use tempfile::TempDir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use smos_adapters::TokioDelay;
use smos_adapters::config::SmosConfig;
use smos_adapters::opencode::{self, SessionSource};
use smos_adapters::{OllamaEmbedding, OllamaExtractor, SurrealStore, SystemClock};
use smos_application::ports::FactRepository;
use smos_application::use_cases::ImportOpencodeSession;
use smos_domain::{MemoryKey, SessionId};

use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Fixture loader — `tests/fixtures/sample_transcript.json` is checked into
// the repo so every test parses the same canonical synthetic transcript.
// ---------------------------------------------------------------------------

/// Read the in-repo sample transcript fixture as a `serde_json::Value`.
fn sample_transcript() -> Value {
    let raw = include_str!("fixtures/sample_transcript.json");
    serde_json::from_str(raw).expect("sample_transcript.json parses")
}

// ---------------------------------------------------------------------------
// Harness — fresh SurrealStore + mock Ollama providers per test.
// ---------------------------------------------------------------------------

/// Build a fresh store in an isolated RocksDB tempdir + run migrations.
async fn fresh_store() -> (SurrealStore, TempDir) {
    let tmp = TempDir::new().expect("tempdir");
    let db_path = tmp.path().join("smos.db");
    let db = Surreal::new::<RocksDb>(db_path.to_string_lossy().to_string())
        .await
        .expect("rocksdb");
    db.use_ns("test").use_db("test").await.expect("use ns/db");
    let store = SurrealStore::from_client(db);
    store.run_migrations().await.expect("migrations");
    (store, tmp)
}

/// Build the import use case wired to mock Ollama endpoints at `ollama.uri()`.
async fn build_import(
    store: SurrealStore,
    ollama_uri: String,
) -> ImportOpencodeSession<
    SurrealStore,
    SurrealStore,
    OllamaEmbedding,
    OllamaExtractor,
    SystemClock,
    TokioDelay,
> {
    let extraction_cfg = smos_adapters::config::LlmExtractionConfig {
        url: ollama_uri.clone(),
        timeout_seconds: 5,
        ..smos_adapters::config::LlmExtractionConfig::default()
    };
    let embedding_cfg = smos_adapters::config::EmbeddingConfig {
        url: ollama_uri,
        timeout_seconds: 5,
        ..smos_adapters::config::EmbeddingConfig::default()
    };
    let embedder = OllamaEmbedding::new(Arc::new(embedding_cfg)).expect("embedder");
    let extractor = OllamaExtractor::new(Arc::new(extraction_cfg)).expect("extractor");
    let confidence_cfg = Arc::new(SmosConfig::default().confidence);
    let extraction_cfg = Arc::new(SmosConfig::default().extraction);

    ImportOpencodeSession {
        facts: store.clone(),
        sessions: store,
        embedder,
        extractor,
        clock: SystemClock,
        delay: TokioDelay,
        confidence_cfg,
        extraction_cfg,
        enable_response_extraction: true,
        min_chars: 15,
    }
}

/// Mount Ollama `/api/chat` returning the supplied bullet-list facts.
async fn mount_chat_facts(ollama: &MockServer, facts: Vec<String>) {
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

/// Mount Ollama `/api/embeddings` returning the SAME 1024-dim unit
/// embedding (axis 0) for every call. Use this for single-fact imports
/// where semantic dedup is irrelevant.
async fn mount_embeddings(ollama: &MockServer) {
    let vector = common::unit_embedding_1024(0).as_slice().to_vec();
    Mock::given(method("POST"))
        .and(path("/api/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"embedding": vector})))
        .mount(ollama)
        .await;
}

/// Stateful `/api/embeddings` responder that returns a distinct 1024-dim
/// unit embedding (axis 0, 1, 2, …) on successive calls, then clamps to the
/// last axis. Multi-fact imports require per-fact embeddings that are
/// orthogonal — otherwise the persist-facts Layer 2 semantic dedup sees
/// cosine similarity = 1.0 between every pair and collapses all extracted
/// facts into the first one.
struct DistinctUnitEmbeddings {
    counter: Arc<AtomicUsize>,
    max_axes: usize,
}

impl DistinctUnitEmbeddings {
    /// Build a responder with `max_axes` distinct orthogonal embeddings.
    /// `max_axes == 0` is rejected up front: every caller must declare the
    /// exact number of distinct facts it expects to embed (≥ 1), and a zero
    /// would silently degrade back to the all-equal-embeddings trap that
    /// caused `facts_extracted` to collapse to 1.
    fn new(max_axes: usize) -> Self {
        assert!(
            max_axes >= 1,
            "DistinctUnitEmbeddings needs at least one axis to vary"
        );
        Self {
            counter: Arc::new(AtomicUsize::new(0)),
            max_axes,
        }
    }
}

impl Respond for DistinctUnitEmbeddings {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        let idx = self.counter.fetch_add(1, Ordering::SeqCst);
        // Two independent guarantees combine here:
        //   - `assert!` in `new` enforces the SEMANTIC invariant
        //     `max_axes >= 1` (≥ 1 distinct embedding), without which every
        //     fact would collapse back to the all-equal-embeddings trap.
        //   - `saturating_sub` is an ARITHMETIC safety net: it never
        //     underflows (clamps to 0), so even a future caller that
        //     bypasses the constructor could not introduce a panic.
        // `min` then clamps every call past the last axis to the last
        // declared embedding.
        let axis = idx.min(self.max_axes.saturating_sub(1));
        let vector = common::unit_embedding_1024(axis).as_slice().to_vec();
        ResponseTemplate::new(200).set_body_json(json!({"embedding": vector}))
    }
}

/// Mount Ollama `/api/embeddings` returning a distinct orthogonal unit
/// embedding for each successive call, so each extracted fact gets its own
/// vector and the persist-facts Layer 2 dedup does not collapse them.
async fn mount_distinct_embeddings(ollama: &MockServer, fact_count: usize) {
    Mock::given(method("POST"))
        .and(path("/api/embeddings"))
        .respond_with(DistinctUnitEmbeddings::new(fact_count))
        .mount(ollama)
        .await;
}

fn memory_key() -> MemoryKey {
    MemoryKey::from_raw("testproj").expect("memory key")
}

fn session_id(tag: u8) -> SessionId {
    SessionId::from_raw(&format!("sess_{:012x}", tag as u64)).expect("session id")
}

// ---------------------------------------------------------------------------
// Parser + import integration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn parse_synthetic_transcript_drops_user_and_keeps_assistants() {
    let transcript = sample_transcript();

    let turns = opencode::parse_transcript(&transcript);

    // msg_1 is user → dropped. msg_2..msg_5 are assistant → kept. 4 turns.
    assert_eq!(turns.len(), 4);
    assert_eq!(turns[0].message_id, "msg_2");
    assert_eq!(turns[0].agent, "head-of-development");
    // Reasoning part must not surface in content.
    assert!(turns[0].content.starts_with("TTL=10"));
    // Tool call present on msg_2.
    assert_eq!(turns[0].tool_calls.len(), 1);
    assert_eq!(turns[0].tool_calls[0].name, "read_file");
}

#[tokio::test]
async fn import_saves_pending_facts_from_assistant_turns() {
    let ollama = MockServer::start().await;
    let (store, _tmp) = fresh_store().await;
    mount_chat_facts(
        &ollama,
        vec!["TTL=10 prevents the token refresh loop".to_string()],
    )
    .await;
    mount_embeddings(&ollama).await;

    let transcript = sample_transcript();
    // Take only the first assistant turn (msg_2) so the scripted extractor
    // response count lines up.
    let mut turns = opencode::parse_transcript(&transcript);
    turns.truncate(1);

    let import = build_import(store.clone(), ollama.uri()).await;
    let stats = import
        .execute(turns, &memory_key(), &session_id(1), None)
        .await
        .expect("import");

    assert_eq!(stats.turns_processed, 1);
    assert_eq!(stats.facts_extracted, 1);

    let pending = FactRepository::list_pending(&store, &memory_key())
        .await
        .unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(
        pending[0].content(),
        "TTL=10 prevents the token refresh loop"
    );
}

#[tokio::test]
async fn import_filters_turns_below_min_chars_without_tool_calls() {
    let ollama = MockServer::start().await;
    let (store, _tmp) = fresh_store().await;
    // Only one extraction result is scripted; msg_5 ("ok", 2 chars) MUST be
    // skipped so it does not consume the single response.
    mount_chat_facts(&ollama, vec!["from msg_2 long content".to_string()]).await;
    mount_embeddings(&ollama).await;

    let transcript = sample_transcript();
    let mut turns = opencode::parse_transcript(&transcript);
    // Keep msg_2 (long text + tool call) and msg_5 ("ok", short).
    turns.drain(1..3); // drop msg_3 + msg_4, keep msg_2 and msg_5

    let import = build_import(store.clone(), ollama.uri()).await;
    let stats = import
        .execute(turns, &memory_key(), &session_id(1), None)
        .await
        .expect("import");

    assert_eq!(stats.turns_processed, 1);
    assert_eq!(stats.turns_skipped, 1);
    assert_eq!(stats.facts_extracted, 1);
}

#[tokio::test]
async fn import_applies_agent_filter() {
    let ollama = MockServer::start().await;
    let (store, _tmp) = fresh_store().await;
    // Two extraction results: one for each head-of-development turn (msg_2,
    // msg_3); msg_4 (dreaming) is filtered out and must NOT consume a result.
    mount_chat_facts(
        &ollama,
        vec!["from msg_2".to_string(), "from msg_3".to_string()],
    )
    .await;
    // Two extracted facts → two distinct orthogonal embeddings so the
    // persist-facts Layer 2 semantic dedup does not collapse them.
    mount_distinct_embeddings(&ollama, 2).await;

    let transcript = sample_transcript();
    // Keep msg_2 (hod), msg_3 (hod), msg_4 (dreaming).
    let mut turns = opencode::parse_transcript(&transcript);
    turns.truncate(3);

    let import = build_import(store.clone(), ollama.uri()).await;
    let filter = vec!["head-of-development".to_string()];
    let stats = import
        .execute(turns, &memory_key(), &session_id(1), Some(&filter))
        .await
        .expect("import");

    assert_eq!(stats.turns_processed, 2);
    assert_eq!(stats.turns_skipped, 1);
    assert_eq!(stats.facts_extracted, 2);
}

#[tokio::test]
async fn import_offset_limit_window() {
    let ollama = MockServer::start().await;
    let (store, _tmp) = fresh_store().await;
    mount_chat_facts(&ollama, vec!["from msg_3".to_string()]).await;
    mount_embeddings(&ollama).await;

    let transcript = sample_transcript();
    let mut turns = opencode::parse_transcript(&transcript);
    // Equivalent to `--offset 1 --limit 1`: drop msg_2, take only msg_3.
    turns.drain(..1);
    turns.truncate(1);

    let import = build_import(store.clone(), ollama.uri()).await;
    let stats = import
        .execute(turns, &memory_key(), &session_id(1), None)
        .await
        .expect("import");

    assert_eq!(stats.turns_processed, 1);
    assert_eq!(stats.facts_extracted, 1);
}

#[tokio::test]
async fn import_includes_tool_calls_in_extraction_input() {
    let ollama = MockServer::start().await;
    let (store, _tmp) = fresh_store().await;
    // The extractor mock echoes a fact whose content proves it was reached at
    // all — verifying that the tool-call trailer reaches the model is the job
    // of `extract_facts_from_response` unit tests; here we only assert the
    // turn with tool_calls was processed.
    mount_chat_facts(&ollama, vec!["auth.rs uses JWT".to_string()]).await;
    mount_embeddings(&ollama).await;

    // Build a synthetic turn with tool calls and very short text — without the
    // tool_calls it would be filtered by min_chars.
    let turn = smos_application::use_cases::import_opencode_session::AssistantTurn {
        message_id: "msg_tool".into(),
        agent: "head-of-development".into(),
        content: "ok".into(),
        tool_calls: vec![smos_domain::chat::ToolCall {
            name: "read_file".into(),
            arguments: smos_domain::chat::ToolArguments::from_json(r#"{"path":"auth.rs"}"#),
        }],
    };

    let import = build_import(store.clone(), ollama.uri()).await;
    let stats = import
        .execute(vec![turn], &memory_key(), &session_id(1), None)
        .await
        .expect("import");

    assert_eq!(
        stats.turns_processed, 1,
        "tool-call turn kept despite short text"
    );
    assert_eq!(stats.facts_extracted, 1);
}

#[tokio::test]
async fn import_reuses_extract_facts_use_case_for_dry_run_safety() {
    // DRY verification: ImportOpencodeSession delegates to
    // ExtractFactsFromResponse, so the SAME pending state shape appears
    // after import as after a single live response extraction. The fixture
    // wires the same providers and asserts the resulting fact is Pending with
    // the right source session.
    let ollama = MockServer::start().await;
    let (store, _tmp) = fresh_store().await;
    mount_chat_facts(&ollama, vec!["the canonical fact".to_string()]).await;
    mount_embeddings(&ollama).await;

    let transcript = sample_transcript();
    let mut turns = opencode::parse_transcript(&transcript);
    turns.truncate(1);

    let sid = session_id(7);
    let import = build_import(store.clone(), ollama.uri()).await;
    let _ = import
        .execute(turns, &memory_key(), &sid, None)
        .await
        .expect("import");

    let pending = FactRepository::list_pending(&store, &memory_key())
        .await
        .unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].source_sessions().distinct_count(), 1);
    // The pending fact was registered under the import's session id.
    assert_eq!(pending[0].source_sessions().iter().next().unwrap(), &sid);
}

#[tokio::test]
async fn import_is_idempotent_on_repeated_runs() {
    // Cross-session confirmation: importing the same fact from a second
    // session must NOT add a duplicate. The original fact's provenance grows
    // instead (and may promote through the validation gate to Accepted — that
    // is the canonical "multi-source confirmation" path).
    let ollama = MockServer::start().await;
    let (store, _tmp) = fresh_store().await;
    let shared_fact = "shared knowledge fact across imports";
    // Each turn re-observes the same fact — extractor returns it for every
    // turn. Two sessions observe it once each.
    mount_chat_facts(&ollama, vec![shared_fact.to_string()]).await;
    mount_embeddings(&ollama).await;

    let turn = smos_application::use_cases::import_opencode_session::AssistantTurn {
        message_id: "msg".into(),
        agent: "head-of-development".into(),
        content: shared_fact.into(),
        tool_calls: vec![],
    };

    // First import — new pending fact.
    let first = build_import(store.clone(), ollama.uri()).await;
    let _ = first
        .execute(vec![turn.clone()], &memory_key(), &session_id(1), None)
        .await
        .unwrap();
    let fact_id = smos_domain::FactId::from_content(shared_fact);
    let after_first = FactRepository::get(&store, &fact_id, &memory_key())
        .await
        .unwrap()
        .expect("fact stored on first import");
    assert_eq!(
        after_first.source_sessions().distinct_count(),
        1,
        "first import registers one session"
    );

    // Second import from a different session — confirmation, not a new fact.
    let second = build_import(store.clone(), ollama.uri()).await;
    let stats = second
        .execute(vec![turn], &memory_key(), &session_id(2), None)
        .await
        .unwrap();
    assert_eq!(
        stats.facts_extracted, 0,
        "re-observation must confirm, not duplicate"
    );

    let after_second = FactRepository::get(&store, &fact_id, &memory_key())
        .await
        .unwrap()
        .expect("fact still present after second import");
    assert_eq!(
        after_second.source_sessions().distinct_count(),
        2,
        "provenance grew to two sessions"
    );
}

// ---------------------------------------------------------------------------
// HTTP discovery — wiremock as fake opencode server
// ---------------------------------------------------------------------------

#[tokio::test]
async fn discovery_probe_finds_mock_opencode_on_custom_port() {
    // Bind a wiremock on an ephemeral port, then run `probe_ports` against a
    // port list that contains only that port. This exercises the same
    // parallel `join_all` + first-alive-wins logic `probe_http` uses, without
    // binding the canonical `DEFAULT_PORTS` (which risks flakiness when those
    // ports are already in use on the test host).
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status": "ok"})))
        .mount(&server)
        .await;
    let port = server.address().port();

    let client = reqwest::Client::new();
    let source = opencode::probe_ports(&client, &[port]).await;
    assert!(
        matches!(source, Some(SessionSource::Http { .. })),
        "probe_ports must find the mocked opencode server"
    );
    assert_eq!(source.unwrap().kind_str(), "http");
}

#[tokio::test]
async fn discovery_probe_returns_none_when_no_port_is_alive() {
    // Port 1 is privileged — no user-space process can bind on most OSes, so
    // the probe's TCP connect fails fast. The probe must return None without
    // hanging.
    let client = reqwest::Client::new();
    let source = opencode::probe_ports(&client, &[1]).await;
    assert!(
        source.is_none(),
        "probe_ports must return None on dead port"
    );
}

#[tokio::test]
async fn discovery_probe_rejects_ollama_error_envelope() {
    // Bind a mock returning an Ollama-style 200 + error envelope, then prove
    // probe_ports rejects it (looks_alive is false). This is the full
    // `probe_one_port` path, not just the helper.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"error": "model 'x' not found"})),
        )
        .mount(&server)
        .await;
    let port = server.address().port();

    let client = reqwest::Client::new();
    let source = opencode::probe_ports(&client, &[port]).await;
    assert!(
        source.is_none(),
        "probe_ports must reject Ollama-style error envelope"
    );
}

#[tokio::test]
async fn discovery_resolve_source_with_explicit_port_bypasses_probe() {
    // `--port <port>` short-circuits the probe; no /health mock is needed.
    let server = MockServer::start().await;
    let port = server.address().port();

    let client = reqwest::Client::new();
    let source = opencode::resolve_source(&client, Some(port)).await;
    assert!(matches!(source, SessionSource::Http { .. }));
    assert_eq!(source.kind_str(), "http");
}

#[tokio::test]
async fn discovery_fetch_session_export_assembles_info_and_messages() {
    let server = MockServer::start().await;
    let port = server.address().port();
    Mock::given(method("GET"))
        .and(path("/session/ses_abc"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "ses_abc", "title": "demo"
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/session/ses_abc/message"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "hi"}]}
        ])))
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let source = SessionSource::Http { port };
    let transcript = opencode::fetch_session_export(&source, &client, "ses_abc")
        .await
        .expect("fetch");
    assert_eq!(transcript["info"]["title"], "demo");
    let turns = opencode::parse_transcript(&transcript);
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].content, "hi");
}

#[tokio::test]
async fn discovery_list_sessions_unwraps_envelope() {
    let server = MockServer::start().await;
    let port = server.address().port();
    Mock::given(method("GET"))
        .and(path("/session"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "sessions": [
                {"id": "ses_a", "title": "first"},
                {"id": "ses_b", "title": "second"}
            ]
        })))
        .mount(&server)
        .await;

    let client = reqwest::Client::new();
    let source = SessionSource::Http { port };
    let sessions = opencode::list_sessions(&source, &client)
        .await
        .expect("list");
    assert_eq!(sessions.len(), 2);
    assert_eq!(sessions[0]["id"], "ses_a");
}

// ---------------------------------------------------------------------------
// --from-file mode (parse-only integration through the import use case)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn from_file_mode_parses_without_discovery() {
    // The fixture transcript is loaded directly from disk — no HTTP probe, no
    // CLI subprocess. This is the path `smos import --from-file` uses for
    // offline / CI imports.
    let transcript = sample_transcript();
    let turns = opencode::parse_transcript(&transcript);
    assert!(!turns.is_empty(), "fixture must yield at least one turn");

    // Sanity: the first turn is the head-of-development assistant message.
    assert_eq!(turns[0].agent, "head-of-development");
}

// ---------------------------------------------------------------------------
// Full pipeline: parse → import → store
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_pipeline_imports_fixture_into_store() {
    let ollama = MockServer::start().await;
    let (store, _tmp) = fresh_store().await;
    // Scripted extractor: returns one fact per assistant turn (4 turns in the
    // fixture minus msg_5 which is below min_chars and has no tool_calls →
    // 3 processed turns).
    mount_chat_facts(
        &ollama,
        vec![
            "fact from msg_2".to_string(),
            "fact from msg_3".to_string(),
            "fact from msg_4".to_string(),
        ],
    )
    .await;
    // Three extracted facts → three distinct orthogonal embeddings so the
    // persist-facts Layer 2 semantic dedup does not collapse them.
    mount_distinct_embeddings(&ollama, 3).await;

    let transcript = sample_transcript();
    let turns = opencode::parse_transcript(&transcript);

    let import = build_import(store.clone(), ollama.uri()).await;
    let stats = import
        .execute(turns, &memory_key(), &session_id(1), None)
        .await
        .expect("import");

    assert_eq!(stats.turns_processed, 3);
    assert_eq!(stats.turns_skipped, 1, "msg_5 (ok) skipped");
    assert_eq!(stats.facts_extracted, 3);

    // Import is synchronous (no background spawn), so by the time `execute`
    // returns every successful extraction has been persisted. A direct
    // `list_pending` reads the same rows the use case just wrote.
    let pending = FactRepository::list_pending(&store, &memory_key())
        .await
        .unwrap();
    assert_eq!(pending.len(), 3);
}
