use super::*;
use smos_domain::{Embedding, FactStatus, NewPendingRequest};

#[tokio::test]
async fn execute_disabled_returns_zero_without_calling_extractor() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();
    let extractor = ScriptedExtractor::new(vec![Err(ProviderError::Unavailable(
        "must not be called".into(),
    ))]);
    let fix = Fix::new();
    let mut uc = build(
        &facts,
        &sessions,
        &extractor,
        &fix.embedder,
        &fix.clock,
        &fix.cfg,
        &fix.extraction_cfg,
    );
    uc.enable_response_extraction = false;

    let n = uc
        .execute("", "TTL=10 prevents refresh loop", &[], &mk(), &sid(1))
        .await
        .unwrap();
    assert_eq!(n, 0);
}

#[tokio::test]
async fn execute_short_input_returns_zero_without_calling_extractor() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();
    let extractor = ScriptedExtractor::new(vec![Err(ProviderError::Unavailable(
        "must not be called".into(),
    ))]);
    let fix = Fix::new();
    let uc = build(
        &facts,
        &sessions,
        &extractor,
        &fix.embedder,
        &fix.clock,
        &fix.cfg,
        &fix.extraction_cfg,
    );

    // "ok" is 2 chars < MIN_INPUT_CHARS (15).
    let n = uc.execute("", "ok", &[], &mk(), &sid(1)).await.unwrap();
    assert_eq!(n, 0);
}

#[tokio::test]
async fn execute_saves_new_pending_fact_and_registers_it() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();
    let extractor = ScriptedExtractor::new(vec![Ok(vec![
        "TTL=10 prevents the token refresh loop".to_string(),
    ])]);
    let fix = Fix::new();
    let uc = build(
        &facts,
        &sessions,
        &extractor,
        &fix.embedder,
        &fix.clock,
        &fix.cfg,
        &fix.extraction_cfg,
    );

    let n = uc
        .execute(
            "",
            "we changed TTL to 10 to stop the loop",
            &[],
            &mk(),
            &sid(1),
        )
        .await
        .unwrap();

    assert_eq!(n, 1);
    let fact = facts
        .get_clone(&FactId::from_content(
            "TTL=10 prevents the token refresh loop",
        ))
        .expect("fact saved");
    assert_eq!(fact.status(), FactStatus::Pending);
    assert_eq!(
        sessions.pending.lock().unwrap().len(),
        1,
        "fact registered on session pending list"
    );
}

#[tokio::test]
async fn execute_unavailable_extractor_skips_gracefully() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();
    // A single Unavailable result: the use case must return Ok(0)
    // immediately WITHOUT retrying. `call_count == 1` (not 3) is the
    // invariant that proves the early-exit on Unavailable.
    let extractor = ScriptedExtractor::new(vec![Err(ProviderError::Unavailable(
        "connection refused".into(),
    ))]);
    let fix = Fix::new();
    let uc = build(
        &facts,
        &sessions,
        &extractor,
        &fix.embedder,
        &fix.clock,
        &fix.cfg,
        &fix.extraction_cfg,
    );

    let n = uc
        .execute("", "some real content long enough", &[], &mk(), &sid(1))
        .await
        .unwrap();
    assert_eq!(n, 0);
    assert_eq!(
        extractor.call_count(),
        1,
        "Unavailable must skip retries — extractor called exactly once"
    );
    assert!(facts.is_empty(), "no fact persisted on graceful skip");
}

#[tokio::test]
async fn execute_retries_on_request_failed_then_succeeds() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();
    let extractor = ScriptedExtractor::new(vec![
        Err(ProviderError::RequestFailed("500".into())),
        Err(ProviderError::RequestFailed("500".into())),
        Ok(vec!["auth.rs uses JWT for tokens".to_string()]),
    ]);
    let fix = Fix::new();
    let uc = build(
        &facts,
        &sessions,
        &extractor,
        &fix.embedder,
        &fix.clock,
        &fix.cfg,
        &fix.extraction_cfg,
    );

    let n = uc
        .execute("", "the auth module uses JWT", &[], &mk(), &sid(1))
        .await
        .unwrap();
    assert_eq!(n, 1);
}

#[tokio::test]
async fn execute_gives_up_after_all_attempts_fail() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();
    let extractor = ScriptedExtractor::new(vec![
        Err(ProviderError::RequestFailed("500".into())),
        Err(ProviderError::RequestFailed("500".into())),
        Err(ProviderError::RequestFailed("500".into())),
    ]);
    let fix = Fix::new();
    let uc = build(
        &facts,
        &sessions,
        &extractor,
        &fix.embedder,
        &fix.clock,
        &fix.cfg,
        &fix.extraction_cfg,
    );

    let result = uc
        .execute("", "content long enough to pass gate", &[], &mk(), &sid(1))
        .await;
    assert!(result.is_err(), "final failure propagates as Err");
    assert!(facts.is_empty());
}

#[tokio::test]
async fn execute_strips_smos_noise_before_extraction() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();
    let extractor = ScriptedExtractor::new(vec![Ok(vec!["a clean fact".to_string()])]);
    let fix = Fix::new();
    let uc = build(
        &facts,
        &sessions,
        &extractor,
        &fix.embedder,
        &fix.clock,
        &fix.cfg,
        &fix.extraction_cfg,
    );

    let content = "real content about the deployment\n<!-- smos:sess_abcdef012345 -->\n<smos-memory session=\"s\">x</smos-memory>";
    let n = uc.execute("", content, &[], &mk(), &sid(1)).await.unwrap();
    assert_eq!(n, 1);
}

#[tokio::test]
async fn execute_cross_session_confirms_existing_fact() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();

    // Seed a fact from session 1.
    let first = Fact::new_pending(NewPendingRequest {
        content: "shared fact content here",
        memory_key: mk(),
        session: sid(1),
        embedding: Embedding::new(vec![1.0]).unwrap(),
        extracted_at: Timestamp::from_unix_secs(1_700_000_000).unwrap(),
        base_confidence: ConfidenceConfig::default().base,
    })
    .unwrap();
    let fid = first.id().clone();
    facts.seed(first);

    let extractor = ScriptedExtractor::new(vec![Ok(vec!["shared fact content here".to_string()])]);
    let fix = Fix::new();
    let uc = build(
        &facts,
        &sessions,
        &extractor,
        &fix.embedder,
        &fix.clock,
        &fix.cfg,
        &fix.extraction_cfg,
    );

    // Same fact observed from session 2 → confirmation, not a new fact.
    let n = uc
        .execute("", "shared fact content here", &[], &mk(), &sid(2))
        .await
        .unwrap();
    assert_eq!(n, 0, "confirmation does not count as a new fact");

    let confirmed = facts.get_clone(&fid).expect("fact still present");
    assert_eq!(
        confirmed.source_sessions().distinct_count(),
        2,
        "provenance grew to two sessions"
    );
    assert!(
        sessions.pending.lock().unwrap().is_empty(),
        "confirmation must not register on the pending list"
    );
}

/// The extraction pipeline MUST hand distinct embeddings to distinct
/// extracted facts — otherwise Layer 2 dedup would collapse two
/// unrelated facts (cosine ~1) and silently lose data. The
/// `RecordingEmbedder` returns a content-derived one-hot vector so
/// two distinct facts end up with cosine similarity 0; this test
/// pins that contract by checking the recorded calls + the resulting
/// store state.
#[tokio::test]
async fn recording_embedder_yields_distinct_vectors_for_distinct_facts() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();
    let extractor = ScriptedExtractor::new(vec![Ok(vec![
        "alpha configuration directive".to_string(),
        "beta configuration directive".to_string(),
    ])]);
    let (embedder, calls) = RecordingEmbedder::new();
    let clock = clock();
    let cfg = cfg();
    let extraction_cfg = extraction_cfg();
    let uc = build_with_recording_embedder(
        &facts,
        &sessions,
        &extractor,
        &embedder,
        &clock,
        &cfg,
        &extraction_cfg,
    );

    let n = uc
        .execute("", "content covering both directives", &[], &mk(), &sid(1))
        .await
        .unwrap();
    assert_eq!(n, 2, "two distinct facts persisted");
    assert_eq!(
        calls.lock().unwrap().len(),
        2,
        "embedder called once per extracted fact"
    );
    // Two distinct FactIds in the store → no collapse happened.
    let id_a = FactId::from_content("alpha configuration directive");
    let id_b = FactId::from_content("beta configuration directive");
    assert!(facts.contains(&id_a));
    assert!(facts.contains(&id_b));
}

// -----------------------------------------------------------------------
// Empty raw fact in extraction batch — must not crash
// -----------------------------------------------------------------------

/// An empty string in the extracted facts list surfaces as `Err` —
/// the pipeline propagates the underlying domain failure rather than
/// silently dropping the empty entry or persisting a malformed fact.
/// The whole batch fails: the call site (background extraction task)
/// logs the error and the facts that would have been persisted in
/// the same batch are lost too. A future refactor that filters
/// empty raw facts BEFORE the `Fact::new_pending` constructor would
/// change this test from `is_err()` to `n == 1`; that change is
/// intentional and the test should be updated alongside it.
#[tokio::test]
async fn execute_propagates_err_when_batch_contains_empty_raw_fact() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();
    // Mix one empty + one real fact in the extractor output.
    let extractor = ScriptedExtractor::new(vec![Ok(vec![
        String::new(),
        "real fact that should still persist".to_string(),
    ])]);
    let fix = Fix::new();
    let uc = build(
        &facts,
        &sessions,
        &extractor,
        &fix.embedder,
        &fix.clock,
        &fix.cfg,
        &fix.extraction_cfg,
    );

    let result = uc
        .execute(
            "",
            "content long enough to clear MIN_INPUT_CHARS",
            &[],
            &mk(),
            &sid(1),
        )
        .await;
    assert!(
        result.is_err(),
        "empty raw fact must surface as Err (the only safe non-silent path)"
    );
}

/// A long user message MUST NOT let a short assistant reply through the gate.
/// This pins the gate-on-assistant-signal invariant: the gate considers only
/// the cleaned assistant content + tool calls, never the user message. A future
/// regression that moves the gate onto the combined input would let this case
/// through and call the extractor — this test would catch it.
#[tokio::test]
async fn execute_long_user_message_does_not_let_short_assistant_reply_through_gate() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();
    let extractor = ScriptedExtractor::new(vec![Err(ProviderError::Unavailable(
        "must not be called".into(),
    ))]);
    let fix = Fix::new();
    let uc = build(
        &facts,
        &sessions,
        &extractor,
        &fix.embedder,
        &fix.clock,
        &fix.cfg,
        &fix.extraction_cfg,
    );

    let long_user = "Could you please explain in great detail how the authentication subsystem works, including all the moving parts and the rationale behind each decision that was made along the way?";
    let n = uc
        .execute(long_user, "ok", &[], &mk(), &sid(1))
        .await
        .unwrap();
    assert_eq!(
        n, 0,
        "short assistant reply must short-circuit regardless of user message length"
    );
    assert_eq!(
        extractor.call_count(),
        0,
        "gate is on assistant signal only — extractor never invoked"
    );
}

// -----------------------------------------------------------------------
// user_message threading — role markup + gate semantics
// -----------------------------------------------------------------------

/// A non-empty user message is prepended with explicit "User:/Assistant:"
/// role markup so the extractor sees the question→answer pair. The recorded
/// extractor input is the exact combined string.
#[tokio::test]
async fn execute_with_user_message_frames_input_with_role_markup() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();
    let extractor = ScriptedExtractor::new(vec![Ok(vec!["a fact".to_string()])]);
    let fix = Fix::new();
    let uc = build(
        &facts,
        &sessions,
        &extractor,
        &fix.embedder,
        &fix.clock,
        &fix.cfg,
        &fix.extraction_cfg,
    );

    let n = uc
        .execute(
            "  how does auth work?  ",
            "the auth uses JWT",
            &[],
            &mk(),
            &sid(1),
        )
        .await
        .unwrap();
    assert_eq!(n, 1);

    let inputs = extractor.inputs();
    assert_eq!(inputs.len(), 1, "extractor invoked exactly once");
    assert_eq!(
        inputs[0].0, "User:\nhow does auth work?\n\nAssistant:\nthe auth uses JWT",
        "user message trimmed and framed with role labels"
    );
}

/// An empty user message (the import-path shape) produces a label-free input
/// byte-identical to the pre-markup extraction shape — no "Assistant:" prefix.
#[tokio::test]
async fn execute_with_empty_user_message_omits_role_labels() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();
    let extractor = ScriptedExtractor::new(vec![Ok(vec!["a fact".to_string()])]);
    let fix = Fix::new();
    let uc = build(
        &facts,
        &sessions,
        &extractor,
        &fix.embedder,
        &fix.clock,
        &fix.cfg,
        &fix.extraction_cfg,
    );

    uc.execute("", "the auth uses JWT", &[], &mk(), &sid(1))
        .await
        .unwrap();

    let inputs = extractor.inputs();
    assert_eq!(
        inputs[0].0, "the auth uses JWT",
        "no role labels when user_message is empty"
    );
}

/// A response that is ONLY a `<think>` block (no factual answer) must short-
/// circuit: after noise stripping the assistant signal is empty, so extraction
/// is skipped and the extractor is never called.
#[tokio::test]
async fn execute_with_think_only_response_skips_extraction() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();
    let extractor = ScriptedExtractor::new(vec![Err(ProviderError::Unavailable(
        "must not be called".into(),
    ))]);
    let fix = Fix::new();
    let uc = build(
        &facts,
        &sessions,
        &extractor,
        &fix.embedder,
        &fix.clock,
        &fix.cfg,
        &fix.extraction_cfg,
    );

    let think_only =
        "<think>long reasoning that is well over the 15-char floor but carries no fact</think>";
    let n = uc
        .execute("", think_only, &[], &mk(), &sid(1))
        .await
        .unwrap();
    assert_eq!(
        n, 0,
        "think-only response has no extractable assistant signal"
    );
    assert_eq!(
        extractor.call_count(),
        0,
        "extractor never invoked when assistant signal is empty after think-strip"
    );
}
