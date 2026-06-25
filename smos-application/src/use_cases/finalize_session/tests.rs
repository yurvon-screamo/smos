// single cohesive test module - the finalize_session test surface is one
// interlocking set of drift-priority / merge / conflict fixtures that share
// builders and NLI-verdict helpers; splitting it would duplicate the shared
// scaffolding without improving clarity. Kept whole under the R8 size-exception
// (REFACTOR_PLAN.md section 8, R8 acceptance).

//! Classicist unit tests for `FinalizeSession`.
//!
//! The fakes (`InMemoryFacts`, `InMemorySessions`, `ScriptedNliClassifier`)
//! come from [`crate::testkit`] so the use case can be exercised without
//! spinning up SurrealDB or a native NLI backend. E2E coverage against a
//! real `SurrealStore` lives in `smos-adapters/tests/e2e_finalize.rs`.

use super::*;
use crate::testkit::{InMemoryFacts, InMemorySessions, ScriptedNliClassifier};

use smos_domain::config::{ConfidenceConfig, MergeConfig, NliConfig};
use smos_domain::enums::NliLabel;
use smos_domain::{
    Embedding, FactStatus, MemoryKey, NewPendingRequest, NliScores, SessionId, SessionState,
    Timestamp,
};

// ---- Fakes live in `crate::testkit` (shared across use-case tests) ----

/// NLI verdict that always returns `Neutral` (above the no-contradiction
/// threshold but below entailment). Used when tests do not care about the
/// specific label, only that the NLI backend was reachable.
fn neutral_available() -> NliResult {
    NliResult {
        label: NliLabel::Neutral,
        scores: NliScores {
            entailment: 0.2,
            neutral: 0.7,
            contradiction: 0.1,
        },
        available: true,
    }
}

fn entailment_available() -> NliResult {
    NliResult {
        label: NliLabel::Entailment,
        scores: NliScores {
            entailment: 0.9,
            neutral: 0.08,
            contradiction: 0.02,
        },
        available: true,
    }
}

fn contradiction_available() -> NliResult {
    NliResult {
        label: NliLabel::Contradiction,
        scores: NliScores {
            entailment: 0.05,
            neutral: 0.1,
            contradiction: 0.85,
        },
        available: true,
    }
}

// ---- Fixtures ----

fn memory_key() -> MemoryKey {
    MemoryKey::from_raw("origa").unwrap()
}
fn sid(n: u8) -> SessionId {
    SessionId::from_raw(&format!("sess_{:012x}", n as u64)).unwrap()
}
fn ts() -> Timestamp {
    Timestamp::from_unix_secs(1_700_000_000).unwrap()
}

/// Build a pending fact whose content-derived id is deterministic.
fn pending(content: &str, embedding: Vec<f32>) -> Fact {
    Fact::new_pending(NewPendingRequest {
        content,
        memory_key: memory_key(),
        session: sid(1),
        embedding: Embedding::new(embedding).unwrap(),
        extracted_at: ts(),
        base_confidence: ConfidenceConfig::default().base,
    })
    .unwrap()
}

/// Build an accepted fact (single source, base confidence lifted above the
/// accept threshold via `set_status_and_confidence`).
fn accepted(content: &str, embedding: Vec<f32>) -> Fact {
    let mut f = Fact::new_pending(NewPendingRequest {
        content,
        memory_key: memory_key(),
        session: sid(2),
        embedding: Embedding::new(embedding).unwrap(),
        extracted_at: ts(),
        base_confidence: ConfidenceConfig::default().base,
    })
    .unwrap();
    f.set_status_and_confidence(
        FactStatus::Accepted,
        smos_domain::Confidence::new(0.9).unwrap(),
        &ConfidenceConfig::default(),
    )
    .unwrap();
    f
}

/// Build a session state carrying `owned` pending fact ids.
fn session_with_pending(owned: Vec<FactId>) -> SessionState {
    let mut state = SessionState::new(sid(1), memory_key(), ts());
    state.add_pending(&owned);
    state
}

/// Shared fixture: confidence / NLI / merge configs owned by the test so
/// the returned use case can borrow them for its whole lifetime.
/// Mirrors the `Fix` pattern in `extract_facts_from_response`.
struct Fix {
    confidence_cfg: ConfidenceConfig,
    nli_cfg: NliConfig,
    merge_cfg: MergeConfig,
}
impl Fix {
    fn new() -> Self {
        Self {
            confidence_cfg: ConfidenceConfig::default(),
            nli_cfg: NliConfig::default(),
            merge_cfg: MergeConfig::default(),
        }
    }
}

fn build<'a>(
    facts: &'a InMemoryFacts,
    sessions: &'a InMemorySessions,
    classifier: &'a ScriptedNliClassifier,
    fix: &'a Fix,
) -> FinalizeSession<'a, InMemoryFacts, InMemorySessions, ScriptedNliClassifier> {
    FinalizeSession {
        facts,
        sessions,
        classifier,
        confidence_cfg: &fix.confidence_cfg,
        nli_cfg: &fix.nli_cfg,
        merge_cfg: &fix.merge_cfg,
    }
}

// -----------------------------------------------------------------------
// Happy-path tests
// -----------------------------------------------------------------------

#[tokio::test]
async fn execute_no_session_returns_empty_stats() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    let classifier = ScriptedNliClassifier::new(vec![]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
    assert_eq!(stats.processed, 0);
    assert_eq!(stats.finalized, 0);
    assert!(classifier.calls().is_empty(), "no NLI call without pending");
}

/// Regression guard for the operator-facing bug: HTTP extraction persists
/// `fact.source_sessions` but NEVER writes a `SessionState` row, so the
/// previous implementation (which read `SessionState.pending_facts()`
/// for ownership) reported "nothing to do" while 24 pending facts sat in
/// the store. The fix derives ownership from `source_sessions` instead,
/// so a missing SessionState must NOT mask real pending facts.
#[tokio::test]
async fn execute_processes_pending_facts_even_when_session_state_is_absent() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    // NO `sessions.seed(...)` — the HTTP path leaves SessionState empty.
    // The pending fact still carries `source_sessions = [sid(1)]`
    // (the `pending()` fixture sets it via `Fact::new_pending`), which
    // is the only ownership signal the use case consults after the fix.
    let fact = pending("user prefers rust over go", vec![1.0, 0.0, 0.0]);
    let fact_id = fact.id().clone();
    facts.seed(fact);

    let classifier = ScriptedNliClassifier::new(vec![]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
    assert_eq!(
        stats.processed, 1,
        "missing SessionState must not mask the fact"
    );
    assert_eq!(stats.finalized, 1);
    let finalized = facts.get_clone(&fact_id).expect("fact still present");
    assert_eq!(finalized.status(), FactStatus::Pending);
}

/// A pending fact whose `source_sessions` does NOT contain the target
/// session is skipped — finalize is scoped to one session's ownership,
/// not to every pending fact in the namespace.
#[tokio::test]
async fn execute_skips_pending_fact_owned_by_a_different_session() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    // `pending()` fixture sets source_sessions = [sid(1)] — finalizing
    // sid(2) must NOT pick it up.
    let fact = pending("user prefers rust over go", vec![1.0, 0.0, 0.0]);
    let fact_id = fact.id().clone();
    facts.seed(fact);

    let classifier = ScriptedNliClassifier::new(vec![]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(2), &memory_key()).await.unwrap();
    assert_eq!(stats.processed, 0);
    // The fact survives untouched.
    let untouched = facts.get_clone(&fact_id).expect("fact still present");
    assert_eq!(untouched.status(), FactStatus::Pending);
}

#[tokio::test]
async fn execute_empty_session_returns_empty_stats() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    sessions.seed(SessionState::new(sid(1), memory_key(), ts()));
    let classifier = ScriptedNliClassifier::new(vec![]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
    assert_eq!(stats.processed, 0);
}

#[tokio::test]
async fn execute_standalone_promotes_pending_fact_with_no_candidate() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    // Pending fact with a unique embedding → no candidate above the merge
    // threshold (no accepted fact exists at all).
    let fact = pending("user prefers rust over go", vec![1.0, 0.0, 0.0]);
    let fact_id = fact.id().clone();
    facts.seed(fact);
    sessions.seed(session_with_pending(vec![fact_id.clone()]));

    let classifier = ScriptedNliClassifier::new(vec![]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
    assert_eq!(stats.processed, 1);
    assert_eq!(stats.finalized, 1);
    assert_eq!(stats.merged, 0);
    assert_eq!(stats.conflicts, 0);
    // Single-source, base confidence (0.5) → Pending (validation gate).
    let finalized = facts.get_clone(&fact_id).expect("fact still present");
    assert_eq!(finalized.status(), FactStatus::Pending);
    assert!(
        classifier.calls().is_empty(),
        "no NLI call without candidate"
    );
    assert!(
        sessions.pending_of(&sid(1)).is_empty(),
        "owned pending cleared"
    );
}

#[tokio::test]
async fn execute_entailment_merges_pending_into_existing() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    let existing = accepted("ttl=10 prevents refresh loop", vec![1.0, 0.0, 0.0]);
    let existing_id = existing.id().clone();
    facts.seed(existing);
    // Pending twin: identical embedding (cosine 1.0 ≥ 0.85 merge threshold).
    let pending_fact = pending("ttl=10 stops the refresh loop", vec![1.0, 0.0, 0.0]);
    let pending_id = pending_fact.id().clone();
    facts.seed(pending_fact.clone());
    sessions.seed(session_with_pending(vec![pending_id.clone()]));

    let classifier = ScriptedNliClassifier::new(vec![Ok(entailment_available())]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
    assert_eq!(stats.processed, 1);
    assert_eq!(stats.merged, 1);
    assert_eq!(stats.rejected, 1);
    assert_eq!(stats.finalized, 0);

    // Existing fact grew provenance (union of source sessions) and was
    // reclassified with the entailment verdict (no_contradiction_bonus).
    let merged = facts.get_clone(&existing_id).expect("existing present");
    assert!(merged.source_sessions().distinct_count() >= 2);
    // Pending twin was rejected.
    let twin = facts.get_clone(&pending_id).expect("pending present");
    assert_eq!(twin.status(), FactStatus::Rejected);
    assert!(
        sessions.pending_of(&sid(1)).is_empty(),
        "owned pending cleared"
    );
}

#[tokio::test]
async fn execute_contradiction_flags_bidirectional_conflict() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    let existing = accepted("ttl=60 seconds", vec![1.0, 0.0, 0.0]);
    let existing_id = existing.id().clone();
    facts.seed(existing);
    let pending_fact = pending("ttl=10 seconds", vec![1.0, 0.0, 0.0]);
    let pending_id = pending_fact.id().clone();
    facts.seed(pending_fact.clone());
    sessions.seed(session_with_pending(vec![pending_id.clone()]));

    let classifier = ScriptedNliClassifier::new(vec![Ok(contradiction_available())]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
    assert_eq!(stats.processed, 1);
    assert_eq!(stats.conflicts, 1);
    assert_eq!(stats.merged, 0);
    assert_eq!(stats.finalized, 0);

    // Both sides carry the bidirectional conflict flag.
    let existing_after = facts.get_clone(&existing_id).expect("existing present");
    let pending_after = facts.get_clone(&pending_id).expect("pending present");
    assert!(existing_after.conflicts_with().contains(&pending_id));
    assert!(pending_after.conflicts_with().contains(&existing_id));
    // Status UNCHANGED on both sides (Accepted stays Accepted, Pending stays Pending).
    assert_eq!(existing_after.status(), FactStatus::Accepted);
    assert_eq!(pending_after.status(), FactStatus::Pending);
    // No valid_until tombstone on either side (drift is not a death).
    assert!(existing_after.valid_until().is_none());
    assert!(pending_after.valid_until().is_none());
}

// -----------------------------------------------------------------------
// Drift-priority walk
// -----------------------------------------------------------------------

#[tokio::test]
async fn drift_priority_walk_contradiction_beats_earlier_neutral() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    // Two accepted facts, both above the cosine threshold. The closer
    // candidate ("similar") would yield `Neutral` — the use case must
    // keep scanning so the contradiction against the less-similar
    // candidate ("drift") still wins.
    let closer = accepted("rust is memory safe", vec![1.0, 0.0, 0.0]);
    let closer_id = closer.id().clone();
    let farther = accepted("rust leaks memory everywhere", vec![0.9, 0.1, 0.0]);
    let farther_id = farther.id().clone();
    facts.seed(closer);
    facts.seed(farther);
    let pending_fact = pending("rust is memory safe language", vec![1.0, 0.0, 0.0]);
    let pending_id = pending_fact.id().clone();
    facts.seed(pending_fact.clone());
    sessions.seed(session_with_pending(vec![pending_id.clone()]));

    // find_merge_candidates sorts by cosine descending, so the first NLI
    // call hits "closer" (Neutral), the second hits "farther" (Contradiction).
    let classifier =
        ScriptedNliClassifier::new(vec![Ok(neutral_available()), Ok(contradiction_available())]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
    assert_eq!(
        stats.conflicts, 1,
        "drift must win over the earlier neutral"
    );
    assert_eq!(stats.merged, 0, "no merge despite the neutral candidate");

    // The contradiction was flagged against "farther" (the contradicting
    // candidate), NOT against "closer".
    let pending_after = facts.get_clone(&pending_id).expect("pending present");
    assert!(
        pending_after.conflicts_with().contains(&farther_id),
        "drift flag points to the contradicting candidate"
    );
    assert!(
        !pending_after.conflicts_with().contains(&closer_id),
        "no spurious drift flag on the neutral candidate"
    );
}

#[tokio::test]
async fn drift_priority_walk_keeps_merge_pick_but_still_scans_for_contradiction() {
    // Entailment candidate first, contradiction second → contradiction wins,
    // the earlier merge pick is NOT committed.
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    let entailed = accepted("the api runs on port 8080", vec![1.0, 0.0, 0.0]);
    let entailed_id = entailed.id().clone();
    let drift = accepted("the api runs on port 9090", vec![0.95, 0.05, 0.0]);
    let drift_id = drift.id().clone();
    facts.seed(entailed);
    facts.seed(drift);
    let pending_fact = pending("the api runs on port 8080 today", vec![1.0, 0.0, 0.0]);
    let pending_id = pending_fact.id().clone();
    facts.seed(pending_fact.clone());
    sessions.seed(session_with_pending(vec![pending_id.clone()]));

    let classifier = ScriptedNliClassifier::new(vec![
        Ok(entailment_available()),
        Ok(contradiction_available()),
    ]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
    assert_eq!(stats.conflicts, 1);
    assert_eq!(stats.merged, 0);
    // The entailed candidate was NOT modified (no merge committed).
    let entailed_after = facts.get_clone(&entailed_id).expect("entailed present");
    assert_eq!(
        entailed_after.source_sessions().distinct_count(),
        1,
        "merge not committed for the entailed candidate"
    );
    // The drift candidate was flagged.
    let drift_after = facts.get_clone(&drift_id).expect("drift present");
    assert!(drift_after.conflicts_with().contains(&pending_id));
}

// -----------------------------------------------------------------------
// C3 guard — already-flagged pairs skip the sidecar
// -----------------------------------------------------------------------

#[tokio::test]
async fn c3_guard_skips_nli_for_already_flagged_conflict_pair() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    let mut existing = accepted("ttl=60 seconds", vec![1.0, 0.0, 0.0]);
    let mut pending_fact = pending("ttl=10 seconds", vec![1.0, 0.0, 0.0]);
    // Pre-flag the pair so the C3 guard fires before any sidecar call.
    existing.flag_conflict(pending_fact.id().clone()).unwrap();
    pending_fact.flag_conflict(existing.id().clone()).unwrap();
    let existing_id = existing.id().clone();
    let pending_id = pending_fact.id().clone();
    facts.seed(existing);
    facts.seed(pending_fact.clone());
    sessions.seed(session_with_pending(vec![pending_id.clone()]));

    // The classifier would have returned contradiction, but the C3 guard
    // must short-circuit before any call.
    let classifier = ScriptedNliClassifier::new(vec![Ok(contradiction_available())]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
    assert_eq!(stats.processed, 1);
    // The C3 guard skipped every candidate → standalone promotion.
    assert_eq!(stats.finalized, 1);
    assert_eq!(stats.conflicts, 0);
    assert!(
        classifier.calls().is_empty(),
        "C3 guard must skip every sidecar call"
    );
    // Existing flags UNCHANGED (no double-flag).
    let existing_after = facts.get_clone(&existing_id).expect("existing present");
    assert_eq!(existing_after.conflicts_with().len(), 1);
    assert!(existing_after.conflicts_with().contains(&pending_id));
    // Pending twin also keeps its pre-flagged conflict link — the C3
    // guard leaves both sides untouched, which is the contract that
    // keeps a re-finalized session idempotent (no spurious
    // double-flag, no leak of the conflict to fresh candidates).
    let pending_after = facts.get_clone(&pending_id).expect("pending present");
    assert_eq!(pending_after.conflicts_with().len(), 1);
    assert!(
        pending_after.conflicts_with().contains(&existing_id),
        "pending twin must retain its pre-existing conflict flag"
    );
}

// -----------------------------------------------------------------------
// Multi-contradiction — pending fact drifts against 2+ existing facts
// -----------------------------------------------------------------------

/// A pending fact that contradicts MULTIPLE accepted facts flags every
/// contradiction it finds. Drift-priority means the FIRST
/// contradiction wins for the *outcome* (the loop returns
/// `Conflict` on the first one), but `resolve_one` continues scanning
/// only until the first contradiction — it does NOT keep flagging
/// after the drift is observed. This test pins that semantics: the
/// second contradicting candidate is NOT visited once the first
/// contradiction has fired.
#[tokio::test]
async fn multi_contradiction_returns_after_first_drift() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    let existing_a = accepted("ttl=60 seconds", vec![1.0, 0.0, 0.0]);
    let existing_b = accepted("ttl=30 seconds", vec![0.95, 0.05, 0.0]);
    let a_id = existing_a.id().clone();
    let b_id = existing_b.id().clone();
    facts.seed(existing_a);
    facts.seed(existing_b);
    let pending_fact = pending("ttl=10 seconds", vec![1.0, 0.0, 0.0]);
    let pending_id = pending_fact.id().clone();
    facts.seed(pending_fact.clone());
    sessions.seed(session_with_pending(vec![pending_id.clone()]));

    // First candidate returns contradiction → loop returns
    // immediately. The second verdict (also contradiction) is never
    // consumed.
    let classifier = ScriptedNliClassifier::new(vec![Ok(contradiction_available())]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
    assert_eq!(stats.conflicts, 1);
    assert_eq!(stats.processed, 1);
    assert_eq!(
        classifier.calls().len(),
        1,
        "first contradiction must short-circuit; second candidate not visited"
    );

    // The pending twin carries exactly ONE drift flag (against
    // whichever candidate was visited first — the merge-candidate
    // order is deterministic via cosine).
    let pending_after = facts.get_clone(&pending_id).expect("pending present");
    assert_eq!(
        pending_after.conflicts_with().len(),
        1,
        "exactly one drift flag on the pending twin"
    );
    // Sanity: the flagged id is one of the two existing facts.
    let flagged = pending_after
        .conflicts_with()
        .iter()
        .next()
        .expect("flag set");
    assert!(*flagged == a_id || *flagged == b_id);
}

// -----------------------------------------------------------------------
// Exact-match short-circuit
// -----------------------------------------------------------------------

#[tokio::test]
async fn exact_match_skips_sidecar_and_merges_identical_pair() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    let existing = accepted("identical fact content", vec![1.0, 0.0, 0.0]);
    let existing_id = existing.id().clone();
    facts.seed(existing);
    // Pending twin has the SAME content → exact-match short-circuit.
    // Note: FactId is content-derived, so two identical-content facts
    // share the same id. We bypass that here by seeding the pending twin
    // under a different content hash via the lowercase trick (POC normalises
    // case + whitespace, so "IDENTICAL FACT CONTENT" exact-matches the
    // existing lower-case form).
    let pending_fact = pending("IDENTICAL FACT CONTENT", vec![1.0, 0.0, 0.0]);
    let pending_id = pending_fact.id().clone();
    facts.seed(pending_fact.clone());
    sessions.seed(session_with_pending(vec![pending_id.clone()]));

    let classifier = ScriptedNliClassifier::new(vec![Ok(contradiction_available())]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
    // Exact-match returns entailment immediately → merge committed. The
    // scripted contradiction verdict MUST NOT be consumed.
    assert_eq!(stats.merged, 1);
    assert_eq!(stats.conflicts, 0);
    assert!(
        classifier.calls().is_empty(),
        "exact-match must short-circuit before any sidecar call"
    );
    let merged = facts.get_clone(&existing_id).expect("existing present");
    assert!(merged.source_sessions().distinct_count() >= 2);
}

// -----------------------------------------------------------------------
// Graceful degradation
// -----------------------------------------------------------------------

#[tokio::test]
async fn sidecar_unavailable_keeps_pending_fact_gracefully() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    let existing = accepted("rust is memory safe", vec![1.0, 0.0, 0.0]);
    facts.seed(existing);
    let pending_fact = pending("rust guarantees memory safety", vec![1.0, 0.0, 0.0]);
    let pending_id = pending_fact.id().clone();
    facts.seed(pending_fact.clone());
    sessions.seed(session_with_pending(vec![pending_id.clone()]));

    // Every NLI call is Unavailable — the use case must not raise.
    let classifier = ScriptedNliClassifier::new(vec![Err(ProviderError::Unavailable(
        "sidecar crashed".into(),
    ))]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc
        .execute(&sid(1), &memory_key())
        .await
        .expect("graceful Ok");
    // No outcome tallied (skip does not increment any counter).
    assert_eq!(stats.finalized, 0);
    assert_eq!(stats.merged, 0);
    assert_eq!(stats.conflicts, 0);
    // The pending fact survives unchanged.
    let pending_after = facts.get_clone(&pending_id).expect("pending present");
    assert_eq!(pending_after.status(), FactStatus::Pending);
    assert!(pending_after.conflicts_with().is_empty());
}

#[tokio::test]
async fn sidecar_replies_available_false_keeps_pending_fact_gracefully() {
    // The sidecar sometimes replies with its own graceful-degradation
    // placeholder (label=neutral, available=false) when the model raised
    // on a malformed input or the sidecar's stdout closed before the
    // reply landed. The use case must treat `available = false` exactly
    // like `Err(Unavailable)` — the pending fact stays pending so a
    // permanently broken sidecar cannot silently promote facts past the
    // drift-detection gate.
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    let existing = accepted("rust is memory safe", vec![1.0, 0.0, 0.0]);
    facts.seed(existing);
    let pending_fact = pending("rust guarantees memory safety", vec![1.0, 0.0, 0.0]);
    let pending_id = pending_fact.id().clone();
    facts.seed(pending_fact.clone());
    sessions.seed(session_with_pending(vec![pending_id.clone()]));

    // Reply shape mirrors the "classifier unavailable" verdict produced
    // by the NLI backend on a transport/runtime failure (see
    // `ProviderError::Unavailable` mapping in `NativeNliClassifier`).
    let unavailable_verdict = NliResult {
        label: NliLabel::Neutral,
        scores: NliScores {
            entailment: 0.0,
            neutral: 1.0,
            contradiction: 0.0,
        },
        available: false,
    };
    let classifier = ScriptedNliClassifier::new(vec![Ok(unavailable_verdict)]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc
        .execute(&sid(1), &memory_key())
        .await
        .expect("graceful Ok");
    assert_eq!(stats.finalized, 0, "available=false must NOT promote");
    assert_eq!(stats.merged, 0);
    assert_eq!(stats.conflicts, 0);
    let pending_after = facts.get_clone(&pending_id).expect("pending present");
    assert_eq!(pending_after.status(), FactStatus::Pending);
    assert!(
        pending_after.conflicts_with().is_empty(),
        "no drift flag without a real verdict"
    );
}

#[tokio::test]
async fn batch_continues_after_single_pair_failure() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    // Three pending facts, two with candidates and one standalone. Each
    // candidate has a distinct content so the matcher can return a
    // deterministic verdict regardless of `HashMap` iteration order.
    let existing = accepted("shared anchor fact here", vec![1.0, 0.0, 0.0]);
    facts.seed(existing);
    // p1: similar but the matcher marks it as NLI-unavailable → skip pair.
    let p1 = pending("shared anchor fact here too", vec![1.0, 0.0, 0.0]);
    // p2: similar, matcher returns entailment → merge.
    let p2 = pending("shared anchor fact but longer", vec![1.0, 0.0, 0.0]);
    // p3: orthogonal embedding → no candidate → standalone promotion.
    let p3 = pending("totally unrelated pending fact", vec![0.0, 1.0, 0.0]);
    let p1_id = p1.id().clone();
    let p3_id = p3.id().clone();
    facts.seed(p1.clone());
    facts.seed(p2.clone());
    facts.seed(p3.clone());
    sessions.seed(session_with_pending(vec![
        p1.id().clone(),
        p2.id().clone(),
        p3.id().clone(),
    ]));

    // Order-independent matcher: keyed on the hypothesis text (the pending
    // twin) so HashMap iteration order over the pending list does not
    // change the outcome.
    let classifier = ScriptedNliClassifier::matching(|_premise, hypothesis| match hypothesis {
        "shared anchor fact here too" => Err(ProviderError::Unavailable("transient".into())),
        "shared anchor fact but longer" => Ok(entailment_available()),
        other => Err(ProviderError::InvalidResponse(format!(
            "unexpected hypothesis: {other}"
        ))),
    });
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
    // One merge (p2 → existing), one finalize (p3 standalone), one skip
    // (p1 stayed pending because the sidecar was unreachable).
    assert_eq!(stats.processed, 3);
    assert_eq!(stats.merged, 1);
    assert_eq!(stats.finalized, 1);
    let p1_after = facts.get_clone(&p1_id).expect("p1 present");
    assert_eq!(p1_after.status(), FactStatus::Pending, "p1 stayed pending");
    let p3_after = facts.get_clone(&p3_id).expect("p3 present");
    // p3 standalone: single source, base confidence → still Pending.
    assert_eq!(p3_after.status(), FactStatus::Pending);
}

// -----------------------------------------------------------------------
// Bookkeeping cleanup
// -----------------------------------------------------------------------

#[tokio::test]
async fn finalize_clears_owned_pending_ids_after_drain() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    // Two pending facts, both owned by the session. After finalize the
    // session's pending list must be empty (both owned ids drained).
    let p1 = pending("first standalone pending fact", vec![1.0, 0.0, 0.0]);
    let p2 = pending("second standalone pending fact", vec![0.0, 1.0, 0.0]);
    let p1_id = p1.id().clone();
    let p2_id = p2.id().clone();
    facts.seed(p1);
    facts.seed(p2);
    sessions.seed(session_with_pending(vec![p1_id, p2_id]));

    let classifier = ScriptedNliClassifier::new(vec![]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
    assert_eq!(stats.processed, 2);
    assert!(
        sessions.pending_of(&sid(1)).is_empty(),
        "owned pending ids cleared after finalize"
    );
}

// -----------------------------------------------------------------------
// Stats contract
// -----------------------------------------------------------------------

#[tokio::test]
async fn stats_default_is_zeroed() {
    let stats = FinalizeStats::default();
    assert_eq!(stats.processed, 0);
    assert_eq!(stats.finalized, 0);
    assert_eq!(stats.merged, 0);
    assert_eq!(stats.conflicts, 0);
    assert_eq!(stats.rejected, 0);
    assert!(stats.session_id.is_empty());
}

#[tokio::test]
async fn stats_session_id_echoed_in_output() {
    let facts = InMemoryFacts::default();
    let sessions = InMemorySessions::default();
    sessions.seed(SessionState::new(sid(7), memory_key(), ts()));
    let classifier = ScriptedNliClassifier::new(vec![]);
    let fix = Fix::new();
    let uc = build(&facts, &sessions, &classifier, &fix);

    let stats = uc.execute(&sid(7), &memory_key()).await.unwrap();
    assert_eq!(stats.session_id, sid(7).as_str());
}

// -----------------------------------------------------------------------
// Golden snapshot -- resolve_one outcome matrix (R8 behaviour lock)
// -----------------------------------------------------------------------
// Captures the current FactOutcome for each (pending, pool, nli-verdicts)
// combination produced by `resolve_one`. Written BEFORE the R8 structural
// split (ScanState extraction) so the drift-priority algorithm is locked.
// After R8 this test must pass WITHOUT any assertion change.
// golden snapshot -- must not change across R8
#[tokio::test]
async fn resolve_one_outcome_matrix_golden() {
    // Row 1 -- candidates empty -> standalone finalize.
    {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let classifier = ScriptedNliClassifier::new(vec![]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);
        let p = pending("standalone with no candidate at all", vec![1.0, 0.0, 0.0]);
        let mut pool: Vec<Fact> = vec![];
        let outcome = uc.resolve_one(&p, &mut pool).await;
        assert_eq!(
            outcome,
            FactOutcome::Finalized,
            "row1: empty pool -> standalone finalize"
        );
    }

    // Row 2 -- exact-text match short-circuits to merge (no NLI call).
    {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        // Scripted contradiction that MUST NOT be consumed.
        let classifier = ScriptedNliClassifier::new(vec![Ok(contradiction_available())]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);
        let existing = accepted("identical fact content", vec![1.0, 0.0, 0.0]);
        // CASE-normalised exact match of the existing content.
        let p = pending("IDENTICAL FACT CONTENT", vec![1.0, 0.0, 0.0]);
        let mut pool = vec![existing];
        let outcome = uc.resolve_one(&p, &mut pool).await;
        assert_eq!(
            outcome,
            FactOutcome::Merged,
            "row2: exact-text match -> merge"
        );
        assert!(
            classifier.calls().is_empty(),
            "row2: exact-match must not call NLI"
        );
    }

    // Row 3 -- single entailment candidate -> merge.
    {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let classifier = ScriptedNliClassifier::new(vec![Ok(entailment_available())]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);
        let existing = accepted("ttl=10 prevents refresh loop", vec![1.0, 0.0, 0.0]);
        let p = pending("ttl=10 stops the refresh loop", vec![1.0, 0.0, 0.0]);
        let mut pool = vec![existing];
        let outcome = uc.resolve_one(&p, &mut pool).await;
        assert_eq!(outcome, FactOutcome::Merged, "row3: entailment -> merge");
    }

    // Row 4 -- single contradiction candidate -> conflict flag.
    {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let classifier = ScriptedNliClassifier::new(vec![Ok(contradiction_available())]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);
        let existing = accepted("ttl=60 seconds", vec![1.0, 0.0, 0.0]);
        let p = pending("ttl=10 seconds", vec![1.0, 0.0, 0.0]);
        let mut pool = vec![existing];
        let outcome = uc.resolve_one(&p, &mut pool).await;
        assert_eq!(
            outcome,
            FactOutcome::Conflict,
            "row4: contradiction -> conflict"
        );
    }

    // Row 5 -- first entailment + second contradiction -> drift wins (Conflict).
    {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let classifier = ScriptedNliClassifier::new(vec![
            Ok(entailment_available()),
            Ok(contradiction_available()),
        ]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);
        let entailed = accepted("the api runs on port 8080", vec![1.0, 0.0, 0.0]);
        let drift = accepted("the api runs on port 9090", vec![0.95, 0.05, 0.0]);
        let p = pending("the api runs on port 8080 today", vec![1.0, 0.0, 0.0]);
        let mut pool = vec![entailed, drift];
        let outcome = uc.resolve_one(&p, &mut pool).await;
        assert_eq!(
            outcome,
            FactOutcome::Conflict,
            "row5: drift-priority - contradiction overrides earlier entailment"
        );
    }

    // Row 6 -- entailment + neutral -> first entailment wins (Merged).
    {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let classifier =
            ScriptedNliClassifier::new(vec![Ok(entailment_available()), Ok(neutral_available())]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);
        let entailed = accepted("the api runs on port 8080", vec![1.0, 0.0, 0.0]);
        let neutral_cand = accepted("the api runs on port 9090", vec![0.95, 0.05, 0.0]);
        let p = pending("the api runs on port 8080 today", vec![1.0, 0.0, 0.0]);
        let mut pool = vec![entailed, neutral_cand];
        let outcome = uc.resolve_one(&p, &mut pool).await;
        assert_eq!(
            outcome,
            FactOutcome::Merged,
            "row6: first entailment wins; later neutral does not override"
        );
    }

    // Row 7 -- NLI unavailable for every candidate -> Skipped (stay pending).
    {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let classifier = ScriptedNliClassifier::new(vec![
            Err(ProviderError::Unavailable("backend down".into())),
            Err(ProviderError::Unavailable("backend down".into())),
        ]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);
        let a = accepted("content alpha marker", vec![1.0, 0.0, 0.0]);
        let b = accepted("content beta marker", vec![0.9, 0.1, 0.0]);
        let p = pending("content gamma marker", vec![1.0, 0.0, 0.0]);
        let mut pool = vec![a, b];
        let outcome = uc.resolve_one(&p, &mut pool).await;
        assert_eq!(
            outcome,
            FactOutcome::Skipped,
            "row7: NLI never observed -> skip (stay pending)"
        );
    }

    // Row 8 -- C3 guard only (already-flagged pair) -> standalone finalize.
    {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        // Would-be contradiction is never consumed - the C3 guard fires first.
        let classifier = ScriptedNliClassifier::new(vec![Ok(contradiction_available())]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);
        let mut existing = accepted("ttl=60 seconds", vec![1.0, 0.0, 0.0]);
        let mut p = pending("ttl=10 seconds", vec![1.0, 0.0, 0.0]);
        existing.flag_conflict(p.id().clone()).unwrap();
        p.flag_conflict(existing.id().clone()).unwrap();
        let mut pool = vec![existing];
        let outcome = uc.resolve_one(&p, &mut pool).await;
        assert_eq!(
            outcome,
            FactOutcome::Finalized,
            "row8: C3 guard -> standalone finalize (nli_observed via flag)"
        );
        assert!(
            classifier.calls().is_empty(),
            "row8: C3 guard skips every NLI call"
        );
    }
}
