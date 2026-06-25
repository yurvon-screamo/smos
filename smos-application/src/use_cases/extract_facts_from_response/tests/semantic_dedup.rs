use super::*;
use smos_domain::{Embedding, Fact, FactId, NewPendingRequest, Timestamp};

/// Rephrased re-observation (different SHA1) is caught at Layer 2 via
/// cosine similarity and routed through cross-session confirmation
/// instead of leaving the fact stuck at single-source confidence.
#[tokio::test]
async fn persist_facts_layer2_semantic_match_confirms_existing_fact() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();

    // Seed an existing fact from session 1 under one phrasing.
    let stored = Fact::new_pending(NewPendingRequest {
        content: "the token cache uses TTL=60 to avoid stale entries",
        memory_key: mk(),
        session: sid(1),
        embedding: Embedding::new(vec![1.0]).unwrap(),
        extracted_at: Timestamp::from_unix_secs(1_700_000_000).unwrap(),
        base_confidence: ConfidenceConfig::default().base,
    })
    .unwrap();
    let stored_id = stored.id().clone();
    facts.seed(stored);

    // The extractor rephrased the same concept differently → its FactId
    // will differ, so Layer 1 (exact match) misses. Layer 2 must catch
    // it because the scripted `search_similar` returns the stored fact
    // with similarity 0.98 (above the 0.95 threshold).
    facts.script_dedup_hits(vec![hit_for(
        &facts.get_clone(&stored_id).expect("seeded fact"),
        0.98,
        mk(),
    )]);

    let rephrased = "token cache TTL is 60 to prevent stale entries";
    let extractor = ScriptedExtractor::new(vec![Ok(vec![rephrased.to_string()])]);
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

    let n = uc.execute(rephrased, &[], &mk(), &sid(2)).await.unwrap();

    assert_eq!(
        n, 0,
        "semantic duplicate must confirm, not create a new fact"
    );
    let confirmed = facts
        .get_clone(&stored_id)
        .expect("seeded fact still present");
    assert_eq!(
        confirmed.source_sessions().distinct_count(),
        2,
        "semantic match grows provenance to two sessions"
    );
    assert!(
        facts.get_clone(&FactId::from_content(rephrased)).is_none(),
        "no new fact id created for the rephrased variant"
    );
    assert!(
        sessions.pending.lock().unwrap().is_empty(),
        "semantic confirmation must not register on the pending list"
    );
}

/// Below the cosine threshold the semantic layer must NOT collapse two
/// different phrasings: the new fact is stored as a separate pending
/// entry (Layer 3 fallback).
#[tokio::test]
async fn persist_facts_layer2_below_threshold_creates_new_fact() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();

    let stored = Fact::new_pending(NewPendingRequest {
        content: "auth module uses Argon2id for password hashing",
        memory_key: mk(),
        session: sid(1),
        embedding: Embedding::new(vec![1.0]).unwrap(),
        extracted_at: Timestamp::from_unix_secs(1_700_000_000).unwrap(),
        base_confidence: ConfidenceConfig::default().base,
    })
    .unwrap();
    let stored_id = stored.id().clone();
    facts.seed(stored);

    // Similarity 0.80 < 0.95 threshold → Layer 2 must NOT fire.
    facts.script_dedup_hits(vec![hit_for(
        &facts.get_clone(&stored_id).expect("seeded fact"),
        0.80,
        mk(),
    )]);

    let new_content = "TLS handshake failure in the upstream pool";
    let extractor = ScriptedExtractor::new(vec![Ok(vec![new_content.to_string()])]);
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

    let n = uc.execute(new_content, &[], &mk(), &sid(2)).await.unwrap();

    assert_eq!(n, 1, "below-threshold similarity must create a new fact");
    let new_id = FactId::from_content(new_content);
    assert!(
        facts.contains(&new_id),
        "new fact persisted under its own FactId"
    );
    assert_eq!(
        sessions.pending.lock().unwrap().len(),
        1,
        "new fact registered on the pending list"
    );
}

/// `metadata.distance = None` (store did not surface a distance) must
/// NOT collapse two phrasings even when the underlying row would
/// otherwise match — Layer 2 cannot make a decision without a distance,
/// so it falls through to Layer 3 (create new). This guards against
/// silent dedup when a future adapter forgets to populate distance.
#[tokio::test]
async fn persist_facts_layer2_missing_distance_falls_through_to_new_fact() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();

    let stored = Fact::new_pending(NewPendingRequest {
        content: "config reload triggers a graceful drain",
        memory_key: mk(),
        session: sid(1),
        embedding: Embedding::new(vec![1.0]).unwrap(),
        extracted_at: Timestamp::from_unix_secs(1_700_000_000).unwrap(),
        base_confidence: ConfidenceConfig::default().base,
    })
    .unwrap();
    let stored_id = stored.id().clone();
    facts.seed(stored);

    let mut hit = hit_for(
        &facts.get_clone(&stored_id).expect("seeded fact"),
        1.0,
        mk(),
    );
    hit.metadata.distance = None;
    facts.script_dedup_hits(vec![hit]);

    let new_content = "config reload drains gracefully on SIGHUP";
    let extractor = ScriptedExtractor::new(vec![Ok(vec![new_content.to_string()])]);
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

    let n = uc.execute(new_content, &[], &mk(), &sid(2)).await.unwrap();

    assert_eq!(
        n, 1,
        "missing distance must not collapse — fall through to new fact"
    );
}

/// Tunable threshold: when the operator lowers
/// `dedup_cosine_threshold`, an above-threshold hit at 0.85 (which the
/// default 0.95 would have rejected) now confirms the existing fact.
/// Confirms the config field actually flows into the dedup decision.
#[tokio::test]
async fn persist_facts_layer2_threshold_lowered_collapses_0_85_pair() {
    let facts = InMemoryFacts::default();
    let sessions = RecordingSessions::default();

    let stored = Fact::new_pending(NewPendingRequest {
        content: "indexer batches at most 1024 documents per commit",
        memory_key: mk(),
        session: sid(1),
        embedding: Embedding::new(vec![1.0]).unwrap(),
        extracted_at: Timestamp::from_unix_secs(1_700_000_000).unwrap(),
        base_confidence: ConfidenceConfig::default().base,
    })
    .unwrap();
    let stored_id = stored.id().clone();
    facts.seed(stored);

    // Similarity 0.85 — above the operator-lowered 0.80 threshold.
    facts.script_dedup_hits(vec![hit_for(
        &facts.get_clone(&stored_id).expect("seeded fact"),
        0.85,
        mk(),
    )]);

    let rephrased = "the indexer caps batches at 1024 documents";
    let extractor = ScriptedExtractor::new(vec![Ok(vec![rephrased.to_string()])]);
    let mut fix = Fix::new();
    fix.extraction_cfg = ExtractionConfig {
        dedup_cosine_threshold: 0.80,
    };
    let uc = build(
        &facts,
        &sessions,
        &extractor,
        &fix.embedder,
        &fix.clock,
        &fix.cfg,
        &fix.extraction_cfg,
    );

    let n = uc.execute(rephrased, &[], &mk(), &sid(2)).await.unwrap();

    assert_eq!(
        n, 0,
        "lowered threshold collapses the 0.85 pair via semantic match"
    );
    let confirmed = facts
        .get_clone(&stored_id)
        .expect("seeded fact still present");
    assert_eq!(
        confirmed.source_sessions().distinct_count(),
        2,
        "semantic collapse grows provenance"
    );
}
