//! InMemoryFacts + InMemorySessions parity invariants.

use super::*;

#[tokio::test]
async fn save_then_list_accepted_returns_it() {
    let facts = InMemoryFacts::default();
    let f = accepted("the sky is blue", vec![0.1, 0.2]);
    facts.save(&f).await.unwrap();

    let listed = facts.list_accepted(&mk("testkit")).await.unwrap();
    let contents: Vec<String> = listed.iter().map(|x| x.content().to_string()).collect();
    assert!(
        contents.iter().any(|c| c == "the sky is blue"),
        "accepted fact must be listed: {contents:?}"
    );
}

#[tokio::test]
async fn save_then_get_roundtrips() {
    let facts = InMemoryFacts::default();
    let f = pending("roundtrip payload", vec![0.5]);
    let id = f.id().clone();
    facts.save(&f).await.unwrap();

    let got = facts.get(&id, &mk("testkit")).await.unwrap();
    let got = got.expect("fact must be retrievable");
    assert_eq!(got.id(), &id);
}

#[tokio::test]
async fn list_pending_filters_by_status() {
    let facts = InMemoryFacts::default();
    facts.seed(pending("p1", vec![0.1]));
    facts.seed(pending("p2", vec![0.2]));
    facts.seed(accepted("a1", vec![0.3]));

    let pending_list = facts.list_pending(&mk("testkit")).await.unwrap();
    assert_eq!(pending_list.len(), 2, "only pending facts returned");
    assert!(
        pending_list
            .iter()
            .all(|f| f.status() == FactStatus::Pending)
    );
}

#[tokio::test]
async fn list_keys_for_session_dedups_preserves_order() {
    let facts = InMemoryFacts::default();
    facts.seed(fact_in("in A", mk("alpha"), sid(1)));
    facts.seed(fact_in("in B", mk("beta"), sid(1)));
    facts.seed(fact_in("dup A again", mk("alpha"), sid(1)));
    facts.seed(fact_in("other session", mk("gamma"), sid(2)));

    let keys = facts.list_memory_keys_for_session(&sid(1)).await.unwrap();
    let key_strings: Vec<String> = keys.iter().map(|k| k.as_str().to_string()).collect();
    let deduped: std::collections::HashSet<&String> = key_strings.iter().collect();
    assert_eq!(deduped.len(), key_strings.len(), "no duplicate keys");
    assert_eq!(key_strings.len(), 2, "alpha and beta only");
    assert!(key_strings.iter().any(|k| k == "alpha"));
    assert!(key_strings.iter().any(|k| k == "beta"));
    assert!(
        !key_strings.iter().any(|k| k == "gamma"),
        "other session excluded"
    );
}

#[tokio::test]
async fn dedup_idempotent_on_same_session_fact() {
    let sessions = InMemorySessions::default();
    let candidate = vec![FactId::from_content("idempotent candidate")];

    let first = sessions
        .dedup_and_mark(&sid(1), &mk("testkit"), &candidate)
        .await
        .unwrap();
    let second = sessions
        .dedup_and_mark(&sid(1), &mk("testkit"), &candidate)
        .await
        .unwrap();

    assert_eq!(first.len(), 1, "first call returns the new candidate");
    assert!(second.is_empty(), "repeat call returns nothing");
}

#[tokio::test]
async fn inmemory_sessions_get_or_create_then_save_roundtrips() {
    let sessions = InMemorySessions::default();
    let mut state = sessions
        .get_or_create(&sid(1), &mk("testkit"))
        .await
        .unwrap();
    state.add_pending(&[FactId::from_content("pending one")]);
    sessions.save(&sid(1), &state).await.unwrap();

    assert_eq!(sessions.pending_of(&sid(1)).len(), 1);
}
