//! Shared helpers used across multiple dreaming tools.
//!
//! Centralises the small parsing / projection helpers that every tool needs
//! (memory_key parsing, status parsing, the JSON view of a `Fact`, the
//! rate-limit slot acquisition, and the audit-privileged rehydrate). Keeping
//! these in one module removes the duplication that previously existed
//! across `read.rs`, `search.rs`, and `mutate.rs`.

use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};
use smos_domain::{Confidence, Fact, FactId, FactStatus, MemoryKey, Timestamp};

use super::ToolError;

/// Parse a `MemoryKey` from a tool-input string.
///
/// Returns [`ToolError::InvalidInput`] with a clear prefix so the LLM sees
/// "memory_key: <reason>" in the tool error and can self-correct.
pub fn parse_memory_key(s: &str) -> Result<MemoryKey, ToolError> {
    MemoryKey::from_raw(s).map_err(|e| ToolError::InvalidInput(format!("memory_key: {e}")))
}

/// Parse a `FactId` from a tool-input string.
pub fn parse_fact_id(s: &str) -> Result<FactId, ToolError> {
    FactId::from_raw(s).map_err(|e| ToolError::InvalidInput(format!("fact_id: {e}")))
}

/// Parse a fact status keyword from a tool-input string.
pub fn parse_status(s: &str) -> Result<FactStatus, ToolError> {
    match s {
        "pending" => Ok(FactStatus::Pending),
        "accepted" => Ok(FactStatus::Accepted),
        "rejected" => Ok(FactStatus::Rejected),
        other => Err(ToolError::InvalidInput(format!(
            "status must be 'pending' | 'accepted' | 'rejected', got {other:?}"
        ))),
    }
}

/// Compact JSON view of a `Fact` returned to the LLM by the read tools.
///
/// The full `Fact` aggregate exposes ~15 fields; surfacing all of them in the
/// tool output would burn context tokens on data the LLM does not need (e.g.
/// `embedding`, `last_access_at`). This view trims the projection to the
/// fields the auditor's prompt actually instructs the LLM to consult.
pub fn fact_to_view(fact: &Fact) -> Value {
    json!({
        "id": fact.id().as_str(),
        "memory_key": fact.memory_key().as_str(),
        "content": fact.content(),
        "fact_type": fact.fact_type().as_str(),
        "confidence": fact.confidence().value(),
        "status": fact.status().as_str(),
        "valid_from": fact.valid_from().as_unix_secs(),
        "valid_until": fact.valid_until().map(|ts| ts.as_unix_secs()),
        "extracted_at": fact.extracted_at().as_unix_secs(),
        "source_sessions_count": fact.source_sessions().distinct_count(),
        "conflicts_with": fact.conflicts_with().iter().map(|c| c.as_str()).collect::<Vec<_>>(),
        "heat_base": fact.heat_base().value(),
    })
}

/// Acquire a slot against the per-run cap. Returns `Err(RateLimitExceeded)`
/// if the counter is already at-or-past `cap`; otherwise atomically
/// increments and returns the new count.
///
/// `Ordering::Relaxed` is sufficient: the counters are observed only by the
/// single audit task (rig's tool-calling loop is sequential within one
/// `prompt` call), and there is no other memory operation that needs to be
/// ordered relative to the increment.
pub fn acquire_slot(
    counter: &AtomicUsize,
    cap: usize,
    label: &'static str,
) -> Result<usize, ToolError> {
    let prior = counter.load(Ordering::Relaxed);
    if prior >= cap {
        return Err(ToolError::RateLimitExceeded(label));
    }
    Ok(counter.fetch_add(1, Ordering::Relaxed) + 1)
}

/// Rehydrate `source` with swapped confidence / status / valid_until. All
/// other fields are copied verbatim from `source`.
///
/// Used by the write tools that need to bypass the workflow transition
/// invariants (e.g. the audit must be able to demote an `Accepted` fact to
/// `Rejected`). The rehydrate path enforces every DATA invariant
/// (`valid_until > valid_from`, `id == FactId::from_content(content)`,
/// confidence in `[0,1]`) so the result is just as sound as the original row.
pub fn rehydrate_with(
    source: &Fact,
    confidence: Confidence,
    status: FactStatus,
    valid_until: Option<Timestamp>,
) -> Result<Fact, ToolError> {
    Ok(Fact::rehydrate(
        source.id().clone(),
        source.memory_key().clone(),
        smos_domain::FactContent::new(source.content().to_string())?,
        source.fact_type(),
        confidence,
        status,
        source.valid_from(),
        valid_until,
        source.extracted_at(),
        source.source_sessions().clone(),
        source.conflicts_with().to_vec(),
        source.heat_base(),
        source.last_access_at(),
        source.embedding().cloned(),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_memory_key_rejects_empty() {
        assert!(parse_memory_key("").is_err());
    }

    #[test]
    fn parse_status_round_trips_known_values() {
        assert_eq!(parse_status("pending").unwrap(), FactStatus::Pending);
        assert_eq!(parse_status("accepted").unwrap(), FactStatus::Accepted);
        assert_eq!(parse_status("rejected").unwrap(), FactStatus::Rejected);
        assert!(parse_status("garbage").is_err());
    }

    #[test]
    fn acquire_slot_rejects_at_cap() {
        let counter = AtomicUsize::new(0);
        let _ = acquire_slot(&counter, 2, "test").unwrap();
        let _ = acquire_slot(&counter, 2, "test").unwrap();
        let err = acquire_slot(&counter, 2, "test").expect_err("cap reached");
        assert!(matches!(err, ToolError::RateLimitExceeded("test")));
    }

    #[test]
    fn rehydrate_with_preserves_unchanged_fields() {
        let session = smos_domain::SessionId::from_raw("sess_aaaaaaaaaaaa").unwrap();
        let emb = smos_domain::Embedding::new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let fact = Fact::new_pending(
            "hello",
            MemoryKey::from_raw("origa").unwrap(),
            session,
            emb,
            Timestamp::from_unix_secs(1_700_000_000).unwrap(),
            0.5,
        )
        .unwrap();
        let rebuilt = rehydrate_with(
            &fact,
            Confidence::new(0.99).unwrap(),
            FactStatus::Rejected,
            None,
        )
        .expect("rehydrate");
        assert_eq!(rebuilt.content(), "hello");
        assert_eq!(rebuilt.confidence().value(), 0.99);
        assert_eq!(rebuilt.status(), FactStatus::Rejected);
        assert_eq!(rebuilt.memory_key().as_str(), "origa");
    }
}
