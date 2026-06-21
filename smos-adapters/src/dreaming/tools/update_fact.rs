//! `update_fact` dreaming tool.

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::{Value, json};
use smos_application::ports::FactRepository;
use smos_domain::{Confidence, Fact};

use super::ToolError;
use super::shared::{parse_memory_key, parse_status, rehydrate_with};
use crate::storage::surreal_store::SurrealStore;

/// Update a fact's confidence and/or status in place.
///
/// Not rate-limited: updates are not destructive and the LLM is unlikely to
/// generate excessive update calls in a single audit. If abuse becomes a real
/// problem, gate this tool behind a third `AuditLimits` field rather than
/// silently reusing the merge or deletion counter.
pub struct UpdateFactTool {
    pub store: SurrealStore,
}

#[derive(Debug, Deserialize)]
pub struct UpdateFactArgs {
    pub memory_key: String,
    pub fact_id: String,
    pub confidence: Option<f32>,
    pub status: Option<String>,
}

impl Tool for UpdateFactTool {
    const NAME: &'static str = "update_fact";
    type Args = UpdateFactArgs;
    type Output = Value;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "Update a fact's confidence and/or status in place. \
                          At least one of `confidence` or `status` must be \
                          supplied; omitted fields are left unchanged."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "memory_key": {"type": "string"},
                    "fact_id": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "status": {"type": "string", "enum": ["pending", "accepted", "rejected"]}
                },
                "required": ["memory_key", "fact_id"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.confidence.is_none() && args.status.is_none() {
            return Err(ToolError::InvalidInput(
                "at least one of `confidence` or `status` must be provided".into(),
            ));
        }
        let mk = parse_memory_key(&args.memory_key)?;
        let fid = super::shared::parse_fact_id(&args.fact_id)?;
        let mut fact = self.store.get(&fid, &mk).await?.ok_or_else(|| {
            ToolError::NotFound(format!("fact {} in {}", args.fact_id, args.memory_key))
        })?;
        apply_update(&mut fact, args.confidence, args.status.as_deref())?;
        self.store.save(&fact).await?;
        tracing::info!(
            tool = Self::NAME,
            fact_id = %args.fact_id,
            memory_key = %args.memory_key,
            confidence = ?args.confidence,
            status = ?args.status,
            "update_fact applied"
        );
        Ok(json!({
            "updated": args.fact_id,
            "status": fact.status().as_str(),
            "confidence": fact.confidence().value()
        }))
    }
}

/// Pure helper: mutate a `Fact` in place with the optional fields. Kept
/// free of IO so it can be unit-tested directly.
///
/// Implementation note: instead of going through `Fact::set_status_and_confidence`
/// (which enforces the workflow transition invariants — terminal statuses
/// cannot move), the audit needs to override an Accepted fact when correcting
/// a confidence score or marking an Accepted fact for deletion. The cleanest
/// path that respects every data invariant (id matches content, valid_until >
/// valid_from, confidence in [0,1]) without coupling the domain layer to
/// audit concerns is to rehydrate a new `Fact` with the swapped fields.
/// `rehydrate` runs every data-level check, so the result is just as sound as
/// the original row.
fn apply_update(
    fact: &mut Fact,
    confidence: Option<f32>,
    status: Option<&str>,
) -> Result<(), ToolError> {
    let new_confidence = match confidence {
        Some(c) => Confidence::new(c)?,
        None => fact.confidence(),
    };
    let new_status = match status {
        Some(s) => parse_status(s)?,
        None => fact.status(),
    };
    *fact = rehydrate_with(fact, new_confidence, new_status, fact.valid_until())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use smos_domain::{Embedding, FactStatus, MemoryKey, SessionId};

    fn fixture() -> Fact {
        let session = SessionId::from_raw("sess_aaaaaaaaaaaa").unwrap();
        let emb = Embedding::new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        Fact::new_pending(
            "hello",
            MemoryKey::from_raw("origa").unwrap(),
            session,
            emb,
            smos_domain::Timestamp::from_unix_secs(1_700_000_000).unwrap(),
            0.5,
        )
        .unwrap()
    }

    #[test]
    fn apply_update_with_no_fields_is_a_no_op() {
        let mut fact = fixture();
        let original_conf = fact.confidence().value();
        apply_update(&mut fact, None, None).expect("no-op");
        assert_eq!(fact.confidence().value(), original_conf);
        assert_eq!(fact.status(), FactStatus::Pending);
    }

    #[test]
    fn apply_update_rejects_bad_status_string() {
        let mut fact = fixture();
        let err = apply_update(&mut fact, None, Some("garbage")).expect_err("bad status");
        assert!(err.to_string().contains("status must be"));
    }

    #[test]
    fn apply_update_changes_confidence_and_status() {
        let mut fact = fixture();
        apply_update(&mut fact, Some(0.95), Some("accepted")).expect("update");
        assert_eq!(fact.confidence().value(), 0.95);
        assert_eq!(fact.status(), FactStatus::Accepted);
    }
}
