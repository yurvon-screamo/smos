//! `delete_fact` dreaming tool.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::{Value, json};
use smos_application::ports::{Clock, FactRepository};
use smos_domain::FactStatus;

use super::shared::{acquire_slot, parse_fact_id, parse_memory_key, rehydrate_with};
use super::{AuditLimits, ToolError};
use crate::storage::surreal_store::SurrealStore;

/// Delete a fact by setting its status to `Rejected` and tombstoning it with
/// `valid_until = now`. The fact row is preserved (not physically removed) so
/// the operation is reversible via a subsequent `update_fact`.
pub struct DeleteFactTool {
    pub store: SurrealStore,
    pub limits: AuditLimits,
    pub counter: Arc<AtomicUsize>,
    /// Wall-clock captured per call so the tombstone timestamp reflects when
    /// the deletion happened, not when the audit run started. Uses the
    /// application-layer `Clock` port directly so the same `Arc<dyn Clock>`
    /// that drives the watcher / extraction can be shared with the audit.
    pub clock: Arc<dyn Clock + Send + Sync>,
}

#[derive(Debug, Deserialize)]
pub struct DeleteFactArgs {
    pub memory_key: String,
    pub fact_id: String,
}

impl Tool for DeleteFactTool {
    const NAME: &'static str = "delete_fact";
    type Args = DeleteFactArgs;
    type Output = Value;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "Soft-delete a fact: mark it Rejected and set \
                          valid_until to now. The fact row is preserved so \
                          the operation is reversible via update_fact. \
                          Rate-limited by max_deletions_per_run."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "memory_key": {"type": "string"},
                    "fact_id": {"type": "string"}
                },
                "required": ["memory_key", "fact_id"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        acquire_slot(
            &self.counter,
            self.limits.max_deletions,
            "max_deletions_per_run",
        )?;
        let mk = parse_memory_key(&args.memory_key)?;
        let fid = parse_fact_id(&args.fact_id)?;
        let fact = self.store.get(&fid, &mk).await?.ok_or_else(|| {
            ToolError::NotFound(format!("fact {} in {}", args.fact_id, args.memory_key))
        })?;
        // Soft-delete = tombstone + Reject. Rehydrate path enforces the
        // data invariant `valid_until > valid_from` while bypassing the
        // workflow transition rule that would otherwise block
        // Accepted -> Rejected (the audit must be able to delete an
        // already-Accepted fact when the LLM has reasoned that it is noise).
        let tombstone = self.clock.now();
        let deleted = rehydrate_with(
            &fact,
            fact.confidence(),
            FactStatus::Rejected,
            Some(tombstone),
        )?;
        self.store.save(&deleted).await?;
        tracing::info!(
            tool = Self::NAME,
            fact_id = %args.fact_id,
            memory_key = %args.memory_key,
            "delete_fact applied (soft delete)"
        );
        Ok(json!({"deleted": args.fact_id}))
    }
}
