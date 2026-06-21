//! `merge_facts` dreaming tool.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::{Value, json};
use smos_application::ports::FactRepository;

use super::shared::{acquire_slot, parse_fact_id, parse_memory_key};
use super::{AuditLimits, ToolError};
use crate::storage::surreal_store::SurrealStore;

/// Merge two facts: union source_sessions + conflicts_with of `source_id`
/// into `target_id`, then re-save `target_id`. The source fact is NOT
/// deleted (the LLM must call `delete_fact` separately once it has confirmed
/// the merge). This two-step protocol keeps the operation recoverable: a
/// wrong merge call never loses data.
pub struct MergeFactsTool {
    pub store: SurrealStore,
    pub limits: AuditLimits,
    pub counter: Arc<AtomicUsize>,
}

#[derive(Debug, Deserialize)]
pub struct MergeFactsArgs {
    pub memory_key: String,
    pub source_id: String,
    pub target_id: String,
}

impl Tool for MergeFactsTool {
    const NAME: &'static str = "merge_facts";
    type Args = MergeFactsArgs;
    type Output = Value;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "Merge source fact into target fact within the same \
                          memory_key. Unions source_sessions and conflicts_with \
                          onto the target. The source fact is NOT deleted; the \
                          caller must delete_fact it explicitly after verifying \
                          the merge."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "memory_key": {"type": "string"},
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"}
                },
                "required": ["memory_key", "source_id", "target_id"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.source_id == args.target_id {
            return Err(ToolError::InvalidInput(
                "source_id and target_id must differ".into(),
            ));
        }
        acquire_slot(&self.counter, self.limits.max_merges, "max_merges_per_run")?;
        let mk = parse_memory_key(&args.memory_key)?;
        let source_fid = parse_fact_id(&args.source_id)?;
        let target_fid = parse_fact_id(&args.target_id)?;
        let mut target = self
            .store
            .get(&target_fid, &mk)
            .await?
            .ok_or_else(|| ToolError::NotFound(format!("target fact {}", args.target_id)))?;
        let source = self
            .store
            .get(&source_fid, &mk)
            .await?
            .ok_or_else(|| ToolError::NotFound(format!("source fact {}", args.source_id)))?;
        target.merge_into(&source)?;
        self.store.save(&target).await?;
        tracing::info!(
            tool = Self::NAME,
            source_id = %args.source_id,
            target_id = %args.target_id,
            "merge_facts applied"
        );
        Ok(json!({"merged_into": args.target_id, "source": args.source_id}))
    }
}
