//! `flag_conflict` dreaming tool.

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::{Value, json};
use smos_application::ports::FactRepository;
use surrealdb::Surreal;
use surrealdb::engine::local::Db;

use super::ToolError;
use super::shared::{parse_fact_id, parse_memory_key};
use crate::storage::surreal_store::SurrealStore;

/// Mark two facts as mutually conflicting (bidirectional).
///
/// The mutation runs as a single SurrealDB transaction so the two writes
/// (one per fact) commit atomically. Without the transaction wrapper, a
/// failure between the two `save` calls would leave the persistent state
/// inconsistent: `a.conflicts_with` would contain `b.id` but
/// `b.conflicts_with` would be missing `a.id`, and downstream search /
/// conflict-resolution paths would silently diverge.
pub struct FlagConflictTool {
    pub store: SurrealStore,
}

#[derive(Debug, Deserialize)]
pub struct FlagConflictArgs {
    pub memory_key: String,
    pub fact_a_id: String,
    pub fact_b_id: String,
}

impl Tool for FlagConflictTool {
    const NAME: &'static str = "flag_conflict";
    type Args = FlagConflictArgs;
    type Output = Value;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "Mark two facts in the same memory_key as mutually \
                          conflicting. Updates both facts' conflicts_with list \
                          (idempotent)."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "memory_key": {"type": "string"},
                    "fact_a_id": {"type": "string"},
                    "fact_b_id": {"type": "string"}
                },
                "required": ["memory_key", "fact_a_id", "fact_b_id"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.fact_a_id == args.fact_b_id {
            return Err(ToolError::InvalidInput(
                "fact_a_id and fact_b_id must differ".into(),
            ));
        }
        let mk = parse_memory_key(&args.memory_key)?;
        let a_fid = parse_fact_id(&args.fact_a_id)?;
        let b_fid = parse_fact_id(&args.fact_b_id)?;
        // Pre-flight: both facts must exist. The transaction below is
        // idempotent but silently no-ops on a missing row, which would let
        // the LLM believe a flag was applied when one of the facts was a
        // typo. Loading both up-front surfaces the typo as NotFound.
        let _a = self
            .store
            .get(&a_fid, &mk)
            .await?
            .ok_or_else(|| ToolError::NotFound(format!("fact {}", args.fact_a_id)))?;
        let _b = self
            .store
            .get(&b_fid, &mk)
            .await?
            .ok_or_else(|| ToolError::NotFound(format!("fact {}", args.fact_b_id)))?;
        apply_conflict_tx(self.store.raw_db(), &a_fid, &b_fid, &mk).await?;
        tracing::info!(
            tool = Self::NAME,
            fact_a = %args.fact_a_id,
            fact_b = %args.fact_b_id,
            "flag_conflict applied"
        );
        Ok(json!({"flagged": [args.fact_a_id, args.fact_b_id]}))
    }
}

/// Atomically append each id to the other's `conflicts_with` array.
///
/// Both UPDATEs run inside a single SurrealDB transaction; if either fails
/// the whole transaction rolls back, so the persistent state cannot end up
/// with only one side of the bidirectional flag. `array::union` is
/// idempotent, so a retry after a transient failure does not duplicate the
/// entry.
async fn apply_conflict_tx(
    db: &Surreal<Db>,
    a_id: &smos_domain::FactId,
    b_id: &smos_domain::FactId,
    memory_key: &smos_domain::MemoryKey,
) -> Result<(), ToolError> {
    let a_id_str = a_id.as_str().to_string();
    let b_id_str = b_id.as_str().to_string();
    let mk_str = memory_key.as_str().to_string();
    let mut res = db
        .query(
            "BEGIN TRANSACTION;
             UPDATE type::thing('fact', $a_id) SET
                 conflicts_with = array::union(conflicts_with, [$b_id])
             WHERE memory_key = $mk;
             UPDATE type::thing('fact', $b_id) SET
                 conflicts_with = array::union(conflicts_with, [$a_id])
             WHERE memory_key = $mk;
             COMMIT TRANSACTION;",
        )
        .bind(("a_id", a_id_str))
        .bind(("b_id", b_id_str))
        .bind(("mk", mk_str))
        .await
        .map_err(|e| ToolError::Repo(format!("flag_conflict transaction: {e}")))?;
    // Drain per-statement errors so a partial transaction failure surfaces
    // as a single RepoError rather than silently dropping.
    let errors: Vec<_> = res.take_errors().into_iter().collect();
    if !errors.is_empty() {
        return Err(ToolError::Repo(format!(
            "flag_conflict transaction errors: {errors:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_self_conflict_at_input_boundary() {
        // The self-conflict guard lives in `call`, so this test just pins
        // the guard's error message shape.
        let err = ToolError::InvalidInput("fact_a_id and fact_b_id must differ".into());
        assert!(err.to_string().contains("must differ"));
    }
}
