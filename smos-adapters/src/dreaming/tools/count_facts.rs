//! `count_facts` dreaming tool.

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::{Value, json};
use smos_application::ports::FactRepository;

use super::ToolError;
use super::shared::parse_memory_key;
use crate::storage::surreal_store::SurrealStore;

/// Count facts in a namespace, optionally filtered by status.
///
/// Counts pending + accepted together when no status is given because that is
/// the "what should the auditor inspect" view; `rejected` is excluded by
/// default (terminal state, no audit value).
pub struct CountFactsTool {
    pub store: SurrealStore,
}

#[derive(Debug, Deserialize)]
pub struct CountFactsArgs {
    pub memory_key: String,
    pub status: Option<String>,
}

impl Tool for CountFactsTool {
    const NAME: &'static str = "count_facts";
    type Args = CountFactsArgs;
    type Output = Value;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "Count facts in a memory namespace, optionally filtered \
                          by status. Defaults to accepted + pending."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "memory_key": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "accepted", "rejected"]
                    }
                },
                "required": ["memory_key"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let mk = parse_memory_key(&args.memory_key)?;
        let count = match args.status.as_deref() {
            None => {
                let accepted = self.store.list_accepted(&mk).await?;
                let pending = self.store.list_pending(&mk).await?;
                accepted.len() + pending.len()
            }
            Some("accepted") => self.store.list_accepted(&mk).await?.len(),
            Some("pending") => self.store.list_pending(&mk).await?.len(),
            Some("rejected") => 0,
            Some(other) => {
                return Err(ToolError::InvalidInput(format!(
                    "status must be 'pending' | 'accepted' | 'rejected', got {other:?}"
                )));
            }
        };
        Ok(json!({ "memory_key": args.memory_key, "count": count }))
    }
}
