//! `list_facts` dreaming tool.

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::{Value, json};
use smos_application::ports::FactRepository;
use smos_domain::FactStatus;

use super::ToolError;
use super::shared::{fact_to_view, parse_memory_key, parse_status};
use crate::storage::surreal_store::SurrealStore;

/// Lists facts in a memory namespace, optionally filtered by status.
pub struct ListFactsTool {
    pub store: SurrealStore,
}

#[derive(Debug, Deserialize)]
pub struct ListFactsArgs {
    pub memory_key: String,
    pub status: Option<String>,
    /// Soft cap on the number of facts returned. Defaults to 100 when
    /// omitted so a fat namespace does not blow the LLM's context window.
    #[serde(default = "default_list_limit")]
    pub limit: usize,
}

fn default_list_limit() -> usize {
    100
}

impl Tool for ListFactsTool {
    const NAME: &'static str = "list_facts";
    type Args = ListFactsArgs;
    type Output = Value;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "List facts stored in a memory namespace, optionally \
                          filtered by status ('pending' | 'accepted' | 'rejected')."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "memory_key": {"type": "string", "description": "Namespace to query."},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "accepted", "rejected"],
                        "description": "Optional status filter."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Max facts to return (default 100)."
                    }
                },
                "required": ["memory_key"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let mk = parse_memory_key(&args.memory_key)?;
        let facts = match args.status.as_deref() {
            None => {
                let mut acc = self.store.list_accepted(&mk).await?;
                acc.extend(self.store.list_pending(&mk).await?);
                acc
            }
            Some(s) => {
                let status = parse_status(s)?;
                match status {
                    FactStatus::Accepted => self.store.list_accepted(&mk).await?,
                    FactStatus::Pending => self.store.list_pending(&mk).await?,
                    FactStatus::Rejected => Vec::new(),
                }
            }
        };
        let view: Vec<Value> = facts
            .iter()
            .take(args.limit.max(1))
            .map(fact_to_view)
            .collect();
        tracing::info!(
            tool = Self::NAME,
            memory_key = %args.memory_key,
            status = ?args.status,
            returned = view.len(),
            total = facts.len(),
            "list_facts"
        );
        Ok(json!({ "facts": view, "total": facts.len() }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_facts_args_default_limit_is_100() {
        let args: ListFactsArgs = serde_json::from_str(r#"{"memory_key":"origa"}"#).expect("parse");
        assert_eq!(args.limit, 100);
        assert_eq!(args.memory_key, "origa");
        assert!(args.status.is_none());
    }

    #[test]
    fn list_facts_args_accepts_explicit_status_and_limit() {
        let args: ListFactsArgs =
            serde_json::from_str(r#"{"memory_key":"origa","status":"pending","limit":5}"#)
                .expect("parse");
        assert_eq!(args.status.as_deref(), Some("pending"));
        assert_eq!(args.limit, 5);
    }
}
