//! `list_memory_keys` dreaming tool — namespace discovery.
//!
//! The LLM cannot infer which `memory_key` values exist in the store, so the
//! `count_facts` / `list_facts` family of per-namespace tools is unusable
//! without this entry point. Calling `list_memory_keys` first is the
//! discovery step that lets the auditor iterate over real namespaces instead
//! of guessing (and emitting `memory_key: ""`, which the domain rejects).

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::{Value, json};
use smos_application::ports::FactRepository;

use super::ToolError;
use crate::storage::surreal_store::SurrealStore;

/// Lists every distinct `memory_key` namespace present in the fact store.
pub struct ListMemoryKeysTool {
    pub store: SurrealStore,
}

/// Tool input. The tool takes no parameters — an empty JSON object is the
/// only valid shape — so the struct deliberately has no fields.
#[derive(Debug, Deserialize)]
pub struct ListMemoryKeysArgs {}

impl Tool for ListMemoryKeysTool {
    const NAME: &'static str = "list_memory_keys";
    type Args = ListMemoryKeysArgs;
    type Output = Value;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "List every distinct memory_key (namespace) stored in \
                          the database. Call this FIRST to discover which \
                          namespaces exist before calling count_facts or \
                          list_facts on a specific namespace."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let keys = self.store.list_memory_keys().await?;
        let view: Vec<String> = keys.iter().map(|k| k.as_str().to_string()).collect();
        tracing::info!(tool = Self::NAME, count = view.len(), "list_memory_keys");
        Ok(json!({ "memory_keys": view }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_memory_keys_args_deserializes_empty_object() {
        let args: ListMemoryKeysArgs = serde_json::from_str("{}").expect("parse empty object");
        let _ = args;
    }

    #[test]
    fn tool_name_is_stable() {
        assert_eq!(ListMemoryKeysTool::NAME, "list_memory_keys");
    }
}
