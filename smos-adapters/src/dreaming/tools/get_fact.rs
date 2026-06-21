//! `get_fact` dreaming tool.

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::Value;
use smos_application::ports::FactRepository;

use super::ToolError;
use super::shared::{fact_to_view, parse_fact_id, parse_memory_key};
use crate::storage::surreal_store::SurrealStore;

/// Fetch a single fact by id within a memory namespace.
pub struct GetFactTool {
    pub store: SurrealStore,
}

#[derive(Debug, Deserialize)]
pub struct GetFactArgs {
    pub memory_key: String,
    pub fact_id: String,
}

impl Tool for GetFactTool {
    const NAME: &'static str = "get_fact";
    type Args = GetFactArgs;
    type Output = Value;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "Fetch a single fact by id within a memory namespace.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "memory_key": {"type": "string"},
                    "fact_id": {"type": "string", "description": "SHA1 hex FactId."}
                },
                "required": ["memory_key", "fact_id"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let mk = parse_memory_key(&args.memory_key)?;
        let fact_id = parse_fact_id(&args.fact_id)?;
        match self.store.get(&fact_id, &mk).await? {
            Some(fact) => Ok(fact_to_view(&fact)),
            None => Err(ToolError::NotFound(format!(
                "fact {} in memory_key {}",
                args.fact_id, args.memory_key
            ))),
        }
    }
}
