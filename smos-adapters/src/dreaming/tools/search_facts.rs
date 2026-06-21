//! `search_facts` dreaming tool.

use std::sync::Arc;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::{Value, json};
use smos_application::ports::{EmbeddingProvider, FactRepository};

use super::ToolError;
use super::shared::parse_memory_key;
use crate::OllamaEmbedding;
use crate::storage::surreal_store::SurrealStore;

/// Semantic nearest-neighbour search for facts in a memory namespace.
///
/// Embeds the query via the injected [`OllamaEmbedding`], then delegates to
/// the production [`FactRepository::search_similar`] path. The embedder is a
/// concrete type rather than `dyn EmbeddingProvider` because the application
/// port uses native async fn in trait whose future is not automatically
/// `Send` — rig's `Tool::call` requires the returned future to be
/// `Send + Sync`, and the concrete type's future satisfies that (its
/// implementation goes through `tokio::spawn_blocking` which produces a
/// `Send` future).
pub struct SearchFactsTool {
    pub store: SurrealStore,
    pub embedder: Arc<OllamaEmbedding>,
}

#[derive(Debug, Deserialize)]
pub struct SearchFactsArgs {
    pub memory_key: String,
    pub query: String,
    #[serde(default = "default_search_limit")]
    pub limit: usize,
}

fn default_search_limit() -> usize {
    10
}

impl Tool for SearchFactsTool {
    const NAME: &'static str = "search_facts";
    type Args = SearchFactsArgs;
    type Output = Value;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "Semantic search over accepted facts in a memory \
                          namespace. Returns nearest neighbours by cosine \
                          similarity with their distance and stored metadata."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "memory_key": {"type": "string"},
                    "query": {"type": "string", "description": "Text to embed and match."},
                    "limit": {"type": "integer", "minimum": 1, "description": "Max hits (default 10)."}
                },
                "required": ["memory_key", "query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.query.trim().is_empty() {
            return Err(ToolError::InvalidInput("query must not be empty".into()));
        }
        let mk = parse_memory_key(&args.memory_key)?;
        let embedding = self
            .embedder
            .embed(&args.query)
            .await?
            .ok_or_else(|| ToolError::Provider("embedder returned no vector".into()))?;
        let hits = self
            .store
            .search_similar(embedding, &mk, args.limit.max(1))
            .await?;
        let view: Vec<Value> = hits
            .iter()
            .map(|h| {
                json!({
                    "id": h.id.as_str(),
                    "document": h.document,
                    "status": h.metadata.status,
                    "confidence": h.metadata.confidence,
                    "distance": h.metadata.distance,
                })
            })
            .collect();
        tracing::info!(
            tool = Self::NAME,
            memory_key = %args.memory_key,
            returned = view.len(),
            "search_facts"
        );
        Ok(json!({ "hits": view }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_facts_args_uses_default_limit_when_omitted() {
        let args: SearchFactsArgs =
            serde_json::from_str(r#"{"memory_key":"origa","query":"rust"}"#).expect("parse");
        assert_eq!(args.limit, 10);
        assert_eq!(args.query, "rust");
    }
}
