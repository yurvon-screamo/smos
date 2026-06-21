//! `nli_classify` dreaming tool.

use std::sync::Arc;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::{Value, json};
use smos_application::ports::NliClassifier;

use super::ToolError;
use crate::NativeNliClassifier;

/// Natural-language-inference verdict between a premise and a hypothesis.
///
/// Calls the injected classifier (the same `NativeNliClassifier` instance the
/// session watcher uses, so the audit and finalize paths share the same
/// softmax thresholds and model weights).
pub struct NliClassifyTool {
    pub classifier: Arc<NativeNliClassifier>,
}

#[derive(Debug, Deserialize)]
pub struct NliClassifyArgs {
    pub premise: String,
    pub hypothesis: String,
}

impl Tool for NliClassifyTool {
    const NAME: &'static str = "nli_classify";
    type Args = NliClassifyArgs;
    type Output = Value;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "Run natural-language-inference on a premise / \
                          hypothesis pair. Returns label and per-class \
                          softmax scores (entailment / neutral / contradiction)."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "premise": {"type": "string"},
                    "hypothesis": {"type": "string"}
                },
                "required": ["premise", "hypothesis"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.premise.trim().is_empty() || args.hypothesis.trim().is_empty() {
            return Err(ToolError::InvalidInput(
                "premise and hypothesis must not be empty".into(),
            ));
        }
        let result = self
            .classifier
            .classify(&args.premise, &args.hypothesis)
            .await?;
        Ok(json!({
            "label": result.label.as_str(),
            "available": result.available,
            "scores": {
                "entailment": result.scores.entailment,
                "neutral": result.scores.neutral,
                "contradiction": result.scores.contradiction,
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nli_classify_args_parses_pair() {
        let args: NliClassifyArgs =
            serde_json::from_str(r#"{"premise":"a","hypothesis":"b"}"#).expect("parse");
        assert_eq!(args.premise, "a");
        assert_eq!(args.hypothesis, "b");
    }
}
