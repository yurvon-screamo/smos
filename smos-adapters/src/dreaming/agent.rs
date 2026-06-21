//! `run_audit` entry point and rig agent wiring.
//!
//! The agent is built once per audit run with fresh rate-limit counters, then
//! prompted with a single instruction that kicks off the full audit workflow
//! described in [`super::prompts::SYSTEM_PROMPT`]. rig's tool-calling loop
//! executes the actual fact queries and mutations; the per-tool atomic
//! counters record exactly how many deletions and merges happened.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, anyhow};
use rig::agent::AgentBuilder;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::{ollama, openrouter};
use smos_application::ports::Clock;

use super::prompts::{self, AUDIT_TRIGGER_PROMPT};
use super::report::AuditReport;
use super::tools::delete_fact::DeleteFactTool;
use super::tools::flag_conflict::FlagConflictTool;
use super::tools::merge_facts::MergeFactsTool;
use super::tools::update_fact::UpdateFactTool;
use super::tools::write_report::WriteReportTool;
use super::tools::{
    AuditLimits, CountFactsTool, GetFactTool, ListFactsTool, NliClassifyTool, SearchFactsTool,
};
use crate::config::AuditConfig;
use crate::storage::surreal_store::SurrealStore;
use crate::{NativeNliClassifier, OllamaEmbedding};

/// Resolve `"${ENV_VAR}"` placeholders in a config string. Returns the
/// literal verbatim when it is not a placeholder; returns an empty string
/// when the placeholder env var is unset (so the downstream caller can
/// surface a clear auth error rather than panicking).
pub fn resolve_env_var(value: &str) -> String {
    if let Some(var) = value.strip_prefix("${").and_then(|s| s.strip_suffix('}')) {
        std::env::var(var).unwrap_or_default()
    } else {
        value.to_string()
    }
}

/// Run one audit using the configured provider.
///
/// Dispatches on [`AuditConfig::llm_provider`] and constructs the matching
/// rig completion model. The two provider branches produce different
/// concrete `CompletionModel` types, so the actual agent building + prompt
/// loop is delegated to the generic [`run_audit_with_model`].
pub async fn run_audit(
    config: &AuditConfig,
    store: SurrealStore,
    classifier: Arc<NativeNliClassifier>,
    embedder: Arc<OllamaEmbedding>,
    clock: Arc<dyn Clock + Send + Sync>,
) -> anyhow::Result<AuditReport> {
    match config.llm_provider.as_str() {
        "cloud" => {
            let api_key = resolve_env_var(&config.cloud_api_key);
            // Fail-fast on missing API key. The audit is typically a cron
            // job; surfacing the auth error at server startup or at the
            // manual `smos audit` invocation is far more useful than letting
            // the first cron tick discover the missing key via a 401 from
            // the LLM provider at 03:00 UTC.
            if api_key.trim().is_empty() {
                return Err(anyhow!(
                    "audit.cloud_api_key resolved to an empty string — set the \
                     env var referenced by cloud_api_key (or pass a literal key) \
                     before enabling the audit"
                ));
            }
            let client = openrouter::Client::from_url(&api_key, &config.cloud_base_url);
            let model = client.completion_model(&config.cloud_model);
            run_audit_with_model(config, model, store, classifier, embedder, clock).await
        }
        "local" => {
            let client = ollama::Client::from_url(&config.local_url);
            let model = client.completion_model(&config.local_model);
            run_audit_with_model(config, model, store, classifier, embedder, clock).await
        }
        other => Err(anyhow!("unknown audit.llm_provider: {other:?}")),
    }
}

/// Generic audit runner: builds the agent with the supplied completion model
/// and prompts it. Generic over `M` so the cloud (OpenRouter) and local
/// (Ollama) provider branches unify on one body.
async fn run_audit_with_model<M>(
    config: &AuditConfig,
    model: M,
    store: SurrealStore,
    classifier: Arc<NativeNliClassifier>,
    embedder: Arc<OllamaEmbedding>,
    clock: Arc<dyn Clock + Send + Sync>,
) -> anyhow::Result<AuditReport>
where
    M: rig::completion::CompletionModel + 'static,
{
    let deletion_counter = Arc::new(AtomicUsize::new(0));
    let merge_counter = Arc::new(AtomicUsize::new(0));
    let limits = AuditLimits {
        max_deletions: config.max_deletions_per_run,
        max_merges: config.max_merges_per_run,
    };
    let report_dir = PathBuf::from(&config.report_dir);

    let agent = AgentBuilder::new(model)
        .preamble(prompts::SYSTEM_PROMPT)
        .tool(ListFactsTool {
            store: store.clone(),
        })
        .tool(SearchFactsTool {
            store: store.clone(),
            embedder: embedder.clone(),
        })
        .tool(GetFactTool {
            store: store.clone(),
        })
        .tool(CountFactsTool {
            store: store.clone(),
        })
        .tool(NliClassifyTool {
            classifier: classifier.clone(),
        })
        .tool(UpdateFactTool {
            store: store.clone(),
        })
        .tool(MergeFactsTool {
            store: store.clone(),
            limits,
            counter: merge_counter.clone(),
        })
        .tool(FlagConflictTool {
            store: store.clone(),
        })
        .tool(DeleteFactTool {
            store: store.clone(),
            limits,
            counter: deletion_counter.clone(),
            clock: clock.clone(),
        })
        .tool(WriteReportTool {
            report_dir,
            clock: clock.clone(),
        })
        .build();

    tracing::info!(
        provider = %config.llm_provider,
        cloud_model = %config.cloud_model,
        local_model = %config.local_model,
        max_deletions = config.max_deletions_per_run,
        max_merges = config.max_merges_per_run,
        "starting SMOS dreaming audit"
    );

    let response = agent
        .prompt(AUDIT_TRIGGER_PROMPT)
        .await
        .context("audit agent prompt failed")?;
    let deletions = deletion_counter.load(Ordering::Relaxed);
    let merges = merge_counter.load(Ordering::Relaxed);
    let timestamp = clock.now();
    tracing::info!(deletions, merges, "audit complete");
    Ok(AuditReport {
        deletions,
        merges,
        response,
        timestamp,
    })
}

#[cfg(test)]
#[path = "agent_tests.rs"]
mod tests;
