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
use rig::providers::openrouter;
use smos_application::ports::Clock;

use super::prompts::{self, AUDIT_TRIGGER_PROMPT};
use super::report::AuditReport;
use super::tools::delete_fact::DeleteFactTool;
use super::tools::flag_conflict::FlagConflictTool;
use super::tools::list_memory_keys::ListMemoryKeysTool;
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
/// rig completion model. The two provider branches produce the same
/// concrete `CompletionModel` type, so the actual agent building + prompt
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
            // The local branch talks to a `llama-server` instance, which
            // speaks the OpenAI Chat Completions wire format. rig 0.14's
            // `ollama::Client` posts to the ollama-native `api/chat` path
            // and its `openai::Client` posts to the OpenAI Responses API
            // (`/responses`); neither is implemented by `llama-server`. The
            // `openrouter::Client` targets `/chat/completions` and attaches
            // only a Bearer header (ignored by an unauthenticated
            // `llama-server`), so it is the correct OpenAI-compatible
            // transport for the local model. `local_url` carries the host
            // root, so `/v1` is appended to match the chat-completions path.
            let base_url = local_audit_base_url(&config.local_url);
            let client = openrouter::Client::from_url("", &base_url);
            let model = client.completion_model(&config.local_model);
            run_audit_with_model(config, model, store, classifier, embedder, clock).await
        }
        other => Err(anyhow!("unknown audit.llm_provider: {other:?}")),
    }
}

/// Resolve the OpenAI-compatible base URL for the local audit provider.
///
/// `audit.local_url` carries the `llama-server` host root (e.g.
/// `http://localhost:28082`); the OpenAI Chat Completions transport the
/// local branch uses posts to `{base}/chat/completions`, so `/v1` must be
/// appended to match the path `llama-server` serves. Trailing slashes are
/// trimmed first so a configured `http://host:port/` normalises cleanly.
fn local_audit_base_url(local_url: &str) -> String {
    format!("{}/v1", local_url.trim_end_matches('/'))
}

/// Every concrete dependency the 11 audit tools need, grouped so the tool
/// assembly in [`attach_audit_tools`] reads by field name instead of a long
/// positional list.
struct AuditToolDeps {
    store: SurrealStore,
    classifier: Arc<NativeNliClassifier>,
    embedder: Arc<OllamaEmbedding>,
    limits: AuditLimits,
    merge_counter: Arc<AtomicUsize>,
    deletion_counter: Arc<AtomicUsize>,
    report_dir: PathBuf,
    clock: Arc<dyn Clock + Send + Sync>,
}

/// Attach the 11 dreaming audit tools to `builder` in the fixed registration
/// order (the same order the inline `.tool(...)` chain used before this
/// helper existed).
///
/// Returning a `Vec<...>` is not achievable with rig 0.14: its `Tool` trait is
/// `Sized` (not object-safe) and [`AgentBuilder::tool`] takes `impl Tool`, so
/// the heterogeneous tool instances cannot be collected into a single
/// container. Threading the builder through this function is the faithful
/// adaptation — it still centralises the tool list + registration order in one
/// named place, which is the actual goal of the slice.
fn attach_audit_tools<M: rig::completion::CompletionModel + 'static>(
    builder: AgentBuilder<M>,
    deps: AuditToolDeps,
) -> AgentBuilder<M> {
    let AuditToolDeps {
        store,
        classifier,
        embedder,
        limits,
        merge_counter,
        deletion_counter,
        report_dir,
        clock,
    } = deps;
    builder
        .tool(ListMemoryKeysTool {
            store: store.clone(),
        })
        .tool(ListFactsTool {
            store: store.clone(),
        })
        .tool(SearchFactsTool {
            store: store.clone(),
            embedder,
        })
        .tool(GetFactTool {
            store: store.clone(),
        })
        .tool(CountFactsTool {
            store: store.clone(),
        })
        .tool(NliClassifyTool { classifier })
        .tool(UpdateFactTool {
            store: store.clone(),
        })
        .tool(MergeFactsTool {
            store: store.clone(),
            limits,
            counter: merge_counter,
        })
        .tool(FlagConflictTool {
            store: store.clone(),
        })
        .tool(DeleteFactTool {
            store,
            limits,
            counter: deletion_counter,
            clock: clock.clone(),
        })
        .tool(WriteReportTool { report_dir, clock })
}

/// Generic audit runner: builds the agent with the supplied completion model
/// and prompts it. Generic over `M` so the cloud (OpenRouter) and local
/// (`llama-server` via the OpenAI-compatible transport) provider branches
/// unify on one body.
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
    // Fresh per-run mutation counters. The same Arc is shared with the bounded
    // write tools (cloned into `deps` below), so the final `.load()` reflects
    // every mutation the agent made through MergeFactsTool / DeleteFactTool.
    let merge_counter = Arc::new(AtomicUsize::new(0));
    let deletion_counter = Arc::new(AtomicUsize::new(0));

    let deps = AuditToolDeps {
        store,
        classifier,
        embedder,
        limits: AuditLimits {
            max_deletions: config.max_deletions_per_run,
            max_merges: config.max_merges_per_run,
        },
        merge_counter: merge_counter.clone(),
        deletion_counter: deletion_counter.clone(),
        report_dir: PathBuf::from(&config.report_dir),
        clock: clock.clone(),
    };

    let agent = attach_audit_tools(
        AgentBuilder::new(model).preamble(prompts::SYSTEM_PROMPT),
        deps,
    )
    .build();

    tracing::info!(
        provider = %config.llm_provider,
        cloud_model = %config.cloud_model,
        local_model = %config.local_model,
        max_deletions = config.max_deletions_per_run,
        max_merges = config.max_merges_per_run,
        "starting SMOS dreaming audit"
    );

    // rig 0.14's `PromptRequest` defaults `max_depth = 0` (single-turn), which
    // makes the tool-calling loop never engage — every prompt that needs a
    // tool surface fails with `MaxDepthError: (reached limit: 0)`. Configuring
    // a non-zero multi-turn depth is load-bearing for the audit workflow
    // because every fact query, mutation, and report write happens through a
    // `rig::tool::Tool` impl.
    let response = agent
        .prompt(AUDIT_TRIGGER_PROMPT)
        .multi_turn(config.max_tool_rounds)
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
