//! `smos audit` — manual one-shot dreaming audit runner.
//!
//! Mirrors what the scheduler does on every cron tick, but blocks the
//! foreground so an operator can watch the audit progress in real time.
//! Returns a non-zero exit code if the audit failed so the runner is
//! scriptable from CI or a watchdog.

use std::sync::Arc;

use anyhow::Result;
use smos_application::ports::Clock;

use crate::SystemClock;
use crate::cli::tracing_setup::init_tracing_for_server;
use crate::config::SmosConfig;
use crate::dreaming::run_audit;
use crate::nli::build_classifier;
use crate::providers::OllamaEmbedding;
use crate::storage::surreal_store::SurrealStore;

/// The LLM provider selected for an audit run.
///
/// Type-safe replacement for the previous `Option<String>` field: a parse
/// error surfaces at clap parse time rather than at the provider match in
/// [`run_audit`], and the match in `run_audit_cli` becomes exhaustive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditProvider {
    Cloud,
    Local,
}

impl AuditProvider {
    /// Parse from a CLI string. Returns `Err` with a clear message so the
    /// operator sees the bad value at the `smos audit` invocation rather
    /// than at the LLM provider dispatch.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "cloud" => Ok(Self::Cloud),
            "local" => Ok(Self::Local),
            other => Err(anyhow::anyhow!(
                "invalid --provider {other:?}: expected 'cloud' or 'local'"
            )),
        }
    }

    fn as_config_str(self) -> &'static str {
        match self {
            Self::Cloud => "cloud",
            Self::Local => "local",
        }
    }
}

/// CLI args for the `smos audit` subcommand.
#[derive(Debug, Clone)]
pub struct AuditArgs {
    /// Override the configured LLM provider (`cloud` or `local`).
    pub provider: Option<AuditProvider>,
    /// Dry-run: validate the provider configuration and bail out BEFORE
    /// building the NLI backend / embedder. Verifies wiring without
    /// spending tokens or loading the ~643 MB ONNX model.
    pub dry_run: bool,
}

/// Run `smos audit`.
pub async fn run_audit_cli(config_path: &str, args: AuditArgs) -> Result<()> {
    let mut config = SmosConfig::load(config_path)?;
    init_tracing_for_server(&config.server);

    if let Some(provider) = args.provider {
        config.audit.llm_provider = provider.as_config_str().to_string();
        // Re-run validation after the override. `validate_audit_always`
        // checks the audit fields REGARDLESS of `audit.enabled` so a
        // `--provider cloud` invocation against a config with an empty
        // `cloud_base_url` fails at startup, not at the first provider
        // call. (`SmosConfig::validate` skips audit checks when
        // `enabled=false`, which is correct for `smos serve` but wrong
        // for the manual runner where the operator explicitly opted in.)
        config.validate_audit_always()?;
    }

    // Dry-run shortcut: validate provider wiring BEFORE building any ML
    // infrastructure. A previous version of this runner built the NLI
    // classifier + embedder first and THEN bailed out of `--dry-run`,
    // which defeated the "verify wiring without paying the model-load cost"
    // purpose of the flag.
    if args.dry_run {
        tracing::info!(
            provider = %config.audit.llm_provider,
            cloud_model = %config.audit.cloud_model,
            local_model = %config.audit.local_model,
            "audit dry-run: configuration validated, skipping LLM prompt and ML backend build"
        );
        return Ok(());
    }

    let store = SurrealStore::connect(
        &config.surreal.path,
        &config.surreal.namespace,
        &config.surreal.database,
    )
    .await?;
    store.run_migrations().await?;

    // The audit needs an NLI classifier (for `nli_classify`) and an embedder
    // (for `search_facts`). Both are constructed eagerly so a missing
    // dependency fails the audit at startup rather than mid-conversation
    // when the LLM calls the corresponding tool.
    //
    // The reranker is intentionally NOT constructed: the dreaming audit
    // tools do not call it, and an unreachable reranker endpoint would
    // block an audit that otherwise would have run fine.
    let classifier = build_classifier(&config)
        .await
        .map_err(|e| anyhow::anyhow!("NLI classifier build failed: {e:#}"))?;
    let embedder = OllamaEmbedding::new(Arc::new(config.embedding.clone()))
        .map_err(|e| anyhow::anyhow!("Ollama embedder build failed: {e:#}"))?;
    let clock: Arc<dyn Clock + Send + Sync> = Arc::new(SystemClock);

    let audit_config = config.audit.clone();
    let report = run_audit(
        &audit_config,
        store,
        Arc::new(classifier),
        Arc::new(embedder),
        clock,
    )
    .await?;
    tracing::info!(
        deletions = report.deletions,
        merges = report.merges,
        "audit complete"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_provider_parse_accepts_known_values() {
        assert_eq!(AuditProvider::parse("cloud").unwrap(), AuditProvider::Cloud);
        assert_eq!(AuditProvider::parse("local").unwrap(), AuditProvider::Local);
    }

    #[test]
    fn audit_provider_parse_rejects_unknown() {
        assert!(AuditProvider::parse("garbage").is_err());
        assert!(AuditProvider::parse("").is_err());
    }

    #[test]
    fn audit_provider_round_trips_through_config_str() {
        for p in [AuditProvider::Cloud, AuditProvider::Local] {
            let s = p.as_config_str();
            assert_eq!(AuditProvider::parse(s).unwrap(), p);
        }
    }
}
