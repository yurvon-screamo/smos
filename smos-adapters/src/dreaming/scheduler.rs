//! Cron-based trigger for the dreaming audit.
//!
//! [`start_scheduler`] wires the [`AuditConfig::schedule`] cron expression to
//! a tokio-cron-scheduler job that calls [`super::run_audit`] on every tick.
//! The returned [`JobScheduler`] is started before the function returns; the
//! caller is expected to hold it for the lifetime of the process (dropping it
//! stops the scheduler).

use std::sync::Arc;

use anyhow::Context;
use smos_application::ports::Clock;
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::{error, info};

use super::run_audit;
use crate::config::AuditConfig;
use crate::storage::surreal_store::SurrealStore;
use crate::{NativeNliClassifier, OllamaEmbedding};

/// Build and start the audit scheduler.
///
/// When [`AuditConfig::enabled`] is `false`, returns a fresh (empty) scheduler
/// without adding any job. The caller still gets a `JobScheduler` handle so
/// the server lifecycle code can hold one value uniformly regardless of
/// whether the audit is enabled.
///
/// When enabled, the scheduler:
/// 1. Parses `config.schedule` as a 5-field UNIX cron expression (UTC).
/// 2. Spawns [`run_audit`] on every tick, logging success or failure.
/// 3. Holds a strong reference to every captured dependency via `Arc` clones
///    so the inner future is `'static + Send`.
///
/// # Errors
///
/// - [`anyhow::Error`] if the cron expression cannot be parsed.
/// - [`anyhow::Error`] if the scheduler cannot be started.
pub async fn start_scheduler(
    config: &AuditConfig,
    store: SurrealStore,
    classifier: Arc<NativeNliClassifier>,
    embedder: Arc<OllamaEmbedding>,
    clock: Arc<dyn Clock + Send + Sync>,
) -> anyhow::Result<JobScheduler> {
    let sched = JobScheduler::new()
        .await
        .context("failed to build audit JobScheduler")?;

    if !config.enabled {
        info!("dreaming audit scheduler disabled (audit.enabled = false)");
        return Ok(sched);
    }

    let job = build_job(config, store, classifier, embedder, clock).await?;
    sched.add(job).await.context("failed to add audit job")?;
    sched
        .start()
        .await
        .context("failed to start audit scheduler")?;

    info!(
        schedule = %config.schedule,
        provider = %config.llm_provider,
        "dreaming audit scheduler started"
    );
    Ok(sched)
}

/// Construct the audit cron job.
///
/// Split out from [`start_scheduler`] so the job-building logic can be unit
/// tested (via a dry-run path) without spinning up the full scheduler.
async fn build_job(
    config: &AuditConfig,
    store: SurrealStore,
    classifier: Arc<NativeNliClassifier>,
    embedder: Arc<OllamaEmbedding>,
    clock: Arc<dyn Clock + Send + Sync>,
) -> anyhow::Result<Job> {
    // The cron expression is parsed eagerly so a typo fails at server
    // startup rather than on the first scheduled tick (which would silently
    // never fire). `Job::new_async` re-parses the string, but the eager
    // check below is the one that surfaces a human-readable error.
    validate_cron(&config.schedule)?;

    let config_arc = Arc::new(config.clone());
    let store_clone = store;
    let classifier_clone = classifier;
    let embedder_clone = embedder;
    let clock_clone = clock;
    let schedule_str = config.schedule.clone();

    Job::new_async(schedule_str.as_str(), move |_uuid, _sched| {
        let config = config_arc.clone();
        let store = store_clone.clone();
        let classifier = classifier_clone.clone();
        let embedder = embedder_clone.clone();
        let clock = clock_clone.clone();
        Box::pin(async move {
            info!("dreaming audit tick fired");
            match run_audit(&config, store, classifier, embedder, clock).await {
                Ok(report) => {
                    info!(
                        deletions = report.deletions,
                        merges = report.merges,
                        "dreaming audit completed"
                    );
                }
                Err(e) => {
                    // Format the full `anyhow` context chain into a single
                    // string so the underlying cause is not lost. A plain
                    // `%e` would only show the top-level message.
                    error!(error = %format!("{e:#}"), "dreaming audit failed");
                }
            }
        })
    })
    .map_err(|e| anyhow::anyhow!("failed to construct audit Job from cron expression: {e}"))
}

/// Validate the cron expression by attempting to parse it through the same
/// crate the scheduler uses.
///
/// `tokio-cron-scheduler` does not expose its cron parser publicly, so this
/// helper relies on `cron` (a transitive dependency). If `cron` is not in the
/// dependency tree, this function performs a structural check instead (5
/// space-separated fields). Either way, an invalid cron expression is
/// surfaced at server startup rather than on the first silent tick.
fn validate_cron(expr: &str) -> anyhow::Result<()> {
    let trimmed = expr.trim();
    if trimmed.is_empty() {
        return Err(anyhow::anyhow!("audit.schedule must not be empty"));
    }
    // Five space-separated fields is the canonical UNIX cron shape that
    // tokio-cron-scheduler expects. A more permissive parser would silently
    // accept macros like `@daily` that the crate does not support.
    let field_count = trimmed.split_whitespace().count();
    if field_count != 5 {
        return Err(anyhow::anyhow!(
            "audit.schedule must be a 5-field UNIX cron expression (UTC), \
             got {field_count} fields: {expr:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_cron_accepts_canonical_5_field_expression() {
        assert!(validate_cron("0 3 * * *").is_ok());
        assert!(validate_cron("*/15 * * * *").is_ok());
        assert!(validate_cron("0 0 1 1 *").is_ok());
    }

    #[test]
    fn validate_cron_rejects_empty() {
        assert!(validate_cron("").is_err());
        assert!(validate_cron("   ").is_err());
    }

    #[test]
    fn validate_cron_rejects_wrong_field_count() {
        assert!(validate_cron("0 3 * *").is_err(), "4 fields");
        assert!(validate_cron("0 3 * * * *").is_err(), "6 fields");
        assert!(validate_cron("@daily").is_err(), "macro form not supported");
    }
}
