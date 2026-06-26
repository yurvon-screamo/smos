//! Tracing subscriber installation shared by every CLI subcommand.
//!
//! Two entry points cover the two log formats the SMOS binaries need:
//! - [`init_tracing_default`] — plain human-readable output (used by
//!   `smos import` and `smos doctor`, where there is no server config to
//!   pick a format from).
//! - [`init_tracing_for_server`] — picks JSON vs. pretty from
//!   `ServerConfig::log_format` so the proxy's structured logs match the
//!   operator's deployment choice.
//! - [`init_tracing_for_service`] — Windows-only rolling-file appender; a
//!   service runs with no attached console, so `fmt()` (which targets
//!   stdout / stderr) would silently drop every log line.

use anyhow::{Context, Result};

use crate::config::ServerConfig;

/// Default level filter used when `RUST_LOG` is not set. `smos=debug` keeps
/// SMOS-owned spans verbose while silence everything else to `info`.
const DEFAULT_FILTER: &str = "info,smos=debug";

/// Install the tracing subscriber with the default (human-readable) format.
/// `RUST_LOG` overrides `DEFAULT_FILTER`.
pub fn init_tracing_default() {
    use tracing_subscriber::EnvFilter;

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

/// Install the tracing subscriber picking JSON vs. pretty from
/// `server_config.log_format`. `RUST_LOG` overrides `DEFAULT_FILTER`.
///
/// `log_format = "json"` emits structured JSON logs (production / log
/// shipping); any other value emits human-readable colourised output for
/// local development.
pub fn init_tracing_for_server(server_config: &ServerConfig) {
    use tracing_subscriber::EnvFilter;

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));

    match server_config.log_format.as_str() {
        "pretty" => {
            tracing_subscriber::fmt().with_env_filter(filter).init();
        }
        _ => {
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .json()
                .init();
        }
    }
}

/// Default daily log file name under `<smos_home>/logs/` for the service
/// entry point. Kept as a constant so the install-time hint and the
/// appender agree on the operator-facing path.
#[cfg(windows)]
const SERVICE_LOG_BASENAME: &str = "smos-service.log";

/// Install a file tracing subscriber for the Windows service process. A
/// service has no console, so the regular `fmt()` subscriber (stdout /
/// stderr) would lose every log line; this wires a daily rolling file
/// under `<smos_home>/logs/` instead.
///
/// Uses a SYNCHRONOUS appender (not `tracing_appender::non_blocking`):
/// the service has no throughput pressure, and a non-blocking worker
/// would drop the last buffered lines on process exit — including the
/// terminal `error!` that explains why the service failed to start.
/// Synchronous writes guarantee every log line reaches disk before the
/// process is allowed to proceed.
///
/// Never fatal: if `<smos_home>/logs/` cannot be created or the
/// subscriber is already installed, the error is propagated but the
/// caller treats it as non-fatal so the service can still report
/// `RUNNING` to SCM.
#[cfg(windows)]
pub fn init_tracing_for_service() -> Result<()> {
    use tracing_appender::rolling;
    use tracing_subscriber::EnvFilter;

    let log_dir = crate::paths::smos_home().join("logs");
    std::fs::create_dir_all(&log_dir)
        .with_context(|| format!("failed to create service log dir {}", log_dir.display()))?;

    let file_appender = rolling::daily(&log_dir, SERVICE_LOG_BASENAME);

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(file_appender)
        .with_ansi(false)
        .json()
        .try_init()
        .map_err(|e| anyhow::anyhow!("tracing subscriber already installed: {e}"))?;
    tracing::info!(log_dir = %log_dir.display(), "smos service tracing initialised");
    Ok(())
}
