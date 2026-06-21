//! Tracing subscriber installation shared by every CLI subcommand.
//!
//! Two entry points cover the two log formats the SMOS binaries need:
//! - [`init_tracing_default`] — plain human-readable output (used by
//!   `smos import` and `smos doctor`, where there is no server config to
//!   pick a format from).
//! - [`init_tracing_for_server`] — picks JSON vs. pretty from
//!   `ServerConfig::log_format` so the proxy's structured logs match the
//!   operator's deployment choice.

use crate::config::ServerConfig;

/// Default level filter used when `RUST_LOG` is not set. `smos=debug` keeps
/// SMOS-owned spans verbose while silence everything else to `info`.
const DEFAULT_FILTER: &str = "info,smos=debug";

/// Install the tracing subscriber with the default (human-readable) format.
/// `RUST_LOG` overrides [`DEFAULT_FILTER`].
pub fn init_tracing_default() {
    use tracing_subscriber::EnvFilter;

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

/// Install the tracing subscriber picking JSON vs. pretty from
/// `server_config.log_format`. `RUST_LOG` overrides [`DEFAULT_FILTER`].
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
