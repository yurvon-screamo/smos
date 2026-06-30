//! CLI forwarding resolver — decides whether a forwardable subcommand
//! (`search`, `finalize`, `import raw`, `import opencode`) should execute
//! locally (opening RocksDB in-process) or forward over HTTP to a running
//! `smos serve` instance.
//!
//! # Why forwarding exists
//!
//! [`crate::SurrealStore::connect`] opens embedded RocksDB, which takes a
//! process-exclusive OS lock on the `LOCK` file inside the data directory.
//! When `smos serve` runs, it holds that lock for its whole lifetime —
//! SurrealDB does not expose a read-only open and does not release the lock
//! on demand. Any CLI subcommand that opens the same store then fails with
//! `IO error: while lock file: LOCK: held`.
//!
//! Forwarding routes the request through the service's HTTP API (under
//! `/v1/cli/*`), so the CLI never opens RocksDB and the running service
//! executes the SAME use case the local branch would have. The hexagonal
//! invariant is preserved: server handler and local branch both invoke the
//! identical use-case struct.
//!
//! # Resolution rules (outer CLI layer)
//!
//! 1. `--local` flag wins unconditionally.
//! 2. Non-loopback `server.host` wins unconditionally (fail-secure default
//!    against talking to a remote / foreign SMOS instance).
//! 3. `[cli].forward_mode = "local"` config override wins.
//! 4. Otherwise: probe `GET /health` on the configured `host:port`. A 200
//!    response enables forwarding; anything else (connection refused,
//!    timeout, non-200) falls back to local.
//!
//! The pure half of the decision ([`should_consider_forward`]) is
//! unit-testable without IO; the async probe ([`probe_server`]) is exercised
//! via wiremock in the integration suite.

use std::time::Duration;

use crate::config::SmosConfig;
use crate::http::axum_server::is_loopback_host;

/// Resolved execution mode for one forwardable subcommand invocation.
#[derive(Debug)]
pub enum ExecMode {
    /// Open RocksDB in-process and invoke the use case directly.
    Local,
    /// Forward over HTTP to a running `smos serve` instance. The CLI does
    /// NOT open the store and does NOT invoke the use case — it transports
    /// the request and streams the response body verbatim to stdout.
    Remote {
        client: reqwest::Client,
        base_url: String,
    },
}

/// Operator-controlled knobs that feed [`ExecMode::resolve`].
#[derive(Debug, Clone, Copy)]
pub struct ExecModeOptions {
    /// `true` when `--local` was passed. Forces [`ExecMode::Local`]
    /// regardless of config or probe outcome.
    pub force_local: bool,
    /// Health-probe deadline. Tight default so a down server does not stall
    /// the CLI by the full TCP timeout.
    pub probe_timeout: Duration,
}

impl Default for ExecModeOptions {
    fn default() -> Self {
        Self {
            force_local: false,
            probe_timeout: Duration::from_millis(250),
        }
    }
}

impl ExecModeOptions {
    /// Build options from the global `--local` flag and `[cli]` config.
    /// The probe timeout is read from `[cli].forward_probe_timeout_ms`
    /// (validated to be `> 0` by [`SmosConfig::validate`]).
    pub fn from_config(force_local: bool, config: &SmosConfig) -> Self {
        Self {
            force_local,
            probe_timeout: Duration::from_millis(config.cli.forward_probe_timeout_ms),
        }
    }
}

/// Resolve the execution mode for one invocation.
///
/// Probes the loopback `/health` endpoint when the static gates pass. Probe
/// failures (connection refused, timeout, non-200) silently fall back to
/// [`ExecMode::Local`]; the operator-visible signal is the per-command
/// stderr notice emitted by [`announce_forward`] / the local-branch tracing.
pub async fn resolve(config: &SmosConfig, opts: &ExecModeOptions) -> ExecMode {
    let Some((host, port)) = should_consider_forward(config, opts) else {
        return ExecMode::Local;
    };
    let client = match reqwest::Client::builder()
        .timeout(opts.probe_timeout)
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!(
                error = %e,
                "failed to build probe client; falling back to local execution"
            );
            return ExecMode::Local;
        }
    };
    match probe_server(&client, &host, port).await {
        Some(base_url) => ExecMode::Remote { client, base_url },
        None => ExecMode::Local,
    }
}

/// Pure half of the decision — gates that do not require IO. Returns the
/// `(host, port)` to probe when forwarding should be CONSIDERED (the probe
/// still has to succeed), or `None` when local execution is mandated.
///
/// Exposed (not folded into [`resolve`]) so the decision tree is unit-tested
/// without spinning up a wiremock server.
pub fn should_consider_forward(
    config: &SmosConfig,
    opts: &ExecModeOptions,
) -> Option<(String, u16)> {
    if opts.force_local {
        return None;
    }
    if config.cli.forward_mode == "local" {
        return None;
    }
    let host = config.server.host.as_str();
    if !is_loopback_host(host) {
        tracing::debug!(
            host = host,
            "server.host is non-loopback; CLI forwarding disabled (use --local). \
             Refusing to forward to a possibly-foreign SMOS instance."
        );
        return None;
    }
    Some((host.to_string(), config.server.port))
}

/// Probe `GET /health` on `host:port`. Returns the base URL when the server
/// is reachable and reports ok, `None` on any failure (connection refused,
/// timeout, non-200). A 200 alone is enough — endpoint-existence is verified
/// lazily on the actual POST (a 404 on `/v1/cli/<cmd>` triggers a documented
/// fallback in the runner).
async fn probe_server(client: &reqwest::Client, host: &str, port: u16) -> Option<String> {
    let url = format!("http://{host}:{port}/health");
    let response = client.get(&url).send().await.ok()?;
    if response.status() != reqwest::StatusCode::OK {
        tracing::debug!(
            status = %response.status(),
            "health probe returned non-200; falling back to local execution"
        );
        return None;
    }
    Some(format!("http://{host}:{port}"))
}

/// Emit the one-line stderr notice that a command is being forwarded. The
/// operator sees the target URL and the override flag in the same line, so
/// debugging "why is my CLI talking to the network?" is one grep away.
pub fn announce_forward(command: &str, base_url: &str) {
    eprintln!("smos: forwarding {command} to {base_url} (use --local to force local execution)");
}

/// Detect a RocksDB lock-contention error in a chained `anyhow::Error`.
///
/// Used by every forwardable runner's local branch to emit the actionable
/// TOCTOU recovery message when the probe said "server-down" but the
/// service started between probe and `SurrealStore::connect` — the local
/// connect then fails with the canonical lock error.
///
/// NARROW match — only lock-contention tokens:
/// `"lock file"`, `"lock: held"`, `"LOCK: held"` (case-insensitive on the
/// whole error chain). Does NOT match generic RocksDB errors (e.g. column
/// family corruption) — those are NOT lock contention and should surface
/// verbatim.
///
/// The string-snip approach is the only option — SurrealDB 2.x wraps the
/// RocksDB error into a generic `RepoError::ConnectFailed(String)`
/// without preserving the underlying typed cause.
pub fn is_lock_error(error: &anyhow::Error) -> bool {
    let chain = format!("{error:#}").to_lowercase();
    chain.contains("lock file") || chain.contains("lock: held")
}

/// Emit the standard TOCTOU lock-recovery message to stderr. Used by all
/// four forwardable runners (search, finalize, raw-import, opencode) when
/// the local branch hits a RocksDB lock after the probe said "server-down"
/// — the service started between probe and `SurrealStore::connect`.
pub fn emit_lock_recovery_message() {
    emit_lock_held_message(
        "the service may have started after the probe; retry the command — \
         the next probe will detect the running service, or pass --local to \
         force a local attempt",
    );
}

/// Emit a lock-held message for always-local commands (e.g.
/// `smos import-dir`) that do not forward. The operator is told another
/// SMOS process holds the lock, not that a probe race occurred.
pub fn emit_lock_held_message_no_forwarding() {
    emit_lock_held_message(
        "another smos process (e.g. `smos serve`) holds the lock; stop it \
         or use a forwardable command (e.g. `smos import raw`)",
    );
}

/// Core lock-held message emitter. `detail` is the actionable hint
/// appropriate for the calling context (forwarding vs always-local).
fn emit_lock_held_message(detail: &str) {
    eprintln!("smos: local execution failed because the database lock is held ({detail}).");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SmosConfig;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn loopback_config() -> SmosConfig {
        let mut cfg = SmosConfig::default();
        cfg.server.host = "127.0.0.1".into();
        cfg.server.port = 0;
        cfg
    }

    fn non_loopback_config() -> SmosConfig {
        let mut cfg = SmosConfig::default();
        cfg.server.host = "0.0.0.0".into();
        cfg
    }

    // ---- should_consider_forward: pure decision tree --------------------

    #[test]
    fn should_consider_forward_returns_none_for_force_local() {
        let cfg = loopback_config();
        let opts = ExecModeOptions {
            force_local: true,
            probe_timeout: Duration::from_millis(50),
        };
        assert!(should_consider_forward(&cfg, &opts).is_none());
    }

    #[test]
    fn should_consider_forward_returns_none_for_non_loopback_host() {
        let cfg = non_loopback_config();
        let opts = ExecModeOptions::default();
        assert!(should_consider_forward(&cfg, &opts).is_none());
    }

    #[test]
    fn should_consider_forward_returns_none_for_local_forward_mode() {
        let mut cfg = loopback_config();
        cfg.cli.forward_mode = "local".into();
        let opts = ExecModeOptions::default();
        assert!(should_consider_forward(&cfg, &opts).is_none());
    }

    #[test]
    fn should_consider_forward_returns_host_port_for_loopback_auto() {
        let mut cfg = loopback_config();
        cfg.server.port = 9999;
        let opts = ExecModeOptions::default();
        let got = should_consider_forward(&cfg, &opts).expect("loopback + auto => probe");
        assert_eq!(got.0, "127.0.0.1");
        assert_eq!(got.1, 9999);
    }

    // ---- probe_server: IO behaviour via wiremock -------------------------

    async fn probe(client: &reqwest::Client, server: &MockServer) -> Option<String> {
        let url = server.uri();
        let health_url = format!("{url}/health");
        let resp = client.get(&health_url).send().await.ok()?;
        if resp.status() != reqwest::StatusCode::OK {
            return None;
        }
        Some(url)
    }

    #[tokio::test]
    async fn probe_returns_base_url_on_200() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"status":"ok"})),
            )
            .mount(&server)
            .await;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(200))
            .build()
            .unwrap();
        let got = probe(&client, &server).await.expect("200 => Some");
        assert_eq!(got, server.uri());
    }

    #[tokio::test]
    async fn probe_returns_none_on_non_200() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(200))
            .build()
            .unwrap();
        assert!(probe(&client, &server).await.is_none());
    }

    #[tokio::test]
    async fn probe_returns_none_on_connection_refused() {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(200))
            .build()
            .unwrap();
        // Ephemeral port that is overwhelmingly likely to be closed.
        let result = client.get("http://127.0.0.1:1/health").send().await;
        assert!(result.is_err(), "connection refused must surface as Err");
    }

    // ---- resolve: end-to-end decision wiring ----------------------------

    #[tokio::test]
    async fn resolve_force_local_short_circuits_without_probe() {
        let mut cfg = loopback_config();
        cfg.server.port = 1;
        let opts = ExecModeOptions {
            force_local: true,
            probe_timeout: Duration::from_millis(50),
        };
        let mode = resolve(&cfg, &opts).await;
        assert!(
            matches!(mode, ExecMode::Local),
            "--local wins without probe"
        );
    }

    #[tokio::test]
    async fn resolve_non_loopback_short_circuits_without_probe() {
        let mut cfg = non_loopback_config();
        cfg.server.port = 1;
        let opts = ExecModeOptions::default();
        let mode = resolve(&cfg, &opts).await;
        assert!(matches!(mode, ExecMode::Local), "non-loopback => Local");
    }

    #[tokio::test]
    async fn resolve_loopback_auto_probes_health_and_forwards_on_200() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;
        let mut cfg = loopback_config();
        let uri = server.uri();
        let (_scheme, rest) = uri.split_once("://").unwrap();
        let (host, port_str) = rest.rsplit_once(':').unwrap();
        cfg.server.host = host.to_string();
        cfg.server.port = port_str.parse().unwrap();
        let opts = ExecModeOptions {
            force_local: false,
            probe_timeout: Duration::from_secs(1),
        };
        let mode = resolve(&cfg, &opts).await;
        match mode {
            ExecMode::Remote { base_url, .. } => {
                assert_eq!(base_url, server.uri());
            }
            ExecMode::Local => panic!("probe should have succeeded"),
        }
    }

    #[tokio::test]
    async fn resolve_loopback_auto_falls_back_when_probe_fails() {
        let mut cfg = loopback_config();
        cfg.server.port = 1;
        let opts = ExecModeOptions {
            force_local: false,
            probe_timeout: Duration::from_millis(50),
        };
        let mode = resolve(&cfg, &opts).await;
        assert!(matches!(mode, ExecMode::Local), "probe fail => Local");
    }

    // ---- is_lock_error: narrow lock-contention detection -----------------

    #[test]
    fn is_lock_error_recognises_canonical_rocksdb_message() {
        let err = anyhow::anyhow!("IO error: while lock file: /tmp/db/LOCK: held");
        assert!(is_lock_error(&err));
    }

    #[test]
    fn is_lock_error_recognises_held_variant() {
        let err = anyhow::anyhow!("IO error: while lock file: LOCK: held");
        assert!(is_lock_error(&err));
    }

    #[test]
    fn is_lock_error_does_not_match_generic_rocksdb_errors() {
        // Pre-cleanup this matched via the `"rocksdb"` substring — too
        // broad. A column-family corruption is NOT lock contention.
        let err = anyhow::anyhow!("RocksDB: unable to open column family");
        assert!(
            !is_lock_error(&err),
            "generic RocksDB errors must not match"
        );
    }

    #[test]
    fn is_lock_error_does_not_match_unrelated_errors() {
        let err = anyhow::anyhow!("connection refused");
        assert!(!is_lock_error(&err));
    }

    #[test]
    fn is_lock_error_is_case_insensitive() {
        let err = anyhow::anyhow!("io error: While LOCK File: foo");
        assert!(is_lock_error(&err));
    }
}
