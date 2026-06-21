//! opencode discovery — locate a live session source.
//!
//! Two data sources are supported, tried in order:
//!
//! 1. **HTTP probe** — `GET /health` against a small set of common localhost
//!    ports, in PARALLEL. The first port whose body looks like an opencode
//!    status object wins. Parallel probing caps worst-case latency at one
//!    timeout (~2 s) instead of `ports.len() * timeout`.
//! 2. **CLI fallback** — when no port responds, the local `opencode` CLI is
//!    used. Reads the local SQLite DB directly; no GPU.
//!
//! Mirrors `smos-poc/scripts/opencode_source.py::probe_http` /
//! `resolve_source` / `fetch_session_list` / `fetch_session_export`.

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use smos_application::errors::ProviderError;

use crate::opencode::cli::run_opencode_cli;

/// Common opencode server ports, tried in parallel during the HTTP probe.
///
/// 11434 is Ollama — kept LAST as a sentinel because its JSON error body is
/// rejected by [`looks_alive`], so the probe never mistakes Ollama for
/// opencode.
pub const DEFAULT_PORTS: &[u16] = &[4096, 3000, 8080, 4097, 8888, 11434];

const HTTP_PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const HTTP_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Resolved way to reach opencode session data.
///
/// `Serialize` so the CLI binary can print the chosen source for operators.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum SessionSource {
    Http { port: u16 },
    Cli,
}

impl SessionSource {
    /// Stable label for log lines and CLI banners.
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Http { .. } => "http",
            Self::Cli => "cli",
        }
    }
}

/// Resolve a session source: explicit port → HTTP probe → CLI fallback.
///
/// Pass `Some(port)` to skip probing and force HTTP against that port.
pub async fn resolve_source(client: &Client, port: Option<u16>) -> SessionSource {
    if let Some(port) = port {
        return SessionSource::Http { port };
    }
    if let Some(http) = probe_http(client).await {
        return http;
    }
    tracing::info!("no opencode HTTP server found; falling back to CLI");
    SessionSource::Cli
}

/// Probe every port in [`DEFAULT_PORTS`] in parallel; return the first alive
/// source (lowest port that responds), or `None`.
///
/// Thin wrapper around [`probe_ports`] for the canonical port list. Kept as a
/// separate function so call sites read like the POC `probe_http()` and so
/// tests can exercise the parallel logic against a custom port list.
pub async fn probe_http(client: &Client) -> Option<SessionSource> {
    probe_ports(client, DEFAULT_PORTS).await
}

/// Probe every port in `ports` in parallel; return the first alive source
/// (lowest port that responds), or `None`.
///
/// Exposed so integration tests can drive the same parallel `join_all` +
/// first-alive-wins logic against a single wiremock port without binding the
/// canonical [`DEFAULT_PORTS`] (which risks flakiness when those ports are
/// already in use on the test host).
pub async fn probe_ports(client: &Client, ports: &[u16]) -> Option<SessionSource> {
    use futures::future::join_all;

    let probes = ports.iter().map(|&port| {
        let client = client.clone();
        async move { probe_one_port(&client, port).await }
    });
    let results = join_all(probes).await;
    results.into_iter().flatten().next()
}

/// Probe a single port: `GET /health`, then apply [`looks_alive`] to the body.
async fn probe_one_port(client: &Client, port: u16) -> Option<SessionSource> {
    let url = format!("http://localhost:{port}/health");
    let resp = match client.get(&url).timeout(HTTP_PROBE_TIMEOUT).send().await {
        Ok(r) => r,
        Err(_) => return None,
    };
    if resp.status() != reqwest::StatusCode::OK {
        return None;
    }
    let body = resp.text().await.ok()?;
    if looks_alive(&body) {
        tracing::info!("opencode HTTP server found on port {port}");
        Some(SessionSource::Http { port })
    } else {
        None
    }
}

/// Heuristic: a 200 body that starts with `{` and contains no `error` /
/// `not found` keyword. Rejects Ollama's `{"error": …}` JSON envelope (which
/// Ollama serves with 200 on some setups when the path is unknown).
///
/// # Known limitation (inherited from the POC)
///
/// The substring match is intentionally coarse — a real opencode payload like
/// `{"errors_count": 0}` or `{"error_rate": 0.0}` would be misclassified as
/// not-alive. Mirrors `opencode_source.py::_looks_alive` verbatim for parity
/// with the canonical Python implementation; tightening it is a POC-level
/// decision, not a Rust-implementation one.
fn looks_alive(body: &str) -> bool {
    let head = body.trim().to_lowercase();
    if !head.starts_with('{') {
        return false;
    }
    !head.contains("error") && !head.contains("not found")
}

/// Fetch the session list from `source`.
///
/// HTTP: `GET /session` → JSON array, or `{ "sessions": [...] }` envelope.
/// CLI: `opencode session list --format json`.
pub async fn list_sessions(
    source: &SessionSource,
    client: &Client,
) -> Result<Vec<Value>, DiscoveryError> {
    match source {
        SessionSource::Http { port } => {
            let url = format!("http://localhost:{port}/session");
            let data = http_get_json(client, &url).await?;
            Ok(extract_session_array(data))
        }
        SessionSource::Cli => {
            let raw = run_opencode_cli(&["session", "list", "--format", "json"])
                .await
                .map_err(discovery_from_provider)?;
            let data: Value = parse_json(&raw)?;
            Ok(match data {
                Value::Array(arr) => arr,
                _ => Vec::new(),
            })
        }
    }
}

/// Fetch the full transcript of one session as a normalized value.
///
/// HTTP path assembles `{"info": …, "messages": […]}` from
/// `GET /session/{id}` followed by `GET /session/{id}/message`. CLI path uses
/// `opencode export <id>` which already emits this shape.
pub async fn fetch_session_export(
    source: &SessionSource,
    client: &Client,
    session_id: &str,
) -> Result<Value, DiscoveryError> {
    match source {
        SessionSource::Http { port } => {
            let base = format!("http://localhost:{port}/session/{session_id}");
            let info = http_get_json(client, &base).await?;
            let messages = http_get_json(client, &format!("{base}/message")).await?;
            Ok(serde_json::json!({ "info": info, "messages": messages }))
        }
        SessionSource::Cli => {
            let raw = run_opencode_cli(&["export", session_id])
                .await
                .map_err(discovery_from_provider)?;
            let data: Value = parse_json(&raw)?;
            Ok(match data {
                Value::Object(_) => data,
                _ => serde_json::json!({ "info": {}, "messages": [] }),
            })
        }
    }
}

/// GET a URL and parse the JSON body, mapping transport/parse failures to
/// [`DiscoveryError::Http`].
async fn http_get_json(client: &Client, url: &str) -> Result<Value, DiscoveryError> {
    let resp = client
        .get(url)
        .timeout(HTTP_READ_TIMEOUT)
        .send()
        .await
        .map_err(|e| DiscoveryError::Http(url.to_string(), e.to_string()))?;
    let resp = resp
        .error_for_status()
        .map_err(|e| DiscoveryError::Http(url.to_string(), e.to_string()))?;
    resp.json()
        .await
        .map_err(|e| DiscoveryError::Http(url.to_string(), e.to_string()))
}

/// Strip the `sessions` envelope if the body is `{ "sessions": [...] }`;
/// pass arrays through; everything else becomes an empty list.
fn extract_session_array(data: Value) -> Vec<Value> {
    match data {
        Value::Array(arr) => arr,
        Value::Object(mut obj) => match obj.remove("sessions") {
            Some(Value::Array(arr)) => arr,
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// Parse a JSON string into a [`Value`], mapping parse failures to
/// [`DiscoveryError::Json`].
fn parse_json(raw: &str) -> Result<Value, DiscoveryError> {
    if raw.trim().is_empty() {
        return Ok(Value::Array(Vec::new()));
    }
    serde_json::from_str(raw).map_err(DiscoveryError::from)
}

/// Map a [`ProviderError`] returned by the CLI wrapper into a discovery error.
fn discovery_from_provider(e: ProviderError) -> DiscoveryError {
    match e {
        ProviderError::Unavailable(msg) => DiscoveryError::CliNotFound(msg),
        ProviderError::Timeout(d) => DiscoveryError::CliTimeout(d),
        other => DiscoveryError::CliFailed(other.to_string()),
    }
}

/// Discovery-layer error: HTTP transport, CLI invocation, JSON parse.
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    #[error("HTTP request failed for {0}: {1}")]
    Http(String, String),
    #[error("opencode CLI not found: {0}")]
    CliNotFound(String),
    #[error("opencode CLI timed out after {0:?}")]
    CliTimeout(Duration),
    #[error("opencode CLI failed: {0}")]
    CliFailed(String),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_alive_accepts_status_object() {
        assert!(looks_alive("{\"status\":\"ok\"}"));
        assert!(looks_alive("  {\"ok\":true}  "));
    }

    #[test]
    fn looks_alive_rejects_ollama_error_envelope() {
        assert!(!looks_alive("{\"error\":\"not found\"}"));
        assert!(!looks_alive("{\"error\":\"model missing\"}"));
    }

    #[test]
    fn looks_alive_rejects_non_json_body() {
        assert!(!looks_alive("<html>not found</html>"));
        assert!(!looks_alive(""));
        assert!(!looks_alive("ok"));
    }

    #[test]
    fn looks_alive_rejects_html_with_404_marker() {
        assert!(!looks_alive("<!DOCTYPE html><html>not found</html>"));
    }

    #[test]
    fn extract_session_array_from_top_level_array() {
        let data = serde_json::json!([{"id": "a"}, {"id": "b"}]);
        let arr = extract_session_array(data);
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn extract_session_array_from_sessions_envelope() {
        let data = serde_json::json!({"sessions": [{"id": "a"}]});
        let arr = extract_session_array(data);
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn extract_session_array_from_object_without_envelope_is_empty() {
        let data = serde_json::json!({"foo": 1});
        assert!(extract_session_array(data).is_empty());
    }

    #[test]
    fn extract_session_array_from_scalar_is_empty() {
        assert!(extract_session_array(serde_json::json!(42)).is_empty());
        assert!(extract_session_array(Value::Null).is_empty());
    }

    #[test]
    fn parse_json_empty_string_returns_empty_array() {
        let v = parse_json("").unwrap();
        assert_eq!(v, Value::Array(Vec::new()));
        let v = parse_json("   \n").unwrap();
        assert_eq!(v, Value::Array(Vec::new()));
    }

    #[test]
    fn parse_json_valid_object_round_trips() {
        let v = parse_json("{\"a\":1}").unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn parse_json_invalid_string_errors() {
        assert!(parse_json("{not json}").is_err());
    }

    #[test]
    fn source_kind_str_matches_variant() {
        assert_eq!(SessionSource::Cli.kind_str(), "cli");
        assert_eq!(SessionSource::Http { port: 4096 }.kind_str(), "http");
    }
}
