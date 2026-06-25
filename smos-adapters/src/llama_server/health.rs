//! Health-check helpers for `llama-server` processes.
//!
//! `llama-server` exposes an HTTP health endpoint at `/health`. We probe it
//! both before launching a service (to detect an already-running instance)
//! and after spawning (to wait until the model has finished loading).

use std::time::Duration;

use anyhow::Result;
use reqwest::Client;

/// Maximum number of health probes before bailing.
const MAX_ATTEMPTS: u32 = 30;

/// Sleep between health probes.
const PROBE_INTERVAL: Duration = Duration::from_secs(1);

/// Per-probe HTTP timeout. Kept short so the probe loop is responsive even
/// when the server is still loading the model and ignoring requests.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Build a fresh pooled HTTP client for the probe loop. Constructed once per
/// manager so we are not paying TLS handshake cost on every probe.
pub fn probe_client() -> Result<Client> {
    Ok(Client::builder().timeout(PROBE_TIMEOUT).build()?)
}

/// Canonical health URL for the service listening on `port`.
pub fn service_health_url(port: u16) -> String {
    format!("http://localhost:{port}/health")
}

/// `true` when a process is already responding on `port`. Any HTTP response
/// (even a non-2xx) counts as "responding" — we only need to know that a
/// server is alive enough to answer at all.
pub async fn is_port_responding(client: &Client, port: u16) -> bool {
    client.get(service_health_url(port)).send().await.is_ok()
}

/// Poll the service's `/health` endpoint until it answers (or until
/// [`MAX_ATTEMPTS`] probes have failed). Returns `Ok(())` on the first
/// successful response and an error otherwise.
pub async fn wait_for_health(client: &Client, name: &str, port: u16) -> Result<()> {
    for attempt in 1..=MAX_ATTEMPTS {
        if is_port_responding(client, port).await {
            tracing::info!(service = name, port, attempts = attempt, "ready");
            return Ok(());
        }
        tokio::time::sleep(PROBE_INTERVAL).await;
    }
    let per_attempt = PROBE_INTERVAL.as_secs() + PROBE_TIMEOUT.as_secs();
    let total = u64::from(MAX_ATTEMPTS) * per_attempt;
    anyhow::bail!(
        "{name} on port {port} did not become healthy within ~{total}s \
         ({MAX_ATTEMPTS} attempts × {per_attempt}s each = probe interval + probe timeout)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_url_uses_localhost() {
        assert_eq!(service_health_url(28181), "http://localhost:28181/health");
    }

    #[tokio::test]
    async fn is_port_responding_returns_false_for_dead_port() {
        let client = probe_client().expect("client");
        // Bind a throwaway TCP listener and HOLD it (do not drop) across
        // the probe. A listener that never speaks HTTP still accepts the
        // probe's connection but never returns an HTTP response, so the
        // probe times out and reports "not responding" - the same answer
        // a truly dead port gives. Holding the port removes the TOCTOU
        // window: no other process can bind an HTTP server on it between
        // bind and probe, so the test is deterministic and runs by
        // default. `listener` is dropped at end of scope (after the probe).
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("bind");
        let port = listener.local_addr().expect("addr").port();
        assert!(!is_port_responding(&client, port).await);
    }
}
