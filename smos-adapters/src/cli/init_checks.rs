//! Inline setup probes for `smos init` — `llama-server` health + reranker +
//! SurrealDB.
//!
//! Each probe is deliberately lightweight: it answers "is the box ready to
//! `smos serve`?" and prints a ✓ / ✗ row with a remediation hint. Detailed
//! diagnostics (NLI cache, config linting, full stats, Markdown report)
//! belong to `smos doctor` — `init` never delegates to the doctor module so
//! the setup wizard stays decoupled from the diagnostic surface.
//!
//! Lives in its own module so [`super::init_runner`] stays focused on
//! orchestration; the probes are pure IO + reporting.

use std::time::Duration;

use anyhow::{Context, Result};

use crate::SurrealStore;
use crate::cli::init_path::find_in_path;
use crate::config::{RerankerConfig, SurrealConfig};

const LLAMA_PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const RERANKER_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// The three `llama-server` ports the default config points the local roles
/// at: embedding (28081), extraction (28082), reranker (28181). `init` probes
/// each one so the operator hears about a missing service before the first
/// `smos serve` request fails with HTTP 503.
const LLAMA_PORTS: &[(u16, &str)] = &[
    (28081, "embedding"),
    (28082, "extraction"),
    (28181, "reranker"),
];

/// Probe `/health` on every port in [`LLAMA_PORTS`]. A miss on any port is
/// reported with a remediation hint that points at `auto_launch` (the
/// easiest fix) and the manual `llama-server` invocation. An already-running
/// server is reported as ✓ so the operator sees what was reused vs. what
/// `smos serve` will spawn via `[llama_cpp].auto_launch`.
pub(super) async fn check_llama_servers() {
    let client = match reqwest::Client::builder().build() {
        Ok(c) => c,
        Err(e) => {
            println!("  ✗ Cannot construct HTTP client: {e}");
            println!("    Verify rustls / native-tls setup and re-run");
            return;
        }
    };
    for (port, role) in LLAMA_PORTS {
        let url = format!("http://localhost:{port}/health");
        match probe_http(&client, &url, LLAMA_PROBE_TIMEOUT).await {
            Ok(()) => println!("  ✓ llama-server ({role}) reachable at http://localhost:{port}"),
            Err(_) => {
                println!("  ✗ llama-server ({role}) not reachable at http://localhost:{port}");
                println!("    Start it: llama-server --model <{role}.gguf> --port {port}");
                println!("    Or rely on [llama_cpp] auto_launch = true in config.toml");
            }
        }
    }
}

/// Check `llama-server` is discoverable on `PATH`. The auto-launch manager
/// (and any `[llama_cpp]` startup) depends on it, so a miss is reported as ✗
/// with the build pointer.
pub(super) fn check_llama_server() {
    match find_in_path("llama-server") {
        Some(path) => println!("  ✓ Found: {}", path.display()),
        None => {
            println!("  ✗ llama-server not found on PATH");
            println!("    Build it: https://github.com/ggerganov/llama.cpp");
            println!(
                "    Required for embedding / extraction / reranker — every chat-completion request fails without it"
            );
        }
    }
}

/// Probe the reranker `/health` endpoint. A miss is a soft warning (✗), not
/// fatal: the operator may legitimately start the reranker after init, or
/// point `[reranker]` at a remote host.
pub(super) async fn check_reranker(config: &RerankerConfig) {
    let client = match reqwest::Client::builder().build() {
        Ok(c) => c,
        Err(e) => {
            println!("  ✗ Cannot construct HTTP client: {e}");
            println!("    Verify rustls / native-tls setup and re-run");
            return;
        }
    };
    let url = format!("{}/health", config.url.trim_end_matches('/'));
    match probe_http(&client, &url, RERANKER_PROBE_TIMEOUT).await {
        Ok(()) => println!("  ✓ Reranker reachable at {}", config.url),
        Err(_) => {
            println!("  ✗ Reranker not reachable at {}", config.url);
            println!("    Start: llama-server --model <qwen3-reranker.gguf> --port 28181");
            println!("    Or enable [llama_cpp] auto_launch = true in config.toml");
        }
    }
}

/// Issue a bounded GET and succeed on any HTTP response — the goal is "is
/// something listening?", not "did it return 2xx?". A connection failure or
/// timeout becomes `Err`. The caller supplies a pooled `Client` so the
/// init wizard amortises connection setup across every probe instead of
/// paying for a fresh TLS handshake per port.
async fn probe_http(client: &reqwest::Client, url: &str, timeout: Duration) -> Result<()> {
    client
        .get(url)
        .timeout(timeout)
        .send()
        .await
        .with_context(|| format!("probe {url} failed"))?;
    Ok(())
}

/// Connect to SurrealDB and apply migrations. Reuses the production
/// bootstrap path ([`SurrealStore::connect`] + [`SurrealStore::run_migrations`])
/// so init validates exactly what `smos serve` will later use.
pub(super) async fn init_database(config: &SurrealConfig) {
    let store = match SurrealStore::connect(&config.path, &config.namespace, &config.database).await
    {
        Ok(s) => s,
        Err(e) => {
            println!("  ✗ Cannot connect to database: {e}");
            println!(
                "    Path: {}. Ensure the parent directory is writable.",
                config.path
            );
            return;
        }
    };
    match store.run_migrations().await {
        Ok(()) => println!("  ✓ Database ready — migrations applied ({})", config.path),
        Err(e) => {
            println!("  ✗ Database migrations failed: {e}");
            println!("    Delete the db directory and re-run: rm -rf ~/.smos/db");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EmbeddingConfig, LlmExtractionConfig, RerankerConfig};
    use crate::llama_server::LlamaCppConfig;

    /// Acquire the workspace-wide env-test lock — `LlamaCppConfig::default()`
    /// resolves model paths through `SmosPaths::resolve()`, which reads
    /// `SMOS_HOME`.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_env_lock::lock()
    }

    /// Extract the trailing port from a `http://host:port` base URL. Returns
    /// `None` when the URL does not end with a numeric port (config error —
    /// the defaults always do).
    fn port_of(url: &str) -> Option<u16> {
        url.rsplit(':').next()?.parse().ok()
    }

    /// The three probed ports must stay distinct — a collision would make
    /// one role's probe mask another's, so an operator would see a misleading
    /// ✓ on a service that is actually down.
    #[test]
    fn llama_ports_are_distinct() {
        let ports: std::collections::HashSet<u16> = LLAMA_PORTS.iter().map(|(p, _)| *p).collect();
        assert_eq!(ports.len(), LLAMA_PORTS.len(), "llama-server ports collide");
    }

    /// `LLAMA_PORTS` must stay in lock-step with the canonical config
    /// defaults: each role's probed port equals the port derived from the
    /// matching config-default URL (`LlmExtractionConfig::default().url`,
    /// `EmbeddingConfig::default().url`, `RerankerConfig::default().url`)
    /// AND the per-service `[llama_cpp.*]` port declared by
    /// `LlamaCppConfig::default()`. A drift here would make `smos init`
    /// probe a port that no configured URL points at — the operator would
    /// see a misleading ✓ on a service that is actually down.
    #[test]
    fn llama_ports_match_config_defaults() {
        let _g = lock();
        let by_role: std::collections::HashMap<&str, u16> =
            LLAMA_PORTS.iter().map(|(p, r)| (*r, *p)).collect();

        let extraction_url_port = port_of(&LlmExtractionConfig::default().url)
            .expect("LlmExtractionConfig::default().url ends with a port");
        let embedding_url_port = port_of(&EmbeddingConfig::default().url)
            .expect("EmbeddingConfig::default().url ends with a port");
        let reranker_url_port = port_of(&RerankerConfig::default().url)
            .expect("RerankerConfig::default().url ends with a port");

        let llama_cpp = LlamaCppConfig::default();

        assert_eq!(
            by_role["extraction"], extraction_url_port,
            "extraction: LLAMA_PORTS vs LlmExtractionConfig::default().url"
        );
        assert_eq!(
            by_role["extraction"], llama_cpp.extraction.port,
            "extraction: LLAMA_PORTS vs [llama_cpp.extraction].port"
        );

        assert_eq!(
            by_role["embedding"], embedding_url_port,
            "embedding: LLAMA_PORTS vs EmbeddingConfig::default().url"
        );
        assert_eq!(
            by_role["embedding"], llama_cpp.embedding.port,
            "embedding: LLAMA_PORTS vs [llama_cpp.embedding].port"
        );

        assert_eq!(
            by_role["reranker"], reranker_url_port,
            "reranker: LLAMA_PORTS vs RerankerConfig::default().url"
        );
        assert_eq!(
            by_role["reranker"], llama_cpp.reranker.port,
            "reranker: LLAMA_PORTS vs [llama_cpp.reranker].port"
        );
    }
}
