//! Inline setup probes for `smos init` — Ollama + llama-server + reranker +
//! SurrealDB.
//!
//! Each probe is deliberately lightweight: it answers "is the box ready to
//! `smos serve`?" and prints a ✓ / ✗ row with a remediation hint. Detailed
//! diagnostics (per-model validation, NLI cache, config linting, full stats,
//! Markdown report) belong to `smos doctor` — `init` never delegates to the
//! doctor module so the setup wizard stays decoupled from the diagnostic
//! surface.
//!
//! Lives in its own module so [`super::init_runner`] stays focused on
//! orchestration; the probes are pure IO + reporting.

use std::time::Duration;

use anyhow::{Context, Result};

use crate::SurrealStore;
use crate::cli::init_path::find_in_path;
use crate::config::{RerankerConfig, SurrealConfig};

const OLLAMA_PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const RERANKER_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Models the default config expects to find in the local Ollama registry.
///
/// `smos init` pulls these so a fresh box is ready to `smos serve` without a
/// second command. The list mirrors the three Ollama-local roles in
/// [`crate::cli::init_defaults::DEFAULT_CONFIG_TOML`] (chat person "bob",
/// extraction, embedding). Cloud-routed persons (OpenRouter / OpenAI / …) are
/// intentionally NOT pulled here — they live outside Ollama, and `smos doctor`
/// validates them. The subset relation is pinned by
/// [`tests::required_ollama_models_stay_in_sync_with_default_config`].
const REQUIRED_OLLAMA_MODELS: &[&str] = &[
    "granite4.1:3b",
    "qwen3.5:2b",
    "hf.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:latest",
];

/// Probe Ollama's `/api/tags`, then ensure every required model is present —
/// pulling the missing ones via the `ollama` CLI. An unreachable Ollama is
/// reported with the install hint and the run continues (the operator may
/// run init again after starting `ollama serve`).
pub(super) async fn check_ollama_and_pull_models(base_url: &str) {
    match fetch_ollama_tags(base_url).await {
        Ok(available) => {
            println!(
                "  ✓ Ollama reachable ({} model{} available)",
                available.len(),
                if available.len() == 1 { "" } else { "s" }
            );
            for model in REQUIRED_OLLAMA_MODELS {
                if available.iter().any(|a| a == *model) {
                    println!("  ✓ Model {model} — already pulled");
                } else {
                    pull_ollama_model(model).await;
                }
            }
        }
        Err(e) => {
            println!("  ✗ Ollama not reachable at {base_url}");
            println!("    {e}");
            println!("    Install from https://ollama.com then run: ollama serve");
        }
    }
}

/// Ollama `/api/tags` response shape — only the fields init reads.
#[derive(Debug, serde::Deserialize)]
struct TagsResponse {
    models: Vec<TagsModel>,
}

#[derive(Debug, serde::Deserialize)]
struct TagsModel {
    name: String,
}

async fn fetch_ollama_tags(base_url: &str) -> Result<Vec<String>> {
    let client = reqwest::Client::builder()
        .build()
        .context("HTTP client construction failed")?;
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    let response = client
        .get(&url)
        .timeout(OLLAMA_PROBE_TIMEOUT)
        .send()
        .await
        .with_context(|| format!("cannot connect to {base_url}"))?;
    let tags: TagsResponse = response
        .json()
        .await
        .context("Ollama /api/tags returned invalid JSON")?;
    Ok(tags.models.into_iter().map(|m| m.name).collect())
}

/// Pull one model via the `ollama` CLI. The CLI streams progress to the
/// inherited stdout/stderr, so the operator sees a live download bar. Any
/// non-zero exit or a missing `ollama` binary is reported with a retry hint
/// rather than aborting the whole init.
async fn pull_ollama_model(model: &str) {
    println!("  ⟳ Pulling {model} (this can take a while on first pull)...");
    let status = tokio::process::Command::new("ollama")
        .args(["pull", model])
        .status()
        .await;
    match status {
        Ok(s) if s.success() => println!("  ✓ Model {model} — pulled"),
        Ok(s) => {
            println!("  ✗ Model {model} — pull failed (exit {:?})", s.code());
            println!("    Retry: ollama pull {model}");
        }
        Err(e) => {
            println!("  ✗ Model {model} — cannot run `ollama` CLI: {e}");
            println!("    Make sure the `ollama` binary is on PATH");
        }
    }
}

/// Check `llama-server` is discoverable on `PATH`. The reranker (and any
/// `[llama_cpp]` auto-launch) depend on it, so a miss is reported as ✗ with
/// the build pointer.
pub(super) fn check_llama_server() {
    match find_in_path("llama-server") {
        Some(path) => println!("  ✓ Found: {}", path.display()),
        None => {
            println!("  ✗ llama-server not found on PATH");
            println!("    Build it: https://github.com/ggerganov/llama.cpp");
            println!("    Required for the reranker — enrichment fails without it");
        }
    }
}

/// Probe the reranker `/health` endpoint. A miss is a soft warning (✗), not
/// fatal: the operator may legitimately start the reranker after init, or
/// point `[reranker]` at a remote host.
pub(super) async fn check_reranker(config: &RerankerConfig) {
    let url = format!("{}/health", config.url.trim_end_matches('/'));
    match probe_http(&url, RERANKER_PROBE_TIMEOUT).await {
        Ok(()) => println!("  ✓ Reranker reachable at {}", config.url),
        Err(_) => {
            println!("  ✗ Reranker not reachable at {}", config.url);
            println!("    Start: llama-server --model <qwen3-reranker.gguf> --port 8181");
            println!("    Or enable [llama_cpp] auto_launch = true in config.toml");
        }
    }
}

/// Issue a bounded GET and succeed on any HTTP response — the goal is "is
/// something listening?", not "did it return 2xx?". A connection failure or
/// timeout becomes `Err`.
async fn probe_http(url: &str, timeout: Duration) -> Result<()> {
    let client = reqwest::Client::builder()
        .build()
        .context("HTTP client construction failed")?;
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
    use crate::cli::init_defaults::DEFAULT_CONFIG_TOML;

    /// Acquire the workspace-wide env-test lock — this test parses the
    /// default config, whose `SurrealConfig::default()` resolves paths via
    /// `SMOS_HOME`.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_env_lock::lock()
    }

    /// `REQUIRED_OLLAMA_MODELS` must stay a subset of the Ollama-local model
    /// ids declared in [`DEFAULT_CONFIG_TOML`] (chat persons + extraction +
    /// embedding). Pinning the subset catches drift the moment either side
    /// changes — otherwise `smos init` would silently pull a model the
    /// default config no longer references (or miss one it newly does).
    #[test]
    fn required_ollama_models_stay_in_sync_with_default_config() {
        let _g = lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let prior = std::env::var("SMOS_HOME").ok();
        // SAFETY: same lock-protected guarantee.
        unsafe {
            std::env::set_var("SMOS_HOME", tmp.path());
        }
        let cfg = crate::config::SmosConfig::load_from_str(DEFAULT_CONFIG_TOML)
            .expect("default toml must parse");
        unsafe {
            match prior {
                Some(v) => std::env::set_var("SMOS_HOME", v),
                None => std::env::remove_var("SMOS_HOME"),
            }
        }

        let mut declared: Vec<String> = cfg.persons.values().map(|p| p.model.clone()).collect();
        declared.push(cfg.llm_extraction.model.clone());
        declared.push(cfg.embedding.model.clone());

        for required in REQUIRED_OLLAMA_MODELS {
            assert!(
                declared.iter().any(|d| d == *required),
                "REQUIRED_OLLAMA_MODELS lists {required:?} which the default config does not \
                 declare as a person / extraction / embedding model — the two have drifted.",
            );
        }
    }
}
