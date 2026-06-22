//! [`LlamaCppManager`] — spawn / probe / kill lifecycle for `llama-server`.
//!
//! The manager owns one [`Child`] per launched service in a
//! `tokio::sync::Mutex<HashMap<String, Child>>`. `Command::spawn` is a cheap
//! synchronous syscall (fork+exec on Unix, CreateProcess on Windows), so it
//! is invoked directly from the async context without `spawn_blocking` —
//! the children map is only ever touched via the async-aware
//! `lock().await`.
//!
//! `stdout` and `stderr` are both inherited from the SMOS process so
//! `llama-server` log lines reach the operator's journald / file logger
//! without SMOS doing any draining. A `Stdio::piped()` here would deadlock
//! once the OS pipe buffer fills up — SMOS never reads from the pipe in the
//! current lifecycle.

use std::collections::HashMap;
use std::process::{Child, Stdio};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use reqwest::Client;
use tokio::sync::Mutex;

use crate::llama_server::config::{LlamaCppConfig, LlamaCppServiceConfig};
use crate::llama_server::health::{is_port_responding, probe_client, wait_for_health};
use crate::paths::expand_tilde;

/// Service-slot identifier used as the `HashMap` key inside [`Self::children`].
const EMBEDDING: &str = "embedding";
const RERANKER: &str = "reranker";
const EXTRACTION: &str = "extraction";

/// Lifecycle handle for the configured `llama-server` processes.
///
/// Constructed from [`LlamaCppConfig`]; [`Self::launch_all`] probes every
/// configured service's port and spawns a fresh `llama-server` when nothing
/// is listening yet; [`Self::shutdown_all`] kills every spawned child. The
/// struct is safe to clone when the caller wants to drop a second handle
/// into a shutdown path — the children live behind a shared `Arc<Mutex<…>>`.
#[derive(Clone)]
pub struct LlamaCppManager {
    config: Arc<LlamaCppConfig>,
    children: Arc<Mutex<HashMap<String, Child>>>,
    probe_client: Client,
}

impl LlamaCppManager {
    /// Build the manager and its probe client. Construction does NOT spawn
    /// any process — call [`Self::launch_all`] to do that.
    pub fn new(config: LlamaCppConfig) -> Result<Self> {
        let probe_client = probe_client()?;
        Ok(Self {
            config: Arc::new(config),
            children: Arc::new(Mutex::new(HashMap::new())),
            probe_client,
        })
    }

    /// Launch every configured service that is not already responding on
    /// its port. Services with `port == 0` or empty `model_path` are
    /// skipped (see [`LlamaCppServiceConfig::is_configured`]).
    pub async fn launch_all(&self) -> Result<()> {
        if !self.config.auto_launch {
            tracing::info!("llama.cpp auto-launch disabled");
            return Ok(());
        }

        for (name, service) in self.configured_services() {
            if is_port_responding(&self.probe_client, service.port).await {
                tracing::info!(service = name, port = service.port, "already running, skip");
                continue;
            }
            self.launch_service(name, &service).await?;
            wait_for_health(&self.probe_client, name, service.port).await?;
        }
        Ok(())
    }

    /// Return the configured services in launch order (embedding →
    /// reranker → extraction) with already-expanded paths. Half-configured
    /// services (`port == 0` or empty `model_path`) are filtered out so
    /// the operator can keep the default section around without spawning
    /// processes for services they do not need.
    fn configured_services(&self) -> Vec<(&'static str, LlamaCppServiceConfig)> {
        let candidates = [
            (EMBEDDING, &self.config.embedding),
            (RERANKER, &self.config.reranker),
            (EXTRACTION, &self.config.extraction),
        ];
        candidates
            .into_iter()
            .filter(|(_, s)| s.is_configured())
            .map(|(name, s)| (name, expand_paths(s)))
            .collect()
    }

    /// Spawn one `llama-server` and register the resulting [`Child`] in
    /// [`Self::children`]. The lock is held only for the HashMap insert;
    /// `Command::spawn` itself runs before the lock is acquired.
    async fn launch_service(
        &self,
        name: &'static str,
        service: &LlamaCppServiceConfig,
    ) -> Result<()> {
        tracing::info!(
            service = name,
            port = service.port,
            model = %service.model_path,
            "launching llama-server"
        );

        let child = spawn_llama_server(name, &self.config.binary, service)?;

        let mut children = self.children.lock().await;
        children.insert(name.to_string(), child);
        Ok(())
    }

    /// Kill every spawned child and drop the handles. Idempotent — calling
    /// it twice is safe (the second call walks an empty map).
    pub async fn shutdown_all(&self) {
        let mut children = self.children.lock().await;
        for (name, child) in children.iter_mut() {
            tracing::info!(service = name, "stopping llama-server");
            let _ = child.kill();
            let _ = child.wait();
        }
        children.clear();
    }
}

/// Expand `~` in the service's `model_path`. The other fields are passed
/// through unchanged. The result is owned so the caller can hand it to
/// the spawn helper without borrowing the config.
fn expand_paths(service: &LlamaCppServiceConfig) -> LlamaCppServiceConfig {
    LlamaCppServiceConfig {
        model_path: expand_tilde(&service.model_path)
            .to_string_lossy()
            .into_owned(),
        port: service.port,
        extra_args: service.extra_args.clone(),
    }
}

/// Build the `llama-server` `Command` and spawn it. `stdin` is closed
/// (`llama-server` never reads it) and `stdout` / `stderr` are inherited so
/// the server's log lines reach the operator's journald / container
/// json-file logger without SMOS having to drain a pipe. A piped stderr
/// nobody reads would deadlock the server once the OS pipe buffer (~4 KB
/// on Windows) fills.
fn spawn_llama_server(
    name: &'static str,
    binary: &str,
    service: &LlamaCppServiceConfig,
) -> Result<Child> {
    let mut cmd = std::process::Command::new(binary);
    cmd.arg("--model").arg(&service.model_path);
    cmd.arg("--port").arg(service.port.to_string());
    cmd.args(&service.extra_args);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());
    cmd.spawn()
        .map_err(|e| anyhow!("failed to launch llama-server for {name}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unconfigured() -> LlamaCppConfig {
        LlamaCppConfig {
            binary: "llama-server".into(),
            auto_launch: true,
            embedding: LlamaCppServiceConfig::default(),
            reranker: LlamaCppServiceConfig::default(),
            extraction: LlamaCppServiceConfig::default(),
        }
    }

    #[test]
    fn configured_services_filters_half_configured_entries() {
        let mut cfg = unconfigured();
        cfg.embedding = LlamaCppServiceConfig {
            model_path: "/m/e.gguf".into(),
            port: 8081,
            extra_args: vec![],
        };
        // reranker + extraction stay default (port 0) and MUST be filtered.
        let manager = LlamaCppManager::new(cfg).expect("build");
        let services = manager.configured_services();
        assert_eq!(services.len(), 1);
        assert_eq!(services[0].0, EMBEDDING);
    }

    #[tokio::test]
    async fn launch_all_is_noop_when_auto_launch_disabled() {
        let mut cfg = unconfigured();
        cfg.auto_launch = false;
        let manager = LlamaCppManager::new(cfg).expect("build");
        manager.launch_all().await.expect("disabled is ok");
    }
}
