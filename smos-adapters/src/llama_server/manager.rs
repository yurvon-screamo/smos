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

        let idle_supported = probe_idle_support(&self.config.binary);
        let idle_seconds = effective_idle_seconds(self.config.idle_timeout_seconds);
        if wants_idle_flag(idle_seconds) && !idle_supported {
            tracing::warn!(
                binary = %self.config.binary,
                "llama-server does not advertise --sleep-idle-seconds; \
                 VRAM idle-unload disabled for this run"
            );
        }

        let child = spawn_llama_server(
            name,
            &self.config.binary,
            service,
            idle_args(idle_seconds, idle_supported),
        )?;

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
///
/// `idle_args` carries the optional `--sleep-idle-seconds <n>` argv pair;
/// an empty slice leaves the command unchanged.
fn spawn_llama_server(
    name: &'static str,
    binary: &str,
    service: &LlamaCppServiceConfig,
    idle_args: Vec<String>,
) -> Result<Child> {
    let mut cmd = std::process::Command::new(binary);
    cmd.arg("--model").arg(&service.model_path);
    cmd.arg("--port").arg(service.port.to_string());
    cmd.args(&service.extra_args);
    cmd.args(&idle_args);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());
    cmd.spawn()
        .map_err(|e| anyhow!("failed to launch llama-server for {name}: {e}"))
}

/// Resolve the effective idle timeout. `None` and `Some(0)` both disable
/// the flag — `None` is treated as "operator never set it" (the default
/// fills in 300 on the config side, but defensive code here keeps the
/// manager correct in isolation), `Some(0)` is the explicit opt-out.
fn effective_idle_seconds(configured: Option<u64>) -> Option<u64> {
    match configured {
        Some(0) | None => None,
        Some(seconds) => Some(seconds),
    }
}

/// `true` when the configured idle timeout requests the flag at all.
fn wants_idle_flag(seconds: Option<u64>) -> bool {
    seconds.is_some()
}

/// Build the argv pair for `--sleep-idle-seconds` when both the operator
/// wants it AND the underlying `llama-server` build advertises support.
/// Returns an empty `Vec` otherwise.
fn idle_args(seconds: Option<u64>, supported: bool) -> Vec<String> {
    if let Some(s) = seconds
        && supported
    {
        return vec!["--sleep-idle-seconds".into(), s.to_string()];
    }
    Vec::new()
}

/// Probe `llama-server --help` for the `--sleep-idle-seconds` flag.
///
/// Falls back to `false` on any spawn/IO failure: when the probe cannot
/// run we MUST NOT add the flag blindly — passing an unknown option to
/// `llama-server` makes it exit 1 on launch, turning every service into a
/// 30-second health-probe timeout. A WARN log is emitted by the caller so
/// the operator sees why VRAM idle-unload did not engage.
fn probe_idle_support(binary: &str) -> bool {
    let output = std::process::Command::new(binary).arg("--help").output();
    match output {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            stdout.contains("--sleep-idle-seconds") || stdout.contains("sleep-idle")
        }
        Err(e) => {
            tracing::warn!(
                binary = binary,
                error = %e,
                "could not run `llama-server --help`; assuming no --sleep-idle-seconds support"
            );
            false
        }
    }
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
            idle_timeout_seconds: Some(300),
        }
    }

    #[test]
    fn configured_services_filters_half_configured_entries() {
        let mut cfg = unconfigured();
        cfg.embedding = LlamaCppServiceConfig {
            model_path: "/m/e.gguf".into(),
            port: 28081,
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

    #[test]
    fn effective_idle_seconds_disables_for_zero_or_none() {
        assert_eq!(effective_idle_seconds(None), None);
        assert_eq!(effective_idle_seconds(Some(0)), None);
        assert_eq!(effective_idle_seconds(Some(1)), Some(1));
        assert_eq!(effective_idle_seconds(Some(300)), Some(300));
    }

    #[test]
    fn wants_idle_flag_only_when_some_seconds() {
        assert!(!wants_idle_flag(None));
        // Some(0) maps to None in effective_idle_seconds, so callers always
        // pass that resolved value here — wants_idle_flag only ever sees a
        // real number when it should return true.
        assert!(wants_idle_flag(Some(300)));
    }

    #[test]
    fn idle_args_empty_when_unsupported_or_disabled() {
        assert!(idle_args(None, true).is_empty(), "disabled → no flag");
        assert!(
            idle_args(Some(300), false).is_empty(),
            "unsupported binary → no flag"
        );
    }

    #[test]
    fn idle_args_emits_pair_when_supported_and_enabled() {
        let args = idle_args(Some(300), true);
        assert_eq!(args, vec!["--sleep-idle-seconds".to_string(), "300".into()]);
    }

    #[test]
    fn probe_idle_support_returns_false_when_binary_missing() {
        // A binary that does not exist cannot be probed — the helper must
        // return false (NOT panic) so serve continues without the flag.
        let supported = probe_idle_support("/this/binary/does/not/exist/llama-server");
        assert!(!supported);
    }
}
