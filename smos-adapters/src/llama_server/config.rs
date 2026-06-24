//! Configuration types for the llama.cpp auto-launch manager.
//!
//! These live in their own file (rather than inline in
//! [`crate::config::SmosConfig`]) because the auto-launch surface carries
//! its own nested defaults (per-service ports, per-service extra args) and
//! keeps the top-level config file readable.

use serde::{Deserialize, Serialize};

use crate::paths::SmosPaths;

/// Top-level llama.cpp auto-launch configuration (`[llama_cpp]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlamaCppConfig {
    /// `llama-server` binary name (resolved via `PATH`) or absolute path.
    pub binary: String,
    /// When `true`, [`super::LlamaCppManager::launch_all`] spawns the
    /// configured services; when `false`, it is a no-op log line.
    pub auto_launch: bool,
    /// Embedding service (consumed by [`crate::providers::OllamaEmbedding`]
    /// when its URL points at `http://localhost:<port>`).
    pub embedding: LlamaCppServiceConfig,
    /// Reranker service (consumed by [`crate::providers::LlamaCppReranker`]
    /// when its URL points at `http://localhost:<port>`).
    pub reranker: LlamaCppServiceConfig,
    /// Extraction service (consumed by [`crate::providers::OllamaExtractor`]
    /// when its URL points at `http://localhost:<port>`).
    pub extraction: LlamaCppServiceConfig,
    /// Idle timeout (seconds) after which `llama-server` should unload the
    /// model from VRAM via the `--sleep-idle-seconds` CLI flag. Defaults
    /// to `Some(300)` (5 minutes); set to `Some(0)` to disable the
    /// behaviour entirely. The flag is only appended when the underlying
    /// `llama-server` build advertises support (probed via `--help`) — a
    /// missing flag is logged as WARN and serve continues without it.
    pub idle_timeout_seconds: Option<u64>,
}

/// One llama.cpp service: model path + port + extra CLI args.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LlamaCppServiceConfig {
    /// Filesystem path to the GGUF model. `~` is expanded to the user home
    /// at launch time. Empty path means "this service is not configured";
    /// [`super::LlamaCppManager`] skips it instead of spawning a process
    /// with no model.
    pub model_path: String,
    /// TCP port the `llama-server` listens on. `0` means "not configured"
    /// and the manager skips the service.
    pub port: u16,
    /// Extra CLI arguments forwarded verbatim to `llama-server` (e.g.
    /// `["--ctx-size", "2048"]`).
    pub extra_args: Vec<String>,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        let paths = SmosPaths::resolve();
        let mk = |name: &str| paths.models.join(name).to_string_lossy().into_owned();
        Self {
            binary: "llama-server".into(),
            auto_launch: true,
            embedding: LlamaCppServiceConfig {
                model_path: mk("jina-embeddings-v5.gguf"),
                port: 28081,
                extra_args: vec!["--ctx-size".into(), "2048".into(), "--embeddings".into()],
            },
            reranker: LlamaCppServiceConfig {
                model_path: mk("qwen3-reranker.gguf"),
                port: 28181,
                extra_args: vec!["--ctx-size".into(), "8192".into()],
            },
            extraction: LlamaCppServiceConfig {
                model_path: mk("nemotron-3-nano-4b.gguf"),
                port: 28082,
                extra_args: vec!["--ctx-size".into(), "4096".into()],
            },
            idle_timeout_seconds: Some(300),
        }
    }
}

impl LlamaCppServiceConfig {
    /// `true` when both a model path and a non-zero port are configured.
    /// Used by [`super::LlamaCppManager`] to skip half-configured services
    /// instead of spawning a `llama-server` that would immediately exit.
    pub fn is_configured(&self) -> bool {
        !self.model_path.trim().is_empty() && self.port != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_enables_auto_launch() {
        let cfg = LlamaCppConfig::default();
        assert!(
            cfg.auto_launch,
            "auto_launch defaults to true so `smos serve` spawns llama-server \
             out of the box"
        );
        assert_eq!(cfg.binary, "llama-server");
    }

    #[test]
    fn default_assigns_distinct_ports() {
        let cfg = LlamaCppConfig::default();
        let ports = [cfg.embedding.port, cfg.reranker.port, cfg.extraction.port];
        let unique: std::collections::HashSet<_> = ports.iter().collect();
        assert_eq!(unique.len(), 3, "service ports must be distinct");
    }

    #[test]
    fn default_idle_timeout_is_five_minutes() {
        let cfg = LlamaCppConfig::default();
        assert_eq!(
            cfg.idle_timeout_seconds,
            Some(300),
            "VRAM-idle default is 5 minutes; opt-out is Some(0), not None"
        );
    }

    #[test]
    fn service_is_configured_requires_model_and_port() {
        let unconfigured = LlamaCppServiceConfig::default();
        assert!(!unconfigured.is_configured());

        let with_port = LlamaCppServiceConfig {
            port: 28081,
            ..Default::default()
        };
        assert!(!with_port.is_configured(), "empty model still rejected");

        let full = LlamaCppServiceConfig {
            model_path: "/x/m.gguf".into(),
            port: 28081,
            extra_args: vec![],
        };
        assert!(full.is_configured());
    }
}
