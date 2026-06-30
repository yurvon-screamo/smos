use super::types::*;

// ---------------------------------------------------------------------------
// Default impls
// ---------------------------------------------------------------------------

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8888,
            shutdown_extraction_grace_seconds: 30,
            enable_response_extraction: true,
            graceful_degradation: true,
            log_format: "json".into(),
        }
    }
}

impl Default for SurrealConfig {
    fn default() -> Self {
        let paths = crate::paths::SmosPaths::resolve();
        Self {
            path: paths.db.join("smos.db").to_string_lossy().into_owned(),
            namespace: "smos".into(),
            database: "smos".into(),
        }
    }
}

impl Default for LlmExtractionConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:28082".into(),
            model: "qwen3.5-2b".into(),
            api_key: String::new(),
            timeout_seconds: 30,
            temperature: 0.0,
            seed: 42,
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:28081".into(),
            model: "hf.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:latest".into(),
            dimensions: 1024,
            api_key: String::new(),
            timeout_seconds: 30,
        }
    }
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:28181".into(),
            model: "qwen3-reranker".into(),
            timeout_seconds: 60,
        }
    }
}

impl Default for NliBackendConfig {
    fn default() -> Self {
        let paths = crate::paths::SmosPaths::resolve();
        Self {
            model: "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli".into(),
            cache_dir: paths.models.to_string_lossy().into_owned(),
            device: "auto".into(),
            ort_cache_dir: paths.models.join("ort").to_string_lossy().into_owned(),
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            timeout_seconds: 1800,
            pending_overflow_threshold: 20,
            scan_interval_seconds: 15,
        }
    }
}

impl Default for AuditConfig {
    fn default() -> Self {
        let paths = crate::paths::SmosPaths::resolve();
        Self {
            enabled: false,
            schedule: "0 3 * * *".into(),
            llm_provider: "cloud".into(),
            cloud_model: "z-ai/glm-4.6".into(),
            cloud_api_key: String::new(),
            cloud_base_url: "https://openrouter.ai/api/v1".into(),
            local_model: "qwen3.5-2b".into(),
            local_url: "http://localhost:28082".into(),
            max_deletions_per_run: 50,
            max_merges_per_run: 100,
            max_tool_rounds: 10,
            audit_timeout_secs: 300,
            report_dir: paths.reports.to_string_lossy().into_owned(),
        }
    }
}

impl Default for GitConfig {
    fn default() -> Self {
        let paths = crate::paths::SmosPaths::resolve();
        Self {
            repo_url: String::new(),
            branch: "main".into(),
            auto_push: false,
            local_path: paths
                .home
                .join("git")
                .join("memory")
                .to_string_lossy()
                .into_owned(),
            disable_gpg_sign: true,
        }
    }
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            forward_mode: "auto".into(),
            forward_probe_timeout_ms: 250,
        }
    }
}

// ---------------------------------------------------------------------------
// Defaults for serde `default = "..."` attributes
// ---------------------------------------------------------------------------

pub(crate) fn default_auth_header() -> String {
    "Authorization".into()
}

pub(crate) fn default_provider_timeout() -> u64 {
    120
}

// ---------------------------------------------------------------------------
// ProviderConfig helpers
// ---------------------------------------------------------------------------

impl ProviderConfig {
    /// Construct with the canonical defaults for the optional fields. Used by
    /// tests and by `smos init` when scaffolding a default config.
    pub fn new(name: &str, url: &str) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            api_key_env: String::new(),
            auth_header: default_auth_header(),
            timeout_seconds: default_provider_timeout(),
        }
    }

    /// Resolve the API key by reading the env var named in `api_key_env`.
    /// Returns an empty string when `api_key_env` is empty (the "no auth"
    /// case for a local `llama-server`). A missing env var also yields an
    /// empty string so a misconfigured `api_key_env` surfaces as an
    /// unauthenticated request (visible as a 401 from the upstream) rather
    /// than a startup panic.
    pub fn resolve_api_key(&self) -> String {
        if self.api_key_env.is_empty() {
            return String::new();
        }
        std::env::var(&self.api_key_env).unwrap_or_default()
    }
}
