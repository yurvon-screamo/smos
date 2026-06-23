//! SMOS proxy configuration (`smos.toml`).
//!
//! The config is layered: sections present in the TOML file override the
//! [`Default`] values; any section missing from the file falls back to its
//! canonical default. This keeps the in-repo `smos.toml` minimal (operators
//! override only what they need) while `cargo run --bin smos -- serve` still
//! works with no file at all.
//!
//! The external `config` crate is referenced as `::config` because this module
//! is itself named `config` — the leading `::` unambiguously reaches the
//! external crate instead of recursing into `crate::config`.
//!
//! # Section map
//!
//! | TOML section        | Rust field                 | Notes                          |
//!|---------------------|----------------------------|--------------------------------|
//! | `[surreal]`         | [`SurrealConfig`]          |                                |
//! | `[server]`          | [`ServerConfig`]           |                                |
//! | `[[providers]]`     | [`ProviderConfig`]         | OpenAI-compatible LLM endpoints. |
//! | `[persons.*]`       | [`PersonConfig`]           | LLM personas = memory keys.    |
//! | `[llm_extraction]`  | [`LlmExtractionConfig`]    |                                |
//! | `[embedding]`       | [`EmbeddingConfig`]        |                                |
//! | `[reranker]`        | [`RerankerConfig`]         |                                |
//! | `[retrieval]`       | [`RetrievalConfig`]        | Re-exported from `smos-domain`.|
//! | `[merge]`           | [`MergeConfig`]            | Re-exported from `smos-domain`.|
//! | `[confidence]`      | [`ConfidenceConfig`]       | Re-exported from `smos-domain`.|
//! | `[heat]`            | [`HeatConfig`]             | Re-exported from `smos-domain`.|
//! | `[nli]`             | [`NliConfig`]              | Domain verdict thresholds.     |
//! | `[nli_backend]`     | [`NliBackendConfig`]       | Adapter-only: native ort/ONNX `model` + `cache_dir`. |
//! | `[extraction]`      | [`ExtractionConfig`]       | Re-exported from `smos-domain`.|
//! | `[session]`         | [`SessionConfig`]          |                                |
//! | `[audit]`           | [`AuditConfig`]            | Dreaming agent (LLM audit).    |
//! | `[llama_cpp]`       | [`LlamaCppConfig`]         | Auto-launch `llama-server`     |
//! |                     |                            | processes on `smos serve`.     |
//! | `[git]`             | [`GitConfig`]              | Git-backed memory sync.        |

use serde::{Deserialize, Serialize};
pub use smos_domain::config::{
    ConfidenceConfig, ExtractionConfig, HeatConfig, MergeConfig, NliConfig, RetrievalConfig,
};

/// Error surface for [`SmosConfig`] loading + validation.
///
/// Wraps the foreign `::config::ConfigError` (file IO + deserialisation
/// failures) and adds a [`Self::Validation`] variant for the semantic range
/// checks enforced by [`SmosConfig::validate`]. A dedicated enum (instead of
/// re-using `::config::ConfigError` directly) is required because the foreign
/// type has no `Validation` variant and we cannot extend it; conflating the
/// two failure modes into a single string-typed error would also lose the
/// `std::error::Error::source` chain.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// File IO or TOML/JSON deserialisation failure, surfaced verbatim from
    /// the `config` crate.
    #[error(transparent)]
    Load(#[from] ::config::ConfigError),

    /// One or more semantic range / consistency checks failed. The string
    /// joins every problem found in one pass so an operator fixing a
    /// misconfigured `smos.toml` sees every issue at once instead of
    /// running `smos serve` N times to discover them one by one.
    #[error("config validation failed: {0}")]
    Validation(String),
}

/// Root configuration.
///
/// Sections that originate in `smos-domain` (`retrieval`, `merge`,
/// `confidence`, `heat`, `nli`) are re-exported from this module so callers
/// have a single import path. Sections that only make sense at the adapter
/// boundary (`surreal`, `server`, `providers`, `persons`, `llm_extraction`,
/// `embedding`, `reranker`, `session`) live here.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SmosConfig {
    #[serde(default)]
    pub surreal: SurrealConfig,

    #[serde(default)]
    pub server: ServerConfig,

    /// LLM chat-completion endpoints declared via `[[providers]]`. Each
    /// entry is one OpenAI-compatible upstream (`llama-server`, OpenRouter,
    /// etc.). The proxy forwards each request to exactly one provider,
    /// chosen by the person → provider map (`[persons.*]`).
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,

    /// LLM personas — the routing map. Each person is simultaneously a
    /// memory key (the namespace under which extracted facts land) and a
    /// routing entry (which provider to use, which upstream model, and
    /// which persona `.md` to inject as the system message).
    #[serde(default)]
    pub persons: std::collections::HashMap<String, PersonConfig>,

    /// Provider-agnostic config for the fact-extraction LLM
    /// (`/v1/chat/completions`-style endpoint, served by `llama-server`).
    #[serde(default)]
    pub llm_extraction: LlmExtractionConfig,

    /// Provider-agnostic config for the embedding model.
    #[serde(default)]
    pub embedding: EmbeddingConfig,

    #[serde(default)]
    pub reranker: RerankerConfig,

    #[serde(default)]
    pub retrieval: RetrievalConfig,

    #[serde(default)]
    pub merge: MergeConfig,

    #[serde(default)]
    pub confidence: ConfidenceConfig,

    #[serde(default)]
    pub heat: HeatConfig,

    /// NLI verdict thresholds (domain layer). Drives the
    /// `is_contradiction` / `is_entailment` / `decide_merge` predicates.
    #[serde(default)]
    pub nli: NliConfig,

    /// Native ort/ONNX backend for NLI inference. Adapter-only: the model id
    /// and cache directory are interpreter-level data that the domain layer
    /// never reads — keeping them out of `smos-domain::NliConfig` preserves
    /// the layering invariant ("domain types carry no IO-boundary data").
    #[serde(default)]
    pub nli_backend: NliBackendConfig,

    /// Semantic dedup safety net for fact extraction (`persist_facts` step 2).
    /// Backs the cosine-similarity gate the extractor falls back to when
    /// `FactId = SHA1(content)` exact match misses a rephrased re-observation.
    #[serde(default)]
    pub extraction: ExtractionConfig,

    #[serde(default)]
    pub session: SessionConfig,

    /// SMOS Dreaming Agent — autonomous periodic audit of stored memory
    /// (deletions of trivial facts, semantic-duplicate merges, conflict
    /// flagging, markdown report). Disabled by default so a fresh `smos.toml`
    /// never silently spends LLM tokens.
    #[serde(default)]
    pub audit: AuditConfig,

    /// llama.cpp auto-launch. When enabled, `smos serve` spawns the configured
    /// `llama-server` processes for embedding / reranker / extraction at
    /// startup so the operator does not have to start them by hand. Each
    /// section's `port` is probed first; an already-running service is
    /// reused as-is.
    #[serde(default)]
    pub llama_cpp: crate::llama_server::LlamaCppConfig,

    /// Git-backed memory sync. When `repo_url` is set, SMOS writes extracted
    /// facts to a local clone of the repo as markdown files (frontmatter +
    /// body) and commits + (optionally) pushes after every FinalizeSession.
    /// The `smos import git <url>` subcommand reads the same layout back
    /// into SurrealDB so two SMOS instances can share memory through git.
    #[serde(default)]
    pub git: GitConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SurrealConfig {
    pub path: String,
    pub namespace: String,
    pub database: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub shutdown_extraction_grace_seconds: u64,
    pub enable_response_extraction: bool,
    pub graceful_degradation: bool,
    pub log_format: String,
}

/// One LLM chat-completion provider. Multiple providers can be declared via
/// `[[providers]]`; the active one per request is chosen by the person →
/// provider map (`[persons.*].provider`).
///
/// ```toml
/// [[providers]]
/// name = "llama-local"
/// url = "http://localhost:28082/v1/chat/completions"
/// api_key_env = ""        # env var name; empty = no auth header sent
///
/// timeout_seconds = 120   # optional, defaults to 120
/// auth_header = "Authorization"  # optional, defaults to "Authorization"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Operator-facing identifier referenced from `[persons.*].provider`.
    /// MUST be unique within the `[[providers]]` array; a duplicate is a
    /// config error surfaced by [`SmosConfig::validate`].
    pub name: String,
    /// Full chat-completions URL (with path).
    pub url: String,
    /// Name of the environment variable that carries the API key. Empty
    /// means "no auth" (suitable for a local `llama-server`). Resolved at
    /// startup via `std::env::var`, never written to disk. Keeping the
    /// env-var name (rather than the literal key) follows the same
    /// secret-hygiene rule as the dreaming agent's `${ENV_VAR}` placeholder.
    #[serde(default)]
    pub api_key_env: String,
    /// Header name to carry the auth token. Defaults to `Authorization`
    /// (OpenAI / `llama-server`). Override to `api-key` for Azure-style
    /// endpoints.
    #[serde(default = "default_auth_header")]
    pub auth_header: String,
    /// Per-request HTTP timeout. Defaults to 120 s.
    #[serde(default = "default_provider_timeout")]
    pub timeout_seconds: u64,
}

/// LLM persona — simultaneously a memory key and a routing entry.
///
/// Each person is the value of `request.model` on the wire: a client that
/// wants persona `bob` sends `{"model": "bob", ...}`. The proxy resolves
/// the person to:
/// - a memory key (the person name, validated through `MemoryKey::from_raw`),
/// - a provider (looked up by `provider` in the `[[providers]]` array),
/// - an upstream model id (rewritten into `request.model` before forward),
/// - an optional persona `.md` (loaded from `persona` and prepended to the
///   system message).
///
/// ```toml
/// [persons.bob]
/// provider = "llama-local"
/// model = "nemotron-3-nano-4b"
/// persona = "~/.smos/persons/bob.md"  # optional
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonConfig {
    /// Name of the provider entry (MUST match one `[[providers]].name`).
    pub provider: String,
    /// Upstream model id forwarded as `request.model`.
    pub model: String,
    /// Filesystem path to the persona `.md` file. The `~` prefix is expanded
    /// to the user home directory at load time. Empty path = no persona
    /// injection (the request is forwarded verbatim, the person name is
    /// still used as the memory key).
    #[serde(default)]
    pub persona: String,
}

/// LLM fact-extraction endpoint config (provider-agnostic).
///
/// Backs the post-response extraction pipeline. The endpoint is expected to
/// be an OpenAI-compatible `/v1/chat/completions` shape
/// (`{model, messages, temperature, seed, stream}`), e.g. a `llama-server`
/// instance started with the configured extraction GGUF. Cloud providers are
/// supported as long as they accept that request envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmExtractionConfig {
    /// API base URL (no path suffix). The extractor appends
    /// `/v1/chat/completions`.
    pub url: String,
    /// Model id passed in the `model` field of `/v1/chat/completions`.
    pub model: String,
    /// Optional API key for cloud providers (a local `llama-server` ignores
    /// the field).
    #[serde(default)]
    pub api_key: String,
    /// Per-request HTTP timeout.
    pub timeout_seconds: u64,
    /// Sampling temperature passed as the top-level `temperature`. `0.0`
    /// (greedy decoding) is the near-deterministic baseline.
    pub temperature: f32,
    /// Sampling seed passed as the top-level `seed`. Pairing
    /// `temperature = 0.0` with a pinned `seed` makes the extractor re-yield
    /// the same bullet list across runs on the same backend.
    pub seed: u32,
}

/// Embedding endpoint config (provider-agnostic).
///
/// Backs the topic-embedding step of the enrich pipeline. The endpoint is
/// expected to be an OpenAI-compatible `/v1/embeddings` shape
/// (`{model, input}`), e.g. a `llama-server` instance started with the
/// configured embedding GGUF. Cloud providers are supported as long as they
/// accept that envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// API base URL (no path suffix). The adapter appends `/v1/embeddings`.
    /// May differ from [`LlmExtractionConfig::url`] so the embedder can run
    /// on a different host (or a different provider entirely).
    pub url: String,
    /// Model id passed in the `model` field of `/v1/embeddings`.
    pub model: String,
    /// Vector dimensionality. MUST match the HNSW index declared in
    /// `surreal_schema::FACT_DDL`. The default 1024 matches the canonical
    /// Jina v5 retrieval-GGUF config; override only if you re-index.
    pub dimensions: usize,
    /// Optional API key for cloud providers (a local `llama-server` ignores
    /// the field).
    #[serde(default)]
    pub api_key: String,
    /// Per-request HTTP timeout.
    pub timeout_seconds: u64,
}

/// llama.cpp reranker server connection.
///
/// The adapter expects an OpenAI-compatible `/v1/rerank` endpoint (e.g. the
/// `llama-server` binary shipped with llama.cpp when started with a reranker
/// model such as Qwen3-Reranker).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankerConfig {
    /// Base URL of the reranker server (no path suffix).
    pub url: String,
    /// Model id passed in the `model` field of `/v1/rerank`.
    pub model: String,
    /// Per-request HTTP timeout.
    pub timeout_seconds: u64,
}

/// Native ort/ONNX backend for NLI inference — adapter-only sibling of the
/// domain [`NliConfig`].
///
/// The domain layer never interprets `model` or `cache_dir`; they are read
/// exactly once at startup by [`crate::nli::build_classifier`] and passed to
/// the ort session build. Keeping them in this adapter-side struct (rather
/// than the domain `NliConfig`) preserves the "domain carries no
/// IO-boundary data" invariant.
///
/// `deny_unknown_fields` mirrors the domain `NliConfig`: a typo here fails
/// loudly at startup instead of silently dropping the configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NliBackendConfig {
    /// Hugging Face model id downloaded by the native backend. The default
    /// matches the POC's benchmark winner (DeBERTa-v3 large, MNLI + FEVER +
    /// ANLI + ling-wanli).
    pub model: String,
    /// Local directory used to cache the ONNX model + tokenizer artifacts
    /// downloaded from HF Hub. The native backend writes a flat
    /// `model_quantized.onnx` + `tokenizer.json` here.
    pub cache_dir: String,
}

/// Per-session lifecycle tunables (§3 session detection, §5 pending overflow).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionConfig {
    /// Inactivity duration after which a session is eligible for finalize.
    pub timeout_seconds: u64,
    /// Pending-fact count that triggers an early session-end (§5 overflow).
    #[serde(default)]
    pub pending_overflow_threshold: usize,
    /// Watcher scan cadence. The session watcher wakes every
    /// `scan_interval_seconds` to look for expired / overflowed sessions and
    /// trigger FinalizeSession.
    #[serde(default)]
    pub scan_interval_seconds: u64,
}

/// SMOS Dreaming Agent configuration.
///
/// The dreaming agent is an autonomous LLM-driven auditor that runs on a cron
/// schedule, reviews stored facts, and applies bounded mutations (deletions,
/// merges, conflict flags) before writing a markdown report. The agent
/// operates through `rig::tool::Tool` impls that gate every write operation
/// behind per-run rate limits — a misbehaving LLM cannot nuke the memory
/// store because `DeleteFactTool` refuses calls past `max_deletions_per_run`.
///
/// Provider selection is `"cloud" | "local"`:
/// - `"cloud"` — OpenRouter (or any OpenAI-compatible chat-completions
///   endpoint) identified by `cloud_*` fields. The `cloud_api_key` field
///   accepts either a literal key or the placeholder `"${ENV_VAR}"`, which
///   `dreaming::resolve_env_var` expands via [`std::env::var`]. The
///   placeholder form keeps secrets out of `smos.toml`.
/// - `"local"` — an OpenAI-compatible chat server (default
///   `http://localhost:28082`, i.e. the `llama-server` extraction port). No
///   API key required.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AuditConfig {
    /// Master switch. When `false` the scheduler never starts and `smos audit`
    /// is a no-op. Defaults to `false` so an operator shipping the default
    /// `smos.toml` never silently incurs LLM costs.
    pub enabled: bool,
    /// Cron expression (5-field UNIX style, UTC). Defaults to `0 3 * * *`
    /// (03:00 UTC daily).
    pub schedule: String,
    /// `"cloud"` (default) or `"local"`. Unknown values are rejected by
    /// `dreaming::run_audit` at runtime.
    pub llm_provider: String,
    /// Cloud model id passed to the OpenRouter completions endpoint.
    pub cloud_model: String,
    /// Cloud API key. Accepts `"${ENV_VAR}"` placeholder form; see
    /// [`crate::dreaming::resolve_env_var`].
    pub cloud_api_key: String,
    /// Cloud base URL (no path). Defaults to OpenRouter.
    pub cloud_base_url: String,
    /// Local model id forwarded as `request.model` to the local
    /// OpenAI-compatible chat server (e.g. `nemotron-3-nano-4b`).
    pub local_model: String,
    /// Local chat-server base URL.
    pub local_url: String,
    /// Hard cap on the number of `delete_fact` calls the agent may issue in a
    /// single audit run. Past the cap the tool returns a rate-limit error to
    /// the LLM.
    pub max_deletions_per_run: usize,
    /// Hard cap on the number of `merge_facts` calls per run.
    pub max_merges_per_run: usize,
    /// Maximum number of tool-calling rounds the rig agent may take per
    /// audit run. rig 0.14's `PromptRequest` defaults to single-turn
    /// (`max_depth = 0`), which prevents the tool loop from ever engaging
    /// and surfaces as `MaxDepthError: (reached limit: 0)` on the first
    /// prompt that expects a tool call. The audit workflow drives every
    /// fact query, mutation, and report write through a `rig::tool::Tool`,
    /// so this MUST be > 0; 10 is the canonical headroom for a full
    /// list → search → merge → flag → report sweep.
    pub max_tool_rounds: usize,
    /// Filesystem directory where `write_report` drops the markdown audit
    /// report. Created on first write if missing.
    pub report_dir: String,
}

/// Git-backed memory sync configuration.
///
/// When `repo_url` is non-empty, SMOS dual-writes extracted facts to a local
/// clone of the repo as markdown files (one fact per `.md`, frontmatter +
/// body) and commits + (optionally) pushes after every FinalizeSession. The
/// layout is read back by `smos import git <url>` so two SMOS instances can
/// share memory through git.
///
/// Empty `repo_url` disables sync — the section stays in the default config
/// as documentation but no clone, commit, or push happens.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GitConfig {
    /// Git repository URL. Private repos use the system's SSH credentials
    /// (no inline tokens). Empty disables sync.
    pub repo_url: String,
    /// Branch to commit + push to. Defaults to `main`.
    pub branch: String,
    /// When `true`, `commit_and_push` runs `git push` after the commit. Off
    /// by default — pushing is an opt-in side effect; the operator who
    /// wants live remote sync flips this once `[git].repo_url` is wired up
    /// and the credentials are verified working.
    pub auto_push: bool,
    /// Local clone path. `~` expands to the user home at load time. Defaults
    /// to `~/.smos/git/memory` so the canonical SMOS home layout stays
    /// self-contained.
    pub local_path: String,
    /// Disable GPG-signing the SMOS commits even when the operator's global
    /// git config sets `commit.gpgsign = true`. Defaults to `true`: a SMOS
    /// process running under a service account rarely has a configured GPG
    /// agent, and a missing passphrase prompt would block `commit_and_push`
    /// forever. Set to `false` to honour the operator's global signing
    /// setting (requires the agent to be reachable from the SMOS process).
    pub disable_gpg_sign: bool,
}

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
            model: "nemotron-3-nano-4b".into(),
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
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            timeout_seconds: 1800,
            pending_overflow_threshold: 20,
            scan_interval_seconds: 60,
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
            local_model: "nemotron-3-nano-4b".into(),
            local_url: "http://localhost:28082".into(),
            max_deletions_per_run: 50,
            max_merges_per_run: 100,
            max_tool_rounds: 10,
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

// ---------------------------------------------------------------------------
// Defaults for serde `default = "..."` attributes
// ---------------------------------------------------------------------------

fn default_auth_header() -> String {
    "Authorization".into()
}

fn default_provider_timeout() -> u64 {
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

// ---------------------------------------------------------------------------
// SmosConfig loading
// ---------------------------------------------------------------------------

impl SmosConfig {
    /// Load from a TOML file (overridden by `SMOS__*` environment variables).
    /// Returns defaults when the file is missing so the proxy runs
    /// out-of-the-box without a config file; sections absent from a partial
    /// file also fall back to their defaults via `#[serde(default)]`.
    ///
    /// Environment overrides use the `SMOS__` prefix and a `__` section
    /// separator.
    pub fn load(path: &str) -> Result<Self, ConfigError> {
        let mut builder = ::config::Config::builder();
        if std::path::Path::new(path).exists() {
            // The existence pre-check is load-bearing: `File::with_name`
            // treats its argument as a *stem* and, when the exact file is
            // absent, probes `<name>.toml` (the only registered extension
            // under the `features = ["toml"]` build) — so
            // `with_name("smos.toml")` would search for `smos.toml.toml`.
            // Skipping the source when the path does not exist avoids that
            // probe entirely. Do not remove the pre-check without
            // replacing it; `File::from(Path)` does NOT bypass the probe
            // in config 0.14.x (it funnels through the same `find_file`
            // extension-search path as `with_name`).
            builder = builder.add_source(::config::File::with_name(path));
        }
        builder = builder.add_source(::config::Environment::with_prefix("SMOS").separator("__"));
        let cfg: SmosConfig = builder.build()?.try_deserialize()?;
        // Fail-fast on invalid config: an operator who ships a config with a
        // bad confidence range or a missing embedding dimension should hear
        // about it at startup, not on the first request that hits the broken
        // path. `validate` collects EVERY problem in one pass so a single
        // startup error is enough to fix a half-broken TOML.
        cfg.validate()?;
        Ok(cfg)
    }

    /// Load directly from a TOML string. Used by `smos init`'s self-test
    /// so the canonical default config (shipped as an inline literal) is
    /// validated without going through the file system. Environment
    /// overrides are NOT applied here — the caller already controls the
    /// input verbatim.
    pub fn load_from_str(toml: &str) -> Result<Self, ConfigError> {
        let cfg: SmosConfig = ::config::Config::builder()
            .add_source(::config::File::from_str(toml, ::config::FileFormat::Toml))
            .build()?
            .try_deserialize()?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate every cross-field invariant and range bound in one pass.
    ///
    /// Returns `Ok(())` when every check passes; otherwise returns
    /// [`ConfigError::Validation`] carrying a `;`-joined list of every
    /// problem found so the operator can fix them all in one editing pass
    /// instead of discovering them one `smos serve` invocation at a time.
    ///
    /// The checks mirror the invariants the rest of the code already assumes:
    ///
    /// - `embedding.dimensions == 1024` — must match the HNSW index DDL.
    /// - `confidence.*` ranges + `accept_threshold >= pending_threshold`.
    /// - `extraction.dedup_cosine_threshold` in `[-1, 1]`.
    /// - `llm_extraction.temperature` in `[0, 2]`.
    /// - `session.timeout_seconds > 0`.
    /// - `server.port > 0`.
    /// - `retrieval.top_k_initial > 0` and `retrieval.top_k_final > 0`
    ///   (a zero would either short-circuit the pipeline or surface as a
    ///   mysterious HTTP 503 once the reranker is consulted).
    /// - `reranker.url` non-empty (reranker is a hard dependency: every
    ///   request fails with HTTP 503 while the URL is missing or the server
    ///   is unreachable; an operator who blanks the field gets a startup
    ///   error instead of a silent quality drop).
    /// - `providers` non-empty (the proxy needs at least one provider to
    ///   forward chat completions to) and every provider carries a non-empty
    ///   URL + non-zero timeout + a unique name.
    /// - Every `[persons.*].provider` MUST reference an existing
    ///   `[[providers]].name` (a typo would surface as a 503 on the first
    ///   request that uses the person; surfacing it at startup keeps the
    ///   failure mode loud and immediate).
    /// - `nli.contradiction_threshold` in `[0, 1]`.
    /// - `merge.cosine_threshold` in `[-1, 1]`.
    /// - `audit.*` semantic checks — only enforced when `audit.enabled = true`
    ///   (a disabled audit is opt-in; see [`SmosConfig::validate_audit_always`]
    ///   for the variant that checks audit fields regardless of the enabled
    ///   flag, used by `smos audit --provider` to catch typos before the run).
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut errors: Vec<String> = Vec::new();

        if self.embedding.dimensions != 1024 {
            errors.push(format!(
                "embedding.dimensions must be 1024 (HNSW index dimension), got {}",
                self.embedding.dimensions
            ));
        }

        if !(0.0..=1.0).contains(&self.confidence.base) {
            errors.push(format!(
                "confidence.base must be in [0,1], got {}",
                self.confidence.base
            ));
        }
        if !(0.0..=1.0).contains(&self.confidence.accept_threshold) {
            errors.push(format!(
                "confidence.accept_threshold must be in [0,1], got {}",
                self.confidence.accept_threshold
            ));
        }
        if !(0.0..=1.0).contains(&self.confidence.pending_threshold) {
            errors.push(format!(
                "confidence.pending_threshold must be in [0,1], got {}",
                self.confidence.pending_threshold
            ));
        }
        if self.confidence.accept_threshold < self.confidence.pending_threshold {
            errors.push(format!(
                "confidence.accept_threshold ({}) must be >= pending_threshold ({})",
                self.confidence.accept_threshold, self.confidence.pending_threshold
            ));
        }

        if !(-1.0..=1.0).contains(&self.extraction.dedup_cosine_threshold) {
            errors.push(format!(
                "extraction.dedup_cosine_threshold must be in [-1,1], got {}",
                self.extraction.dedup_cosine_threshold
            ));
        }

        if !(0.0..=2.0).contains(&self.llm_extraction.temperature) {
            errors.push(format!(
                "llm_extraction.temperature must be in [0,2], got {}",
                self.llm_extraction.temperature
            ));
        }

        if self.session.timeout_seconds == 0 {
            errors.push("session.timeout_seconds must be > 0".into());
        }

        if self.server.port == 0 {
            errors.push("server.port must be > 0".into());
        }

        if self.retrieval.top_k_final == 0 {
            // `top_k_final == 0` would make `RerankProvider::rerank` return
            // `Ok(vec![])` (the legitimate "nothing to do" path), which the
            // fail-closed enrich pipeline converts into
            // `ProviderError::InvalidResponse("reranker returned empty
            // results")` → every chat-completion request fails with HTTP
            // 503. Reject at startup so the operator hears about it as a
            // config error, not as a mysterious 503.
            errors.push("retrieval.top_k_final must be > 0".into());
        }

        if self.retrieval.top_k_initial == 0 {
            errors.push("retrieval.top_k_initial must be > 0".into());
        }

        if self.reranker.url.trim().is_empty() {
            errors.push(
                "reranker.url must not be empty — reranker is required for enrichment".into(),
            );
        }

        validate_providers(&self.providers, &mut errors);
        validate_persons(&self.persons, &self.providers, &mut errors);

        if !(0.0..=1.0).contains(&self.nli.contradiction_threshold) {
            errors.push(format!(
                "nli.contradiction_threshold must be in [0,1], got {}",
                self.nli.contradiction_threshold
            ));
        }

        if !(-1.0..=1.0).contains(&self.merge.cosine_threshold) {
            errors.push(format!(
                "merge.cosine_threshold must be in [-1,1], got {}",
                self.merge.cosine_threshold
            ));
        }

        if self.audit.enabled {
            errors.extend(self.validate_audit_fields());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Validation(errors.join("; ")))
        }
    }

    /// Validate the audit fields REGARDLESS of `audit.enabled`.
    ///
    /// Used by `smos audit` (the manual one-shot runner) so a typo in
    /// `cloud_base_url` or an unknown `llm_provider` is surfaced at the
    /// invocation rather than as a runtime error mid-audit. The full
    /// [`SmosConfig::validate`] only checks audit fields when
    /// `audit.enabled = true`, which is correct for `smos serve` (where
    /// the audit is off by default and a stale config should not block
    /// server startup) but wrong for the manual runner.
    pub fn validate_audit_always(&self) -> Result<(), ConfigError> {
        let errors = self.validate_audit_fields();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Validation(errors.join("; ")))
        }
    }

    /// Shared semantic checks for the audit section. Returns the (possibly
    /// empty) list of problems; the caller decides whether to fail or
    /// accumulate them into a wider validation pass.
    fn validate_audit_fields(&self) -> Vec<String> {
        let mut errors: Vec<String> = Vec::new();
        if self.audit.schedule.trim().is_empty() {
            errors.push("audit.schedule must not be empty when audit is enabled".into());
        }
        let provider = self.audit.llm_provider.as_str();
        if !matches!(provider, "cloud" | "local") {
            errors.push(format!(
                "audit.llm_provider must be 'cloud' or 'local', got {provider:?}"
            ));
        }
        if provider == "cloud" && self.audit.cloud_base_url.trim().is_empty() {
            errors.push("audit.cloud_base_url must not be empty for the cloud provider".into());
        }
        if provider == "local" && self.audit.local_url.trim().is_empty() {
            errors.push("audit.local_url must not be empty for the local provider".into());
        }
        if self.audit.max_tool_rounds == 0 {
            // `max_tool_rounds = 0` reproduces the rig 0.14 default
            // (`max_depth = 0`), which short-circuits the tool-calling loop
            // and makes the audit fail with `MaxDepthError` on the first
            // prompt that needs a tool. Reject at startup so the operator
            // hears about it as a config error, not as a mid-audit crash.
            errors.push("audit.max_tool_rounds must be > 0".into());
        }
        errors
    }
}

/// Validate the `[[providers]]` array in isolation.
///
/// Pushed out of [`SmosConfig::validate`] so the body of `validate` stays
/// readable. The same checks run whether the array came from TOML or from
/// `SmosConfig::default` + programmatic edits (which is how the E2E helpers
/// build a config).
fn validate_providers(providers: &[ProviderConfig], errors: &mut Vec<String>) {
    if providers.is_empty() {
        errors.push("providers must not be empty".into());
        return;
    }
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for (i, p) in providers.iter().enumerate() {
        if p.timeout_seconds == 0 {
            errors.push(format!("providers[{i}].timeout_seconds must be > 0"));
        }
        if p.url.is_empty() {
            errors.push(format!("providers[{i}].url must not be empty"));
        }
        if p.name.is_empty() {
            errors.push(format!("providers[{i}].name must not be empty"));
            continue;
        }
        if !seen.insert(p.name.as_str()) {
            errors.push(format!(
                "providers[{i}].name = {:?} is duplicated; provider names must be unique",
                p.name
            ));
        }
    }
}

/// Validate the `[persons.*]` map. Each person MUST:
/// - reference a provider that exists in the `[[providers]]` array,
/// - declare a non-empty upstream model,
/// - carry a name that is a valid `MemoryKey` (the name is used as the
///   memory namespace at runtime — `[persons."a/b"]` would 400 every
///   request because the router rejects path-traversal characters).
///
/// Surfacing these at startup matches the documented "loud and immediate"
/// validation philosophy (see [`SmosConfig::validate`]).
fn validate_persons(
    persons: &std::collections::HashMap<String, PersonConfig>,
    providers: &[ProviderConfig],
    errors: &mut Vec<String>,
) {
    let provider_names: std::collections::HashSet<&str> =
        providers.iter().map(|p| p.name.as_str()).collect();
    for (name, person) in persons {
        // Validate the person name as a MemoryKey. The router re-validates
        // per request (defence in depth against programmatic edits), but a
        // typo in the TOML key SHOULD surface at startup rather than as a
        // 400 on the first request.
        if let Err(e) = smos_domain::MemoryKey::from_raw(name) {
            errors.push(format!(
                "persons.{name:?}: invalid memory key — {e}. \
                 Person names MUST satisfy the MemoryKey rules (alphanumeric \
                 first char, no path separators, no '..')."
            ));
        }
        if !provider_names.contains(person.provider.as_str()) {
            errors.push(format!(
                "persons.{name}.provider = {:?} does not match any [[providers]].name; \
                 known providers: {{{}}}",
                person.provider,
                providers
                    .iter()
                    .map(|p| p.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if person.model.is_empty() {
            errors.push(format!("persons.{name}.model must not be empty"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Acquire the workspace-wide env-test lock. See
    /// [`crate::test_env_lock`] for why this is required.
    fn _lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_env_lock::lock()
    }

    fn one_provider() -> ProviderConfig {
        ProviderConfig::new("u", "http://u")
    }

    #[test]
    fn default_has_canonical_values() {
        let _g = _lock();
        let cfg = SmosConfig::default();
        assert_eq!(cfg.server.port, 8888);
        assert_eq!(cfg.server.host, "127.0.0.1");
        assert!(cfg.providers.is_empty());
        assert!(cfg.persons.is_empty());
        assert_eq!(cfg.surreal.namespace, "smos");
        assert_eq!(cfg.nli.contradiction_threshold, 0.5);
        assert_eq!(cfg.nli.entailment_threshold, 0.6);
        assert!(cfg.nli_backend.model.starts_with("MoritzLaurer/"));
        assert_eq!(cfg.llm_extraction.model, "nemotron-3-nano-4b");
        assert_eq!(cfg.llm_extraction.seed, 42);
        assert_eq!(cfg.embedding.dimensions, 1024);
    }

    /// The surreal + nli_backend defaults now anchor on `~/.smos` (or
    /// `SMOS_HOME` when set). Pinned so a refactor that drops the
    /// `SmosPaths::resolve` call from the default impl does not silently
    /// regress to the legacy `./data` path.
    #[test]
    fn default_paths_are_anchored_on_smos_home() {
        let _g = _lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let prior = std::env::var("SMOS_HOME").ok();
        // SAFETY: this test holds `CONFIG_TEST_LOCK`, serialising every
        // config test in this binary.
        unsafe {
            std::env::set_var("SMOS_HOME", tmp.path());
        }
        let cfg = SmosConfig::default();
        unsafe {
            match prior {
                Some(v) => std::env::set_var("SMOS_HOME", v),
                None => std::env::remove_var("SMOS_HOME"),
            }
        }
        assert!(
            cfg.surreal
                .path
                .starts_with(tmp.path().to_string_lossy().as_ref()),
            "expected surreal.path under SMOS_HOME, got {}",
            cfg.surreal.path
        );
        assert!(
            cfg.nli_backend
                .cache_dir
                .starts_with(tmp.path().to_string_lossy().as_ref()),
            "expected nli_backend.cache_dir under SMOS_HOME, got {}",
            cfg.nli_backend.cache_dir
        );
    }

    #[test]
    fn load_missing_file_falls_back_to_defaults_then_fails_validation_on_empty_providers() {
        // Defaults parse fine when the file is missing, but `load()` runs
        // `validate()` after parsing. The default config has `providers = []`,
        // which violates the "must not be empty" rule — so the operator-
        // facing result is a clear Validation error that points at the
        // missing providers rather than a silent zero-providers state that
        // would only surface at the first request.
        let _g = _lock();
        let result = SmosConfig::load("definitely-does-not-exist.toml");
        let err = result.expect_err("defaults without providers must fail validation");
        let msg = err.to_string();
        assert!(
            msg.contains("providers must not be empty"),
            "expected validation message about empty providers, got: {msg}"
        );
    }

    #[test]
    fn load_partial_file_fills_missing_sections_from_defaults() {
        let _g = _lock();
        let tmp = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("tempfile");
        // Include a provider so validation passes — the test is about
        // section-merging, not about provider semantics.
        std::fs::write(
            tmp.path(),
            "[server]\nhost = \"0.0.0.0\"\nport = 9999\n\
             [[providers]]\nname = \"u\"\nurl = \"http://u\"\ntimeout_seconds = 9\n",
        )
        .expect("write");
        let cfg = SmosConfig::load(tmp.path().to_str().unwrap()).expect("parse + validate");
        assert_eq!(cfg.server.host, "0.0.0.0");
        assert_eq!(cfg.server.port, 9999);
        assert_eq!(cfg.surreal.namespace, "smos");
    }

    #[test]
    fn load_full_file_overrides_all_sections() {
        let _g = _lock();
        // `embedding.dimensions` MUST be 1024 (HNSW index dimension) — the
        // validation gate rejects any other value at startup.
        let toml = "[surreal]\npath = \"./x.db\"\nnamespace = \"ns\"\ndatabase = \"db\"\n\
                    [server]\nhost = \"h\"\nport = 1\nshutdown_extraction_grace_seconds = 5\n\
                    enable_response_extraction = false\ngraceful_degradation = false\nlog_format = \"pretty\"\n\
                    [[providers]]\nname = \"u\"\nurl = \"u\"\napi_key_env = \"SMOS_KEY\"\nauth_header = \"api-key\"\ntimeout_seconds = 9\n\
                    [llm_extraction]\nurl = \"http://llm:28082\"\nmodel = \"qwen\"\ntimeout_seconds = 11\n\
                    temperature = 0.2\nseed = 7\n\
                    [embedding]\nurl = \"http://embed:28081\"\nmodel = \"jina\"\ndimensions = 1024\ntimeout_seconds = 11\n\
                    [reranker]\nurl = \"http://reranker:28181\"\nmodel = \"rr\"\ntimeout_seconds = 7\n\
                    [retrieval]\ntop_k_initial = 30\ntop_k_final = 3\nmin_confidence = 0.6\nmin_topic_chars = 2\n\
                    [merge]\ncosine_threshold = 0.8\n\
                    [confidence]\nbase = 0.4\nmulti_source_bonus = 0.1\nno_contradiction_bonus = 0.05\naccept_threshold = 0.65\npending_threshold = 0.3\n\
                    [heat]\ndecay_rate = 0.02\nmin_threshold = 0.15\n\
                    [nli]\ncontradiction_threshold = 0.55\nentailment_threshold = 0.65\n\
                    [nli_backend]\nmodel = \"cross-encoder/nli-deberta-v3\"\ncache_dir = \"/var/cache/smos/nli\"\n\
                    [extraction]\ndedup_cosine_threshold = 0.92\n\
                    [session]\ntimeout_seconds = 600\npending_overflow_threshold = 15\nscan_interval_seconds = 30\n";
        let tmp = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("tempfile");
        std::fs::write(tmp.path(), toml).expect("write");
        let cfg = SmosConfig::load(tmp.path().to_str().unwrap()).expect("parse + validate");
        assert_eq!(cfg.server.host, "h");
        assert_eq!(cfg.server.port, 1);
        assert!(!cfg.server.enable_response_extraction);
        assert_eq!(cfg.server.log_format, "pretty");
        assert_eq!(cfg.providers.len(), 1);
        assert_eq!(cfg.providers[0].auth_header, "api-key");
        assert_eq!(cfg.providers[0].timeout_seconds, 9);
        assert_eq!(cfg.providers[0].api_key_env, "SMOS_KEY");
        assert_eq!(cfg.surreal.path, "./x.db");
        assert_eq!(cfg.llm_extraction.url, "http://llm:28082");
        assert_eq!(cfg.llm_extraction.model, "qwen");
        assert_eq!(cfg.llm_extraction.timeout_seconds, 11);
        assert_eq!(cfg.llm_extraction.seed, 7);
        assert_eq!(cfg.llm_extraction.temperature, 0.2);
        assert_eq!(cfg.embedding.url, "http://embed:28081");
        assert_eq!(cfg.embedding.model, "jina");
        assert_eq!(cfg.embedding.dimensions, 1024);
        assert_eq!(cfg.reranker.url, "http://reranker:28181");
        assert_eq!(cfg.reranker.model, "rr");
        assert_eq!(cfg.reranker.timeout_seconds, 7);
        assert_eq!(cfg.retrieval.top_k_initial, 30);
        assert_eq!(cfg.retrieval.top_k_final, 3);
        assert_eq!(cfg.merge.cosine_threshold, 0.8);
        assert_eq!(cfg.confidence.accept_threshold, 0.65);
        assert_eq!(cfg.heat.min_threshold, 0.15);
        assert_eq!(cfg.nli.contradiction_threshold, 0.55);
        assert_eq!(cfg.nli.entailment_threshold, 0.65);
        assert_eq!(cfg.nli_backend.model, "cross-encoder/nli-deberta-v3");
        assert_eq!(cfg.nli_backend.cache_dir, "/var/cache/smos/nli");
        assert_eq!(cfg.extraction.dedup_cosine_threshold, 0.92);
        assert_eq!(cfg.session.timeout_seconds, 600);
        assert_eq!(cfg.session.pending_overflow_threshold, 15);
        assert_eq!(cfg.session.scan_interval_seconds, 30);
    }

    #[test]
    fn new_sections_default_when_omitted_from_partial_file() {
        let _g = _lock();
        let tmp = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("tempfile");
        // Add a provider so validation passes; the test verifies that the
        // sections OMITTED from the partial file fall back to defaults.
        std::fs::write(
            tmp.path(),
            "[server]\nport = 7777\n\
             [[providers]]\nname = \"u\"\nurl = \"http://u\"\ntimeout_seconds = 9\n",
        )
        .expect("write");
        let cfg = SmosConfig::load(tmp.path().to_str().unwrap()).expect("parse + validate");
        assert_eq!(cfg.server.port, 7777);
        assert_eq!(cfg.llm_extraction.timeout_seconds, 30);
        assert!(cfg.embedding.model.starts_with("hf.co/jinaai"));
        assert_eq!(cfg.reranker.model, "qwen3-reranker");
        assert_eq!(cfg.retrieval.top_k_final, 5);
        assert_eq!(cfg.session.pending_overflow_threshold, 20);
    }

    #[test]
    fn config_roundtrips_through_serde_json() {
        let _g = _lock();
        let cfg = SmosConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: SmosConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.server.port, cfg.server.port);
        assert_eq!(back.providers.len(), cfg.providers.len());
    }

    // --- Provider / person parsing -------------------------------------

    /// The canonical `[[providers]]` + `[persons.X]` shape parses into the
    /// two collections the routing layer expects.
    #[test]
    fn providers_and_persons_section_parses() {
        let _g = _lock();
        let toml = "[[providers]]\n\
                    name = \"llama-local\"\n\
                    url = \"http://localhost:28082/v1/chat/completions\"\n\
                    api_key_env = \"\"\n\
                    auth_header = \"Authorization\"\n\
                    timeout_seconds = 120\n\
                    [[providers]]\n\
                    name = \"openrouter\"\n\
                    url = \"https://openrouter.ai/api/v1/chat/completions\"\n\
                    api_key_env = \"OPENROUTER_API_KEY\"\n\
                    timeout_seconds = 90\n\
                    [persons.bob]\n\
                    provider = \"llama-local\"\n\
                    model = \"nemotron-3-nano-4b\"\n\
                    persona = \"~/.smos/persons/bob.md\"\n\
                    [persons.alice]\n\
                    provider = \"openrouter\"\n\
                    model = \"z-ai/glm-5.2\"\n";
        let tmp = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("tempfile");
        std::fs::write(tmp.path(), toml).expect("write");
        let cfg = SmosConfig::load(tmp.path().to_str().unwrap()).expect("parse + validate");
        assert_eq!(cfg.providers.len(), 2);
        assert_eq!(cfg.providers[0].name, "llama-local");
        assert_eq!(cfg.providers[1].name, "openrouter");
        assert_eq!(cfg.providers[1].api_key_env, "OPENROUTER_API_KEY");
        // Second provider inherits the default `auth_header` since the TOML
        // omits it.
        assert_eq!(cfg.providers[1].auth_header, "Authorization");

        let bob = cfg.persons.get("bob").expect("person bob");
        assert_eq!(bob.provider, "llama-local");
        assert_eq!(bob.model, "nemotron-3-nano-4b");
        assert_eq!(bob.persona, "~/.smos/persons/bob.md");

        let alice = cfg.persons.get("alice").expect("person alice");
        assert_eq!(alice.provider, "openrouter");
        assert_eq!(alice.model, "z-ai/glm-5.2");
        assert_eq!(alice.persona, "", "persona defaults to empty");
    }

    /// `ProviderConfig::resolve_api_key` reads the env var named in
    /// `api_key_env`. Empty `api_key_env` MUST yield an empty string (the
    /// "no auth" case for a local `llama-server`) instead of consulting any
    /// default env var.
    #[test]
    fn resolve_api_key_reads_named_env_var() {
        let _g = _lock();
        let prior = std::env::var("SMOS_TEST_PROVIDER_KEY").ok();
        // SAFETY: this test holds `CONFIG_TEST_LOCK`.
        unsafe {
            std::env::set_var("SMOS_TEST_PROVIDER_KEY", "sk-from-env");
        }
        let provider = ProviderConfig {
            name: "p".into(),
            url: "http://p".into(),
            api_key_env: "SMOS_TEST_PROVIDER_KEY".into(),
            auth_header: "Authorization".into(),
            timeout_seconds: 9,
        };
        assert_eq!(provider.resolve_api_key(), "sk-from-env");
        // SAFETY: same serialisation guarantee.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("SMOS_TEST_PROVIDER_KEY", v),
                None => std::env::remove_var("SMOS_TEST_PROVIDER_KEY"),
            }
        }

        // Empty api_key_env MUST short-circuit to empty string.
        let unauth = ProviderConfig {
            api_key_env: String::new(),
            ..provider
        };
        assert_eq!(unauth.resolve_api_key(), "");
    }

    // --- Legacy section guards -----------------------------------------
    //
    // These tests pin the intentional behaviour: a TOML carrying legacy
    // sections/fields still LOADS (serde has no `deny_unknown_fields`) but
    // the legacy values NEVER affect the canonical config. A future engineer
    // who re-adds a bridge will break one of these tests, which is the
    // point — the intent is documented in code, not just in commit history.

    /// A leftover unknown section (e.g. a historical `[ollama]` block from a
    /// pre-llama.cpp config) does NOT populate `[llm_extraction]` /
    /// `[embedding]`. The legacy fields are silently dropped at deserialize
    /// time and the canonical sections keep their defaults.
    #[test]
    fn legacy_unknown_section_does_not_bridge_into_canonical_sections() {
        let _g = _lock();
        let toml = "[ollama]\n\
                    url = \"http://legacy:11434\"\n\
                    embedding_model = \"legacy-embed\"\n\
                    extraction_model = \"legacy-extract\"\n\
                    timeout_seconds = 17\n\
                    [[providers]]\nname = \"u\"\nurl = \"http://u\"\ntimeout_seconds = 9\n";
        let tmp = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("tempfile");
        std::fs::write(tmp.path(), toml).expect("write");
        let cfg = SmosConfig::load(tmp.path().to_str().unwrap()).expect("parse + validate");
        // Defaults preserved — legacy fields did NOT bleed through.
        assert_eq!(cfg.llm_extraction.url, "http://localhost:28082");
        assert_eq!(cfg.llm_extraction.model, "nemotron-3-nano-4b");
        assert_eq!(cfg.llm_extraction.timeout_seconds, 30);
        assert!(cfg.embedding.model.starts_with("hf.co/jinaai"));
        assert_eq!(cfg.embedding.timeout_seconds, 30);
    }

    /// `[nli_backend]` is the CANONICAL adapter-side section (carrying
    /// `model` + `cache_dir`); the domain-side `[nli]` section now holds
    /// only verdict thresholds. This test pins the layering invariant: an
    /// operator-supplied `[nli_backend]` populates `cfg.nli_backend`, and
    /// `cfg.nli` (the domain thresholds) stays at its defaults unless the
    /// operator also overrides `[nli]`.
    #[test]
    fn nli_backend_section_is_canonical_and_does_not_touch_domain_thresholds() {
        let _g = _lock();
        let toml = "[nli_backend]\n\
                    model = \"cross-encoder/nli-deberta-v3\"\n\
                    cache_dir = \"/var/cache/smos/nli\"\n\
                    [[providers]]\nname = \"u\"\nurl = \"http://u\"\ntimeout_seconds = 9\n";
        let tmp = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("tempfile");
        std::fs::write(tmp.path(), toml).expect("write");
        let cfg = SmosConfig::load(tmp.path().to_str().unwrap()).expect("parse + validate");
        // Adapter section picked up the override.
        assert_eq!(cfg.nli_backend.model, "cross-encoder/nli-deberta-v3");
        assert_eq!(cfg.nli_backend.cache_dir, "/var/cache/smos/nli");
        // Domain thresholds stayed at their defaults — the layering
        // invariant is intact.
        assert_eq!(cfg.nli.contradiction_threshold, 0.5);
        assert_eq!(cfg.nli.entailment_threshold, 0.6);
    }

    /// Putting `model` (an adapter-only field) under `[nli]` MUST fail loudly
    /// at startup. `NliConfig` carries `#[serde(deny_unknown_fields)]` so the
    /// parser rejects the misplacement instead of silently dropping it.
    #[test]
    fn nli_section_with_adapter_field_fails_loudly() {
        let _g = _lock();
        let toml = "[nli]\n\
                    contradiction_threshold = 0.5\n\
                    entailment_threshold = 0.6\n\
                    model = \"accidental-misplacement\"\n";
        let tmp = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("tempfile");
        std::fs::write(tmp.path(), toml).expect("write");
        let result = SmosConfig::load(tmp.path().to_str().unwrap());
        assert!(
            result.is_err(),
            "operator misplacing `model` under `[nli]` must fail loudly, not silently drop"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("model") && err_msg.contains("unknown"),
            "error must identify the unknown field; got: {err_msg}"
        );
    }

    /// Symmetric loud-failure for the adapter side: an unknown field under
    /// `[nli_backend]` MUST fail loudly. `NliBackendConfig` carries the same
    /// `#[serde(deny_unknown_fields)]` so a typo (`modle = "..."`) does not
    /// silently fall back to the default model.
    #[test]
    fn nli_backend_section_with_unknown_field_fails_loudly() {
        let _g = _lock();
        let toml = "[nli_backend]\n\
                    modle = \"typo-for-model\"\n\
                    cache_dir = \"./data/nli_cache\"\n";
        let tmp = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("tempfile");
        std::fs::write(tmp.path(), toml).expect("write");
        let result = SmosConfig::load(tmp.path().to_str().unwrap());
        assert!(
            result.is_err(),
            "typo in `[nli_backend]` must fail loudly, not silently fall back to defaults"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("modle") && err_msg.contains("unknown"),
            "error must identify the unknown field; got: {err_msg}"
        );
    }

    /// A leftover `[nli_sidecar]` section (Python sidecar, removed) does
    /// NOT abort startup and does NOT populate any field. Pinned so a
    /// future change that re-introduces sidecar parsing breaks this test.
    #[test]
    fn legacy_nli_sidecar_section_is_silently_ignored() {
        let _g = _lock();
        let toml = "[nli_sidecar]\n\
                    python = \"python\"\n\
                    script = \"x.py\"\n\
                    cache_dir = \"./legacy\"\n\
                    [[providers]]\nname = \"u\"\nurl = \"http://u\"\ntimeout_seconds = 9\n";
        let tmp = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("tempfile");
        std::fs::write(tmp.path(), toml).expect("write");
        let cfg = SmosConfig::load(tmp.path().to_str().unwrap()).expect("parse + validate");
        // The default cache_dir is anchored on SMOS_HOME; the legacy value
        // `./legacy` did NOT bleed through.
        assert_ne!(cfg.nli_backend.cache_dir, "./legacy");
    }

    /// The legacy `[[upstream.providers]]` array (now replaced by
    /// `[[providers]]`) does NOT populate `cfg.providers`. The fields are
    /// silently dropped at deserialize time and `cfg.providers` stays empty,
    /// which the validator flags with the canonical "providers must not be
    /// empty" error.
    #[test]
    fn legacy_upstream_providers_section_does_not_bridge_into_providers() {
        let _g = _lock();
        let toml = "[[upstream.providers]]\n\
                    name = \"legacy\"\n\
                    url = \"http://legacy\"\n\
                    api_key = \"legacy\"\n";
        let tmp = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("tempfile");
        std::fs::write(tmp.path(), toml).expect("write");
        let result = SmosConfig::load(tmp.path().to_str().unwrap());
        let err = result.expect_err("legacy [[upstream.providers]] must NOT bridge");
        let msg = err.to_string();
        assert!(
            msg.contains("providers must not be empty"),
            "expected validation to flag empty providers (proof that no bridge \
             happened); got: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // validate() — range / consistency checks
    // -----------------------------------------------------------------------

    #[test]
    fn validate_accepts_default_plus_one_provider() {
        // The minimum config that should pass validation: defaults + one
        // provider. Anchors the lower bound of what `smos serve` accepts.
        let mut cfg = SmosConfig::default();
        cfg.providers.push(one_provider());
        assert!(cfg.validate().is_ok(), "default + 1 provider must validate");
    }

    #[test]
    fn validate_rejects_wrong_embedding_dimensions() {
        let mut cfg = SmosConfig::default();
        cfg.embedding.dimensions = 512;
        cfg.providers.push(one_provider());
        let err = cfg.validate().expect_err("dimensions != 1024 must fail");
        let msg = err.to_string();
        assert!(msg.contains("embedding.dimensions"), "msg = {msg}");
        assert!(msg.contains("1024"), "msg = {msg}");
    }

    #[test]
    fn validate_rejects_confidence_out_of_range() {
        let mut cfg = SmosConfig::default();
        cfg.confidence.base = 1.5;
        cfg.providers.push(one_provider());
        let err = cfg.validate().expect_err("base > 1 must fail");
        assert!(err.to_string().contains("confidence.base"));
    }

    #[test]
    fn validate_rejects_accept_below_pending_threshold() {
        let mut cfg = SmosConfig::default();
        cfg.confidence.accept_threshold = 0.3;
        cfg.confidence.pending_threshold = 0.5;
        cfg.providers.push(one_provider());
        let err = cfg.validate().expect_err("accept < pending must fail");
        let msg = err.to_string();
        assert!(msg.contains("accept_threshold"), "msg = {msg}");
        assert!(msg.contains("pending_threshold"), "msg = {msg}");
    }

    #[test]
    fn validate_rejects_empty_providers() {
        let cfg = SmosConfig::default();
        let err = cfg.validate().expect_err("no providers must fail");
        assert!(
            err.to_string().contains("providers must not be empty"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_rejects_provider_with_empty_url() {
        let mut cfg = SmosConfig::default();
        let mut p = ProviderConfig::new("u", "");
        p.timeout_seconds = 9;
        cfg.providers.push(p);
        let err = cfg.validate().expect_err("empty url must fail");
        assert!(err.to_string().contains("url must not be empty"));
    }

    #[test]
    fn validate_rejects_provider_with_zero_timeout() {
        let mut cfg = SmosConfig::default();
        let mut p = ProviderConfig::new("u", "http://u");
        p.timeout_seconds = 0;
        cfg.providers.push(p);
        let err = cfg.validate().expect_err("zero timeout must fail");
        assert!(err.to_string().contains("timeout_seconds must be > 0"));
    }

    #[test]
    fn validate_rejects_provider_with_empty_name() {
        let mut cfg = SmosConfig::default();
        let mut p = ProviderConfig::new("", "http://u");
        p.timeout_seconds = 9;
        cfg.providers.push(p);
        let err = cfg.validate().expect_err("empty name must fail");
        assert!(err.to_string().contains("name must not be empty"));
    }

    #[test]
    fn validate_rejects_duplicate_provider_names() {
        let mut cfg = SmosConfig::default();
        cfg.providers.push(ProviderConfig::new("dup", "http://a"));
        cfg.providers.push(ProviderConfig::new("dup", "http://b"));
        let err = cfg.validate().expect_err("duplicate name must fail");
        let msg = err.to_string();
        assert!(msg.contains("duplicated"), "msg = {msg}");
        assert!(msg.contains("dup"), "msg = {msg}");
    }

    #[test]
    fn validate_rejects_person_referencing_unknown_provider() {
        let mut cfg = SmosConfig::default();
        cfg.providers.push(ProviderConfig::new("known", "http://a"));
        let mut persons = HashMap::new();
        persons.insert(
            "bob".into(),
            PersonConfig {
                provider: "typo".into(),
                model: "nemotron-3-nano-4b".into(),
                persona: String::new(),
            },
        );
        cfg.persons = persons;
        let err = cfg.validate().expect_err("unknown provider must fail");
        let msg = err.to_string();
        assert!(msg.contains("persons.bob.provider"), "msg = {msg}");
        assert!(msg.contains("typo"), "msg = {msg}");
    }

    #[test]
    fn validate_rejects_person_with_empty_model() {
        let mut cfg = SmosConfig::default();
        cfg.providers.push(ProviderConfig::new("p", "http://a"));
        let mut persons = HashMap::new();
        persons.insert(
            "bob".into(),
            PersonConfig {
                provider: "p".into(),
                model: String::new(),
                persona: String::new(),
            },
        );
        cfg.persons = persons;
        let err = cfg.validate().expect_err("empty model must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("persons.bob.model must not be empty"),
            "msg = {msg}"
        );
    }

    #[test]
    fn validate_rejects_person_with_unsafe_name_at_startup() {
        // A person name with path-traversal characters fails MemoryKey
        // validation. The startup validator MUST catch this so the
        // operator hears about it before the first request hits the
        // routing layer.
        let mut cfg = SmosConfig::default();
        cfg.providers.push(ProviderConfig::new("p", "http://a"));
        let mut persons = HashMap::new();
        persons.insert(
            "a/b".into(),
            PersonConfig {
                provider: "p".into(),
                model: "nemotron-3-nano-4b".into(),
                persona: String::new(),
            },
        );
        cfg.persons = persons;
        let err = cfg
            .validate()
            .expect_err("unsafe person name must fail at startup");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid memory key"),
            "expected 'invalid memory key' in msg, got: {msg}"
        );
        assert!(msg.contains("a/b"), "expected person name in msg: {msg}");
    }

    #[test]
    fn validate_accepts_person_referencing_known_provider() {
        let mut cfg = SmosConfig::default();
        cfg.providers.push(ProviderConfig::new("p", "http://a"));
        let mut persons = HashMap::new();
        persons.insert(
            "bob".into(),
            PersonConfig {
                provider: "p".into(),
                model: "nemotron-3-nano-4b".into(),
                persona: String::new(),
            },
        );
        cfg.persons = persons;
        assert!(cfg.validate().is_ok(), "valid person must validate");
    }

    #[test]
    fn validate_rejects_empty_reranker_url() {
        // The reranker is a hard dependency — an operator who blanks the URL
        // must get a startup error pointing at the field instead of
        // discovering the dependency via an HTTP 503 on the first request.
        let mut cfg = SmosConfig::default();
        cfg.reranker.url = String::new();
        cfg.providers.push(one_provider());
        let err = cfg.validate().expect_err("empty reranker url must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("reranker.url must not be empty"),
            "msg = {msg}"
        );
    }

    #[test]
    fn validate_rejects_whitespace_only_reranker_url() {
        // `trim().is_empty()` catches whitespace-only strings so a typo like
        // `url = "   "` is treated identically to an empty string.
        let mut cfg = SmosConfig::default();
        cfg.reranker.url = "   ".into();
        cfg.providers.push(one_provider());
        let err = cfg
            .validate()
            .expect_err("whitespace-only reranker url must fail");
        assert!(err.to_string().contains("reranker.url must not be empty"));
    }

    #[test]
    fn validate_collects_multiple_errors_in_one_message() {
        // Two unrelated problems: bad dimensions AND no providers. The
        // operator should see both in a single error so they can fix them
        // in one editing pass.
        let mut cfg = SmosConfig::default();
        cfg.embedding.dimensions = 768;
        // providers stays empty
        let err = cfg.validate().expect_err("multi-error case");
        let msg = err.to_string();
        assert!(msg.contains("embedding.dimensions"), "msg = {msg}");
        assert!(msg.contains("providers must not be empty"), "msg = {msg}");
        assert!(
            msg.contains(";"),
            "multiple errors joined by ';' in msg = {msg}"
        );
    }

    // --- AuditConfig behaviour -------------------------------------------

    #[test]
    fn audit_section_disabled_by_default() {
        let _g = _lock();
        let cfg = SmosConfig::default();
        assert!(!cfg.audit.enabled, "audit must be off by default");
        assert_eq!(cfg.audit.schedule, "0 3 * * *");
        assert_eq!(cfg.audit.llm_provider, "cloud");
        assert_eq!(cfg.audit.max_deletions_per_run, 50);
        assert_eq!(cfg.audit.max_merges_per_run, 100);
    }

    #[test]
    fn audit_validation_skipped_when_disabled() {
        // Audit off => bad provider string does NOT fail validation. The
        // audit is opt-in; a stale `llm_provider` typo in a deployment that
        // never enables the audit should not block server startup.
        let mut cfg = SmosConfig::default();
        cfg.audit.enabled = false;
        cfg.audit.llm_provider = "garbage".into();
        cfg.providers.push(one_provider());
        assert!(cfg.validate().is_ok(), "disabled audit must not validate");
    }

    #[test]
    fn audit_validation_rejects_unknown_provider_when_enabled() {
        let mut cfg = SmosConfig::default();
        cfg.audit.enabled = true;
        cfg.audit.llm_provider = "garbage".into();
        cfg.providers.push(one_provider());
        let err = cfg.validate().expect_err("bad provider must fail");
        assert!(err.to_string().contains("audit.llm_provider"));
    }

    #[test]
    fn audit_validation_rejects_empty_schedule_when_enabled() {
        let mut cfg = SmosConfig::default();
        cfg.audit.enabled = true;
        cfg.audit.schedule = "   ".into();
        cfg.providers.push(one_provider());
        let err = cfg.validate().expect_err("empty schedule must fail");
        assert!(err.to_string().contains("audit.schedule"));
    }

    #[test]
    fn audit_section_roundtrips_through_serde_json() {
        let cfg = SmosConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: SmosConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.audit.schedule, cfg.audit.schedule);
        assert_eq!(back.audit.cloud_model, cfg.audit.cloud_model);
        assert_eq!(
            back.audit.max_deletions_per_run,
            cfg.audit.max_deletions_per_run
        );
    }
}
