use serde::{Deserialize, Serialize};
pub use smos_domain::config::{
    ConfidenceConfig, ExtractionConfig, HeatConfig, MergeConfig, NliConfig, RetrievalConfig,
};

use super::defaults::{default_auth_header, default_provider_timeout};

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
/// The domain layer never interprets `model`, `cache_dir`, `device` or
/// `ort_cache_dir`; they are read exactly once at startup by
/// [`crate::nli::build_classifier`] and passed to the ort session build.
/// Keeping them in this adapter-side struct (rather than the domain
/// `NliConfig`) preserves the "domain carries no IO-boundary data"
/// invariant.
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
    /// Device selection policy: `"auto"` (default) probes the host at
    /// startup; `"cpu"`, `"directml"`, `"cuda"`, `"metal"` force a
    /// specific device. See [`crate::nli::device::detect_device`] for
    /// the resolution rules.
    pub device: String,
    /// Local directory used to cache the dynamically-downloaded ONNX
    /// Runtime shared library (one subdirectory per device — `cpu/`,
    /// `cuda/`, `directml/`, `metal/`). See [`crate::nli::ort_cache`].
    pub ort_cache_dir: String,
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
