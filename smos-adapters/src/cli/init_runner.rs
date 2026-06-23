//! `smos init` — first-time setup runner.
//!
//! Materialises `~/.smos` (or `$SMOS_HOME`) with every well-known
//! subdirectory and drops a default `config.toml` at its root when one is
//! not already present. Re-running `smos init` is idempotent: existing
//! directories are kept, an existing `config.toml` is NEVER overwritten so
//! the operator's edits survive a re-init.

use anyhow::Result;

use crate::paths::{SmosPaths, ensure_smos_home};

/// The canonical default `config.toml` written by [`run_init`] when no
/// config exists yet.
///
/// A verbatim inline copy of the workspace-root `smos.toml` so the binary
/// stays self-contained: a single-file distribution can drop the `smos`
/// binary on a fresh box and `smos init` produces a working config without
/// any extra assets. Inlining (rather than `include_str!` of a file outside
/// the crate) keeps `cargo publish` working — an out-of-crate path is not
/// packaged into the crate tarball.
///
/// KEEP IN SYNC with `smos.toml` at the workspace root: that file is the
/// active config when running `smos serve` from the repo root, while this
/// literal is what installed binaries ship. Drift between the two silently
/// gives developers and end users different defaults.
pub const DEFAULT_CONFIG_TOML: &str = r#"# SMOS proxy default configuration.
# Copy/edit next to the binary (or pass via a custom path) and run:
#   cargo run --bin smos -- serve
# Any section omitted here falls back to the built-in defaults in
# smos-adapters/src/config.rs.
#
# This same file is installed verbatim under ~/.smos/config.toml by
# `smos init` — re-running `smos init` is idempotent and NEVER overwrites an
# existing config.

[surreal]
# Defaults to ~/.smos/db/smos.db (resolved by SmosPaths at startup). Override
# only when you need to point at a different RocksDB path.
# path = "~/.smos/db/smos.db"
namespace = "smos"
database = "smos"

[server]
host = "127.0.0.1"
port = 8888
shutdown_extraction_grace_seconds = 30
enable_response_extraction = true
graceful_degradation = true
log_format = "json"

# ---------------------------------------------------------------------------
# Providers — OpenAI-compatible LLM chat-completion endpoints.
# ---------------------------------------------------------------------------
#
# Each entry is one upstream (Ollama, OpenRouter, OpenAI, vLLM, …). The proxy
# forwards every chat-completion request to exactly one provider, chosen by
# the [persons.*] map. There is no round-robin / failover any more: routing
# is per-person, not per-pool.

[[providers]]
name = "ollama-local"
url = "http://localhost:11434/v1/chat/completions"
api_key_env = ""                    # env var name; empty = no auth header sent
# auth_header = "Authorization"     # default; set to "api-key" for Azure
# timeout_seconds = 120             # default

# Example second provider. Uncomment and set OPENROUTER_API_KEY in the
# environment to route `[persons.alice]`-style entries through OpenRouter.
# [[providers]]
# name = "openrouter"
# url = "https://openrouter.ai/api/v1/chat/completions"
# api_key_env = "OPENROUTER_API_KEY"

# ---------------------------------------------------------------------------
# Persons — LLM personas = memory keys.
# ---------------------------------------------------------------------------
#
# A person is simultaneously:
#   - a memory key (the namespace under which extracted facts land),
#   - a provider + upstream model (the routing target),
#   - an optional persona .md (prepended to the request as a system message).
#
# Clients send `{"model": "<person-name>", ...}`; the proxy rewrites
# `request.model` to the upstream model declared below before forwarding.

[persons.bob]
provider = "ollama-local"           # MUST match a [[providers]].name
model = "granite4.1:3b"             # upstream model id
persona = "~/.smos/persons/bob.md"  # optional; `~` expands to user home

# Example second person.
# [persons.alice]
# provider = "openrouter"
# model = "z-ai/glm-5.2"
# persona = "~/.smos/persons/alice.md"

# ---------------------------------------------------------------------------
# LLM for fact extraction (Qwen3.5:2b or any Ollama-/OpenAI-compatible model).
# ---------------------------------------------------------------------------
[llm_extraction]
url = "http://localhost:11434"           # API base (the adapter appends /api/chat)
model = "qwen3.5:2b"
api_key = ""                              # optional, for cloud providers
timeout_seconds = 30
# Deterministic extraction: temperature=0 disables random sampling, seed pins
# the RNG so two runs against the same input re-yield the same bullet list.
# This is what keeps FactId = SHA1(content) stable across re-extraction runs.
temperature = 0.0
seed = 42

# Embedding model for vector search (may be a different provider than the
# extraction LLM).
[embedding]
url = "http://localhost:11434"           # the adapter appends /api/embeddings
model = "hf.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:latest"
dimensions = 1024                          # MUST match the HNSW index DDL
api_key = ""
timeout_seconds = 30

# Reranker — required for enrichment.
# The enrich pipeline reranks vector-search survivors via a cross-encoder
# before injecting them into the request. Without a reachable reranker,
# enrichment returns HTTP 503 (no fallback to vector-order-only ranking).
# Start the llama.cpp reranker with a Qwen3-Reranker GGUF model:
#   ./llama-server --model qwen3-reranker-0.6b-q8_0.gguf --port 8181
# The adapter expects an OpenAI-compatible /v1/rerank endpoint.
[reranker]
url = "http://localhost:8181"
model = "qwen3-reranker"
timeout_seconds = 60

[retrieval]
top_k_initial = 50
top_k_final = 5
min_confidence = 0.7
min_topic_chars = 3

[merge]
cosine_threshold = 0.85

[confidence]
base = 0.5
multi_source_bonus = 0.2
no_contradiction_bonus = 0.1
accept_threshold = 0.7
pending_threshold = 0.4

# NLI verdict thresholds (domain layer). Drives the `is_contradiction` /
# `is_entailment` / `decide_merge` predicates. Soft caps so a borderline
# softmax distribution does not flip a verdict.
[nli]
contradiction_threshold = 0.5
entailment_threshold = 0.6

# Native ort + ONNX Runtime backend for NLI inference — adapter-only sibling
# of `[nli]`. The model id and cache directory are interpreter-level data
# that the domain layer never reads; keeping them under their own section
# preserves the layering invariant ("domain types carry no IO-boundary
# data"). Putting `model` / `cache_dir` under `[nli]` is a loud startup
# error — both sections carry `#[serde(deny_unknown_fields)]`, so the
# parser rejects the misplacement instead of silently dropping it.
[nli_backend]
model = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# Defaults to ~/.smos/models; uncomment to override.
# cache_dir = "~/.smos/models"

# Semantic dedup safety net for fact extraction (persist_facts step 2).
# When the extractor rephrases a fact just enough to break the FactId =
# SHA1(content) exact match, the semantic layer catches it via cosine
# similarity and routes it through cross-session confirmation instead of
# leaving the fact stuck at single-source confidence.
[extraction]
dedup_cosine_threshold = 0.95

[heat]
decay_rate = 0.03
min_threshold = 0.2

[session]
timeout_seconds = 1800
pending_overflow_threshold = 20
scan_interval_seconds = 60

# SMOS Dreaming Agent — autonomous LLM-driven memory audit. Disabled by
# default; flip `enabled = true` to opt in. The agent runs on a cron schedule,
# reviews stored facts via rig tool-calling, applies bounded mutations
# (deletions, merges, conflict flags), and writes a markdown report.
[audit]
enabled = false
schedule = "0 3 * * *"
llm_provider = "cloud"
cloud_model = "z-ai/glm-4.6"
# Set this to "${OPENROUTER_API_KEY}" to keep the secret out of TOML; the
# dreaming module expands the placeholder via std::env::var at runtime.
cloud_api_key = ""
cloud_base_url = "https://openrouter.ai/api/v1"
local_model = "granite4.1:3b"
local_url = "http://localhost:11434"
max_deletions_per_run = 50
max_merges_per_run = 100
# rig 0.14's PromptRequest defaults to single-turn (max_depth = 0); without a
# non-zero multi-turn depth the tool-calling loop never engages and the audit
# fails with `MaxDepthError: (reached limit: 0)` on the first prompt that needs
# a tool. 10 is the canonical headroom for a full audit sweep.
max_tool_rounds = 10
# Defaults to ~/.smos/reports; uncomment to override.
# report_dir = "~/.smos/reports"

# ---------------------------------------------------------------------------
# llama.cpp auto-launch.
# ---------------------------------------------------------------------------
#
# When `auto_launch = true`, `smos serve` spawns the configured
# `llama-server` processes at startup so the operator does not have to start
# them by hand. Each service's port is probed first; an already-running
# server is reused. Disable if you launch llama-server yourself or use a
# remote / cloud provider.
[llama_cpp]
binary = "llama-server"
auto_launch = false

[llama_cpp.embedding]
model_path = "~/.smos/models/jina-embeddings-v5.gguf"
port = 8081
extra_args = ["--ctx-size", "2048"]

[llama_cpp.reranker]
model_path = "~/.smos/models/qwen3-reranker.gguf"
port = 8181
extra_args = ["--ctx-size", "8192"]

[llama_cpp.extraction]
model_path = "~/.smos/models/qwen3.5-2b.gguf"
port = 8082
extra_args = ["--ctx-size", "4096"]

# ---------------------------------------------------------------------------
# Git-backed memory sync.
# ---------------------------------------------------------------------------
#
# When `repo_url` is non-empty, SMOS dual-writes extracted facts to a local
# clone of the repo as markdown files (one fact per .md) and commits + pushes
# after every FinalizeSession. The layout is read back by
# `smos import git <url>` so two SMOS instances can share memory through git.
#
# Empty `repo_url` disables sync; the local_path still works for offline use.
[git]
repo_url = ""
branch = "main"
auto_push = false
local_path = "~/.smos/git/memory"
disable_gpg_sign = true            # SMOS commits are unsigned by default
"#;

/// Entry point invoked by the unified `smos` binary's `Init` subcommand.
///
/// Steps:
/// 1. [`ensure_smos_home`] creates every well-known subdirectory.
/// 2. If `~/.smos/config.toml` does NOT exist, write [`DEFAULT_CONFIG_TOML`]
///    to it. An existing config is left untouched (logged).
/// 3. Drop a stub `persons/bob.md` when one is not already present so the
///    operator has a concrete starting point (the default config references
///    this file via `[persons.bob].persona`). Existing files are kept.
/// 4. Print the resolved paths + a pointer to the persona file so the
///    operator knows where to drop persona `.md` content.
pub fn run_init() -> Result<()> {
    let home = ensure_smos_home()?;
    let paths = SmosPaths::resolve();
    let config_path = &paths.config;

    if config_path.exists() {
        tracing::info!(
            config_path = %config_path.display(),
            "config already exists; left untouched"
        );
        println!("Config already exists at {}", config_path.display());
    } else {
        std::fs::write(config_path, DEFAULT_CONFIG_TOML)?;
        tracing::info!(
            config_path = %config_path.display(),
            "wrote default config"
        );
        println!("Created default config at {}", config_path.display());
    }

    // Drop a stub persona file so the default `[persons.bob]` entry has a
    // working target. The operator is expected to edit it; we never
    // overwrite an existing file (the operator may have crafted one
    // already).
    let bob_persona = paths.persons.join("bob.md");
    if !bob_persona.exists() {
        std::fs::write(&bob_persona, DEFAULT_PERSONA_BOB_MD)?;
        println!("Created stub persona at {}", bob_persona.display());
        println!(
            "Edit {} to customise the bob persona (this is the system prompt \
             injected on every chat-completion request that names model \"bob\").",
            bob_persona.display()
        );
    }

    println!("SMOS home: {}", home.display());
    println!(
        "Directory structure created under {} (db/, models/, persons/, logs/, reports/).",
        home.display()
    );
    Ok(())
}

/// Minimal stub persona shipped with the default `[persons.bob]` entry.
///
/// Kept inline (not loaded from a file) so the binary stays self-contained.
/// The operator is expected to replace it with the real persona content
/// after `smos init`.
const DEFAULT_PERSONA_BOB_MD: &str = "\
# Bob persona\n\
\n\
Replace this file with the system prompt you want injected as the leading\n\
`system` message for every chat-completion request that names `model: \"bob\"`\n\
in its body. The content is forwarded verbatim to the upstream provider\n\
declared in `[persons.bob].provider`.\n\
";

/// Resolve the canonical config path WITHOUT writing anything. Used by
/// the unified `smos` binary so `--config <path>` overrides take effect
/// for `smos serve` regardless of whether `smos init` was ever run.
///
/// Exposed here (rather than inlined at the call site) so the priority
/// chain — CLI override > `./smos.toml` > `~/.smos/config.toml` — stays
/// documented in one place.
pub fn resolve_effective_config_path(cli_override: Option<&str>) -> std::path::PathBuf {
    crate::paths::resolve_config_path(cli_override)
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Acquire the workspace-wide env-test lock. See
    /// [`crate::test_env_lock`] for why this is required.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_env_lock::lock()
    }
    /// `run_init` against a fresh `SMOS_HOME` tempdir creates the
    /// directory tree AND drops the default config. Re-running is
    /// idempotent: the second invocation must NOT overwrite the file the
    /// operator may have edited.
    #[test]
    fn run_init_is_idempotent_and_writes_default_config_once() {
        let _g = lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let prior = std::env::var("SMOS_HOME").ok();
        // SAFETY: `INIT_TEST_LOCK` is held for the duration of the env
        // mutation + read, and the prior value is restored before return
        // so other tests in the binary see the original state.
        unsafe {
            std::env::set_var("SMOS_HOME", tmp.path());
        }

        run_init().expect("first init");
        let config_path = tmp.path().join("config.toml");
        assert!(config_path.is_file(), "config must exist after first init");
        let first_content = std::fs::read_to_string(&config_path).expect("read first config");

        // Simulate an operator edit: overwrite the config with custom
        // content. The second init MUST preserve it.
        std::fs::write(&config_path, "# operator-edited config\n").expect("write edit");
        run_init().expect("second init");
        let second_content = std::fs::read_to_string(&config_path).expect("read second config");
        assert_eq!(
            second_content, "# operator-edited config\n",
            "second init must NOT overwrite an existing config"
        );
        assert_ne!(
            second_content, first_content,
            "sanity: the two reads must differ"
        );

        unsafe {
            match prior {
                Some(v) => std::env::set_var("SMOS_HOME", v),
                None => std::env::remove_var("SMOS_HOME"),
            }
        }
    }

    /// `run_init` drops a stub `persons/bob.md` so the default
    /// `[persons.bob].persona` reference resolves to a working file. The
    /// second invocation must NOT overwrite an operator-edited persona.
    #[test]
    fn run_init_drops_stub_bob_persona_idempotently() {
        let _g = lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let prior = std::env::var("SMOS_HOME").ok();
        // SAFETY: same lock-protected guarantee.
        unsafe {
            std::env::set_var("SMOS_HOME", tmp.path());
        }

        run_init().expect("first init");
        let bob_path = tmp.path().join("persons").join("bob.md");
        assert!(
            bob_path.is_file(),
            "stub persona must exist after first init"
        );

        // Edit the persona; second init MUST preserve the edit.
        std::fs::write(&bob_path, "# operator persona\n").expect("edit");
        run_init().expect("second init");
        let content = std::fs::read_to_string(&bob_path).expect("read");
        assert_eq!(
            content, "# operator persona\n",
            "second init must NOT overwrite an existing persona file"
        );

        unsafe {
            match prior {
                Some(v) => std::env::set_var("SMOS_HOME", v),
                None => std::env::remove_var("SMOS_HOME"),
            }
        }
    }
    /// `DEFAULT_CONFIG_TOML` is the inline default config shipped in the
    /// binary. Pinning that it parses back into a valid
    /// [`crate::config::SmosConfig`] catches a typo in either the literal
    /// or the parser before it ships.
    #[test]
    fn default_config_toml_parses_into_valid_smos_config() {
        let _g = lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let prior = std::env::var("SMOS_HOME").ok();
        // SAFETY: same serialisation guarantee.
        unsafe {
            std::env::set_var("SMOS_HOME", tmp.path());
        }
        let cfg = crate::config::SmosConfig::load_from_str(DEFAULT_CONFIG_TOML)
            .expect("default toml must parse + validate");
        unsafe {
            match prior {
                Some(v) => std::env::set_var("SMOS_HOME", v),
                None => std::env::remove_var("SMOS_HOME"),
            }
        }
        assert!(!cfg.providers.is_empty(), "default must ship >= 1 provider");
        assert!(!cfg.persons.is_empty(), "default must ship >= 1 person");
    }

    /// `resolve_effective_config_path` mirrors the documented priority chain.
    #[test]
    fn resolve_effective_config_path_prefers_cli_override() {
        let p = resolve_effective_config_path(Some("/explicit/path.toml"));
        assert_eq!(p, std::path::PathBuf::from("/explicit/path.toml"));
    }

    /// `smos_home` is the single source of truth for the home directory.
    /// Pinned so a refactor that drops the env-var override breaks the
    /// test rather than silently changing the resolution order.
    #[test]
    fn smos_home_is_exported_and_callable() {
        let _ = crate::paths::smos_home();
    }
}
