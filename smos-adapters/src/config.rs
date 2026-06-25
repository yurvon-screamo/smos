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
//! |---------------------|----------------------------|--------------------------------|
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
//! | `[llama_cpp]`       | `LlamaCppConfig`           | Auto-launch `llama-server`     |
//! |                     |                            | processes on `smos serve`.     |
//! | `[git]`             | [`GitConfig`]              | Git-backed memory sync.        |
//!
//! # Module layout
//!
//! Historically a single 1710-line `config.rs`; split (R5) into focused
//! submodules without changing the public API. [`types`] carries the struct +
//! enum definitions, [`defaults`] the `Default` impls and serde-default
//! helpers, [`validate`] the cross-field invariants, [`loader`] the layered
//! TOML+env loading. Everything remains reachable at `crate::config::*`
//! through the `pub use types::*` re-export below — external callers see no
//! difference.

pub mod defaults;
pub mod loader;
pub mod types;
pub mod validate;

#[cfg(test)]
mod tests;

pub use types::*;
