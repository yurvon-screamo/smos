//! `smos` — concrete implementations of the application-layer ports.
//!
//! This crate is the *only* place IO happens in the SMOS Rust port. Each
//! adapter implements a `smos_application::ports` trait against a specific
//! external system (SurrealDB for persistence, system clock for time, HTTP
//! for LLM upstream, OpenAI-compatible `llama-server` endpoints for
//! embeddings/extraction/rerank, ort + ONNX Runtime for NLI).
//!
//! See `smos-poc/ТРЕБОВАНИЯ.md` for the canonical specification and
//! `smos-application` for the port shapes.

pub mod cli;
pub mod config;
pub mod doctor;
pub mod dreaming;
pub mod git_sync;
pub mod http;
pub mod llama_server;
pub mod nli;
pub mod opencode;
pub mod paths;
pub mod providers;
pub mod runtime;
pub mod storage;
pub mod upstream;

#[cfg(test)]
mod paths_tests;

pub use config::{
    AuditConfig, EmbeddingConfig, GitConfig, LlmExtractionConfig, NliBackendConfig, PersonConfig,
    ProviderConfig, RerankerConfig, ServerConfig, SessionConfig, SmosConfig, SurrealConfig,
};
pub use llama_server::{LlamaCppConfig, LlamaCppManager};
pub use nli::NativeNliClassifier;
pub use opencode::{DiscoveryError, SessionSource};
pub use paths::{
    SmosPaths, ensure_smos_home, expand_tilde, resolve_config_path, smos_home, user_home_dir,
};
pub use providers::{LlamaCppReranker, NoopExtractor, OllamaEmbedding, OllamaExtractor};
pub use runtime::TokioDelay;
pub use runtime::{SessionWatcher, WatcherConfig, WatcherDeps};
pub use storage::surreal_store::SurrealStore;
pub use storage::system_clock::SystemClock;
pub use storage::system_id_generator::SystemIdGenerator;
pub use upstream::reqwest_upstream::{ReqwestUpstream, ReqwestUpstreamRouter};

/// Workspace-wide lock that serialises every env-var-mutating unit test in
/// this binary (paths, config, init_runner). Process-global env vars race
/// when tests run in parallel; acquiring this lock at the top of every
/// env-touching test (and holding it until the prior value is restored)
/// serialises them so the rest of the suite observes a stable env.
///
/// Hidden behind `#[cfg(test)]` so production builds do not carry the
/// dependency, and `#[doc(hidden)]` so it does not show up in generated
/// docs — it is a test-only contract.
#[cfg(test)]
#[doc(hidden)]
pub mod test_env_lock {
    use std::sync::{Mutex, MutexGuard};

    static WORKSPACE_ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Acquire the workspace-wide env-test lock. The returned guard MUST
    /// be held for the lifetime of every `unsafe { std::env::set_var() }`
    /// block so a parallel test in the same binary cannot observe the
    /// mutation mid-flight.
    pub fn lock() -> MutexGuard<'static, ()> {
        WORKSPACE_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }
}
