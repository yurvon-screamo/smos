//! `smos-adapters` — concrete implementations of the application-layer ports.
//!
//! This crate is the *only* place IO happens in the SMOS Rust port. Each
//! adapter implements a `smos_application::ports` trait against a specific
//! external system (SurrealDB for persistence, system clock for time, HTTP
//! for LLM upstream, Ollama for embeddings/rerank, ort + ONNX Runtime for
//! NLI).
//!
//! See `smos-poc/ТРЕБОВАНИЯ.md` for the canonical specification and
//! `smos-application` for the port shapes.

pub mod cli;
pub mod config;
pub mod doctor;
pub mod dreaming;
pub mod http;
pub mod nli;
pub mod opencode;
pub mod providers;
pub mod runtime;
pub mod storage;
pub mod upstream;

pub use config::{
    AuditConfig, EmbeddingConfig, LlmExtractionConfig, NliBackendConfig, RerankerConfig,
    ServerConfig, SessionConfig, SmosConfig, SurrealConfig, UpstreamConfig, UpstreamProvider,
    UpstreamStrategy,
};
pub use nli::NativeNliClassifier;
pub use opencode::{DiscoveryError, SessionSource};
pub use providers::{LlamaCppReranker, NoopExtractor, OllamaEmbedding, OllamaExtractor};
pub use runtime::SessionWatcher;
pub use runtime::TokioDelay;
pub use storage::surreal_store::SurrealStore;
pub use storage::system_clock::SystemClock;
pub use storage::system_id_generator::SystemIdGenerator;
pub use upstream::reqwest_upstream::{ReqwestUpstream, ReqwestUpstreamPool};
