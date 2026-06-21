//! Port traits — abstract capabilities the application depends on.
//!
//! Each trait is a *port* in the hexagonal sense: it captures one external
//! capability (persistence, embedding, NLI, clock, …) without committing to a
//! concrete implementation. Native `async fn` in trait is used (no
//! `#[async_trait]`); dispatch is generic over `T: Trait`.
//!
//! # On `async_fn_in_trait`
//!
//! Rust's `async_fn_in_trait` lint warns that the returned `Future` of an
//! `async fn` in a trait has no explicit `Send` bound. We allow it here on
//! purpose: we want *call sites* to choose whether they need `Send` (most do,
//! via tokio) instead of baking it into the trait surface. Concrete adapters
//! in `smos-adapters` are written to return `Send` futures, and use cases
//! that spawn tasks require `T: Trait + Send + Sync + 'static` at the call
//! site, which propagates the `Send` requirement through the bound.

pub mod clock;
pub mod delay;
pub mod embedding_provider;
pub mod fact_repository;
pub mod id_generator;
pub mod llm_extractor;
pub mod llm_upstream;
pub mod nli_classifier;
pub mod rerank_provider;
pub mod session_repository;

pub use clock::Clock;
pub use delay::Delay;
pub use embedding_provider::EmbeddingProvider;
pub use fact_repository::FactRepository;
pub use id_generator::IdGenerator;
pub use llm_extractor::LlmExtractor;
pub use llm_upstream::LlmUpstream;
pub use nli_classifier::NliClassifier;
pub use rerank_provider::RerankProvider;
pub use session_repository::SessionRepository;
