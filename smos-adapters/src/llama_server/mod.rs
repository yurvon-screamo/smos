//! `llama-server` process lifecycle manager.
//!
//! When `auto_launch` is enabled, [`LlamaCppManager`] spawns one
//! `llama-server` process per configured service (embedding, reranker,
//! extraction) at `smos serve` startup, probes their `/health` endpoint
//! until they answer, and kills them again on shutdown. The manager NEVER
//! touches a service that is already responding on its port — an operator
//! who already started a `llama-server` by hand (or another SMOS instance)
//! is reused as-is.
//!
//! This module is named after the binary it manages (`llama-server`),
//! NOT after the upstream library (`llama.cpp`); the HTTP reranker client
//! that talks to an already-running server lives in
//! [`crate::providers::llama_cpp`] and carries the library name.

pub mod config;
mod health;
mod manager;

pub use config::{LlamaCppConfig, LlamaCppServiceConfig};
pub use manager::LlamaCppManager;
