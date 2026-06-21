//! `smos-application` — port traits, transport types, and error hierarchy.
//!
//! This layer is intentionally IO-free in the same way `smos-domain` is: it
//! only defines *what* the system does (port traits + DTOs) and *how it can
//! fail* (errors). Concrete adapters (surreal store, HTTP upstream, Ollama
//! provider, …) live in `smos-adapters`.
//!
//! The `helpers` module hosts protocol-shaped pure functions (HTTP/JSON/regex
//! parsing) that build on top of the domain value objects. They are NOT domain
//! concerns — the domain layer is restricted to entities and value objects.
//!
//! # Lint policy
//!
//! `async_fn_in_trait` is allowed workspace-wide for the port traits. The
//! returned futures do not carry an explicit `Send` bound on the trait
//! surface; concrete adapters are written to return `Send` futures, and use
//! cases that need `Send` propagate the requirement via `T: Trait + Send +
//! Sync + 'static` bounds at the spawn site.

#![allow(async_fn_in_trait)]

pub mod errors;
pub mod helpers;
pub mod ports;
pub mod types;
pub mod use_cases;

pub use errors::UseCaseError;
