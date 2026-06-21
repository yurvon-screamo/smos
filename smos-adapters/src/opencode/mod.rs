//! opencode integration — discovery + transcript parsing for the import path.
//!
//! This module is the Rust counterpart of `smos-poc/scripts/opencode_source.py`.
//! It is consumed by the `smos import` subcommand (Slice-8) to locate an
//! opencode session (HTTP probe with CLI fallback), parse its transcript into
//! assistant turns, and feed those turns into [`ImportOpencodeSession`].
//!
//! # Layering
//!
//! - [`transcript`] is pure: JSON in, domain-shaped turns out. Lives in the
//!   adapter layer because it depends on the opencode wire shape (which is an
//!   adapter-boundary concern, not a domain concept).
//! - [`discovery`] does IO (HTTP probe, CLI subprocess) but returns plain
//!   `serde_json::Value`s so callers can decide how strict to be.
//! - [`cli`] is a thin `tokio::process::Command` wrapper around the local
//!   `opencode` binary.
//!
//! [`ImportOpencodeSession`]: smos_application::use_cases::ImportOpencodeSession

pub mod cli;
pub mod discovery;
pub mod transcript;

pub use discovery::{
    DiscoveryError, SessionSource, fetch_session_export, list_sessions, probe_http, probe_ports,
    resolve_source,
};
pub use transcript::parse_transcript;
