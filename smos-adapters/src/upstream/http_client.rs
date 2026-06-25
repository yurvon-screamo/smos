//! Canonical `reqwest::Client` factory for SMOS adapters.
//!
//! Every production HTTP probe (NLI archive download, `smos init` llama-server
//! health checks, `smos doctor` connectivity probes) builds its client here so
//! the builder settings (timeout, pooling, TLS defaults) have one source of
//! truth. The two entry points mirror the two builder shapes the call sites
//! used inline before this module existed:
//!
//! - [`with_timeout`] — the per-request timeout pattern (archive downloads,
//!   bounded health probes).
//! - [`default_client`] — the bare default builder (no per-request timeout;
//!   the caller bounds the request itself via `RequestBuilder::timeout`).
//!
//! Test code keeps using `reqwest::Client::new()` directly — those are
//! intentionally default, short-lived test clients with no pooling needs.

use std::time::Duration;

use reqwest::Client;

/// Build a pooled `reqwest::Client` with the supplied per-request timeout.
pub fn with_timeout(timeout: Duration) -> Result<Client, reqwest::Error> {
    Client::builder().timeout(timeout).build()
}

/// Build a pooled `reqwest::Client` with default builder settings.
pub fn default_client() -> Result<Client, reqwest::Error> {
    Client::builder().build()
}
