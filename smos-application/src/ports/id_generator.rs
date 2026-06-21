//! `IdGenerator` port — injectable source of fresh session ids.
//!
//! Production code uses `SystemIdGenerator` from `smos-adapters`; tests
//! inject a fake so the ids are predictable. The port exists so the
//! domain stays free of the `SessionId::new()` constructor (reading
//! system entropy is an IO concern — same layering rule that motivates
//! the [`Clock`] port for wall-clock time).
//!
//! [`Clock`]: super::Clock

use smos_domain::SessionId;

/// Fresh-session-id boundary.
pub trait IdGenerator {
    /// Mint a fresh, never-before-seen session id.
    fn new_session_id(&self) -> SessionId;
}
