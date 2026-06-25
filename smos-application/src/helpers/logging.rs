//! Fail-open logging helper.
//!
//! The [`log_nonfatal!`] macro unifies the `if let Err(e) = … { tracing::warn!(…) }`
//! swallow pattern that recurs across the finalize / merge / save paths.
//! Semantics are identical to the inlined form: the error is logged at WARN
//! and execution continues. Structured fields are passed through verbatim —
//! only the `error = %e` binding is injected by the macro itself, so callers
//! must NOT repeat it.

/// Log a non-fatal error at WARN and continue.
///
/// Expands to `if let Err(e) = … { tracing::warn!(error = %e, …) }`. The
/// injected `error = %e` binding is the only thing the macro adds itself;
/// callers pass any additional structured fields and the message literal
/// verbatim. `tracing` accepts fields in any order before the message, so
/// the injected `error = %e` is valid whether the caller adds fields or
/// only a context string.
#[macro_export]
macro_rules! log_nonfatal {
    ($result:expr, $($args:tt)*) => {
        if let Err(e) = $result {
            tracing::warn!(error = %e, $($args)*);
        }
    };
}
