use super::*;

mod defaults_parse_legacy;
mod validate_audit;

/// Acquire the workspace-wide env-test lock. See
/// [`crate::test_env_lock`] for why this is required.
fn _lock() -> std::sync::MutexGuard<'static, ()> {
    crate::test_env_lock::lock()
}

fn one_provider() -> ProviderConfig {
    ProviderConfig::new("u", "http://u")
}
