//! `llama-server` auto-launch helper extracted out of [`server_runner`].
//!
//! Owning the spawn-or-skip decision in its own module keeps
//! [`server_runner`] under the file-size budget and gives the spawn path a
//! single, well-tested entry point.

use crate::llama_server::{LlamaCppConfig, LlamaCppManager};

/// Build and launch the llama.cpp process manager.
///
/// Returns `None` (and logs the reason) when:
/// - `auto_launch = false` (the operator launches `llama-server` by hand); or
/// - the probe client could not be built; or
/// - `launch_all()` reported a non-recoverable error (binary missing,
///   model path wrong, health probe timeout).
///
/// The HTTP server keeps running in every `None` case so chat completions
/// stay available — the operator can still point `[embedding]` /
/// `[reranker]` at external servers.
pub async fn spawn_llama_cpp(config: &LlamaCppConfig) -> Option<LlamaCppManager> {
    if !config.auto_launch {
        tracing::info!("llama.cpp auto-launch disabled; not spawning llama-server");
        return None;
    }
    let manager = match LlamaCppManager::new(config.clone()) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(
                error = %format!("{e:#}"),
                "llama.cpp manager build failed; auto-launch disabled \
                 (HTTP server keeps running)."
            );
            return None;
        }
    };
    if let Err(e) = manager.launch_all().await {
        tracing::warn!(
            error = %format!("{e:#}"),
            "llama.cpp auto-launch hit an error; partial launch may be in effect \
             (HTTP server keeps running). Run `smos doctor` to inspect the running \
             services."
        );
    }
    Some(manager)
}
