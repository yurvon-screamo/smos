//! Pure helpers for `smos import`: pagination windowing, discovery error
//! mapping, opencode→SMOS session-id derivation, dry-run turn printing.
//!
//! Kept separate from `import_runner` so the unit tests can live next to
//! the functions they cover and the runner stays focused on the IO
//! orchestration.

use anyhow::{Context, Result};
use sha1::{Digest, Sha1};

use smos_application::use_cases::import_opencode_session::AssistantTurn;
use smos_domain::SessionId;

use crate::opencode::DiscoveryError;

/// Apply offset + limit pagination to the parsed turns.
pub fn apply_offset_limit(
    mut turns: Vec<AssistantTurn>,
    offset: usize,
    limit: Option<usize>,
) -> Vec<AssistantTurn> {
    let offset = offset.min(turns.len());
    turns.drain(..offset);
    if let Some(limit) = limit {
        turns.truncate(limit);
    }
    turns
}

/// Print every turn to stdout; used by `--dry-run` so operators can verify
/// the parser output before committing to a model round-trip.
pub fn print_dry_run(turns: &[AssistantTurn]) {
    for (i, turn) in turns.iter().enumerate() {
        println!("\n--- Turn {} ({}) ---", i + 1, turn.agent);
        println!("Message ID: {}", turn.message_id);
        println!("Content: {}", turn.content);
        if !turn.tool_calls.is_empty() {
            println!("Tool calls:");
            for tc in &turn.tool_calls {
                println!("  - {}({})", tc.name, tc.arguments);
            }
        }
    }
}

/// Map a [`DiscoveryError`] to an `anyhow::Error` with a single, consistent
/// message so the operator does not need to know the variant names.
pub fn map_discovery_error(e: DiscoveryError) -> anyhow::Error {
    match e {
        DiscoveryError::Http(url, msg) => {
            anyhow::anyhow!("HTTP request to {url} failed: {msg}")
        }
        DiscoveryError::CliNotFound(msg) => {
            anyhow::anyhow!("{msg}. Start `opencode serve` or install the CLI on PATH.")
        }
        DiscoveryError::CliTimeout(d) => anyhow::anyhow!("opencode CLI timed out after {d:?}"),
        DiscoveryError::CliFailed(msg) => anyhow::anyhow!("opencode CLI failed: {msg}"),
        DiscoveryError::Json(e) => {
            anyhow::anyhow!("opencode response was not valid JSON: {e}")
        }
    }
}

/// Resolve `raw` (an opencode session id) to a SMOS [`SessionId`].
///
/// opencode session ids do NOT always match the SMOS pattern
/// (`sess_<12 lowercase hex>`). When they do not, the id is deterministically
/// derived from a SHA-1 hash of the raw value: the same `raw` always maps to
/// the same SMOS id, so re-importing the same `--from-file` transcript grows
/// the provenance of the same fact instead of minting fresh random ids every
/// time (which would inflate cross-session confirmation counts and let
/// low-confidence facts cross the accept threshold on repeats).
pub fn derive_session_id(raw: &str) -> SessionId {
    if let Ok(id) = SessionId::from_raw(raw) {
        return id;
    }
    let mut hasher = Sha1::new();
    hasher.update(raw.as_bytes());
    let digest = hasher.finalize();
    let hex: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
    let derived = format!("sess_{hex}");
    // `info` (not `warn`) — opencode session ids routinely use `ses_` (single
    // s) while SMOS requires `sess_`, so the derivation path is the COMMON
    // case on every discovery import, not an anomaly. Operators must not be
    // trained to ignore a warning that fires on healthy runs.
    tracing::info!(
        original = raw,
        derived = %derived,
        "opencode session id does not match the SMOS pattern; \
         derived a deterministic SMOS id from its SHA-1 hash"
    );
    SessionId::from_raw(&derived).unwrap_or_else(|_| {
        // Defensive: SHA-1 of an arbitrary byte string always yields 12
        // lowercase hex chars under the construction above, so this branch
        // is unreachable in practice. Falling back to a static valid id
        // keeps the helper total — a `panic!` here would crash `smos import`
        // halfway through a long-running import on a deterministic derivation
        // path, which is the worst possible time to crash.
        tracing::error!(
            original = raw,
            derived = %derived,
            "SHA-1-derived session id failed SessionId validation; using static fallback"
        );
        SessionId::from_raw("sess_000000000000")
            .expect("static fallback id is valid by construction")
    })
}

/// Validate `raw` as a SMOS memory_key; surfaces a friendlier `anyhow`
/// context than the raw domain error.
pub fn parse_memory_key(raw: &str) -> Result<smos_domain::MemoryKey> {
    smos_domain::MemoryKey::from_raw(raw).context("invalid --memory-key (must be a safe namespace)")
}

#[cfg(test)]
mod tests {
    //! Unit tests for the pure helpers. The full CLI surface (discovery +
    //! parsing + import) is exercised by `tests/e2e_import.rs`; here we only
    //! cover the logic that is invisible to that suite (offset/limit
    //! windowing, error mapping, deterministic session-id derivation).

    use super::*;
    use std::time::Duration;

    fn turn(agent: &str, content: &str) -> AssistantTurn {
        AssistantTurn {
            message_id: format!("msg_{agent}"),
            agent: agent.to_string(),
            content: content.to_string(),
            tool_calls: Vec::new(),
        }
    }

    #[test]
    fn apply_offset_limit_drops_first_n_turns() {
        let turns = vec![turn("a", "1"), turn("b", "2"), turn("c", "3")];
        let out = apply_offset_limit(turns, 1, None);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].agent, "b");
        assert_eq!(out[1].agent, "c");
    }

    #[test]
    fn apply_offset_limit_truncates_to_limit() {
        let turns = vec![turn("a", "1"), turn("b", "2"), turn("c", "3")];
        let out = apply_offset_limit(turns, 0, Some(2));
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].agent, "a");
        assert_eq!(out[1].agent, "b");
    }

    #[test]
    fn apply_offset_limit_offset_beyond_length_yields_empty() {
        let turns = vec![turn("a", "1")];
        let out = apply_offset_limit(turns, 5, None);
        assert!(out.is_empty());
    }

    #[test]
    fn apply_offset_limit_offset_clamped_to_length() {
        let turns = vec![turn("a", "1"), turn("b", "2")];
        // offset=10 with len=2 must clamp to 2, not panic.
        let out = apply_offset_limit(turns, 10, None);
        assert!(out.is_empty());
    }

    #[test]
    fn apply_offset_limit_offset_plus_limit_window() {
        let turns = vec![
            turn("a", "1"),
            turn("b", "2"),
            turn("c", "3"),
            turn("d", "4"),
            turn("e", "5"),
        ];
        // Equivalent to `--offset 2 --limit 2`: drop first 2, keep next 2.
        let out = apply_offset_limit(turns, 2, Some(2));
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].agent, "c");
        assert_eq!(out[1].agent, "d");
    }

    #[test]
    fn map_discovery_error_http_message_contains_url() {
        let err = map_discovery_error(DiscoveryError::Http(
            "http://localhost:4096/health".into(),
            "connection refused".into(),
        ));
        let msg = format!("{err:#}");
        assert!(msg.contains("localhost:4096"));
        assert!(msg.contains("connection refused"));
    }

    #[test]
    fn map_discovery_error_cli_not_found_offers_remediation() {
        let err = map_discovery_error(DiscoveryError::CliNotFound("not found".into()));
        let msg = format!("{err:#}");
        assert!(msg.contains("opencode serve"));
        assert!(msg.contains("PATH"));
    }

    #[test]
    fn map_discovery_error_cli_timeout_includes_duration() {
        let err = map_discovery_error(DiscoveryError::CliTimeout(Duration::from_secs(120)));
        let msg = format!("{err:#}");
        assert!(msg.contains("120"));
    }

    #[test]
    fn map_discovery_error_json_mentions_json() {
        let err = map_discovery_error(DiscoveryError::Json(
            serde_json::from_str::<serde_json::Value>("{bad}").unwrap_err(),
        ));
        let msg = format!("{err:#}");
        assert!(msg.to_lowercase().contains("json"));
    }

    #[test]
    fn map_discovery_error_cli_failed_surfaces_message() {
        let err = map_discovery_error(DiscoveryError::CliFailed("exited 1: bad args".into()));
        let msg = format!("{err:#}");
        assert!(msg.contains("opencode CLI failed"));
        assert!(msg.contains("exited 1"));
    }

    #[test]
    fn derive_session_id_accepts_valid_smoss_pattern() {
        let id = derive_session_id("sess_abcdef012345");
        assert_eq!(id.as_str(), "sess_abcdef012345");
    }

    #[test]
    fn derive_session_id_derives_deterministic_id_for_non_matching_input() {
        // Same input → same derived id (idempotent re-imports).
        let a = derive_session_id("opencode-session-xyz");
        let b = derive_session_id("opencode-session-xyz");
        assert_eq!(a, b);
        // Derived id must match the SMOS pattern.
        assert!(a.as_str().starts_with("sess_"));
        assert_eq!(a.as_str().len(), "sess_".len() + 12);
    }

    #[test]
    fn derive_session_id_different_inputs_yield_different_ids() {
        let a = derive_session_id("opencode-session-aaa");
        let b = derive_session_id("opencode-session-bbb");
        assert_ne!(a, b);
    }
}
