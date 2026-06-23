//! `route_request` — resolve a person name into a routing decision.
//!
//! Replaces the legacy `parse_model("memory_key:model")` helper with a
//! config-driven lookup against the `[persons.*]` map declared in
//! `smos.toml`. Each person is simultaneously:
//!
//! - a **memory key** (the namespace under which extracted facts land), and
//! - a **provider + model pair** (the upstream route), and
//! - an optional **persona `.md`** (a system-message payload to inject).
//!
//! The router performs three validations per request:
//! 1. The requested model must name a configured person
//!    ([`RouteError::UnknownPerson`]).
//! 2. The person's `provider` MUST reference an existing provider entry
//!    ([`RouteError::UnknownProvider`]).
//! 3. The person name MUST be a valid [`MemoryKey`]
//!    ([`RouteError::InvalidMemoryKey`]).
//!
//! Persona loading is fail-soft: a missing file degrades to
//! `persona_content: None` and the request is forwarded without a system
//! message injection. This mirrors the proxy's fail-open contract for the
//! enrichment pipeline — a misconfigured persona file is a startup mistake,
//! not a 503.
//!
//! # Architecture note
//!
//! The router lives in the IO-free `smos-application` crate. To stay
//! decoupled from the adapter-side config types
//! (`smos::config::{ProviderConfig, PersonConfig}`), it consumes
//! lightweight [`ProviderEntry`] / [`PersonEntry`] views. The adapter
//! constructs these views from its own config when invoking the router — a
//! cheap field-by-field copy at the HTTP boundary that keeps the
//! application layer free of `serde` / `config` dependencies beyond what it
//! already pulls in.

use std::collections::HashMap;
use std::path::PathBuf;

use serde_json::Value;
use smos_domain::MemoryKey;

/// Minimal view of a `[[providers]]` entry needed for routing.
///
/// Constructed by the adapter from `smos::config::ProviderConfig`.
/// Only the `name` field matters for routing decisions — the URL + api-key
/// live in the adapter and are looked up separately by the HTTP layer.
#[derive(Debug, Clone)]
pub struct ProviderEntry {
    pub name: String,
}

/// Minimal view of a `[persons.*]` entry needed for routing.
///
/// Constructed by the adapter from `smos::config::PersonConfig`.
#[derive(Debug, Clone)]
pub struct PersonEntry {
    pub provider: String,
    pub model: String,
    /// Filesystem path to the persona `.md`. The router expands `~` and
    /// reads the file when this field is non-empty. Empty path = no persona.
    pub persona: String,
}

/// One routing decision returned by [`route_request`].
///
/// Carries every piece of state the request pipeline needs to forward a
/// chat-completion call to the right provider with the right model and the
/// right persona system message.
#[derive(Debug, Clone)]
pub struct PersonRoute {
    /// Memory namespace = person name. Used by enrichment + extraction to
    /// scope retrieved / persisted facts.
    pub memory_key: MemoryKey,
    /// Provider name from `[[providers]].name`. The HTTP layer looks up the
    /// concrete URL + auth header from the provider map.
    pub provider_name: String,
    /// Upstream model id (e.g. `granite4.1:3b`). The HTTP layer rewrites
    /// `request.model` to this before forwarding.
    pub upstream_model: String,
    /// Expanded persona path (`~` resolved). The adapter loads the file
    /// asynchronously (via `tokio::fs::read_to_string`) so the application
    /// layer stays IO-free. `None` when the person declares no persona
    /// path (`persona = ""` in TOML).
    pub persona_path: Option<PathBuf>,
}

/// Errors returned by [`route_request`].
///
/// Each variant carries enough context for the HTTP layer to render a useful
/// 400 / 500 response. The variant names are stable and tested.
#[derive(Debug, thiserror::Error)]
pub enum RouteError {
    #[error("unknown person '{0}'. Configure under [persons.{0}] in smos.toml")]
    UnknownPerson(String),

    #[error(
        "unknown provider '{0}' referenced by person '{1}'. \
         Configure under [[providers]] in smos.toml"
    )]
    UnknownProvider(String, String),

    #[error("invalid memory key '{0}': {1}")]
    InvalidMemoryKey(String, String),
}

/// Resolve a requested person name into a routing decision.
///
/// `requested_model` is the raw value of `ChatRequest::model` (e.g.
/// `"bob"`). Validation order:
/// 1. The requested name MUST be a valid [`MemoryKey`] — path-traversal
///    characters and other unsafe inputs are rejected BEFORE any config
///    lookup so an attacker cannot probe the persons map with crafted
///    names.
/// 2. The requested name MUST name a configured person
///    ([`RouteError::UnknownPerson`]).
/// 3. The person's `provider` MUST reference an existing provider entry
///    ([`RouteError::UnknownProvider`]).
///
/// The provider list is consulted only for the existence check — the URL +
/// api-key resolution happens in the HTTP adapter (`ReqwestUpstreamRouter`)
/// so the application layer stays IO-free.
///
/// Persona loading is deliberately NOT performed here. The router returns
/// the expanded persona path (`persona_path`) and the adapter loads the
/// file asynchronously via `tokio::fs` so the application layer never
/// blocks the async runtime on file IO.
pub fn route_request(
    requested_model: &str,
    persons: &HashMap<String, PersonEntry>,
    providers: &[ProviderEntry],
) -> Result<PersonRoute, RouteError> {
    // Step 1 — structural validation. An unsafe name rejects the request
    // before the persons map is consulted so a hostile client cannot
    // enumerate configured persons via differential error messages.
    let memory_key = MemoryKey::from_raw(requested_model)
        .map_err(|e| RouteError::InvalidMemoryKey(requested_model.to_string(), e.to_string()))?;

    // Step 2 — person exists.
    let person = persons
        .get(requested_model)
        .ok_or_else(|| RouteError::UnknownPerson(requested_model.to_string()))?;

    // Step 3 — provider reference resolves.
    if !providers.iter().any(|p| p.name == person.provider) {
        return Err(RouteError::UnknownProvider(
            person.provider.clone(),
            requested_model.to_string(),
        ));
    }

    let persona_path = if person.persona.is_empty() {
        None
    } else {
        Some(expand_tilde(&person.persona))
    };

    Ok(PersonRoute {
        memory_key,
        provider_name: person.provider.clone(),
        upstream_model: person.model.clone(),
        persona_path,
    })
}

/// Load a persona `.md` file by already-expanded path.
///
/// Synchronous helper used by `HandleChatCompletion` to load the persona
/// declared in `[persons.X].persona`. The path comes pre-expanded from
/// [`route_request`] (which calls [`expand_tilde`] on the raw TOML value),
/// so this function does NOT touch the env vars again.
///
/// Returns `None` when the file is missing or unreadable so a misconfigured
/// persona degrades to "no system message" rather than a 503. The fail-open
/// matches the proxy's enrichment contract.
///
/// # Why synchronous IO is acceptable here
///
/// Persona `.md` files are tiny (typically < 1 KB). After the first read,
/// the OS page cache serves every subsequent access in microseconds. The
/// LLM-proxy round-trip dominates request latency by 3–4 orders of
/// magnitude, so the cold-cache read on the first request per persona is
/// noise. Keeping the helper sync avoids pulling `tokio::fs` into the
/// otherwise-async-runtime-agnostic application crate.
pub fn load_persona_at(path: &PathBuf) -> Option<String> {
    match std::fs::read_to_string(path) {
        Ok(content) => Some(content),
        Err(e) => {
            tracing::warn!(
                persona_path = %path.display(),
                error = %e,
                "failed to load persona file; persona injection skipped (fail-soft)"
            );
            None
        }
    }
}

/// Load a persona `.md` file by raw string path, expanding a leading `~/`
/// first. Retained as a convenience for unit tests that exercise the
/// path-expansion + read pipeline with a raw config-style string.
pub fn load_persona(path: &str) -> Option<String> {
    let expanded = expand_tilde(path);
    load_persona_at(&expanded)
}

/// Expand a leading `~/`, `~\`, or bare `~` to the OS user home directory.
///
/// Reads `HOME` (unix) / `USERPROFILE` (windows) directly so the helper
/// stays IO-free at the crate-dependency level — no `dirs` crate pull-in,
/// no async runtime needed. The canonical home resolver in
/// `smos::paths::expand_tilde` mirrors this implementation; the
/// two MUST stay in sync so persona paths resolve identically on both sides
/// of the application / adapter boundary.
pub fn expand_tilde(path: &str) -> PathBuf {
    let stripped = path
        .strip_prefix("~/")
        .or_else(|| path.strip_prefix("~\\"))
        .or_else(|| path.strip_prefix("~"));
    if let Some(rest) = stripped
        && let Some(home) = user_home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(path)
}

/// Resolve the OS user home directory without pulling in the `dirs` crate.
///
/// Mirrors `smos::paths::user_home_dir` BEHAVIOUR-FOR-BEHAVIOUR so
/// persona paths resolve identically on both sides of the application /
/// adapter boundary. Both implementations:
/// - consult `USERPROFILE` first on Windows (covers roaming profiles +
///   service accounts where `USERPROFILE` is set explicitly),
/// - fall back to `HOMEDRIVE` + `HOMEPATH` on Windows when `USERPROFILE`
///   is missing (the canonical fallback used by cmd.exe / PowerShell for
///   stripped-down accounts),
/// - consult `HOME` on unix.
///
/// The duplication is intentional (keeping `smos-application` IO-free at the
/// crate-dependency level) but the two implementations MUST stay
/// behaviourally identical. The `paths::tests::user_home_dir_matches_application_router`
/// test pins this by asserting that both functions return the same value
/// for every env-var combination exercised.
pub fn user_home_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        if let Some(p) = std::env::var_os("USERPROFILE").filter(|s| !s.is_empty()) {
            return Some(PathBuf::from(p));
        }
        let drive = std::env::var_os("HOMEDRIVE");
        let path = std::env::var_os("HOMEPATH");
        match (drive, path) {
            (Some(d), Some(p)) => {
                let mut combined = PathBuf::from(d);
                combined.push(p);
                Some(combined)
            }
            _ => None,
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("HOME")
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
    }
}

/// Inject a persona into the message stream as the system message.
///
/// Behaviour:
/// - Empty messages → insert a new system message at index 0.
/// - First message is already `system` → PREPEND the persona (existing
///   system content is preserved and joined by a blank line separator so
///   the persona reads as a preamble rather than a concatenation).
/// - First message is not `system` → insert a new system message at index 0.
///
/// The persona is the source of identity for the upstream model, so it MUST
/// be the first thing the model reads. Prepending (rather than appending)
/// preserves the operator-authored system prompt ordering when the caller
/// already provides one.
pub fn inject_persona_into_messages(messages: &mut Vec<Value>, persona: &str) {
    if messages.is_empty() {
        messages.push(serde_json::json!({"role": "system", "content": persona}));
        return;
    }
    let first = &mut messages[0];
    let is_system = first
        .get("role")
        .and_then(Value::as_str)
        .map(|r| r == "system")
        .unwrap_or(false);
    if is_system {
        let existing = first
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        first["content"] = Value::String(format!("{persona}\n\n{existing}"));
    } else {
        messages.insert(0, serde_json::json!({"role": "system", "content": persona}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(name: &str) -> ProviderEntry {
        ProviderEntry { name: name.into() }
    }

    fn person(provider: &str, model: &str, persona: &str) -> PersonEntry {
        PersonEntry {
            provider: provider.into(),
            model: model.into(),
            persona: persona.into(),
        }
    }

    fn build_persons(entries: &[(&str, &str, &str, &str)]) -> HashMap<String, PersonEntry> {
        let mut map = HashMap::new();
        for (name, provider, model, persona) in entries {
            map.insert(
                name.to_string(),
                PersonEntry {
                    provider: provider.to_string(),
                    model: model.to_string(),
                    persona: persona.to_string(),
                },
            );
        }
        map
    }

    fn build_providers(names: &[&str]) -> Vec<ProviderEntry> {
        names.iter().map(|n| provider(n)).collect()
    }

    // --- route_request ----------------------------------------------

    #[test]
    fn route_request_happy_path_returns_memory_key_provider_model() {
        let providers = build_providers(&["llama-local"]);
        let persons = build_persons(&[("bob", "llama-local", "granite4.1:3b", "")]);
        let route = route_request("bob", &persons, &providers).expect("route");
        assert_eq!(route.memory_key.as_str(), "bob");
        assert_eq!(route.provider_name, "llama-local");
        assert_eq!(route.upstream_model, "granite4.1:3b");
        assert!(route.persona_path.is_none());
    }

    #[test]
    fn route_request_unknown_person_returns_unknown_person_error() {
        let providers = build_providers(&["llama-local"]);
        let persons = HashMap::new();
        let err = route_request("ghost", &persons, &providers).expect_err("unknown");
        assert!(matches!(err, RouteError::UnknownPerson(name) if name == "ghost"));
    }

    #[test]
    fn route_request_unknown_provider_returns_unknown_provider_error() {
        let providers = build_providers(&["llama-local"]);
        let persons = build_persons(&[("bob", "typo", "granite4.1:3b", "")]);
        let err = route_request("bob", &persons, &providers).expect_err("unknown");
        match err {
            RouteError::UnknownProvider(provider_name, person) => {
                assert_eq!(provider_name, "typo");
                assert_eq!(person, "bob");
            }
            other => panic!("expected UnknownProvider, got {other:?}"),
        }
    }

    #[test]
    fn route_request_invalid_memory_key_returns_invalid_memory_key_error() {
        // A person name with a path separator fails MemoryKey validation,
        // even though the TOML parser would not have accepted it either.
        // The router is defensive: it re-validates so a programmatic config
        // edit cannot bypass the domain invariant.
        let providers = build_providers(&["llama-local"]);
        let mut persons = HashMap::new();
        persons.insert(
            "a/b".to_string(),
            person("llama-local", "granite4.1:3b", ""),
        );
        let err = route_request("a/b", &persons, &providers).expect_err("invalid key");
        assert!(matches!(err, RouteError::InvalidMemoryKey(_, _)));
    }

    #[test]
    fn route_request_returns_persona_path_when_declared() {
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        std::fs::write(tmp.path(), "You are Bob.").expect("write");
        let providers = build_providers(&["llama-local"]);
        let mut persons = HashMap::new();
        persons.insert(
            "bob".into(),
            person("llama-local", "granite4.1:3b", tmp.path().to_str().unwrap()),
        );
        let route = route_request("bob", &persons, &providers).expect("route");
        // route_request returns the EXPANDED path; the actual file read
        // happens in `load_persona_at` (called by HandleChatCompletion).
        let path = route.persona_path.expect("persona path");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "You are Bob.");
    }

    #[test]
    fn route_request_persona_path_absent_when_not_declared() {
        let providers = build_providers(&["llama-local"]);
        let persons = build_persons(&[("bob", "llama-local", "granite4.1:3b", "")]);
        let route = route_request("bob", &persons, &providers).expect("route");
        assert!(
            route.persona_path.is_none(),
            "empty persona string MUST yield None path"
        );
    }

    #[test]
    fn load_persona_at_returns_none_for_missing_file_without_panic() {
        let path = PathBuf::from("/definitely/does/not/exist.md");
        assert!(load_persona_at(&path).is_none());
    }

    // --- expand_tilde -----------------------------------------------

    #[test]
    fn expand_tilde_passes_through_absolute_paths() {
        assert_eq!(expand_tilde("/etc/passwd"), PathBuf::from("/etc/passwd"));
        assert_eq!(
            expand_tilde("relative/path"),
            PathBuf::from("relative/path")
        );
        assert_eq!(expand_tilde(""), PathBuf::from(""));
    }

    // --- inject_persona_into_messages -------------------------------

    #[test]
    fn inject_persona_into_empty_messages_creates_system_message() {
        let mut messages: Vec<Value> = vec![];
        inject_persona_into_messages(&mut messages, "be bob");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "be bob");
    }

    #[test]
    fn inject_persona_prepends_to_existing_system_message() {
        let mut messages: Vec<Value> = vec![
            serde_json::json!({"role": "system", "content": "existing"}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        inject_persona_into_messages(&mut messages, "persona");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "persona\n\nexisting");
    }

    #[test]
    fn inject_persona_inserts_system_message_when_first_is_user() {
        let mut messages: Vec<Value> = vec![serde_json::json!({"role": "user", "content": "hi"})];
        inject_persona_into_messages(&mut messages, "persona");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "persona");
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn inject_persona_into_existing_empty_system_content_uses_persona_only() {
        // When the existing system message has empty content, the joined
        // payload is `<persona>\n\n` — i.e. the persona wins, the trailing
        // whitespace is harmless for upstream LLMs that strip it. Pinned
        // so a future "smart trim" refactor does not silently change the
        // payload shape.
        let mut messages: Vec<Value> = vec![serde_json::json!({"role": "system", "content": ""})];
        inject_persona_into_messages(&mut messages, "persona");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "persona\n\n");
    }
}
