//! `[persons.*]` consistency checks for the doctor.
//!
//! Each person is a routing entry — the doctor reports whether the
//! provider each person references still exists in `[[providers]]` and
//! whether the optional persona `.md` resolves to a real file. These are
//! the same invariants [`crate::config::SmosConfig::validate`] enforces
//! for `provider` references; the doctor surfaces them again so a config
//! edited AFTER startup (without re-running `smos serve`) does not stay
//! broken until the first 503.
//!
//! Pure checks — no IO beyond `Path::exists`. Doctor never mutates the
//! config or the persona files.

use super::super::types::CheckResult;
use crate::config::SmosConfig;
use crate::paths::expand_tilde;

/// For every entry in `[persons.*]`, emit one row per condition checked:
/// - provider reference exists in `[[providers]]` (FAIL if missing),
/// - optional persona `.md` resolves to a file (FAIL if missing; skipped
///   entirely when no persona is declared).
pub fn check_persons(config: &SmosConfig) -> Vec<CheckResult> {
    let mut results = Vec::new();

    for (name, person) in &config.persons {
        results.push(check_person_provider(name, &person.provider, config));
        if !person.persona.is_empty() {
            results.push(check_person_persona(name, &person.persona));
        }
    }

    results
}

fn check_person_provider(name: &str, provider: &str, config: &SmosConfig) -> CheckResult {
    let provider_exists = config.providers.iter().any(|p| p.name == provider);
    if provider_exists {
        CheckResult::pass(
            format!("Person '{name}' provider"),
            format!("provider '{provider}' resolved"),
        )
    } else {
        CheckResult::fail(
            format!("Person '{name}' provider"),
            format!("provider '{provider}' not found"),
        )
        .with_recommendation("check [[providers]] in config.toml")
    }
}

fn check_person_persona(name: &str, persona: &str) -> CheckResult {
    let persona_path = expand_tilde(persona);
    if persona_path.exists() {
        CheckResult::pass(
            format!("Person '{name}' persona"),
            persona_path.display().to_string(),
        )
    } else {
        CheckResult::fail(
            format!("Person '{name}' persona"),
            format!("not found: {}", persona_path.display()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PersonConfig, ProviderConfig};
    use std::collections::HashMap;

    fn build_config(
        providers: Vec<ProviderConfig>,
        persons: HashMap<String, PersonConfig>,
    ) -> SmosConfig {
        SmosConfig {
            providers,
            persons,
            ..SmosConfig::default()
        }
    }

    #[test]
    fn check_persons_passes_when_provider_exists_and_persona_skipped() {
        let mut persons = HashMap::new();
        persons.insert(
            "bob".into(),
            PersonConfig {
                provider: "llama-local".into(),
                model: "m".into(),
                persona: String::new(),
            },
        );
        let providers = vec![ProviderConfig::new("llama-local", "http://x")];
        let cfg = build_config(providers, persons);

        let rows = check_persons(&cfg);
        assert_eq!(rows.len(), 1, "only the provider row should be emitted");
        assert!(rows[0].status.is_pass());
    }

    #[test]
    fn check_persons_fails_when_provider_missing() {
        let mut persons = HashMap::new();
        persons.insert(
            "bob".into(),
            PersonConfig {
                provider: "ghost".into(),
                model: "m".into(),
                persona: String::new(),
            },
        );
        let cfg = build_config(
            vec![ProviderConfig::new("llama-local", "http://x")],
            persons,
        );

        let rows = check_persons(&cfg);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].status.is_fail());
        assert!(
            rows[0]
                .recommendation
                .as_deref()
                .unwrap_or("")
                .contains("[[providers]]"),
            "FAIL row must point operators at the [[providers]] array"
        );
    }

    #[test]
    fn check_persons_emits_persona_row_only_when_persona_set() {
        let mut persons = HashMap::new();
        persons.insert(
            "bob".into(),
            PersonConfig {
                provider: "llama-local".into(),
                model: "m".into(),
                persona: "/nonexistent/bob.md".into(),
            },
        );
        let providers = vec![ProviderConfig::new("llama-local", "http://x")];
        let cfg = build_config(providers, persons);

        let rows = check_persons(&cfg);
        assert_eq!(rows.len(), 2, "provider + persona rows expected");
        assert!(rows[0].status.is_pass(), "provider exists");
        assert!(rows[1].status.is_fail(), "persona path does not exist");
    }

    #[test]
    fn check_persons_is_empty_when_no_persons_configured() {
        let cfg = build_config(vec![ProviderConfig::new("p", "http://x")], HashMap::new());
        let rows = check_persons(&cfg);
        assert!(rows.is_empty(), "no persons → no rows");
    }
}
