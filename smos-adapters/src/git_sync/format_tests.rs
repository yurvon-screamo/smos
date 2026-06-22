//! Tests for [`super::format`]. Extracted into its own file so
//! [`super::format`] stays under the file-size budget.

#![cfg(test)]

use super::format::{parse_fact_md, render_fact_md};
use crate::git_sync::test_support::sample_fact;

#[test]
fn render_and_parse_round_trip() {
    let fact = sample_fact("Rust is memory-safe.", "origa");
    let md = render_fact_md(&fact).expect("render");
    assert!(md.starts_with("---\n"));
    let (fm, body) = parse_fact_md(&md).expect("round-trip parses");
    assert_eq!(body, "Rust is memory-safe.");
    assert_eq!(fm.memory_key, "origa");
    assert_eq!(fm.status, "accepted");
    let rebuilt = fm.to_fact(&body).expect("rehydrate");
    assert_eq!(rebuilt.id(), fact.id());
    assert_eq!(rebuilt.content(), fact.content());
    assert_eq!(rebuilt.status(), fact.status());
}

#[test]
fn parse_returns_none_for_missing_frontmatter() {
    assert!(parse_fact_md("just body").is_none());
}

#[test]
fn parse_returns_none_for_unclosed_frontmatter() {
    assert!(parse_fact_md("---\nid = \"x\"\nbody never closes").is_none());
}

#[test]
fn parse_returns_none_for_invalid_toml() {
    assert!(parse_fact_md("---\nthis is not = valid toml\n---\nbody").is_none());
}

#[test]
fn to_fact_falls_back_to_entity_for_unknown_fact_type() {
    // The frontmatter carries an unknown `fact_type` (e.g. produced by a
    // future SMOS version that introduces a new variant). The rehydrate
    // path MUST NOT abort the import — it falls back to `Entity` and
    // emits a WARN so the operator notices. The body round-trips intact.
    let fact = sample_fact("body with unknown type", "origa");
    let mut md = render_fact_md(&fact).expect("render");
    let needle = "fact_type = \"entity\"";
    let replacement = "fact_type = \"future_variant\"";
    md = md.replacen(needle, replacement, 1);
    assert!(
        md.contains(replacement),
        "fixture setup: replacement applied"
    );

    let (fm, body) = parse_fact_md(&md).expect("round-trip parses");
    assert_eq!(body, "body with unknown type");
    assert_eq!(fm.fact_type, "future_variant");
    let rebuilt = fm.to_fact(&body).expect("rehydrate falls back, not errors");
    assert_eq!(rebuilt.fact_type(), smos_domain::FactType::Entity);
}
