use super::*;
use std::collections::HashMap;

// -----------------------------------------------------------------------
// validate() — range / consistency checks
// -----------------------------------------------------------------------

#[test]
fn validate_accepts_default_plus_one_provider() {
    // The minimum config that should pass validation: defaults + one
    // provider. Anchors the lower bound of what `smos serve` accepts.
    let mut cfg = SmosConfig::default();
    cfg.providers.push(one_provider());
    assert!(cfg.validate().is_ok(), "default + 1 provider must validate");
}

#[test]
fn validate_rejects_wrong_embedding_dimensions() {
    let mut cfg = SmosConfig::default();
    cfg.embedding.dimensions = 512;
    cfg.providers.push(one_provider());
    let err = cfg
        .validate()
        .expect_err("non-canonical dimensions must fail");
    let msg = err.to_string();
    assert!(msg.contains("embedding.dimensions"), "msg = {msg}");
    assert!(msg.contains("1024"), "msg = {msg}");
}

#[test]
fn validate_rejects_confidence_out_of_range() {
    let mut cfg = SmosConfig::default();
    cfg.confidence.base = 1.5;
    cfg.providers.push(one_provider());
    let err = cfg.validate().expect_err("base > 1 must fail");
    assert!(err.to_string().contains("confidence.base"));
}

#[test]
fn validate_rejects_accept_below_pending_threshold() {
    let mut cfg = SmosConfig::default();
    cfg.confidence.accept_threshold = 0.3;
    cfg.confidence.pending_threshold = 0.5;
    cfg.providers.push(one_provider());
    let err = cfg.validate().expect_err("accept < pending must fail");
    let msg = err.to_string();
    assert!(msg.contains("accept_threshold"), "msg = {msg}");
    assert!(msg.contains("pending_threshold"), "msg = {msg}");
}

#[test]
fn validate_rejects_empty_providers() {
    let cfg = SmosConfig::default();
    let err = cfg.validate().expect_err("no providers must fail");
    assert!(
        err.to_string().contains("providers must not be empty"),
        "got: {err}"
    );
}

#[test]
fn validate_rejects_provider_with_empty_url() {
    let mut cfg = SmosConfig::default();
    let mut p = ProviderConfig::new("u", "");
    p.timeout_seconds = 9;
    cfg.providers.push(p);
    let err = cfg.validate().expect_err("empty url must fail");
    assert!(err.to_string().contains("url must not be empty"));
}

#[test]
fn validate_rejects_provider_with_zero_timeout() {
    let mut cfg = SmosConfig::default();
    let mut p = ProviderConfig::new("u", "http://u");
    p.timeout_seconds = 0;
    cfg.providers.push(p);
    let err = cfg.validate().expect_err("zero timeout must fail");
    assert!(err.to_string().contains("timeout_seconds must be > 0"));
}

#[test]
fn validate_rejects_provider_with_empty_name() {
    let mut cfg = SmosConfig::default();
    let mut p = ProviderConfig::new("", "http://u");
    p.timeout_seconds = 9;
    cfg.providers.push(p);
    let err = cfg.validate().expect_err("empty name must fail");
    assert!(err.to_string().contains("name must not be empty"));
}

#[test]
fn validate_rejects_duplicate_provider_names() {
    let mut cfg = SmosConfig::default();
    cfg.providers.push(ProviderConfig::new("dup", "http://a"));
    cfg.providers.push(ProviderConfig::new("dup", "http://b"));
    let err = cfg.validate().expect_err("duplicate name must fail");
    let msg = err.to_string();
    assert!(msg.contains("duplicated"), "msg = {msg}");
    assert!(msg.contains("dup"), "msg = {msg}");
}

#[test]
fn validate_rejects_person_referencing_unknown_provider() {
    let mut cfg = SmosConfig::default();
    cfg.providers.push(ProviderConfig::new("known", "http://a"));
    let mut persons = HashMap::new();
    persons.insert(
        "bob".into(),
        PersonConfig {
            provider: "typo".into(),
            model: "qwen3.5-2b".into(),
            persona: String::new(),
        },
    );
    cfg.persons = persons;
    let err = cfg.validate().expect_err("unknown provider must fail");
    let msg = err.to_string();
    assert!(msg.contains("persons.bob.provider"), "msg = {msg}");
    assert!(msg.contains("typo"), "msg = {msg}");
}

#[test]
fn validate_rejects_person_with_empty_model() {
    let mut cfg = SmosConfig::default();
    cfg.providers.push(ProviderConfig::new("p", "http://a"));
    let mut persons = HashMap::new();
    persons.insert(
        "bob".into(),
        PersonConfig {
            provider: "p".into(),
            model: String::new(),
            persona: String::new(),
        },
    );
    cfg.persons = persons;
    let err = cfg.validate().expect_err("empty model must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("persons.bob.model must not be empty"),
        "msg = {msg}"
    );
}

#[test]
fn validate_rejects_person_with_unsafe_name_at_startup() {
    // A person name with path-traversal characters fails MemoryKey
    // validation. The startup validator MUST catch this so the
    // operator hears about it before the first request hits the
    // routing layer.
    let mut cfg = SmosConfig::default();
    cfg.providers.push(ProviderConfig::new("p", "http://a"));
    let mut persons = HashMap::new();
    persons.insert(
        "a/b".into(),
        PersonConfig {
            provider: "p".into(),
            model: "qwen3.5-2b".into(),
            persona: String::new(),
        },
    );
    cfg.persons = persons;
    let err = cfg
        .validate()
        .expect_err("unsafe person name must fail at startup");
    let msg = err.to_string();
    assert!(
        msg.contains("invalid memory key"),
        "expected 'invalid memory key' in msg, got: {msg}"
    );
    assert!(msg.contains("a/b"), "expected person name in msg: {msg}");
}

#[test]
fn validate_accepts_person_referencing_known_provider() {
    let mut cfg = SmosConfig::default();
    cfg.providers.push(ProviderConfig::new("p", "http://a"));
    let mut persons = HashMap::new();
    persons.insert(
        "bob".into(),
        PersonConfig {
            provider: "p".into(),
            model: "qwen3.5-2b".into(),
            persona: String::new(),
        },
    );
    cfg.persons = persons;
    assert!(cfg.validate().is_ok(), "valid person must validate");
}

#[test]
fn validate_rejects_empty_reranker_url() {
    // The reranker is a hard dependency — an operator who blanks the URL
    // must get a startup error pointing at the field instead of
    // discovering the dependency via an HTTP 503 on the first request.
    let mut cfg = SmosConfig::default();
    cfg.reranker.url = String::new();
    cfg.providers.push(one_provider());
    let err = cfg.validate().expect_err("empty reranker url must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("reranker.url must not be empty"),
        "msg = {msg}"
    );
}

#[test]
fn validate_rejects_whitespace_only_reranker_url() {
    // `trim().is_empty()` catches whitespace-only strings so a typo like
    // `url = "   "` is treated identically to an empty string.
    let mut cfg = SmosConfig::default();
    cfg.reranker.url = "   ".into();
    cfg.providers.push(one_provider());
    let err = cfg
        .validate()
        .expect_err("whitespace-only reranker url must fail");
    assert!(err.to_string().contains("reranker.url must not be empty"));
}

#[test]
fn validate_collects_multiple_errors_in_one_message() {
    // Two unrelated problems: bad dimensions AND no providers. The
    // operator should see both in a single error so they can fix them
    // in one editing pass.
    let mut cfg = SmosConfig::default();
    cfg.embedding.dimensions = 768;
    // providers stays empty
    let err = cfg.validate().expect_err("multi-error case");
    let msg = err.to_string();
    assert!(msg.contains("embedding.dimensions"), "msg = {msg}");
    assert!(msg.contains("providers must not be empty"), "msg = {msg}");
    assert!(
        msg.contains(";"),
        "multiple errors joined by ';' in msg = {msg}"
    );
}

// --- AuditConfig behaviour -------------------------------------------

#[test]
fn audit_section_disabled_by_default() {
    let _g = _lock();
    let cfg = SmosConfig::default();
    assert!(!cfg.audit.enabled, "audit must be off by default");
    assert_eq!(cfg.audit.schedule, "0 3 * * *");
    assert_eq!(cfg.audit.llm_provider, "cloud");
    assert_eq!(cfg.audit.max_deletions_per_run, 50);
    assert_eq!(cfg.audit.max_merges_per_run, 100);
}

#[test]
fn audit_validation_skipped_when_disabled() {
    // Audit off => bad provider string does NOT fail validation. The
    // audit is opt-in; a stale `llm_provider` typo in a deployment that
    // never enables the audit should not block server startup.
    let mut cfg = SmosConfig::default();
    cfg.audit.enabled = false;
    cfg.audit.llm_provider = "garbage".into();
    cfg.providers.push(one_provider());
    assert!(cfg.validate().is_ok(), "disabled audit must not validate");
}

#[test]
fn audit_validation_rejects_unknown_provider_when_enabled() {
    let mut cfg = SmosConfig::default();
    cfg.audit.enabled = true;
    cfg.audit.llm_provider = "garbage".into();
    cfg.providers.push(one_provider());
    let err = cfg.validate().expect_err("bad provider must fail");
    assert!(err.to_string().contains("audit.llm_provider"));
}

#[test]
fn audit_validation_rejects_empty_schedule_when_enabled() {
    let mut cfg = SmosConfig::default();
    cfg.audit.enabled = true;
    cfg.audit.schedule = "   ".into();
    cfg.providers.push(one_provider());
    let err = cfg.validate().expect_err("empty schedule must fail");
    assert!(err.to_string().contains("audit.schedule"));
}

#[test]
fn audit_section_roundtrips_through_serde_json() {
    let cfg = SmosConfig::default();
    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: SmosConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.audit.schedule, cfg.audit.schedule);
    assert_eq!(back.audit.cloud_model, cfg.audit.cloud_model);
    assert_eq!(
        back.audit.max_deletions_per_run,
        cfg.audit.max_deletions_per_run
    );
}
