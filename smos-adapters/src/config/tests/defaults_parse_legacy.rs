use super::*;

#[test]
fn default_has_canonical_values() {
    let _g = _lock();
    let cfg = SmosConfig::default();
    assert_eq!(cfg.server.port, 8888);
    assert_eq!(cfg.server.host, "127.0.0.1");
    assert!(cfg.providers.is_empty());
    assert!(cfg.persons.is_empty());
    assert_eq!(cfg.surreal.namespace, "smos");
    assert_eq!(cfg.nli.contradiction_threshold, 0.5);
    assert_eq!(cfg.nli.entailment_threshold, 0.6);
    assert!(cfg.nli_backend.model.starts_with("MoritzLaurer/"));
    assert_eq!(cfg.llm_extraction.model, "qwen3.5-2b");
    assert_eq!(cfg.llm_extraction.seed, 42);
    assert_eq!(cfg.embedding.dimensions, 1024);
}

/// The surreal + nli_backend defaults now anchor on `~/.smos` (or
/// `SMOS_HOME` when set). Pinned so a refactor that drops the
/// `SmosPaths::resolve` call from the default impl does not silently
/// regress to the legacy `./data` path.
#[test]
fn default_paths_are_anchored_on_smos_home() {
    let _g = _lock();
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let prior = std::env::var("SMOS_HOME").ok();
    // SAFETY: this test holds `CONFIG_TEST_LOCK`, serialising every
    // config test in this binary.
    unsafe {
        std::env::set_var("SMOS_HOME", tmp.path());
    }
    let cfg = SmosConfig::default();
    unsafe {
        match prior {
            Some(v) => std::env::set_var("SMOS_HOME", v),
            None => std::env::remove_var("SMOS_HOME"),
        }
    }
    assert!(
        cfg.surreal
            .path
            .starts_with(tmp.path().to_string_lossy().as_ref()),
        "expected surreal.path under SMOS_HOME, got {}",
        cfg.surreal.path
    );
    assert!(
        cfg.nli_backend
            .cache_dir
            .starts_with(tmp.path().to_string_lossy().as_ref()),
        "expected nli_backend.cache_dir under SMOS_HOME, got {}",
        cfg.nli_backend.cache_dir
    );
}

#[test]
fn load_missing_file_falls_back_to_defaults_then_fails_validation_on_empty_providers() {
    // Defaults parse fine when the file is missing, but `load()` runs
    // `validate()` after parsing. The default config has `providers = []`,
    // which violates the "must not be empty" rule — so the operator-
    // facing result is a clear Validation error that points at the
    // missing providers rather than a silent zero-providers state that
    // would only surface at the first request.
    let _g = _lock();
    let result = SmosConfig::load("definitely-does-not-exist.toml");
    let err = result.expect_err("defaults without providers must fail validation");
    let msg = err.to_string();
    assert!(
        msg.contains("providers must not be empty"),
        "expected validation message about empty providers, got: {msg}"
    );
}

#[test]
fn load_partial_file_fills_missing_sections_from_defaults() {
    let _g = _lock();
    let tmp = tempfile::Builder::new()
        .suffix(".toml")
        .tempfile()
        .expect("tempfile");
    // Include a provider so validation passes — the test is about
    // section-merging, not about provider semantics.
    std::fs::write(
        tmp.path(),
        "[server]\nhost = \"0.0.0.0\"\nport = 9999\n\
             [[providers]]\nname = \"u\"\nurl = \"http://u\"\ntimeout_seconds = 9\n",
    )
    .expect("write");
    let cfg = SmosConfig::load(tmp.path().to_str().unwrap()).expect("parse + validate");
    assert_eq!(cfg.server.host, "0.0.0.0");
    assert_eq!(cfg.server.port, 9999);
    assert_eq!(cfg.surreal.namespace, "smos");
}

#[test]
fn load_full_file_overrides_all_sections() {
    let _g = _lock();
    // `embedding.dimensions` MUST be 1024 (HNSW index dimension) — the
    // validation gate rejects any other value at startup.
    let toml = "[surreal]\npath = \"./x.db\"\nnamespace = \"ns\"\ndatabase = \"db\"\n\
                    [server]\nhost = \"h\"\nport = 1\nshutdown_extraction_grace_seconds = 5\n\
                    enable_response_extraction = false\ngraceful_degradation = false\nlog_format = \"pretty\"\n\
                    [[providers]]\nname = \"u\"\nurl = \"u\"\napi_key_env = \"SMOS_KEY\"\nauth_header = \"api-key\"\ntimeout_seconds = 9\n\
                    [llm_extraction]\nurl = \"http://llm:28082\"\nmodel = \"qwen\"\ntimeout_seconds = 11\n\
                    temperature = 0.2\nseed = 7\n\
                    [embedding]\nurl = \"http://embed:28081\"\nmodel = \"jina\"\ndimensions = 1024\ntimeout_seconds = 11\n\
                    [reranker]\nurl = \"http://reranker:28181\"\nmodel = \"rr\"\ntimeout_seconds = 7\n\
                    [retrieval]\ntop_k_initial = 30\ntop_k_final = 3\nmin_confidence = 0.6\nmin_topic_chars = 2\n\
                    [merge]\ncosine_threshold = 0.8\n\
                    [confidence]\nbase = 0.4\nmulti_source_bonus = 0.1\nno_contradiction_bonus = 0.05\naccept_threshold = 0.65\npending_threshold = 0.3\n\
                    [heat]\ndecay_rate = 0.02\nmin_threshold = 0.15\n\
                    [nli]\ncontradiction_threshold = 0.55\nentailment_threshold = 0.65\n\
                    [nli_backend]\nmodel = \"cross-encoder/nli-deberta-v3\"\ncache_dir = \"/var/cache/smos/nli\"\n\
                    [extraction]\ndedup_cosine_threshold = 0.92\n\
                    [session]\ntimeout_seconds = 600\npending_overflow_threshold = 15\nscan_interval_seconds = 30\n";
    let tmp = tempfile::Builder::new()
        .suffix(".toml")
        .tempfile()
        .expect("tempfile");
    std::fs::write(tmp.path(), toml).expect("write");
    let cfg = SmosConfig::load(tmp.path().to_str().unwrap()).expect("parse + validate");
    assert_eq!(cfg.server.host, "h");
    assert_eq!(cfg.server.port, 1);
    assert!(!cfg.server.enable_response_extraction);
    assert_eq!(cfg.server.log_format, "pretty");
    assert_eq!(cfg.providers.len(), 1);
    assert_eq!(cfg.providers[0].auth_header, "api-key");
    assert_eq!(cfg.providers[0].timeout_seconds, 9);
    assert_eq!(cfg.providers[0].api_key_env, "SMOS_KEY");
    assert_eq!(cfg.surreal.path, "./x.db");
    assert_eq!(cfg.llm_extraction.url, "http://llm:28082");
    assert_eq!(cfg.llm_extraction.model, "qwen");
    assert_eq!(cfg.llm_extraction.timeout_seconds, 11);
    assert_eq!(cfg.llm_extraction.seed, 7);
    assert_eq!(cfg.llm_extraction.temperature, 0.2);
    assert_eq!(cfg.embedding.url, "http://embed:28081");
    assert_eq!(cfg.embedding.model, "jina");
    assert_eq!(cfg.embedding.dimensions, 1024);
    assert_eq!(cfg.reranker.url, "http://reranker:28181");
    assert_eq!(cfg.reranker.model, "rr");
    assert_eq!(cfg.reranker.timeout_seconds, 7);
    assert_eq!(cfg.retrieval.top_k_initial, 30);
    assert_eq!(cfg.retrieval.top_k_final, 3);
    assert_eq!(cfg.merge.cosine_threshold, 0.8);
    assert_eq!(cfg.confidence.accept_threshold, 0.65);
    assert_eq!(cfg.heat.min_threshold, 0.15);
    assert_eq!(cfg.nli.contradiction_threshold, 0.55);
    assert_eq!(cfg.nli.entailment_threshold, 0.65);
    assert_eq!(cfg.nli_backend.model, "cross-encoder/nli-deberta-v3");
    assert_eq!(cfg.nli_backend.cache_dir, "/var/cache/smos/nli");
    assert_eq!(cfg.extraction.dedup_cosine_threshold, 0.92);
    assert_eq!(cfg.session.timeout_seconds, 600);
    assert_eq!(cfg.session.pending_overflow_threshold, 15);
    assert_eq!(cfg.session.scan_interval_seconds, 30);
}

#[test]
fn new_sections_default_when_omitted_from_partial_file() {
    let _g = _lock();
    let tmp = tempfile::Builder::new()
        .suffix(".toml")
        .tempfile()
        .expect("tempfile");
    // Add a provider so validation passes; the test verifies that the
    // sections OMITTED from the partial file fall back to defaults.
    std::fs::write(
        tmp.path(),
        "[server]\nport = 7777\n\
             [[providers]]\nname = \"u\"\nurl = \"http://u\"\ntimeout_seconds = 9\n",
    )
    .expect("write");
    let cfg = SmosConfig::load(tmp.path().to_str().unwrap()).expect("parse + validate");
    assert_eq!(cfg.server.port, 7777);
    assert_eq!(cfg.llm_extraction.timeout_seconds, 30);
    assert!(cfg.embedding.model.starts_with("hf.co/jinaai"));
    assert_eq!(cfg.reranker.model, "qwen3-reranker");
    assert_eq!(cfg.retrieval.top_k_final, 5);
    assert_eq!(cfg.session.pending_overflow_threshold, 20);
}

#[test]
fn config_roundtrips_through_serde_json() {
    let _g = _lock();
    let cfg = SmosConfig::default();
    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: SmosConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.server.port, cfg.server.port);
    assert_eq!(back.providers.len(), cfg.providers.len());
}

// --- Provider / person parsing -------------------------------------

/// The canonical `[[providers]]` + `[persons.X]` shape parses into the
/// two collections the routing layer expects.
#[test]
fn providers_and_persons_section_parses() {
    let _g = _lock();
    let toml = "[[providers]]\n\
                    name = \"llama-local\"\n\
                    url = \"http://localhost:28082/v1/chat/completions\"\n\
                    api_key_env = \"\"\n\
                    auth_header = \"Authorization\"\n\
                    timeout_seconds = 120\n\
                    [[providers]]\n\
                    name = \"openrouter\"\n\
                    url = \"https://openrouter.ai/api/v1/chat/completions\"\n\
                    api_key_env = \"OPENROUTER_API_KEY\"\n\
                    timeout_seconds = 90\n\
                    [persons.bob]\n\
                    provider = \"llama-local\"\n\
                    model = \"qwen3.5-2b\"\n\
                    persona = \"~/.smos/persons/bob.md\"\n\
                    [persons.alice]\n\
                    provider = \"openrouter\"\n\
                    model = \"z-ai/glm-5.2\"\n";
    let tmp = tempfile::Builder::new()
        .suffix(".toml")
        .tempfile()
        .expect("tempfile");
    std::fs::write(tmp.path(), toml).expect("write");
    let cfg = SmosConfig::load(tmp.path().to_str().unwrap()).expect("parse + validate");
    assert_eq!(cfg.providers.len(), 2);
    assert_eq!(cfg.providers[0].name, "llama-local");
    assert_eq!(cfg.providers[1].name, "openrouter");
    assert_eq!(cfg.providers[1].api_key_env, "OPENROUTER_API_KEY");
    // Second provider inherits the default `auth_header` since the TOML
    // omits it.
    assert_eq!(cfg.providers[1].auth_header, "Authorization");

    let bob = cfg.persons.get("bob").expect("person bob");
    assert_eq!(bob.provider, "llama-local");
    assert_eq!(bob.model, "qwen3.5-2b");
    assert_eq!(bob.persona, "~/.smos/persons/bob.md");

    let alice = cfg.persons.get("alice").expect("person alice");
    assert_eq!(alice.provider, "openrouter");
    assert_eq!(alice.model, "z-ai/glm-5.2");
    assert_eq!(alice.persona, "", "persona defaults to empty");
}

/// `ProviderConfig::resolve_api_key` reads the env var named in
/// `api_key_env`. Empty `api_key_env` MUST yield an empty string (the
/// "no auth" case for a local `llama-server`) instead of consulting any
/// default env var.
#[test]
fn resolve_api_key_reads_named_env_var() {
    let _g = _lock();
    let prior = std::env::var("SMOS_TEST_PROVIDER_KEY").ok();
    // SAFETY: this test holds `CONFIG_TEST_LOCK`.
    unsafe {
        std::env::set_var("SMOS_TEST_PROVIDER_KEY", "sk-from-env");
    }
    let provider = ProviderConfig {
        name: "p".into(),
        url: "http://p".into(),
        api_key_env: "SMOS_TEST_PROVIDER_KEY".into(),
        auth_header: "Authorization".into(),
        timeout_seconds: 9,
    };
    assert_eq!(provider.resolve_api_key(), "sk-from-env");
    // SAFETY: same serialisation guarantee.
    unsafe {
        match prior {
            Some(v) => std::env::set_var("SMOS_TEST_PROVIDER_KEY", v),
            None => std::env::remove_var("SMOS_TEST_PROVIDER_KEY"),
        }
    }

    // Empty api_key_env MUST short-circuit to empty string.
    let unauth = ProviderConfig {
        api_key_env: String::new(),
        ..provider
    };
    assert_eq!(unauth.resolve_api_key(), "");
}

// --- Legacy section guards -----------------------------------------
//
// These tests pin the intentional behaviour: a TOML carrying legacy
// sections/fields still LOADS (serde has no `deny_unknown_fields`) but
// the legacy values NEVER affect the canonical config. A future engineer
// who re-adds a bridge will break one of these tests, which is the
// point — the intent is documented in code, not just in commit history.

/// A leftover unknown section (e.g. a historical `[ollama]` block from a
/// pre-llama.cpp config) does NOT populate `[llm_extraction]` /
/// `[embedding]`. The legacy fields are silently dropped at deserialize
/// time and the canonical sections keep their defaults.
#[test]
fn legacy_unknown_section_does_not_bridge_into_canonical_sections() {
    let _g = _lock();
    let toml = "[ollama]\n\
                    url = \"http://legacy:11434\"\n\
                    embedding_model = \"legacy-embed\"\n\
                    extraction_model = \"legacy-extract\"\n\
                    timeout_seconds = 17\n\
                    [[providers]]\nname = \"u\"\nurl = \"http://u\"\ntimeout_seconds = 9\n";
    let tmp = tempfile::Builder::new()
        .suffix(".toml")
        .tempfile()
        .expect("tempfile");
    std::fs::write(tmp.path(), toml).expect("write");
    let cfg = SmosConfig::load(tmp.path().to_str().unwrap()).expect("parse + validate");
    // Defaults preserved — legacy fields did NOT bleed through.
    assert_eq!(cfg.llm_extraction.url, "http://localhost:28082");
    assert_eq!(cfg.llm_extraction.model, "qwen3.5-2b");
    assert_eq!(cfg.llm_extraction.timeout_seconds, 30);
    assert!(cfg.embedding.model.starts_with("hf.co/jinaai"));
    assert_eq!(cfg.embedding.timeout_seconds, 30);
}

/// `[nli_backend]` is the CANONICAL adapter-side section (carrying
/// `model` + `cache_dir`); the domain-side `[nli]` section now holds
/// only verdict thresholds. This test pins the layering invariant: an
/// operator-supplied `[nli_backend]` populates `cfg.nli_backend`, and
/// `cfg.nli` (the domain thresholds) stays at its defaults unless the
/// operator also overrides `[nli]`.
#[test]
fn nli_backend_section_is_canonical_and_does_not_touch_domain_thresholds() {
    let _g = _lock();
    let toml = "[nli_backend]\n\
                    model = \"cross-encoder/nli-deberta-v3\"\n\
                    cache_dir = \"/var/cache/smos/nli\"\n\
                    [[providers]]\nname = \"u\"\nurl = \"http://u\"\ntimeout_seconds = 9\n";
    let tmp = tempfile::Builder::new()
        .suffix(".toml")
        .tempfile()
        .expect("tempfile");
    std::fs::write(tmp.path(), toml).expect("write");
    let cfg = SmosConfig::load(tmp.path().to_str().unwrap()).expect("parse + validate");
    // Adapter section picked up the override.
    assert_eq!(cfg.nli_backend.model, "cross-encoder/nli-deberta-v3");
    assert_eq!(cfg.nli_backend.cache_dir, "/var/cache/smos/nli");
    // Domain thresholds stayed at their defaults — the layering
    // invariant is intact.
    assert_eq!(cfg.nli.contradiction_threshold, 0.5);
    assert_eq!(cfg.nli.entailment_threshold, 0.6);
}

/// Putting `model` (an adapter-only field) under `[nli]` MUST fail loudly
/// at startup. `NliConfig` carries `#[serde(deny_unknown_fields)]` so the
/// parser rejects the misplacement instead of silently dropping it.
#[test]
fn nli_section_with_adapter_field_fails_loudly() {
    let _g = _lock();
    let toml = "[nli]\n\
                    contradiction_threshold = 0.5\n\
                    entailment_threshold = 0.6\n\
                    model = \"accidental-misplacement\"\n";
    let tmp = tempfile::Builder::new()
        .suffix(".toml")
        .tempfile()
        .expect("tempfile");
    std::fs::write(tmp.path(), toml).expect("write");
    let result = SmosConfig::load(tmp.path().to_str().unwrap());
    assert!(
        result.is_err(),
        "operator misplacing `model` under `[nli]` must fail loudly, not silently drop"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("model") && err_msg.contains("unknown"),
        "error must identify the unknown field; got: {err_msg}"
    );
}

/// Symmetric loud-failure for the adapter side: an unknown field under
/// `[nli_backend]` MUST fail loudly. `NliBackendConfig` carries the same
/// `#[serde(deny_unknown_fields)]` so a typo (`modle = "..."`) does not
/// silently fall back to the default model.
#[test]
fn nli_backend_section_with_unknown_field_fails_loudly() {
    let _g = _lock();
    let toml = "[nli_backend]\n\
                    modle = \"typo-for-model\"\n\
                    cache_dir = \"./data/nli_cache\"\n";
    let tmp = tempfile::Builder::new()
        .suffix(".toml")
        .tempfile()
        .expect("tempfile");
    std::fs::write(tmp.path(), toml).expect("write");
    let result = SmosConfig::load(tmp.path().to_str().unwrap());
    assert!(
        result.is_err(),
        "typo in `[nli_backend]` must fail loudly, not silently fall back to defaults"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("modle") && err_msg.contains("unknown"),
        "error must identify the unknown field; got: {err_msg}"
    );
}

/// A leftover `[nli_sidecar]` section (Python sidecar, removed) does
/// NOT abort startup and does NOT populate any field. Pinned so a
/// future change that re-introduces sidecar parsing breaks this test.
#[test]
fn legacy_nli_sidecar_section_is_silently_ignored() {
    let _g = _lock();
    let toml = "[nli_sidecar]\n\
                    python = \"python\"\n\
                    script = \"x.py\"\n\
                    cache_dir = \"./legacy\"\n\
                    [[providers]]\nname = \"u\"\nurl = \"http://u\"\ntimeout_seconds = 9\n";
    let tmp = tempfile::Builder::new()
        .suffix(".toml")
        .tempfile()
        .expect("tempfile");
    std::fs::write(tmp.path(), toml).expect("write");
    let cfg = SmosConfig::load(tmp.path().to_str().unwrap()).expect("parse + validate");
    // The default cache_dir is anchored on SMOS_HOME; the legacy value
    // `./legacy` did NOT bleed through.
    assert_ne!(cfg.nli_backend.cache_dir, "./legacy");
}

/// The legacy `[[upstream.providers]]` array (now replaced by
/// `[[providers]]`) does NOT populate `cfg.providers`. The fields are
/// silently dropped at deserialize time and `cfg.providers` stays empty,
/// which the validator flags with the canonical "providers must not be
/// empty" error.
#[test]
fn legacy_upstream_providers_section_does_not_bridge_into_providers() {
    let _g = _lock();
    let toml = "[[upstream.providers]]\n\
                    name = \"legacy\"\n\
                    url = \"http://legacy\"\n\
                    api_key = \"legacy\"\n";
    let tmp = tempfile::Builder::new()
        .suffix(".toml")
        .tempfile()
        .expect("tempfile");
    std::fs::write(tmp.path(), toml).expect("write");
    let result = SmosConfig::load(tmp.path().to_str().unwrap());
    let err = result.expect_err("legacy [[upstream.providers]] must NOT bridge");
    let msg = err.to_string();
    assert!(
        msg.contains("providers must not be empty"),
        "expected validation to flag empty providers (proof that no bridge \
             happened); got: {msg}"
    );
}
