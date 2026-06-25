use super::types::{ConfigError, PersonConfig, ProviderConfig, SmosConfig};
use crate::storage::surreal_schema::EMBEDDING_DIM;

// ---------------------------------------------------------------------------
// SmosConfig loading
// ---------------------------------------------------------------------------

impl SmosConfig {
    /// Validate every cross-field invariant and range bound in one pass.
    ///
    /// Returns `Ok(())` when every check passes; otherwise returns
    /// [`ConfigError::Validation`] carrying a `;`-joined list of every
    /// problem found so the operator can fix them all in one editing pass
    /// instead of discovering them one `smos serve` invocation at a time.
    ///
    /// The checks mirror the invariants the rest of the code already assumes:
    ///
    /// - `embedding.dimensions == 1024` — must match the HNSW index DDL.
    /// - `confidence.*` ranges + `accept_threshold >= pending_threshold`.
    /// - `extraction.dedup_cosine_threshold` in `[-1, 1]`.
    /// - `llm_extraction.temperature` in `[0, 2]`.
    /// - `session.timeout_seconds > 0`.
    /// - `server.port > 0`.
    /// - `retrieval.top_k_initial > 0` and `retrieval.top_k_final > 0`
    ///   (a zero would either short-circuit the pipeline or surface as a
    ///   mysterious HTTP 503 once the reranker is consulted).
    /// - `reranker.url` non-empty (reranker is a hard dependency: every
    ///   request fails with HTTP 503 while the URL is missing or the server
    ///   is unreachable; an operator who blanks the field gets a startup
    ///   error instead of a silent quality drop).
    /// - `providers` non-empty (the proxy needs at least one provider to
    ///   forward chat completions to) and every provider carries a non-empty
    ///   URL + non-zero timeout + a unique name.
    /// - Every `[persons.*].provider` MUST reference an existing
    ///   `[[providers]].name` (a typo would surface as a 503 on the first
    ///   request that uses the person; surfacing it at startup keeps the
    ///   failure mode loud and immediate).
    /// - `nli.contradiction_threshold` in `[0, 1]`.
    /// - `merge.cosine_threshold` in `[-1, 1]`.
    /// - `audit.*` semantic checks — only enforced when `audit.enabled = true`
    ///   (a disabled audit is opt-in; see [`SmosConfig::validate_audit_always`]
    ///   for the variant that checks audit fields regardless of the enabled
    ///   flag, used by `smos audit --provider` to catch typos before the run).
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut errors: Vec<String> = Vec::new();

        if self.embedding.dimensions != EMBEDDING_DIM {
            errors.push(format!(
                "embedding.dimensions must be {EMBEDDING_DIM} (HNSW index dimension), got {}",
                self.embedding.dimensions
            ));
        }

        if !(0.0..=1.0).contains(&self.confidence.base) {
            errors.push(format!(
                "confidence.base must be in [0,1], got {}",
                self.confidence.base
            ));
        }
        if !(0.0..=1.0).contains(&self.confidence.accept_threshold) {
            errors.push(format!(
                "confidence.accept_threshold must be in [0,1], got {}",
                self.confidence.accept_threshold
            ));
        }
        if !(0.0..=1.0).contains(&self.confidence.pending_threshold) {
            errors.push(format!(
                "confidence.pending_threshold must be in [0,1], got {}",
                self.confidence.pending_threshold
            ));
        }
        if self.confidence.accept_threshold < self.confidence.pending_threshold {
            errors.push(format!(
                "confidence.accept_threshold ({}) must be >= pending_threshold ({})",
                self.confidence.accept_threshold, self.confidence.pending_threshold
            ));
        }

        if !(-1.0..=1.0).contains(&self.extraction.dedup_cosine_threshold) {
            errors.push(format!(
                "extraction.dedup_cosine_threshold must be in [-1,1], got {}",
                self.extraction.dedup_cosine_threshold
            ));
        }

        if !(0.0..=2.0).contains(&self.llm_extraction.temperature) {
            errors.push(format!(
                "llm_extraction.temperature must be in [0,2], got {}",
                self.llm_extraction.temperature
            ));
        }

        if self.session.timeout_seconds == 0 {
            errors.push("session.timeout_seconds must be > 0".into());
        }

        if self.server.port == 0 {
            errors.push("server.port must be > 0".into());
        }

        if self.retrieval.top_k_final == 0 {
            // `top_k_final == 0` would make `RerankProvider::rerank` return
            // `Ok(vec![])` (the legitimate "nothing to do" path), which the
            // fail-closed enrich pipeline converts into
            // `ProviderError::InvalidResponse("reranker returned empty
            // results")` → every chat-completion request fails with HTTP
            // 503. Reject at startup so the operator hears about it as a
            // config error, not as a mysterious 503.
            errors.push("retrieval.top_k_final must be > 0".into());
        }

        if self.retrieval.top_k_initial == 0 {
            errors.push("retrieval.top_k_initial must be > 0".into());
        }

        if self.reranker.url.trim().is_empty() {
            errors.push(
                "reranker.url must not be empty — reranker is required for enrichment".into(),
            );
        }

        validate_providers(&self.providers, &mut errors);
        validate_persons(&self.persons, &self.providers, &mut errors);

        if !(0.0..=1.0).contains(&self.nli.contradiction_threshold) {
            errors.push(format!(
                "nli.contradiction_threshold must be in [0,1], got {}",
                self.nli.contradiction_threshold
            ));
        }

        if !(-1.0..=1.0).contains(&self.merge.cosine_threshold) {
            errors.push(format!(
                "merge.cosine_threshold must be in [-1,1], got {}",
                self.merge.cosine_threshold
            ));
        }

        if self.audit.enabled {
            errors.extend(self.validate_audit_fields());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Validation(errors.join("; ")))
        }
    }

    /// Validate the audit fields REGARDLESS of `audit.enabled`.
    ///
    /// Used by `smos audit` (the manual one-shot runner) so a typo in
    /// `cloud_base_url` or an unknown `llm_provider` is surfaced at the
    /// invocation rather than as a runtime error mid-audit. The full
    /// [`SmosConfig::validate`] only checks audit fields when
    /// `audit.enabled = true`, which is correct for `smos serve` (where
    /// the audit is off by default and a stale config should not block
    /// server startup) but wrong for the manual runner.
    pub fn validate_audit_always(&self) -> Result<(), ConfigError> {
        let errors = self.validate_audit_fields();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Validation(errors.join("; ")))
        }
    }

    /// Shared semantic checks for the audit section. Returns the (possibly
    /// empty) list of problems; the caller decides whether to fail or
    /// accumulate them into a wider validation pass.
    fn validate_audit_fields(&self) -> Vec<String> {
        let mut errors: Vec<String> = Vec::new();
        if self.audit.schedule.trim().is_empty() {
            errors.push("audit.schedule must not be empty when audit is enabled".into());
        }
        let provider = self.audit.llm_provider.as_str();
        if !matches!(provider, "cloud" | "local") {
            errors.push(format!(
                "audit.llm_provider must be 'cloud' or 'local', got {provider:?}"
            ));
        }
        if provider == "cloud" && self.audit.cloud_base_url.trim().is_empty() {
            errors.push("audit.cloud_base_url must not be empty for the cloud provider".into());
        }
        if provider == "local" && self.audit.local_url.trim().is_empty() {
            errors.push("audit.local_url must not be empty for the local provider".into());
        }
        if self.audit.max_tool_rounds == 0 {
            // `max_tool_rounds = 0` reproduces the rig 0.14 default
            // (`max_depth = 0`), which short-circuits the tool-calling loop
            // and makes the audit fail with `MaxDepthError` on the first
            // prompt that needs a tool. Reject at startup so the operator
            // hears about it as a config error, not as a mid-audit crash.
            errors.push("audit.max_tool_rounds must be > 0".into());
        }
        errors
    }
}
/// Validate the `[[providers]]` array in isolation.
///
/// Pushed out of [`SmosConfig::validate`] so the body of `validate` stays
/// readable. The same checks run whether the array came from TOML or from
/// `SmosConfig::default` + programmatic edits (which is how the E2E helpers
/// build a config).
fn validate_providers(providers: &[ProviderConfig], errors: &mut Vec<String>) {
    if providers.is_empty() {
        errors.push("providers must not be empty".into());
        return;
    }
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for (i, p) in providers.iter().enumerate() {
        if p.timeout_seconds == 0 {
            errors.push(format!("providers[{i}].timeout_seconds must be > 0"));
        }
        if p.url.is_empty() {
            errors.push(format!("providers[{i}].url must not be empty"));
        }
        if p.name.is_empty() {
            errors.push(format!("providers[{i}].name must not be empty"));
            continue;
        }
        if !seen.insert(p.name.as_str()) {
            errors.push(format!(
                "providers[{i}].name = {:?} is duplicated; provider names must be unique",
                p.name
            ));
        }
    }
}

/// Validate the `[persons.*]` map. Each person MUST:
/// - reference a provider that exists in the `[[providers]]` array,
/// - declare a non-empty upstream model,
/// - carry a name that is a valid `MemoryKey` (the name is used as the
///   memory namespace at runtime — `[persons."a/b"]` would 400 every
///   request because the router rejects path-traversal characters).
///
/// Surfacing these at startup matches the documented "loud and immediate"
/// validation philosophy (see [`SmosConfig::validate`]).
fn validate_persons(
    persons: &std::collections::HashMap<String, PersonConfig>,
    providers: &[ProviderConfig],
    errors: &mut Vec<String>,
) {
    let provider_names: std::collections::HashSet<&str> =
        providers.iter().map(|p| p.name.as_str()).collect();
    for (name, person) in persons {
        // Validate the person name as a MemoryKey. The router re-validates
        // per request (defence in depth against programmatic edits), but a
        // typo in the TOML key SHOULD surface at startup rather than as a
        // 400 on the first request.
        if let Err(e) = smos_domain::MemoryKey::from_raw(name) {
            errors.push(format!(
                "persons.{name:?}: invalid memory key — {e}. \
                 Person names MUST satisfy the MemoryKey rules (alphanumeric \
                 first char, no path separators, no '..')."
            ));
        }
        if !provider_names.contains(person.provider.as_str()) {
            errors.push(format!(
                "persons.{name}.provider = {:?} does not match any [[providers]].name; \
                 known providers: {{{}}}",
                person.provider,
                providers
                    .iter()
                    .map(|p| p.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if person.model.is_empty() {
            errors.push(format!("persons.{name}.model must not be empty"));
        }
    }
}
