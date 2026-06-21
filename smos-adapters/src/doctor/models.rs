//! Pure helpers for Ollama model availability matching.
//!
//! The doctor checks three required models (upstream chat model, embedding
//! model, extraction model) against the list returned by `GET /api/tags`.
//! The match logic is factored out of the IO path so unit tests can verify
//! the partial-match heuristic without spinning up Ollama.
//!
//! # Match heuristic
//!
//! Ollama `model` strings are case-sensitive record ids (`hf.co/...:latest`).
//! The doctor never knows the exact GGUF tag ahead of time — operators pull
//! variants (`Q8_0`, `F16`, …) — so the matcher treats two strings as
//! equivalent when one is a prefix of the other up to the first `:` tag
//! separator, OR when the `hf.co/<publisher>/<model>` segment matches
//! ignoring the quantisation suffix.

/// One row in the required-models table.
#[derive(Debug, Clone)]
pub struct ExpectedModel {
    /// Human-friendly label for the doctor output (`upstream chat model`).
    pub role: &'static str,
    /// Model id configured in `smos.toml` (e.g.
    /// `hf.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:latest`).
    pub configured: String,
}

impl ExpectedModel {
    pub fn new(role: &'static str, configured: impl Into<String>) -> Self {
        Self {
            role,
            configured: configured.into(),
        }
    }
}

/// Decide whether `available` satisfies `expected`. Used by the doctor and
/// by the unit tests; the heuristic lives in one place so adding a new
/// matching rule is a one-line change.
pub fn model_matches(expected: &str, available: &str) -> bool {
    if expected.eq_ignore_ascii_case(available) {
        return true;
    }
    let av_norm = strip_tag(available);
    let ex_norm = strip_tag(expected);
    if ex_norm.eq_ignore_ascii_case(av_norm) {
        return true;
    }
    // HuggingFace-style `hf.co/<publisher>/<repo>` ids: compare the trailing
    // repo segment so `:Q8_0` vs `:latest` and `<repo>-GGUF:latest` vs
    // `<repo>:F16` do not produce false negatives.
    let av_repo = last_segment(av_norm);
    let ex_repo = last_segment(ex_norm);
    !av_repo.is_empty() && av_repo.eq_ignore_ascii_case(ex_repo)
}

/// Strip the Ollama tag suffix (`:latest`, `:Q8_0`, …) if present. The model
/// id is split at the FIRST `:` so `hf.co/...` paths are preserved.
fn strip_tag(id: &str) -> &str {
    match id.find(':') {
        Some(idx) => &id[..idx],
        None => id,
    }
}

/// Return the segment after the last `/` (the HuggingFace repo name) so
/// `hf.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:latest` reduces
/// to `jina-embeddings-v5-text-small-retrieval-GGUF`. Returns the input
/// unchanged when no `/` is present.
fn last_segment(id: &str) -> &str {
    match id.rfind('/') {
        Some(idx) => &id[idx + 1..],
        None => id,
    }
}

/// Compute the per-model `CheckResult` rows from a configured expectation set
/// and a list of models that Ollama actually returned. Pure — the IO layer
/// (the doctor binary) supplies the inputs.
pub fn match_expected_models<'a>(
    expected: &'a [ExpectedModel],
    available: &'a [String],
) -> Vec<(ExpectedModel, bool)> {
    expected
        .iter()
        .map(|e| {
            let hit = available.iter().any(|a| model_matches(&e.configured, a));
            (e.clone(), hit)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_id_match_case_insensitive() {
        assert!(model_matches("granite4.1:3b", "granite4.1:3b"));
        assert!(model_matches("Granite4.1:3B", "granite4.1:3b"));
    }

    #[test]
    fn tag_suffix_difference_still_matches() {
        assert!(model_matches(
            "hf.co/jinaai/jina-embeddings-v5:latest",
            "hf.co/jinaai/jina-embeddings-v5:Q8_0"
        ));
    }

    #[test]
    fn hf_repo_name_match_overrides_quantisation_suffix() {
        // Operator pulled `...-GGUF:latest` while the config wants
        // `...-retrieval-GGUF:latest` — the repo segment differs, so this
        // must NOT match (different models).
        assert!(!model_matches(
            "hf.co/jinaai/jina-embeddings-v5-retrieval-GGUF:latest",
            "hf.co/jinaai/jina-embeddings-v5-GGUF:latest"
        ));
        // Same repo segment, different quantisation tag → matches.
        assert!(model_matches(
            "hf.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:latest",
            "hf.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:Q8_0"
        ));
    }

    #[test]
    fn different_publisher_does_not_match() {
        assert!(!model_matches(
            "hf.co/jinaai/jina-embeddings-v5:latest",
            "hf.co/openai/text-embedding-3:latest"
        ));
    }

    #[test]
    fn empty_available_list_matches_nothing() {
        let expected = vec![
            ExpectedModel::new("upstream", "granite4.1:3b"),
            ExpectedModel::new("embedding", "hf.co/jinaai/jina-embeddings-v5:latest"),
        ];
        let out = match_expected_models(&expected, &[]);
        assert_eq!(out.len(), 2);
        assert!(!out[0].1);
        assert!(!out[1].1);
    }

    #[test]
    fn full_expected_set_matches_all_available() {
        let expected = vec![
            ExpectedModel::new("upstream", "granite4.1:3b"),
            ExpectedModel::new("embedding", "hf.co/jinaai/jina-embeddings-v5:latest"),
            ExpectedModel::new("extraction", "qwen3.5:2b"),
        ];
        let available = vec![
            "granite4.1:3b".to_string(),
            "hf.co/jinaai/jina-embeddings-v5:latest".to_string(),
            "qwen3.5:2b".to_string(),
            "unused-model:7b".to_string(),
        ];
        let out = match_expected_models(&expected, &available);
        assert!(out.iter().all(|(_, hit)| *hit));
    }

    #[test]
    fn partial_match_reports_missing_models() {
        let expected = vec![
            ExpectedModel::new("upstream", "granite4.1:3b"),
            ExpectedModel::new("extraction", "qwen3.5:2b"),
        ];
        let available = vec!["granite4.1:3b".to_string()];
        let out = match_expected_models(&expected, &available);
        assert!(out[0].1);
        assert!(!out[1].1);
    }

    #[test]
    fn last_segment_handles_no_slash() {
        // `last_segment` does NOT strip the tag; the matcher pipeline runs
        // `strip_tag` first. Verifying the raw helper keeps that boundary
        // intact so future callers don't accidentally re-strip.
        assert_eq!(last_segment("granite4.1:3b"), "granite4.1:3b");
        assert_eq!(
            last_segment("hf.co/jinaai/jina-embeddings-v5:latest"),
            "jina-embeddings-v5:latest"
        );
    }
}
