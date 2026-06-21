//! Native NLI backend integration tests (ort + ONNX Runtime).
//!
//! Each test constructs a real [`NativeNliClassifier`] against the
//! DeBERTa-v3 ONNX export. The tests download ~643 MB on the first run and
//! reuse the local cache afterwards; subsequent runs only pay the
//! session-build cost. Marked `#[ignore]` so they stay out of the default
//! `cargo t` flow — run via `cargo tall` (or `cargo test -- --ignored`).

use std::path::PathBuf;
use std::sync::OnceLock;

use smos_adapters::NativeNliClassifier;
use smos_application::ports::NliClassifier;
use smos_domain::enums::NliLabel;

/// Canonical DeBERTa-v3 MNLI/FEVER/ANLI model id used by every other SMOS
/// NLI surface. Hard-coded (rather than read from `smos.toml`) so the test
/// is self-contained and survives a config edit that changes the default.
const MODEL_ID: &str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli";

/// Shared cache directory for the test binary. The first test to run pays
/// the download cost; every subsequent test reuses the cached artifacts.
/// A process-level [`OnceLock`] keeps a single [`TempDir`] alive for the
/// duration of the test binary — when the binary exits, `TempDir`'s `Drop`
/// cleans up the cache so the test does not leak ~643 MB onto the host.
///
/// Tests must NOT delete the directory manually; the cleanup is owned by
/// the `OnceLock`'s `TempDir`.
static SHARED_CACHE: OnceLock<tempfile::TempDir> = OnceLock::new();

fn cache_dir() -> PathBuf {
    let tmp = SHARED_CACHE.get_or_init(|| {
        tempfile::Builder::new()
            .prefix("smos_nli_test_")
            .tempdir()
            .expect("failed to create temp dir for native NLI test cache")
    });
    tmp.path().to_path_buf()
}

/// Construct the classifier, panicking on failure. The first invocation in
/// a CI run downloads the model; the panic message preserves the original
/// error so a flaky network failure surfaces verbatim in the log.
fn setup_classifier() -> NativeNliClassifier {
    let cache = cache_dir();
    NativeNliClassifier::new(MODEL_ID, cache).expect(
        "failed to construct NativeNliClassifier — check HF Hub connectivity \
         and that the cache directory is writable",
    )
}

#[tokio::test]
#[ignore = "requires 643MB DeBERTa ONNX model download"]
async fn native_nli_classifies_canonical_entailment() {
    let clf = setup_classifier();
    let result = clf
        .classify("A dog runs in the park.", "An animal moves outdoors.")
        .await
        .expect("classify must succeed for a canonical pair");

    assert!(result.available, "verdict must be marked available");
    assert_eq!(result.label, NliLabel::Entailment);
    assert!(
        result.scores.entailment > 0.7,
        "entailment score must dominate; got {:?}",
        result.scores
    );
}

#[tokio::test]
#[ignore = "requires 643MB DeBERTa ONNX model download"]
async fn native_nli_classifies_canonical_contradiction() {
    let clf = setup_classifier();
    let result = clf
        .classify("The sky is blue today.", "The sky is red today.")
        .await
        .expect("classify must succeed");

    assert!(result.available);
    assert_eq!(result.label, NliLabel::Contradiction);
    assert!(
        result.scores.contradiction > 0.5,
        "contradiction score must dominate; got {:?}",
        result.scores
    );
}

#[tokio::test]
#[ignore = "requires 643MB DeBERTa ONNX model download"]
async fn native_nli_classifies_canonical_neutral() {
    let clf = setup_classifier();
    let result = clf
        .classify("I like apples.", "The weather is sunny today.")
        .await
        .expect("classify must succeed");

    assert!(result.available);
    assert_eq!(result.label, NliLabel::Neutral);
}

#[tokio::test]
#[ignore = "requires 643MB DeBERTa ONNX model download"]
async fn native_nli_handles_long_input_without_panicking() {
    // DeBERTa-v3 max position embeddings = 512; the tokenizer truncates
    // when `truncation = true` is passed, so a pathological input must NOT
    // raise. The verdict quality on truncated input is intentionally NOT
    // asserted — truncation can legitimately shift the label.
    let clf = setup_classifier();
    let long_text = "word ".repeat(1000);
    let result = clf.classify(&long_text, "a short hypothesis").await;
    assert!(result.is_ok(), "long input must not raise");
}

#[tokio::test]
#[ignore = "requires 643MB DeBERTa ONNX model download"]
async fn native_nli_softmax_distribution_sums_to_one() {
    // Numerical-stability smoke check: every verdict's softmax distribution
    // must sum to 1.0 within f32 epsilon. The native backend's softmax is
    // max-subtracted, so even pathological logits stay in range.
    let clf = setup_classifier();
    let result = clf
        .classify("A man is eating.", "A person is consuming food.")
        .await
        .expect("classify must succeed");

    let sum = result.scores.entailment + result.scores.neutral + result.scores.contradiction;
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax must sum to 1.0 within f32 epsilon; got {sum}"
    );
}
