//! `NativeNliClassifier` — `NliClassifier` port backed by ort + ONNX Runtime.
//!
//! Single-process native implementation. The classifier owns:
//!
//! - one ort [`Session`] built against the detected [`DeviceKind`];
//! - one [`Tokenizer`] loaded from the HF-cached `tokenizer.json`;
//! - a [`std::sync::Mutex`] guarding the session — `Session::run` takes
//!   `&mut self`, so concurrent classify calls (e.g. from the session watcher
//!   + the HTTP extractor) must serialise at the session level.
//!
//! The mutex is `std::sync::Mutex` rather than `tokio::sync::Mutex` because
//! ort inference is purely CPU-bound (or blocks on the EP's own queue) and
//! never crosses an await point while held — using the std variant avoids
//! forcing every caller onto the async runtime for a sub-50ms operation.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ort::session::Session;
use ort::value::Tensor;
use smos_application::errors::ProviderError;
use smos_application::ports::NliClassifier;
use smos_application::types::NliResult;
use smos_domain::NliScores;
use smos_domain::enums::NliLabel;
use tokenizers::Tokenizer;

use super::device::{self, DeviceKind};
use super::model_cache;

/// DeBERTa-v3 NLI label order in the model's output logits tensor.
///
/// All MoritzLaurer DeBERTa-v3 MNLI exports (the canonical SMOS NLI model
/// family) emit `[entailment, neutral, contradiction]`. Hard-coded because
/// the label order is part of the trained weights, not a runtime property —
/// pinning it here keeps a wrong-order ONNX export from silently flipping
/// the contradiction verdict on every fact.
const LOGIT_ENTAILMENT: usize = 0;
const LOGIT_NEUTRAL: usize = 1;
const LOGIT_CONTRADICTION: usize = 2;

/// Native NLI classifier over ort + ONNX Runtime.
///
/// ## Concurrency: serialized inference
///
/// `Session::run` requires `&mut self`, so all NLI classifications are
/// serialized on a single `Mutex`. Concurrent finalize for N sessions
/// executes inference sequentially.
///
/// This is a **deliberate trade-off** for single-process SMOS:
/// - NLI runs in background finalize, not per-request hot path
/// - Sequential inference is predictable and avoids GPU memory contention
/// - Micro-batching multiple facts into one inference pass is a future
///   scalability improvement (not yet implemented; today every fact
///   incurs one `Session::run`)
///
/// For high-throughput deployments, consider spawning multiple
/// `NativeNliClassifier` instances with round-robin dispatch.
pub struct NativeNliClassifier {
    // `Arc<Mutex<Session>>` because `Session::run` takes `&mut self`: every
    // concurrent classify call must serialise at the session level. The
    // `Arc` lets us hand a cheap owned clone to `spawn_blocking` (which
    // requires `'static + Send`) without leaving a borrowed reference tied
    // to `&self`'s lifetime.
    session: Arc<Mutex<Session>>,
    tokenizer: Tokenizer,
    device: DeviceKind,
}

impl NativeNliClassifier {
    /// Build a classifier against a HF-cached model.
    ///
    /// `model_id` is the HF Hub repo (e.g.
    /// `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`). `cache_dir`
    /// is the canonical flat layout populated by [`model_cache`]. Both the
    /// ONNX graph and `tokenizer.json` are downloaded on first use and reused
    /// thereafter.
    ///
    /// Heavy: the first call on a fresh cache downloads ~643 MB. The session
    /// build itself also takes 1–5 s (graph optimisation + EP initialisation);
    /// production wiring should construct the classifier once at startup, not
    /// per request.
    pub fn new(model_id: &str, cache_dir: PathBuf) -> Result<Self, ProviderError> {
        let cache: &Path = &cache_dir;
        let model_path = model_cache::ensure_model_cached(model_id, cache)
            .map_err(|e| ProviderError::Unavailable(e.to_string()))?;
        let tokenizer_path = model_cache::ensure_tokenizer_cached(model_id, cache)
            .map_err(|e| ProviderError::Unavailable(e.to_string()))?;

        let device = device::detect_device();
        tracing::info!(device = device.as_str(), "native NLI device detected");

        let model_str = model_path.to_str().ok_or_else(|| {
            ProviderError::Unavailable(format!(
                "model path is not valid UTF-8: {}",
                model_path.display()
            ))
        })?;
        let session = device::build_session(model_str, device)
            .map_err(|e| ProviderError::Unavailable(format!("ort session build: {e}")))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| ProviderError::Unavailable(format!("tokenizer load: {e}")))?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer,
            device,
        })
    }

    /// Expose the detected device for diagnostics and the doctor check.
    pub fn device(&self) -> DeviceKind {
        self.device
    }
}

impl NliClassifier for NativeNliClassifier {
    async fn classify(&self, premise: &str, hypothesis: &str) -> Result<NliResult, ProviderError> {
        // All four pipeline stages (tokenise, tensor build, ort run, extract)
        // are CPU-bound, sync operations. The trait method is async only so
        // the use case's `Send`-bounded consumers can call it; we still must
        // avoid blocking a tokio worker for ~30–300 ms of DeBERTa-v3
        // inference. `spawn_blocking` lifts the whole pipeline onto the
        // dedicated blocking-pool thread and surfaces the result via an
        // async wrapper.
        //
        // The closure must be `'static + Send` (per `spawn_blocking`'s
        // contract). The session Arc and tokenizer clone are owned, but
        // `premise` / `hypothesis` arrive as `&str` tied to the caller's
        // lifetime, so they must be promoted to owned `String`s before the
        // closure captures them. The strings are then re-borrowed inside
        // `run_inference` so the inference path keeps the cheap `&str`
        // signature the tokenizer expects.
        let session = self.session.clone();
        let tokenizer = self.tokenizer.clone();
        let premise_owned = premise.to_owned();
        let hypothesis_owned = hypothesis.to_owned();
        tokio::task::spawn_blocking(move || {
            run_inference(&session, &tokenizer, &premise_owned, &hypothesis_owned)
        })
        .await
        .map_err(|e| ProviderError::Unavailable(format!("native NLI worker task: {e}")))?
    }
}

/// Synchronous inference pipeline — runs on a `spawn_blocking` thread.
///
/// `&Mutex<Session>` and `Tokenizer` are both `Send + Sync`, so they cross
/// the blocking-pool boundary safely. Error mapping follows the
/// [`ProviderError`] semantics: transport/runtime failures map to
/// [`Unavailable`] (so the use case's graceful-degradation branch leaves the
/// fact pending); genuinely malformed model output maps to
/// [`InvalidResponse`].
///
/// [`Unavailable`]: ProviderError::Unavailable
/// [`InvalidResponse`]: ProviderError::InvalidResponse
fn run_inference(
    session: &Mutex<Session>,
    tokenizer: &Tokenizer,
    premise: &str,
    hypothesis: &str,
) -> Result<NliResult, ProviderError> {
    // 1. Tokenise the (premise, hypothesis) pair. `true` enables the
    //    tokenizer's truncation strategy so inputs longer than the model's
    //    max position embeddings (512 for DeBERTa-v3) do not raise — they
    //    are clipped to the configured max length. A tokenizer error is
    //    a runtime/transport failure, not a malformed model response.
    let encoding = tokenizer
        .encode((premise, hypothesis), true)
        .map_err(|e| ProviderError::Unavailable(format!("tokenize: {e}")))?;

    let input_ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();
    let token_type_ids = encoding.get_type_ids();

    let seq_len = input_ids.len();
    if seq_len == 0 {
        // Unreachable in practice (DeBERTa-v3 always emits CLS/SEP) but a
        // defensive guard costs nothing and prevents an empty-tensor panic
        // at the ort boundary if a future tokenizer change breaks the
        // invariant.
        return Err(ProviderError::InvalidResponse(
            "tokenized input is empty".into(),
        ));
    }

    // 2. Convert u32 token ids into the i64 tensors DeBERTa-v3 expects.
    //    tokenizers always emits dense contiguous slices, so a single
    //    map+collect is the cheapest path.
    let input_ids_i64: Vec<i64> = input_ids.iter().map(|&v| v as i64).collect();
    let attention_i64: Vec<i64> = attention_mask.iter().map(|&v| v as i64).collect();
    let token_type_i64: Vec<i64> = token_type_ids.iter().map(|&v| v as i64).collect();

    // 3. Build ort input tensors. Batch dimension is always 1 (one pair per
    //    call); `from_array` copies the Vec into the ONNX Runtime allocator's
    //    owned buffer. Tensor construction failures are runtime/allocator
    //    issues, not malformed model output.
    let input_ids_tensor =
        Tensor::from_array((vec![1_usize, seq_len], input_ids_i64.into_boxed_slice()))
            .map_err(|e| ProviderError::Unavailable(format!("input_ids tensor: {e}")))?;
    let attention_tensor =
        Tensor::from_array((vec![1_usize, seq_len], attention_i64.into_boxed_slice()))
            .map_err(|e| ProviderError::Unavailable(format!("attention_mask tensor: {e}")))?;
    let token_type_tensor =
        Tensor::from_array((vec![1_usize, seq_len], token_type_i64.into_boxed_slice()))
            .map_err(|e| ProviderError::Unavailable(format!("token_type_ids tensor: {e}")))?;

    let inputs = ort::inputs! {
        "input_ids" => input_ids_tensor,
        "attention_mask" => attention_tensor,
        "token_type_ids" => token_type_tensor,
    };

    // 4. Run inference. `Session::run` takes `&mut self`, so the lock is
    //    unavoidable; std::sync::Mutex is fine because the hold is bounded
    //    by inference latency and this whole function executes on a
    //    blocking-pool thread. Poisoned mutex = a prior call panicked
    //    mid-inference = the session state is suspect; surface it as
    //    Unavailable so the use case degrades gracefully instead of
    //    crashing the worker. A transient EP failure (GPU OOM, runtime
    //    hiccup) is also Unavailable — the model produced no response at
    //    all, which is transport-level, not malformed-payload.
    let mut session_guard = session
        .lock()
        .map_err(|_| ProviderError::Unavailable("ort session mutex poisoned".into()))?;
    let outputs = session_guard
        .run(inputs)
        .map_err(|e| ProviderError::Unavailable(format!("ort run: {e}")))?;

    // 5. Extract the [1, 3] logits tensor. `try_extract_tensor` returns a
    //    flat slice + shape view; indexing the three NLI slots avoids
    //    pulling in ndarray for one softmax computation. A missing
    //    `logits` output, a wrong dtype, or a too-short tensor IS a
    //    genuine `InvalidResponse` — the model returned something we
    //    cannot interpret.
    let logits_value = outputs
        .get("logits")
        .ok_or_else(|| ProviderError::InvalidResponse("model output missing 'logits'".into()))?;
    let (_shape, logits) = logits_value
        .try_extract_tensor::<f32>()
        .map_err(|e| ProviderError::InvalidResponse(format!("extract logits: {e}")))?;

    if logits.len() < 3 {
        return Err(ProviderError::InvalidResponse(format!(
            "logits tensor has fewer than 3 elements: {}",
            logits.len()
        )));
    }

    let entailment = logits[LOGIT_ENTAILMENT];
    let neutral = logits[LOGIT_NEUTRAL];
    let contradiction = logits[LOGIT_CONTRADICTION];

    let scores = softmax(entailment, neutral, contradiction);
    let label = argmax_label(&scores);

    Ok(NliResult {
        label,
        scores,
        available: true,
    })
}

/// Numerically stable softmax over the three NLI logits.
///
/// Subtracting the max logit before exponentiating prevents overflow when the
/// model emits a very confident verdict (e.g. entailment logit > 30); the
/// resulting distribution is unchanged because the max cancels in the
/// denominator. Pure function so the unit-test suite can verify the
/// `sum == 1.0` invariant without constructing an ort session.
fn softmax(entailment: f32, neutral: f32, contradiction: f32) -> NliScores {
    let max_logit = entailment.max(neutral).max(contradiction);
    let exp_e = (entailment - max_logit).exp();
    let exp_n = (neutral - max_logit).exp();
    let exp_c = (contradiction - max_logit).exp();
    let sum = exp_e + exp_n + exp_c;

    NliScores {
        entailment: exp_e / sum,
        neutral: exp_n / sum,
        contradiction: exp_c / sum,
    }
}

/// Pick the dominant NLI label given a softmax distribution. Ties resolve in
/// the order `Entailment > Neutral > Contradiction` — matches DeBERTa-v3
/// `argmax` semantics where the first slot wins a flat tie.
fn argmax_label(scores: &NliScores) -> NliLabel {
    if scores.entailment >= scores.neutral && scores.entailment >= scores.contradiction {
        NliLabel::Entailment
    } else if scores.neutral >= scores.contradiction {
        NliLabel::Neutral
    } else {
        NliLabel::Contradiction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_for_typical_logits() {
        let scores = softmax(2.0, 1.0, 0.5);
        let sum = scores.entailment + scores.neutral + scores.contradiction;
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn softmax_is_numerically_stable_on_extreme_logits() {
        // Without max-subtraction, exp(50) overflows to inf and the
        // distribution collapses to NaN. The guard keeps the dominant slot
        // at ~1.0 and the others at ~0.0.
        let scores = softmax(50.0, 0.0, -50.0);
        assert!(scores.entailment > 0.99);
        assert!(scores.neutral < 1e-6);
        assert!(scores.contradiction < 1e-6);
    }

    #[test]
    fn softmax_preserves_argmax_order() {
        let scores = softmax(0.1, 5.0, 0.2);
        assert!(scores.neutral > scores.entailment);
        assert!(scores.neutral > scores.contradiction);
    }

    #[test]
    fn argmax_label_picks_entailment_when_dominant() {
        let scores = NliScores {
            entailment: 0.8,
            neutral: 0.1,
            contradiction: 0.1,
        };
        assert_eq!(argmax_label(&scores), NliLabel::Entailment);
    }

    #[test]
    fn argmax_label_picks_contradiction_when_dominant() {
        let scores = NliScores {
            entailment: 0.05,
            neutral: 0.05,
            contradiction: 0.9,
        };
        assert_eq!(argmax_label(&scores), NliLabel::Contradiction);
    }

    #[test]
    fn argmax_label_breaks_tie_towards_entailment() {
        // DeBERTa's argmax contract: the first slot wins a flat tie.
        let scores = NliScores {
            entailment: 0.5,
            neutral: 0.5,
            contradiction: 0.0,
        };
        assert_eq!(argmax_label(&scores), NliLabel::Entailment);
    }

    #[test]
    fn argmax_label_breaks_tie_between_neutral_and_contradiction() {
        let scores = NliScores {
            entailment: 0.0,
            neutral: 0.5,
            contradiction: 0.5,
        };
        assert_eq!(argmax_label(&scores), NliLabel::Neutral);
    }

    #[test]
    fn logit_slot_constants_match_deberta_v3_label_order() {
        // The legacy Python NLI backend's LABEL_ORDER tuple hard-coded this
        // order; the native classifier must reproduce it exactly so the
        // persisted verdicts match across the migration.
        assert_eq!(LOGIT_ENTAILMENT, 0);
        assert_eq!(LOGIT_NEUTRAL, 1);
        assert_eq!(LOGIT_CONTRADICTION, 2);
    }
}
