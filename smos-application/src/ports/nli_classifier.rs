//! `NliClassifier` port — natural-language-inference verdicts.
//!
//! Maps to the POC's DeBERTa classifier (`smos/nli.py`). Pure verdict
//! aggregation (`NliResult`) lives in `smos-domain::value_objects::nli`;
//! this trait only models the (premise, hypothesis) → scores call.
//!
//! # Why this trait is NOT dyn-compatible
//!
//! The trait uses native `async fn in trait` without an explicit `Send`
//! bound on the returned future. Native async-fn-in-trait futures borrow
//! from `&self`, which is fundamentally incompatible with the type-erasure
//! `dyn Trait` requires (the compiler cannot build a vtable for an
//! associated future type without `Send + 'static` bounds). This is a
//! deliberate architectural choice: it keeps the port free of
//! `Pin<Box<dyn Future>>` boilerplate and lets the adapter layer pick the
//! runtime per call site.
//!
//! Concrete consequence: consumers (`SessionWatcher`,
//! `FinalizeSession`) take a generic `NC: NliClassifier` parameter rather
//! than `Arc<dyn NliClassifier>`. The adapter layer exposes the concrete
//! backend (`NativeNliClassifier` in `smos-adapters::nli`) directly.

use smos_domain::NliResult;

use crate::errors::ProviderError;

/// NLI model boundary (DeBERTa-v3-large cross-encoder).
pub trait NliClassifier {
    /// Classify the relationship between `premise` and `hypothesis`.
    async fn classify(&self, premise: &str, hypothesis: &str) -> Result<NliResult, ProviderError>;
}
