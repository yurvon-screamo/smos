//! `FactOutcome` - per-fact resolution result of the finalize walk.
//!
//! Split out of `finalize_session.rs` (R8); behaviour-preserving. The enum
//! body is the verbatim original.

/// Per-fact resolution outcome. Internal to the use case; surfaced in
/// [`super::FinalizeStats`] via the `tally` step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FactOutcome {
    /// Pending fact was reclassified standalone (no candidate, or only neutral
    /// NLI verdicts). Status may be `Accepted` / `Pending` / `Rejected`
    /// depending on the validation gate.
    Finalized,
    /// Pending fact was merged into an existing accepted fact and the twin
    /// was marked `Rejected`.
    Merged,
    /// Pending fact drifted (contradiction) and was bidirectionally flagged
    /// against an existing fact. Status unchanged on both sides.
    Conflict,
    /// Pending fact could not be resolved (NLI unavailable, save failed, ...).
    /// Stays `Pending` for the next cycle. NOT tallied into any counter so
    /// operators can detect "facts stuck in pending" via `processed - (finalized
    /// + merged + conflicts)`.
    Skipped,
}
