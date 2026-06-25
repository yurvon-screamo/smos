# ADR-0004: ScanState encapsulation for the drift-priority walk

- **Status:** Accepted
- **Date:** 2026-06-25
- **Supersedes:** —

## Context

`FinalizeSession::resolve_one` is the drift-priority NLI walk described in
[`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) → "NLI pipeline". It iterates
every merge candidate for a pending fact and tracks four pieces of state
across the iteration:

- `merge_pick: Option<(Fact, NliResult)>` — the first entailment candidate,
  held until the scan completes contradiction-free.
- `last_observed_nli: Option<NliResult>` — feeds the
  `no_contradiction_bonus` on the standalone promotion path.
- `nli_observed: bool` — gates the bonus; an `available = false` NLI reply
  does not flip it.
- `pool: Vec<Fact>` — the growing comparison pool.

These were four loose `let mut` bindings at the top of `resolve_one`. The
function had cyclomatic complexity 15 and cognitive complexity 23, and the
drift-priority invariant — "first contradiction wins, entailment is held
until the scan is contradiction-free, an unavailable NLI reply does not
count as observed" — was enforced by four independent mutable variables
that a reader had to simulate in their head.

The walk is the most subtle algorithm in SMOS. A future edit that, say,
flips `nli_observed` at the wrong point silently breaks drift detection
and there is no compile-time signal.

## Decision

Encapsulate the four accumulators in a `ScanState` struct whose methods
are the **only** way to mutate them. The state transitions are 1:1 with
the original code — this is a behaviour-preserving refactor (AAD-5 in
`docs/REFACTOR_PLAN.md`), not an algorithm rewrite.

```rust
struct ScanState {
    merge_pick: Option<(Fact, NliResult)>,
    last_observed_nli: Option<NliResult>,
    nli_observed: bool,
    pool: Vec<Fact>,
}

impl ScanState {
    fn observe_verdict(&mut self, candidate: &Fact, verdict: &NliResult) { ... }
    fn commit_merge_pick(&mut self) -> Option<(Fact, NliResult)> { ... }
    // ...
}
```

`observe_verdict` encodes the three drift-priority rules in one place:

1. A `verdict.available == false` reply is a no-op (does not flip
   `nli_observed`, does not set `last_observed_nli`).
2. The first entailment becomes `merge_pick` but the scan continues.
3. A contradiction short-circuits the caller immediately (the method
   returns a signal the loop matches on).

The algorithm itself — candidate ordering, the C3 guard short-circuit, the
exact-match short-circuit, the standalone promotion carrying
`last_observed_nli` — is byte-for-byte the same. Only the accumulator
plumbing moved into methods.

To lock the behaviour, a **golden snapshot matrix** test
(`resolve_one_outcome_matrix_golden`) was added. It captures the
`FactOutcome` (variants `Finalized`, `Merged`, `Conflict`, `Skipped`) for
a fixed matrix of pending/candidate/NLI configurations, captured
**from the pre-refactor code** so the snapshot is the source of truth.
The test is ~8 rows and pins the drift-priority semantics independent of
the `ScanState` plumbing.

## Alternatives Considered

- **Leave the four mutable bindings in `resolve_one`.** Rejected: the
  cognitive complexity was already 23 and the drift-priority invariant
  was implicit in the order of mutations. The next edit to the walk
  (a new outcome variant, a tuned threshold) would have to re-derive the
  invariant from scratch.
- **Rewrite the walk.** Rejected: the drift-priority algorithm has
  non-obvious invariants (C3 guard, exact-match short-circuit,
  `available = false` does not count as observed — see ARCHITECTURE.md).
  A rewrite would lose the behaviour guarantee. The refactor target is
  readability, not a new algorithm.

## Consequences

- **Positive:** The drift-priority rules live in `observe_verdict` /
  `commit_merge_pick`, not scattered across a 60-line loop body. A reader
  sees the three rules as three branches in one method.
- **Positive:** The golden snapshot matrix pins the observable behaviour.
  A refactor that breaks drift detection fails the snapshot before it
  fails a downstream enrichment test.
- **Positive:** The existing `drift_priority_walk_*`,
  `multi_contradiction_*`, and `c3_guard_*` tests pass **without any body
  changes** — they exercise the public `resolve_one` API, which is
  unchanged. This is the behaviour-preservation signal.
- **Negative:** `ScanState` is a small struct with private fields and a
  handful of methods — one more type to read. The trade is that the four
  accumulators are no longer loose locals; they cannot drift out of sync
  with the rules.
- **Negative:** The state-transition methods are 1:1 with the original
  code today. If a future change genuinely needs a new state transition
  (e.g. a fifth accumulator), the `ScanState` API has to grow rather than
  a new `let mut` appearing in the function body. That is the point —
  growing the API is a visible, reviewed change.
