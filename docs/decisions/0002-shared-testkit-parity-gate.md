# ADR-0002: Shared testkit with a parity gate

- **Status:** Accepted
- **Date:** 2026-06-25
- **Supersedes:** —

## Context

Three use-case test modules each carried their own copy of the same
in-memory test doubles:

- `InMemoryFacts` (a `FactRepository` fake backed by a `HashMap`).
- `InMemorySessions` (a `SessionRepository` fake).
- `ScriptedNliClassifier` (returns a pre-programmed `NliResult` sequence).
- `ScriptedExtractor` (returns a pre-programmed extraction result).
- `ConstantEmbedder` (returns the same embedding for every input).
- `RecordingEmbedder` (records every embed call for assertion).
- `FixedClock` (returns a pinned `Timestamp`).
- `NoOpDelay` (a `Delay` impl that does not sleep).

That was roughly 300 lines of duplicated test infrastructure, and the
pattern was viral: every new use case copied the fakes from a neighbour to
get its tests to compile, then drifted its own copy. Two copies of
`InMemoryFacts` had already diverged in their `save_pending` semantics — a
real signal that the duplication was rotting.

## Decision

Consolidate the eight in-memory fakes into
`smos_application::testkit`. The testkit is a normal (non-dev) module of
`smos-application` so that integration tests in the `smos` crate can import
it via `use smos_application::testkit::*;`.

The migration was gated by a **parity test**: 11 invariant tests covering
the fakes' observable behaviour were written against the testkit versions
**before** any use-case test module was migrated. They had to be green
before the per-module copies were deleted. This is AAD-1 in
`docs/REFACTOR_PLAN.md` (behaviour-preservation via baseline + parity tests).

Scope boundary: the testkit holds only the **application-layer** in-memory
fakes. The adapter-layer `MockNliClassifier` (lives in the `smos` crate,
exercises the `ort` wiring) is deliberately **not** in the testkit — it is
a different concern at a different layer.

## Alternatives Considered

- **Keep one copy per use-case module.** Rejected: the duplication was
  already diverging (`save_pending` semantics drift between two
  `InMemoryFacts` copies). The DRY violation grew linearly with every new
  use case.
- **A separate `smos-testkit` dev-only crate.** Rejected: overkill for eight
  fakes. It would add a fourth crate to the workspace, a new `Cargo.toml`
  to maintain, and a publish boundary that does not exist in production.
  The testkit is test-only in spirit but it lives inside
  `smos-application` next to the ports it imitates, which is where a reader
  looks for them.

## Consequences

- **Positive:** ~250 lines de-duplicated. One implementation of each fake;
  one place to fix a fake's behaviour.
- **Positive:** Unified fake behaviour across every use-case test. The
  drift between the two `InMemoryFacts` copies is gone — there is now one
  `save_pending` semantic.
- **Positive:** The parity-gate is a safety net for future fake edits.
  Changing a fake's behaviour without updating its invariant tests fails
  CI, which protects the use-case tests that depend on the fake.
- **Negative:** Use-case test modules now depend on a shared module. A
  genuinely use-case-specific fake still has to live in the use-case's own
  test module — the testkit is only for the cross-cutting doubles. The
  boundary is "if two use cases would copy it, it goes in the testkit".
- **Negative:** The testkit is a non-dev module, so it ships with the
  `smos-application` artifact. The fakes are cheap (no IO, no deps beyond
  what the crate already pulls) and are documented as test-only.
