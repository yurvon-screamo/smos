# ADR-0001: Record-struct constructors for Fact / SessionState

- **Status:** Accepted
- **Date:** 2026-06-25
- **Supersedes:** —

## Context

`Fact` and `SessionState` aggregates were reconstructed from persistence via
positional constructors:

- `Fact::rehydrate(...)` — 14 positional arguments.
- `Fact::new_pending(...)` — 6 positional arguments.
- `SessionState::rehydrate(...)` — 6 positional arguments.

Several arguments share the same type (`Timestamp`, `Confidence`,
`FactContent`). Swapping two arguments of the same type compiles and is a
silent correctness bug. Three persistence adapters (`SurrealStore` and its
test fakes) plus the in-tree test fixtures called `rehydrate` directly, so a
signature drift propagated to 40+ call sites.

The 14-argument `Fact::rehydrate` was the worst offender: the row-to-aggregate
mapping in `surreal_store.rs` had to thread every field by position, and any
new persisted field meant re-touching every call site with no compiler help
on field identity.

## Decision

Migrate the positional constructors to **record-struct** style:

- `Fact::rehydrate(FactRecord { ... })` — `FactRecord` is a plain struct
  with one named field per rehydrated attribute.
- `Fact::new_pending(NewPendingRequest { ... })` — covers the
  freshly-extracted pending fact path.
- `SessionState::rehydrate(SessionRecord { ... })`.

The migration followed an **additive** pattern so that no slice in the
refactor train was behavior-changing on its own (matches AAD-2 in
`docs/REFACTOR_PLAN.md`):

1. **R2** introduces the record-struct and a new constructor next to the
   positional one. Both compile; nothing calls the new one yet.
2. **R3** migrates every caller to the record-struct form, then deletes the
   positional constructor.
3. The record-struct is renamed to the canonical name once the positional
   form is gone.

`FactRecord` doubles as the serialisable snapshot shape — the persistence
mapper builds it from a SurrealDB row and the aggregate consumes it, so the
row-to-aggregate mapping has a single typed hop instead of 14 positional ones.

## Alternatives Considered

- **Builder pattern.** A `FactRehydrateBuilder` with chained `.field(...)`
  calls. Rejected: it solves the named-argument problem but is overkill for a
  pure rehydration path that always supplies every field — the builder would
  need runtime "field not set" tracking for a shape that is structurally
  total. The record-struct gives named fields at the call site with no
  runtime state machine.
- **Keep positional.** Rejected: the positional form already produced silent
  same-type-swap bugs and blocked adding persisted fields cleanly. The cost
  grew with every new field.
- **Tuple-struct (`Fact(Timestamp, Timestamp, ...)`).** Rejected: tuple
  fields are unnamed (`.0`, `.1`, ...), which is strictly worse than the
  positional `fn` for readability and identical for type-safety.

## Consequences

- **Positive:** Call sites name every field. Adding a persisted field is now
  a record-struct edit plus the call-site literal update, with the compiler
  pointing at every caller. `FactRecord` is reused as the serialisable
  snapshot in the SurrealStore mapper and in test fixtures via struct
  literals.
- **Positive:** Test fixtures read as field-by-field literals; the 14-arg
  positional noise is gone from `tests/common` and the persistence adapter
  tests.
- **Negative:** One extra type per aggregate (`FactRecord`,
  `NewPendingRequest`, `SessionRecord`). The trade is acceptable: the types
  are plain data carriers with no behaviour, and they document the
  rehydration shape explicitly.
- **Negative:** Constructing a record literal is one more line at the call
  site than a positional call. Field naming earns that line.

40+ call sites were migrated in R3. No test body changed — only constructor
invocations. The 898-test baseline stayed green across R2 and R3.
