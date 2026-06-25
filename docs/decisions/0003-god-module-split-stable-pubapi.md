# ADR-0003: God-module split with a stable public API

- **Status:** Accepted
- **Date:** 2026-06-25
- **Supersedes:** —

## Context

Four modules in the workspace had grown past the point where a reader could
hold their responsibility set in working memory:

| Module | Lines | Cyclomatic | Problem |
|---|---|---|---|
| `smos/src/config.rs` | 1706 | high | Mixed config types, defaults, validation, layered loader, and tests in one file. |
| `smos/src/storage/surreal_store.rs` | 1241 | 168 | Five responsibilities in one file: row types, row-to-aggregate mapping, `FactRepository` impl, `SessionRepository` impl, HNSW vector search. |
| `smos-application/src/use_cases/finalize_session.rs` | 1562 | high | The NLI walk, merge logic, outcome construction, and tests interleaved. |
| `smos-application/src/use_cases/extract_facts.rs` | 1371 | high | Extraction orchestration, dedup, noise filtering, persistence, and tests. |

The size blocked evolution: a change to vector search touched the same file
as a change to session persistence, and the diff context was unmanageable.
`surreal_store.rs` at cyclo 168 was the worst — it mixed five distinct
responsibilities behind one `mod` and made the boundary between
row-mapping and repository logic invisible.

## Decision

Split each god-module into a **directory module** with one file per
responsibility, and re-export the original public API from the directory's
`mod.rs` (or `lib.rs` where appropriate) so that **every external caller
keeps compiling unchanged**. This is AAD-3 in `docs/REFACTOR_PLAN.md`:
module split without a public-API change.

The hard contract is `cargo doc --workspace --no-deps` produces **zero
warnings**. A re-export that drops doc visibility, an orphaned `pub use`,
or a renamed item surfaces as a rustdoc warning, so the doc build is the
gate that the public API actually remained stable.

Resulting layout:

```
config/
├── types.rs
├── defaults.rs
├── validate.rs
├── loader.rs
└── tests.rs

storage/
├── rows.rs              # row types (DB-shaped)
├── mapping.rs           # row ↔ aggregate mapping (FactRecord etc.)
├── fact_repository.rs   # FactRepository impl
├── session_repository.rs
└── vector_search.rs     # HNSW vector search

finalize_session/
├── scan.rs              # candidate gathering + drift-priority walk
├── merge.rs             # merge commit logic
├── outcome.rs           # FactOutcome construction
└── tests.rs

extract_facts/
├── … (one file per responsibility)
└── tests.rs
```

Every leaf module is ≤ 500 lines. The public surface (`pub use` from the
directory root) is identical to the pre-split surface.

## Alternatives Considered

- **Rewrite from scratch.** Rejected: `surreal_store.rs` and
  `finalize_session.rs` carry subtle invariants (the drift-priority walk,
  the `owned_ids` snapshot before the first await, the HNSW DDL coupling).
  A rewrite would have to re-derive all of them and would lose the
  behaviour-preservation guarantee that the 898-test baseline gives the
  split.
- **Leave them.** Rejected: maintainability was already degrading. Two of
  the four files had active change pressure (vector search tuning,
  finalize outcome variants) and every touch paid the file-size cost.

## Consequences

- **Positive:** Each leaf module has one responsibility and is ≤ 500 lines.
  A change to vector search no longer touches the same file as a change to
  session persistence.
- **Positive:** The public API is unchanged. Every external caller
  (integration tests, the `smos` binary, downstream crates) compiles
  against the same paths as before the split.
- **Positive:** `cargo doc --workspace --no-deps` = 0 warnings is now a
  standing invariant, enforced at every doc build. A future split that
  breaks re-exports fails this gate immediately.
- **Negative:** The directory layout is one level deeper. Readers used to
  `surreal_store.rs` now navigate to `storage/fact_repository.rs`. The
  `mod.rs` re-exports make the public surface flat, so this only affects
  readers of the file tree, not callers.
- **Negative:** Internal `pub(crate)` items sometimes have to be lifted to
  `pub(super)` so sibling files in the same directory can see them. This
  widens the visibility slightly within the directory but never escapes
  the crate.
