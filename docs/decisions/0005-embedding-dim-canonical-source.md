# ADR-0005: EMBEDDING_DIM canonical source

- **Status:** Accepted
- **Date:** 2026-06-25
- **Supersedes:** —

## Context

The embedding dimensionality for the HNSW vector index is `1024` (Jina v5).
Before this refactor, the literal `1024` appeared in two places:

1. `smos/src/config.rs` — config validation rejected an embedding whose
   dimensionality did not equal `1024`.
2. `smos/tests/common` — test fixtures built embeddings of width `1024`.

Both literals were independently pinned to the HNSW DDL in
`storage/surreal_schema.rs`, which itself hard-coded `1024` in the
`DEFINE INDEX ... HNSW DIM 1024` statement. Three sources of truth, all
manual, all tied to the upstream embedding model. Changing the model (a
different Jina variant, a different vendor) would have required touching
all three and hoping they stayed in sync — a silent dimensional mismatch
between the index and the embeddings would corrupt vector search at query
time, not at startup.

The canonical value already existed in the domain layer as
`Embedding::EXPECTED_DIM` (an associated const on the `Embedding` value
object), but nothing in the adapter layer read it.

## Decision

`Embedding::EXPECTED_DIM` in `smos-domain` is the **single canonical
source**. The adapter re-exports it under the name the adapter already used:

```rust
// smos/src/storage/surreal_schema.rs
pub const EMBEDDING_DIM: usize = Embedding::EXPECTED_DIM;
```

Note the form: it is a `const` whose value is `Embedding::EXPECTED_DIM`, **not**
`pub use Embedding::EXPECTED_DIM;`. The `pub use` form does not compile:

```
error[E0432]: unresolved import `Embedding::EXPECTED_DIM`
```

`Embedding` is a struct, and `pub use` addresses items (modules, types,
functions), not associated consts on a struct. A `const` alias is the
idiomatic way to re-export an associated const under a different path.

The schema DDL, the config validation, and the test fixtures now all read
`EMBEDDING_DIM` (or, from within the domain, `Embedding::EXPECTED_DIM`).

An invariant test (`embedding_dim_alias_matches_domain_expected`) asserts
that the adapter alias equals the domain canonical value, so a future edit
that re-pins the alias to a literal fails CI.

## Alternatives Considered

- **`pub use Embedding::EXPECTED_DIM;` in the adapter.** Rejected: does not
  compile (E0432). `pub use` cannot re-export an associated const on a
  struct under a different name. This was the first attempt and the
  compiler error is what motivated the `const` alias form.
- **A new `EMBEDDING_DIM` const in the schema module, set to `1024`.**
  Rejected: this would create a third source of truth (schema, domain,
  config) instead of consolidating to one. The whole point is that the
  domain value object owns the dimensionality and the adapter reads from
  it.
- **Move `EXPECTED_DIM` out of the domain into a shared constants module.**
  Rejected: `Embedding::EXPECTED_DIM` is already the right home — the
  dimensionality is a property of the `Embedding` value object. Moving it
  to a `constants` module would divorce it from the type it describes.

## Consequences

- **Positive:** Single source of truth in the domain layer. The schema
  DDL, config validation, and test fixtures all read through one alias.
- **Positive:** Changing the embedding model is now a one-line edit to
  `Embedding::EXPECTED_DIM` plus a migration of the HNSW index. The
  adapter code does not need to know the new value.
- **Positive:** The invariant test guards against a future re-divergence.
  A literal `1024` reintroduced into config or tests does not fail the
  compiler, but the alias test would catch a value mismatch if someone
  re-pins the alias.
- **Negative:** The adapter reads the dimensionality via a `const` alias
  rather than directly. One extra hop at the call site
  (`surreal_schema::EMBEDDING_DIM` instead of a literal). The hop is the
  point — it routes every reader through the canonical value.
- **Negative:** A literal `1024` in newly-written code still compiles. The
  convention is enforced by review and the AGENTS.md note, not by the
  compiler. The invariant test catches a value mismatch only when someone
  updates the alias, not when someone adds a fresh literal.
