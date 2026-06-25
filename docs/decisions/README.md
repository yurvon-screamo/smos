# Architecture Decision Records

This directory holds SMOS's ADRs — short, dated records of architectural
decisions that future contributors (humans and agents) need to understand
the codebase as it is today.

## Format

Each ADR is one Markdown file named `NNNN-kebab-case-title.md`, where
`NNNN` is a zero-padded sequence number. The sections are:

- **Status** — `Accepted`, `Superseded`, `Deprecated`.
- **Date** — ISO 8601 (`YYYY-MM-DD`).
- **Context** — the problem that forced the decision.
- **Decision** — what was decided, with enough detail to be actionable.
- **Alternatives Considered** — what was rejected and why.
- **Consequences** — positive, negative, and the trade-offs accepted.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-record-struct-constructors.md) | Record-struct constructors for `Fact` / `SessionState` | Accepted |
| [0002](0002-shared-testkit-parity-gate.md) | Shared `smos_application::testkit` with a parity gate | Accepted |
| [0003](0003-god-module-split-stable-pubapi.md) | God-module split with a stable public API | Accepted |
| [0004](0004-scanstate-encapsulation-drift-priority.md) | `ScanState` encapsulation for the drift-priority walk | Accepted |
| [0005](0005-embedding-dim-canonical-source.md) | `EMBEDDING_DIM` canonical source | Accepted |

## When to write an ADR

Write one when a decision is hard to reverse, affects multiple modules, or
trades off an invariant a future editor would not infer from the code. Do
not write one for a local refactor that a reader can follow from the diff.

A new ADR increments the sequence number (no renumbering) and adds a row
to the table above.
