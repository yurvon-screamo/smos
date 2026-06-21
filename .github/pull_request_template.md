## Summary

One or two sentences: what does this PR change and why?

## Motivation

- What problem does this solve?
- Is there an issue this closes? (Use `Closes #NNN`.)
- If this is a new feature, what is the use case?

## Change type

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (something existing stops working)
- [ ] Documentation only
- [ ] Refactor / cleanup (no behaviour change)
- [ ] Test-only

## Layering check

If this PR touches code in `smos-domain` or `smos-application`:

- [ ] No new IO imports (`tokio`, `serde_json`, `surrealdb`, `axum`, `reqwest`, `ort`) added to `smos-domain` or `smos-application`.
- [ ] Any new IO boundary is a port trait in `smos-application`, implemented in `smos-adapters.

## Local gate

Run before requesting review:

```bash
cargo build --workspace
cargo t
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

- [ ] `cargo build --workspace` passes
- [ ] `cargo t` passes (no new `#[ignore]` without a documented external dependency)
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo fmt --check` passes
- [ ] `cargo tall` passes (only if the change touches the native NLI path or a live-Ollama surface)

## Tests

- [ ] Bug fix: a regression test that fails before this PR and passes after.
- [ ] New feature: at least one test covering the happy path and one covering a failure mode.
- [ ] No `#[ignore]` added to silence a failing test that does not depend on an external service.

## Trade-offs

If this change has a downside (performance, complexity, new dependency, layering compromise), state it here. Reviewer-visible trade-offs get merged faster than hidden ones.

## Documentation

- [ ] `README.md` updated if a user-facing behaviour changed.
- [ ] `docs/ARCHITECTURE.md` updated if an internal contract changed.
- [ ] `smos.toml` comments updated if a new config field was added.
- [ ] `AGENTS.md` updated if a new contributor-facing convention was introduced.
