# Contributing to SMOS

Thanks for considering a contribution. This document covers the development setup, the architecture you will be working in, the test surface, the code style we enforce, and the PR process.

## Development setup

Requirements:

- **Rust 1.96+** (pinned via [`rust-toolchain.toml`](rust-toolchain.toml)).
- **Ollama** running locally for any test that touches extraction, embeddings, or NLI end-to-end.
- (Optional) a CUDA / DirectML / Metal / WebGPU-capable GPU if you want to exercise a GPU EP.

```bash
git clone https://github.com/yurvon-screamo/smos.git
cd smos
cargo build --workspace          # smoke-compile every crate
cargo t                          # default test surface (no external deps)
```

If `cargo t` is green, you have a working tree. `cargo tall` (the full surface, including the 643 MB DeBERTa-v3 ONNX download and live Ollama calls) is only needed when your change touches the native NLI path or a live-Ollama integration surface.

## Architecture overview

SMOS is a 3-crate Cargo workspace in the Hexagonal / DDD style. The dependency direction is enforced one way:

```
smos-domain  ←  smos-application  ←  smos-adapters
   (pure)         (ports + use         (every concrete
                    cases)              IO implementation)
```

A layering violation fails to compile: `smos-domain`'s `Cargo.toml` declares no tokio, no serde_json, no surrealdb.

### `smos-domain` — pure logic

| Module | Contents |
|---|---|
| `entities/` | `Fact`, `Session` — the long-lived domain objects. |
| `value_objects/` | `FactId`, `FactContent`, `Confidence`, `Cosine`, `Embedding`, `Heat`, `MemoryKey`, `NliResult`, `SessionId`, `SourceSessions`, `Timestamp`. |
| `enums/` | `FactStatus`, `FactType`, `MergeReason`, `NliLabel`. |
| `chat.rs` | Tool-call argument shapes (opaque payload types, no JSON parsing in domain). |
| `config.rs` | Domain-relevant thresholds (NLI cuts, confidence bonuses). |
| `error.rs` | Domain error type. |

Rules: no IO, no async runtime, no `serde_json` in production code (it is a dev-dependency for round-trip tests only). Comments and invariants live here.

### `smos-application` — ports + use cases

| Module | Contents |
|---|---|
| `ports/` | `Clock`, `Delay`, `EmbeddingProvider`, `FactRepository`, `IdGenerator`, `LlmExtractor`, `LlmUpstream`, `NliClassifier`, `RerankProvider`, `SessionRepository`. |
| `use_cases/` | `enrich_request`, `extract_facts_from_response`, `finalize_session`, `handle_chat_completion`, `import_opencode_session`. |
| `helpers/` | `memory_block`, `model_parser`, `noise_filter`, `openai_content`, `request_enricher`, `retrieval_planner`, `session_marker`, `topic_extractor`. |
| `types/` | `chat_request`, `chat_response`, `enrichment_messages`, `merge_result`, `rerank_result`, `search_hit`. |
| `errors/` | Application-layer error type. |

Rules: ports are `async fn` **without** a `Send` bound. The bound is added at the adapter call site, which keeps the application layer runtime-agnostic.

### `smos-adapters` — every concrete IO implementation

| Module | Contents |
|---|---|
| `storage/` | `SurrealStore`, `surreal_schema`, `SystemClock`, `SystemIdGenerator`. |
| `nli/` | `native_nli`, `runtime`, `model_cache`, `device` — the `ort` + ONNX Runtime NLI backend. |
| `http/` | `axum_server`, `routes/`, `stream_transform`, `error_mapper`. |
| `upstream/` | `reqwest_upstream`, `sse_parser`, `streaming_buffer`. |
| `providers/` | `ollama/` (extraction + embeddings), `llama_cpp/` (reranker), `noop/`. |
| `opencode/` | Session discovery for the `import` subcommand. |
| `doctor/` | Environment validation for `smos doctor`. |
| `dreaming/` | Optional audit agent (`rig-core` tool-calling). |
| `runtime/` | Service install + supervisor glue. |
| `cli/` | `clap` subcommand wiring. |
| `bin/smos.rs` | The unified binary entry point. |
| `config.rs` | Layered TOML config (defaults + override). |

Rules: this is the only crate that may import `tokio`, `serde_json`, `surrealdb`, `axum`, `reqwest`, `ort`. Every IO boundary in the system routes through a port trait defined in `smos-application`.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for data flow, memory lifecycle, and NLI pipeline internals.

## Testing strategy

Every test that does **not** carry `#[ignore]` runs under the default `cargo t`. The previous feature-gate tier system was removed because the gates hid pre-existing regressions and made the plain command a developer types silently skip every `e2e_*` binary.

| Alias | Scope |
|---|---|
| `cargo tf` | `smos-domain` + `smos-application` only — fastest feedback loop. |
| `cargo t` | Workspace unit tests + every embedded-SurrealDB / wiremock e2e binary. Default pre-commit. |
| `cargo ti` | Alias kept for backwards compatibility — same scope as `cargo t`. |
| `cargo tall` | Adds the `#[ignore]` tests: native NLI model download + live Ollama. Pre-release only. |

### `#[ignore]` policy

`#[ignore]` is reserved for **external dependencies** only:

- **Native NLI model download** — `#[ignore = "requires 643MB DeBERTa ONNX model download"]`
- **Live Ollama** — `#[ignore = "requires live Ollama on localhost:11434"]`

A bug in our own code (including a SurrealQL syntax mistake) is **not** a reason to `#[ignore]` a test — fix it. If you add a new `tests/*.rs` binary, decide its category up front:

1. **Pure unit helpers** (no IO, no async runtime) → no special handling.
2. **Embedded-SurrealDB / wiremock / TCP listener** → universal, runs by default. No gating.
3. **Needs a live Ollama / model download** → `#[ignore]` per test, with a reason.

See [`AGENTS.md`](AGENTS.md) for the full rationale.

## Code style

### Mandatory

- **`cargo fmt --all --check`** must pass. Default rustfmt style (K&R braces, 4-space indent).
- **`cargo clippy --workspace --all-targets -- -D warnings`** must pass. Warnings are CI failures, not suggestions.
- **No `unwrap()` / `expect()` in production code** outside of test modules, build scripts, and const-initialisation contexts where panicking is the only option. Map errors through `thiserror` types or `anyhow::Result`.
- **No silent failures.** A swallowed error is worse than an explicit panic. If you cannot handle an error, propagate it; if propagation is impossible, log at `error!` and surface a typed failure.
- **Comments in English.** `///` doc-comments are welcome on public API.
- **Git commits in English.** Use the conventional-commit prefix (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`) and an imperative subject line under 72 characters.

### Encouraged

- **Doc-comments on every public port trait and use case entry point.** A new contributor should be able to read `smos-application/src/ports/*.rs` and understand the system.
- **Explicit invariants as comments.** When a domain rule is non-obvious (e.g. "FactId = SHA1(content) keeps exact duplicates stable across re-extraction"), say so next to the code that enforces it.
- **Hexagonal layering over convenience.** If you are tempted to import `tokio` in `smos-domain` or `serde_json` in `smos-application`, stop — the answer is a port trait.
- **Fail-open at the IO boundary, fail-closed in the domain.** Enrichment failures degrade gracefully; domain invariants panic or return typed errors.

### Forbidden

- Adding `tokio`, `serde_json`, `surrealdb`, `axum`, `reqwest`, or `ort` as a dependency of `smos-domain` or `smos-application`.
- `unwrap()` / `expect()` / `panic!()` in adapter code paths that handle user requests. Use typed errors.
- Adding a new feature flag without documenting it in the README GPU table and `AGENTS.md`.
- Marking a non-external-dependency test `#[ignore]` to make CI green.

## PR process

1. **Fork and branch.** Branch from `main`, name it `<type>/<short-description>` (e.g. `fix/surreal-lock-retry`).
2. **Write tests first when fixing a bug.** A regression test that fails before your fix and passes after is the proof the bug existed and is gone.
3. **Run the full local gate before pushing:**

   ```bash
   cargo build --workspace
   cargo t
   cargo clippy --workspace --all-targets -- -D warnings
   cargo fmt --all --check
   ```

   Run `cargo tall` only if your change touches the native NLI path or a live-Ollama surface.

4. **Open a PR against `main`.** Fill in the PR template (the repo has one at [`.github/pull_request_template.md`](.github/pull_request_template.md)). Reference the issue number if applicable.
5. **Describe the why, not just the what.** A reviewer reading the diff should understand the problem before reading the solution. If the change has a trade-off, state it.
6. **Respond to review.** Squash or rebase per reviewer preference; the merge is via squash by default to keep history linear.

### What gets a PR rejected

- Failing `cargo t`, `cargo clippy -D warnings`, or `cargo fmt --check`.
- Layering violations (IO imports in domain/application).
- A `#[ignore]` added to silence a failing test that does not depend on an external service.
- Marketing-style language in code comments, commit messages, or docs (no "revolutionary", "seamless", "next-gen"). State what the code does.

## Getting help

- **Architecture questions** — open a [Discussion](https://github.com/yurvon-screamo/smos/discussions) or an issue labelled `question`.
- **Bugs** — open an issue using the bug report template.
- **Feature ideas** — open an issue labelled `enhancement` and outline the use case before any code.

Thank you for reading this far.
