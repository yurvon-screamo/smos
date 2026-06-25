# AGENTS.md — SMOS Rust workspace

Conventions specific to the `smos-rust/` workspace. The parent
[`D:\uwuwu_agent\AGENTS.md`](../AGENTS.md) still applies; this file adds
SMOS-specific rules.

## Testing

SMOS uses a single test surface (defined in
[`.cargo/config.toml`](.cargo/config.toml)). Every test that does NOT carry
`#[ignore]` runs under `cargo t`; `cargo tall` additionally runs the
`#[ignore]` tests.

| Alias       | When to run                                                |
|-------------|------------------------------------------------------------|
| `cargo tf`  | After editing `smos-domain` or `smos-application` only.   |
| `cargo t`   | Default pre-commit check. Runs every non-`#[ignore]` test. |
| `cargo ti`  | Alias kept for compat — same scope as `cargo t`.            |
| `cargo tall`| Pre-release. Includes every `#[ignore]` test (the 643 MB DeBERTa-v3 ONNX download). |

See [README.md](README.md) → Testing for the full breakdown.

### `#[ignore]` policy

Tests must pass ALWAYS. If a test cannot pass without an external dependency
(e.g. the 643 MB DeBERTa-v3 ONNX model download), mark it
`#[ignore = "<reason>"]`:

- **Native NLI model download** —
  `#[ignore = "requires 643MB DeBERTa ONNX model download"]`

`#[ignore]` is reserved for **external dependencies**. A bug in our own
code (including a SurrealQL syntax mistake) is NOT a reason to `#[ignore]`
a test — fix it. The previous batch of "pre-existing SurrealDB 2.x
regression" markers was a layer of hiding: `array::contains` is not a
SurrealQL function (use the `CONTAINS` operator), and `array::difference`
is the symmetric difference A△B (use `array::complement` for the relative
complement A\B). Both are now fixed and the tests run by default.

When adding a new `tests/*.rs` binary, decide its category up front:

1. **Pure unit helpers** (no IO, no async runtime) → no special handling.
2. **Embedded-SurrealDB / wiremock / TCP listener** → universal, runs by
   default. No gating.
3. **Needs the 643 MB DeBERTa model download** → `#[ignore]` per test
   with the reason above.

### Feature gates (smos)

The NLI backend is always native (ort + ONNX Runtime). There are **no
compile-time GPU features** — `ort` is built with `load-dynamic`, so the
matching ONNX Runtime shared library (CPU, CUDA, DirectML, or the macOS
CoreML build) is downloaded into `~/.smos/models/ort/<device>/` on first
use and loaded via `ORT_DYLIB_PATH`. The same `smos` binary runs on every
GPU vendor without rebuilds.

Device selection is a runtime config value (`[nli_backend].device`):

- `"auto"` (default) — probe the host at startup.
  - Windows: CUDA (NVIDIA only, fastest EP for ort), then DirectML
    (Intel Arc / AMD / NVIDIA via DX12), then CPU.
  - Linux: CUDA, then CPU.
  - macOS: Metal / CoreML on Apple Silicon, else CPU.
- `"cpu"` / `"directml"` / `"cuda"` / `"metal"` — force a specific device.

There are no test-gating features. Tests that need a live external dependency
(the 643 MB DeBERTa-v3 ONNX download) carry `#[ignore = "<reason>"]` and run
via `cargo tall`.

## Quality gates (run before declaring a task done)

```bash
cargo t                              # universal, no feature flags
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

Run `cargo tall` only when the change touches the native NLI path.

## Architecture reminders

- Three-crate workspace: `smos-domain` (pure, no IO) ← `smos-application`
  (ports + use cases, runtime-agnostic) ← `smos` (the only crate
  that performs IO).
- Do not introduce tokio / serde_json / surrealdb deps in `smos-domain`.
- Async port traits are `Send`-bounded at the adapter call site, not at the
  port definition.
- Comments and git commits are in English; doc-comments (`///`) are welcome
  on public API.
