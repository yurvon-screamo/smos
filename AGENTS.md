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
| `cargo tall`| Pre-release. Includes every `#[ignore]` test (643 MB model download + live Ollama). |

See [README.md](README.md) → Testing for the full breakdown.

### `#[ignore]` policy

Tests must pass ALWAYS. If a test cannot pass without an external dependency
(live Ollama, model download), mark it `#[ignore = "<reason>"]`:

- **Native NLI model download** —
  `#[ignore = "requires 643MB DeBERTa ONNX model download"]`
- **Live Ollama** —
  `#[ignore = "requires live Ollama on localhost:11434"]`

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
3. **Needs a live Ollama / model download** → `#[ignore]` per test with the
   reason above.

### Feature gates (smos-adapters)

The NLI backend is always native (ort + ONNX Runtime). The remaining
features are GPU execution providers (mutually exclusive — pick at most one):

- `nli-cuda`     — enables ort's CUDA EP (NVIDIA).
- `nli-directml` — enables ort's DirectML EP for DirectX 12-enabled GPUs on
  Windows (Intel Arc, AMD, NVIDIA). Recommended for Intel Arc on Windows.
- `nli-metal`    — enables ort's CoreML EP (Apple Silicon).
- `nli-webgpu`   — enables ort's WebGPU EP (cross-platform: Vulkan/DX12/Metal).
  Note: ort cannot combine WebGPU with CUDA in a single prebuilt binary; pick
  one GPU feature.

There are no test-gating features. Tests that need a live external dependency
(live Ollama, 643 MB DeBERTa ONNX download) carry `#[ignore = "<reason>"]`
and run via `cargo tall`.

## Quality gates (run before declaring a task done)

```bash
cargo t                              # universal, no feature flags
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

Run `cargo tall` only when the change touches the native NLI path or any
live-Ollama integration surface.

## Architecture reminders

- Three-crate workspace: `smos-domain` (pure, no IO) ← `smos-application`
  (ports + use cases, runtime-agnostic) ← `smos-adapters` (the only crate
  that performs IO).
- Do not introduce tokio / serde_json / surrealdb deps in `smos-domain`.
- Async port traits are `Send`-bounded at the adapter call site, not at the
  port definition.
- Comments and git commits are in English; doc-comments (`///`) are welcome
  on public API.
