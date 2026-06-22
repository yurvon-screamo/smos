<div align="center">

# SMOS — Semantic Memory OS

OpenAI-compatible **semantic memory proxy** for AI agents.

Give any OpenAI-compatible client long-term memory without changing a line of its code.

[![Rust](https://img.shields.io/badge/rust-1.96-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-665%2B-green.svg)](#testing)
[![Edition](https://img.shields.io/badge/edition-2024-lightgrey.svg)](https://blog.rust-lang.org/2024/11/28/Rust-2024.html)

</div>

---

## Why SMOS?

- **🔌 Zero-friction memory.** Set `OPENAI_BASE_URL=http://localhost:8888/v1` and memory works. No SDK to import, no prompts to rewrite, no client-side code.
- **🧠 NLI contradiction detection.** A DeBERTa-v3 model judges each candidate fact via entailment/contradiction/neutral verdicts — not cosine similarity. `TTL=60` vs `TTL=10` is flagged as a conflict, the LLM receives both and decides.
- **🧬 Semantic deduplication.** `TTL=10 prevents refresh` and `Token lifetime of 10 minutes avoids loops` collapse into one fact. Embeddings catch rephrasings that escape the `FactId = SHA1(content)` exact match.
- **🔁 Cross-session confirmation.** A fact extracted in two independent sessions gets a confidence boost. Multi-source observations are trustworthy; single-source claims stay pending until corroborated.
- **🦀 Native Rust, no Python.** NLI runs on `ort` + ONNX Runtime. Storage is embedded SurrealDB (RocksDB + HNSW vector index). One binary, one data directory, no external services.
- **⚡ Multi-provider + GPU.** Round-robin or failover across Ollama, OpenAI, OpenRouter. NLI inference runs on CUDA, DirectML, Metal, or WebGPU — opt-in per build.

## Quick Start

### 1. Prerequisites

- **Rust 1.96+** (`rustup update stable`)
- **Ollama** running locally (`ollama serve`)
- **llama.cpp** build with `llama-server` on PATH (for the reranker; see step 4)
- **GPU** (optional but recommended): Intel Arc, NVIDIA, or Apple Silicon

### 2. Clone & build

```bash
git clone https://github.com/yurvon-screamo/smos.git
cd smos

# Default — CPU NLI inference
cargo build --release --bin smos

# Pick exactly one GPU feature for NLI acceleration:
cargo build --release --bin smos --features smos-adapters/nli-directml   # Windows + Intel Arc / AMD / NVIDIA
cargo build --release --bin smos --features smos-adapters/nli-cuda      # Windows / Linux + NVIDIA
cargo build --release --bin smos --features smos-adapters/nli-metal     # macOS + Apple Silicon
cargo build --release --bin smos --features smos-adapters/nli-webgpu    # universal (Vulkan/DX12/Metal)
```

### 3. Pull the required Ollama models

```bash
ollama pull granite4.1:3b                                              # upstream LLM
ollama pull qwen3.5:2b                                                 # extraction LLM
ollama pull hf.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:latest   # embeddings
```

### 4. Start the reranker

SMOS reranks vector-search survivors via a Qwen3-Reranker cross-encoder
before injecting them into the request. The reranker is a **hard
dependency**: if it is unreachable or returns an empty result, every
chat-completion request fails with **HTTP 503** rather than silently
shipping vector-order-only ranking. `smos doctor` reports it as FAIL.

Download a Qwen3-Reranker GGUF from HuggingFace, then start the
llama.cpp reranker server:

```bash
# Download qwen3-reranker-0.6b-q8_0.gguf (or any Qwen3-Reranker GGUF variant)
./llama-server --model qwen3-reranker-0.6b-q8_0.gguf --port 8181
```

The adapter expects an OpenAI-compatible `/v1/rerank` endpoint. `smos doctor`
probes `http://localhost:8181/health` and FAILs when the reranker is
unreachable.

### 5. Configure

Edit `smos.toml` next to the binary (see the [Configuration](#configuration)
section). Minimal working setup:

```toml
[[upstream.providers]]
name = "ollama-local"
url = "http://localhost:11434/v1/chat/completions"
api_key = "ollama"

[llm_extraction]
url = "http://localhost:11434"
model = "qwen3.5:2b"
temperature = 0.0
seed = 42

[embedding]
url = "http://localhost:11434"
model = "hf.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:latest"
dimensions = 1024

# Reranker — hard dependency. Empty URL is a startup validation error;
# an unreachable server turns every request into HTTP 503.
[reranker]
url = "http://localhost:8181"
model = "qwen3-reranker"
timeout_seconds = 60

[nli_backend]
model = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
cache_dir = "./data/nli_cache"
```

### 6. Run

```bash
# Start the proxy
./target/release/smos serve

# Health check
curl http://localhost:8888/health
# → {"status":"ok","version":"0.1.0"}
```

The proxy listens on `127.0.0.1:8888` by default. First startup downloads
the DeBERTa-v3 ONNX model (~643 MB) into `[nli_backend].cache_dir`.
Pre-warm before going live:

```bash
./target/release/smos finalize sess_dummy --memory-key shared
```

### 7. Connect your AI client

**opencode / cursor / any OpenAI SDK:**

```bash
export OPENAI_BASE_URL=http://localhost:8888/v1
export OPENAI_API_KEY=smos
opencode --model myproject:granite4.1:3b
```

**curl:**

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"myproject:granite4.1:3b","stream":true,"messages":[{"role":"user","content":"hello"}]}'
```

The `myproject:` prefix is the **memory key** — SMOS partitions memory by
project namespace. A bare model name (no prefix) maps to the `shared`
namespace.

## How it works

```
User → opencode / cursor / curl
        │ POST /v1/chat/completions
        │ model: "myproject:granite4.1:3b"
        ▼
┌─────────────────────── SMOS PROXY ───────────────────────┐
│                                                            │
│  1. PARSE      strip memory_key prefix, detect/reuse       │
│                 session id                                  │
│  2. ENRICH     embed topic → HNSW vector search → rerank   │
│                 → heat filter → atomic dedup → inject      │
│                 <smos-memory> block into request           │
│  3. FORWARD    reqwest POST → upstream LLM (SSE streaming  │
│                 passthrough, response untouched)           │
│  4. EXTRACT    (background) Qwen3.5:2b pulls facts from    │
│                 the response, stores as pending            │
│  5. FINALIZE   (background, on session timeout)            │
│                 DeBERTa-v3 NLI → merge / conflict /        │
│                 promote                                     │
│  6. AUDIT      (background, cron) optional dreaming agent  │
│                 runs cloud LLM tool-calling cleanup        │
│                                                            │
└────────────────────────────────────────────────────────────┘
        │
        ▼
   Upstream LLM (Ollama / OpenAI / OpenRouter / …)
```

Stages 4, 5, and 6 run **off the request path** — the response always
returns to the client as soon as the LLM stream completes.

**Failure modes:**

- **Embedder / vector-search / dedup failures** — fail-open. The request
  forwards unenriched (no `<smos-memory>` block). A flaky memory
  subsystem never breaks the user's chat.
- **Reranker failure** (provider error or empty result) — **fail-closed**.
  The request returns HTTP 503 "SMOS provider unavailable: …". Silent
  vector-order-only ranking is not exposed to the client.
- **Upstream failure** — propagated per the §12 status matrix (4xx
  verbatim, 5xx/timeout → 502).

## Commands

| Command | Description |
|---|---|
| `smos serve` | Start the HTTP proxy server (session watcher + native NLI). |
| `smos import --from-file <file> --memory-key <key>` | Import a transcript into memory. |
| `smos import --list` | List discoverable opencode sessions. |
| `smos import <session> --dry-run` | Parse turns only, no model calls, no writes. |
| `smos doctor` | Environment validation + SurrealDB stats. **Fails when the reranker is unreachable.** |
| `smos doctor --stats` | Quick SurrealDB stats (no model round-trips). |
| `smos doctor --report <path>` | Generate a Markdown report. |
| `smos finalize <session_id> --memory-key <key>` | Manual finalize trigger (single session drain). Pre-warms the NLI model when called with a dummy session. |
| `smos audit [--provider cloud\|local] [--dry-run]` | One-shot dreaming-agent run in the foreground (independent of `[audit].enabled`, which only gates the in-server cron). `--dry-run` validates config without loading the NLI model. |
| `smos service install` | Install as system service (systemd / Windows service / launchd). |
| `smos service uninstall` | Remove the system service. |
| `smos service start / stop / restart / status` | Control the installed service. |

Global flag: `--config <path>` (defaults to `smos.toml` next to the binary).

## Configuration

`smos.toml` (next to the binary, or via `--config <path>`) is **layered** —
sections present in the file override built-in defaults; any omitted section
falls back. See [`smos.toml`](smos.toml) for the canonical example and
[`smos-adapters/src/config.rs`](smos-adapters/src/config.rs) for every field.

| Section | Purpose |
|---|---|
| `[surreal]` | Embedded RocksDB path + namespace/database. |
| `[server]` | Bind host/port, shutdown grace, extraction toggle, log format. |
| `[[upstream.providers]]` | Multi-provider LLM endpoints (round-robin / failover). |
| `[upstream.strategy]` | `single` / `round_robin` / `failover`. |
| `[llm_extraction]` | Fact extraction LLM (model, temperature, seed). |
| `[embedding]` | Vector embedding model (model, dimensions). |
| `[reranker]` | llama.cpp reranker URL (`/v1/rerank`). An unreachable reranker makes every request fail with HTTP 503. Empty `url` is a startup validation error. |
| `[retrieval]` | top-K initial/final, `min_topic_chars`, `min_confidence`. |
| `[merge]` | Cosine candidate-selection threshold for merge detection. |
| `[confidence]` | Base + multi-source/no-contradiction bonuses, accept/pending cut. |
| `[nli]` | NLI softmax thresholds (contradiction/entailment). |
| `[nli_backend]` | Native ort/ONNX model id + cache directory. |
| `[extraction]` | Semantic dedup cosine threshold. |
| `[heat]` | Decay rate, min threshold (boosts recently-active facts). |
| `[session]` | Timeout, pending overflow threshold, watcher scan interval. |
| `[audit]` | Optional dreaming agent (schedule, model, mutation caps). |

Secrets: the `[audit].cloud_api_key` field expands `${OPENROUTER_API_KEY}`
via `std::env::var` at runtime, so secrets stay out of TOML. The same
pattern works for any `api_key` field — and the convenience env var
`SMOS__UPSTREAM__API_KEY` broadcasts onto every `[[upstream.providers]]`
entry whose TOML `api_key = ""`.

## GPU support

Each GPU execution provider is opt-in via a cargo feature flag. Pick
**at most one** per build — ort's prebuilt binary matrix cannot satisfy
every combination (e.g. CUDA and WebGPU cannot coexist in a single
binary).

| Feature flag | Platform | GPU |
|---|---|---|
| *(default)* | All | CPU |
| `nli-directml` | Windows | Intel Arc, AMD, NVIDIA (DirectX 12) |
| `nli-cuda` | Windows, Linux | NVIDIA |
| `nli-metal` | macOS | Apple Silicon |
| `nli-webgpu` | All | Universal (Vulkan / DX12 / Metal) |

At startup `smos serve` logs the detected device and falls back to CPU
automatically if the selected provider cannot initialise. HTTP keeps
serving even if NLI fails — only the session watcher is disabled.

## Testing

Every test that does **not** carry `#[ignore]` runs under the default
`cargo t`. Embedded SurrealDB + wiremock + in-process axum run without
external services.

| Alias | Scope | Tests | Time |
|---|---|---|---|
| `cargo tf` | `smos-domain` + `smos-application` only | ~351 | ~2s |
| `cargo t` | All unit tests + embedded-SurrealDB / wiremock e2e | 665 (+ 5 ignored) | ~60s warm |
| `cargo ti` | Alias kept for compat — same scope as `cargo t` | 665 | ~60s warm |
| `cargo tall` | Adds native NLI model tests (643 MB download + live Ollama) | 665 + 5 ignored | ~10 min |

`#[ignore]` is reserved for **external dependencies** (model download,
live Ollama). A bug in our own code is never a reason to `#[ignore]` a
test — see [`AGENTS.md`](AGENTS.md) for the policy.

**Dev workflow:**

- After editing `smos-domain` or `smos-application`: `cargo tf`
- Before commit: `cargo t`
- Before release: `cargo tall`

**Lint gate (CI-equivalent):**

```bash
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development setup,
testing strategy, code style, and PR process. The full layer-by-layer
breakdown, data flow, and NLI pipeline internals live there as well.

Short version: fork → branch → `cargo t` + `cargo clippy -- -D warnings` +
`cargo fmt --check` → PR. The lint gate is mandatory.

## License

MIT — see [`LICENSE`](LICENSE).

---

Built by [turbin_y](https://github.com/yurvon-screamo). Feedback, bug
reports, and architecture questions are welcome in
[Issues](https://github.com/yurvon-screamo/smos/issues).
