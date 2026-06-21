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

## What is SMOS?

SMOS sits between your AI client (opencode, cursor, curl, anything OpenAI-compatible) and an upstream LLM. Before each request, it retrieves relevant facts from past sessions and injects them into the prompt. After each response, it extracts new facts. In the background, NLI-driven consolidation merges duplicates, flags contradictions, and promotes trustworthy facts.

The agent sees a smarter conversation partner. It does not know SMOS exists.

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
- **GPU** (optional but recommended): Intel Arc, NVIDIA, or Apple Silicon

### 2. Clone

```bash
git clone https://github.com/yurvon-screamo/smos.git
cd smos
```

### 3. Pull the required Ollama models

```bash
ollama pull granite4.1:3b                                              # upstream LLM
ollama pull qwen3.5:2b                                                 # extraction LLM
ollama pull hf.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:latest   # embeddings
```

### 4. Start the reranker (REQUIRED for production-quality enrichment)

The enrich pipeline reranks vector-search survivors via a Qwen3-Reranker
cross-encoder. Without it the pipeline runs in **degraded mode**
(vector-order-only ranking) and logs a WARN on every request.

Download a Qwen3-Reranker GGUF from HuggingFace, then start the llama.cpp
reranker server:

```bash
# Download qwen3-reranker-0.6b-q8_0.gguf (or any Qwen3-Reranker GGUF variant)
./llama-server --model qwen3-reranker-0.6b-q8_0.gguf --port 8181
```

The adapter expects an OpenAI-compatible `/v1/rerank` endpoint. `smos doctor`
probes `http://localhost:8181/health` and warns when the reranker is
unreachable.

### 5. Build

```bash
# Default — CPU NLI inference
cargo build --release --bin smos

# Pick exactly one GPU feature for NLI acceleration:
cargo build --release --bin smos --features smos-adapters/nli-directml   # Windows + Intel Arc / AMD / NVIDIA
cargo build --release --bin smos --features smos-adapters/nli-cuda      # Windows / Linux + NVIDIA
cargo build --release --bin smos --features smos-adapters/nli-metal     # macOS + Apple Silicon
cargo build --release --bin smos --features smos-adapters/nli-webgpu    # universal (Vulkan/DX12/Metal)
```

### 6. Configure

Edit `smos.toml` next to the binary (see the [Configuration](#configuration) section). Minimal working setup:

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

# REQUIRED for production-quality enrichment (degrades to vector-order-only
# ranking when the reranker is unreachable).
[reranker]
url = "http://localhost:8181"
model = "qwen3-reranker"
timeout_seconds = 60

[nli_backend]
model = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
cache_dir = "./data/nli_cache"
```

### 7. Run

```bash
# Start the proxy
./target/release/smos serve

# Health check
curl http://localhost:8888/health
# → {"status":"ok","version":"0.1.0"}
```

The proxy listens on `127.0.0.1:8888` by default. First startup downloads the DeBERTa-v3 ONNX model (~643 MB) into `[nli_backend].cache_dir`. Pre-warm before going live:

```bash
./target/release/smos finalize sess_dummy --memory-key shared
```

### 8. Connect your AI client

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

The `myproject:` prefix is the **memory key** — SMOS partitions memory by project namespace. A bare model name (no prefix) maps to the `shared` namespace.

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

Stages 4, 5, and 6 run **off the request path** — the response always returns to the client as soon as the LLM stream completes. Enrichment (stage 2) is fail-open: if it fails, the request proceeds unenriched rather than erroring out.

## Features

| Feature | What it does | How |
|---|---|---|
| **Memory injection** | Adds a `<smos-memory>` block with relevant past facts to each request. | Top-K HNSW search → Qwen3-Reranker → heat decay → dedup. |
| **Fail-open enrichment** | A retrieval/embedding failure never blocks a request. | Enrichment is best-effort; failures degrade to unenriched forward. |
| **Streaming passthrough** | Upstream SSE streams reach the client byte-for-byte. | `reqwest::Response::bytes_stream()` forwarded through `axum`. |
| **Deterministic extraction** | Same response → same facts, same `FactId`, every run. | `temperature = 0`, `seed = 42`, `FactId = SHA1(content)`. |
| **Semantic dedup** | Catches rephrased duplicates the SHA1 exact match misses. | Cosine ≥ 0.95 on embeddings (`[extraction].dedup_cosine_threshold`). |
| **NLI verdict** | Classifies each pending fact against accepted facts. | DeBERTa-v3-mnli via `ort` + ONNX Runtime, in-process. |
| **Conflict surfacing** | Contradictions are flagged, not silently overwritten. | Bidirectional `conflicts_with` flag; LLM sees both sides. |
| **Cross-session confirmation** | Independent extraction across sessions boosts confidence. | `multi_source_bonus = 0.2` when ≥ 2 unique sessions observe a fact. |
| **Multi-provider upstream** | Round-robin or failover across N upstream endpoints. | `single` / `round_robin` / `failover` strategies. |
| **Session marker injection** | Streaming responses are tagged for conversation-turn linking. | SSE passthrough injects an out-of-band marker per turn. |
| **Graceful shutdown** | Ctrl+C / SIGTERM drains every in-flight task before exit. | HTTP drain → extraction drain (grace window) → session watcher drain. |
| **Service install** | Runs as systemd unit, Windows service, or launchd job. | `smos service install / start / stop / status`. |
| **Dreaming agent (opt-in)** | Periodic LLM-driven memory audit with bounded mutations. | `rig-core` tool-calling, cron-scheduled, writes a markdown report. |

## Architecture

3-crate Cargo workspace in the Hexagonal / DDD style:

```
smos-domain          entities, value objects, pure domain logic
                     (no IO, no async runtime, no serde_json)

smos-application     port traits + use cases (runtime-agnostic async)
                     ports are `async fn` WITHOUT a `Send` bound — the
                     adapter call site adds the bound, application layer
                     stays runtime-neutral

smos-adapters        SurrealStore, ort+ONNX NLI, Ollama, axum HTTP,
                     reqwest upstream, clap CLI, dreaming agent
                     (the only crate that performs IO)
```

Dependency direction is enforced one way: `domain ← application ← adapters`. The `smos-domain` `Cargo.toml` declares no tokio / serde_json / surrealdb — a layering violation fails to compile.

| Component | Technology |
|---|---|
| HTTP server | axum 0.8 |
| Upstream forward | reqwest 0.12 (rustls, streaming SSE) |
| Storage | SurrealDB 2.x embedded (RocksDB + HNSW vector index) |
| NLI | ort 2.0.0-rc.12 + ONNX Runtime (DeBERTa-v3-large-mnli) |
| Embeddings | Jina v5 (1024d) via Ollama |
| Extraction | Qwen3.5:2b via Ollama |
| Reranker | Qwen3-Reranker via llama.cpp (REQUIRED for production-quality enrichment; degrades to vector-order-only ranking if unavailable) |
| Audit agent | rig-core 0.14 + tokio-cron-scheduler (opt-in) |
| Release profile | LTO = true, codegen-units = 1 |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full layer-by-layer breakdown, data flow, memory lifecycle, and NLI pipeline internals.

## Memory lifecycle

```
                    ┌──────────────┐
   extraction ────▶ │   pending    │   FactId = SHA1(content)
                    └──────┬───────┘   stored on response complete
                           │
                  finalize trigger
                  (session timeout 30 min,
                   or `smos finalize`)
                           │
                           ▼
              ┌─────────────────────────┐
              │   DeBERTa-v3 NLI walk    │
              │   (drift-priority,       │
              │    C3 guard, exact-match)│
              └──┬──────────┬───────────┘
                 │          │
        entailment      contradiction
        (merge)         (flag conflicts_with)
                 │          │
                 ▼          ▼
        ┌────────────┐  ┌──────────────────────┐
        │  accepted  │  │ both kept, surfaced  │
        │            │  │ to the LLM on next   │
        │ confidence │  │ enrichment as a pair │
        │ ≥ 0.7      │  └──────────────────────┘
        └────────────┘
                 ▲
                 │ neutral + confidence ≥ 0.7
                 │ (standalone promotion)
                 │
            ┌────────────┐
            │  rejected  │  confidence < pending_threshold (0.4)
            └────────────┘
```

**Confidence formula** (per fact, recomputed on finalize):

```
0.5 base
  + 0.2 if 2+ unique sessions extracted this fact (multi_source_bonus)
  + 0.1 if NLI ran and found no contradiction   (no_contradiction_bonus)
```

`accept_threshold = 0.7` — single-source facts stay pending until a second session corroborates them or the NLI walk grants the no-contradiction bonus.

## Configuration

`smos.toml` (next to the binary, or via `--config <path>`) is **layered** — sections present in the file override built-in defaults; any omitted section falls back. See [`smos.toml`](smos.toml) for the canonical example and [`smos-adapters/src/config.rs`](smos-adapters/src/config.rs) for every field.

| Section | Purpose |
|---|---|
| `[surreal]` | Embedded RocksDB path + namespace/database. |
| `[server]` | Bind host/port, shutdown grace, extraction toggle, log format. |
| `[[upstream.providers]]` | Multi-provider LLM endpoints (round-robin / failover). |
| `[upstream.strategy]` | `single` / `round_robin` / `failover`. |
| `[llm_extraction]` | Fact extraction LLM (model, temperature, seed). |
| `[embedding]` | Vector embedding model (model, dimensions). |
| `[reranker]` | REQUIRED llama.cpp reranker URL (`/v1/rerank`) for production-quality enrichment. |
| `[retrieval]` | top-K initial/final, `min_topic_chars`, `min_confidence`. |
| `[merge]` | Cosine candidate-selection threshold for merge detection. |
| `[confidence]` | Base + multi-source/no-contradiction bonuses, accept/pending cut. |
| `[nli]` | NLI softmax thresholds (contradiction/entailment). |
| `[nli_backend]` | Native ort/ONNX model id + cache directory. |
| `[extraction]` | Semantic dedup cosine threshold. |
| `[heat]` | Decay rate, min threshold (boosts recently-active facts). |
| `[session]` | Timeout, pending overflow threshold, watcher scan interval. |
| `[audit]` | Optional dreaming agent (schedule, model, mutation caps). |

Secrets: the `[audit].cloud_api_key` field expands `${OPENROUTER_API_KEY}` via `std::env::var` at runtime, so secrets stay out of TOML. The same pattern works for any `api_key` field.

## Subcommands

| Command | Description |
|---|---|
| `smos serve` | Start the HTTP proxy server (session watcher + native NLI). |
| `smos import --from-file <file> --memory-key <key>` | Import a transcript into memory. |
| `smos import --list` | List discoverable opencode sessions. |
| `smos import <session> --dry-run` | Parse turns only, no model calls, no writes. |
| `smos doctor` | Environment validation + SurrealDB stats. |
| `smos doctor --stats` | Quick SurrealDB stats (no model round-trips). |
| `smos doctor --report <path>` | Generate a Markdown report. |
| `smos finalize <session_id> --memory-key <key>` | Manual finalize trigger (single session drain). |
| `smos audit [--provider cloud\|local] [--dry-run]` | One-shot dreaming-agent run in the foreground (independent of `[audit].enabled`, which only gates the in-server cron). `--dry-run` validates config without loading the NLI model. |
| `smos service install` | Install as system service (systemd / Windows service / launchd). |
| `smos service uninstall` | Remove the system service. |
| `smos service start / stop / restart / status` | Control the installed service. |

Global flag: `--config <path>` (defaults to `smos.toml` next to the binary).

## GPU support

Each GPU execution provider is opt-in via a cargo feature flag. Pick **at most one** per build — ort's prebuilt binary matrix cannot satisfy every combination (e.g. CUDA and WebGPU cannot coexist in a single binary).

| Feature flag | Platform | GPU |
|---|---|---|
| *(default)* | All | CPU |
| `nli-directml` | Windows | Intel Arc, AMD, NVIDIA (DirectX 12) |
| `nli-cuda` | Windows, Linux | NVIDIA |
| `nli-metal` | macOS | Apple Silicon |
| `nli-webgpu` | All | Universal (Vulkan / DX12 / Metal) |

At startup `smos serve` logs the detected device and falls back to CPU automatically if the selected provider cannot initialise. HTTP keeps serving even if NLI fails — only the session watcher is disabled.

## Testing

Every test that does **not** carry `#[ignore]` runs under the default `cargo t`. Embedded SurrealDB + wiremock + in-process axum run without external services.

| Alias | Scope | Tests | Time |
|---|---|---|---|
| `cargo tf` | `smos-domain` + `smos-application` only | ~351 | ~2s |
| `cargo t` | All unit tests + embedded-SurrealDB / wiremock e2e | 665 (+ 5 ignored) | ~60s warm |
| `cargo ti` | Alias kept for compat — same scope as `cargo t` | 665 | ~60s warm |
| `cargo tall` | Adds native NLI model tests (643 MB download + live Ollama) | 665 + 5 ignored | ~10 min |

`#[ignore]` is reserved for **external dependencies** (model download, live Ollama). A bug in our own code is never a reason to `#[ignore]` a test — see [`AGENTS.md`](AGENTS.md) for the policy.

**Dev workflow:**

- After editing `smos-domain` or `smos-application`: `cargo tf`
- Before commit: `cargo t`
- Before release: `cargo tall`

**Lint gate (CI-equivalent):**

```bash
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

See [`PERFORMANCE.md`](PERFORMANCE.md) for warm/cold timings and the slowest test binaries.

## Production deployment

- **Storage** — SurrealDB embedded (RocksDB). Set `[surreal].path` to a persistent directory; the proxy creates it on first run. No external database process.
- **Upstream** — point `[[upstream.providers]]` at your OpenAI-compatible endpoints. Multiple providers enable round-robin (`mode = "round_robin"`) or failover (`mode = "failover"`).
- **CORS** — permissive by default (`CorsLayer::permissive()`). Safe on `127.0.0.1`. Before binding to a non-localhost address, configure `[server].allowed_origins`. The startup log warns when the bind is non-localhost with permissive CORS.
- **Graceful shutdown** — Ctrl+C / SIGTERM triggers a coordinated drain: HTTP connections finish → in-flight extraction tasks drain (bounded by `shutdown_extraction_grace_seconds`) → watcher drains pending sessions. Native NLI is in-process and needs no explicit teardown.
- **Pre-warm NLI cache** — `smos finalize sess_dummy --memory-key shared` downloads the 643 MB DeBERTa-v3 ONNX model before the first production request.
- **Service install** — `smos service install` registers the proxy as a systemd unit (Linux), a Windows service, or a launchd job (macOS). Use `smos service start` to bring it up.

### Docker (notes)

SMOS is a single static-ish binary plus a data directory. A minimal container copies the release binary and the `data/` volume; expose `8888`. The DeBERTa-v3 ONNX cache and the SurrealDB RocksDB file both live under `[surreal].path` and `[nli_backend].cache_dir` — mount both as volumes to survive container restarts. A reference Dockerfile is not yet shipped; track [#contributions welcome](#contributing).

## Known limitations

- **First-run download.** The DeBERTa-v3 ONNX export is ~643 MB and lands under `[nli_backend].cache_dir` on first NLI run. Pre-warm with `smos finalize sess_dummy`.
- **Single-process storage.** Embedded SurrealDB holds a single RocksDB lock — run one `smos serve` per data directory. Horizontal scaling needs a shared upstream and per-instance memory keys.
- **Extraction model choice.** Determinism (`temperature = 0`, `seed = 42`) is a property of the configured extraction LLM. Cloud models that ignore the seed will produce drift in `FactId` stability.
- **CORS scope.** Permissive by default for local single-user use. Tighten `[server].allowed_origins` before any non-localhost bind.
- **Audit agent is opt-in.** The dreaming agent is disabled by default and requires a cloud API key (or a sufficiently capable local model) to run.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development setup, architecture overview, testing strategy, code style, and PR process.

Short version: fork → branch → `cargo t` + `cargo clippy -- -D warnings` + `cargo fmt --check` → PR. The lint gate is mandatory.

## License

MIT — see [`LICENSE`](LICENSE).

---

Built by [turbin_y](https://github.com/yurvon-screamo). Feedback, bug reports, and architecture questions are welcome in [Issues](https://github.com/yurvon-screamo/smos/issues).
