<div align="center">

# SMOS — Semantic Memory Operating System

**An operating system for AI agent memory.** SMOS manages the full memory lifecycle — storage, retrieval, consolidation, conflict resolution — the way a traditional OS manages processes and files.

Give any OpenAI-compatible client long-term memory without changing a line of its code.

[![Rust](https://img.shields.io/badge/rust-1.96-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-679%2B-green.svg)](#testing)
[![Edition](https://img.shields.io/badge/edition-2024-lightgrey.svg)](https://blog.rust-lang.org/2024/11/28/Rust-2024.html)

</div>

---

## Why SMOS?

- 🔌 **Zero-friction memory.** Point an existing client at `http://localhost:8888/v1` and memory works. No SDK to import, no prompts to rewrite, no client-side code.
- ⚖️ **Contradiction-aware storage.** When a new fact conflicts with an existing one, both versions are preserved and flagged — never silently overwritten. The downstream LLM receives both and decides.
- 🧬 **Semantic deduplication.** Rephrasings of the same fact (`TTL=10 prevents refresh` and `Token lifetime of 10 minutes avoids loops`) collapse into one record.
- 🔁 **Cross-session confirmation.** A fact observed in two independent sessions gets a confidence boost; single-source claims stay pending until corroborated.
- 🦀 **Native Rust, no Python.** Inference runs on `ort` + ONNX Runtime. Storage is embedded SurrealDB (RocksDB + HNSW vector index). One binary, one data directory, no external services.
- 🎮 **Optional GPU.** NLI inference runs on CUDA, DirectML, Metal, or WebGPU — opt-in per build.

## Inspiration

SMOS builds on academic research in AI agent memory:

- **[MemoryOS: Memory OS of AI Agent](https://arxiv.org/abs/2506.06326)** (Kang et al., 2025) — proposes hierarchical memory management for AI agents. SMOS adopts a similar lifecycle (`pending → accepted → conflict-flagged`) with consolidation driven by natural-language inference rather than hand-tuned heuristics.

- **[The Price of Meaning: Why Every Semantic Memory System Forgets](https://arxiv.org/html/2603.27116v1)** (2026) — shows interference is fundamental in semantic memory: every store that decides what to keep also decides what to lose. SMOS sidesteps this by preserving both sides of a contradiction and flagging them, instead of picking a winner.

## Install

```bash
# Option 1: Install from source (compiles, ~15 min)
cargo install --git https://github.com/yurvon-screamo/smos

# Option 2: Prebuilt binary (after v0.1.0 release)
cargo binstall smos

# Option 3: npm
npm install -g @yurvon-screamo/smos
```

## Quick Start

### 1. Prerequisites

- **Rust 1.96+** — `rustup update stable`
- **Ollama** running locally (`ollama serve`) — or any OpenAI-compatible cloud provider
- **GPU** (recommended): Intel Arc, NVIDIA, or Apple Silicon

### Install from source (quickest)

```bash
# Install directly from GitHub (compiles from source, ~15 min)
cargo install --git https://github.com/yurvon-screamo/smos

# With GPU support:
cargo install --git https://github.com/yurvon-screamo/smos --features smos/nli-directml
```

Or build from a local clone:

```bash
git clone https://github.com/yurvon-screamo/smos.git
cd smos
```

### 2. Build

```bash
# Default build — CPU inference
cargo build --release --bin smos

# Or pick one GPU feature for NLI acceleration:
cargo build --release --bin smos --features smos/nli-directml   # Windows + Intel Arc / AMD / NVIDIA
cargo build --release --bin smos --features smos/nli-cuda      # Windows / Linux + NVIDIA
cargo build --release --bin smos --features smos/nli-metal     # macOS + Apple Silicon
cargo build --release --bin smos --features smos/nli-webgpu    # universal (Vulkan/DX12/Metal)
```

### 3. Resource requirements

| Component | Disk | RAM |
|---|---|---|
| SMOS binary | ~76 MB | ~100 MB |
| NLI model (auto-download) | ~643 MB | ~700 MB |
| Build artifacts (temp) | ~2 GB | — |

### 4. Initialize

```bash
smos init      # creates ~/.smos with a default config + a stub persona
smos doctor    # environment validation + SurrealDB stats
```

`smos init` is idempotent. It creates every well-known subdirectory (`db/`, `models/`, `persons/`, `git/`, `logs/`, `reports/`, `cache/`) and never overwrites an existing `config.toml`. Re-running it is always safe.

### 5. Configure

Edit `~/.smos/config.toml` (or `smos.toml` next to the binary — both are respected). Minimal working setup:

```toml
[[providers]]
name = "ollama-local"
url = "http://localhost:11434/v1/chat/completions"
api_key_env = ""                       # env var name; empty = no auth header sent

[persons.bob]
provider = "ollama-local"              # must match a [[providers]].name
model = "granite4.1:3b"               # upstream model id
persona = "~/.smos/persons/bob.md"     # optional; ~ expands to user home

[llm_extraction]
url = "http://localhost:11434"
model = "qwen3.5:2b"
temperature = 0.0
seed = 42

[embedding]
url = "http://localhost:11434"
model = "hf.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:latest"
dimensions = 1024

[reranker]
url = "http://localhost:8181"
model = "qwen3-reranker"
timeout_seconds = 60

[nli_backend]
model = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# cache_dir defaults to ~/.smos/models; first startup downloads the ONNX model (~643 MB)
```

See the [Configuration](#configuration) table for every section and [`smos.toml`](smos.toml) for the canonical example.

Persona file (`~/.smos/persons/bob.md`) is plain markdown:

```markdown
You are Bob, a Rust systems programming assistant.
Focus on memory safety and performance.
Be concise. Prefer code over long explanations.
Reply in English.
Cite the crate and version when suggesting a dependency.
```

SMOS injects the persona once per conversation as a system message; the client's own messages take over afterwards.

### 6. Run

```bash
# Pull the Ollama models referenced by the config above:
ollama pull granite4.1:3b
ollama pull qwen3.5:2b
ollama pull hf.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:latest

# Start the reranker server:
llama-server --model qwen3-reranker-0.6b-q8_0.gguf --port 8181

# Start SMOS:
smos serve

# Health check:
curl http://localhost:8888/health
# → {"status":"ok","version":"0.1.0"}
```

SMOS listens on `127.0.0.1:8888` by default. The first startup downloads the NLI ONNX model (~643 MB) into `[nli_backend].cache_dir`.

### 7. Connect

Point any OpenAI-compatible client at SMOS:

```bash
export OPENAI_BASE_URL=http://localhost:8888/v1
export OPENAI_API_KEY=smos
opencode --model bob          # "bob" is the person name from config
```

Or with curl:

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bob","stream":true,"messages":[{"role":"user","content":"hello"}]}'
```

## Persons & Providers

SMOS routes by **persons**, not by parsing a `memory_key:model` string. A person is simultaneously a memory namespace, a routing target, and (optionally) a persona file.

```toml
[[providers]]
name = "ollama-local"
url = "http://localhost:11434/v1/chat/completions"
api_key_env = ""

[persons.bob]
provider = "ollama-local"
model = "granite4.1:3b"
persona = "~/.smos/persons/bob.md"
```

When a client sends `{"model": "bob", ...}`, SMOS:

1. Uses `"bob"` as the **memory isolation key** — extracted facts land under this namespace.
2. Looks up `[persons.bob]` → rewrites `request.model` to the upstream `granite4.1:3b`.
3. Routes the rewritten request to the matching `[[providers]]` entry by `provider` name.
4. Prepends the persona file contents as a `system` message when `persona` is set.

A model name that is not a configured person returns HTTP 400 — every request must name a real `[persons.*]` entry. Add a person (or point `provider` at an existing `[[providers]]` entry) before sending traffic.

## How it works

```
                ┌─────────────────────────────────────────────────────┐
   OpenAI       │  1. ROUTE     resolve person → provider + model     │
   client  ───▶ │              + persona rewrite                       │
                │                                                      │
                │  2. ENRICH    embed topic → vector search → rerank   │
                │              → dedup → inject <smos-memory> block    │
                │                                                      │
                │  3. FORWARD   rewrite → upstream provider (SSE)      │
                └────────────────────────┬─────────────────────────────┘
                                         │
                          response streamed back to client
                                         │
                ┌────────────────────────▼─────────────────────────────┐
                │  4. EXTRACT   pull candidate facts from the response  │
                │              (off the request path)                   │
                │                                                      │
                │  5. FINALIZE  on session timeout — consolidation,    │
                │              merges, conflict flags, confidence      │
                │              promotion; optional git-sync push       │
                │                                                      │
                │  6. AUDIT     optional dreaming-agent cron run that  │
                │              reviews stored facts and prunes         │
                └─────────────────────────────────────────────────────┘
```

Steps 1–3 run on the request path; the response reaches the client as soon as the upstream stream completes. Steps 4–6 run asynchronously, so a flaky memory subsystem never blocks the chat.

## Commands

| Command | Description |
|---|---|
| `smos init` | Create `~/.smos` with default config + stub persona. Idempotent. |
| `smos serve` | Start the HTTP proxy (session watcher + native NLI + optional llama.cpp auto-launch). |
| `smos import <session> [--memory-key <k>]` | Import an opencode session transcript into memory. |
| `smos import --from-file <f> [--memory-key <k>]` | Import from a local opencode-export JSON file. |
| `smos import --list` | List discoverable opencode sessions. |
| `smos import <session> --dry-run` | Parse turns only, no model calls, no writes. |
| `smos import-git <url>` | Import facts from a git repo written by SMOS git-sync. |
| `smos doctor` | Environment validation + SurrealDB stats. |
| `smos doctor --stats` | Quick SurrealDB stats (no model round-trips). |
| `smos doctor --report <path>` | Generate a Markdown report. |
| `smos finalize <session> [--memory-key <k>]` | Manual single-session consolidation drain trigger. |
| `smos audit [--provider cloud\|local] [--dry-run]` | One-shot dreaming-agent run in the foreground. |
| `smos service install / uninstall / start / stop / status` | Manage SMOS as a system service (systemd / Windows / launchd). |

Global flag: `--config <path>` (defaults to `./smos.toml`, then `~/.smos/config.toml`).

## Configuration

`smos.toml` is **layered** — sections present in the file override built-in defaults; any omitted section falls back. Resolution order: `--config <path>` → `./smos.toml` (CWD) → `~/.smos/config.toml`. See [`smos.toml`](smos.toml) for the canonical example.

| Section | Purpose |
|---|---|
| `[surreal]` | Embedded RocksDB path (defaults to `~/.smos/db/smos.db`) + namespace/database. |
| `[server]` | Bind host/port, shutdown grace, extraction toggle, graceful degradation, log format. |
| `[[providers]]` | OpenAI-compatible LLM chat-completion endpoints. One entry per upstream. |
| `[persons.<name>]` | Person = memory key + provider + upstream model + optional persona file. Drives routing. |
| `[llm_extraction]` | Fact extraction LLM (model, temperature, seed, timeout). |
| `[embedding]` | Vector embedding model (model, dimensions, timeout). |
| `[reranker]` | Cross-encoder reranker URL (`/v1/rerank`) used by the enrich step. |
| `[retrieval]` | top-K initial/final, `min_topic_chars`, `min_confidence`. |
| `[merge]` | Cosine candidate-selection threshold for merge detection. |
| `[confidence]` | Base + multi-source/no-contradiction bonuses, accept/pending cut. |
| `[nli]` | Verdict thresholds (contradiction/entailment). |
| `[nli_backend]` | Native ort/ONNX model id + cache directory (`~/.smos/models`). |
| `[extraction]` | Semantic dedup cosine threshold. |
| `[heat]` | Decay rate, min threshold (boosts recently-active facts). |
| `[session]` | Timeout, pending overflow threshold, watcher scan interval. |
| `[audit]` | Optional dreaming agent (schedule, model, mutation caps). |
| `[llama_cpp]` | Auto-launch config for `llama-server` (embedding / reranker / extraction models). |
| `[git]` | Git-backed memory sync (`repo_url`, `branch`, `auto_push`, `local_path`). |

**Secrets:** each `[[providers]]` entry uses `api_key_env` to name the environment variable holding the API key (empty = no auth header), so secrets stay out of TOML. `[audit].cloud_api_key` expands `${OPENROUTER_API_KEY}` via `std::env::var` at runtime.

### llama.cpp auto-launch

When `[llama_cpp].auto_launch = true`, `smos serve` spawns the configured `llama-server` processes at startup. Each service's port is probed first; an already-running server is reused. Disable it if you launch `llama-server` yourself or use a remote/cloud provider.

```toml
[llama_cpp]
binary = "llama-server"
auto_launch = false

[llama_cpp.reranker]
model_path = "~/.smos/models/qwen3-reranker.gguf"
port = 8181
extra_args = ["--ctx-size", "8192"]
```

### Git-backed memory sync

When `[git].repo_url` is non-empty, SMOS dual-writes extracted facts to a local clone of the repo as markdown files (one fact per `.md`) and commits them after every `FinalizeSession`. It also pushes when `auto_push = true` (off by default, so you control when memory leaves the box). A second SMOS instance re-hydrates the facts via `smos import-git <url>`. An empty `repo_url` disables sync; `local_path` still works for offline use.

```toml
[git]
repo_url = ""
branch = "main"
auto_push = false
local_path = "~/.smos/git/memory"
disable_gpg_sign = true              # SMOS commits are unsigned by default
```

## GPU Support

Each GPU execution provider is opt-in via a cargo feature flag. Pick at most one per build — ort's prebuilt binary matrix cannot satisfy every combination (e.g. CUDA and WebGPU cannot coexist in a single binary).

| Feature flag | Platform | GPU |
|---|---|---|
| *(default)* | All | CPU |
| `nli-directml` | Windows | Intel Arc, AMD, NVIDIA (DirectX 12) |
| `nli-cuda` | Windows, Linux | NVIDIA |
| `nli-metal` | macOS | Apple Silicon |
| `nli-webgpu` | All | Universal (Vulkan / DX12 / Metal) |

At startup `smos serve` logs the detected device and falls back to CPU automatically if the selected provider cannot initialise. HTTP keeps serving even if NLI fails — only the session watcher is disabled.

## Known Limitations

- **Single-source facts stay pending.** A fact extracted from only one session caps at confidence 0.5. Promotion to `accepted` (threshold 0.7) requires multi-source confirmation (the same fact extracted in 2+ independent sessions) or a merge during finalize.
- **NLI recall ~50%.** The NLI model catches about half of semantic duplicates (paraphrased facts). The semantic dedup layer (cosine ≥ 0.95) catches most of the rest, but some near-duplicates may slip through as independent facts.
- **Context window limit.** The dreaming agent loads all facts for a memory key into a single LLM prompt. For 100+ facts in one namespace, this may exceed the model's context window.
- **No multi-user isolation.** SMOS partitions by memory key (person name), not by user. All clients sharing a SMOS instance see the same namespaces.

## Testing

Every test that does not carry `#[ignore]` runs under the default `cargo t`. Embedded SurrealDB + wiremock + in-process axum run without external services.

| Alias | Scope | Tests | Time |
|---|---|---|---|
| `cargo tf` | `smos-domain` + `smos-application` only | ~350 | ~2s |
| `cargo t` | All unit tests + embedded-SurrealDB / wiremock e2e | 679 (+ 6 ignored) | ~60s warm |
| `cargo ti` | Alias kept for compat — same scope as `cargo t` | 679 | ~60s warm |
| `cargo tall` | Adds native NLI model tests (643 MB download + live Ollama) | 679 + 6 ignored | ~10 min |

`#[ignore]` is reserved for **external dependencies** (model download, live Ollama). A bug in our own code is never a reason to `#[ignore]` a test — see [`AGENTS.md`](AGENTS.md) for the policy.

**Dev workflow:** domain/application edits → `cargo tf`; pre-commit → `cargo t`; pre-release → `cargo tall`.

## License

MIT — see [`LICENSE`](LICENSE).

---

Built by [turbin_y](https://github.com/yurvon-screamo). Feedback, bug reports, and architecture questions are welcome in [Issues](https://github.com/yurvon-screamo/smos/issues).
