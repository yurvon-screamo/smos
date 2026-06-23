<div align="center">

# SMOS — Semantic Memory Operating System

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)
[![npm](https://img.shields.io/npm/v/@yurvon_screamo/smos.svg)](https://www.npmjs.com/package/@yurvon_screamo/smos)

</div>

**An operating system for AI agent memory.** SMOS sits between your AI client and any OpenAI-compatible model, giving it long-term semantic memory — storage, retrieval, consolidation, conflict resolution — without changing a line of client code.

Point your client at `http://localhost:8888/v1`, send `{"model": "bob"}`, and the conversation remembers.

---

## Install

Three ways to get the `smos` binary:

```bash
# Quick — prebuilt binary (no compiler needed)
cargo binstall smos

# From source (compiles, ~15 min)
cargo install --git https://github.com/yurvon-screamo/smos

# npm
npm install -g @yurvon_screamo/smos
```

> Building from source and want GPU acceleration for the inference engine?
> Add a `--features` flag — see [GPU Support](#gpu-support-for-source-builds).

Verify:

```bash
smos --version
```

---

## Setup

After [Install](#install) gives you the `smos` binary, follow these steps in order. Each one says **what** it does and **why** you need it.

### 1. Install dependencies

SMOS talks to one local binary: **`llama-server`** from [llama.cpp](https://github.com/ggerganov/llama.cpp). Build it and put it on `PATH`. The GGUF weights for the three roles SMOS uses (chat + extraction, embedding, reranker) are downloaded automatically by `smos init` — you do not need to fetch them by hand.

**llama.cpp** — runs the chat / extraction LLM, the embedding model, and the cross-encoder reranker that enrichment depends on. Build `llama-server` from <https://github.com/ggerganov/llama.cpp> and put the binary on `PATH`. `smos init` (next step) downloads the Nemotron-3-Nano-4B, Jina-Embeddings-v5, and Qwen3-Reranker GGUFs into `~/.smos/models/` for you.

### 2. Setup

```bash
smos init
```

This single command:

- Creates `~/.smos/` with a default `config.toml`, the working directories (`db/`, `models/`, `persons/`, `logs/`, `reports/`), and a stub persona at `persons/bob.md`.
- Checks for `llama-server` on `PATH`.
- Downloads the GGUF models (~4 GB total) into `~/.smos/models/`:
  - `nemotron-3-nano-4b.gguf` — extraction LLM (from `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF`).
  - `jina-embeddings-v5.gguf` — embedding model (from `jinaai/jina-embeddings-v5-text-small-retrieval-GGUF`).
  - `qwen3-reranker.gguf` — cross-encoder reranker (from `DevQuasar/Qwen.Qwen3-Reranker-0.6B-GGUF`).
- Probes `/health` on the three configured llama-server ports (28081 embedding, 28082 extraction, 28181 reranker).
- Initializes the database (SurrealDB migrations).
- Reports what is ready and what still needs attention.

Already-downloaded models are skipped, so re-running `smos init` only retries the failed ones. Fix any `✗` items shown, then run `smos init` again to verify. For a deeper audit (NLI cache, stats, a Markdown report), run `smos doctor`.

Now edit `~/.smos/config.toml` so it matches your setup: provider URLs, model ids, and which `[persons.*]` identity routes where. See [Configuration](#configuration).

Want to launch the three `llama-server` processes yourself instead of letting `smos serve` auto-spawn them? Run them against the downloaded weights:

```bash
# Embedding server (port 28081)
llama-server --model ~/.smos/models/jina-embeddings-v5.gguf --port 28081

# Chat + extraction server (port 28082)
llama-server --model ~/.smos/models/nemotron-3-nano-4b.gguf --port 28082

# Reranker (port 28181)
llama-server --model ~/.smos/models/qwen3-reranker.gguf --port 28181
```

Or rely on `auto_launch = true` (the default under `[llama_cpp]` in the config) — `smos serve` will spawn them for you (an already-running server on the same port is reused).

### 3. Start

```bash
smos serve
```

The first start downloads the DeBERTa NLI model (~643 MB) into `~/.smos/models/`. Subsequent starts are instant — the model is cached. The proxy listens on `127.0.0.1:8888`.

### 4. Verify it works

```bash
curl http://localhost:8888/health
# → {"status":"ok","version":"0.1.1"}
```

### 5. Install as a service (optional)

```bash
smos service install
```

Registers SMOS as a system service (systemd on Linux, launchd on macOS, Windows Service on Windows) that starts at boot.

```bash
smos service start      # start now
smos service stop       # stop
smos service status     # current state
smos service uninstall  # remove
```

---

## Configuration

Base config lives at `~/.smos/config.toml` (or `smos.toml` next to the binary — both are respected). The most important sections are providers, persons, and persona files.

### Providers and Persons

A **provider** is one upstream OpenAI-compatible endpoint (`llama-server`, OpenRouter, OpenAI, vLLM…). A **person** is a named identity that bundles a memory namespace, a routing target, and an optional persona.

```toml
[[providers]]
name = "llama-local"
url = "http://localhost:28082/v1/chat/completions"
api_key_env = ""                       # env var name; empty = no auth header

[persons.bob]
provider = "llama-local"               # must match a [[providers]].name
model = "nemotron-3-nano-4b"           # upstream model id
persona = "~/.smos/persons/bob.md"     # optional; ~ expands to user home
```

When a client sends `{"model": "bob", ...}`, SMOS:

1. Uses `"bob"` as the **memory isolation key** — extracted facts land under this namespace.
2. Rewrites `model` to the upstream `nemotron-3-nano-4b` and routes to provider `"llama-local"`.
3. Prepends the persona file contents as a `system` message.

A model name that is not a configured person returns HTTP 400 — every request must name a real `[persons.*]` entry.

### Persona file

`~/.smos/persons/bob.md` is plain markdown:

```markdown
You are Bob, a Rust systems programming assistant.
Focus on memory safety and performance.
Be concise. Prefer code over long explanations.
Reply in English.
```

SMOS injects the persona once per conversation as a system message; the client's own messages take over afterwards.

### llama.cpp auto-launch

`auto_launch = true` is the default — `smos serve` spawns the embedding, reranker, and extraction `llama-server` processes itself. Each port is probed first — an already-running server is reused. Disable it only when you launch `llama-server` yourself or route through a remote provider.

```toml
[llama_cpp]
binary = "llama-server"
auto_launch = true

[llama_cpp.embedding]
model_path = "~/.smos/models/jina-embeddings-v5.gguf"
port = 28081
extra_args = ["--ctx-size", "2048"]

[llama_cpp.reranker]
model_path = "~/.smos/models/qwen3-reranker.gguf"
port = 28181
extra_args = ["--ctx-size", "8192"]

[llama_cpp.extraction]
model_path = "~/.smos/models/nemotron-3-nano-4b.gguf"
port = 28082
extra_args = ["--ctx-size", "4096"]
```

### Git sync (optional)

Dual-write every extracted fact to a local git repo as markdown files — backup, versioning, and import onto another machine.

```toml
[git]
repo_url = "git@github.com:user/smos-memory.git"
branch = "main"
auto_push = true                       # push after each session finalize
local_path = "~/.smos/git/memory"
disable_gpg_sign = true
```

On a second machine, re-hydrate the facts with:

```bash
smos import-git git@github.com:user/smos-memory.git
```

Empty `repo_url` disables sync. Secrets stay out of TOML: provider API keys are read from the env var named in `api_key_env`.

### All sections

Any section omitted from your `config.toml` falls back to the built-in default. See [`smos.toml`](smos.toml) for the canonical, fully-commented example.

| Section | Purpose |
|---|---|
| `[surreal]` | Embedded RocksDB path + namespace/database. |
| `[server]` | Bind host/port, shutdown grace, log format. |
| `[[providers]]` | OpenAI-compatible chat-completion endpoints. One per upstream. |
| `[persons.<name>]` | Person = memory key + provider + upstream model + optional persona. |
| `[llm_extraction]` | Fact-extraction LLM (model, temperature, seed, timeout). |
| `[embedding]` | Vector embedding model (model, dimensions, timeout). |
| `[reranker]` | Cross-encoder reranker URL (`/v1/rerank`). |
| `[retrieval]` | top-K initial/final, `min_topic_chars`, `min_confidence`. |
| `[merge]` | Cosine threshold for merge candidate selection. |
| `[confidence]` | Base + multi-source/no-contradiction bonuses, accept/pending cut. |
| `[nli]` | Verdict thresholds (contradiction/entailment). |
| `[nli_backend]` | Native ONNX model id + cache directory. |
| `[extraction]` | Semantic dedup cosine threshold. |
| `[heat]` | Decay rate, min threshold (boosts recently-active facts). |
| `[session]` | Timeout, pending overflow, watcher scan interval. |
| `[audit]` | Optional dreaming agent (schedule, model, mutation caps). |
| `[llama_cpp]` | Auto-launch config for `llama-server` processes. |
| `[git]` | Git-backed memory sync (`repo_url`, `branch`, `auto_push`). |

Global flag: `--config <path>` (resolution order: `--config` → `./smos.toml` → `~/.smos/config.toml`).

---

## Use with AI clients

Any client that speaks the OpenAI Chat Completions API works. Point it at SMOS and use the **person name** as the model.

### opencode

```bash
export OPENAI_BASE_URL=http://localhost:8888/v1
export OPENAI_API_KEY=smos
opencode --model bob
```

### Cursor

Settings → Models → OpenAI API Base URL: `http://localhost:8888/v1`
Model name: `bob`

### Cloud providers (OpenRouter, OpenAI, …)

Add a cloud provider and route a person through it:

```toml
[[providers]]
name = "openrouter"
url = "https://openrouter.ai/api/v1/chat/completions"
api_key_env = "OPENROUTER_API_KEY"

[persons.alice]
provider = "openrouter"
model = "z-ai/glm-5.2"
persona = "~/.smos/persons/alice.md"
```

```bash
export OPENROUTER_API_KEY=sk-or-...
opencode --model alice
```

### curl

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bob","messages":[{"role":"user","content":"hello"}]}'
```

---

## Commands

| Command | Description |
|---|---|
| `smos init` | One-command setup: `~/.smos` bootstrap + GGUF model download + `llama-server` / reranker health probes + DB migrations. Idempotent. |
| `smos serve` | Start the HTTP proxy. |
| `smos doctor` | Validate environment + show SurrealDB stats. |
| `smos doctor --stats` | Quick memory stats (no model round-trips). |
| `smos doctor --report <path>` | Generate a Markdown health report. |
| `smos import --from-file <f>` | Import an opencode transcript JSON into memory. |
| `smos import --list` | List discoverable opencode sessions. |
| `smos import-dir <path>` | Bulk import from a directory (`*.md`, `*.txt`, `*.json`). |
| `smos import-git <url>` | Re-hydrate facts from a git-synced memory repo. |
| `smos finalize <session>` | Manually trigger memory consolidation for one session. |
| `smos audit` | Run the dreaming agent once (memory cleanup / merges / pruning). |
| `smos service install` | Install SMOS as a system service (auto-starts at boot). |
| `smos service start/stop/status/uninstall` | Manage the installed service. |

---

## GPU Support (for source builds)

GPU acceleration for the NLI inference engine is opt-in via a cargo feature flag when installing from source. Pick **at most one** per build.

```bash
cargo install --git https://github.com/yurvon-screamo/smos --features smos/nli-directml
```

| Feature flag | Platform | GPU |
|---|---|---|
| *(default)* | All | CPU |
| `nli-directml` | Windows | Intel Arc, AMD, NVIDIA (DirectX 12) |
| `nli-cuda` | Windows, Linux | NVIDIA |
| `nli-metal` | macOS | Apple Silicon |
| `nli-webgpu` | All | Universal (Vulkan / DX12 / Metal) |

If the selected provider can't initialise at startup, SMOS logs the issue and falls back to CPU automatically — the HTTP server keeps serving.

---

## Inspiration

SMOS builds on academic research in AI agent memory:

- **[MemoryOS: Memory OS of AI Agent](https://arxiv.org/abs/2506.06326)** (Kang et al., 2025) — hierarchical memory management for AI agents. SMOS adopts a similar lifecycle (`pending → accepted → conflict-flagged`) driven by natural-language inference rather than hand-tuned heuristics.

- **[The Price of Meaning: Why Every Semantic Memory System Forgets](https://arxiv.org/html/2603.27116v1)** (2026) — interference is fundamental in semantic memory: every store that decides what to keep also decides what to lose. SMOS sidesteps this by preserving both sides of a contradiction and flagging them, instead of picking a winner.

---

## License

MIT — see [`LICENSE`](LICENSE).
