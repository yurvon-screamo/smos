<div align="center">

# SMOS — Semantic Memory Operating System

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org)
[![npm](https://img.shields.io/npm/v/@yurvon_screamo/smos.svg)](https://www.npmjs.com/package/@yurvon_screamo/smos)

</div>

A memory proxy for AI agents. SMOS sits between any OpenAI-compatible client and any OpenAI-compatible model, giving it long-term semantic memory — storage, retrieval, consolidation, conflict resolution — without changing a line of client code.

Point your client at `http://localhost:8888/v1`, send `{"model": "bob"}`, and the conversation remembers.

---

## Install

```bash
# Prebuilt binary (recommended — no compiler needed)
cargo binstall smos

# From source (compiles, ~15 min)
cargo install --git https://github.com/yurvon-screamo/smos

# npm
npm install -g @yurvon_screamo/smos
```

Verify:

```bash
smos --version
```

### GPU support (source builds only)

GPU acceleration for the inference engine is opt-in via a cargo feature flag when installing from source. Pick **at most one** per build.

| GPU | Feature flag | Platform |
|---|---|---|
| Intel Arc / AMD | `--features smos/nli-directml` | Windows |
| NVIDIA | `--features smos/nli-cuda` | Windows, Linux |
| Apple Silicon | `--features smos/nli-metal` | macOS |
| Universal (Vulkan / DX12 / Metal) | `--features smos/nli-webgpu` | All |
| *(none — default)* | — | All (CPU) |

Example:

```bash
cargo install --git https://github.com/yurvon-screamo/smos --features smos/nli-directml
```

If the selected provider cannot initialise at startup, SMOS logs the issue and falls back to CPU automatically — the HTTP server keeps serving.

---

## Setup

Follow these steps in order after [Install](#install) gives you the `smos` binary.

### 1. Install llama.cpp

SMOS uses [llama.cpp](https://github.com/ggerganov/llama.cpp) for all model inference. Build it and ensure the `llama-server` binary is on your `PATH`. The GGUF weights for the three roles SMOS uses are downloaded automatically in the next step — you do not need to fetch them by hand.

### 2. Initialize

```bash
smos init
```

This single command:

- Creates `~/.smos/` with a default `config.toml`, working directories (`db/`, `models/`, `persons/`, `logs/`, `reports/`), and a stub persona at `persons/bob.md`.
- Checks for `llama-server` on `PATH`.
- Downloads the GGUF models (~4 GB total) into `~/.smos/models/`:
  - `nemotron-3-nano-4b.gguf` — extraction + chat LLM.
  - `jina-embeddings-v5.gguf` — embedding model.
  - `qwen3-reranker.gguf` — cross-encoder reranker.
- Probes `/health` on the three configured ports (28081 embedding, 28082 extraction, 28181 reranker).
- Initializes the database (SurrealDB migrations).
- Reports what is ready and what still needs attention.

Already-downloaded models are skipped, so re-running `smos init` only retries the failed ones. Fix any `✗` items shown, then run `smos init` again to verify. For a deeper audit (NLI cache, stats, a Markdown report), run `smos doctor`.

### 3. Start

```bash
smos serve
```

With `auto_launch = true` (the default), SMOS spawns the three `llama-server` processes itself on first start — an already-running server on the same port is reused. The first start also downloads the DeBERTa NLI model (~643 MB) into `~/.smos/models/`; subsequent starts are instant.

Verify it works:

```bash
curl http://localhost:8888/health
# → {"status":"ok","version":"0.1.2"}
```

### 4. Install as a service (optional)

```bash
smos service install      # auto-starts at boot
smos service start        # start now
smos service status       # current state
smos service stop         # stop
smos service uninstall    # remove
```

Registered as systemd (Linux), launchd (macOS), or a Windows Service.

---

## Configure providers and agents

Providers and agents are declared in `~/.smos/config.toml` (or `smos.toml` next to the binary — both are respected). You can edit the file directly, or inspect the resolved configuration via CLI.

### Inspect current configuration

```bash
smos config show          # full resolved config as TOML (defaults merged in)
smos config providers     # list providers: name → URL
smos config persons       # list agents: name → provider / model
```

These commands are read-only. To change configuration, edit the TOML.

### Add a provider

A **provider** is one upstream OpenAI-compatible endpoint (`llama-server`, OpenRouter, OpenAI, vLLM…).

```toml
[[providers]]
name = "llama-local"
url = "http://localhost:28082/v1/chat/completions"
api_key_env = ""                       # env var name; empty = no auth header

# Cloud example — uncomment and set OPENROUTER_API_KEY in the environment
# [[providers]]
# name = "openrouter"
# url = "https://openrouter.ai/api/v1/chat/completions"
# api_key_env = "OPENROUTER_API_KEY"
```

### Configure an agent (person)

A **person** bundles a memory namespace, a routing target, and an optional persona. When a client sends `{"model": "bob", ...}`, SMOS uses `"bob"` as the memory isolation key, rewrites `model` to the upstream model, and routes to the declared provider.

```toml
[persons.bob]
provider = "llama-local"               # must match a [[providers]].name
model = "nemotron-3-nano-4b"           # upstream model id
persona = "~/.smos/persons/bob.md"     # optional; ~ expands to user home

# [persons.alice]
# provider = "openrouter"
# model = "z-ai/glm-5.2"
# persona = "~/.smos/persons/alice.md"
```

A model name that is not a configured person returns HTTP 400 — every request must name a real `[persons.*]` entry.

### Create a persona file

`~/.smos/persons/bob.md` is plain markdown, injected once per conversation as a `system` message:

```markdown
You are Bob, a Rust systems programming assistant.
Focus on memory safety and performance.
Be concise. Prefer code over long explanations.
Reply in English.
```

---

## Connect your AI client

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
| `smos init` | One-command setup: bootstrap `~/.smos`, download GGUF models, probe `llama-server`, run DB migrations. Idempotent. |
| `smos serve` | Start the HTTP proxy (auto-launches `llama-server` processes). |
| `smos doctor` | Validate environment + show SurrealDB stats. |
| `smos doctor --stats` | Quick memory stats (no model round-trips). |
| `smos doctor --report <path>` | Generate a Markdown health report. |
| `smos doctor --skip-llama` | Skip the `llama-server` + reranker probes. |
| `smos config show` | Print the full resolved configuration as TOML. |
| `smos config providers` | List configured providers (name → URL). |
| `smos config persons` | List configured agents (name → provider / model). |
| `smos import --from-file <f>` | Import an opencode transcript JSON into memory. |
| `smos import --list` | List discoverable opencode sessions. |
| `smos import-dir <path>` | Bulk import from a directory (`*.md`, `*.txt`, `*.json`, …). |
| `smos import-git <url>` | Re-hydrate facts from a git-synced memory repo. |
| `smos finalize <session>` | Manually trigger memory consolidation for one session. |
| `smos audit` | Run the dreaming agent once (memory cleanup / merges / pruning). |
| `smos service install` | Install SMOS as a system service (auto-starts at boot). |

Global flag: `--config <path>` (resolution order: `--config` → `./smos.toml` → `~/.smos/config.toml`).

---

## Configuration reference

All configuration lives in `~/.smos/config.toml`. Any section omitted falls back to the built-in default — see [`smos.toml`](smos.toml) for the canonical, fully-commented example.

| Section | Purpose |
|---|---|
| `[[providers]]` | OpenAI-compatible chat-completion endpoints. One per upstream. |
| `[persons.<name>]` | Person = memory key + provider + upstream model + optional persona. |
| `[llama_cpp]` | Auto-launch config for `llama-server` processes (ports, model paths). |
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
| `[git]` | Git-backed memory sync (`repo_url`, `branch`, `auto_push`). |
| `[surreal]` | Embedded RocksDB path + namespace/database. |
| `[server]` | Bind host/port, shutdown grace, log format. |

### Git sync (optional)

Dual-write every extracted fact to a local git repo as markdown files — backup, versioning, and import onto another machine.

```toml
[git]
repo_url = "git@github.com:user/smos-memory.git"
branch = "main"
auto_push = true
local_path = "~/.smos/git/memory"
disable_gpg_sign = true
```

On a second machine, re-hydrate the facts with `smos import-git <url>`. Empty `repo_url` disables sync. Provider API keys are read from the env var named in `api_key_env`, so secrets never land in TOML.

---

## Inspiration

SMOS builds on academic research in AI agent memory:

- **[MemoryOS: Memory OS of AI Agent](https://arxiv.org/abs/2506.06326)** (Kang et al., 2025) — hierarchical memory management for AI agents. SMOS adopts a similar lifecycle (`pending → accepted → conflict-flagged`) driven by natural-language inference rather than hand-tuned heuristics.
- **[The Price of Meaning: Why Every Semantic Memory System Forgets](https://arxiv.org/html/2603.27116v1)** (2026) — interference is fundamental in semantic memory: every store that decides what to keep also decides what to lose. SMOS sidesteps this by preserving both sides of a contradiction and flagging them, instead of picking a winner.

---

## License

MIT — see [`LICENSE`](LICENSE).
