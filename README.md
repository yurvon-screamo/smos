<div align="center">

# SMOS — Semantic Memory Operating System

**An OpenAI-compatible memory proxy that gives any AI coding agent persistent long-term memory — without code changes, without an MCP server, without a framework.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.96%20edition%202024-orange.svg)](https://www.rust-lang.org)
[![crates.io](https://img.shields.io/crates/v/smos.svg)](https://crates.io/crates/smos)
[![npm](https://img.shields.io/npm/v/@yurvon_screamo/smos.svg)](https://www.npmjs.com/package/@yurvon_screamo/smos)
[![Release](https://github.com/yurvon-screamo/smos/actions/workflows/release.yml/badge.svg)](https://github.com/yurvon-screamo/smos/actions/workflows/release.yml)

</div>

## Quick start

```bash
npm install -g @yurvon_screamo/smos   # or: cargo binstall smos
smos init                              # one-time: downloads ~4 GB of local models
smos serve                             # starts on http://localhost:8888
```

Point Cursor (or Claude Code, opencode, Cline, Aider, Continue.dev) at
`http://localhost:8888/v1` and use `bob` as the model name. That assistant
now remembers across sessions.

**One prerequisite:** [`llama-server`](https://github.com/ggerganov/llama.cpp)
on your `PATH`. SMOS uses it to run three tiny models locally — extraction,
embeddings, reranking. The largest is 4B parameters. These run on a laptop
CPU with integrated graphics — no GPU, no API keys, no cloud bills, no data
leaving your machine. Prefer cloud providers instead? SMOS supports that
too — see [Configure](#configure).

---

Open a new chat in Cursor and your assistant starts from scratch. Switch to
Claude Code or opencode and you re-explain why the cache TTL is 10 seconds,
not 60 — your architecture, your conventions, every decision you already
made. The model is stateless. The tool is replaceable. The memory should
not be.

SMOS fixes this. It is a transparent proxy that sits between your AI client
and the upstream LLM. Every response is mined for facts automatically — the
agent does nothing, the agent forgets nothing. Point any OpenAI-compatible
client at SMOS and your assistant remembers across sessions, across tools,
across model swaps. Works with local llama.cpp, OpenAI, OpenRouter, vLLM —
any OpenAI-compatible upstream. Run fully local for privacy, or point it
at your existing cloud provider.

---

## How it works

```
Client ──▶ SMOS ──▶ upstream LLM (GPT-4o, Claude, local, …)
              │
              ├── 1. ENRICH    inject relevant facts into the request
              ├── 2. FORWARD   stream response back at full LLM speed
              ├── 3. EXTRACT   mine the response for facts (after delivery)
              └── 4. FINALIZE  DeBERTa NLI resolves merges and conflicts
                                (after delivery)
```

Steps 3 and 4 run **off the request path** — the client receives the
response as soon as the upstream LLM finishes. Extraction and consolidation
never add latency. If any step fails, the system degrades gracefully: the
request forwards unenriched, facts stay pending for the next cycle, HTTP
keeps serving.

For the full pipeline, memory lifecycle, and NLI internals, see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Why SMOS

- **Memory is part of the API, not a tool.** Every response is mined for
  facts automatically. The agent cannot forget to save, because the agent
  is not involved in saving. Extraction runs off the request path — zero
  added latency.
- **No external database.** Embedded SurrealDB (RocksDB + HNSW vector
  index). No Postgres, no Neo4j, no Qdrant, no Docker. One binary, one
  directory.
- **Contradictions are detected, not overwritten.** A DeBERTa-v3 NLI model
  evaluates each merge candidate. Both sides of a contradiction are
  preserved and surfaced to the LLM — not silently overwritten. The
  theoretical basis: ["The Price of Meaning"](https://arxiv.org/abs/2603.27116)
  (2026) proves vector-only retrieval degrades through semantic
  interference; external verification is necessary.
- **Multi-persona isolation.** Bob for Rust, Alice for ML, Charlie for
  DevOps — each a separate memory namespace. One SMOS instance, N isolated
  assistants.
- **Runs on any laptop.** Three tiny local models (4 GB total) handle
  extraction, embeddings, and reranking on CPU. Tested on a laptop with
  integrated Intel graphics — no GPU, no API keys, no cloud bills. Your
  conversations never leave your machine.

---

## Persons: name your assistant

Every AI client sends a `model` field in the request. SMOS uses that field
as a **person name** — and each person is a memory namespace, a routing
target, and an optional persona.

When Cursor sends `{"model": "bob", ...}`, SMOS:

1. Uses `"bob"` as the memory isolation key
2. Rewrites `model` to the upstream model declared for Bob
3. Routes the request to Bob's provider
4. Injects Bob's persona as a system message
5. Enriches the request with facts from Bob's memory namespace

Create **Alice** for ML engineering and **Charlie** for DevOps — each with
its own memory, provider, and persona. Alice never mixes your Rust types
with your Python pipelines. Swap GPT-4o for a local model and Bob stays
Bob — identity lives at the OS layer, not in a chat log you rebuild by
hand.

See [Configure → Agents (persons)](#agents-persons) for the TOML.

---

## What you need

- **~5 GB disk** for local models (one-time download: 4 GB GGUF + 643 MB
  DeBERTa NLI).
- **`llama-server`** on your `PATH` — get it from
  [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
  or build from source. Runs on any modern laptop; GPU is optional, not
  required.
- **Any OpenAI-compatible AI client** — Cursor, Claude Code, opencode,
  Cline, Continue.dev, Aider, Windsurf, or plain `curl`.

No Postgres. No Neo4j. No Docker. No cloud account. No API key (unless you
choose to use a cloud provider as your upstream).

---

## Install

### Prebuilt binary (recommended)

```bash
cargo binstall smos
```

The same binary runs on CPU and (when one is detected) on the host GPU.
SMOS probes the hardware at startup, downloads the matching ONNX Runtime
shared library into `~/.smos/models/ort/` on first use (~5–300 MB
depending on the device), and falls back to CPU if no GPU is available.
No feature flags, no per-vendor rebuild.

Detected devices:

- **Windows** — CUDA (NVIDIA only), DirectML (Intel Arc, AMD, NVIDIA via
  DirectX 12), CPU fallback.
- **Linux** — CUDA (NVIDIA), CPU fallback.
- **macOS** — Metal / CoreML on Apple Silicon, CPU fallback.

Override the probe by setting `[nli_backend].device = "cpu" | "directml"
| "cuda" | "metal"` in `~/.smos/config.toml`.

### From source

```bash
cargo install smos
```

### npm

```bash
npm install -g @yurvon_screamo/smos
```

Verify:

```bash
smos --version
```

---

## Setup

### Step 1 — Get llama-server

SMOS uses [llama.cpp](https://github.com/ggerganov/llama.cpp) to run three
tiny models locally — a 4B extraction LLM, an embedding model, and a
reranker. These are small enough to run on a laptop CPU with integrated
graphics. No GPU required.

**Quickest path:**

- Download a prebuilt binary from
  [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
  (look for `llama-server` in the assets for your platform).
- Or build from source:
  `git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build --config Release`
- Ensure `llama-server` is on your `PATH` (`llama-server --help` should
  work from any directory).

The GGUF model weights for the three roles are downloaded automatically in
the next step — you do not need to fetch them by hand.

> **Prefer cloud?** Skip llama-server entirely. Set `[llama_cpp].auto_launch
> = false` in `~/.smos/config.toml` and point `[llm_extraction]`,
> `[embedding]`, and `[reranker]` at any OpenAI-compatible cloud provider.
> See [Configure](#configure).

### Step 2 — Initialize

```bash
smos init
```

This single command:

- Creates `~/.smos/` with a default `config.toml`, working directories
  (`db/`, `models/`, `persons/`, `logs/`, `reports/`), and a stub persona at
  `persons/bob.md`.
- Checks for `llama-server` on `PATH`.
- Downloads the GGUF models (~4 GB total) into `~/.smos/models/`:
  - `nemotron-3-nano-4b.gguf` — extraction + chat LLM.
  - `jina-embeddings-v5.gguf` — embedding model.
  - `qwen3-reranker.gguf` — cross-encoder reranker.
- Probes `/health` on the three configured ports (28081 embedding, 28082
  extraction, 28181 reranker).
- Initializes the database (SurrealDB migrations).
- Reports what is ready and what still needs attention.

Already-downloaded models are skipped, so re-running `smos init` only retries
the failed ones. Fix any `✗` items shown, then run `smos init` again to verify.
For a deeper audit (NLI cache, stats, a Markdown report), run `smos doctor`.

### Step 3 — Start

```bash
smos serve
```

With `auto_launch = true` (the default), SMOS spawns the three `llama-server`
processes itself on first start — an already-running server on the same port is
reused. The first start also downloads the DeBERTa NLI model (~643 MB) into
`~/.smos/models/`; subsequent starts are instant.

Verify it works:

```bash
curl http://localhost:8888/health
# → {"status":"ok","version":"0.1.7"}
```

### Step 4 — Install as a service (optional)

```bash
smos service install      # auto-starts at boot
smos service start        # start now
smos service status       # current state
smos service stop         # stop
smos service uninstall    # remove
```

Registered as systemd (Linux), launchd (macOS), or a Windows Service.

---

## Configure

All configuration lives in `~/.smos/config.toml`. `smos init` creates it with
safe defaults; edit the file by hand from there. Any section omitted falls back
to the built-in default.

### Inspect current configuration

```bash
smos config show          # full resolved config as TOML (defaults merged in)
smos config providers     # list providers: name → URL
smos config persons       # list agents: name → provider / model
```

These commands are read-only. To change configuration, edit the TOML.

### Providers

A **provider** is one upstream OpenAI-compatible endpoint (`llama-server`,
OpenRouter, OpenAI, vLLM…). One entry per upstream; there is no round-robin or
failover — routing is per-agent.

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

### Agents (persons)

A **person** bundles a memory namespace, a routing target, and an optional
persona. When a client sends `{"model": "bob", ...}`, SMOS uses `"bob"` as the
memory isolation key, rewrites `model` to the upstream model, and routes to the
declared provider.

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

A model name that is not a configured person returns HTTP 400 — every request
must name a real `[persons.*]` entry.

### Persona files

`~/.smos/persons/bob.md` is plain markdown, injected once per conversation as a
`system` message:

```markdown
You are Bob, a Rust systems programming assistant.
Focus on memory safety and performance.
Be concise. Prefer code over long explanations.
Reply in English.
```

### Git memory sync (optional)

Dual-write every extracted fact to a local git repo as markdown files — backup,
versioning, and re-hydration onto another machine. Empty `repo_url` disables
sync.

```toml
[git]
repo_url = "git@github.com:user/smos-memory.git"
branch = "main"
auto_push = true
local_path = "~/.smos/git/memory"
disable_gpg_sign = true
```

On a second machine, re-hydrate the facts with `smos import-git <url>`. Provider
API keys are read from the env var named in `api_key_env`, so secrets never land
in TOML.

### Advanced: llama.cpp auto-launch

By default, `smos serve` spawns the three `llama-server` processes itself and
reuses any server already bound to the configured port. Override the binary,
ports, model paths, or extra CLI args here; flip `auto_launch = false` if you
launch `llama-server` yourself or use a remote / cloud provider.

```toml
[llama_cpp]
binary = "llama-server"
auto_launch = true
# Unload models from VRAM after this many seconds idle (5 min default).
# Set to 0 to disable. Only appended when llama-server supports the flag.
idle_timeout_seconds = 300

[llama_cpp.embedding]
model_path = "~/.smos/models/jina-embeddings-v5.gguf"
port = 28081
extra_args = ["--ctx-size", "2048", "--embeddings"]

[llama_cpp.reranker]
model_path = "~/.smos/models/qwen3-reranker.gguf"
port = 28181
extra_args = ["--ctx-size", "8192"]

[llama_cpp.extraction]
model_path = "~/.smos/models/nemotron-3-nano-4b.gguf"
port = 28082
extra_args = ["--ctx-size", "4096"]
```

### Full configuration reference

See [`smos.toml`](smos.toml) for the canonical, fully-commented example.

| Section | Purpose |
|---|---|
| `[[providers]]` | OpenAI-compatible chat-completion endpoints. One per upstream. |
| `[persons.<name>]` | Person = memory key + provider + upstream model + optional persona. |
| `[git]` | Git-backed memory sync (`repo_url`, `branch`, `auto_push`). |
| `[llama_cpp]` | Auto-launch config for `llama-server` processes (ports, model paths). |
| `[llm_extraction]` | Fact-extraction LLM (model, temperature, seed, timeout). |
| `[embedding]` | Vector embedding model (model, dimensions, timeout). |
| `[reranker]` | Cross-encoder reranker URL (`/v1/rerank`). |
| `[retrieval]` | top-K initial/final, `min_topic_chars`, `min_confidence`. |
| `[merge]` | Cosine threshold for merge candidate selection. |
| `[confidence]` | Base + multi-source/no-contradiction bonuses, accept/pending cut. |
| `[nli]` | Verdict thresholds (contradiction/entailment). |
| `[nli_backend]` | Native ONNX model id + cache directory + device selection. |
| `[extraction]` | Semantic dedup cosine threshold. |
| `[heat]` | Decay rate, min threshold (boosts recently-active facts). |
| `[session]` | Timeout, pending overflow, watcher scan interval. |
| `[audit]` | Optional dreaming agent (schedule, model, mutation caps). |
| `[surreal]` | Embedded RocksDB path + namespace/database. |
| `[server]` | Bind host/port, shutdown grace, log format. |

---

## Connect your AI client

Any client that speaks the OpenAI Chat Completions API works — Cursor,
Claude Code, opencode, Cline, Continue.dev, Aider, Windsurf, and anything
else that lets you set a custom base URL. Point it at SMOS and use the
**person name** as the model.

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

For other OpenAI-compatible clients, the pattern is the same: set the
base URL to `http://localhost:8888/v1`, set any API key (SMOS does not
validate it by default), and use the person name as the model.

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
| `smos import raw "<text>"` | Extract facts from arbitrary free-form text. |
| `smos import raw --stdin` | Same, reading the text body from stdin. |
| `smos finalize <session>` | Manually trigger memory consolidation for one session. |
| `smos audit` | Run the dreaming agent once (memory cleanup / merges / pruning). |
| `smos service install` | Install SMOS as a system service (auto-starts at boot). |

Global flag: `--config <path>` to point at a non-default config file.

---

## Known limitations

Honest scope, not marketing hedging:

- **643 MB DeBERTa-v3 ONNX download on first start.** Subsequent starts
  are instant. The model is cached under `~/.smos/models/`.
- **`llama-server` on `PATH` for local inference.** SMOS auto-launches
  the three `llama-server` processes (extraction, embedding, reranker)
  when `auto_launch = true`. The models are tiny (4 GB total) and run on
  CPU. To use cloud providers instead, set `auto_launch = false` and
  point the extraction / embedding / reranker URLs at your provider.
- **Extraction model is English-optimized.** Nemotron-3-Nano-4B is
  multilingual, but accuracy is highest on English. The DeBERTa NLI model
  is English-only.
- **Single-process SurrealDB lock.** One SMOS instance per database path.
  No built-in horizontal scaling. Multi-machine sync via the git backend.
- **Not benchmarked on LOCOMO.** The NLI contradiction detection is the
  architectural choice, not a benchmark number.

---

## Inspiration

SMOS builds on academic research in AI agent memory:

- **[MemoryOS: Memory OS of AI Agent](https://arxiv.org/abs/2506.06326)**
  (Kang et al., 2025, EMNLP 2025 Oral) — hierarchical memory management
  for AI agents. SMOS adopts a similar lifecycle
  (`pending → accepted → conflict-flagged`) driven by natural-language
  inference rather than hand-tuned heuristics.
- **[The Price of Meaning: Why Every Semantic Memory System Forgets](https://arxiv.org/abs/2603.27116)**
  (Ray Barman et al., 2026) — interference is fundamental in semantic
  memory: every store that decides what to keep also decides what to
  lose, and pure vector retrieval is mathematically proven to degrade.
  SMOS sidesteps this by preserving both sides of a contradiction and
  flagging them, instead of picking a winner — and by layering DeBERTa
  NLI on top of cosine retrieval as the external verification the paper
  calls necessary.

---

## License

MIT — see [`LICENSE`](LICENSE).
