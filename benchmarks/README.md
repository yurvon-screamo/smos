# SMOS × BEAM benchmark harness

Fork of [`mem0ai/memory-benchmarks`](https://github.com/mem0ai/memory-benchmarks)
(commit `4b61c5d31b9c668a12b4f5e78064248a02c82d2b`, `main`) rewired to drive
**SMOS** instead of Mem0. The BEAM benchmark (arXiv:2510.27246) measures
long-term-memory recall across 100 conversations × 4 size buckets
(100K–10M tokens) × 20 probing questions.

## What changed vs upstream

Only two integration points were touched — the BEAM scoring/judging/prompts
code is **untouched**:

1. **`benchmarks/common/smos_client.py`** (new) — `SMOSClient`, a drop-in async
   adapter presenting the `Mem0Client` interface (`add` / `search` /
   `delete_user` + async context manager). Each operation shells out to the
   unified `smos` binary against one benchmark config.
2. **`benchmarks/beam/run.py`** — `Mem0Client(...)` → `SMOSClient(...)`, plus a
   single `await mem0.finalize_pending()` hook between Phase 1 (ingest) and
   Phase 2 (search + answer + judge). No other edits.

`prompts.py`, `metrics.py`, `schema.py`, `llm_client.py`, `utils.py` are
verbatim from upstream.

## How SMOS maps onto the BEAM contract

| BEAM op (`Mem0Client`) | SMOS CLI                             | Notes                                                                                    |
|------------------------|--------------------------------------|------------------------------------------------------------------------------------------|
| `add(messages, user_id)`   | `smos import raw "<text>" --memory-key <sanitize(user_id)>` | Concatenates message contents. Parses `Memory key:` / `Session:` from stdout. |
| `search(query, user_id, top_k)` | `smos search "<query>" --person <sanitize(user_id)> --top-k <top_k>` | Returns SMOS' rerank-ordered JSON array (score = rerank relevance, higher = better). |
| `delete_user(user_id)`     | no-op                                | Isolation via a unique sanitized `memory_key` per BEAM `user_id`.                        |
| finalize (between phases)  | `smos finalize <session_id>`         | Discovery path (no `--memory-key`); promotes pending facts to Accepted so `search` sees them. |

`smos search` returns **Accepted** facts only, and freshly-extracted facts
land as **Pending** — so the finalize hook between ingest and search is
**mandatory**, not optional.

## Concurrency: strictly sequential

SMOS' RocksDB store takes a **single-writer lock** at `SurrealStore::connect`
time. Two concurrent `smos` subprocesses against the same `[surreal].path`
will contend on that lock. The BEAM `run.py` loop is itself a sequential
for-loop, so keep it that way: **one `smos` subprocess at a time** per
database path. This is why `SMOSClient` runs each op as a fresh subprocess
(connect → op → drop) rather than holding a long-lived connection.

## Subprocess timeout

`SMOSClient` waits up to `SMOS_SUBPROCESS_TIMEOUT` seconds per subprocess
(default **1800 s / 30 min**, overridable via the env var). The default is
deliberately high: on a 1M run, `finalize_pending` collapses every BEAM user
into ONE `smos finalize` subprocess (see "Finalize model" below), which
reloads the 643 MB DeBERTa NLI model and drains every pending fact across
every memory_key before returning. Lower the value only for smoke runs.

## Finalize model (one shared session id)

`smos import raw` derives a **deterministic** session id
(`derive_session_id("raw-import")`), so every BEAM `user_id` lands facts under
the SAME session id (in distinct `memory_key` namespaces). `finalize_pending`
dedupes by session id and runs `smos finalize <sid>` **once** via the
discovery path (no `--memory-key`), which scans every `memory_key` that
session touched and promotes their pending facts to Accepted in a single pass.
Finalize is idempotent, so a timeout-interrupted run is safe to resume.

### Known limitation (1M scale)

The shared session id means the single `session` row accumulates provenance
across ALL BEAM users, and each `smos import raw` re-reads that growing
document. This is a pre-existing property of `import raw` (not introduced by
this integration) and is out of scope to change here. On a full 1M run it adds
per-chunk overhead; if it becomes a bottleneck, the fix is per-conversation
session derivation in `import raw` (future work, explicitly deferred).

## Prerequisites

1. **SMOS binary on PATH** (or set `SMOS_BINARY=/path/to/smos`). Build with
   `cargo build --release --bin smos` in the repo root.
2. **Three llama-server endpoints** running (the SMOS-internal models, NOT the
   BEAM reader/judge):
   - embedding: `http://localhost:28081` (Jina v5, dim 1024)
   - reranker:  `http://localhost:28181` (Qwen3-Reranker)
   - extraction: `http://localhost:28082` (Nemotron-3-Nano-4B)
3. **DeBERTa-v3 NLI model** cached in `~/.smos/models` (643 MB; downloaded on
   first `smos finalize`). Required because finalize runs the NLI classifier
   to promote pending facts to Accepted.
4. **Python deps**: `pip install -r benchmarks/requirements.txt`.
5. **BEAM reader/judge LLMs**: set via the harness env vars (see "Running"
   below) — these are the models the user is fetching.

## Configuration

`smos.bench.toml` (repo root) is the benchmark config. It uses a **separate**
RocksDB path (`~/.smos/beam-rocksdb/beam.db`) so the benchmark never collides
with the user's main SMOS store, and wide retrieval
(`top_k_initial=300`, `top_k_final=200`, `min_confidence=0.0`) so recall is
measured at the BEAM cutoffs, not pre-filtered.

Point SMOS at it via the `SMOS_CONFIG` env var (or the `--config` flag):

```bash
export SMOS_CONFIG=smos.bench.toml
```

Validate it without running anything:

```bash
smos config show --config smos.bench.toml
```

## Running

### Smoke (1 conversation, 100K, predict-only — no judge needed)

```bash
python -m benchmarks.beam.run \
  --project-name smos-smoke \
  --backend oss \
  --chat-sizes 100K \
  --conversations 0 \
  --predict-only
```

### Full 1M run

```bash
python -m benchmarks.beam.run \
  --project-name smos-1m \
  --backend oss \
  --chat-sizes 1M \
  --answerer-model <reader-model-id> \
  --judge-model <judge-model-id> \
  --provider <openai|openrouter|...>
```

The reader/judge LLM credentials come from the environment the BEAM
`LLMClient` reads (see `benchmarks/common/llm_client.py`).

## Adapter smoke test (no models / no GPU)

`SMOSClient` can be exercised end-to-end against a **mock** `smos` binary,
which verifies the parsing/subprocess wiring without any llama-server or NLI
model. Run:

```bash
python benchmarks/common/test_smos_client.py
```

This stubs `import raw` / `finalize` / `search` and asserts that
`add` → `finalize_pending` → `search` round-trips a fact through the adapter.
The real end-to-end smoke (`add` → `finalize` → `search` returning ≥1 Accepted
fact) requires the llama-server endpoints + NLI model and is run as part of
the BEAM smoke above.
