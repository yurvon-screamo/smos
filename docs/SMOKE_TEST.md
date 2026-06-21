# SMOS Smoke Test

Manual checklist for end-to-end verification of SMOS Rust production against a real `opencode` client.

The test takes ~30 minutes split across five phases. Phase 1 validates the environment; Phases 2-4 exercise the live pipeline (chat → enrichment → extraction → finalize); Phase 5 produces the report artefact.

> **Do not** run any of the live phases against a workspace you cannot afford to lose. The doctor + server mutate `./data/smos.db` and trigger real model inference against your local Ollama.

---

## Prerequisites

Before starting, confirm:

- [ ] Rust 1.96+ installed (`rustup show`)
- [ ] Ollama running locally (`ollama serve`)
- [ ] Required Ollama models pulled:
  ```bash
  ollama pull granite4.1:3b
  ollama pull hf.co/jinaai/jinaai-jina-embeddings-v5-text-small-retrieval-GGUF:latest
  ollama pull qwen3.5:2b
  ```
- [ ] DeBERTa-v3 ONNX model cacheable (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, ~643 MB on first run; lands under `[nli].cache_dir`)
- [ ] `opencode` CLI installed and on PATH
- [ ] (Optional) llama.cpp server running on `:8181` for the reranker
- [ ] Working directory is the workspace root: `D:\uwuwu_agent\smos-rust`

> No Python, no `torch`, no `transformers` — SMOS uses native ort + ONNX Runtime for NLI.

---

## Phase 1 — Environment validation (5 min)

### 1.1 Run doctor

```bash
cargo run --release --bin smos -- doctor
```

Expected: every check **PASS** except the optional reranker (a **WARN** is fine).

- [ ] `smos binary` **PASS**
- [ ] `Ollama connectivity` **PASS**
- [ ] All required models **PASS** (granite4.1:3b, Jina v5, qwen3.5:2b)
- [ ] `Reranker` **PASS** or **WARN**
- [ ] `SurrealDB` + `SurrealDB migrations` + `SurrealDB stats` **PASS**

If any check **FAIL** — fix before continuing. Common fixes:

- Missing model: `ollama pull <name>`
- SurrealDB locked: stop any running `smos serve`, delete `./data/smos.db`, retry

### 1.2 Pre-seed the database

```bash
cargo run --release --bin smos -- import \
  --from-file scripts/smoke_seed.json \
  --memory-key smoke-test
```

Expected (numbers may vary ±2 depending on extraction model temperature):

```
Parsed 3 assistant turns
After offset/limit: 3 turns to process

=== Import complete ===
Session:      ses_smokeseed0001
Memory key:   smoke-test
Processed:    3 turns
Skipped:      0 turns
New facts:    5
```

- [ ] Import succeeds
- [ ] 5 new facts reported (3 assistant turns × 1-2 facts each)

### 1.3 Verify seed with `--stats`

```bash
cargo run --release --bin smos -- doctor --stats
```

Expected stats block:

```
facts: 5 (accepted: 0, pending: 5, rejected: 0)
sessions: 1 (active: 0, ended: 1)
```

- [ ] Pending count ≥ 5
- [ ] 1 session recorded

---

## Phase 2 — Live smoke test (15 min)

### 2.1 Start SMOS server (Terminal 1)

```bash
cargo run --release --bin smos -- serve
```

Expected startup logs:

```
INFO starting SMOS proxy version=0.1.0
INFO surrealdb connected, migrations applied
INFO NLI backend started for session watcher model=MoritzLaurer/...
INFO session watcher started scan_interval_secs=60
INFO SMOS HTTP server listening host=127.0.0.1 port=8888
```

- [ ] Server starts without errors
- [ ] Native NLI backend started for the watcher (or warn that it failed — HTTP still serves)
- [ ] Listening on configured port

**Keep this terminal open.** All SMOS logs will stream here.

### 2.2 Start opencode (Terminal 2)

PowerShell:

```powershell
$env:OPENAI_BASE_URL = "http://localhost:8888/v1"
$env:OPENAI_API_KEY = "smos-smoke-test"
opencode
```

Bash:

```bash
export OPENAI_BASE_URL=http://localhost:8888/v1
export OPENAI_API_KEY=smos-smoke-test
opencode
```

- [ ] opencode launches normally
- [ ] No connection errors in either terminal

In the opencode model selector choose: **`smoke-test:granite4.1:3b`**

The `smoke-test:` prefix is the memory key — SMOS routes the suffix to the upstream model.

### 2.3 Test turns

Send each prompt in opencode. After each response, watch the SMOS logs in Terminal 1.

#### Turn 1 — topic that matches seeded facts

```
What authentication does the project use?
```

Expected SMOS logs:

```
INFO enrichment topic="What authentication does the project use?"
INFO enrichment injected=2 source=smoke-test
INFO upstream forwarded to=granite4.1:3b
INFO response streaming marker_injected=true session=sess_...
INFO extraction spawned async=true
```

- [ ] Response rendered in opencode (quality does not matter for the smoke test)
- [ ] SMOS logs show `injected >= 1` fact
- [ ] Session id consistent across turns

#### Turn 2 — second topic with seeded match

```
What's the tech stack?
```

- [ ] SMOS logs show `injected >= 1` fact (Rust + Leptos seed)

#### Turn 3 — topic with no seeded match

```
What is 3+2?
```

Expected SMOS logs:

```
INFO enrichment topic="What is 3+2?"
INFO enrichment injected=0 reason="no hits"
INFO upstream forwarded enriched=false
```

- [ ] Response received (`5`)
- [ ] SMOS logs confirm `0` facts injected

#### Turn 4 — code generation (extraction path)

```
Write a Rust function that returns the current UTC time as an ISO 8601 string.
```

Expected after the response finishes streaming:

```
INFO extraction parsed_content_chars=NNN
INFO extraction model_returned_facts=K
INFO extraction saved_pending_facts=K
INFO session pending_count=N
```

- [ ] Response received
- [ ] SMOS logs show extraction ran
- [ ] At least 1 new pending fact saved

#### Turn 5 — repeat Turn 1 (dedup verification)

```
What authentication does the project use?
```

Expected SMOS logs:

```
INFO enrichment candidates=2
INFO dedup new=0 reason="already injected this session"
INFO upstream forwarded enriched=false
```

- [ ] Same session id as Turn 1
- [ ] `new=0` in dedup log (session-level dedup works)

### 2.4 Verify extraction state

```bash
# Terminal 3
cargo run --release --bin smos -- doctor --stats
```

Expected:

```
facts: 6+ (accepted: 0, pending: 6+, rejected: 0)
sessions: 1 (active: 1)
```

- [ ] Pending count increased from Phase 1 (extraction pipeline works)

---

## Phase 3 — Finalize (5 min)

### 3.1 Manual finalize trigger

Find the session id in SMOS logs (Terminal 1) — it looks like `sess_<12 hex chars>`.

While the server keeps running, in Terminal 4:

```bash
cargo run --release --bin smos -- finalize sess_<your-session-id>
```

Expected:

```
INFO starting finalize trigger session=sess_... model=MoritzLaurer/...
INFO loading pending facts count=6
INFO finalizing fact=... (per-fact logs)
INFO finalize complete processed=6 finalized=N merged=M conflicts=K
```

- [ ] Finalize succeeds
- [ ] Native NLI classifier classified the candidate pairs

### 3.2 Verify finalize

```bash
cargo run --release --bin smos -- doctor --stats
```

Expected:

```
facts: 6+ (accepted: N, pending: 0-or-low, rejected: M)
```

- [ ] Pending count dropped
- [ ] Accepted count > 0 (some facts promoted)
- [ ] Rejected count possibly > 0 (trivial facts filtered by NLI)

---

## Phase 4 — Graceful shutdown (1 min)

### 4.1 Stop SMOS server

In Terminal 1, press **Ctrl+C**.

Expected logs (in order):

```
INFO Ctrl+C received initiating graceful shutdown
INFO draining in_flight_requests
INFO draining extraction_tasks grace=30s
INFO draining sessions count=1
INFO session watcher stopped
INFO graceful shutdown complete
```

- [ ] No orphan processes (check Task Manager / `ps aux | grep smos`)
- [ ] SurrealDB file is consistent (no corruption markers in logs)
- [ ] Exit code 0

### 4.2 Post-shutdown doctor

```bash
cargo run --release --bin smos -- doctor
```

- [ ] All checks still **PASS**
- [ ] Stats reflect the finalised state

---

## Phase 5 — Report

```bash
cargo run --release --bin smos -- doctor --report smoke_report.md
```

- [ ] Markdown report generated at `./smoke_report.md`
- [ ] Contains: header with timestamp + config path, summary table, stats section, recommendations list

---

## Troubleshooting

### `FAIL Required model: granite4.1:3b`

```
ollama pull granite4.1:3b
```

Re-run `cargo run --release --bin smos -- doctor` to confirm.

### `WARN NLI backend failed to start` / watcher disabled

The native ort + ONNX Runtime backend could not initialise. Likely causes:

- First-run model download failed — check network access to HF Hub and free
  disk space under `[nli].cache_dir` (default `./data/nli_cache`).
  The DeBERTa-v3 ONNX export is ~643 MB.
- Selected GPU EP cannot initialise on the host — re-build with a
  different GPU feature (or no GPU feature) per `README.md` → Native NLI
  backend.
- ort binary download mismatch — delete `target/` and re-build.

HTTP still serves without NLI; only the watcher is disabled. Fix and
restart.

### `FAIL SurrealDB` with `database is locked`

Another process holds the RocksDB lock:

1. Stop any running `smos serve` (Ctrl+C in Terminal 1).
2. Delete `./data/smos.db`.
3. Retry the doctor.

### opencode connection refused

- Verify `OPENAI_BASE_URL=http://localhost:8888/v1` (note the `/v1` suffix).
- Verify `smos serve` is running: `curl http://localhost:8888/health` should return 200.
- Verify the model selector picked the `smoke-test:granite4.1:3b` entry (the `:`-prefixed token is the memory key).

### Enrichment shows `injected=0` for a topic you seeded

- Confirm the seed import succeeded: `cargo run --release --bin smos -- doctor --stats` should list pending facts under the `smoke-test` memory key.
- Confirm the model selector used the `smoke-test:` prefix — without it, the proxy uses the `shared` namespace, which has no facts.
- Confirm `retrieval.min_topic_chars` (default 3) is satisfied by your prompt.

### Extraction shows 0 new facts after a long response

- Check SMOS logs for `kill-switch` or `extraction disabled` messages.
- Verify `enable_response_extraction = true` in `smos.toml`.
- Confirm the response was longer than `MIN_INPUT_CHARS` (15 chars) — short replies are intentionally skipped.

### Finalize reports `processed=0`

- The session id you passed to `smos finalize` does not match any recorded session. Copy the id from SMOS logs (the `session=sess_...` field on every enrichment log line).

---

## Success criteria

Smoke test passes when **every** checkbox in Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5 is ticked.

If any step fails, collect:

1. SMOS server logs (Terminal 1)
2. `cargo run --release --bin smos -- doctor --report` output
3. The exact step that failed + expected vs actual behaviour

Report to the Tech Lead for triage.
