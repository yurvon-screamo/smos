# SMOS Architecture

This document covers the layer-by-layer structure of the SMOS workspace, the request pipeline, the memory lifecycle, and the NLI pipeline internals. It is the reference a contributor should read before changing anything in `smos-application` or `smos-adapters`.

For setup, testing, and code style see [`CONTRIBUTING.md`](../CONTRIBUTING.md). For user-facing usage see [`README.md`](../README.md).

## Contents

- [Overview](#overview)
- [Layered architecture](#layered-architecture)
- [Domain layer](#domain-layer-smos-domain)
- [Application layer](#application-layer-smos-application)
- [Adapters layer](#adapters-layer-smos-adapters)
- [Request pipeline (data flow)](#request-pipeline-data-flow)
- [Memory lifecycle](#memory-lifecycle)
- [NLI pipeline](#nli-pipeline)
- [Confidence scoring](#confidence-scoring)
- [Configuration layering](#configuration-layering)
- [Fail-open contract](#fail-open-contract)
- [Session ownership](#session-ownership)
- [Crate dependency rules](#crate-dependency-rules)

## Overview

SMOS is a 3-crate Cargo workspace in the Hexagonal / DDD style. The dependency direction is enforced one way: a layering violation fails to compile.

```
┌──────────────────────────────────────────────────────────────┐
│                    smos-adapters                              │
│  SurrealStore · native NLI (ort+ONNX) · axum · reqwest ·     │
│  Ollama · llama.cpp · CLI · dreaming agent · service install │
└──────────────────────────────────────────────────────────────┘
                            ▲ implements
┌──────────────────────────────────────────────────────────────┐
│                  smos-application                            │
│  Port traits (async, NO Send bound) · use cases · helpers ·  │
│  types · errors                                              │
└──────────────────────────────────────────────────────────────┘
                            ▲ depends on
┌──────────────────────────────────────────────────────────────┐
│                     smos-domain                              │
│  Entities · value objects · enums · config thresholds ·      │
│  chat payload shapes · error type                            │
│  (no IO, no async runtime, no serde_json)                    │
└──────────────────────────────────────────────────────────────┘
```

| Crate | Depends on | IO | Async runtime |
|---|---|---|---|
| `smos-domain` | (std + serde + thiserror + time + sha1) | none | none |
| `smos-application` | `smos-domain` | none (only port traits) | async fns, no tokio dep |
| `smos-adapters` | `smos-domain`, `smos-application` + tokio, surrealdb, ort, axum, reqwest, clap, rig-core | all | tokio multi-thread |

## Layered architecture

### Domain layer (`smos-domain`)

Pure logic. No IO, no async runtime, no `serde_json` in production code. The crate exposes its own opaque payload types (e.g. `chat::ToolArguments`) so adapters own the JSON boundary. `serde_json` is a dev-only dependency for round-trip tests.

| Module | Contents |
|---|---|
| `entities/fact.rs` | `Fact` aggregate root. Owns `find_merge_candidates`, `mark_conflict`, provenance lists (`source_sessions`, `conflicts_with`), and the per-fact confidence recomputation. |
| `entities/session.rs` | `Session` / `SessionState` — conversation lifecycle and pending-fact bookkeeping. |
| `value_objects/` | `FactId`, `FactContent`, `Confidence`, `Cosine`, `Embedding`, `Heat`, `MemoryKey`, `NliResult` + `NliScores`, `SessionId`, `SourceSessions`, `Timestamp`. |
| `enums/` | `FactStatus` (`Pending` / `Accepted` / `Rejected`), `FactType`, `MergeReason` (`Merged` / `Drift` / `NeutralSkipped`), `NliLabel` (`Entailment` / `Neutral` / `Contradiction`). |
| `chat.rs` | Tool-call argument shapes. |
| `config.rs` | Domain thresholds (`NliConfig`, `ConfidenceConfig`, `MergeConfig`). The adapter-only `[nli_backend]` data (model id, cache dir) never crosses into this layer. |
| `error.rs` | Domain error type. |

Invariants live next to the code that enforces them. Example: `FactId = SHA1(content)` is the dedup identity and keeps exact duplicates stable across re-extraction runs — the comment sits in `value_objects/fact_id.rs`.

### Application layer (`smos-application`)

Port traits and use cases. Runtime-agnostic: ports are `async fn` **without** a `Send` bound. The bound is added at the adapter call site (the only place that needs a multi-thread runtime), which keeps the application layer usable from a single-thread executor in principle.

| Module | Contents |
|---|---|
| `ports/` | `Clock`, `Delay`, `EmbeddingProvider`, `FactRepository`, `IdGenerator`, `LlmExtractor`, `LlmUpstream`, `NliClassifier`, `RerankProvider`, `SessionRepository`. |
| `use_cases/` | `handle_chat_completion`, `enrich_request`, `extract_facts_from_response`, `finalize_session`, `import_opencode_session`. |
| `helpers/` | `memory_block` (renders the `<smos-memory>` injection), `model_parser` (splits `memory_key:model`), `noise_filter`, `openai_content`, `request_enricher`, `retrieval_planner`, `session_marker`, `topic_extractor`. |
| `types/` | `chat_request`, `chat_response`, `enrichment_messages`, `merge_result`, `rerank_result`, `search_hit`. |
| `errors/` | `UseCaseError`, `ProviderError`. |

A use case is the smallest unit that has business meaning. Each one takes borrowed references to its dependencies and returns a typed result — no globals, no hidden state.

### Adapters layer (`smos-adapters`)

Every concrete IO implementation in the system. This is the only crate that may import `tokio`, `serde_json`, `surrealdb`, `axum`, `reqwest`, `ort`, `clap`, `rig-core`.

| Module | Implements | Notes |
|---|---|---|
| `storage/surreal_store.rs` | `FactRepository`, `SessionRepository` | Embedded SurrealDB (RocksDB + HNSW vector index). Single-process lock. |
| `storage/surreal_schema.rs` | — | Migrations applied on first connect. |
| `storage/system_clock.rs` | `Clock` | `time::OffsetDateTime::now_utc`. |
| `storage/system_id_generator.rs` | `IdGenerator` | ULID-style session / fact ids. |
| `nli/native_nli.rs` | `NliClassifier` | DeBERTa-v3 via `ort` + ONNX Runtime, in-process. |
| `nli/runtime.rs` | — | Session watcher that drains pending facts via `FinalizeSession`. |
| `nli/model_cache.rs` | — | HF Hub download + on-disk model cache. |
| `nli/device.rs` | — | GPU EP detection + CPU fallback. |
| `http/axum_server.rs` | — | The proxy HTTP server. |
| `http/routes/` | — | `/v1/chat/completions`, `/health`, etc. |
| `http/stream_transform.rs` | — | SSE passthrough + session marker injection. |
| `http/error_mapper.rs` | — | Use-case errors → HTTP responses. |
| `upstream/reqwest_upstream.rs` | `LlmUpstream` | Streaming SSE forward; `single` / `round_robin` / `failover` strategies. |
| `upstream/sse_parser.rs` | — | Incremental SSE parser. |
| `upstream/streaming_buffer.rs` | — | Bounded buffer for streaming responses. |
| `providers/ollama/` | `LlmExtractor`, `EmbeddingProvider` | Qwen3.5:2b extraction, Jina v5 embeddings. |
| `providers/llama_cpp/` | `RerankProvider` | Qwen3-Reranker (optional). |
| `providers/noop/` | — | In-process mocks for tests. |
| `opencode/` | — | Session discovery for `smos import`. |
| `doctor/` | — | Environment validation for `smos doctor`. |
| `dreaming/` | — | Optional audit agent: `rig-core` tool-calling, cron-scheduled, bounded mutations, markdown report. |
| `runtime/` | — | Service install (systemd / Windows service / launchd) + supervisor glue. |
| `cli/` | — | `clap` subcommand wiring. |
| `config.rs` | — | Layered TOML config (built-in defaults + file override + `--config <path>`). |
| `bin/smos.rs` | — | The unified binary entry point. |

## Request pipeline (data flow)

```
Client ──POST /v1/chat/completions──▶  axum route
                                         │
                                         ▼
                          handle_chat_completion  (use case)
                                         │
                  ┌──────────────────────┴───────────────────────┐
                  │                                                │
                  ▼                                                ▼
          1. PARSE                                          (deferred)
   model_parser splits                                    (extraction,
   "memory_key:upstream_model"                            finalize run
   → MemoryKey + upstream model name                       off-path)
   detect or reuse session id
                  │
                  ▼
          2. ENRICH  (enrich_request + request_enricher + retrieval_planner)
   topic_extractor  ─▶ EmbeddingProvider.embed(topic)
                   ─▶ FactRepository.search(memory_key, embedding, top_k_initial)
                                  │  HNSW vector search
                                  ▼
                   RerankProvider.rerank(topic, hits)  (optional, llama.cpp)
                                  │
                                  ▼
                   heat_filter     (decay * recency)
                   atomic dedup    (FactId exact + cosine ≥ 0.95)
                                  │
                                  ▼
                   memory_block.render(accepted facts)
                                  │
                                  ▼
                   <smos-memory> injected into request.messages
                                  │
                                  ▼
          3. FORWARD  (reqwest_upstream)
                   upstream strategy picks provider
                   (single / round_robin / failover)
                   POST → upstream LLM (SSE stream)
                                  │
                                  ▼
                   stream_transform  (SSE passthrough + session marker)
                                  │
                                  ▼
                            Client receives stream
                                  │
                                  ▼
                   (response complete)
                                  │
                                  ▼
          4. EXTRACT  (extract_facts_from_response, OFF the request path)
                   LlmExtractor.extract(response)  (Qwen3.5:2b, temp=0, seed=42)
                   noise_filter
                   semantic dedup  (FactId SHA1 + cosine ≥ 0.95)
                   FactRepository.save_pending(facts, session_id, memory_key)
                                  │
                                  ▼
                   facts sit at status = Pending, confidence = base (0.5)
                                  │
                                  ▼
          5. FINALIZE  (finalize_session, OFF the request path)
                   triggered by session timeout (30 min default)
                   or `smos finalize <session_id>`
                   or watcher scan interval
                                  │
                                  ▼
                   DeBERTa-v3 NLI walk over pending vs accepted
                   (see NLI pipeline below)
                                  │
                                  ▼
                   merge / conflict / promote / reject
                                  │
                                  ▼
          6. AUDIT  (optional dreaming agent, OFF the request path)
                   cron-scheduled rig-core tool-calling
                   bounded deletions / merges / conflict flags
                   markdown report under [audit].report_dir
```

Stages 4, 5, 6 run **off the request path**. The client receives the response as soon as the upstream stream completes — extraction, finalize, and audit never block a request. Enrichment (stage 2) is fail-open: a retrieval / embedding / rerank failure degrades to an unenriched forward rather than erroring out.

## Memory lifecycle

```
                    ┌──────────────┐
   extraction ────▶ │   pending    │   FactId = SHA1(content)
   (stage 4)        └──────┬───────┘   confidence = 0.5 (base)
                           │           status = Pending
                  finalize trigger
                  (session timeout 30 min,
                   watcher scan_interval 60 s,
                   or `smos finalize <session_id>`)
                           │
                           ▼
              ┌─────────────────────────┐
              │   DeBERTa-v3 NLI walk    │
              │   (drift-priority,       │   see "NLI pipeline" below
              │    C3 guard, exact-match)│
              └──┬──────────┬───────────┘
                 │          │
        entailment      contradiction
        (merge)         (flag conflicts_with bidirectionally)
                 │          │
                 ▼          ▼
        ┌────────────┐  ┌──────────────────────────┐
        │  accepted  │  │ both facts kept; surfaced│
        │            │  │ to the LLM on the next   │
        │ confidence │  │ enrichment as a conflict │
        │ ≥ 0.7      │  │ pair (the LLM decides)   │
        └────────────┘  └──────────────────────────┘
                 ▲
                 │ neutral + confidence ≥ accept_threshold
                 │ (standalone promotion)
                 │
            ┌────────────┐
            │  rejected  │  confidence < pending_threshold (0.4)
            └────────────┘
```

Per-fact confidence is recomputed on finalize (see [Confidence scoring](#confidence-scoring)).

The lifecycle is **idempotent**: re-running `smos finalize <session_id>` on an already-drained session is a no-op. Snapshotted `owned_ids` (taken before the first await) ensure concurrent extraction appends during finalize are preserved for the next cycle.

## NLI pipeline

The finalize walk is the most subtle part of SMOS. It is implemented in [`smos-application/src/use_cases/finalize_session.rs`](../smos-application/src/use_cases/finalize_session.rs) (`FinalizeSession::resolve_one`).

### Problem

A naive finalize would pick the top accepted candidate by cosine similarity, run NLI once, and commit the verdict. That masks drift: a contradiction against a *less-similar* candidate is silently swallowed by a neutral/entailment hit on the top candidate. A pending `TTL=60` may contradict an accepted `TTL=10` that ranks second by embedding similarity, while the top match is an unrelated neutral.

### Resolution: drift-priority walk

For each pending fact, `resolve_one`:

1. Gathers candidates via `Fact::find_merge_candidates(pool, merge_cfg)` — cosine ≥ `merge.cosine_threshold` (default 0.85).
2. Iterates **every** candidate, not just the top one.
3. Tracks a `merge_pick: Option<(Fact, NliResult)>` — the first entailment candidate becomes the merge pick, but the scan continues so a later less-similar candidate can still surface a contradiction.
4. Tracks `last_observed_nli: Option<NliResult>` — feeds the `no_contradiction_bonus` on the standalone promotion path.
5. First contradiction wins immediately (flag both sides + return). No earlier entailment candidate is committed before the contradiction is observed, because drift is a stronger signal than merge.
6. If the scan completes contradiction-free with an entailment candidate, the merge is committed.
7. Otherwise the pending fact is promoted standalone, carrying `last_observed_nli` for the bonus.

The comparison pool **grows** as standalone facts are promoted — a later pending fact can merge with one that was itself pending a moment ago. Merges and conflicts consume the pending twin without growing the pool.

### C3 guard

Already-flagged conflict pairs skip the expensive NLI call entirely. The conflict was resolved by an earlier finalize cycle; calling the model again would just re-confirm it.

```
if pending.conflicts_with().contains(existing.id())
    || existing.conflicts_with().contains(pending.id())
{
    nli_observed = true;
    continue;  // skip the NLI call, pair still counts as "observed"
}
```

Without the C3 guard, a pending twin of an already-flagged pair would be stuck in `pending` forever: every cycle would skip the same pair and report "NLI never observed", blocking the `no_contradiction_bonus` and the standalone promotion.

### Exact-match short-circuit

Identical text is entailment by definition. The model call is skipped and a canonical `NliResult::exact_match_result()` is returned:

```rust
NliResult {
    label: NliLabel::Entailment,
    scores: NliScores { entailment: 1.0, neutral: 0.0, contradiction: 0.0 },
    available: true,
}
```

This avoids DeBERTa-v3's known quirk of returning `neutral` on byte-identical pairs. Normalisation is whitespace- and case-insensitive (`FactContent::text_equals_normalized`).

### Graceful degradation

The NLI classifier may be unreachable (model download failed, GPU EP refused to initialise, OOM). The backend returns a placeholder `NliResult { available: false, ... }`. Downstream code treats that as "not checked", not "no contradiction detected":

- An `available = false` reply does **not** bump `nli_observed` — otherwise a permanently broken backend would silently promote facts without drift detection.
- The pending fact stays pending for the next cycle. No data is lost.

This is the fail-open contract: HTTP keeps serving, the watcher logs a warning, and pending facts queue up until NLI recovers.

### Layered dedup

Deduplication runs in three layers, each cheaper than the next:

| Layer | Where | Mechanism | Cost |
|---|---|---|---|
| 0 | Extraction (stage 4) | `FactId = SHA1(content)` exact match | hash lookup |
| 1 | Extraction (stage 4) | Cosine ≥ `extraction.dedup_cosine_threshold` (default 0.95) | one embedding + similarity |
| 2 | Finalize (stage 5) | Drift-priority NLI walk | one DeBERTa forward per candidate pair |

Layers 0 and 1 catch the common case (re-extracted duplicates, minor rephrasings) without invoking the model. Layer 2 is reserved for the genuinely ambiguous pairs that reach finalize.

## Confidence scoring

Per fact, recomputed on finalize:

```
confidence = base
           + multi_source_bonus      if 2+ unique sessions extracted this fact
           + no_contradiction_bonus  if NLI ran and found no contradiction
```

Defaults (from `smos.toml` → `[confidence]`):

| Field | Default | Meaning |
|---|---|---|
| `base` | `0.5` | Single-source, NLI not yet run. |
| `multi_source_bonus` | `0.2` | Cross-session corroboration. |
| `no_contradiction_bonus` | `0.1` | NLI ran and the fact is drift-free. |
| `accept_threshold` | `0.7` | At or above → `Accepted`. |
| `pending_threshold` | `0.4` | Below → `Rejected`. |

Consequence: a single-source fact starts at 0.5 and stays `Pending`. It promotes only after either a second session extracts it (0.5 + 0.2 = 0.7) or the NLI walk grants the no-contradiction bonus (0.5 + 0.1 = 0.6 — still pending, needs the multi-source boost). This is the cross-session confirmation guarantee: a single observation is never enough to surface a fact in enrichment.

## Configuration layering

`smos.toml` is **layered**: built-in defaults live in code (`smos-adapters/src/config.rs`); sections present in the file override; omitted sections fall back. Every section uses `#[serde(deny_unknown_fields)]`, so a typo or a misplaced field (e.g. putting `model` under `[nli]` instead of `[nli_backend]`) is a loud startup error, not a silent drop.

The layering preserves the domain invariant: domain types carry no IO-boundary data. The `[nli]` section holds domain-relevant thresholds (`contradiction_threshold`, `entailment_threshold`) consumed by `NliResult::is_contradiction` / `is_entailment`. The `[nli_backend]` section holds interpreter-level data (model id, cache dir) that the domain layer never reads.

Secrets: the `[audit].cloud_api_key` field expands `${OPENROUTER_API_KEY}` via `std::env::var` at runtime, so secrets stay out of TOML. The same `${VAR}` expansion works for any `api_key` field.

## Fail-open contract

The system degrades gracefully at every IO boundary:

| Failure | Behaviour |
|---|---|
| Embedding provider unreachable | Enrichment skips, request forwards unenriched. |
| Reranker unreachable | Rerank skipped, top-K by cosine used directly. |
| Vector search returns 0 hits | `<smos-memory>` block omitted, request forwards. |
| Extraction LLM unreachable | No facts saved, response unaffected. |
| NLI backend unreachable | Pending facts stay pending, watcher logs warning. |
| SurrealDB locked (single-process) | Startup fails with explicit error (no silent retry). |
| Upstream LLM error | `failover` strategy tries the next provider; `single` returns the error to the client. |

The only failure that propagates as an HTTP error is an upstream LLM failure under the `single` strategy. Everything else degrades.

## Session ownership

Pending ownership is derived from `Fact.source_sessions`, **not** from `SessionState.pending_facts`. Every fact whose provenance list references `session_id` is in scope for finalize.

Why: the HTTP extraction path never persists a `SessionState` row — it only mutates `fact.source_sessions` at extraction time. Reading `SessionState.pending_facts()` left real pending facts invisible to finalize (the operator-facing "24 pending facts but finalize says nothing to do" bug). `source_sessions` is the only durable provenance signal that survives the request path.

The `memory_key` is supplied by the caller (CLI `--memory-key`, watcher reading `SessionState.memory_key()`) because `source_sessions` does not pin a namespace. The CLI additionally exposes a discovery fallback (`FactRepository::list_memory_keys_for_session`) that iterates every key when the operator does not name one.

`owned_ids` is snapshotted **before the first await** so concurrent extraction appends (which race the drain) survive: only the snapshotted ids are removed from `pending_facts` after finalize. Fresh pending ids appended by another flow during finalize are preserved for the next cycle.

## Crate dependency rules

These are enforced at the `Cargo.toml` level — a violation fails to compile.

- `smos-domain` may depend only on `serde`, `thiserror`, `time`, `sha1`. No `tokio`, no `serde_json`, no `surrealdb`, no `axum`, no `reqwest`, no `ort`.
- `smos-application` may depend on `smos-domain` plus `serde`, `serde_json`, `thiserror`, `time`, `futures`, `bytes`, `tracing`, `regex`. No `tokio`, no `surrealdb`, no HTTP.
- `smos-adapters` is the only crate that performs IO. It depends on both inner crates plus the full IO stack.

Port traits in `smos-application` are `async fn` **without** a `Send` bound. The bound is added at the adapter call site (via `Box<dyn Trait + Send>` or generic bounds on the runtime entry point), which keeps the application layer usable from a single-thread executor in principle and avoids leaking runtime concerns into the domain.

If you find yourself wanting to add `tokio` or `serde_json` to `smos-domain` or `smos-application`, stop — the answer is a port trait in `smos-application` and an implementation in `smos-adapters`.
