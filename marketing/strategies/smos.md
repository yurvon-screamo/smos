# SMOS Marketing Strategy

> **Status:** DRAFT — awaiting HUMAN GATE approval before any publication.
> **Generated:** 2026-06-25
> **Inputs:** 2 background research reports (`docs/research/smos-competitive-research-2026-06-25.md`, `docs/research/seo-keyword-clusters-2026-06-25.md`) + primary sources (`README.md`, `docs/ARCHITECTURE.md`, `smos.toml`, `CONTRIBUTING.md`).
> **Audience:** Solo developers and power users of AI coding assistants (Cursor, Claude Code, opencode, Cline, Continue.dev, Aider, Windsurf).
> **Voice:** Builder-Architect 60% / Pragmatic Operator 30% / Precision Educator 10%. Calm conviction, never enthusiasm.

---

## 1. Executive Summary

SMOS is an OpenAI-compatible semantic memory proxy written in Rust. A client points its base URL at SMOS, uses a person name as the model, and gets persistent long-term memory — no agent code changes, no MCP server, no framework adoption.

Of the 14 memory-for-AI projects analyzed in the competitive research, **only 3 use proxy-based architecture**. Two of them (Mnemo — 1 star, inactive; agent-cli memory proxy — minimal npm package) are not production-grade. **SMOS is the only production-grade proxy-based memory system on the market.**

Five positioning gaps are unique or under-served, and SMOS can own all five: memory-as-API (not a tool), multi-agent isolation as a primary feature, NLI-based contradiction detection, zero external dependencies, and the OS metaphor as an infrastructure (not runtime) interpretation.

---

## 2. Primary Audience & Funnel

| Layer | Audience | Funnel stage | What they search |
|---|---|---|---|
| **Primary** | Power user of AI coding assistants who is tired of re-explaining project context every new chat | Mid-funnel (consideration) | "cursor persistent context", "claude code memory", "claude code memory without MCP", "persistent memory across coding sessions" |
| **Secondary** | Dev building multi-agent system who needs memory infrastructure without framework lock-in | Mid-funnel | "mem0 alternative self-hosted rust", "self hosted memory proxy", "rust llm memory" |
| **Tertiary** | OSS Rust dev evaluating architecture of memory systems | Top-funnel (awareness) | "semantic memory proxy rust", "memory proxy not tool" |

**Funnel reality from SEO research:** generic "AI memory" cluster is saturated (mem0/letta/wikipedia dominate). The unserved keywords are architecture-specific and Rust-specific. **SMOS should target the unserved long-tail, not compete on broad terms.**

---

## 3. Differentiation Framework — 5 Gaps SMOS Owns

| # | Gap | Status | Evidence |
|---|---|---|---|
| **G1** | **Memory IS the API, not a tool the agent calls** | UNIQUE — nobody else positions this way | All serious competitors (mem0, Letta, Zep, Cognee) require explicit save/retrieve tool calls. Research Part 5.3, Gap 1. |
| **G2** | **Multi-agent isolation as a primary feature** | UNDER-SERVED — r/AIMemory thread explicitly identified this gap (Mar 2026) | mem0 has user_id/agent_id scoping but does not market it. Letta requires framework adoption. SMOS: `memory_key` namespace by default. README:26-31, smos.toml:50-71. |
| **G3** | **NLI-based contradiction detection (DeBERTa-v3, not cosine)** | UNIQUE — no competitor uses NLI for this | "The Price of Meaning" (arXiv:2603.27116, MIT/Sentra, Mar 2026) mathematically proves vector-only retrieval degrades through interference. DeBERTa NLI = the "external verification" the paper calls necessary. ARCHITECTURE.md:242-313. |
| **G4** | **Zero external dependencies (embedded SurrealDB RocksDB)** | UNIQUE — every serious competitor requires external DB | mem0: Qdrant/Postgres. Letta: Postgres+Redis. Zep: Neo4j mandatory. Cognee: Neo4j+Postgres+vector. SMOS: one binary + a directory. Cargo.toml:26. |
| **G5** | **OS metaphor as infrastructure (not runtime)** | SHARED but ownable | Letta owns "LLMs as Operating Systems" runtime interpretation. MemoryOS paper (EMNLP 2025 Oral) owns hierarchical storage. SMOS owns the **infrastructure interpretation**: memory as a system service any agent connects to via the standard OpenAI API. README:20-24. |

---

## 4. Competitive Landscape — Honest Comparison

All facts from `docs/research/smos-competitive-research-2026-06-25.md`. Star counts are snapshots as of 2026-06-25 — recheck before publication.

| Project | Stars (snapshot) | Architecture | Requires to run | Contradiction handling | Self-hostable | License |
|---|---|---|---|---|---|---|
| **mem0** | 59,449 | Tool-based (add/search API) | Qdrant or Postgres+pgvector + LLM key | Picks winner (latest fact wins) | Yes (Docker) | Apache-2.0 |
| **Letta** (MemGPT) | 23,521 | Framework-bundled (full runtime) | Postgres + Redis + LLM key | Agent self-edits | Yes (Docker) | Apache-2.0 |
| **Zep / Graphiti** | 27,922 (Graphiti) | Tool-based / SaaS | **Neo4j mandatory** + LLM key | Temporal invalidation (preserves history) | Partial (full platform is SaaS) | Apache-2.0 |
| **Cognee** | 22,279 | Tool-based (pipeline API) | Neo4j + Postgres + vector DB + LLM key | None explicit (relies on graph structure) | Yes (complex) | Apache-2.0 |
| **Hindsight** | 17,556 | Tool-based (4-strategy retrieval) | LLM key + optional Postgres | Cross-encoder reranking resolution | Yes | MIT |
| **LangMem** | 35,737 (LangGraph) | Framework-bundled | LangChain ecosystem | None native (delegates to store) | Yes | MIT |
| **Mnemo** | 1 | Proxy-based (Rust) | Unknown (planned) | Unknown | Planned | MIT |
| **SMOS** | new | **Proxy-based (Rust)** | **Single binary + a directory** | **NLI contradiction preservation (DeBERTa-v3)** | **Yes, fully** | MIT |

**Read this table honestly, not as a sales pitch.** mem0 has 59K stars and a $24M raise — that is real traction we do not have. The point of the table is architectural contrast, not "we are better". Where SMOS wins (zero dependencies, proxy architecture, NLI detection), the contrast is verifiable. Where SMOS is behind (community size, integration ecosystem, LOCOMO benchmarks), we say so in §5.4.

---

## 5. Marketing Bullets — FINAL (3-layer validated)

Each bullet carries: **product evidence** (README/ARCHITECTURE/smos.toml line citation) + **market signal** (what competitors don't do, from research) + **search vocabulary** (exact phrasing devs use, from SEO research).

### 5.1 Headline bullets (top-level positioning — README/HN/landing)

**B1. Memory is part of the API, not a tool the agent calls.**
SMOS is a transparent OpenAI-compatible proxy. Point your client at it, use a person name as the model, get persistent memory. No agent code changes, no MCP server, no SDK swap.
- *Product:* README:291-315, ARCHITECTURE.md:118-198 (off-request-path pipeline).
- *Market:* research confirms nobody else positions this way (Part 5.3 Gap 1).
- *Search:* "memory is part of the API, not a tool the agent calls" — zero SERP results, SMOS owns this phrasing.

**B2. Real memory of what happened, not what the agent decided to write down.**
Extraction runs off the request path, AFTER the client receives the response. Every assistant turn is mined for facts — not just the tool-call arguments the agent chose to persist.
- *Product:* ARCHITECTURE.md:169-198 (stage 4 EXTRACT runs after response).
- *Market:* every tool-based competitor (mem0, Zep, Cognee, Hindsight) depends on agent-initiated saves.
- *Search:* "the agent treated every session like a blank slate" (HN Ask HN 2026-02-04); "mem0 stores memories but doesn't learn patterns" (HN 2026-02-04).

**B3. Single binary. No Postgres. No Neo4j. No Docker. No external database process.**
Embedded SurrealDB (RocksDB + HNSW vector index). Deploy = `smos serve` and a directory.
- *Product:* Cargo.toml:26 (kv-rocksdb), smos.toml:11-16.
- *Market:* unique — mem0 needs Qdrant/Postgres, Zep mandates Neo4j, Cognee needs Neo4j+Postgres+vector, Letta needs Postgres+Redis.
- *Search:* "self hosted memory proxy" — zero authority pages, SMOS can own.

**B4. Bob for Rust, Alice for ML, Charlie for DevOps — each with isolated memory.**
Multi-persona isolation by default. The person name is the namespace boundary. Alice never mixes your Rust types with your Python pipelines.
- *Product:* README:26-31, smos.toml:50-71.
- *Market:* r/AIMemory thread (Mar 2026) explicitly identified multi-agent isolation as under-served. mem0 has scoping but does not market it. Letta requires framework adoption.
- *Search:* "how are you handling persistent memory across multiple AI agents?" (Reddit r/vibecoding).

**B5. Swap GPT-4o for a local model — Bob stays Bob.**
Identity and memory live at the OS layer, not in the chat log. Switch Cursor → Claude → opencode without re-explaining your architecture, conventions, or past decisions.
- *Product:* README:13-31.
- *Search:* "re-explain your architecture, conventions, past decisions" — exact phrasing from HN Show HN: Engram (2026-03-01); "every session starts from zero" (same).

### 5.2 Technical depth bullets (HN first comment / dev.to / architecture page)

**B6. NLI contradiction detection (DeBERTa-v3), not cosine similarity.**
Cosine similarity averages 0.82 between contradicting pairs in our fixture corpus. DeBERTa-v3 NLI owns every merge and conflict verdict. Pure vector retrieval is mathematically proven to degrade through semantic interference — "The Price of Meaning" (arXiv:2603.27116, MIT/Sentra, Mar 2026) calls external verification necessary. The DeBERTa sidecar is that verification.
- *Product:* README:346-360 (Inspiration cites the paper), ARCHITECTURE.md:242-313 (drift-priority walk + NLI pipeline), smos.toml:138-148.
- *Market:* research confirms no competitor uses NLI for contradiction detection.

**B7. Drift-priority walk catches contradictions against non-top candidates.**
A naive finalize picks the top cosine candidate and runs NLI once. SMOS scans every candidate. The first contradiction wins immediately. The merge pick is held but not committed until the scan completes contradiction-free.
- *Product:* ARCHITECTURE.md:242-313.

**B8. Fail-open contract. Memory failure never blocks the request.**
Seven failure modes — embedding, reranker, vector search, extraction, NLI, SurrealDB lock, upstream LLM — all degrade gracefully. HTTP keeps serving. The only HTTP error is upstream LLM failure under the `single` strategy.
- *Product:* ARCHITECTURE.md:344-358.

**B9. Cross-session confidence confirmation. A single observation is never enough.**
A single-source fact starts at confidence 0.5 and stays Pending. It promotes only after a second unique session extracts it (multi_source_bonus +0.2 = 0.7) or after the NLI walk grants the no-contradiction bonus (0.5 + 0.1 = 0.6, still Pending — needs cross-session corroboration).
- *Product:* ARCHITECTURE.md:314-334.

**B10. Self-hosted sovereignty. All inference local via llama.cpp.**
Extraction, embedding, and reranker run through local llama-server processes. Native DeBERTa NLI on device (CUDA on Linux/Windows NVIDIA, DirectML on Windows, Metal on macOS, CPU fallback) — runtime detection, no rebuild.
- *Product:* smos.toml:201-262, README:43-57.

**B11. Rust, hexagonal DDD, 3-crate workspace with compile-enforced layering.**
`smos-domain` (pure, no IO) ← `smos-application` (ports + use cases, runtime-agnostic) ← `smos` (the only crate that performs IO). A layering violation fails to compile.
- *Product:* ARCHITECTURE.md:24-52.

**B12. Git-backed memory sync. Version, backup, multi-machine re-hydration.**
Dual-write every extracted fact to a git repo as markdown. Re-hydrate on another machine with `smos import-git <url>`.
- *Product:* smos.toml:224-239.

### 5.3 Comparison bullets (differentiation tables / Reddit / "vs X" queries)

Each comparison is factual, sourced from competitor research. **No claims of superiority without evidence.** Where SMOS is weaker, we say so.

**B13. vs mem0 (59K stars):** mem0 requires Qdrant or Postgres+pgvector plus an LLM API key. Graph features are gated behind the $249/mo Pro tier. SMOS ships the full feature set in one binary under MIT, with no paid tier. mem0 has 100K+ community and a $24M Series A — SMOS does not compete on community size.

**B14. vs Letta (23K stars):** Letta requires adopting its full agent runtime (framework lock-in). Memory ops go through LLM tool calls, which adds inference overhead per memory operation. SMOS is framework-agnostic — point any OpenAI-compatible client at it via a base URL change. Letta's runtime integration is deeper; SMOS's is wider.

**B15. vs Zep / Graphiti (27K stars):** Zep requires Neo4j (mandatory). The full Zep platform is now primarily SaaS with credit-based pricing. SMOS has no Neo4j, no Postgres, no external database. Zep's bi-temporal knowledge graph is more sophisticated for temporal reasoning; SMOS does not compete on temporal modeling.

**B16. vs Cognee (22K stars):** Cognee requires Neo4j + Postgres + a vector DB + an LLM key. SMOS requires a single binary. Cognee has 30+ ingestion connectors and multi-modal support; SMOS is text-only today.

**B17. vs all tool-based approaches:** Every tool-based memory depends on the agent deciding when to save and what to retrieve. The agent can forget, misjudge, or skip the call. SMOS extracts from every response transparently — extraction is structurally impossible to skip.

### 5.4 Honest limitations (credibility section — mandatory)

**B18. 643 MB DeBERTa-v3 ONNX download on first start.** Subsequent starts are instant. The model is cached under `~/.smos/models/`.

**B19. Requires llama-server on PATH for full local inference.** SMOS auto-launches the three llama-server processes (extraction, embedding, reranker) when `auto_launch = true`. Disable it to use cloud providers.

**B20. Extraction model is English-optimized.** Nemotron-3-Nano-4B is multilingual but accuracy is highest on English. The DeBERTa NLI model is English-only.

**B21. Single-process SurrealDB lock.** One SMOS instance per database path. No built-in horizontal scaling. Multi-machine via git-sync.

**B22. Not benchmarked on LOCOMO.** We are not joining the LOCOMO benchmark wars — credibility in this category is poisoned (the public mem0 vs Zep benchmark dispute, 2025). The NLI contradiction detection is the architectural choice; a benchmark number is not the argument.

**B23. New project, small community.** SMOS is younger than mem0, Letta, and Zep by years. The architecture is the differentiator, not the install base.

---

## 6. Anti-bullets (DO NOT SAY)

These are saturated, dangerous, or pull the wrong audience. Source: research Part 5.2 (saturated angles) and SEO Deliverable 5.5 (anti-patterns).

| DO NOT SAY | WHY |
|---|---|
| "Stop re-explaining yourself" / "LLMs have amnesia" | Over-saturated. Every memory product uses this. Cognee's Show HN with this framing got 9 points. |
| "Memory layer for AI agents" | mem0 owns this phrase. Latecomers look like clones. |
| "SOTA on LOCOMO" / "best on benchmark X" | Benchmark wars destroyed category credibility. Public debunking risk (mem0 vs Zep). |
| "revolutionary", "game-changing", "seamless", "next-gen" | AI-slop markers. Auto-downvote on HN. |
| "Open-source memory" as a differentiator | Table stakes. Every competitor has this. |
| "Built for production" without evidence | Everyone says this. |
| "AI chatbot memory", "conversational AI memory", "AI companion memory" | Pulls wrong audience (chatbot devs, consumer apps). |
| "Memory for ChatGPT" | Pulls non-technical users of the web interface. |

---

## 7. Vocabulary Inventory (verified phrasings to reuse)

These are verbatim phrasings developers use in HN/Reddit/SO threads. Using them verbatim maximizes discoverability and resonance.

### Pain-point phrasings (use in problem framing)

- "every Claude Code or Cursor session starts from zero" — HN Show HN: Engram (2026-03-01)
- "you re-explain your architecture, conventions, past decisions" — same
- "CLAUDE.md helps but it's manual and doesn't scale" — same
- "after context compression, Claude consistently ignores CLAUDE.md" — HN Ask HN (2025-11-10)
- "session memory loss: asks the same questions like a new intern daily" — same
- "the agent treated every session like a blank slate" — HN Ask HN (2026-02-04)
- "memory degradation (entropy) is the silent killer of complex coding projects" — HN (2025-11-27)
- "I already told you this last week" / "you solved this same problem before" — HN (2025-05-29)

### Solution phrasings (use in feature descriptions)

- "persistent memory across sessions"
- "transparent memory layer via OpenAI-compatible proxy" — GitHub HKUDS/nanobot discussion #2728 (2026-04-01)
- "no code changes" / "without code changes"
- "drop-in replacement for OpenAI API"
- "self-hosted memory for AI agents"
- "memory without MCP" / "no MCP server needed"
- "any OpenAI-compatible client"

### Positioning phrasings (SMOS owns these — use verbatim)

- "memory is part of the API, not a tool the agent calls" — zero SERP results, SMOS unique
- "OpenAI-compatible semantic memory proxy in Rust" — recommended Show HN title

---

## 8. Channel & Format Recommendations

### Show HN

**Title (≤80 chars, no marketing words):**
```
Show HN: SMOS — OpenAI-compatible semantic memory proxy in Rust
```

**First comment structure** (300-500 words, per Template 1 in marketer system prompt):
- The problem (1-2 sentences using pain vocabulary from §7)
- What I built (1 sentence, no adjectives)
- Architecture (3-5 lines: proxy pipeline, embedded SurrealDB, NLI sidecar, off-request-path extraction)
- Honest limitations (B18-B23)
- Link to GitHub, not a landing page

**Best posting window:** Tue/Wed/Thu 7-9 AM EST. Respond to every comment in the first 2 hours.

### Reddit

**Subreddfits:** r/LocalLLaMA (primary), r/rust (secondary), r/ClaudeCode (targeted).

**Title pattern (personal-experience framing, not product launch):**
> I'm tired of re-explaining my project to Claude Code in every new session. I built SMOS — an OpenAI-compatible memory proxy in Rust. Memory is part of the API, not a tool the agent calls.

**10:1 rule:** contribute 10 non-self-promo comments per 1 self-promo post in each subreddit.

### GitHub

**Recommended 20 topics** (from SEO Deliverable 4):
```
ai-memory, agent-memory, llm-memory, semantic-memory, openai-compatible,
self-hosted, memory, proxy, rust, coding-agent, local-first,
claude-code, cursor, mem0-alternative, long-term-memory,
openai-api, surrealdb, ai-agents, persistent-context, ai-proxy
```

### dev.to

**Primary tag combination** (max 4, lowercase): `ai`, `llm`, `rust`, `showdev`

**Article angle:** "Why I built SMOS: memory is part of the API, not a tool the agent calls" — longform architecture deep-dive.

---

## 9. Saturated Angles to AVOID (summary)

See §6 Anti-bullets. Reiterated here for visibility:

- Generic "AI memory" — Wikipedia and mem0 dominate SERPs
- "mem0" as a primary keyword — 59K stars, $24M raise, impossible to outrank
- "langchain memory" — LangChain docs own this
- "RAG" — different concept, wrong SERP
- "chatbot memory" — wrong audience

---

## 10. Pre-launch checklist (next steps)

Before any publication:

1. **Factcheck gate** — `.factcheck.json` for this strategy doc must be `gate: READY`. (Created alongside this doc.)
2. **Star count refresh** — competitor star counts in §4 are 2026-06-25 snapshots. Recheck on the day of publication.
3. **README alignment** — verify the README reflects the proxy/memory-as-API positioning consistently (current README is aligned).
4. **GitHub topics** — apply the 20 topics via `gh repo edit --add-topic` (Phase 1 of staged launch).
5. **crates.io / npm publish** — confirm `cargo binstall smos` and `npm install -g @yurvon_screamo/smos` work on a clean machine.
6. **LOCOMO position** — decide whether to explicitly opt out of benchmark wars in the README (B22) or stay silent.
7. **HUMAN GATE** — user approval of the bullet set and channel plan before any paste-ready copy is generated.

After this strategy is approved, the next concrete artifacts are:
- `artifacts/smos/2026-06-25/show-hn.md` — paste-ready Show HN post + first comment
- `artifacts/smos/2026-06-25/reddit-localllama.md` — paste-ready Reddit post
- `artifacts/smos/2026-06-25/devto-architecture.md` — paste-ready dev.to article

---

## Appendix A: Research artifacts

- Competitive research: `docs/research/smos-competitive-research-2026-06-25.md` (14 competitors, 10+ HN posts, 5+ Reddit threads, 3 academic papers)
- SEO research: `docs/research/seo-keyword-clusters-2026-06-25.md` (7 clusters, 30+ keywords, 20 GitHub topics, vocabulary inventory)

## Appendix B: Brand voice check

All bullets pass:
- No forbidden words (delve, tapestry, harness, unlock, revolutionize, leverage, seamless, robust without metric, etc.)
- Concrete numbers in every claim (643 MB, 0.5/0.7 thresholds, star counts, line citations)
- First-person author voice where applicable
- Honest limitations section present
- No "I'm excited to announce" / "thrilled to share"
- No emoji in technical writing
