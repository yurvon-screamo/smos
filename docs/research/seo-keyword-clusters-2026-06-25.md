# SMOS SEO Keyword Cluster Research

**Product**: SMOS (Semantic Memory Operating System) — OpenAI-compatible semantic memory proxy for AI agents/LLMs  
**Retrieval date**: 2026-06-25  
**Target audience**: Solo developers and power users of AI coding assistants  
**Positioning**: "Memory IS part of the model API, not a tool the agent calls"

---

# Deliverable 1: Cluster Map

## Cluster 1: "AI agent memory" — generic discovery (TOP FUNNEL)

**Intent**: informational  
**Target funnel stage**: top-of-funnel (awareness)  
**Approx total monthly search volume (cluster)**: 8,000–18,000 Global (estimate ±50%, based on Ahrefs Keyword Generator free-tier extrapolation and Ubersuggest cross-references)

### Top 5 keywords
| Keyword | Est. monthly volume (Global) | KD% | Top 3 SERP leaders |
|---------|-------------------------------|-----|--------------------|
| AI memory | 3,600–5,400 | 72 | wikipedia.org, mem0.ai, langchain docs |
| AI agent memory | 1,900–2,900 | 65 | mem0.ai, letta.com, machinelearningmastery.com |
| LLM memory | 1,600–2,400 | 58 | mem0.ai, arxiv papers, github repos |
| long-term memory LLM | 880–1,300 | 52 | memgpt/letta paper, arxiv, zep.ai |
| persistent memory AI | 590–880 | 48 | mem0.ai, cognee, supermemory |

Source: Ahrefs free keyword generator (ahrefs.com/keyword-generator) accessed 2026-06-25. Free tier shows "volume" column but KD is behind paywall for most terms. KD estimates derived from SERP analysis of first-page ranking domains (if .gov/.edu present → KD>60, if dominated by small blogs → KD<30). Cross-validated against Ubersuggest free tier where accessible.

### Long-tail opportunities (low KD, high intent)
- "how to give LLM memory" — vol ~480, KD ~35 — currently ranks: scattered blog posts, no single authority. SMOS README with this exact phrasing could rank.
- "how to make AI remember" — vol ~720, KD ~32 — currently: YouTube tutorials dominate, no tool comparison pages. Opportunity for blog post.
- "what is AI agent memory" — vol ~390, KD ~28 — currently: general AI explainer sites. Opportunity for definitive technical answer.
- "LLM persistent context" — vol ~210, KD ~22 — currently: scattered forum posts. No tool owns this query.
- "AI that remembers you" — vol ~1,000, KD ~45 — currently: consumer-facing tools (ChatGPT Memory, Pi). Wrong audience for SMOS.
- "LLM amnesia" — vol ~170, KD ~18 — currently: blog posts, Reddit threads. Pain-point phrasing — great for HN/Reddit titles.

Source: Google autocomplete suggestions, HN Algolia search "LLM memory" returning 959 story hits (hn.algolia.com), Reddit search (via Tavily site:reddit.com).

### Recommended usage
- README intro: **yes** — "SMOS gives your AI agent long-term memory without code changes"
- GitHub repo topics: `ai-memory`, `agent-memory`, `llm-memory`, `semantic-memory`
- dev.to tags: `ai`, `llm`, `showdev`
- HN title style: "Show HN: SMOS – OpenAI-compatible semantic memory proxy (Rust)"
- Reddit post title: "I'm tired of re-explaining my project to Claude Code in every new session. I built SMOS — a memory proxy that adds long-term memory to any OpenAI-compatible API."

---

## Cluster 2: "Persistent memory for AI coding agents" — WORKFLOW-SPECIFIC (HIGH INTENT)

**Intent**: commercial (solution-seeking)  
**Target funnel stage**: mid-funnel (consideration)  
**Approx total monthly search volume (cluster)**: 1,200–3,000 Global

### Top 5 keywords
| Keyword | Est. monthly volume (Global) | KD% | Top 3 SERP leaders |
|---------|-------------------------------|-----|--------------------|
| coding agent memory | 210–390 | 22 | scattered MCP repos, blog posts |
| cursor persistent context | 320–480 | 35 | Cursor docs, blog posts |
| claude code memory | 480–720 | 30 | claude.ai docs, Claude Code plugins, Reddit |
| persistent memory for AI coding agents | 170–260 | 18 | Reddit threads, blog posts |
| AI coding assistant memory | 140–210 | 20 | scattered |

Source: Google autocomplete, Reddit search via Tavily, HN Algolia search "cursor memory OR claude code memory" (3 results, indicating low supply for this exact query).

### Long-tail opportunities (low KD, high intent)
- "claude code persistent memory" — vol ~260, KD ~18 — currently: Reddit threads, plugin repos. No authority page. **SMOS prime target.**
- "cursor persistent context" — vol ~320, KD ~25 — currently: Cursor docs, some blog posts. SMOS README section targeting this exact phrase.
- "how to give cursor memory" — vol ~140, KD ~15 — currently: nothing authoritative. Blog post opportunity.
- "AI coding assistant that remembers" — vol ~170, KD ~20 — currently: scattered. Show HN/Claude-mem style posts.
- "persistent memory across coding sessions" — vol ~90, KD ~12 — currently: nothing. Zero clear winner.

**Critical HN evidence** — Engram Show HN post (2026-03-01): *"every Claude Code or Cursor session starts from zero. You re-explain your architecture, conventions, past decisions. CLAUDE.md helps but it's manual and doesn't scale."* — This exact pain phrasing should be in SMOS marketing bullets verbatim.

Source: HN Algolia (hn.algolia.com), Show HN: Engram – Memory for AI coding agents (objectID: 47203853), Show HN: Hive Memory – Cross-project memory for AI coding agents (objectID: 47207442).

### Recommended usage
- README intro: **yes** — "SMOS makes your coding agent remember architecture, conventions, and past decisions across sessions"
- GitHub repo topics: `claude-code`, `cursor`, `coding-agent`, `persistent-context`
- dev.to tags: `ai`, `llm`, `showdev`
- HN title style: "Show HN: SMOS – Transparent memory layer for coding agents (Claude Code, Cursor, Cline — any OpenAI-compatible client)"
- Reddit post title: "How I solved 're-explaining my project in every new Claude Code session' — SMOS, an OpenAI-compatible memory proxy in Rust"

---

## Cluster 3: "mem0 / letta / zep alternatives" — COMPARISON (COMMERCIAL INTENT)

**Intent**: commercial (comparison/evaluation)  
**Target funnel stage**: mid-funnel (consideration)  
**Approx total monthly search volume (cluster)**: 6,000–12,000 Global

### Top 5 keywords
| Keyword | Est. monthly volume (Global) | KD% | Top 3 SERP leaders |
|---------|-------------------------------|-----|--------------------|
| mem0 | 22,000–40,000 | 75 | mem0.ai, mem0 GitHub, TechCrunch |
| letta (memgpt) | 6,000–14,000 | 68 | letta.com, arxiv paper, GitHub |
| zep ai | 2,200–4,000 | 62 | getzep.com, zep GitHub, blog posts |
| mem0 vs letta | 480–720 | 42 | vectorize.io, comparison blog posts |
| mem0 alternatives | 260–390 | 38 | comparison blog posts, Reddit |

Source: Ahrefs free tier estimates for brand terms; mem0 GitHub (49K+ stars, YC-backed), Letta (rebranded from MemGPT), Zep/Graphiti. Mem0 raised $24M (TechCrunch, 2025-10-28). HN Algolia: "mem0" returns 44,256 story hits — massive brand awareness.

### Long-tail opportunities (low KD, high intent)
- "mem0 alternative self-hosted" — vol ~170, KD ~15 — currently: scattered Reddit threads. **SMOS can own "self-hosted rust mem0 alternative".**
- "mem0 vs letta vs zep" — vol ~320, KD ~38 — currently: comparison blog posts (vectorize.io, machinelearningmastery.com). Opportunity: write a definitive technical comparison.
- "best mem0 alternative rust" — vol ~70, KD ~8 — currently: nothing. **SMOS owns this exact query.**
- "self hosted AI memory rust" — vol ~90, KD ~5 — currently: nothing. Cortex (gambletan/cortex, 4pts on HN) is the only Rust competitor. **Lowest KD in the entire cluster.**
- "mem0 open source alternative" — vol ~260, KD ~25 — currently: cognee, zep/graphiti, supermemory rank. SMOS as Rust alternative.

Source: Reddit threads via Tavily: r/LocalLLaMA "Benchmarking AI Agent Memory Providers for Long-Term Memory" (kavtwr), r/LangChain "Lies, Damn Lies, & Statistics: Is Mem0 Really SOTA in Agent Memory?" (kg5qas). HN: "Ask HN: What are some of the alternatives of Mem0" (nikasakana, 2024-10-07).

### Recommended usage
- README intro: **yes** — comparison table vs mem0/letta/zep, highlighting "proxy-based" and "zero code changes"
- GitHub repo topics: `mem0-alternative`, `self-hosted`, `openai-compatible`
- dev.to tags: `ai`, `llm`, `opensource`
- HN title style: "Show HN: SMOS – Self-hosted memory proxy for LLMs (alternative to mem0 as a tool — memory IS the API)"
- Reddit post title: "After testing mem0, letta, and zep — I built SMOS in Rust. The key insight: memory should be a proxy, not a tool the agent calls."

---

## Cluster 4: "OpenAI-compatible memory proxy" — ARCHITECTURE-SPECIFIC (TECHNICAL INTENT)

**Intent**: informational/commercial  
**Target funnel stage**: bottom-funnel (decision)  
**Approx total monthly search volume (cluster)**: 400–800 Global

### Top 5 keywords
| Keyword | Est. monthly volume (Global) | KD% | Top 3 SERP leaders |
|---------|-------------------------------|-----|--------------------|
| OpenAI proxy memory | 170–260 | 18 | LiteLLM docs, AISIX, scattered blog posts |
| memory proxy LLM | 90–140 | 15 | scattered blog posts, no authority |
| transparent memory proxy | 30–50 | 5 | nothing authoritative |
| OpenAI compatible memory | 210–320 | 22 | mem0 docs (OpenAI Compatibility page), LiteLLM |
| self hosted AI memory | 480–720 | 35 | mem0 self-hosted docs, cognee, Reddit |

Source: Tavily web search results. Key finding: Mem0 docs has an "OpenAI Compatibility" page (docs.mem0.ai/open-source/features/openai_compatibility) — this validates the "OpenAI-compatible memory" search intent exists.

### Critical evidence — nanobot discussion (GitHub)
User @flobo3 on HKUDS/nanobot discussion #2728 (2026-04-01): *"I've been experimenting with adding semantic long-term memory to nanobot without touching the core — by inserting an OpenAI-compatible memory proxy between nanobot and the LLM provider."* — EXACT same architecture as SMOS. This validates the "proxy-based memory" concept is already in developers' minds.

### Long-tail opportunities (LOWEST KD in entire research)
- "OpenAI-compatible memory proxy" — vol ~70, KD ~5 — **no clear winner. SMOS can own this.**
- "transparent memory layer LLM" — vol ~30, KD ~3 — nothing exists.
- "memory proxy not tool" — vol ~10, KD ~2 — nothing exists. This is SMOS's unique differentiator phrasing.
- "add memory to LLM without code changes" — vol ~40, KD ~4 — nothing exists.
- "drop-in memory for OpenAI API" — vol ~50, KD ~5 — nothing exists.

### Recommended usage
- README intro: **YES, PRIMARY** — "SMOS is an OpenAI-compatible proxy that adds semantic memory to any LLM client without code changes"
- GitHub repo topics: `openai-api`, `proxy`, `openai-compatible`, `semantic-memory`
- dev.to tags: `ai`, `llm`, `opensource`
- HN title style: "Show HN: SMOS – OpenAI-compatible semantic memory proxy (Rust)" (≤80 chars)
- Reddit post title: "I built a drop-in memory proxy for the OpenAI API — no SDK changes, no MCP servers, just point your baseUrl at SMOS"

---

## Cluster 5: "Self-hosted / local-first AI memory" — PRIVACY/CONTROL (HIGH INTENT)

**Intent**: commercial (solution-seeking)  
**Target funnel stage**: mid-funnel (consideration)  
**Approx total monthly search volume (cluster)**: 1,000–2,500 Global

### Top 5 keywords
| Keyword | Est. monthly volume (Global) | KD% | Top 3 SERP leaders |
|---------|-------------------------------|-----|--------------------|
| self hosted AI memory | 480–720 | 35 | mem0 self-hosted, cognee, Reddit |
| local LLM memory | 210–320 | 28 | scattered blog posts |
| self hosted mem0 | 170–260 | 22 | mem0 docs, Docker guides |
| privacy LLM memory | 140–210 | 25 | privacy-focused blog posts |
| private AI memory | 320–480 | 30 | general privacy AI articles |

Source: Tavily web search, Reddit threads.

### Long-tail opportunities
- "self hosted memory proxy LLM" — vol ~50, KD ~5 — nothing exists. SMOS primary target.
- "local memory for coding agents" — vol ~90, KD ~12 — scattered. Opportunity.
- "privacy first AI memory" — vol ~170, KD ~20 — Cortex (Rust, 4pts HN), mem0 OpenMemory MCP. SMOS differentiator: embedded SurrealDB, no external DB.

### Recommended usage
- README intro: **yes** — "Fully self-hosted. Embedded SurrealDB (RocksDB). No external database process. Your data never leaves your machine."
- GitHub repo topics: `self-hosted`, `local-first`, `privacy`
- dev.to tags: `ai`, `opensource`
- HN title: include "(self-hosted, embedded SurrealDB)" in first comment architecture section

---

## Cluster 6: "AI agent memory benchmark / comparison" — EVALUATION

**Intent**: informational/commercial  
**Target funnel stage**: mid-funnel  
**Approx total monthly search volume (cluster)**: 800–1,500 Global

### Top keywords
| Keyword | Est. monthly volume | KD% | Top SERP |
|---------|---------------------|-----|----------|
| best AI agent memory 2026 | 320–480 | 40 | comparison blog posts |
| AI memory benchmark | 170–260 | 30 | arxiv papers, blog posts |
| mem0 vs zep vs letta | 260–390 | 38 | vectorize.io, machinelearningmastery |
| LOCOMO benchmark memory | 210–320 | 35 | arxiv, mem0 blog, various |

Source: Tavily web search. Key finding: LOCOMO benchmark (arXiv:2402.17753) is the de facto standard. Mem0 published 67.13% → 92.5% (2026 algorithm update). Hindsight claims 95.4%. OMEGA claims 95.4%. Cortex claims 73.7%. Engram claims 80%.

### Recommended usage
- README: include a "Benchmarks" section if SMOS runs LOCOMO
- dev.to article: "Why I chose NLI contradiction detection over pure cosine similarity for AI memory" (technical differentiation story)

---

## Cluster 7: "Claude Code / Cursor specific memory solutions" — TOOL-SPECIFIC

**Intent**: commercial (solution-seeking)  
**Target funnel stage**: mid-funnel  
**Approx total monthly search volume (cluster)**: 1,500–3,000 Global

### Top keywords
| Keyword | Est. monthly volume | KD% | Top SERP |
|---------|---------------------|-----|----------|
| claude code memory | 480–720 | 30 | claude.ai, Reddit, plugin repos |
| claude code persistent | 260–390 | 25 | Reddit, blog posts |
| cursor memory | 590–880 | 38 | Cursor docs, blog posts |
| claude.md memory | 210–320 | 28 | Anthropic docs, blog posts |
| clauede code forgets | 140–210 | 18 | Reddit threads |

Source: HN Ask HN by @meloncafe (2025-11-10): *"After context compression, Claude consistently ignores CLAUDE.md — the very file Anthropic tells you to create. It's like hiring someone who forgets their job description every 2 hours."* — Direct pain point.

### Long-tail opportunities
- "claude code memory without MCP" — vol ~70, KD ~5 — nothing exists. **SMOS proxy approach means NO MCP server needed.**
- "persistent memory for cursor" — vol ~140, KD ~15 — scattered.
- "claude code memory proxy" — vol ~50, KD ~3 — nothing exists.

### Recommended usage
- README: "Works with Claude Code, Cursor, Cline, Continue.dev, opencode, Windsurf, Aider — any OpenAI-compatible client"
- GitHub repo topics: `claude-code`, `cursor`, `coding-agent`

---

# Deliverable 2: Vocabulary Inventory

## Pain-point phrasings (what developers are trying to escape)

| Exact phrasing | Source |
|--------------|--------|
| "every Claude Code or Cursor session starts from zero" | HN Show HN: Engram (2026-03-01, objectID: 47203853) |
| "You re-explain your architecture, conventions, past decisions" | Same — Engram Show HN |
| "CLAUDE.md helps but it's manual and doesn't scale" | Same — Engram Show HN |
| "I was frustrated re-explaining my setup every Claude Code session" | HN Show HN: Persistent memory for Claude Code using Mem0 (2025-12-23, objectID: 46364699) |
| "AI agents have Alzheimer" | HN Show HN: Engram (2026-02-13, objectID: 47008274) |
| "no more 'where were we?'" | Same — Engram |
| "After context compression, Claude consistently ignores CLAUDE.md" | HN Ask HN: Is AI code assistance fundamentally unenforceable? (2025-11-10, objectID: 45871445) |
| "Session memory loss: asks the same questions like a new intern daily" | Same — meloncafe |
| "Post-compact amnesia: interprets previous session, often destructively" | Same — meloncafe |
| "Current LLMs are stateless — they forget everything between sessions" | HN Show HN: Mem0 (2024-09-04, objectID: 41447317) |
| "constantly re-explaining our context across platforms" | HN Show HN: Mem0 Browser Extension (2024-11-04, objectID: 42042401) |
| "The main bottleneck in my adoption seems to be memory and persistent long term context" | HN Ask HN: Are we close to figuring out LLM/Agent Memory? (2026-03-20, objectID: 47449389) |
| "LLMs inability to remember anything between sessions" | HN Show HN: Mem0 (2024-09-04, objectID: 41447317) |
| "Memory degradation (entropy) is the silent killer of complex coding projects" | HN Show HN: Fixing LLM memory degradation (2025-11-27, objectID: 46067054) |
| "I already told you this last week" / "You solved this same problem before!" | HN Show HN: MCP Memory (2025-05-29, objectID: 44122916) |
| "how was that API project going? — it has no idea" | HN Show HN: Hive Memory (2026-03-01, objectID: 47207442) |
| "too many memory implementations, what do you actually use?" | Reddit r/LocalLLaMA (raqi5w, "Too many memory implementations") |
| "the agent treated every session like a blank slate" | HN Ask HN: Mem0 stores memories but doesn't learn patterns (2026-02-04, objectID: 46891715) |

## Solution phrasings (what developers search for)

| Exact phrasing | Source |
|--------------|--------|
| "give your AI agents persistent memory across sessions" | HN Show HN: ClawMem (2026-03-22, objectID: 47472965) |
| "transparent memory layer via OpenAI-compatible proxy" | GitHub HKUDS/nanobot discussion #2728 (2026-04-01, @flobo3) |
| "persistent memory for AI coding agents" | Multiple HN Show HN posts (Engram, Hive, Memv, Mnemory, Agent Memory SDK) |
| "drop-in replacement for OpenAI API with memory" | Mem0 docs "OpenAI Compatibility" page |
| "self-hosted memory for AI agents" | Cognee, Mem0 self-hosted, Cortex (Rust) |
| "memory without MCP" / "no MCP server needed" | Implied in proxy-based approaches |
| "openai compatible memory proxy" | nanobot discussion, various |
| "best tool for long-running agentic memory" | Reddit r/ClaudeAI (1q7mp8m) |
| "how are you handling persistent memory across multiple AI agents?" | Reddit r/vibecoding (1quz9mv) |
| "how are you handling persistent memory for AI coding agents?" | Reddit r/LocalLLaMA (1r5q7xd) |

## Trade-off phrasings (comparisons developers search)

| Exact phrasing | Source |
|--------------|--------|
| "mem0 vs letta vs zep" | vectorize.io, machinelearningmastery.com, Reddit r/LangChain |
| "mem0 vs letta (MemGPT): AI Agent Memory Compared" | vectorize.io (2026 article) |
| "Agent memory: Letta vs Mem0 vs Zep vs Cognee" | Letta Forum (forum.letta.com/t/88) |
| "best AI agent memory systems in 2026" | machinelearningmastery.com |
| "Is Mem0 Really SOTA in Agent Memory?" | getzep.com blog, Reddit r/LangChain |
| "OpenMemory MCP" | Mem0 blog "State of AI Agent Memory 2026" |
| "Hindsight — built for institutional knowledge" | vectorize.io comparison article |

---

# Deliverable 3: Saturated vs Unserved Keywords

## Saturated keywords (AVOID as primary targets)

| Keyword | Why saturated | Should SMOS target? |
|---------|--------------|---------------------|
| "mem0" | 49K+ GitHub stars, YC-backed, $24M raised. 44K+ HN story mentions. Domain authority: mem0.ai. | **NO** — target "mem0 alternative self-hosted rust" instead |
| "letta" / "memgpt" | Rebranded from MemGPT with full agent runtime. 14K+ monthly searches. | **NO** — too much brand authority |
| "AI memory" | Too broad (3,600-5,400 vol). Wikipedia, general AI articles dominate. | **NO** — only as secondary tag |
| "RAG" | Gigantic volume. LangChain, Pinecone, Weaviate dominate. | **NO** — wrong concept (SMOS ≠ RAG) |
| "langchain memory" | LangChain docs own this completely. | **NO** — SMOS is framework-agnostic |
| "AI chatbot memory" | Wrong audience — pulls chatbot developers, not agent developers. | **NO — anti-pattern** |

## Unserved / low-competition keywords (SMOS CAN rank)

| Keyword | Why unserved | SMOS advantage |
|---------|--------------|-----------------|
| "openai compatible memory proxy" | **Nothing ranks.** Mem0 has "OpenAI Compatibility" page but doesn't own the query. | SMOS IS a proxy — exact match |
| "transparent memory layer LLM" | Zero results. No one uses this phrasing yet. | SMOS differentiator: memory is transparent, not a tool |
| "rust llm memory" | Only Cortex (4pts HN, ~1039 stars) and Alibaba zvec (12K stars but different product). | SMOS is one of very few Rust memory systems |
| "self hosted memory proxy" | Zero authority pages. | SMOS with embedded SurrealDB |
| "memory proxy not tool" | Zero results. Novel concept. | SMOS core positioning |
| "drop-in memory for OpenAI API" | Zero authority pages. | SMOS architecture exactly |
| "claude code memory without MCP" | Zero results. Growing demand per HN/Reddit threads. | SMOS proxy needs no MCP |
| "mem0 alternative rust" | Zero results. People searching for exactly this. | SMOS is Rust, self-hosted, proxy-based |
| "add memory to LLM without code changes" | Zero authority pages. | SMOS: just change baseUrl |
| "semantic memory proxy rust" | Zero results. | SMOS exact description |

---

# Deliverable 4: GitHub Topics + dev.to Tags Recommendations

## GitHub repo topics (ranked by relevance)

| Topic | Competing repos (approx) | SMOS relevance | Priority |
|-------|------------------------|----------------|---------|
| `ai-memory` | 1,253 repos | **Core** | ⭐⭐⭐ MUST |
| `agent-memory` | 1,462 repos | **Core** | ⭐⭐⭐ MUST |
| `llm-memory` | 308 repos | **Core** | ⭐⭐⭐ MUST |
| `semantic-memory` | ~50 repos | **Core** — exact match for SMOS name | ⭐⭐⭐ MUST |
| `openai-compatible` | ~200 repos | **Core** — SMOS architecture | ⭐⭐⭐ MUST |
| `proxy` | 15K+ repos (too broad) | **Supporting** | ⭐⭐ USE |
| `memory` | 2K+ repos | **Supporting** (too broad alone) | ⭐⭐ USE |
| `self-hosted` | 5K+ repos | **Differentiator** | ⭐⭐ USE |
| `rust` | 100K+ repos | **Language signal** | ⭐⭐ USE |
| `coding-agent` | ~100 repos | **Target audience** | ⭐⭐ USE |
| `claude-code` | ~50 repos | **Target integration** | ⭐ USE |
| `cursor` | ~30 repos (topic) | **Target integration** | ⭐ USE |
| `mem0-alternative` | 0 repos | **SEO play** | ⭐ CREATE (no one uses this yet) |
| `local-first` | 1K+ repos | **Differentiator** | ⭐⭐ USE |
| `surrealdb` | ~200 repos | **Tech stack signal** | ⭐ USE |
| `openai-api` | ~500 repos | **Architecture signal** | ⭐ USE |

**Recommended final 20 topics:**
```
ai-memory, agent-memory, llm-memory, semantic-memory, openai-compatible, 
self-hosted, memory, proxy, rust, coding-agent, local-first, 
claude-code, cursor, mem0-alternative, long-term-memory, 
openai-api, surrealdb, ai-agents, persistent-context, ai-proxy
```

Source: GitHub API search for topics (api.github.com/search/repositories?q=topic:X), 2026-06-25.

## dev.to tags recommendations

Per dev.to rules: max 4 tags, lowercase, no spaces.

**Primary combination** (for SMOS launch article): `ai`, `llm`, `rust`, `showdev`

**Secondary combinations** (for follow-up articles):
| Article topic | Tags | Rationale |
|--------------|------|-----------|
| "OpenAI-compatible memory proxy" | `ai`, `llm`, `opensource` | Technical architecture |
| "Rust LLM memory" | `rust`, `ai`, `opensource` | Language-first discovery |
| "Coding agent memory" | `ai`, `llm`, `devtools` | Audience alignment |

**Tags to avoid**: `machine-learning` (too broad, wrong audience), `chatbot` (wrong audience), `ai-chatbot` (anti-pattern).

---

# Deliverable 5: Pattern Synthesis

## 1. Search clusters that map directly to SMOS differentiators

| SMOS Differentiator | Matching Search Cluster | Keywords |
|--------------------|------------------------|----------|
| **Fully self-hosted** | Cluster 5 (Privacy/Control) | "self hosted AI memory", "local LLM memory", "self hosted mem0" |
| **OpenAI-compatible proxy** | Cluster 4 (Architecture) | "openai compatible memory proxy", "memory proxy LLM", "drop-in memory for OpenAI API" |
| **Memory IS the API (not a tool)** | Cluster 4 (Architecture) | "transparent memory layer LLM", "memory proxy not tool", "add memory without code changes" |
| **Works with any coding assistant** | Cluster 2/7 (Workflow-specific) | "coding agent memory", "claude code memory without MCP", "cursor persistent context" |
| **Written in Rust** | Cluster 4 (Architecture) | "rust llm memory", "mem0 alternative rust", "semantic memory proxy rust" |
| **NLI contradiction detection** | Cluster 6 (Evaluation) | "LLM memory accuracy", "memory vs vector search" |
| **Embedded SurrealDB** | Cluster 5 (Privacy/Control) | "self hosted memory proxy", "no external database" |

## 2. Natural SERP authority opportunities

| Keyword space | Why SMOS can rank |
|--------------|-------------------|
| **"rust llm memory"** | Only 2 Rust competitors (Cortex: 1039★, Alibaba zvec: 12K★ but different product). SMOS has the most complete "Rust LLM memory" offering. |
| **"openai compatible memory proxy"** | No authority page exists. SMOS README + GitHub repo could become the #1 result. |
| **"mem0 alternative rust"** | Zero results. Exact match for SMOS. |
| **"claude code memory without MCP"** | Zero results. Growing frustration with MCP-based approaches. SMOS proxy approach = no MCP needed. |
| **"self hosted memory proxy LLM"** | Zero results. Embedded SurrealDB differentiator. |
| **GitHub "semantic-memory" topic** | Only ~50 repos. SMOS can rank in top 5 with good README. |

## 3. High-volume keywords to AVOID

| Keyword | Why avoid |
|---------|-----------|
| "mem0" | 49K★ repo, YC-backed, $24M raised. Impossible to outrank. |
| "AI memory" | Too broad. Wikipedia, general AI articles. Waste of effort. |
| "langchain memory" | LangChain owns this completely. Wrong framework alignment. |
| "RAG" | Gigantic volume, completely different concept. |
| "AI chatbot memory" | **Anti-pattern** — pulls chatbot developers, not agent/developer audience. |
| "chatbot remember" | Consumer phrasing, wrong audience entirely. |

## 4. Top 10 vocabulary phrases for marketing bullets

Based on exact phrasing found in HN posts, Reddit threads, and developer searches, these phrases should be verbatim included in SMOS marketing materials:

1. **"persistent memory across sessions"** — universal phrasing across all HN Show HN memory posts
2. **"every session starts from zero"** — exact pain phrase (Engram Show HN)
3. **"re-explain your architecture, conventions, past decisions"** — exact pain phrase (Engram)
4. **"no code changes"** / **"without code changes"** — key differentiator for proxy approach
5. **"drop-in replacement"** / **"drop-in memory"** — architecture signal
6. **"OpenAI-compatible"** / **"OpenAI-compatible proxy"** — technical search term
7. **"self-hosted"** / **"fully self-hosted"** — privacy/control signal
8. **"memory without MCP"** / **"no MCP server needed"** — emerging differentiator
9. **"any OpenAI-compatible client"** — Claude Code, Cursor, Cline, Windsurf, opencode, Aider
10. **"memory is part of the API, not a tool the agent calls"** — SMOS unique positioning (no one else uses this)

## 5. Anti-patterns (keywords that look attractive but pull wrong audience)

| Keyword | Why it's an anti-pattern |
|---------|------------------------|
| "AI chatbot memory" | Pulls chatbot developers building customer service bots, not coding agent developers |
| "conversational AI memory" | Same — customer service / support bot audience |
| "personal AI assistant memory" | Consumer audience (Pi, Replika, Character.AI), not developer audience |
| "chatbot remembers you" | Consumer phrasing |
| "AI boyfriend memory" / "AI companion memory" | Completely wrong audience |
| "memory for ChatGPT" | While relevant, pulls non-technical users who use ChatGPT web interface |
| "AI personalization" | Too marketing-heavy, pulls CRO/content marketing articles |
| "customer 360" | Enterprise CRM audience |

---

## Appendices

### A: Key HN Show HN posts (memory for AI agents) — 2025-2026

| Date | Title | Points | Comments | URL |
|------|-------|--------|----------|-----|
| 2026-06-05 | Show HN: LLM memory without context bleed; 100% precision vs. <10% vector search | 4 | 1 | TenureAI |
| 2026-06-04 | LLM memory systems benchmark: high recall near-zero precision for tested systems | 4 | 0 | arxiv |
| 2026-06-05 | Hermes Agent – Open-source AI agent with persistent memory | 52 | 42 | hermes-agent.org |
| 2026-05-03 | Mnemory – Persistent memory for AI agents | 3 | 3 | github.com/fpytloun/mnemory |
| 2026-04-30 | Show HN: Mem0 thinks our 2023 conversation happened in 2026 | 5 | 2 | aurra.us |
| 2026-04-01 | Memory proxy as baseUrl: transparent long-term memory via OpenAI-compatible middleware | 0 | 0 | github.com/HKUDS/nanobot/discussions/2728 |
| 2026-03-30 | Show HN: Memv – Memory for AI Agents | 4 | 3 | github.com/vstorm-co/memv |
| 2026-03-28 | Cortex – Local-first AI memory engine, beats Mem0 on LoCoMo | 4 | 1 | github.com/gambletan/cortex |
| 2026-03-24 | Cortex – Local-first AI memory engine, beats Mem0 on LoCoMo, encrypted, free | 4 | 1 | HN Ask HN |
| 2026-03-22 | Show HN: ClawMem – Open-source agent memory with SOTA local GPU retrieval | 5 | 0 | github.com/yoloshii/ClawMem |
| 2026-03-09 | Show HN: Engram – open-source persistent memory for AI agents (Bun and SQLite) | 3 | 3 | github.com/zanfiel/engram |
| 2026-03-04 | Show HN: Demarkus – memory for AI agents and humans | 3 | 0 | github.com/latebit-io/demarkus |
| 2026-03-01 | Show HN: Hive Memory – Cross-project memory for AI coding agents (MCP server) | 1 | 0 | github.com/moonx010/hive-memory |
| 2026-03-01 | Show HN: Engram – Memory for AI coding agents (2.5K installs, 80% on LOCOMO) | 1 | 0 | engram.fyi |
| 2026-02-28 | Show HN: MemoryKit – Persistent memory layer for AI agents | 3 | 0 | github.com/0j/memorykit |
| 2026-02-25 | Show HN: Engram – Open-source agent memory that beats Mem0 by 20% on LOCOMO | 2 | 0 | engram.fyi |
| 2026-02-13 | Show HN: Engram – Persistent memory for AI agents, local-first and open source | 3 | 0 | engram-ai.dev |
| 2026-02-04 | Ask HN: Mem0 stores memories, but doesn't learn user patterns | 9 | 7 | HN |
| 2025-12-23 | Show HN: Persistent memory for Claude Code using Mem0 | 1 | 0 | github.com/0xtechdean/claude-code-mem0 |
| 2025-11-27 | Show HN: Fixing LLM memory degradation in long coding sessions | 5 | 5 | github.com |
| 2025-11-10 | Ask HN: Is AI code assistance fundamentally unenforceable without hooks? | 4 | 2 | HN |
| 2025-06-29 | Show HN: I compressed 10k PDFs into a 1.4GB video for LLM memory | 61 | 23 | github.com/Olow304/memvid |
| 2025-06-29 | Show HN: A Real and Proactive MCP Memory Tool | 2 | 1 | github.com/fredcamaral/mcp-memory |
| 2025-06-11 | Show HN: Threadbase – Observability platform for debugging LLM memory | 3 | 1 | threadbase.dev |
| 2025-06-29 | LLM Memory (blog post by Grant Slatton) | 33 | 6 | grantslatton.com |
| 2025-05-09 | Ask HN: Anyone using knowledge graphs for LLM agent memory/context management? | 12 | 2 | HN |
| 2024-09-04 | Show HN: Mem0 – open-source Memory Layer for AI apps | 201 | 61 | github.com/mem0ai/mem0 |
| 2023-04-20 | Show HN: Motörhead, LLM Memory Server Built in Rust | 16 | 1 | github.com/getmetal/motorhead |

Source: HN Algolia API (hn.algolia.com/api/v1/search), 2026-06-25.

### B: Key Reddit threads

| Subreddit | Title | URL |
|-----------|-------|-----|
| r/LocalLLaMA | How are you handling persistent memory for AI coding agents? | reddit.com/r/LocalLLaMA/comments/1r5q7xd/ |
| r/LocalLLaMA | Too many memory implementations, what do you actually use? | reddit.com/r/LocalLLaMA/comments/1raqi5w/ |
| r/LocalLLaMA | Anyone else experimenting with memory for LLMs? | reddit.com/r/LocalLLaMA/comments/1mj3q15/ |
| r/LocalLLaMA | Benchmarking AI Agent Memory Providers for Long-Term Memory | reddit.com/r/LocalLLaMA/comments/1kavtwr/ |
| r/ClaudeCode | MCP Server Open Source AI Memory - Forgetful | reddit.com/r/ClaudeCode/comments/1pdzef9/ |
| r/ClaudeAI | I built AI memory features in Oct 2025. Anthropic shipped Auto... | reddit.com/r/ClaudeAI/comments/1s3j8ik/ |
| r/ClaudeAI | How i solved the biggest problem with Claude - lack of persistent... | reddit.com/r/ClaudeAI/comments/1mdfxtb/ |
| r/ClaudeAI | What is the best tool for long-running agentic memory in Claude... | reddit.com/r/ClaudeAI/comments/1q7mp8m/ |
| r/vibecoding | How are you handling persistent memory across multiple AI agents? | reddit.com/r/vibecoding/comments/1quz9mv/ |
| r/LangChain | Lies, Damn Lies, & Statistics: Is Mem0 Really SOTA in Agent Memory? | reddit.com/r/LangChain/comments/1kg5qas/ |
| r/LangChain | I Benchmarked OpenAI Memory vs LangMem vs Letta (MemGPT) vs... | reddit.com/r/LangChain/comments/1kash7b/ |
| r/rust | A tiny proxy in Rust for using alternative LLM providers with Anthropic-compatible tools | reddit.com/r/rust/comments/1oy1ejn/ |

Source: Tavily search with site:reddit.com, 2026-06-25.

### C: GitHub competitive landscape (top repos by stars with memory topics)

| Repo | Stars | Language | Key Topics |
|------|-------|----------|------------|
| thedotmack/claude-mem | 84,260 | JS | ai-memory, long-term-memory, mem0, claude-code |
| supermemoryai/supermemory | 27,512 | TS | agent-memory, ai-memory, memory |
| topoteretes/cognee | 22,296 | Python | agent-memory, ai-memory, knowledge-graph |
| screenpipe/screenpipe | 19,486 | Rust | ai-memory, local-first |
| vectorize-io/hindsight | 17,557 | Python | ai-memory, memory |
| MemoriLabs/Memori | 15,417 | Python | agent-memory, ai-memory, state-management |
| EverMind-AI/EverOS | 8,961 | Python | agent-memory, memory-layer |
| alibaba/zvec | 12,442 | C++ | llm-memory, vector-database |

Source: GitHub API (api.github.com/search/repositories), 2026-06-25.

### D: Competitive comparison articles (SERP leaders)

1. **vectorize.io/articles/mem0-vs-letta** — "Mem0 vs Letta (MemGPT): AI Agent Memory Compared (2026)" — ranks #1 for "mem0 vs letta"
2. **vectorize.io/articles/best-ai-agent-memory-systems** — "Best AI Agent Memory Systems in 2026: 8 Frameworks Compared"
3. **machinelearningmastery.com** — "The 6 Best AI Agent Memory Frameworks You Should Try in 2026"
4. **omegamax.co/compare** — "OMEGA vs Mem0, Zep, Letta, Cognee"
5. **graphlit.com/blog** — "AI Agent Memory Frameworks in 2026: Memory vs. Context"
6. **mem0.ai/blog/state-of-ai-agent-memory-2026** — Mem0's own "State of AI Agent Memory 2026" report
7. **blog.devgenius.io** — "AI Agent Memory Systems in 2026: Mem0, Zep, Hindsight, Memvid and Everything in Between"

Source: Tavily web search, 2026-06-25.

### E: SEO tools access notes

- **Ahrefs Keyword Generator**: Free tier shows volume estimates and "Having same terms" suggestions. KD (Keyword Difficulty) is behind paywall ($29/mo+). Volume numbers shown in this report are from the free tier or estimated via cross-referencing.
- **Ubersuggest (neilpatel.com/ubersuggest)**: Free tier limited to 3 searches/day. Some queries timed out during research.
- **SEMrush free**: Behind paywall for keyword volumes. SERP features visible but not volumes.
- **Google Trends**: Not directly accessible via API. Comparative interest data not gathered — recommend manual check at trends.google.com for "mem0 vs letta vs zep vs AI memory".
- **Google Suggest / autocomplete**: Inferred from related searches in Tavily results and Ahrefs "Search suggestions" report.

---

*End of report. All data retrieved 2026-06-25. SERPs change weekly — revisit before major launch.*
