# SMOS Landing Page — Техническое Задание

> **Document type:** Полное ТЗ для разработчика лендинга
> **Created:** 2026-06-25
> **Status:** DRAFT — awaiting HUMAN GATE approval
> **Based on:** Marketing strategy (`marketing/strategies/smos.md`), competitive research, SEO keyword research, README rewrite
> **Target audience:** Solo developers and power users of AI coding assistants

---

## 1. Project Overview

### 1.1 Что строим

Одностраничный landing page для SMOS (Semantic Memory Operating System) — OpenAI-compatible memory proxy для AI coding agents. Страница должна:

- Объяснить что такое SMOS за 10 секунд (above the fold)
- Показать путь к установке за 30 секунд (Quick Start виден без скролла на десктопе)
- Занять SERP #1 по unserved keyword кластерам ("openai compatible memory proxy", "semantic memory proxy rust", "mem0 alternative self-hosted rust")
- Конвертировать посетителя в GitHub star / npm install / `smos init`

### 1.2 Цели и метрики

| Метрика | Target | Как измеряем |
|---|---|---|
| Lighthouse Performance | ≥ 95 | PageSpeed Insights |
| Lighthouse SEO | 100 | PageSpeed Insights |
| LCP (Largest Contentful Paint) | < 1.5s | Core Web Vitals |
| CLS (Cumulative Layout Shift) | < 0.1 | Core Web Vitals |
| INP (Interaction to Next Paint) | < 200ms | Core Web Vitals |
| Time to Quick Start visible | < 2s (above fold on desktop) | Manual / Lighthouse |
| Bounce rate | < 40% | Analytics |
| GitHub stars conversion | ~2-5% of landing visitors | GitHub Insights |
| SERP position for "openai compatible memory proxy" | #1 within 30 days | Google Search Console |

### 1.3 Что НЕ строим

- НЕ multi-page site (пока — только single landing page)
- НЕ blog (Phase 2)
- НЕ interactive demo (Phase 2 — если будет, то как embed/iframe)
- НЕ pricing page (SMOS = MIT, бесплатно)
- НЕ signup / login (self-hosted OSS, no accounts)

---

## 2. Target Audience

### 2.1 Primary persona

**Имя:** Alex
**Кто:** Senior backend developer, использует Cursor + Claude Code ежедневно
**Боль:** «Every new chat starts from scratch. I re-explain my architecture, my conventions, every decision I already made.»
**Что ищет:** "cursor persistent context", "claude code memory", "AI coding assistant memory"
**Скептицизм:** High — видел mem0, Letta, Zep; разочарован benchmark wars
**Что убедит:** Конкретная архитектура (proxy, не tool), single binary (no Docker), self-hosted sovereignty, honest limitations

### 2.2 Secondary persona

**Имя:** Jordan
**Кто:** AI startup founder, строит multi-agent систему
**Боль:** Нужна memory infrastructure без framework lock-in
**Что ищет:** "mem0 alternative self-hosted", "rust llm memory", "multi-agent memory isolation"
**Что убедит:** Multi-persona isolation, Rust performance, OpenAI API compatibility

---

## 3. SEO Strategy

### 3.1 Target keywords (из verified research)

**Primary (own these — zero/low SERP competition):**

| Keyword | Est. KD | Current SERP | SMOS advantage |
|---|---|---|---|
| `openai compatible memory proxy` | ~5 | Nothing ranks | SMOS IS this exactly |
| `semantic memory proxy rust` | ~3 | Zero results | Exact match |
| `memory proxy not tool` | ~2 | Zero results | SMOS unique positioning |
| `self hosted memory proxy` | ~5 | Zero authority pages | Embedded SurrealDB |
| `mem0 alternative self hosted rust` | ~8 | Zero results | Exact match |
| `claude code memory without mcp` | ~5 | Zero results | Proxy needs no MCP |
| `rust llm memory` | ~5 | Only 2 Rust competitors | SMOS most complete |

**Secondary (supporting — higher KD but relevant):**

| Keyword | Est. KD | Strategy |
|---|---|---|
| `AI agent memory` | 65 | Use in H2/H3, don't try to rank #1 |
| `persistent memory AI` | 48 | Use in body copy |
| `self hosted AI memory` | 35 | Dedicated section |
| `local LLM memory` | 28 | Use in self-hosted section |
| `coding agent memory` | 22 | Use in pain-point section |

### 3.2 Title tag (≤60 chars)

```
SMOS — OpenAI-Compatible Semantic Memory Proxy in Rust
```
(54 chars — включает 3 primary keywords: "OpenAI-Compatible", "Semantic Memory Proxy", "Rust")

### 3.3 Meta description (≤155 chars)

```
SMOS gives any AI coding agent persistent memory — without code changes. OpenAI-compatible proxy, self-hosted, single binary, no external database. Rust.
```
(155 chars — включает: "persistent memory", "OpenAI-compatible proxy", "self-hosted", "single binary", "Rust")

### 3.4 URL structure

```
/                    — Landing page (this spec)
/docs                — Redirect → github.com/yurvon-screamo/smos#readme
/github              — Redirect → github.com/yurvon-screamo/smos
/npm                 — Redirect → npmjs.com/package/@yurvon_screamo/smos
/crates              — Redirect → crates.io/crates/smos
```

Все редиректы — 301 permanent.

### 3.5 Semantic HTML structure

```html
<h1>SMOS — Semantic Memory Operating System</h1>     <!-- 1 per page -->

<h2>Quick start</h2>
<h2>Why SMOS</h2>
<h3>Memory is part of the API, not a tool</h3>       <!-- keyword: "memory proxy not tool" -->
<h3>No external database</h3>
<h3>Contradictions detected, not overwritten</h3>
<h3>Multi-persona isolation</h3>
<h3>Runs on any laptop</h3>

<h2>How it works</h2>
<h2>How SMOS compares</h2>                           <!-- keyword: "mem0 alternative" -->
<h2>FAQ</h2>
<h2>Get started</h2>
```

**Правило:** ровно один `<h1>`. Каждый `<h2>` — section header. `<h3>` только внутри секций.

### 3.6 Schema.org structured data

#### SoftwareApplication schema (в `<head>` как JSON-LD)

```json
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "SMOS — Semantic Memory Operating System",
  "description": "An OpenAI-compatible memory proxy that gives any AI coding agent persistent long-term memory — without code changes, without an MCP server, without a framework.",
  "url": "https://smos.dev",
  "downloadUrl": "https://github.com/yurvon-screamo/smos/releases",
  "applicationCategory": "DeveloperApplication",
  "operatingSystem": "Windows, Linux, macOS",
  "programmingLanguage": "Rust",
  "offers": {
    "@type": "Offer",
    "price": "0",
    "priceCurrency": "USD"
  },
  "license": "https://opensource.org/licenses/MIT",
  "softwareRequirements": "llama-server on PATH, ~5 GB disk for models",
  "featureList": [
    "OpenAI-compatible proxy — no agent code changes",
    "Embedded SurrealDB — no external database",
    "DeBERTa-v3 NLI contradiction detection",
    "Multi-persona memory isolation",
    "Self-hosted — runs on laptop CPU"
  ]
}
```

#### FAQPage schema (если есть FAQ секция)

```json
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "Does SMOS work with cloud LLMs like OpenAI and Claude?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Yes. SMOS is an OpenAI-compatible proxy. Point it at any upstream — local llama.cpp, OpenAI, OpenRouter, vLLM. Run fully local for privacy, or use your existing cloud provider."
      }
    }
  ]
}
```

### 3.7 Open Graph + Twitter Card

```html
<meta property="og:type" content="website" />
<meta property="og:title" content="SMOS — OpenAI-Compatible Semantic Memory Proxy in Rust" />
<meta property="og:description" content="Persistent memory for any AI coding agent. No code changes. No MCP server. No external database." />
<meta property="og:url" content="https://smos.dev" />
<meta property="og:image" content="https://smos.dev/og-image.png" />
<meta property="og:image:width" content="1200" />
<meta property="og:image:height" content="630" />

<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="SMOS — Semantic Memory Operating System" />
<meta name="twitter:description" content="Memory IS the API. Not a tool. OpenAI-compatible proxy for AI coding agents." />
<meta name="twitter:image" content="https://smos.dev/og-image.png" />
```

### 3.8 Дополнительные SEO элементы

```html
<link rel="canonical" href="https://smos.dev" />
<meta name="robots" content="index, follow" />
<meta name="author" content="turbin_y" />

<!-- Performance hints -->
<link rel="preconnect" href="https://github.com" />
<link rel="dns-prefetch" href="https://github.com" />
<link rel="preconnect" href="https://crates.io" />

<!-- Sitemap (для будущего) -->
<link rel="sitemap" type="application/xml" href="/sitemap.xml" />
```

### 3.9 robots.txt

```
User-agent: *
Allow: /
Sitemap: https://smos.dev/sitemap.xml
```

---

## 4. Content — Ready-to-Paste Copy

> Весь контент ниже — final copy, готов к вставке. Не требует редактирования.
> Tone: Builder-Architect 60% / Pragmatic Operator 30% / Precision Educator 10%.
> Calm conviction, NEVER enthusiasm. Concrete numbers. No forbidden words.

### 4.1 Hero Section

**H1:**
```
SMOS — Semantic Memory Operating System
```

**Tagline (под H1):**
```
Memory IS the API. Not a tool.
```

**Subheadline:**
```
An OpenAI-compatible proxy that gives any AI coding agent persistent
long-term memory — without code changes, without an MCP server,
without a framework.
```

**CTA buttons:**
- Primary: `Get Started` → scroll to Quick Start
- Secondary: `GitHub` → https://github.com/yurvon-screamo/smos

**Terminal demo (в hero visual area):**
```bash
$ npm install -g @yurvon_screamo/smos
$ smos init          # downloads ~4 GB of tiny local models
$ smos serve         # starts on http://localhost:8888

# Point Cursor at http://localhost:8888/v1
# Use "bob" as the model name.
# Your assistant now remembers across sessions.
```

**Trust badges (под terminal):**
- MIT License
- Rust 1.96 / Edition 2024
- Self-hosted
- ~5 GB disk, no GPU required

### 4.2 Pain Point Section

**Section label (eyebrow):** `THE PROBLEM`

**H2:**
```
Every new chat starts from scratch
```

**Body:**
```
Open a new chat in Cursor and your assistant starts from scratch.
Switch to Claude Code or opencode and you re-explain why the cache
TTL is 10 seconds, not 60 — your architecture, your conventions,
every decision you already made.

The model is stateless. The tool is replaceable. The memory should
not be.
```

**Visual:** Before/After comparison

| Without SMOS | With SMOS |
|---|---|
| Every session: re-explain architecture | Bob knows your architecture from day one |
| Switch Cursor → Claude: context lost | Switch tools: Bob stays Bob |
| Agent must decide what to save | Every response mined automatically |
| Memory = a notebook the agent keeps | Memory = what actually happened |

### 4.3 Solution Section

**Section label:** `THE SOLUTION`

**H2:**
```
A transparent proxy. Point your base URL at it. Done.
```

**Body:**
```
SMOS sits between your AI client and the upstream LLM. Every response
is mined for facts automatically — the agent does nothing, the agent
forgets nothing. Point any OpenAI-compatible client at SMOS and your
assistant remembers across sessions, across tools, across model swaps.

Works with local llama.cpp, OpenAI, OpenRouter, vLLM — any
OpenAI-compatible upstream. Run fully local for privacy, or point it
at your existing cloud provider.
```

**Visual:** Pipeline diagram (тот же, что в README)

```
Client ──▶ SMOS ──▶ upstream LLM (GPT-4o, Claude, local, …)
              │
              ├── 1. ENRICH    inject relevant facts into the request
              ├── 2. FORWARD   stream response back at full LLM speed
              ├── 3. EXTRACT   mine the response for facts (after delivery)
              └── 4. FINALIZE  DeBERTa NLI resolves merges and conflicts
```

### 4.4 Why SMOS — Differentiators (5 Cards)

**Section label:** `WHY SMOS`

**H2:**
```
Five things SMOS does differently
```

**Card 1: Memory is part of the API, not a tool**
```
Every response is mined for facts automatically. The agent cannot
forget to save, because the agent is not involved in saving.
Extraction runs off the request path — zero added latency.
```

**Card 2: No external database**
```
Embedded SurrealDB (RocksDB + HNSW vector index). No Postgres, no
Neo4j, no Qdrant, no Docker. One binary, one directory.
```

**Card 3: Contradictions detected, not overwritten**
```
A DeBERTa-v3 NLI model evaluates each merge candidate. Both sides
of a contradiction are preserved and surfaced to the LLM — not
silently overwritten.
```

**Card 4: Multi-persona isolation**
```
Bob for Rust, Alice for ML, Charlie for DevOps — each a separate
memory namespace. One SMOS instance, N isolated assistants.
```

**Card 5: Runs on any laptop**
```
Three tiny local models (4 GB total) handle extraction, embeddings,
and reranking on CPU. No GPU, no API keys, no cloud bills. Your
conversations never leave your machine.
```

### 4.5 Quick Start Section

**Section label:** `QUICK START`

**H2:**
```
Running in 3 commands
```

**Code block:**
```bash
npm install -g @yurvon_screamo/smos   # or: cargo binstall smos
smos init                              # one-time: downloads ~4 GB
smos serve                             # starts on http://localhost:8888
```

**Note (под кодом):**
```
Point Cursor, Claude Code, opencode, Cline, or Aider at
http://localhost:8888/v1 and use "bob" as the model name.

One prerequisite: llama-server on your PATH. SMOS uses it to run
three tiny models locally — no GPU, no API keys, no cloud bills.
Prefer cloud? Skip llama-server and configure any OpenAI-compatible
provider.
```

### 4.6 How SMOS Compares — Honest Table

**Section label:** `COMPARISON`

**H2:**
```
How SMOS compares to other memory systems
```

**Table:**

| | SMOS | mem0 | Letta | Zep | Cognee |
|---|---|---|---|---|---|
| **Architecture** | Proxy (transparent) | Tool (agent calls) | Framework (runtime) | Tool + SaaS | Tool (pipeline) |
| **External DB** | None (embedded) | Qdrant / Postgres | Postgres + Redis | Neo4j (mandatory) | Neo4j + Postgres + vector |
| **Code changes needed** | None (change base URL) | Yes (SDK calls) | Yes (adopt runtime) | Yes (SDK calls) | Yes (pipeline API) |
| **Self-hosted** | Fully | Docker | Docker | Partial (SaaS-first) | Complex multi-DB |
| **Multi-agent isolation** | Built-in (personas) | user_id scoping | Per-agent identity | Per-user graphs | Namespaces |
| **Contradiction handling** | NLI detection + preserve | Picks winner | Agent self-edits | Temporal invalidation | None explicit |
| **Language** | Rust | Python | Python | Python | Python |
| **License** | MIT | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 |

**Disclaimer под таблицей:**
```
Star counts and feature sets verified as of June 2026. Each project
has different strengths — this table highlights architectural
differences, not superiority.
```

### 4.7 Self-Hosted Sovereignty Section

**Section label:** `SELF-HOSTED`

**H2:**
```
No API keys. No cloud bills. No data leaving your machine.
```

**Body:**
```
SMOS runs three tiny local models — a 4B extraction LLM, an
embedding model, and a reranker. The largest is 4B parameters.
These run on a laptop CPU with integrated graphics.

No GPU required. No OpenAI API key. No monthly subscription.
Your code, your conversations, your decisions — all stay local.

Prefer cloud? SMOS works with OpenAI, OpenRouter, vLLM, and any
OpenAI-compatible provider. The choice is yours.
```

### 4.8 Academic Foundation Section

**Section label:** `RESEARCH`

**H2:**
```
Built on peer-reviewed research
```

**Body:**
```
SMOS is grounded in two recent papers on AI agent memory:

MemoryOS (EMNLP 2025 Oral, Kang et al.) — hierarchical memory
management for AI agents. SMOS adopts a similar lifecycle
(pending → accepted → conflict-flagged) driven by NLI rather
than hand-tuned heuristics.

The Price of Meaning (Ray Barman et al., 2026) — proves that
vector-only retrieval mathematically degrades through semantic
interference. External verification is necessary. SMOS's
DeBERTa-v3 NLI layer is that verification.
```

**Links:**
- [MemoryOS paper →](https://arxiv.org/abs/2506.06326)
- [The Price of Meaning →](https://arxiv.org/abs/2603.27116)

### 4.9 FAQ Section

**Section label:** `FAQ`

**H2:**
```
Common questions
```

**Q: Does SMOS work with cloud LLMs like OpenAI and Claude?**

Yes. SMOS is an OpenAI-compatible proxy. Point it at any upstream — local llama.cpp, OpenAI, OpenRouter, vLLM. Run fully local for privacy, or use your existing cloud provider. The local models (extraction, embedding, reranker) are tiny and run on CPU; your chat model can still be GPT-4o or Claude.

**Q: Do I need a GPU?**

No. The three local models (extraction, embeddings, reranking) are tiny — the largest is 4B parameters. They run on a laptop CPU with integrated graphics. Tested on Intel integrated graphics without issues. If you have a GPU, SMOS detects it automatically (CUDA, DirectML, Metal) and uses it for the DeBERTa NLI model — but it's optional, not required.

**Q: How is this different from mem0 or Letta?**

mem0 and Letta are tools the agent must decide to call — save here, retrieve there. The agent can forget, misjudge, or skip the call. SMOS is a transparent proxy: every response is mined for facts automatically. The agent is not involved in saving. Additionally, SMOS requires zero external databases (no Postgres, no Neo4j, no Docker) — it's a single binary with an embedded database.

**Q: What about MCP servers?**

SMOS doesn't need one. Because it's a proxy (not a tool), there's no MCP server to configure. Point your AI client's base URL at SMOS and you're done.

**Q: Is it production-ready?**

SMOS is a young project (v0.1.7). The architecture is production-oriented — hexagonal DDD, compile-enforced layering, fail-open contract, 665+ tests. But it has not been battle-tested at scale yet. See Known Limitations in the README.

**Q: Can I use it with multiple agents?**

Yes. Each "person" in SMOS is a memory namespace. Create Bob for Rust, Alice for ML, Charlie for DevOps — each isolated. One SMOS instance serves N assistants.

**Q: What about my data privacy?**

All inference (extraction, embedding, reranking, NLI) runs locally via llama.cpp. Your conversations never leave your machine. If you use a cloud provider as your upstream (OpenAI, Claude), those requests go directly from SMOS to the provider — SMOS does not relay them through any third party.

### 4.10 Final CTA Section

**H2:**
```
Give your AI agent a memory.
```

**Body:**
```
Three commands. Five minutes. Your assistant remembers.
```

**CTA:**
- Primary: `npm install -g @yurvon_screamo/smos` (copy-to-clipboard button)
- Secondary: `Read the docs →` → https://github.com/yurvon-screamo/smos
- Tertiary: `Star on GitHub →` → https://github.com/yurvon-screamo/smos

### 4.11 Footer

```
SMOS — Semantic Memory Operating System
MIT License · Built in Rust · Self-hosted by design

GitHub · npm · crates.io · Docs · Architecture

© 2026 turbin_y
```

---

## 5. Design System

### 5.1 Aesthetic direction

**Reference sites (design language benchmarks):**
- [Linear.app](https://linear.app) — dark, clean, code-forward
- [Railway.app](https://railway.app) — gradient accents, developer-focused
- [Cursor.com](https://cursor.com) — minimal, dark, code blocks
- [Astro.build](https://astro.build) — excellent use of space and typography

**Mood:** Dark, technical, precise, confident. NOT playful. NOT corporate. A tool built by an engineer for engineers.

### 5.2 Color palette

```css
:root {
  /* Background — near-black, easy on developer eyes */
  --bg-primary: #0a0a0a;
  --bg-secondary: #111111;
  --bg-surface: #1a1a1a;
  --bg-elevated: #222222;

  /* Text */
  --text-primary: #e8e8e8;
  --text-secondary: #a0a0a0;
  --text-muted: #666666;

  /* Accent — Rust-inspired orange */
  --accent-primary: #ce422b;       /* Rust orange */
  --accent-hover: #e0533a;
  --accent-subtle: rgba(206, 66, 43, 0.1);

  /* Semantic */
  --success: #22c55e;
  --warning: #eab308;
  --error: #ef4444;

  /* Code syntax (for terminal blocks) */
  --code-bg: #0d0d0d;
  --code-text: #e8e8e8;
  --code-comment: #666666;
  --code-string: #22c55e;
  --code-keyword: #ce422b;
  --code-number: #eab308;

  /* Borders */
  --border-subtle: rgba(255, 255, 255, 0.06);
  --border-default: rgba(255, 255, 255, 0.1);
}
```

### 5.3 Typography

```css
:root {
  /* Headings — modern, geometric */
  --font-heading: 'Inter', -apple-system, system-ui, sans-serif;

  /* Body — same family, lighter weight */
  --font-body: 'Inter', -apple-system, system-ui, sans-serif;

  /* Code — monospace, ligatures */
  --font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
}

/* Type scale (desktop) */
h1 { font-size: clamp(2.5rem, 5vw, 3.5rem); font-weight: 700; line-height: 1.1; }
h2 { font-size: clamp(1.75rem, 3vw, 2.25rem); font-weight: 600; line-height: 1.2; }
h3 { font-size: 1.25rem; font-weight: 600; line-height: 1.3; }
body { font-size: 1rem; font-weight: 400; line-height: 1.6; }
code { font-size: 0.9rem; font-family: var(--font-mono); }
```

**Font loading:** Use `font-display: swap` for all web fonts. Prefer system fonts as fallback.

### 5.4 Layout grid

```
Desktop (≥1024px):
  Content max-width: 1152px (72rem)
  Padding: 24px (1.5rem) horizontal
  Hero: full-width background, content max-width

Tablet (768px–1023px):
  Content max-width: 100%
  Padding: 32px horizontal

Mobile (<768px):
  Content max-width: 100%
  Padding: 20px horizontal
  Terminal blocks: horizontal scroll
  Comparison table: horizontal scroll
```

### 5.5 Section spacing

```css
section { padding-block: clamp(4rem, 8vw, 7rem); }
.hero { min-height: 100vh; display: flex; align-items: center; }
```

### 5.6 Component specifications

#### Terminal / code block
```css
.terminal {
  background: var(--code-bg);
  border: 1px solid var(--border-default);
  border-radius: 8px;
  padding: 20px 24px;
  font-family: var(--font-mono);
  font-size: 0.9rem;
  line-height: 1.7;
  overflow-x: auto;
}

.terminal-header {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-subtle);
}

.terminal-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  /* red, yellow, green — standard terminal dots */
}
```

#### Cards (differentiators)
```css
.card {
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: 12px;
  padding: 28px;
  transition: border-color 0.2s, transform 0.2s;
}

.card:hover {
  border-color: var(--accent-primary);
  transform: translateY(-2px);
}
```

#### CTA buttons
```css
.btn-primary {
  background: var(--accent-primary);
  color: white;
  padding: 12px 28px;
  border-radius: 8px;
  font-weight: 600;
  font-size: 1rem;
  transition: background 0.2s;
}

.btn-primary:hover {
  background: var(--accent-hover);
}

.btn-secondary {
  background: transparent;
  color: var(--text-primary);
  border: 1px solid var(--border-default);
  padding: 12px 28px;
  border-radius: 8px;
  font-weight: 500;
}

.btn-secondary:hover {
  border-color: var(--text-primary);
}
```

#### Comparison table
```css
.comparison-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

.comparison-table th {
  text-align: left;
  padding: 16px;
  background: var(--bg-surface);
  border-bottom: 2px solid var(--border-default);
  font-weight: 600;
}

.comparison-table td {
  padding: 14px 16px;
  border-bottom: 1px solid var(--border-subtle);
}

.comparison-table tr:hover td {
  background: var(--bg-secondary);
}

/* SMOS column highlight */
.comparison-table th:nth-child(2),
.comparison-table td:nth-child(2) {
  background: var(--accent-subtle);
}
```

---

## 6. Technical Requirements

### 6.1 Recommended stack

**Primary recommendation: [Astro](https://astro.build)**

Почему:
- Purpose-built для content-heavy sites с SEO приоритетом
- Zero JS by default — islands of interactivity только где нужно
- Static HTML output → идеальный Lighthouse score
- Markdown/MDX support — контент можно писать в MDX
- Built-in SEO features (sitemap, robots.txt, image optimization)

**Alternatives:**
- **Plain HTML/CSS + minimal JS** — simplest, fastest, но manual SEO
- **Next.js (static export)** — familiar для React devs, но overkill для single page
- **SvelteKit** — excellent DX, static adapter available

> Решение стека — за заказчиком. Контент и SEO requirements из этого ТЗ
> не зависят от стека.

### 6.2 Performance requirements

| Metric | Target | How |
|---|---|---|
| Total page weight | < 300 KB (gzipped) | Minimal JS, no heavy frameworks |
| Images | WebP/AVIF, lazy-loaded | `<img loading="lazy">` ниже fold |
| Fonts | `font-display: swap`, preload critical | Only Inter + JetBrains Mono |
| CSS | Inline critical CSS, defer rest | < 20 KB critical |
| JS | < 30 KB gzipped total | Only copy-to-clipboard + scroll-smooth |
| Hosting | CDN with edge caching | Cloudflare Pages / Vercel / Netlify |

### 6.3 JavaScript requirements

Минимальный JS. Только:
1. **Copy-to-clipboard** на code blocks и CTA commands
2. **Smooth scroll** на anchor links
3. **Mobile menu toggle** (если есть nav)

Никакого:
- React / Vue / Svelte hydration (если не Astro islands)
- Analytics SDK (использовать lightweight: Plausible / Umami / Cloudflare Web Analytics)
- Chat widgets
- Cookie banners (нет cookies, нет tracking, нет GDPR problem)

### 6.4 Analytics

**Recommended: [Plausible](https://plausible.io)** или [Cloudflare Web Analytics](https://www.cloudflare.com/web-analytics/)

Почему:
- Privacy-friendly (no cookies, no GDPR consent needed)
- Lightweight script (< 1 KB)
- Developer audience ненавидит Google Analytics

**Events to track:**
- `pageview` — default
- `cta_click` — Get Started / GitHub / npm install
- `code_copy` — copy-to-clipboard events
- `section_scroll` — which sections users reach (optional)

### 6.5 Domain

**Recommended:** `smos.dev` (ideal, short, memorable)

**Alternatives:**
- `smos.yurvon.dev` (subdomain, free)
- `smos-memory.dev`
- `semantic-memory.dev`

Проверить availability через [Domainr](https://domainr.com) или [Namecheap](https://namecheap.com).

### 6.6 Hosting

**Recommended:** Cloudflare Pages (free, fast, edge-cached)

**Alternatives:**
- Vercel (free tier, excellent DX)
- Netlify (free tier)
- GitHub Pages (free, but slower)

### 6.7 SSL/TLS

Automatic via hosting provider (Let's Encrypt). Force HTTPS redirect.

---

## 7. Asset List

### 7.1 OG Image (social sharing)

**Size:** 1200×630 px
**Content:**
- Dark background (#0a0a0a)
- SMOS logo/wordmark
- Tagline: "Memory IS the API. Not a tool."
- Terminal snippet (3 commands)
- Rust orange accent line

### 7.2 Favicon

**Sizes:** 16×16, 32×32, 180×180 (Apple touch icon), 512×512 (PWA)
**Design:** Minimal "S" mark or terminal icon with Rust orange accent

### 7.3 Terminal demo animation (optional, Phase 2)

**Format:** GIF or WebM, < 2 MB
**Content:** Animated terminal showing `smos init` → `smos serve` → curl health check
**Duration:** 5-8 seconds, loop

### 7.4 Comparison table icons (optional)

Small checkmark (✓) / cross (✗) icons for the comparison table feature cells.
**Style:** Monochrome, 16×16, consistent with dark theme.

### 7.5 Logo / Wordmark (if not exists)

**Style:** Wordmark "SMOS" in Inter Bold, with Rust orange dot or accent.
**Variations:**
- Horizontal (logo + tagline) — for hero
- Stacked (logo above, tagline below) — for footer
- Icon only — for favicon

---

## 8. Anti-Patterns — What NOT to Do

### 8.1 Forbidden words (AI-slop markers — instant downvote signal)

NEVER use these anywhere on the page:
```
delve, tapestry, harness, unlock, revolutionize, disrupt, leverage,
supercharge, cutting-edge, state-of-the-art, game-changer, next-gen,
seamless, empower, elevate, foster, fuel, robust, revolutionary,
innovative, comprehensive, transformative
```

Russian equivalents (если будет RU-версия):
```
инновационный, нет аналогов, мирового уровня, прорывной, революционный,
уникальный, масштабный, комплексное решение, передовой, бесшовный
```

### 8.2 Saturated marketing angles (DO NOT USE)

| Angle | Why avoid |
|---|---|
| "Stop re-explaining yourself" | Every memory product says this |
| "LLMs have amnesia" | Over-saturated, Cognee got 9 points on HN with this |
| "Memory layer for AI agents" | mem0 owns this phrase |
| "SOTA on LOCOMO" | Benchmark wars destroyed credibility |
| "AI chatbot memory" | Pulls wrong audience |
| "Built for production" (without evidence) | Everyone says this |

### 8.3 Design anti-patterns

- NO hero stock photos (use code blocks, not smiling developers)
- NO gradient text (instant AI-generated look)
- NO auto-playing video with sound
- NO exit-intent popups
- NO fake countdown timers
- NO artificial scarcity ("Only 3 spots left!")
- NO chat widget in bottom-right corner
- NO cookie consent banner (no cookies = no consent needed)
- NO heavy animations (parallax, scroll-jacking)

### 8.4 Technical anti-patterns

- NO client-side rendering for SEO content (SSR or static)
- NO render-blocking JS
- NO external font requests without `font-display: swap`
- NO uncompressed images
- NO `console.log` in production
- NO tracking pixels beyond analytics

---

## 9. Section Order (Page Layout)

```
┌─────────────────────────────────────────────┐
│ NAV BAR (sticky, minimal)                    │
│ SMOS | Docs · GitHub · npm                   │
├─────────────────────────────────────────────┤
│                                              │
│ HERO (100vh)                                 │
│ H1: SMOS — Semantic Memory Operating System  │
│ Tagline: Memory IS the API. Not a tool.     │
│ Subheadline                                  │
│ [Get Started] [GitHub]                       │
│ Terminal demo (3 commands)                   │
│ Trust badges                                 │
│                                              │
├─────────────────────────────────────────────┤
│ PAIN POINT                                   │
│ "Every new chat starts from scratch"         │
│ Before/After comparison                      │
├─────────────────────────────────────────────┤
│ SOLUTION                                     │
│ "A transparent proxy. Point your base URL."  │
│ Pipeline diagram                             │
├─────────────────────────────────────────────┤
│ QUICK START                                  │
│ 3 commands + note                            │
│ [Copy install command button]                │
├─────────────────────────────────────────────┤
│ WHY SMOS (5 cards)                           │
│ Memory is API | No DB | Contradictions |     │
│ Multi-persona | Runs on laptop               │
├─────────────────────────────────────────────┤
│ COMPARISON TABLE                             │
│ SMOS vs mem0 vs Letta vs Zep vs Cognee       │
├─────────────────────────────────────────────┤
│ SELF-HOSTED                                  │
│ "No API keys. No cloud bills."               │
├─────────────────────────────────────────────┤
│ RESEARCH                                     │
│ MemoryOS + Price of Meaning papers           │
├─────────────────────────────────────────────┤
│ FAQ (6 questions)                            │
├─────────────────────────────────────────────┤
│ FINAL CTA                                    │
│ "Give your AI agent a memory."               │
│ [npm install command] [GitHub] [Docs]        │
├─────────────────────────────────────────────┤
│ FOOTER                                       │
│ Links · License · © 2026                     │
└─────────────────────────────────────────────┘
```

---

## 10. Acceptance Criteria

Лендинг считается готовым когда:

- [ ] Все секции из §4 реализованы с готовым copy
- [ ] Title tag, meta description, schema.org, OG tags из §3 установлены
- [ ] Lighthouse Performance ≥ 95 (mobile)
- [ ] Lighthouse SEO = 100
- [ ] LCP < 1.5s на 4G connection
- [ ] CLS < 0.1
- [ ] INP < 200ms
- [ ] Mobile responsive (320px–1920px tested)
- [ ] Terminal blocks имеют copy-to-clipboard
- [ ] Все ссылки рабочие (GitHub, npm, crates.io, papers)
- [ ] SSL/TLS активен, HTTPS redirect
- [ ] robots.txt + sitemap.xml (если multi-page)
- [ ] Analytics установлен (Plausible/Cloudflare)
- [ ] 0 forbidden words из §8.1
- [ ] 0 saturated angles из §8.2
- [ ] OG image (1200×630) загружена
- [ ] Favicon set (16/32/180/512)

---

## Appendix A: Keyword Placement Map

| Section | Primary keyword placed | Where exactly |
|---|---|---|
| `<title>` | "OpenAI-Compatible Semantic Memory Proxy in Rust" | Title tag |
| Meta description | "OpenAI-compatible proxy", "self-hosted", "single binary" | Meta desc |
| H1 | "Semantic Memory Operating System" | Hero heading |
| Hero subheadline | "OpenAI-compatible proxy", "persistent long-term memory" | Hero body |
| Pain point | "starts from scratch", "re-explain" | Pain body |
| Solution | "transparent proxy", "OpenAI-compatible" | Solution body |
| Why SMOS Card 1 | "memory is part of the API, not a tool" | Card heading |
| Why SMOS Card 2 | "No Postgres, no Neo4j, no Qdrant" | Card body |
| Why SMOS Card 5 | "runs on any laptop", "no cloud bills" | Card body |
| Quick Start | "npm install", "smos init", "smos serve" | Code block |
| Comparison | "mem0 alternative", "self-hosted" | Table + heading |
| FAQ | "cloud LLMs", "MCP server", "mem0", "Letta" | Questions |
| Final CTA | "memory", "npm install" | CTA body |

---

## Appendix B: Brand Voice Reminder

- **Tone:** Measured, technically precise, personal-experience grounded
- **Conviction:** Calm, NEVER enthusiastic
- **Numbers:** Concrete in every claim (4 GB, 4B params, 665+ tests, 0.1.7)
- **First person:** "SMOS runs..." not "SMOS empowers you to..."
- **Limitations:** State them honestly — they build credibility
- **Comparisons:** Factual, not aggressive. "Different architecture" not "better than"
- **CTAs:** Direct, not pushy. "Get Started" not "Supercharge Your AI Now"

---

*End of specification. Developer should be able to build the entire landing page from this document without additional clarification.*
