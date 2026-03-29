# Architecture Overview

## 1. Purpose of This Document

This is the architectural source of truth for the **semantic-index** SKILL project.
It describes the system as it exists today — components, data flows, stability zones.
It does NOT define coding rules (see [Section 8](#8-ai-coding-rules-and-behavioral-contracts)).

**Audience:** AI assistants, new contributors, future-you after a long break.

---

## 2. High-Level System Overview

**What:** A portable SKILL for embedding-based semantic search of codebases and documentation.
**Pattern:** CLI scripts + file-based storage. No daemon, no server, no Docker.
**Primary interface:** AI reads `SKILL.md`, runs Python scripts, parses JSON output.
**Alternative:** MCP server (optional transport, same underlying code).

### Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.10+ | All scripts and library code |
| AST parsing | Tree-sitter | Code chunking (10 languages) |
| XML parsing | xml.etree.ElementTree | DITA documentation chunking |
| Office extraction | PyMuPDF, python-docx, python-pptx | PDF/DOCX/PPTX text extraction |
| Tokenization | tiktoken | Token counting for chunk sizing |
| Vector store | LanceDB (PyArrow) | Embedding storage and similarity search |
| Keyword index | Custom BM25 | Keyword-based retrieval |
| Embeddings (remote) | OpenRouter REST API | Default provider (BAAI/bge-m3) |
| Embeddings (local) | sentence-transformers | Optional zero-cost offline provider |
| Reranking | CrossEncoder (sentence-transformers) | Optional cross-encoder reranking |
| MCP transport | FastMCP | Optional MCP server bridge |

### Architecture Pattern

```
┌──────────────────────────────────────────────────┐
│              AI Assistant                         │
│        (reads SKILL.md, runs scripts)            │
└─────┬──────────────┬──────────────┬──────────────┘
  build_index.py  semantic_search.py  index_status.py
      │              │              │
┌─────┴──────────────┴──────────────┴──────┐
│              scripts/lib/                │
│  config─hasher─chunker─embedder          │
│  store─bm25─fusion─reranker              │
└──────────────────┬───────────────────────┘
          .index/ (per-project)
    config│manifest│cache│lancedb
```

---

## 3. Repository Structure

```
semantic-index/                     # SKILL root (installed to ~/.kiro/skills/)
├── SKILL.md                        # AI-facing instructions (when/how to use)
├── assets/default-config.json      # Default config template
├── references/                     # On-demand docs (languages, models, MCP)
├── scripts/
│   ├── setup.sh                    # One-command venv + deps setup
│   ├── requirements*.txt           # Core + optional dep groups
│   ├── build_index.py              # CLI: build/rebuild index
│   ├── semantic_search.py          # CLI: search by meaning
│   ├── index_status.py             # CLI: index health check
│   ├── migrate_config.py           # CLI: config schema migration
│   ├── mcp_server.py               # Optional MCP server transport
│   └── lib/                        # Python package
│       ├── models.py config.py hasher.py   # Core: data, config, change detection
│       ├── chunker.py              # Dispatch + fallback
│       ├── chunkers/               # code.py markdown.py dita.py office.py common.py
│       ├── embedder.py             # Provider ABC, factory, cache
│       ├── providers/              # openrouter.py huggingface.py
│       ├── store.py bm25.py        # Storage: vector + keyword
│       ├── fusion.py reranker.py   # Search: RRF merge + optional reranking
│       └── constants.py            # Shared constants
└── tests/                          # pytest suite (10 test files)
```

**Critical paths:** Entry points are `build_index.py`, `semantic_search.py`, `index_status.py`.
Config template: `assets/default-config.json`. Per-project data: `<project>/.index/` (gitignored).

---

## 4. Core Components

### 4.1 Chunking Pipeline (`lib/chunkers/`)

Five strategies dispatched by file extension via `chunker.py`:

| Strategy | Module | Formats | Method |
|----------|--------|---------|--------|
| Code | `code.py` | .py .js .ts .go .rs .java .c .cpp .rb .php | Tree-sitter AST → functions, classes, methods |
| Markdown | `markdown.py` | .md .mdx .rst | Header-based section splitting |
| DITA | `dita.py` | .dita .ditamap | XML topic-based parsing |
| Office | `office.py` | .pdf .docx .pptx | Page/heading/slide-based (binary I/O) |
| Fallback | `chunker.py` | everything else | Blank-line splitting |

### 4.2 Embedding System (`lib/embedder.py` + `lib/providers/`)

Provider pattern with lazy imports. Factory selects provider from config.
`openrouter.py` — REST API with batching + retry. `huggingface.py` — local inference, auto device detection.
Embedding cache (`embedding_cache.json`) sits above the provider layer.
Both providers produce identical vectors for the same model — indexes are cross-compatible.

### 4.3 Search & Retrieval

`store.py` (LanceDB cosine similarity, 2x over-fetch) → `bm25.py` (keyword index) →
`fusion.py` (RRF merge) → `reranker.py` (optional cross-encoder, BAAI/bge-reranker-v2-m3).

### 4.4 External Integrations

| Integration | Module | Required |
|-------------|--------|----------|
| OpenRouter API | `providers/openrouter.py` | Default (needs API key) |
| HuggingFace Hub | `providers/huggingface.py` | Optional (model download on first use) |
| MCP protocol | `mcp_server.py` | Optional alternative transport |

---

## 5. Data Flow & Runtime Model

### Indexing Flow

```
project files → hasher.py (SHA-256 diff vs manifest)
                    │
            ┌───────┴────────┐
            │ changed/new    │ unchanged → skip
            ▼                │
       chunker.py            │
  (dispatch by extension)    │
            ▼                │
     embedder.py             │
  (cache check → provider)   │
            ▼                │
  store.py + bm25.py         │
  (vector + keyword index)   │
            ▼                │
     manifest.json update ◄──┘
```

### Search Flow

```
query → embedder.embed_query()
             │
     ┌───────┴───────┐
     ▼               ▼
 store.py         bm25.py          (parallel retrieval)
     │               │
     └───────┬───────┘
             ▼
        fusion.py (RRF)            (merge)
             ▼ (optional)
       reranker.py (cross-encoder) (re-score)
             ▼
      ranked results (JSON)
```

### Configuration Loading Hierarchy

Defaults (`assets/default-config.json`) → `.index/config.json` (per-project) → env vars (`OPENROUTER_API_KEY`, `SEMANTIC_INDEX_*`) → CLI args (`--top-k`, `--mode`, etc.)


---

## 6. Configuration & Environment Assumptions

### Per-Project Index Directory

```
<project-root>/.index/
├── config.json            # User config (created on first run)
├── manifest.json          # SHA-256 file hash manifest
├── embedding_cache.json   # Content hash → vector cache
├── bm25_index.json        # BM25 keyword index
└── lancedb/chunks.lance/  # LanceDB vector store (Arrow format)
```

### Environment Variables

| Variable | Overrides | Required |
|----------|-----------|----------|
| `OPENROUTER_API_KEY` | `embedding.api_key` | Yes (openrouter provider) |
| `SEMANTIC_INDEX_PROVIDER` | `embedding.provider` | No |
| `SEMANTIC_INDEX_MODEL` | `embedding.model` | No |
| `SEMANTIC_INDEX_DIMENSIONS` | `embedding.dimensions` | No |
| `HF_HUB_CACHE` | HuggingFace model cache dir | No |

### Deployment

- Installed to `~/.kiro/skills/semantic-index/` (read-only during use)
- Each project gets its own `.index/` (add to `.gitignore`, fully rebuildable)
- No background processes — all operations are explicit CLI invocations
- Optional deps via `setup.sh` flags: `--with-huggingface`, `--with-office`, `--with-mcp`

---

## 7. Stability Zones

### Module Stability Map

| Zone | Component | Notes |
|------|-----------|-------|
| ✅ Stable | `models.py`, `config.py`, `hasher.py` | Core data model and config. Everything depends on these. |
| ✅ Stable | `chunkers/code.py`, `chunkers/markdown.py` | Battle-tested across real projects. |
| ✅ Stable | `chunker.py` (dispatch) | Extend by adding new chunkers, don't restructure. |
| ✅ Stable | `store.py`, `embedder.py`, `providers/openrouter.py` | Stable interfaces. Retry, batching, caching all done. |
| ✅ Stable | `bm25.py`, `fusion.py` | Hybrid search pipeline. Tune via config, not code. |
| ✅ Stable | CLI scripts (build, search, status) | JSON output format is a contract — don't break it. |
| 🔄 Semi-Stable | `providers/huggingface.py`, `reranker.py` | Working. May evolve with new models/backends. |
| 🔄 Semi-Stable | `chunkers/dita.py` | Functional for standard DITA. Exotic specializations may need updates. |
| 🔄 Semi-Stable | `mcp_server.py` | Functional bridge. May evolve with MCP protocol updates. |
| ⚠️ Experimental | `chunkers/office.py` | PDF/DOCX/PPTX working but edge cases remain (scanned PDFs, complex tables). |
| ⚠️ Experimental | `migrate_config.py` | Config migration. Needs more real-world testing. |
| 🔮 Planned | Parallel chunking (`lib/parallel.py`) | Phase 8 — not yet implemented. |
| 🔮 Planned | Additional providers (Ollama, OpenAI) | Phase 6 — not yet implemented. |

### What NOT to Change

- **CLI JSON output format** — AI assistants parse this; breaking changes break all consumers
- **`EmbeddingProvider` ABC** — both providers implement it; changes require updating both
- **`.index/` directory structure** — existing indexes must remain readable after updates
- **`chunk_file()` signature** — called by `build_index.py` and `mcp_server.py`
- **Config field names** — use `migrate_config.py` for schema evolution

---

## 8. AI Coding Rules and Behavioral Contracts

**This document does NOT define coding rules.**

All coding standards, formatting rules, and stack-specific practices live in
dedicated `AI*.md` files. This section lists them and explains precedence.

### Authoritative AI Rule Files

| File | Scope | Key Topics |
|------|-------|------------|
| `AI.md` | Global Python conventions | PEP8, type hints, docstrings, error handling, imports, testing |
| `AI_SKILL.md` | SKILL-specific conventions | CLI JSON output, stderr logging, exit codes, Tree-sitter, LanceDB, config patterns, 300-line module limit |
| `CLAUDE.md` | AI behavior contract | Rule file discovery order, "propose before coding" mandate |

### Rule Precedence (Highest → Lowest)

1. **User's explicit instruction** in the current conversation
2. **Stack-specific rules** — `AI_SKILL.md` (SKILL conventions)
3. **Global rules** — `AI.md` (Python conventions)
4. **This document** — `ARCHITECTURE.md` (architectural constraints only)
5. **Implicit conventions** inferred from existing codebase

### Conflict Resolution

When rules conflict: **STOP → IDENTIFY → ASK → WAIT**.
Do not guess. Present the conflict with trade-offs and wait for explicit decision.

### Key Architectural Decisions to Preserve

- **SKILL-first**: CLI scripts are primary. MCP is optional transport.
- **File-based storage**: No external databases, no servers. `.index/` is self-contained.
- **Provider abstraction**: Embedding providers interchangeable via config. Don't couple to one.
- **Incremental by default**: SHA-256 manifest. Never re-embed unchanged files.
- **Lazy imports**: Optional deps (torch, PyMuPDF, mcp) imported only when needed.
- **JSON stdout / logs stderr**: All CLI output follows this contract.

---

## 9. Quick Start for AI Assistants

### Pre-Flight Checklist

Before making any changes:

1. Read `CLAUDE.md` — behavioral contract
2. Read `AI.md` + `AI_SKILL.md` — coding rules (especially: 300-line module limit, JSON stdout, exit codes)
3. Check `docs/chats/` — previous implementation context and decisions
4. Check Section 7 above — stability zones tell you what's safe to touch
5. **Propose your solution and wait for approval before writing code**

### Where to Find Things

| Need | Location |
|------|----------|
| How the AI should use the skill | `semantic-index/SKILL.md` |
| Full implementation spec & phases | `IMPLEMENTATION_PLAN.md` |
| Default config values | `semantic-index/assets/default-config.json` |
| Supported languages | `semantic-index/references/supported-languages.md` |
| Embedding model comparison | `semantic-index/references/embedding-models.md` |
| MCP server setup | `semantic-index/references/mcp-server.md` |
| Previous conversations | `docs/chats/` |
| Python coding rules | `AI.md` |
| SKILL coding rules | `AI_SKILL.md` |
| Tests | `semantic-index/tests/` |
