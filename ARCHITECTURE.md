# Semantic Index Skill — Architecture & Implementation Plan

> A portable SKILL for embedding-based indexing and semantic search of codebases and documentation.
> Designed for Claude Code, Cowork, and any SKILL-compatible AI tool.

---

## Table of Contents

1. [Vision & Goals](#1-vision--goals)
2. [SKILL Structure](#2-skill-structure)
3. [SKILL.md Draft](#3-skillmd-draft)
4. [Python Module Architecture](#4-python-module-architecture)
5. [Configuration Schema](#5-configuration-schema)
6. [Chunking Strategy](#6-chunking-strategy)
7. [Embedding Pipeline](#7-embedding-pipeline)
8. [Vector Store & Search](#8-vector-store--search)
9. [File Format Specifications](#9-file-format-specifications)
10. [Implementation Plan](#10-implementation-plan)
11. [Reference Script Specs](#11-reference-script-specs)
12. [Future Roadmap](#12-future-roadmap)
13. [Appendix: Prior Art Comparison](#13-appendix-prior-art-comparison)

---

## 1. Vision & Goals

### What This Is

A SKILL that gives any AI assistant the ability to create and query a semantic index of a project's codebase and documentation. The AI reads the SKILL instructions, runs the bundled Python scripts, and gets back highly relevant code/doc snippets for any natural language query — far better than grep/glob for conceptual searches like "where is authentication handled?" or "how does the payment flow work?".

### Design Principles

- **Portable**: Works anywhere SKILLs work — Claude Code, Cowork, future tools. No daemon, no server, no Docker.
- **Explicit**: User triggers indexing manually. No background watchers, no surprise CPU usage.
- **Local-first index**: The `.index/` folder lives in the project. It's gitignoreable, rebuildable, zero infrastructure.
- **Provider-flexible**: Uses OpenRouter API for embeddings, but the model is configurable. Swapping to local Ollama or OpenAI is a config change, not a code change.
- **Transparent**: The AI understands the entire pipeline because it reads the SKILL and runs the scripts. It can diagnose issues, explain search results, and adjust parameters.

### What This Is NOT

- Not an MCP server (no persistent process, no protocol overhead)
- Not a RAG framework (no retrieval-augmented generation pipeline — just indexing and search)
- Not a replacement for grep/glob (those are better for exact string matches; this is for semantic/conceptual search)

---

## 2. SKILL Structure

```
semantic-index/
├── SKILL.md                          # Main skill instructions (required)
├── scripts/
│   ├── requirements.txt              # Core Python dependencies
│   ├── requirements-huggingface.txt  # Optional: local embedding deps
│   ├── setup.py                      # One-command setup: venv + deps
│   ├── build_index.py                # CLI: build/rebuild the semantic index
│   ├── semantic_search.py            # CLI: search the index by meaning
│   ├── index_status.py               # CLI: show index health & stats
│   └── lib/
│       ├── __init__.py
│       ├── chunker.py                # Chunking dispatch (code / markdown / DITA)
│       ├── chunkers/
│       │   ├── __init__.py
│       │   ├── code.py               # Tree-sitter AST-aware code chunking
│       │   ├── markdown.py           # Header-based markdown chunking
│       │   └── dita.py               # XML-aware DITA topic chunking (Phase 5)
│       ├── embedder.py               # Provider factory + EmbeddingProvider ABC
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── openrouter.py         # OpenRouter REST API provider
│       │   └── huggingface.py        # Local sentence-transformers provider
│       ├── store.py                  # LanceDB vector store wrapper
│       ├── hasher.py                 # File change detection (SHA-256 manifest)
│       ├── config.py                 # Configuration loader
│       └── models.py                 # Data classes (Chunk, SearchResult, etc.)
├── references/
│   ├── supported-languages.md        # Tree-sitter grammar list & file extensions
│   └── embedding-models.md           # Model comparison: dimensions, cost, quality
└── assets/
    └── default-config.json           # Default configuration template
```

### Why This Structure

- **SKILL.md** stays under 500 lines — it tells the AI *when* and *how* to use the scripts, not *how the scripts work internally*
- **scripts/** contains all executable logic — the AI calls these via Bash, never needs to read the internals unless debugging
- **references/** are loaded on demand — the AI reads `embedding-models.md` only when the user asks about model selection
- **lib/** is the actual Python package — clean separation of concerns, each module under 300 lines

---

## 3. SKILL.md Draft

```markdown
---
name: semantic-index
description: >
  Semantic code and documentation indexing with embedding-based search.
  Creates a local .index/ in the project folder for fast conceptual search
  across code and markdown files. Use this skill whenever the user wants to:
  index a codebase or project for semantic search, search code by meaning
  rather than exact text, find where concepts/features/patterns are implemented,
  understand unfamiliar codebases quickly, or asks questions like
  "where is X handled?" or "how does Y work?" about their project.
  Also trigger when the user mentions "semantic search", "index my code",
  "embeddings", "vector search", or "codebase indexing".
---

# Semantic Index

Index code and documentation for meaning-based search using embeddings.

## When To Use This Skill

### Indexing

#### User-initiated
- The user asks to index, re-index, or update the index of their project
- The user opens a new project and wants to set up semantic search

#### AI-initiated
- The project has no .index/ directory and semantic search would be useful
  for the current task — suggest indexing first
- Before searching: run index_status.py to check for stale files. If many files
  are stale (>20% of indexed files), suggest re-indexing before searching
- The user has made significant changes (new modules, large refactors) and
  the AI knows the index is likely outdated

**Important**: Suggest indexing at most once per conversation. If the user
declines or ignores the suggestion, do not bring it up again — fall back
to Grep/Glob/Read and move on with the task. The goal is to be helpful,
not to nag.

### Searching

#### User-initiated
- The user wants to search code by concept, not exact string
- The user asks "where is X handled?" or "how does Y work?"
- The user explicitly asks for semantic/vector/embedding search

#### AI-initiated
- Before implementing a feature: search for similar existing patterns,
  conventions, or related modules to stay consistent with the codebase
- Before fixing a bug: search for related code, similar past fixes, or
  other places where the same pattern appears (the bug might exist elsewhere too)
- When the user mentions "something similar was done before" or "check how
  we handled X" — search for that prior implementation
- When exploring an unfamiliar codebase before making changes: build
  understanding of architecture, naming conventions, and module boundaries
- When the user's task touches a concept that could span multiple files
  and you don't know which ones (e.g., "update the error handling" —
  search for error handling patterns across the project)
- When Grep/Glob would require guessing the exact terminology the codebase
  uses (e.g., the user says "authentication" but the code might use "auth",
  "session", "jwt", "credentials", or "login")

### When NOT to use
- The user knows the exact string to search for (use Grep instead)
- The user wants to find files by name pattern (use Glob instead)
- The total project content comfortably fits within context (e.g., a few
  small files under ~200 lines each). If individual files are large (500+
  lines) or the combined content would exceed ~50K tokens, semantic indexing
  is worthwhile even for just 3-5 files.

## Prerequisites

The skill needs a Python virtual environment with dependencies installed.
On first use, run setup:

```bash
cd <skill-path>/scripts
bash setup.sh
```

This creates a `.venv` in the scripts directory and installs all dependencies.
It only needs to run once per machine.

Embedding provider setup depends on the `embedding.provider` field in
`.index/config.json` (defaults to `"openrouter"`):

- **openrouter**: Requires an API key. Check `OPENROUTER_API_KEY` env var,
  then `config.embedding.api_key`. If neither exists, ask the user.
- **huggingface**: No API key needed. On first run, the model is downloaded
  to `~/.cache/huggingface/hub` (~274MB for Nomic). Subsequent runs load
  from cache. Works fully offline after first download.

If no `.index/config.json` exists yet, the scripts create one on first run.
The provider choice is purely a configuration concern — indexing and search
commands work identically regardless of provider.

## Core Commands

### Indexing

To index the current project:

```bash
<skill-path>/scripts/.venv/bin/python <skill-path>/scripts/build_index.py \
  --project-dir <project-root> \
  [--config <path-to-config.json>]
```

What this does:
1. Scans the project for supported files (code + markdown)
2. Respects .gitignore and .indexignore patterns
3. Computes SHA-256 hashes to detect changed files
4. Chunks files using AST-aware splitting (code) or header-based splitting (markdown)
5. Embeds chunks via the configured provider (OpenRouter API or local HuggingFace)
6. Stores embeddings in `.index/` (LanceDB format)
7. Saves file manifest for incremental re-indexing

Output: prints summary of files indexed, chunks created, time taken.

On re-run, only changed/new files are re-indexed (incremental).

### Searching

To search the index:

```bash
<skill-path>/scripts/.venv/bin/python <skill-path>/scripts/semantic_search.py \
  --project-dir <project-root> \
  --query "your natural language query" \
  [--top-k 10] \
  [--threshold 0.3]
```

Returns ranked results with:
- File path (relative to project root)
- Line range
- Relevance score (0.0–1.0)
- Chunk content (the actual code/text)
- Chunk type (function, class, module-level, markdown-section, etc.)

### Status

To check index health:

```bash
<skill-path>/scripts/.venv/bin/python <skill-path>/scripts/index_status.py \
  --project-dir <project-root>
```

Shows: total files indexed, total chunks, last index time, stale files count,
embedding model used, index size on disk.

## Search Strategy

When helping users explore a codebase, use a layered approach:

1. **First**: Use semantic search via this skill for conceptual queries
2. **Then**: Use Grep/Glob to narrow down or verify specific findings
3. **Finally**: Use Read to examine the actual files in detail

For example, if a user asks "how does authentication work?":
1. `semantic_search.py --query "authentication flow and user login"` → get relevant files/chunks
2. Read the top results to understand the architecture
3. Use Grep if you need to trace specific function calls

## Configuration

The index configuration lives at `.index/config.json` in the project root.
If it doesn't exist, `build_index.py` creates one from defaults on first run.

Key settings the user might want to change:
- `embedding_model`: which model to use (default: "nomic-ai/nomic-embed-text-v1.5")
- `embedding_dimensions`: vector size (default: 768)
- `chunk_max_tokens`: maximum chunk size (default: 512)
- `chunk_overlap_tokens`: overlap between chunks (default: 50)
- `file_extensions`: which file types to index
- `exclude_patterns`: additional ignore patterns beyond .gitignore
- `search_default_top_k`: default number of results (default: 10)
- `search_default_threshold`: minimum similarity score (default: 0.3)

Read `references/embedding-models.md` for guidance on choosing an embedding model.

## Troubleshooting

- **"No index found"**: Run build_index.py first to create the .index/ directory
- **"API key not found"**: Set OPENROUTER_API_KEY env var or add to .index/config.json
- **Slow indexing**: Large projects (>1000 files) take time on first run; subsequent runs are incremental
- **Poor search results**: Try adjusting chunk_max_tokens (smaller = more precise, larger = more context) or switching to a code-specific embedding model
- **"Module not found" errors**: Re-run setup.sh to ensure venv is properly configured
```

---

## 4. Python Module Architecture

### 4.1 Module Dependency Graph

```
build_index.py ──→ config.py ──→ models.py
       │              │
       ├──→ hasher.py │
       ├──→ chunker.py
       ├──→ embedder.py
       └──→ store.py

semantic_search.py ──→ config.py
       ├──→ embedder.py  (to embed the query)
       └──→ store.py     (to search vectors)

index_status.py ──→ config.py
       └──→ store.py     (to read stats)
```

### 4.2 models.py — Data Classes

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class ChunkType(Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE_LEVEL = "module_level"
    MARKDOWN_SECTION = "markdown_section"
    MARKDOWN_FRONTMATTER = "markdown_frontmatter"
    UNKNOWN = "unknown"

@dataclass
class Chunk:
    """A single indexed chunk of code or documentation."""
    id: str                          # SHA-256 of (file_path + content)
    file_path: str                   # Relative to project root
    start_line: int
    end_line: int
    content: str                     # Raw text content
    chunk_type: ChunkType
    language: Optional[str] = None   # e.g., "python", "typescript", "markdown"
    symbol_name: Optional[str] = None  # Function/class name if applicable
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

@dataclass
class SearchResult:
    """A single search hit."""
    chunk: Chunk
    score: float                     # Similarity score 0.0–1.0
    rank: int                        # 1-based rank in results

@dataclass
class IndexStats:
    """Summary of the index state."""
    total_files: int
    total_chunks: int
    last_indexed: str                # ISO timestamp
    stale_files: int                 # Files changed since last index
    embedding_model: str
    embedding_dimensions: int
    index_size_bytes: int
```

### 4.3 config.py — Configuration Loader

Responsibilities:
- Load `.index/config.json` or create from defaults
- Merge with environment variables (env vars override config file)
- Validate configuration
- Provide typed access to all settings

Key behavior:
- `OPENROUTER_API_KEY` env var takes precedence over config file `api_key`
- `SEMANTIC_INDEX_MODEL` env var overrides `embedding_model` in config
- Config file is created on first `build_index.py` run if missing
- Unknown keys in config are preserved (forward compatibility)

### 4.4 hasher.py — File Change Detection

Responsibilities:
- Compute SHA-256 hash of each file
- Maintain `.index/manifest.json` mapping `file_path → {hash, last_indexed, chunk_count}`
- On re-index: compare current hashes to manifest, return list of changed/new/deleted files
- Handle deleted files: remove their chunks from the store

Algorithm:
```
1. Walk project directory, respecting .gitignore + .indexignore
2. For each file, compute SHA-256 of contents
3. Compare to manifest:
   - Hash matches → skip (unchanged)
   - Hash differs → mark for re-index
   - New file → mark for indexing
   - In manifest but not on disk → mark for deletion
4. Return: {to_index: [...], to_delete: [...], unchanged: int}
```

### 4.5 chunker.py — AST-Aware Chunking

This is the most complex module. It dispatches to specialized chunkers based on
file type, with a fallback for unrecognized formats. Strategy-specific logic lives
in `lib/chunkers/` submodules; `chunker.py` handles dispatch and shared utilities.

**Strategy 1 — Code files (via Tree-sitter) → `chunkers/code.py`:**
1. Parse file with Tree-sitter grammar for the detected language
2. Extract top-level nodes: functions, classes, methods
3. For each node:
   - If under `chunk_max_tokens` → one chunk
   - If over → split at logical sub-boundaries (nested functions, if/else blocks)
   - Preserve leading comments/docstrings with their parent node
4. Remaining module-level code (imports, constants, top-level statements) → one "module_level" chunk
5. Each chunk includes `chunk_overlap_tokens` of context from the previous chunk

**Strategy 2 — Markdown files → `chunkers/markdown.py`:**
1. Split on headers (# , ## , ### , etc.)
2. Each section (header + content until next same-or-higher-level header) = one chunk
3. Frontmatter (YAML between `---` delimiters) = separate chunk
4. If a section exceeds `chunk_max_tokens`, split at paragraph boundaries
5. Each chunk inherits parent headers as metadata for context

**Strategy 3 — DITA XML files → `chunkers/dita.py` (Phase 5):**
1. Parse XML via `xml.etree.ElementTree` (stdlib, no extra deps)
2. Detect topic type from root element (`<topic>`, `<concept>`, `<task>`, `<reference>`,
   `<glossentry>`, `<troubleshooting>`) or via DITA `class` attribute fallback
3. Extract `<prolog>` metadata (keywords, audience, category) as chunk context
4. Each topic = one chunk: `<title>` + `<shortdesc>` + body text (tags stripped)
5. If a topic exceeds `chunk_max_tokens`, split at `<section>` boundaries
6. For `.ditamap` files: walk `<topicref>` hierarchy → one "map overview" chunk
7. Propagate `xml:lang` attributes as chunk metadata for multilingual indexing
8. Note `conref`/`conkeyref` in metadata (not resolved; source file indexed separately)

**Fallback (unsupported formats):**
- Split on blank-line-separated blocks
- Respect `chunk_max_tokens` limit
- Mark as `ChunkType.UNKNOWN`

**Dispatch logic in `chunker.py`:**

| File Extension | Strategy | Chunker Module |
|---------------|----------|----------------|
| `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.rb`, `.php`, etc. | Tree-sitter AST | `chunkers/code.py` |
| `.md`, `.mdx`, `.rst` | Header-based | `chunkers/markdown.py` |
| `.dita`, `.ditamap` | XML topic-based | `chunkers/dita.py` |
| `.txt` and others | Fallback | `chunker.py` (inline) |

**Supported languages (Tree-sitter, Phase 1):**
Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP

**Supported languages (regex fallback, Phase 1):**
All other text files matching configured extensions

### 4.6 embedder.py — Provider Abstraction & Factory

The embedding system uses a provider pattern. `embedder.py` contains the abstract
interface and factory function. Concrete providers live in `providers/`.

#### Abstract Interface (embedder.py)

```python
from abc import ABC, abstractmethod
from typing import Optional

class EmbeddingProvider(ABC):
    """Base class for all embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts. Returns list of vectors.
        Provider handles prefixing (e.g., 'search_document:' for Nomic)."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query. Returns one vector.
        Provider handles prefixing (e.g., 'search_query:' for Nomic)."""

    @abstractmethod
    def get_dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""


def create_embedder(config) -> EmbeddingProvider:
    """Factory: instantiate the right provider based on config.embedding.provider.

    Supported providers:
      - "openrouter": REST API via OpenRouter (requires API key)
      - "huggingface": Local inference via sentence-transformers (no API key)
    """
    provider = config.embedding.provider

    if provider == "openrouter":
        from .providers.openrouter import OpenRouterProvider
        return OpenRouterProvider(config)
    elif provider == "huggingface":
        from .providers.huggingface import HuggingFaceProvider
        return HuggingFaceProvider(config)
    else:
        raise ValueError(f"Unknown embedding provider: '{provider}'. "
                         f"Supported: 'openrouter', 'huggingface'")
```

Note: Provider imports are lazy (inside the factory function). This means
`sentence-transformers` is never imported if the user uses OpenRouter, and
`requests` is never imported if the user uses HuggingFace. This keeps
dependencies optional.

#### providers/openrouter.py

The existing OpenRouter implementation, extracted into its own module:

```python
class OpenRouterProvider(EmbeddingProvider):
    def __init__(self, config):
        self.api_key = config.get_api_key()  # env var or config file
        self.model = config.embedding.model
        self.dimensions = config.embedding.dimensions
        self.batch_size = config.embedding.batch_size
        self.document_prefix = config.embedding.document_prefix
        self.query_prefix = config.embedding.query_prefix
        self.max_retries = config.embedding.max_retries
        self.retry_delay = config.embedding.retry_delay_seconds
```

Responsibilities:
- REST calls to `https://openrouter.ai/api/v1/embeddings`
- Auth: `Authorization: Bearer <api_key>`
- Body: `{"model": "<model>", "input": ["text1", ...], "dimensions": N}`
- Batching: up to 100 texts per request
- Retry with exponential backoff
- Prefixes for asymmetric models (Nomic: `search_document:` / `search_query:`)

#### providers/huggingface.py

Local embedding using the `sentence-transformers` library:

```python
class HuggingFaceProvider(EmbeddingProvider):
    def __init__(self, config):
        # Lazy import — only loaded when this provider is selected
        from sentence_transformers import SentenceTransformer

        self.model_id = config.embedding.model
        self._dimensions = config.embedding.dimensions
        self.document_prefix = config.embedding.document_prefix
        self.query_prefix = config.embedding.query_prefix
        self.batch_size = config.embedding.batch_size
        self.device = config.embedding.get("device", None)  # None = auto-detect

        # Load model (downloads on first use to ~/.cache/huggingface/hub)
        self._model = SentenceTransformer(
            self.model_id,
            trust_remote_code=True,
            device=self.device,  # None → auto (CUDA > MPS > CPU)
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self.document_prefix + t for t in texts]
        embeddings = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        prefixed = self.query_prefix + query
        embedding = self._model.encode(
            [prefixed],
            convert_to_numpy=True,
        )
        return embedding[0].tolist()

    def get_dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self.model_id
```

Key behaviors:
- **First run**: Downloads model to `~/.cache/huggingface/hub` (274MB for Nomic).
  Override cache location with `HF_HUB_CACHE` env var.
- **Device auto-detection**: sentence-transformers automatically selects
  CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU. No config needed, but
  `device` field in config can force a specific device.
- **Performance**: ~50-100 chunks/sec on CPU, ~500+ on GPU. A typical 1,600-chunk
  project indexes in 15-30 seconds on CPU.
- **Same model compatibility**: `nomic-ai/nomic-embed-text-v1.5` produces identical
  vectors whether run through OpenRouter or locally. Indexes built with one provider
  can be searched with the other, as long as model + dimensions match.
- **No API key needed**: Fully offline after first model download.

#### Embedding Cache

The cache layer sits above the provider abstraction (in `embedder.py`):

- Store in `.index/embedding_cache.json`: `{content_hash: vector}`
- On indexing: check cache before calling the provider
- Cache is invalidated when model or dimensions change
- Saves cost (OpenRouter) or time (HuggingFace) on incremental re-indexing

### 4.7 store.py — Vector Store (LanceDB)

Responsibilities:
- Create/open LanceDB database in `.index/lancedb/`
- Store chunk embeddings with full metadata
- Perform vector similarity search (cosine similarity)
- Delete chunks by file path (for incremental re-index)
- Report statistics (total chunks, index size, etc.)

LanceDB schema:
```python
import lancedb
import pyarrow as pa

SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("file_path", pa.string()),
    pa.field("start_line", pa.int32()),
    pa.field("end_line", pa.int32()),
    pa.field("content", pa.string()),
    pa.field("chunk_type", pa.string()),
    pa.field("language", pa.string()),
    pa.field("symbol_name", pa.string()),
    pa.field("token_count", pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), list_size=EMBEDDING_DIM)),
])
```

Why LanceDB:
- File-based (no server needed), persists to a directory
- Fast vector search with IVF-PQ indexing for larger datasets
- Native Python, pip-installable
- Apache Arrow format — efficient and portable
- Supports filtering (e.g., search only in `.py` files)
- Active open-source project with good documentation

### 4.8 CLI Scripts

**build_index.py:**
```
Usage: python build_index.py --project-dir <path> [--config <config.json>] [--full]

Arguments:
  --project-dir     Path to the project root (required)
  --config          Path to config file (default: <project-dir>/.index/config.json)
  --full            Force full re-index (ignore manifest, re-index everything)

Exit codes:
  0  Success
  1  Configuration error (missing API key, invalid config)
  2  Indexing error (API failure, parse error)

Output (stdout, JSON):
  {
    "status": "success",
    "files_indexed": 42,
    "files_skipped": 180,
    "files_deleted": 2,
    "chunks_created": 387,
    "duration_seconds": 12.4,
    "embedding_api_calls": 4
  }
```

**semantic_search.py:**
```
Usage: python semantic_search.py --project-dir <path> --query <text> [--top-k N] [--threshold F] [--filter-lang <lang>] [--filter-path <glob>]

Arguments:
  --project-dir     Path to the project root (required)
  --query           Natural language search query (required)
  --top-k           Max results to return (default: from config, usually 10)
  --threshold       Min similarity score 0.0–1.0 (default: from config, usually 0.3)
  --filter-lang     Only search in files of this language (e.g., "python")
  --filter-path     Only search files matching this glob (e.g., "src/**")

Output (stdout, JSON):
  {
    "query": "how does authentication work?",
    "results": [
      {
        "rank": 1,
        "score": 0.847,
        "file_path": "src/auth/middleware.py",
        "start_line": 15,
        "end_line": 48,
        "chunk_type": "function",
        "symbol_name": "verify_jwt_token",
        "language": "python",
        "content": "def verify_jwt_token(request):\n    ..."
      }
    ],
    "total_results": 7,
    "search_duration_ms": 34
  }
```

**index_status.py:**
```
Usage: python index_status.py --project-dir <path>

Output (stdout, JSON):
  {
    "indexed": true,
    "total_files": 222,
    "total_chunks": 1847,
    "last_indexed": "2026-03-19T14:30:00Z",
    "stale_files": 3,
    "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
    "embedding_dimensions": 768,
    "index_size_mb": 12.4,
    "languages": {"python": 120, "typescript": 80, "markdown": 22}
  }
```

---

## 5. Configuration Schema

### .index/config.json

```json
{
  "$schema_version": "1.0",

  "embedding": {
    "provider": "openrouter",
    "api_key": null,
    "model": "nomic-ai/nomic-embed-text-v1.5",
    "dimensions": 768,
    "batch_size": 50,
    "query_prefix": "search_query: ",
    "document_prefix": "search_document: ",
    "max_retries": 3,
    "retry_delay_seconds": 1.0,
    "device": null
  },

  "chunking": {
    "max_tokens": 512,
    "overlap_tokens": 50,
    "min_tokens": 20
  },

  "indexing": {
    "file_extensions": [
      ".py", ".js", ".ts", ".tsx", ".jsx",
      ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp",
      ".rb", ".php",
      ".md", ".mdx", ".txt", ".rst",
      ".dita", ".ditamap"
    ],
    "exclude_patterns": [
      "node_modules/", "venv/", ".venv/", "__pycache__/",
      "dist/", "build/", ".git/", ".index/",
      "*.min.js", "*.min.css", "*.map",
      "package-lock.json", "yarn.lock", "poetry.lock",
      "*.pyc", "*.pyo", "*.so", "*.dylib"
    ],
    "max_file_size_kb": 500,
    "respect_gitignore": true
  },

  "search": {
    "default_top_k": 10,
    "default_threshold": 0.3
  }
}
```

### Provider-Specific Configuration

The `embedding` section works for both providers. The `provider` field selects
which backend processes the embeddings:

**`"openrouter"` (default)** — Remote API:
- Requires `api_key` (or `OPENROUTER_API_KEY` env var)
- `batch_size`: up to 100 texts per API call (default 50)
- `max_retries` and `retry_delay_seconds`: for transient API failures
- `device`: ignored (not applicable)

**`"huggingface"` — Local inference:**
- No `api_key` needed (field ignored)
- `model`: any sentence-transformers compatible model from HuggingFace Hub
- `batch_size`: controls in-process batch size (default 50, can increase on GPU)
- `device`: `null` (auto-detect), `"cpu"`, `"cuda"`, or `"mps"` (Apple Silicon)
- `max_retries`: ignored (no network calls)
- First run downloads model to `~/.cache/huggingface/hub` (override with `HF_HUB_CACHE` env var)

**Example: switching to local HuggingFace:**
```json
{
  "embedding": {
    "provider": "huggingface",
    "model": "nomic-ai/nomic-embed-text-v1.5",
    "dimensions": 768,
    "batch_size": 64,
    "device": null
  }
}
```

The same model name works with both providers. An index built with OpenRouter
can be searched with HuggingFace and vice versa — as long as model + dimensions
match, the vectors are identical.

### .indexignore

Works like `.gitignore` — one pattern per line. Applied in addition to `.gitignore` and `config.exclude_patterns`.

```
# Ignore test fixtures
tests/fixtures/
# Ignore generated code
**/generated/
# Ignore specific large files
data/large-dataset.json
```

### Environment Variables

| Variable | Purpose | Overrides |
|----------|---------|-----------|
| `OPENROUTER_API_KEY` | API key for OpenRouter | `config.embedding.api_key` |
| `SEMANTIC_INDEX_PROVIDER` | Embedding provider | `config.embedding.provider` |
| `SEMANTIC_INDEX_MODEL` | Embedding model name | `config.embedding.model` |
| `SEMANTIC_INDEX_DIMENSIONS` | Vector dimensions | `config.embedding.dimensions` |
| `HF_HUB_CACHE` | HuggingFace model cache dir | Default `~/.cache/huggingface/hub` |

---

## 6. Chunking Strategy

### Code Files — AST-Aware Chunking

The chunker uses Tree-sitter to parse code into an AST, then extracts meaningful units:

```
Source File
├── Imports / top-level statements  →  1 "module_level" chunk
├── Function `authenticate_user`    →  1 "function" chunk
├── Class `UserService`
│   ├── Method `__init__`           →  1 "method" chunk
│   ├── Method `create_user`        →  1 "method" chunk
│   └── Method `validate_email`     →  1 "method" chunk
├── Function `hash_password`        →  1 "function" chunk
└── Top-level code (if __name__)    →  1 "module_level" chunk
```

**Why AST-aware matters**: Naive line-based splitting breaks functions in half, separating the signature from the body. A search for "password hashing" might match the body but lose the function name. AST-aware chunking keeps `hash_password` as one unit, so the search result includes both the name and the implementation.

**Overflow handling**: If a single function/class exceeds `max_tokens`:
1. Try splitting at method boundaries (for classes)
2. Try splitting at nested function boundaries
3. Fall back to splitting at blank lines within the function
4. Last resort: hard split at `max_tokens` with `overlap_tokens` context

**Context enrichment**: Each chunk's metadata includes:
- The file path (so results show where it came from)
- The parent class name (for methods)
- The function/class docstring (even if it's technically part of the body)
- Import statements from the file's module_level chunk (as metadata, not in the content — so the AI can understand dependencies without bloating the chunk)

### Markdown Files — Header-Based Chunking

```
Document
├── Frontmatter (YAML)              →  1 "markdown_frontmatter" chunk
├── # Introduction                   →  1 "markdown_section" chunk
├── ## Getting Started
│   ├── ### Installation             →  1 "markdown_section" chunk
│   └── ### Configuration            →  1 "markdown_section" chunk
├── ## API Reference                 →  1 "markdown_section" chunk (if small)
│   ├── ### Endpoints                →  split into multiple if large
```

Each markdown chunk preserves the header hierarchy as metadata:
```json
{
  "content": "### Installation\n\nRun `npm install semantic-index`...",
  "metadata": {
    "header_path": ["Getting Started", "Installation"],
    "header_level": 3
  }
}
```

---

## 7. Embedding Pipeline

### Request Flow

```
Chunks → Batch (≤50) → Add prefix → OpenRouter API → Vectors → Store
                                         ↑
                               Cache check (skip if cached)
```

### Embedding Model Selection

Default: `nomic-ai/nomic-embed-text-v1.5`

Rationale:
- 768 dimensions (good balance of quality vs. storage)
- 8192 token context window (handles large chunks)
- Open-source model, well-supported on OpenRouter
- Supports task prefixes (`search_document:` / `search_query:`) for asymmetric search
- Strong performance on code + text benchmarks
- Low cost on OpenRouter (~$0.02 per 1M tokens)

Alternatives (documented in `references/embedding-models.md`):
| Model | Dimensions | Context | Best For | OpenRouter ID |
|-------|-----------|---------|----------|---------------|
| Nomic Embed Text v1.5 | 768 | 8192 | General code+docs (default) | `nomic-ai/nomic-embed-text-v1.5` |
| Nomic Embed Code | 768 | 8192 | Code-heavy projects | `nomic-ai/nomic-embed-code-v1` |
| OpenAI text-embedding-3-small | 1536 | 8191 | Highest quality | `openai/text-embedding-3-small` |
| OpenAI text-embedding-3-large | 3072 | 8191 | Maximum accuracy | `openai/text-embedding-3-large` |

### Batch Processing

```python
# Pseudocode for the embedding pipeline
def index_project(project_dir, config):
    # 1. Detect changes
    changes = hasher.detect_changes(project_dir)

    # 2. Delete removed file chunks
    for deleted_file in changes.to_delete:
        store.delete_by_file(deleted_file)

    # 3. Chunk changed/new files
    all_chunks = []
    for file_path in changes.to_index:
        chunks = chunker.chunk_file(file_path, config)
        all_chunks.extend(chunks)

    # 4. Filter out chunks that are already cached (same content hash)
    uncached_chunks = [c for c in all_chunks if not cache.has(c.id)]
    cached_chunks = [c for c in all_chunks if cache.has(c.id)]

    # 5. Embed uncached chunks in batches
    for batch in batched(uncached_chunks, config.embedding.batch_size):
        texts = [config.embedding.document_prefix + c.content for c in batch]
        vectors = embedder.embed_texts(texts)
        for chunk, vector in zip(batch, vectors):
            chunk.vector = vector
            cache.set(chunk.id, vector)

    # 6. Retrieve cached vectors
    for chunk in cached_chunks:
        chunk.vector = cache.get(chunk.id)

    # 7. Upsert all chunks into store
    store.delete_by_files([c.file_path for c in all_chunks])  # remove old chunks
    store.add(all_chunks)

    # 8. Update manifest
    hasher.update_manifest(changes)
```

### Cost Estimation

For a typical project (500 files, ~100K lines, ~1,650 chunks, ~500K tokens):

| Provider | Full Index Cost | Incremental (10 files) | Speed |
|----------|----------------|----------------------|-------|
| **OpenRouter** (Nomic) | ~$0.01 | ~$0.0002 | ~5-15 sec (network-bound) |
| **HuggingFace** (local CPU) | $0.00 | $0.00 | ~15-30 sec (compute-bound) |
| **HuggingFace** (local GPU) | $0.00 | $0.00 | ~3-5 sec |

HuggingFace has a one-time model download cost (~274MB for Nomic, ~600MB for
larger models) but zero ongoing API costs. For teams indexing frequently or
working in air-gapped environments, local is significantly cheaper over time.

---

## 8. Vector Store & Search

### LanceDB Architecture

```
.index/
├── config.json            # Configuration
├── manifest.json          # File hash manifest
├── embedding_cache.json   # Embedding cache (content_hash → vector)
└── lancedb/               # LanceDB database directory
    └── chunks.lance/      # Lance table (Arrow format)
```

### Search Algorithm

```python
def search(query, config, top_k=10, threshold=0.3, filters=None):
    # 1. Embed the query
    query_vector = embedder.embed_query(config.embedding.query_prefix + query)

    # 2. Vector similarity search
    results = store.search(
        vector=query_vector,
        top_k=top_k * 2,  # Over-fetch to allow for filtering
        metric="cosine"
    )

    # 3. Apply filters (language, path glob)
    if filters:
        results = apply_filters(results, filters)

    # 4. Apply threshold
    results = [r for r in results if r.score >= threshold]

    # 5. Truncate to top_k
    results = results[:top_k]

    # 6. Return ranked SearchResult objects
    return [SearchResult(chunk=r.chunk, score=r.score, rank=i+1)
            for i, r in enumerate(results)]
```

### Why Cosine Similarity

Cosine similarity measures the angle between two vectors, making it scale-invariant. This is important because embedding models don't guarantee consistent vector magnitudes across different text lengths. A 10-line function and a 50-line function can have equally "close" embeddings to a query, regardless of their raw vector lengths.

---

## 9. File Format Specifications

### manifest.json

```json
{
  "version": "1.0",
  "last_indexed": "2026-03-19T14:30:00Z",
  "project_dir": "/absolute/path/to/project",
  "files": {
    "src/auth/middleware.py": {
      "hash": "sha256:a1b2c3d4...",
      "last_indexed": "2026-03-19T14:30:00Z",
      "chunk_count": 8,
      "file_size_bytes": 4521
    },
    "docs/getting-started.md": {
      "hash": "sha256:e5f6g7h8...",
      "last_indexed": "2026-03-19T14:28:00Z",
      "chunk_count": 5,
      "file_size_bytes": 2103
    }
  }
}
```

### embedding_cache.json

```json
{
  "version": "1.0",
  "model": "nomic-ai/nomic-embed-text-v1.5",
  "dimensions": 768,
  "entries": {
    "sha256:content_hash_1": [0.0123, -0.0456, ...],
    "sha256:content_hash_2": [0.0789, -0.0012, ...]
  }
}
```

Note: The cache is invalidated entirely if the embedding model or dimensions change. This is tracked by the `model` and `dimensions` fields — if they don't match the current config, the cache is cleared and all chunks are re-embedded.

---

## 10. Implementation Plan

### Phase 1: Foundation (MVP)

**Goal**: Working index + search for Python and JavaScript files + markdown.

| Step | Task | Estimated Effort | Status |
|------|------|-----------------|--------|
| 1.1 | Project scaffolding: create directory structure, `requirements.txt`, `setup.sh` | 30 min | ✅ Done |
| 1.2 | `models.py`: data classes | 30 min | ✅ Done |
| 1.3 | `config.py`: load/create/validate config, env var override | 1 hr | ✅ Done |
| 1.4 | `hasher.py`: file walking (with .gitignore), SHA-256, manifest read/write | 1.5 hr | ✅ Done |
| 1.5 | `chunker.py` — markdown splitting (header-based) | 1.5 hr | ✅ Done |
| 1.6 | `chunker.py` — code splitting with Tree-sitter (Python + JS/TS only) | 3 hr | ✅ Done |
| 1.7 | `embedder.py`: OpenRouter API client with batching, retry, caching | 2 hr | ✅ Done |
| 1.8 | `store.py`: LanceDB wrapper (create, add, search, delete, stats) | 2 hr | ✅ Done |
| 1.9 | `build_index.py`: CLI orchestration script | 1 hr | ✅ Done |
| 1.10 | `semantic_search.py`: CLI search script | 1 hr | ✅ Done |
| 1.11 | `index_status.py`: CLI status script | 30 min | ✅ Done |
| 1.12 | SKILL.md: finalize instructions | 1 hr | ✅ Done |
| 1.13 | End-to-end testing on a real project | 2 hr | ✅ Done |
| 1.14 | Refactor `chunker.py` into `lib/chunkers/` subpackage (`code.py`, `markdown.py`, fallback in `chunker.py` dispatch) | 2 hr | ✅ Done |
| 1.15 | Batch-commit indexing for large repos (memory optimization) | 2 hr | |

**Total Phase 1: ~21 hours**

**Step 1.14 — Refactor chunker.py into subpackage:**

The current `chunker.py` is ~979 lines — over 3× the 300-line module limit
from `AI_SKILL.md`. Before adding more chunking strategies (Phase 2 language
expansion, Phase 5 DITA), split it into the architecture's intended structure:

```
lib/
├── chunker.py              # Dispatch logic, shared utilities, fallback chunker (~150 lines)
└── chunkers/
    ├── __init__.py
    ├── code.py             # Tree-sitter AST-aware code chunking (~350 lines)
    └── markdown.py         # Header-based markdown chunking (~200 lines)
```

- `chunker.py` keeps the public `chunk_file()` function, `_detect_language()`,
  `_make_chunk_id()`, `_count_tokens()`, `_get_tokenizer()`, and the fallback
  `_chunk_text_fallback()`.
- `chunkers/code.py` gets all Tree-sitter logic: `_get_parser()`, `_get_ts_language()`,
  `_chunk_code_with_treesitter()`, `_chunk_class_node()`, `_split_oversized_chunk()`,
  `EXTRACTABLE_NODES`, `METHOD_NODES`, etc.
- `chunkers/markdown.py` gets `_chunk_markdown()`, `_split_text_by_paragraphs()`,
  `_HEADER_RE`, `_FRONTMATTER_RE`.
- Shared helpers (`_make_chunk_id`, `_count_tokens`, `_split_oversized_chunk`) are
  imported from `chunker.py` by the submodules.

No behavioral changes — pure structural refactor. All existing tests (if any)
should pass without modification.

**Step 1.15 — Batch-commit indexing for large repos:**

The current `build_index.py` accumulates all chunks in memory, embeds them all,
then commits to LanceDB in one atomic write. This guarantees index integrity
(if embedding fails, the old index stays intact) but consumes significant memory
on large repos (e.g., 8K chunks × 1024 dims × 4 bytes = ~32MB vectors alone,
plus all content strings).

The improvement: process files in batches of 50-100 files at a time:

1. Chunk a batch of files
2. Embed the batch
3. Delete old chunks for those files from the store
4. Commit the new chunks to LanceDB
5. Repeat for the next batch
6. Update the manifest only after all batches succeed

If a failure occurs mid-way, the manifest won't reflect uncommitted files,
so the next `build_index.py` run re-processes them. The tradeoff is a
potentially partially-updated index on failure (some files at new version,
some at old), but `--full` re-index always recovers to a clean state.

Peak memory drops from O(all_chunks) to O(batch_size_chunks), making
10K+ file monorepos viable without OOM risk.

**Dependencies (requirements.txt):**
```
lancedb>=0.6.0
pyarrow>=14.0.0
tree-sitter>=0.21.0
tree-sitter-python>=0.21.0
tree-sitter-javascript>=0.21.0
tree-sitter-typescript>=0.21.0
requests>=2.31.0
tiktoken>=0.6.0
pathspec>=0.12.0
```

### Phase 2: Language Expansion

**Goal**: Add Tree-sitter grammars for Go, Rust, Java, C/C++, Ruby, PHP.

| Step | Task |
|------|------|
| 2.1 | Add tree-sitter grammars to requirements |
| 2.2 | Implement language-specific AST node queries for each language |
| 2.3 | Test chunking quality across languages |
| 2.4 | Update `references/supported-languages.md` |

### Phase 3: Search Quality Improvements

**Goal**: Better search results through hybrid search and re-ranking.

| Step | Task |
|------|------|
| 3.1 | Add BM25 keyword index alongside vector store |
| 3.2 | Implement Reciprocal Rank Fusion (RRF) to merge BM25 + vector results |
| 3.3 | Add optional re-ranking step (cross-encoder model via OpenRouter) |
| 3.4 | Tunable weights: user can adjust semantic vs. keyword balance |

### Phase 4: HuggingFace Local Embedding Provider

**Goal**: Add local embedding via sentence-transformers as an alternative to
OpenRouter, enabling zero-cost, offline, low-latency indexing and search.

| Step | Task | Details |
|------|------|---------|
| 4.1 | Create `lib/providers/` package | `__init__.py` with provider registry |
| 4.2 | Extract `OpenRouterProvider` from existing `embedder.py` | Move current OpenRouter logic to `providers/openrouter.py`, keep same behavior |
| 4.3 | Define `EmbeddingProvider` ABC in `embedder.py` | `embed_texts()`, `embed_query()`, `get_dimensions()`, `model_name` property |
| 4.4 | Create `create_embedder()` factory in `embedder.py` | Reads `config.embedding.provider`, lazy-imports the right provider |
| 4.5 | Implement `HuggingFaceProvider` in `providers/huggingface.py` | sentence-transformers `SentenceTransformer.encode()`, auto device detection |
| 4.6 | Create `requirements-huggingface.txt` | `sentence-transformers>=3.0.0`, `torch>=2.0.0` |
| 4.7 | Update `setup.sh` to handle optional deps | Add `--with-huggingface` flag and auto-detect provider from existing `.index/config.json`. Current `setup.sh` only installs core deps — it has no HuggingFace awareness. |
| 4.8 | Add `SEMANTIC_INDEX_PROVIDER` env var override in `config.py` | Currently not implemented. Add to `_apply_env_overrides()` so env var overrides `config.embedding.provider`. |
| 4.9 | Update `build_index.py` and `semantic_search.py` | Replace `Embedder(config)` with `create_embedder(config)` |
| 4.10 | Add `device` field to config schema and `Config` class | `null` (auto), `"cpu"`, `"cuda"`, `"mps"` |
| 4.11 | Test: same model produces identical vectors via both providers | Index with OpenRouter, search with HuggingFace (and vice versa) |
| 4.12 | Update `references/embedding-models.md` | Add local model recommendations, RAM requirements, download sizes |

**Dependencies (requirements-huggingface.txt):**
```
sentence-transformers>=3.0.0
torch>=2.0.0
```

These are only installed when using the `huggingface` provider. The core
`requirements.txt` stays lightweight (no torch dependency).

**Implementation notes:**

- The refactor from monolithic `Embedder` class to `EmbeddingProvider` ABC is
  the main code change. Existing OpenRouter logic moves to `providers/openrouter.py`
  with minimal modification.
- The embedding cache layer stays in `embedder.py`, above the provider. Both
  providers benefit from caching identically.
- `HuggingFaceProvider.__init__()` triggers model download on first use. This
  can take 1-5 minutes depending on model size and network speed. The provider
  should print a progress message to stderr: `"Downloading model nomic-ai/nomic-embed-text-v1.5 (~274MB)..."`.
- After download, model loads from disk cache in 2-5 seconds.

### Phase 5: DITA Documentation Support

**Goal**: Add first-class support for indexing DITA XML documentation alongside code and Markdown.

DITA (Darwin Information Typing Architecture) is an OASIS open standard for structured
technical documentation. Files are XML with predictable topic-based structure, making them
well-suited for semantic chunking — each topic covers a single subject by design.

| Step | Task | Details |
|------|------|---------|
| 5.1 | Create `lib/chunkers/dita.py` | XML-aware DITA parser using Python's built-in `xml.etree.ElementTree`. No external dependencies needed. |
| 5.2 | Register DITA file extensions | Add `.dita` and `.ditamap` to default `file_extensions` in config and to chunker dispatch logic. |
| 5.3 | Implement topic-level chunking | Split on `<topic>`, `<concept>`, `<task>`, `<reference>`, `<glossentry>`, `<troubleshooting>` boundaries. Each topic = one chunk (unless it exceeds `chunk_max_tokens`). |
| 5.4 | Extract `<prolog>` metadata | Parse `<keywords>`, `<audience>`, `<category>`, `<author>` from `<prolog>` elements. Prepend as context to chunk text to enrich embedding quality (e.g., `[audience: admin] [keywords: installation, upgrade] ...`). |
| 5.5 | Implement section-level splitting | When a single topic exceeds `chunk_max_tokens`, split at `<section>` boundaries within `<body>` / `<taskbody>` / `<conbody>` / `<refbody>`. Preserve parent `<title>` as context in each sub-chunk. |
| 5.6 | Implement XML text extraction | Strip XML tags for embedding text while preserving original XML in the stored chunk. Concatenate text from `<title>`, `<shortdesc>`, `<p>`, `<li>`, `<step>`, `<cmd>`, `<codeblock>`, `<note>`, etc. Maintain natural reading order. |
| 5.7 | Handle `.ditamap` files | Parse `<topicref>` hierarchy to extract navigation structure, topic titles, and `navtitle` attributes. Index as a single "map overview" chunk per ditamap — useful for queries like "where is the installation guide?" |
| 5.8 | Handle DITA specializations | Custom topic types (industry-specific) extend the base `<topic>` element. Use generic `<topic>` detection as fallback — if the root or any descendant matches `<topic>` or has `class` attribute containing `topic/topic`, treat it as a topic. |
| 5.9 | Handle `xml:lang` propagation | DITA supports `xml:lang` at any element level, inherited by children. Extract and store language info as chunk metadata. Pair with multilingual embedding model (BGE-M3) for cross-lingual DITA search. |
| 5.10 | Handle content references (conref) | For MVP: index raw file content without resolving conrefs. The referenced text will be indexed from its source file. Add a note in chunk metadata when `conref` or `conkeyref` attributes are detected, so search results can indicate partial content. |
| 5.11 | Update `references/supported-languages.md` | Add DITA section documenting supported topic types, recognized elements, and chunking behavior. |
| 5.12 | Test with real DITA content | Validate with concept, task, reference, and glossary entry topics. Verify ditamap indexing. Test multilingual DITA sets with `xml:lang` attributes. |

**Chunking strategy summary for `dita.py`:**

```
.dita file
  └── Parse XML tree
       ├── Extract <prolog> metadata (keywords, audience, category)
       ├── For each <topic>/<concept>/<task>/<reference>:
       │    ├── title + shortdesc + body text → chunk text
       │    ├── If under chunk_max_tokens → one chunk
       │    ├── If over → split at <section> boundaries
       │    └── Attach prolog metadata + xml:lang as chunk metadata
       └── Mark conref/conkeyref references in metadata (not resolved)

.ditamap file
  └── Parse XML tree
       ├── Walk <topicref> hierarchy
       ├── Extract navtitle / href / keys for each reference
       └── One "map overview" chunk with full navigation structure
```

**DITA elements to extract text from (in order):**

`<title>`, `<shortdesc>`, `<abstract>`, `<p>`, `<li>`, `<sli>`, `<dt>`, `<dd>`,
`<note>`, `<section>`, `<example>`, `<step>`, `<cmd>`, `<info>`, `<stepresult>`,
`<result>`, `<prereq>`, `<context>`, `<codeblock>`, `<screen>`, `<msgblock>`,
`<fig>/<title>`, `<table>/<title>`, `<entry>` (table cells), `<stentry>`.

**DITA elements to skip (structural/metadata only):**

`<prolog>` (extracted separately), `<related-links>`, `<link>`, `<topicmeta>` (in maps),
`<navref>`, `<anchor>`, `<data>`, `<data-about>`, `<foreign>`, `<unknown>`.

### Phase 6: Additional Providers (Optional)

**Goal**: Add more provider options for flexibility.

| Step | Task |
|------|------|
| 6.1 | Implement `OllamaProvider` (local Ollama REST API at `localhost:11434`) |
| 6.2 | Implement `OpenAIProvider` (direct OpenAI API, no OpenRouter middleman) |
| 6.3 | Provider auto-detection from model name patterns (e.g., `openai/...` → OpenAI) |

### Phase 7: MCP Bridge (Optional)

**Goal**: Expose the same indexing/search as an MCP server for tools that prefer MCP.

| Step | Task |
|------|------|
| 7.1 | Wrap `build_index.py` / `semantic_search.py` / `index_status.py` as MCP tools |
| 7.2 | Add FastMCP server entry point |
| 7.3 | Keep the SKILL as the primary interface; MCP is an alternative transport |

---

## 11. Reference Script Specs

### setup.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "Setting up semantic-index environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created virtual environment at $VENV_DIR"
fi

# Activate and install core dependencies
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q

# Install HuggingFace dependencies if requested or if config says huggingface
if [ "${1:-}" = "--with-huggingface" ]; then
    echo "Installing HuggingFace local embedding dependencies..."
    pip install -r "$SCRIPT_DIR/requirements-huggingface.txt" -q
elif [ -f "${2:-.index/config.json}" ]; then
    # Auto-detect from config if it exists
    PROVIDER=$(python3 -c "
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
    print(cfg.get('embedding', {}).get('provider', ''))
except: pass
" "${2:-.index/config.json}" 2>/dev/null || true)
    if [ "$PROVIDER" = "huggingface" ]; then
        echo "Config uses HuggingFace provider. Installing local embedding dependencies..."
        pip install -r "$SCRIPT_DIR/requirements-huggingface.txt" -q
    fi
fi

echo "Setup complete. Dependencies installed."
echo "Virtual environment: $VENV_DIR"
```

**Usage:**
- `bash setup.sh` — install core deps only (OpenRouter provider works out of the box)
- `bash setup.sh --with-huggingface` — install core + HuggingFace deps
- If `.index/config.json` exists with `"provider": "huggingface"`, HuggingFace deps
  are auto-installed

### build_index.py — Entry Point Pseudocode

```python
#!/usr/bin/env python3
"""Build/rebuild the semantic index for a project."""

import argparse
import json
import sys
import time

from lib.config import load_config, ensure_index_dir
from lib.hasher import detect_changes, update_manifest
from lib.chunker import chunk_file
from lib.embedder import Embedder
from lib.store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Build the semantic index for a project")
    parser.add_argument("--project-dir", required=True, help="Project root directory")
    parser.add_argument("--config", help="Path to config.json")
    parser.add_argument("--full", action="store_true", help="Force full re-index")
    args = parser.parse_args()

    start_time = time.time()

    # Load configuration
    config = load_config(args.project_dir, args.config)
    ensure_index_dir(args.project_dir)

    # Detect changes
    if args.full:
        changes = detect_changes(args.project_dir, config, force_full=True)
    else:
        changes = detect_changes(args.project_dir, config)

    if not changes.to_index and not changes.to_delete:
        print(json.dumps({"status": "up_to_date", "message": "No changes detected"}))
        sys.exit(0)

    # Initialize components
    embedder = Embedder(config)
    store = VectorStore(args.project_dir, config)

    # Delete removed files from store
    for file_path in changes.to_delete:
        store.delete_by_file(file_path)

    # Chunk and embed new/changed files
    all_chunks = []
    for file_path in changes.to_index:
        chunks = chunk_file(file_path, args.project_dir, config)
        all_chunks.extend(chunks)

    if all_chunks:
        # Remove old chunks for files being re-indexed
        affected_files = set(c.file_path for c in all_chunks)
        for f in affected_files:
            store.delete_by_file(f)

        # Embed in batches
        api_calls = embedder.embed_chunks(all_chunks)

        # Store
        store.add(all_chunks)

    # Update manifest
    update_manifest(args.project_dir, changes)

    duration = time.time() - start_time
    result = {
        "status": "success",
        "files_indexed": len(changes.to_index),
        "files_skipped": changes.unchanged,
        "files_deleted": len(changes.to_delete),
        "chunks_created": len(all_chunks),
        "duration_seconds": round(duration, 1),
        "embedding_api_calls": api_calls if all_chunks else 0,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

### semantic_search.py — Entry Point Pseudocode

```python
#!/usr/bin/env python3
"""Search the semantic index by meaning."""

import argparse
import json
import sys
import time

from lib.config import load_config
from lib.embedder import Embedder
from lib.store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Search the semantic index by meaning")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--filter-lang", default=None)
    parser.add_argument("--filter-path", default=None)
    args = parser.parse_args()

    config = load_config(args.project_dir)
    top_k = args.top_k or config.search.default_top_k
    threshold = args.threshold or config.search.default_threshold

    embedder = Embedder(config)
    store = VectorStore(args.project_dir, config)

    start_time = time.time()

    # Embed the query
    query_vector = embedder.embed_query(args.query)

    # Search
    raw_results = store.search(
        vector=query_vector,
        top_k=top_k * 2,  # over-fetch for filtering
        filters={
            "language": args.filter_lang,
            "file_path_glob": args.filter_path,
        }
    )

    # Apply threshold and truncate
    results = [r for r in raw_results if r["score"] >= threshold][:top_k]

    duration_ms = (time.time() - start_time) * 1000

    output = {
        "query": args.query,
        "results": [
            {
                "rank": i + 1,
                "score": round(r["score"], 4),
                "file_path": r["file_path"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "chunk_type": r["chunk_type"],
                "symbol_name": r["symbol_name"],
                "language": r["language"],
                "content": r["content"],
            }
            for i, r in enumerate(results)
        ],
        "total_results": len(results),
        "search_duration_ms": round(duration_ms),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
```

---

## 12. Future Roadmap

### Near-term (after MVP)

- **Watch mode**: Optional file watcher that auto-indexes on save (opt-in, not default)
- **Multi-project indexes**: Central index store that spans multiple projects
- **Chunk quality scoring**: Automatically detect and flag low-quality chunks (too short, boilerplate, auto-generated)
- **Search history**: Track recent queries for "search again" workflows

### Medium-term

- **Hybrid search (BM25 + vector)**: Significantly better results for queries that mix concepts with specific identifiers
- **Knowledge graph layer**: Extract imports/calls/references between chunks to build a dependency-aware search
- **Incremental embedding updates**: Instead of re-embedding entire files, detect which chunks within a file changed

### Long-term

- **MCP server mode**: Same core logic, exposed as MCP tools for broader compatibility
- **Distributed indexing**: Index remote repositories without cloning (via GitHub API)
- **Multi-modal**: Index images (diagrams, screenshots) using vision embedding models
- **Plugin marketplace**: Publish as a distributable .skill file

---

## 13. Appendix: Prior Art Comparison

| Feature | semantic-index (this SKILL) | Code-Index-MCP | Claude Context | SocratiCode | Code-Memory |
|---------|----------------------------|----------------|----------------|-------------|-------------|
| **Architecture** | SKILL (scripts) | MCP server | MCP server | MCP server | MCP server |
| **Vector DB** | LanceDB (file-based) | None (symbol index) | Milvus Cloud | Qdrant (Docker) | SQLite-vec |
| **Embeddings** | OpenRouter or local HuggingFace | N/A | OpenAI (required) | Ollama (local) | sentence-transformers |
| **Setup** | `bash setup.sh` | pip + config | npm + Zilliz + OpenAI keys | npm + Docker | pip + model download |
| **Infrastructure** | None (file-based) | None | Zilliz Cloud | Docker | None |
| **AST parsing** | Tree-sitter | Tree-sitter (7 langs) | AST (18 langs) | ast-grep (18 langs) | Tree-sitter (8 langs) |
| **Incremental** | SHA-256 manifest | Merkle tree | Checkpointed batches | Checkpointed batches | Not specified |
| **Hybrid search** | Phase 3 (planned) | Regex/fuzzy | BM25 + vector | BM25 + vector (RRF) | BM25 + vector |
| **Offline** | Yes (HuggingFace) / No (OpenRouter) | Yes | No | Yes (Ollama) | Yes |
| **AI-debuggable** | Yes (AI reads + runs scripts) | No (opaque MCP) | No (opaque MCP) | No (opaque MCP) | No (opaque MCP) |
| **Portability** | Any SKILL-compatible tool | Any MCP client | Any MCP client | Any MCP client | Any MCP client |
| **Cost** | Free (HuggingFace) or ~$0.01 (OpenRouter) | Free | Zilliz + OpenAI | Free (self-hosted) | Free |

### Key Differentiators

1. **AI-native transparency**: The AI reads the SKILL instructions and executes the scripts itself. When search results are poor, the AI can reason about why (chunk size? model mismatch? stale index?) and suggest fixes. MCP servers are black boxes.

2. **Zero infrastructure**: No Docker, no cloud accounts, no running servers. Just Python scripts and a file-based database. This makes it work in constrained environments (CI/CD, air-gapped machines, lightweight VMs).

3. **Dual-mode provider flexibility**: Users choose between OpenRouter (remote API, one key for dozens of models) and HuggingFace (local, zero-cost, offline-capable). Switching is a single config field change — same model produces identical vectors in both modes. No other tool offers this seamless local/remote toggle.

4. **Incremental by default**: SHA-256 manifest tracking means re-indexing a large project after changing 3 files costs fractions of a cent and takes seconds.

---

## How To Use This Document

This document is your implementation spec. Hand it to Claude Code with a prompt like:

> "Read the architecture document at `semantic-index-skill-architecture.md`. Implement Phase 1 of the semantic-index SKILL, following the structure, module designs, and configuration schema exactly as specified. Start with project scaffolding (Step 1.1), then implement each module in dependency order: models.py → config.py → hasher.py → chunker.py (markdown first, then Tree-sitter) → embedder.py → store.py → CLI scripts (build_index.py, semantic_search.py, index_status.py) → SKILL.md."

The document is designed so each section is self-contained enough for Claude Code to implement one module at a time without needing to re-read the entire spec.
