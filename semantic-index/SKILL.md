---
name: semantic-index
description: >
  Semantic code and documentation indexing with embedding-based search.
  Creates a local .index/ in the project folder for fast conceptual search
  across code, markdown, DITA XML, and office documents (PDF, DOCX, PPTX).
  Use this skill whenever the user wants to: index a codebase or project
  for semantic search, search code by meaning rather than exact text, find
  where concepts/features/patterns are implemented, understand unfamiliar
  codebases quickly, or asks questions like "where is X handled?" or
  "how does Y work?" about their project. Also trigger when the user
  mentions "semantic search", "index my code", "embeddings", "vector search",
  or "codebase indexing".
---

# Semantic Index

Index code, documentation, and office documents for meaning-based search using embeddings.

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
  other places where the same pattern appears
- When the user mentions "something similar was done before" or "check how
  we handled X" — search for that prior implementation
- When exploring an unfamiliar codebase before making changes: build
  understanding of architecture, naming conventions, and module boundaries
- When the user's task touches a concept that could span multiple files
  and you don't know which ones
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

This creates a `.venv` in the scripts directory and installs core dependencies.
It only needs to run once per machine.

Optional dependency groups (pass as flags to `setup.sh`):
- `--with-huggingface` — local embedding via HuggingFace (no API key needed)
- `--with-office` — PDF, DOCX, PPTX extraction (PyMuPDF, python-docx, python-pptx)
- `--with-mcp` — MCP server transport (see `references/mcp-server.md`)

Example installing everything:
```bash
bash setup.sh --with-huggingface --with-office --with-mcp
```

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

## Path Resolution

Before running any command, resolve these two placeholders:

- `<skill-path>`: Always `~/.kiro/skills/semantic-index`. This is fixed.
- `<project-root>`: The actual workspace root directory. **Always run `pwd`
  first** to get the real path. Never guess from environment variables,
  Machine ID context, or other indirect sources — these can point to
  non-existent or inaccessible paths.

**Common mistake**: Using a path like `/Users/<username>/Documents/workspace`
derived from IDE context variables instead of the actual working directory.
This causes `PermissionError` or "No .index/ directory found" even when the
index exists, because the script tries to create directories under a path
it cannot access.

**Correct pattern**:
```bash
# Step 1: Get the real workspace path
pwd
# Output: /Users/johndoe/src/my-project

# Step 2: Use that exact path in all commands
~/.kiro/skills/semantic-index/scripts/.venv/bin/python \
  ~/.kiro/skills/semantic-index/scripts/index_status.py \
  --project-dir /Users/johndoe/src/my-project
```

## Core Commands

All commands output structured JSON to stdout. Progress and logs go to stderr.

### Indexing

To index the current project:

```bash
<skill-path>/scripts/.venv/bin/python <skill-path>/scripts/build_index.py \
  --project-dir <project-root> \
  [--config <path-to-config.json>] \
  [--full]
```

Arguments:
- `--project-dir` (required): Path to the project root
- `--config`: Path to config.json (default: `<project-root>/.index/config.json`)
- `--full`: Force full re-index, ignoring the manifest

What this does:
1. Scans the project for supported files (code, markdown, DITA XML, office documents)
2. Respects .gitignore and .indexignore patterns
3. Computes SHA-256 hashes to detect changed files
4. Chunks files using format-aware splitting:
   - Code: Tree-sitter AST parsing (functions, classes, methods)
   - Markdown: header-based section splitting
   - DITA XML: topic-aware parsing (concepts, tasks, references, glossary)
   - PDF: page-based splitting with short-page merging
   - DOCX: heading-based sectioning (mirrors markdown strategy)
   - PPTX: slide-based splitting with speaker notes
5. Embeds chunks via the configured provider (OpenRouter API or local HuggingFace)
6. Stores embeddings in `.index/` (LanceDB format) with a BM25 keyword index
7. Saves file manifest for incremental re-indexing

On re-run, only changed/new files are re-indexed (incremental).

Success output:
```json
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

No changes output:
```json
{
  "status": "up_to_date",
  "message": "No changes detected",
  "files_unchanged": 222
}
```

### Searching

To search the index:

```bash
<skill-path>/scripts/.venv/bin/python <skill-path>/scripts/semantic_search.py \
  --project-dir <project-root> \
  --query "your natural language query" \
  [--top-k 10] \
  [--threshold 0.3] \
  [--mode hybrid] \
  [--alpha 0.7] \
  [--rerank] \
  [--filter-lang <lang>] \
  [--filter-path <glob>]
```

Arguments:
- `--project-dir` (required): Path to the project root
- `--query` (required): Natural language search query
- `--top-k`: Max results to return (default: from config, usually 10)
- `--threshold`: Min similarity score 0.0–1.0 (default: from config, usually 0.3)
- `--mode`: Search mode — `vector`, `keyword`, or `hybrid` (default: from config, usually `hybrid`)
- `--alpha`: Hybrid balance — 0.0 = pure keyword, 1.0 = pure vector (default: 0.7)
- `--rerank`: Re-rank results using a cross-encoder model for higher precision (requires HuggingFace deps)
- `--filter-lang`: Only search files of this language (e.g., "python")
- `--filter-path`: Only search files matching this glob (e.g., "src/**")

Search modes:
- `vector` — pure semantic similarity using embeddings
- `keyword` — BM25 keyword matching for when you know specific terms
- `hybrid` (default) — combines both using Reciprocal Rank Fusion

Output:
```json
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

### Status

To check index health:

```bash
<skill-path>/scripts/.venv/bin/python <skill-path>/scripts/index_status.py \
  --project-dir <project-root>
```

Output:
```json
{
  "indexed": true,
  "total_files": 222,
  "total_chunks": 1847,
  "last_indexed": "2026-03-19T14:30:00+00:00",
  "stale_files": 3,
  "embedding_model": "BAAI/bge-m3",
  "embedding_dimensions": 1024,
  "index_size_mb": 12.4,
  "languages": {"python": 120, "typescript": 80, "markdown": 22}
}
```

## Error Handling

All scripts use consistent exit codes:
- `0`: Success
- `1`: Configuration error (missing API key, invalid config, bad arguments)
- `2`: Runtime error (API failure, parse error, I/O error)

Error output (stdout, JSON):
```json
{
  "status": "error",
  "error": "OpenRouter provider requires an API key. Set OPENROUTER_API_KEY env var, add api_key to .index/config.json, or switch to 'huggingface' provider for local embedding.",
  "error_type": "EmbeddingError"
}
```

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
- `embedding.model`: which model to use (default: `BAAI/bge-m3`)
- `embedding.dimensions`: vector size (default: 1024)
- `chunking.max_tokens`: maximum chunk size (default: 512)
- `chunking.overlap_tokens`: overlap between chunks (default: 50)
- `indexing.file_extensions`: which file types to index
- `indexing.exclude_patterns`: additional ignore patterns beyond .gitignore
- `indexing.max_file_size_kb`: max size for text files (default: 500)
- `indexing.max_office_file_size_kb`: max size for office files (default: 50000)
- `search.default_top_k`: default number of results (default: 10)
- `search.default_threshold`: minimum similarity score (default: 0.3)
- `search.mode`: search mode — `vector`, `keyword`, or `hybrid` (default: `hybrid`)
- `search.hybrid_alpha`: hybrid balance 0.0–1.0 (default: 0.7)
- `search.rerank_enabled`: enable cross-encoder reranking (default: false)
- `search.rerank_model`: reranker model (default: `BAAI/bge-reranker-v2-m3`)

Environment variable overrides:
- `OPENROUTER_API_KEY` → overrides `embedding.api_key`
- `SEMANTIC_INDEX_PROVIDER` → overrides `embedding.provider`
- `SEMANTIC_INDEX_MODEL` → overrides `embedding.model`
- `SEMANTIC_INDEX_DIMENSIONS` → overrides `embedding.dimensions`
- `HF_HUB_CACHE` → HuggingFace model cache directory (default `~/.cache/huggingface/hub`)

## Troubleshooting

- **PermissionError or "No such file or directory"**: The `--project-dir`
  path is wrong. Run `pwd` to get the actual workspace root and use that
  exact path. Do not guess paths from IDE context, Machine ID, or
  environment variables — they often point to non-existent locations.
- **"No .index/ directory found" when index exists**: Same cause — the
  `--project-dir` is pointing to a different directory than where `.index/`
  lives. Verify with `ls <project-root>/.index/` before running commands.
- **"No index found"**: Run `build_index.py` first to create the `.index/` directory
- **"No API key found"**: Either set `OPENROUTER_API_KEY` env var / add to config, or switch to `"huggingface"` provider in `.index/config.json` for local embedding with no API key
- **Slow indexing**: Large projects (>1000 files) take time on first run; subsequent runs are incremental
- **Poor search results**: Try adjusting `chunking.max_tokens` (smaller = more precise, larger = more context) or switching to a code-specific embedding model
- **"Module not found" errors**: Re-run `setup.sh` to ensure venv is properly configured
- **Partial index corruption**: Run `build_index.py --full` to force a complete rebuild
