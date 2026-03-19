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

This creates a `.venv` in the scripts directory and installs all dependencies.
It only needs to run once per machine.

The user must have an OpenRouter API key. Check for it:
1. Environment variable `OPENROUTER_API_KEY`
2. Project config at `.index/config.json` → `embedding.api_key` field
3. If neither exists, ask the user to provide one

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
1. Scans the project for supported files (code + markdown)
2. Respects .gitignore and .indexignore patterns
3. Computes SHA-256 hashes to detect changed files
4. Chunks files using AST-aware splitting (code) or header-based splitting (markdown)
5. Embeds chunks via OpenRouter API
6. Stores embeddings in `.index/` (LanceDB format)
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
  [--filter-lang <lang>] \
  [--filter-path <glob>]
```

Arguments:
- `--project-dir` (required): Path to the project root
- `--query` (required): Natural language search query
- `--top-k`: Max results to return (default: from config, usually 10)
- `--threshold`: Min similarity score 0.0–1.0 (default: from config, usually 0.3)
- `--filter-lang`: Only search files of this language (e.g., "python")
- `--filter-path`: Only search files matching this glob (e.g., "src/**")

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
  "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
  "embedding_dimensions": 768,
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
  "error": "No API key found. Set OPENROUTER_API_KEY environment variable or add api_key to .index/config.json",
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
- `embedding.model`: which model to use (default: `nomic-ai/nomic-embed-text-v1.5`)
- `embedding.dimensions`: vector size (default: 768)
- `chunking.max_tokens`: maximum chunk size (default: 512)
- `chunking.overlap_tokens`: overlap between chunks (default: 50)
- `indexing.file_extensions`: which file types to index
- `indexing.exclude_patterns`: additional ignore patterns beyond .gitignore
- `search.default_top_k`: default number of results (default: 10)
- `search.default_threshold`: minimum similarity score (default: 0.3)

Environment variable overrides:
- `OPENROUTER_API_KEY` → overrides `embedding.api_key`
- `SEMANTIC_INDEX_MODEL` → overrides `embedding.model`
- `SEMANTIC_INDEX_DIMENSIONS` → overrides `embedding.dimensions`

## Troubleshooting

- **"No index found"**: Run `build_index.py` first to create the `.index/` directory
- **"No API key found"**: Set `OPENROUTER_API_KEY` env var or add to `.index/config.json`
- **Slow indexing**: Large projects (>1000 files) take time on first run; subsequent runs are incremental
- **Poor search results**: Try adjusting `chunking.max_tokens` (smaller = more precise, larger = more context) or switching to a code-specific embedding model
- **"Module not found" errors**: Re-run `setup.sh` to ensure venv is properly configured
- **Partial index corruption**: Run `build_index.py --full` to force a complete rebuild
