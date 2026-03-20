# Semantic Index Skill

A portable SKILL for embedding-based indexing and semantic search of codebases and documentation. Designed for Claude Code, Cowork, and any SKILL-compatible AI tool.

Instead of grep/glob for exact string matches, this skill lets you search code by meaning — queries like "where is authentication handled?" or "how does the payment flow work?" return the most relevant code and documentation chunks.

## How It Works

1. Scans your project for supported files (code + markdown)
2. Chunks files using AST-aware splitting (Tree-sitter for code, header-based for markdown)
3. Embeds chunks via OpenRouter API (Nomic Embed Text v1.5 by default)
4. Stores embeddings locally in `.index/` (LanceDB format)
5. Searches by cosine similarity against your natural language queries

The `.index/` directory is local, gitignoreable, and fully rebuildable. No servers, no Docker, no infrastructure.

## Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/) API key (for the default embedding provider)

## Installation

```bash
cd semantic-index/scripts
bash setup.sh
```

This creates a `.venv` in the `scripts/` directory and installs all dependencies. Only needs to run once per machine.

### Verify Installation

```bash
semantic-index/scripts/.venv/bin/python -c "import lancedb, tree_sitter, tiktoken; print('All dependencies OK')"
```

## Configuration

### API Key

Set your OpenRouter API key as an environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

Alternatively, add it to `.index/config.json` after first run (the `api_key` field under `embedding`).

### Config File

On first run, `build_index.py` creates `.index/config.json` in your project root with sensible defaults. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding.provider` | `"openrouter"` | Embedding backend (`"openrouter"` or `"huggingface"`) |
| `embedding.model` | `"nomic-ai/nomic-embed-text-v1.5"` | Embedding model |
| `embedding.dimensions` | `768` | Vector dimensionality |
| `embedding.batch_size` | `50` | Texts per API call |
| `chunking.max_tokens` | `512` | Max chunk size in tokens |
| `chunking.overlap_tokens` | `50` | Overlap between adjacent chunks |
| `chunking.min_tokens` | `20` | Minimum chunk size (smaller chunks are discarded) |
| `indexing.file_extensions` | See config | Which file types to index |
| `indexing.exclude_patterns` | See config | Patterns to skip (in addition to `.gitignore`) |
| `indexing.max_file_size_kb` | `500` | Skip files larger than this |
| `search.default_top_k` | `10` | Default number of search results |
| `search.default_threshold` | `0.3` | Minimum similarity score (0.0–1.0) |

### Environment Variable Overrides

Environment variables take precedence over config file values:

| Variable | Overrides |
|----------|-----------|
| `OPENROUTER_API_KEY` | `embedding.api_key` |
| `SEMANTIC_INDEX_MODEL` | `embedding.model` |
| `SEMANTIC_INDEX_DIMENSIONS` | `embedding.dimensions` |

### .indexignore

Create a `.indexignore` file in your project root to exclude additional paths (same syntax as `.gitignore`):

```
tests/fixtures/
**/generated/
data/large-dataset.json
```

## Usage

All commands output structured JSON to stdout. Progress and logs go to stderr.

In the examples below, `SKILL_PATH` refers to the absolute path to the `semantic-index` directory, and `PROJECT` refers to your target project root.

### Build the Index

```bash
# Index a project (incremental — only new/changed files)
$SKILL_PATH/scripts/.venv/bin/python $SKILL_PATH/scripts/build_index.py \
  --project-dir $PROJECT

# Force full re-index (ignore manifest, re-embed everything)
$SKILL_PATH/scripts/.venv/bin/python $SKILL_PATH/scripts/build_index.py \
  --project-dir $PROJECT \
  --full

# Use a custom config file
$SKILL_PATH/scripts/.venv/bin/python $SKILL_PATH/scripts/build_index.py \
  --project-dir $PROJECT \
  --config /path/to/custom-config.json
```

Output on success:
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

If nothing changed since last index:
```json
{
  "status": "up_to_date",
  "message": "No changes detected",
  "files_unchanged": 222
}
```

### Search the Index

```bash
# Basic semantic search
$SKILL_PATH/scripts/.venv/bin/python $SKILL_PATH/scripts/semantic_search.py \
  --project-dir $PROJECT \
  --query "how does authentication work?"

# Limit results and set minimum score
$SKILL_PATH/scripts/.venv/bin/python $SKILL_PATH/scripts/semantic_search.py \
  --project-dir $PROJECT \
  --query "error handling patterns" \
  --top-k 5 \
  --threshold 0.5

# Filter by language
$SKILL_PATH/scripts/.venv/bin/python $SKILL_PATH/scripts/semantic_search.py \
  --project-dir $PROJECT \
  --query "database connection setup" \
  --filter-lang python

# Filter by file path glob
$SKILL_PATH/scripts/.venv/bin/python $SKILL_PATH/scripts/semantic_search.py \
  --project-dir $PROJECT \
  --query "API route definitions" \
  --filter-path "src/**"
```

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

### Check Index Status

```bash
$SKILL_PATH/scripts/.venv/bin/python $SKILL_PATH/scripts/index_status.py \
  --project-dir $PROJECT
```

Output:
```json
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

## Exit Codes

All CLI scripts use consistent exit codes:

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Configuration error (missing API key, invalid config, bad arguments) |
| `2` | Runtime error (API failure, parse error, I/O error, no index) |

On error, JSON is still written to stdout:
```json
{
  "status": "error",
  "error": "No API key found. Set OPENROUTER_API_KEY environment variable or add api_key to .index/config.json",
  "error_type": "EmbeddingError"
}
```

## Supported Languages

AST-aware chunking (Tree-sitter) is available for:
- Python, JavaScript, TypeScript

All other text files matching configured extensions fall back to blank-line splitting.

## Project Structure

```
semantic-index/
├── SKILL.md                    # AI-facing instructions
├── assets/
│   └── default-config.json     # Default configuration template
└── scripts/
    ├── setup.sh                # One-command environment setup
    ├── requirements.txt        # Python dependencies
    ├── build_index.py          # CLI: build/rebuild the index
    ├── semantic_search.py      # CLI: search by meaning
    ├── index_status.py         # CLI: index health check
    └── lib/
        ├── __init__.py
        ├── models.py           # Data classes and exceptions
        ├── config.py           # Configuration loader
        ├── hasher.py           # File change detection (SHA-256)
        ├── chunker.py          # AST-aware code + markdown chunking
        ├── embedder.py         # OpenRouter embedding client + cache
        └── store.py            # LanceDB vector store wrapper
```

## Troubleshooting

**"No API key found"**
Set `OPENROUTER_API_KEY` env var or add `api_key` to `.index/config.json` under the `embedding` section.

**"No index found"**
Run `build_index.py` first to create the `.index/` directory and populate it.

**"Module not found" errors**
Re-run `bash setup.sh` in the `scripts/` directory to ensure the venv is properly configured.

**Slow indexing on first run**
Large projects (1000+ files) take time on the initial full index. Subsequent runs are incremental and only process changed files.

**Poor search results**
- Try smaller `chunking.max_tokens` (more precise chunks) or larger (more context per chunk)
- Lower `search.default_threshold` to see more results
- Check `index_status.py` for stale files — re-index if many files changed

**Stale index**
Run `index_status.py` to check. If `stale_files` is high, re-run `build_index.py` to update.
