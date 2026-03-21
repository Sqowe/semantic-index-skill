# Semantic Index Skill

A portable SKILL for embedding-based indexing and semantic search of codebases and documentation. Designed for Claude Code, Cowork, and any SKILL-compatible AI tool.

Instead of grep/glob for exact string matches, this skill lets you search code by meaning — queries like "where is authentication handled?" or "how does the payment flow work?" return the most relevant code and documentation chunks.

## How It Works

1. Scans your project for supported files (code + markdown)
2. Chunks files using AST-aware splitting (Tree-sitter for code, header-based for markdown, XML-aware for DITA)
3. Embeds chunks via the configured provider (OpenRouter API or local HuggingFace)
4. Stores embeddings locally in `.index/` (LanceDB vectors + BM25 keyword index)
5. Searches using hybrid retrieval (vector similarity + BM25 keyword matching, merged via Reciprocal Rank Fusion)
6. Optionally re-ranks results with a cross-encoder for higher accuracy

The `.index/` directory is local, gitignoreable, and fully rebuildable. No servers, no Docker, no infrastructure.

## Prerequisites

- Python 3.10+
- Git
- An [OpenRouter](https://openrouter.ai/) API key (for the default embedding provider), OR
- The HuggingFace optional dependencies for free local embedding (see [Local Embedding](#local-embedding-huggingface-provider))

## Getting the Skill

Clone this repository and copy the `semantic-index/` directory to your Kiro skills folder:

```bash
# Clone the repo
git clone <repo-url> /tmp/semantic-index-skill

# Copy the skill into Kiro's skills directory
cp -r /tmp/semantic-index-skill/semantic-index ~/.kiro/skills/semantic-index
```

After this, your skill directory should look like:

```
~/.kiro/skills/semantic-index/
├── SKILL.md
├── assets/
│   └── default-config.json
└── scripts/
    ├── setup.sh
    ├── requirements.txt
    ├── build_index.py
    ├── semantic_search.py
    ├── index_status.py
    └── lib/
        └── ... (Python modules)
```

> You can place the skill anywhere you like. The examples below use `~/.kiro/skills/semantic-index` as the skill path. Adjust if you chose a different location.

### How Skill vs. Index Directories Work

The skill is installed once globally (e.g., `~/.kiro/skills/semantic-index/`) and stays read-only during normal use. The `.index/` directory — containing config, manifest, embedding cache, and LanceDB data — is created inside each project you index via `--project-dir`.

```
~/.kiro/skills/semantic-index/   ← skill code (shared, installed once)
~/project-a/.index/              ← index data for project-a
~/project-b/.index/              ← index data for project-b
```

Each project gets its own independent `.index/`, so you can index multiple projects without conflicts. Add `.index/` to your project's `.gitignore` — it's fully rebuildable.

## Installation

Run the setup script to create a virtual environment and install dependencies:

```bash
cd ~/.kiro/skills/semantic-index/scripts
bash setup.sh
```

This creates a `.venv` inside the `scripts/` directory and installs all Python dependencies. Only needs to run once per machine.

### Verify Installation

```bash
~/.kiro/skills/semantic-index/scripts/.venv/bin/python -c \
  "import lancedb, tree_sitter, tiktoken; print('All dependencies OK')"
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
| `embedding.model` | `"BAAI/bge-m3"` | Embedding model |
| `embedding.dimensions` | `1024` | Vector dimensionality |
| `embedding.batch_size` | `50` | Texts per API call |
| `embedding.device` | `null` | Device for local inference: `null` (auto), `"cpu"`, `"cuda"`, `"mps"` |
| `chunking.max_tokens` | `512` | Max chunk size in tokens |
| `chunking.overlap_tokens` | `50` | Overlap between adjacent chunks |
| `chunking.min_tokens` | `20` | Minimum chunk size (smaller chunks are discarded) |
| `indexing.file_extensions` | See config | Which file types to index |
| `indexing.exclude_patterns` | See config | Patterns to skip (in addition to `.gitignore`) |
| `indexing.max_file_size_kb` | `500` | Skip files larger than this |
| `search.default_top_k` | `10` | Default number of search results |
| `search.default_threshold` | `0.3` | Minimum similarity score (0.0–1.0) |
| `search.mode` | `"hybrid"` | Search mode: `"vector"`, `"keyword"`, or `"hybrid"` |
| `search.hybrid_alpha` | `0.7` | Balance: 0.0 = pure keyword, 1.0 = pure vector |
| `search.rerank_enabled` | `false` | Enable cross-encoder reranking (requires HuggingFace deps) |
| `search.rerank_model` | `"BAAI/bge-reranker-v2-m3"` | Cross-encoder model for reranking |
| `search.rerank_top_n` | `10` | Number of results to re-rank |

### Environment Variable Overrides

Environment variables take precedence over config file values:

| Variable | Overrides |
|----------|-----------|
| `OPENROUTER_API_KEY` | `embedding.api_key` |
| `SEMANTIC_INDEX_PROVIDER` | `embedding.provider` |
| `SEMANTIC_INDEX_MODEL` | `embedding.model` |
| `SEMANTIC_INDEX_DIMENSIONS` | `embedding.dimensions` |

### .indexignore

Create a `.indexignore` file in your project root to exclude additional paths (same syntax as `.gitignore`):

```
tests/fixtures/
**/generated/
data/large-dataset.json
```

### Local Embedding (HuggingFace Provider)

Instead of calling the OpenRouter API, you can run embeddings locally using sentence-transformers. This is free, works offline, and keeps all data on your machine.

**Install HuggingFace dependencies:**

```bash
cd ~/.kiro/skills/semantic-index/scripts
bash setup.sh --with-huggingface
```

This installs `sentence-transformers` and `torch` (~2-4 GB) into the skill's venv. The embedding model (~1.1 GB for BGE-M3) is downloaded on first use to `~/.cache/huggingface/hub`.

**Switch to local provider** in `.index/config.json`:

```json
{
  "embedding": {
    "provider": "huggingface",
    "model": "BAAI/bge-m3",
    "dimensions": 1024,
    "device": null
  }
}
```

Or via environment variable:

```bash
export SEMANTIC_INDEX_PROVIDER=huggingface
```

The `device` field controls where inference runs: `null` (auto-detect: CUDA > MPS > CPU), `"cpu"`, `"cuda"`, or `"mps"`.

**Cross-provider compatibility:** Indexes built with OpenRouter can be searched with HuggingFace and vice versa, as long as the model and dimensions match. The vectors are identical.

### Reranking (Cross-Encoder)

For higher-quality search results, enable cross-encoder reranking. After the initial retrieval (vector/keyword/hybrid), a cross-encoder model re-scores each result against the query for more accurate relevance ranking.

Reranking requires the HuggingFace dependencies (`bash setup.sh --with-huggingface`).

**Enable via config** (`.index/config.json`):

```json
{
  "search": {
    "rerank_enabled": true,
    "rerank_model": "BAAI/bge-reranker-v2-m3",
    "rerank_top_n": 10
  }
}
```

**Or via CLI flag:**

```bash
$SKILL/scripts/.venv/bin/python $SKILL/scripts/semantic_search.py \
  --project-dir $PROJECT \
  --query "authentication flow" \
  --rerank
```

Use `--no-rerank` to disable reranking for a single query even when enabled in config.

## Usage

All commands output structured JSON to stdout. Progress and logs go to stderr.

The examples below use:
- `SKILL=~/.kiro/skills/semantic-index` — where the skill is installed
- `PROJECT=~/my-project` — the project you want to index

Set these for convenience, or substitute your own paths:

```bash
SKILL=~/.kiro/skills/semantic-index
PROJECT=~/my-project
```

### Build the Index

```bash
# Index a project (incremental — only new/changed files)
$SKILL/scripts/.venv/bin/python $SKILL/scripts/build_index.py \
  --project-dir $PROJECT

# Force full re-index (ignore manifest, re-embed everything)
$SKILL/scripts/.venv/bin/python $SKILL/scripts/build_index.py \
  --project-dir $PROJECT \
  --full

# Use a custom config file
$SKILL/scripts/.venv/bin/python $SKILL/scripts/build_index.py \
  --project-dir $PROJECT \
  --config /path/to/custom-config.json

# Control batch size for large repos (default: 50 files per batch)
$SKILL/scripts/.venv/bin/python $SKILL/scripts/build_index.py \
  --project-dir $PROJECT \
  --batch-size 25
```

Files are processed in batches (default 50 files) to keep memory usage bounded. Each batch is chunked, embedded, and committed to the store before the next batch starts. This makes indexing viable for large monorepos (10K+ files) without OOM risk. Use `--batch-size` to tune the tradeoff between memory usage and commit overhead.

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
$SKILL/scripts/.venv/bin/python $SKILL/scripts/semantic_search.py \
  --project-dir $PROJECT \
  --query "how does authentication work?"

# Limit results and set minimum score
$SKILL/scripts/.venv/bin/python $SKILL/scripts/semantic_search.py \
  --project-dir $PROJECT \
  --query "error handling patterns" \
  --top-k 5 \
  --threshold 0.5

# Filter by language
$SKILL/scripts/.venv/bin/python $SKILL/scripts/semantic_search.py \
  --project-dir $PROJECT \
  --query "database connection setup" \
  --filter-lang python

# Filter by file path glob
$SKILL/scripts/.venv/bin/python $SKILL/scripts/semantic_search.py \
  --project-dir $PROJECT \
  --query "API route definitions" \
  --filter-path "src/**"

# Search modes: vector-only, keyword-only, or hybrid (default)
$SKILL/scripts/.venv/bin/python $SKILL/scripts/semantic_search.py \
  --project-dir $PROJECT \
  --query "database connection" \
  --mode keyword

# Hybrid search with custom alpha (0.0 = pure keyword, 1.0 = pure vector)
$SKILL/scripts/.venv/bin/python $SKILL/scripts/semantic_search.py \
  --project-dir $PROJECT \
  --query "error handling" \
  --mode hybrid \
  --alpha 0.5

# Re-rank results with a cross-encoder for higher accuracy
$SKILL/scripts/.venv/bin/python $SKILL/scripts/semantic_search.py \
  --project-dir $PROJECT \
  --query "payment processing flow" \
  --rerank
```

Output:
```json
{
  "query": "how does authentication work?",
  "mode": "hybrid",
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
$SKILL/scripts/.venv/bin/python $SKILL/scripts/index_status.py \
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

## Using with AI Assistants

This skill is designed to be used by AI assistants, not just from the terminal. The AI reads `SKILL.md`, understands when and how to run the scripts, and uses the JSON output to answer your questions.

### Kiro

Once the skill is installed at `~/.kiro/skills/semantic-index/`, Kiro automatically discovers it. You can then ask naturally:

- "Index this project for semantic search"
- "Search the codebase for how authentication works"
- "Where is error handling implemented?"
- "Find code related to database migrations"
- "Check if the index is up to date"

Kiro will run the appropriate script, parse the JSON output, and present the results in context. If the index doesn't exist yet, it will suggest creating one first.

### Claude Code

Add the skill to your Claude Code project by placing the `semantic-index/` directory in your project's skill path or referencing it globally. Then use prompts like:

```
Use the semantic-index skill to index this project, then search for
"how does the payment processing flow work?"
```

```
Run index_status.py to check if the index is stale, then re-index
if needed and search for "where are API routes defined?"
```

```
Search the semantic index for code related to "user session management"
and show me the top 5 results with their file paths.
```

### Generic SKILL-Compatible Tools

Any tool that supports the SKILL protocol can use this skill. The AI assistant:

1. Reads `SKILL.md` for instructions on when and how to use the scripts
2. Runs `setup.sh` on first use to install dependencies
3. Calls `build_index.py` to create/update the index
4. Calls `semantic_search.py` with natural language queries
5. Calls `index_status.py` to check index health

All scripts communicate via JSON on stdout, making them easy for any AI to parse and act on.

### Example Workflow in Chat

A typical conversation might look like:

> **You:** "I need to understand how this project handles caching"
>
> **AI:** *runs `index_status.py` → index exists but 12 files are stale*
> *runs `build_index.py` → re-indexes 12 changed files*
> *runs `semantic_search.py --query "caching strategy and cache invalidation"`*
>
> "Based on the semantic search, caching is handled in three places:
> 1. `src/cache/redis_client.py` — Redis connection and basic get/set operations
> 2. `src/middleware/cache.py` — HTTP response caching middleware
> 3. `src/services/user_cache.py` — User session cache with TTL..."

The AI uses the search results as context to give you a grounded, accurate answer with specific file references.

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

AST-aware chunking via Tree-sitter is available for 10 languages:

| Language | Extensions | Key constructs extracted |
|----------|-----------|------------------------|
| Python | `.py` | functions, classes, methods |
| JavaScript | `.js`, `.jsx` | functions, classes, arrow functions |
| TypeScript | `.ts`, `.tsx` | functions, classes, interfaces |
| Go | `.go` | functions, types, method declarations |
| Rust | `.rs` | functions, structs, impl blocks |
| Java | `.java` | classes, methods, interfaces |
| C | `.c`, `.h` | functions, structs, typedefs |
| C++ | `.cpp`, `.hpp` | functions, classes, namespaces |
| Ruby | `.rb` | modules, classes, methods |
| PHP | `.php` | classes, functions, methods |

DITA XML documentation is also supported with XML-aware chunking:

| Language | Extensions | Key constructs extracted |
|----------|-----------|------------------------|
| DITA | `.dita` | topics (concept, task, reference, glossentry, troubleshooting), prolog metadata, sections |
| DITA Map | `.ditamap` | navigation hierarchy, topicref structure |

Markdown files (`.md`, `.mdx`) use header-based chunking. All other text files matching configured extensions fall back to blank-line splitting.

## Project Structure

```
semantic-index/
├── SKILL.md                    # AI-facing instructions
├── assets/
│   └── default-config.json     # Default configuration template
├── references/
│   ├── supported-languages.md  # Tree-sitter grammar list & extensions
│   └── embedding-models.md     # Model comparison guide
└── scripts/
    ├── setup.sh                # One-command environment setup
    ├── requirements.txt        # Core Python dependencies
    ├── requirements-huggingface.txt  # Optional: local embedding deps
    ├── build_index.py          # CLI: build/rebuild the index
    ├── semantic_search.py      # CLI: search by meaning
    ├── index_status.py         # CLI: index health check
    ├── migrate_config.py       # CLI: migrate config to latest schema
    └── lib/
        ├── __init__.py
        ├── models.py           # Data classes and exceptions
        ├── config.py           # Configuration loader
        ├── hasher.py           # File change detection (SHA-256)
        ├── chunker.py          # Chunking dispatch + fallback
        ├── chunkers/
        │   ├── __init__.py
        │   ├── common.py       # Shared chunking utilities
        │   ├── code.py         # Tree-sitter AST-aware code chunking
        │   ├── dita.py         # XML-aware DITA documentation chunking
        │   └── markdown.py     # Header-based markdown chunking
        ├── embedder.py         # EmbeddingProvider ABC, factory, cache
        ├── providers/
        │   ├── __init__.py     # Provider registry
        │   ├── openrouter.py   # OpenRouter REST API provider
        │   └── huggingface.py  # Local sentence-transformers provider
        ├── reranker.py         # Cross-encoder reranker
        ├── store.py            # LanceDB vector store wrapper
        ├── bm25.py             # BM25 keyword index
        └── fusion.py           # Reciprocal Rank Fusion (RRF)
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
