# AI Guidelines for SKILL Projects

This file defines coding conventions specific to building SKILLs — portable AI tool packages
that are executed via CLI scripts by an AI assistant.

---

## SKILL Project Structure

This project does NOT follow the standard `src/` layout. The canonical structure is:

```
semantic-index/
├── SKILL.md                    # AI-facing instructions (required)
├── scripts/
│   ├── setup.sh                # One-command environment setup
│   ├── build_index.py          # CLI entry point: build/rebuild index
│   ├── semantic_search.py      # CLI entry point: search by meaning
│   ├── index_status.py         # CLI entry point: index health check
│   └── lib/                    # Python package (internal modules)
│       ├── __init__.py
│       ├── models.py
│       ├── config.py
│       ├── hasher.py
│       ├── chunker.py
│       ├── embedder.py
│       └── store.py
├── references/                 # On-demand docs for the AI
└── assets/                     # Templates, default configs
```

- `scripts/lib/` is the Python package root — use relative imports within it
- `references/` lives inside the skill directory, not in a top-level `docs/`
- Keep each module under **300 lines** (stricter than the general 800-line rule in AI.md)

---

## CLI Script Conventions

### Output Format

All CLI scripts MUST output structured JSON to **stdout**. This is how the AI consumes results.

```python
# ✅ Good: JSON to stdout
import json
print(json.dumps({"status": "success", "files_indexed": 42}, indent=2))

# ❌ Bad: unstructured text to stdout
print("Indexed 42 files successfully!")
```

### Logging and Progress

Use **stderr** for human-readable progress, warnings, and debug info. Never mix logs into stdout.

```python
import sys
import logging

# Configure logging to stderr
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

# Progress goes to stderr
logger.info("Processing file: %s", file_path)
print("Processing...", file=sys.stderr)

# Results go to stdout as JSON
print(json.dumps(result))
```

### Exit Codes

Use consistent exit codes across all CLI scripts:
- `0` — Success
- `1` — Configuration error (missing API key, invalid config, bad arguments)
- `2` — Runtime error (API failure, parse error, I/O error)

### Argument Parsing

Use `argparse` for all CLI scripts. Always include:
- `--project-dir` (required) — path to the project being indexed/searched
- Sensible defaults from config for optional parameters
- `--help` with clear descriptions

---

## Configuration

### No python-dotenv

This project uses **JSON-based configuration** (`.index/config.json`), NOT `.env` files.
The `python-dotenv` rule from `AI.md` does **not apply** here.

Environment variables are read directly via `os.environ.get()` and override config file values:
- `OPENROUTER_API_KEY` → overrides `config.embedding.api_key`
- `SEMANTIC_INDEX_MODEL` → overrides `config.embedding.model`
- `SEMANTIC_INDEX_DIMENSIONS` → overrides `config.embedding.dimensions`

### Config Loading Pattern

```python
import os
import json

def load_config(project_dir: str, config_path: str | None = None) -> Config:
    """Load config from JSON file, with env var overrides."""
    # 1. Load JSON config (or create defaults)
    # 2. Override with env vars where set
    # 3. Validate required fields
    # 4. Return typed Config object
```

---

## The .index/ Directory

All index artifacts live in `<project-root>/.index/`. This directory is:
- Created by `build_index.py` on first run
- Gitignored (add `.index/` to `.gitignore`)
- Fully rebuildable — safe to delete and re-index

Contents:
```
.index/
├── config.json            # User configuration
├── manifest.json          # File hash manifest (SHA-256)
├── embedding_cache.json   # Content hash → vector cache
└── lancedb/               # LanceDB database directory
```

---

## API Client Patterns (embedder.py)

### Retry with Exponential Backoff

All external API calls must implement retry logic:

```python
import time

def _call_with_retry(self, func, max_retries: int = 3, base_delay: float = 1.0):
    """Retry with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("Attempt %d failed, retrying in %.1fs: %s", attempt + 1, delay, e)
            time.sleep(delay)
```

### Batching

- Send embedding requests in batches (configurable, default 50)
- Never send more than 100 texts per API call (OpenRouter limit)
- Log batch progress to stderr

### Rate Limiting

- Respect HTTP 429 responses — read `Retry-After` header
- Back off gracefully, don't hammer the API

---

## Tree-sitter Conventions

### Grammar Loading

Load Tree-sitter grammars lazily — only when a file of that language is encountered:

```python
# ✅ Good: lazy loading
def _get_parser(self, language: str) -> tree_sitter.Parser:
    if language not in self._parsers:
        self._parsers[language] = self._create_parser(language)
    return self._parsers[language]

# ❌ Bad: loading all grammars at import time
```

### AST Node Queries

Define language-specific node types for extraction:

```python
# What counts as a "top-level definition" per language
EXTRACTABLE_NODES = {
    "python": ["function_definition", "class_definition"],
    "javascript": ["function_declaration", "class_declaration", "lexical_declaration"],
    "typescript": ["function_declaration", "class_declaration", "lexical_declaration"],
}
```

### Fallback

If Tree-sitter parsing fails (corrupt file, unsupported syntax), fall back to blank-line splitting.
Never crash on a single unparseable file — log a warning and continue.

---

## LanceDB Conventions

### Table Management

- One table per index: `chunks`
- Use PyArrow schema for type safety
- Always close/cleanup connections in error paths

### Search

- Over-fetch by 2x when filtering is applied (fetch `top_k * 2`, then filter, then truncate)
- Use cosine similarity metric
- Return scores normalized to 0.0–1.0

---

## Error Handling

### Custom Exceptions

Define project-specific exceptions in `models.py`:

```python
class SemanticIndexError(Exception):
    """Base exception for semantic-index."""

class ConfigError(SemanticIndexError):
    """Configuration is missing or invalid."""

class EmbeddingError(SemanticIndexError):
    """Embedding API call failed."""

class IndexError(SemanticIndexError):
    """Index operation failed."""
```

### Never Silent Failures

Every error must either:
1. Be logged with context and re-raised
2. Be caught, logged, and result in a non-zero exit code with JSON error output

```python
# CLI error output pattern
import json
import sys

def handle_error(error: Exception, exit_code: int = 2) -> None:
    """Output error as JSON and exit."""
    print(json.dumps({
        "status": "error",
        "error": str(error),
        "error_type": type(error).__name__,
    }), file=sys.stdout)
    sys.exit(exit_code)
```
