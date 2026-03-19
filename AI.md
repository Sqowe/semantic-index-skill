# AI Guidelines for Python Project

This file defines how AI assistants should generate or modify code in this repository.  
The goal: keep output consistent, production-ready, and aligned with our coding standards.

---

## General Rules
- Always use **PEP8 style** (formatting, naming, imports).
- Add **type hints** to all functions and classes.
- Include **docstrings** for functions, classes, and modules (Google or NumPy style).
- Write tests for new functionality (pytest).
- Prefer **f-strings** for string formatting.
- Handle exceptions with clear messages, no silent failures.

---

## Project Structure
- Do not generate large monolithic files.
- Keep modules **under ~800 lines** for readability and maintainability.
- Organize code into `src/`, `tests/`, `scripts/` when possible.
- **Exception**: SKILL projects use `scripts/lib/` as the package root (see `AI_SKILL.md` for details).
- Configurations must be stored in `.env` or `config.yaml`, **never hard-coded**.
- Keep test files only in tests/ directory
- Keep **ALL** documentation files, except README.md, in separated docs/ directory
- **Exception**: SKILL projects keep `references/` inside the skill directory, not in `docs/`.

---

## Environment Variables & Configuration
- **ALWAYS use `python-dotenv` when `.env` files are used for configuration**
- **Exception**: Projects using JSON-based config (e.g., SKILL projects with `.index/config.json`) read env vars directly via `os.environ.get()`. See `AI_SKILL.md`.
- Python does NOT automatically load `.env` files - you must explicitly load them
- Add `python-dotenv>=1.0.0` to `requirements.txt`
- Load `.env` at application entry point:
  ```python
  from dotenv import load_dotenv
  from pathlib import Path

  # Load .env file
  env_path = Path(__file__).parent / '.env'
  if env_path.exists():
      load_dotenv(env_path)
  ```
- Provide sensible defaults that work for development (e.g., SQLite paths)
- Auto-detect environment (Docker vs local dev) when paths differ
- Document all environment variables in `.env.example` with comments

---

## Dependencies
- Use `requirements.txt` or `pyproject.toml` for dependencies.
- Prefer standard library when possible before adding new libraries.
- If adding a library, explain why it's needed.
- **Required dependencies for common patterns:**
  - `.env` files → `python-dotenv`
  - Async HTTP → `aiohttp` or `fastapi`
  - WebSockets → `websockets` or `aiohttp`
  - Database migrations → `alembic` (for SQLAlchemy)

---

## Error Handling
- Centralize error handling where possible.
- Use custom exceptions (`class ProjectError(Exception):`) instead of raw `Exception`.
- Log errors using `logging`, not `print`.

---

## Common Pitfalls to Avoid

### 1. Environment Variable Loading
❌ **Bad**: Assuming `.env` files are automatically loaded
```python
# This won't work - .env is ignored!
config = os.environ.get('API_KEY')
```

✅ **Good**: Explicitly load `.env` files
```python
from dotenv import load_dotenv
load_dotenv()
config = os.environ.get('API_KEY')
```

### 2. Path Handling (Docker vs Local)
❌ **Bad**: Hard-coded paths that only work in Docker
```python
database_path = '/data/db.sqlite'  # Fails locally!
```

✅ **Good**: Auto-detect environment
```python
def get_db_path() -> str:
    if os.path.exists('/data') and os.access('/data', os.W_OK):
        return '/data/db.sqlite'  # Docker
    return './data/db.sqlite'  # Local dev
```

### 3. Import Structure
❌ **Bad**: Absolute imports in a package
```python
from config import Config  # Breaks when run as module
```

✅ **Good**: Relative imports within packages
```python
from .config import Config  # Works with python -m
```

### 4. Testing Configuration
- Always test with missing/invalid environment variables
- Provide helpful error messages for missing required config
- Never use production credentials as defaults

---

## Examples
✅ Good:
```python
def load_data(path: str) -> pd.DataFrame:
    """Load CSV file into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)
```

❌ Bad:
```python
def loaddata(p):
    return pd.read_csv(p)  # no error handling, unclear naming
```