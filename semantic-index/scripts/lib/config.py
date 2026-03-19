"""Configuration loader for the semantic index.

Loads settings from .index/config.json with environment variable overrides.
Creates default config on first run if missing.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import ConfigError

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILENAME = "config.json"
INDEX_DIR_NAME = ".index"


@dataclass
class EmbeddingConfig:
    """Embedding provider settings."""

    provider: str = "openrouter"
    api_key: Optional[str] = None
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    dimensions: int = 768
    batch_size: int = 50
    query_prefix: str = "search_query: "
    document_prefix: str = "search_document: "
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class ChunkingConfig:
    """Chunking strategy settings."""

    max_tokens: int = 512
    overlap_tokens: int = 50
    min_tokens: int = 20


@dataclass
class IndexingConfig:
    """File discovery and filtering settings."""

    file_extensions: list[str] = field(default_factory=lambda: [
        ".py", ".js", ".ts", ".tsx", ".jsx",
        ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp",
        ".rb", ".php",
        ".md", ".mdx", ".txt", ".rst",
    ])
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "node_modules/", "venv/", ".venv/", "__pycache__/",
        "dist/", "build/", ".git/", ".index/",
        "*.min.js", "*.min.css", "*.map",
        "package-lock.json", "yarn.lock", "poetry.lock",
        "*.pyc", "*.pyo", "*.so", "*.dylib",
    ])
    max_file_size_kb: int = 500
    respect_gitignore: bool = True


@dataclass
class SearchConfig:
    """Search defaults."""

    default_top_k: int = 10
    default_threshold: float = 0.3


@dataclass
class Config:
    """Top-level configuration container."""

    schema_version: str = "1.0"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    _extra: dict = field(default_factory=dict)


def _get_default_config_path() -> Path:
    """Return path to the bundled default config template."""
    return Path(__file__).parent.parent.parent / "assets" / "default-config.json"


def _dict_to_embedding_config(data: dict) -> EmbeddingConfig:
    """Build EmbeddingConfig from a dict, ignoring unknown keys."""
    known = {f.name for f in EmbeddingConfig.__dataclass_fields__.values()}
    return EmbeddingConfig(**{k: v for k, v in data.items() if k in known})


def _dict_to_chunking_config(data: dict) -> ChunkingConfig:
    known = {f.name for f in ChunkingConfig.__dataclass_fields__.values()}
    return ChunkingConfig(**{k: v for k, v in data.items() if k in known})


def _dict_to_indexing_config(data: dict) -> IndexingConfig:
    known = {f.name for f in IndexingConfig.__dataclass_fields__.values()}
    return IndexingConfig(**{k: v for k, v in data.items() if k in known})


def _dict_to_search_config(data: dict) -> SearchConfig:
    known = {f.name for f in SearchConfig.__dataclass_fields__.values()}
    return SearchConfig(**{k: v for k, v in data.items() if k in known})


def _config_to_dict(config: Config) -> dict:
    """Serialize Config back to a JSON-compatible dict, preserving extra keys."""
    from dataclasses import asdict

    result = {
        "schema_version": config.schema_version,
        "embedding": asdict(config.embedding),
        "chunking": asdict(config.chunking),
        "indexing": asdict(config.indexing),
        "search": asdict(config.search),
    }
    # Preserve unknown keys for forward compatibility
    result.update(config._extra)
    return result


def _apply_env_overrides(config: Config) -> None:
    """Override config values with environment variables where set."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        config.embedding.api_key = api_key

    model = os.environ.get("SEMANTIC_INDEX_MODEL")
    if model:
        config.embedding.model = model

    dimensions = os.environ.get("SEMANTIC_INDEX_DIMENSIONS")
    if dimensions:
        try:
            config.embedding.dimensions = int(dimensions)
        except ValueError:
            logger.warning("Invalid SEMANTIC_INDEX_DIMENSIONS: %s (not an integer)", dimensions)


def load_config(project_dir: str, config_path: Optional[str] = None) -> Config:
    """Load configuration from JSON file with env var overrides.

    Args:
        project_dir: Path to the project root.
        config_path: Optional explicit path to config.json.
            Defaults to <project_dir>/.index/config.json.

    Returns:
        Validated Config object.

    Raises:
        ConfigError: If config file exists but is invalid JSON.
    """
    if config_path:
        cfg_path = Path(config_path)
    else:
        cfg_path = Path(project_dir) / INDEX_DIR_NAME / DEFAULT_CONFIG_FILENAME

    config = Config()

    if cfg_path.exists():
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ConfigError(f"Failed to read config at {cfg_path}: {exc}") from exc

        known_sections = {"schema_version", "embedding", "chunking", "indexing", "search"}
        extra = {k: v for k, v in raw.items() if k not in known_sections}

        config.schema_version = raw.get("schema_version", "1.0")
        if "embedding" in raw:
            config.embedding = _dict_to_embedding_config(raw["embedding"])
        if "chunking" in raw:
            config.chunking = _dict_to_chunking_config(raw["chunking"])
        if "indexing" in raw:
            config.indexing = _dict_to_indexing_config(raw["indexing"])
        if "search" in raw:
            config.search = _dict_to_search_config(raw["search"])
        config._extra = extra
    else:
        logger.info("No config found at %s, using defaults", cfg_path)

    _apply_env_overrides(config)
    return config


def ensure_index_dir(project_dir: str) -> Path:
    """Create the .index/ directory if it doesn't exist.

    Also writes a default config.json if none exists.

    Returns:
        Path to the .index/ directory.
    """
    index_dir = Path(project_dir) / INDEX_DIR_NAME
    index_dir.mkdir(exist_ok=True)

    cfg_path = index_dir / DEFAULT_CONFIG_FILENAME
    if not cfg_path.exists():
        default_template = _get_default_config_path()
        if default_template.exists():
            cfg_path.write_text(default_template.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("Created default config at %s", cfg_path)
        else:
            # Write from in-memory defaults
            config = Config()
            cfg_path.write_text(
                json.dumps(_config_to_dict(config), indent=2) + "\n",
                encoding="utf-8",
            )
            logger.info("Created config from defaults at %s", cfg_path)

    return index_dir
