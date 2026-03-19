"""Data classes for the semantic index pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ChunkType(Enum):
    """Type of content chunk extracted from a source file."""

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

    id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    chunk_type: ChunkType
    language: Optional[str] = None
    symbol_name: Optional[str] = None
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single search hit with relevance score."""

    chunk: Chunk
    score: float
    rank: int


@dataclass
class IndexStats:
    """Summary of the current index state."""

    total_files: int
    total_chunks: int
    last_indexed: str
    stale_files: int
    embedding_model: str
    embedding_dimensions: int
    index_size_bytes: int


@dataclass
class FileChange:
    """Result of comparing current files against the manifest."""

    to_index: list[str] = field(default_factory=list)
    to_delete: list[str] = field(default_factory=list)
    unchanged: int = 0


class SemanticIndexError(Exception):
    """Base exception for semantic-index."""


class ConfigError(SemanticIndexError):
    """Configuration is missing or invalid."""


class EmbeddingError(SemanticIndexError):
    """Embedding API call failed."""


class IndexingError(SemanticIndexError):
    """Index operation failed."""
