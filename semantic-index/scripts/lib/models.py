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
    DITA_TOPIC = "dita_topic"
    DITA_MAP = "dita_map"
    PDF_PAGE = "pdf_page"
    DOCX_SECTION = "docx_section"
    PPTX_SLIDE = "pptx_slide"
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


@dataclass
class TruncationRecord:
    """Record of a chunk that was shortened before embedding.

    Populated by ``Embedder.embed_chunks`` whenever the pre-embed token
    cap cut a chunk's content down so it would fit the model's context
    window. The chunk's ``content`` is replaced with the truncated
    prefix (that is what gets sent to the API), but the full original
    text is preserved on ``chunk.metadata["original_content"]`` so
    search results and later inspection can still read the whole
    chunk. The embedding vector only covers the surviving prefix, so
    semantic search on this chunk will miss anything past
    ``final_tokens``.

    ``final_tokens`` is the token count of the truncated prefix as
    measured by the Embedder. Because the Embedder reserves room for
    the provider's ``document_prefix`` in its target budget, the
    provider's secondary token cap (which counts prefixed text) does
    not have to shorten further — ``final_tokens`` matches what the
    embedding API actually saw.

    Attributes:
        file_path: Project-relative path of the chunk's source file.
        chunk_id: The chunk's content-hash identifier.
        original_tokens: Tokens measured before truncation (the model's
            own tokenizer when available; tiktoken + safety factor
            otherwise).
        final_tokens: Tokens after truncation. The vector covers these
            and only these.
    """

    file_path: str
    chunk_id: str
    original_tokens: int
    final_tokens: int


class SemanticIndexError(Exception):
    """Base exception for semantic-index."""


class ConfigError(SemanticIndexError):
    """Configuration is missing or invalid."""


class EmbeddingError(SemanticIndexError):
    """Embedding API call failed."""


class IndexingError(SemanticIndexError):
    """Index operation failed."""
