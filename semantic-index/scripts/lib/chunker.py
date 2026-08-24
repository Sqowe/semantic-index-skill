"""Chunking dispatch module.

Routes files to the appropriate chunking strategy based on language:
- Markdown files: header-based splitting (chunkers/markdown.py)
- Python/JS/TS: Tree-sitter AST-aware splitting (chunkers/code.py)
- Other: blank-line fallback splitting (chunkers/common.py)

Shared helpers live in chunkers/common.py to avoid circular imports.
"""

import logging
import os
from pathlib import Path

from .chunkers.common import (
    build_chunks,
    chunk_text_fallback,
    count_tokens,
    detect_language,
)
from .config import Config
from .constants import BINARY_FORMATS
from .models import Chunk

logger = logging.getLogger(__name__)

# How far over budget a chunk must be before the safety net treats it as a
# chunker defect rather than rounding.
#
# Chunkers accumulate by summing the size of each part, but the joined text
# tokenizes to slightly more than the sum, because the separators merge with
# their neighbours. That overshoots by a few tokens and is harmless. A missing
# last-resort split is different in kind: with no boundary to cut on, the chunk
# comes out a multiple of the budget. Below the margin, split quietly; above
# it, say so.
OVERSIZE_WARN_RATIO = 1.25

# Languages supported by Tree-sitter (Phase 1 + Phase 2)
TREESITTER_LANGUAGES = {
    "python", "javascript", "typescript",
    "go", "rust", "java", "c", "cpp", "ruby", "php",
}


def _enforce_token_budget(
    chunks: list[Chunk],
    file_path: str,
    config: Config,
) -> list[Chunk]:
    """Split any chunk that still exceeds ``chunking.max_tokens``.

    Every chunker is expected to respect the budget on its own. This is a
    safety net for content none of them can divide — a minified line, a
    base64 blob, a table row wider than the whole budget — and for future
    chunkers that forget the last-resort split. An oversized chunk is not
    harmless: the embedding API rejects it, and the retry logic then
    halves the batch repeatedly to find it, costing a request per level.

    Anything caught here is logged with its file and line range, so a gap in
    a chunker is visible rather than silent — at WARNING when the chunk is
    more than ``OVERSIZE_WARN_RATIO`` over budget, at DEBUG for the few
    tokens of rounding an accumulating chunker normally produces.

    Args:
        chunks: Chunks produced by a chunking strategy.
        file_path: Project-relative path, for logging.
        config: Loaded configuration.

    Returns:
        The chunks, with oversized ones replaced by their split pieces.
    """
    max_tokens = config.chunking.max_tokens

    def tokens_of(chunk: Chunk) -> int:
        """Chunk size, recounted only if the chunker left the field unset."""
        return chunk.token_count or count_tokens(chunk.content)

    if all(tokens_of(chunk) <= max_tokens for chunk in chunks):
        return chunks

    result: list[Chunk] = []
    for chunk in chunks:
        token_count = tokens_of(chunk)
        if token_count <= max_tokens:
            result.append(chunk)
            continue
        level = (
            logging.WARNING
            if token_count > max_tokens * OVERSIZE_WARN_RATIO
            else logging.DEBUG
        )
        logger.log(
            level,
            "Chunker left an oversized chunk in %s lines %d-%d "
            "(%d tokens > %d), splitting at token boundaries",
            file_path, chunk.start_line, chunk.end_line,
            token_count, max_tokens,
        )
        result.extend(build_chunks(
            chunk.content,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            language=chunk.language,
            chunk_type=chunk.chunk_type,
            max_tokens=max_tokens,
            min_tokens=config.chunking.min_tokens,
            symbol_name=chunk.symbol_name,
            metadata=chunk.metadata,
        ))
    return result


def chunk_file(
    file_path: str,
    project_dir: str,
    config: Config,
) -> list[Chunk]:
    """Chunk a single file into semantically meaningful pieces.

    Dispatches to the appropriate chunking strategy based on language:
    - Office documents (PDF/DOCX/PPTX): binary extraction via office chunker
    - Markdown files: header-based splitting
    - Python/JS/TS: Tree-sitter AST-aware splitting
    - Other: blank-line fallback splitting

    Whatever the strategy, the result passes through
    :func:`_enforce_token_budget`, so no chunk leaves this function larger
    than ``chunking.max_tokens``.

    Args:
        file_path: Relative path to the file (from project root).
        project_dir: Absolute path to the project root.
        config: Loaded configuration.

    Returns:
        List of Chunk objects. May be empty if the file is too small.
    """
    abs_path = os.path.join(project_dir, file_path)
    language = detect_language(file_path)

    # Binary formats: delegate to office chunker (handles its own file I/O)
    if language in BINARY_FORMATS:
        from .chunkers.office import chunk_office
        return _enforce_token_budget(
            chunk_office(abs_path, file_path, language, config),
            file_path, config,
        )

    try:
        content = Path(abs_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Failed to read file %s: %s", file_path, exc)
        return []

    if not content.strip():
        return []

    if language == "markdown":
        from .chunkers.markdown import chunk_markdown
        chunks = chunk_markdown(content, file_path, config)
    elif language in ("dita", "ditamap"):
        from .chunkers.dita import chunk_dita
        chunks = chunk_dita(content, file_path, language, config)
    elif language in TREESITTER_LANGUAGES:
        from .chunkers.code import chunk_code_with_treesitter
        chunks = chunk_code_with_treesitter(content, file_path, language, config)
    else:
        chunks = chunk_text_fallback(content, file_path, language, config)

    return _enforce_token_budget(chunks, file_path, config)
