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
from typing import Optional

from .chunkers.common import (
    build_chunks,
    chunk_text_fallback,
    count_tokens,
    detect_language,
)
from .config import Config
from .constants import BINARY_FORMATS
from .models import Chunk
from .tokenizer_resolver import (
    TokenizerWrapper,
    effective_max_tokens,
    resolver_for_config,
)

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


def _effective_chunk_max_tokens(config: Config, tokens: TokenizerWrapper) -> int:
    """Resolve the token budget that the chunker should enforce.

    When the real (per-model) tokenizer is loaded, the configured
    ``chunking.max_tokens`` is used as-is: the count matches the API's.
    When the chunker is running on the tiktoken fallback (because
    ``tokenizers`` is not installed or the model's HF repo is not
    reachable), the chunker-side budget shrinks by the safety factor so
    a chunk that fits in tiktoken units still fits in real units at the
    measured worst-case ratio.

    Returns:
        The integer token budget used for chunking.
    """
    if tokens.kind == "real":
        return config.chunking.max_tokens
    shrunk = int(config.chunking.max_tokens / config.embedding.token_safety_factor)
    # Never go below 1; also never inflate the user-configured budget.
    return max(1, min(config.chunking.max_tokens, shrunk))


def _enforce_token_budget(
    chunks: list[Chunk],
    file_path: str,
    config: Config,
    tokens: Optional[TokenizerWrapper] = None,
    effective_max: Optional[int] = None,
) -> list[Chunk]:
    """Split any chunk that still exceeds the effective chunk budget.

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
        tokens: Tokenizer wrapper used to measure overshoot.
        effective_max: The maximum tokens a chunk should have. This is
            the safety-factor-adjusted budget computed by
            :func:`_effective_chunk_max_tokens`, not necessarily
            ``chunking.max_tokens``.

    Returns:
        The chunks, with oversized ones replaced by their split pieces.
    """
    if tokens is None:
        tokens = resolver_for_config(config)
    if effective_max is None:
        effective_max = _effective_chunk_max_tokens(config, tokens)

    def tokens_of(chunk: Chunk) -> int:
        """Chunk size, recounted only if the chunker left the field unset."""
        return chunk.token_count or count_tokens(chunk.content, tokens=tokens)

    if all(tokens_of(chunk) <= effective_max for chunk in chunks):
        return chunks

    result: list[Chunk] = []
    for chunk in chunks:
        token_count = tokens_of(chunk)
        # SentencePiece-style tokenizers can produce a piece whose
        # re-encoded count slightly exceeds the requested window because
        # the leading-space marker token is not reproduced on re-encode
        # (a 60-token slice may re-encode to 62). A few tokens of
        # overshoot is harmless and within the rounding margin that
        # accumulating chunkers normally produce; only treat chunks as
        # truly oversized when they exceed OVERSIZE_WARN_RATIO of the
        # budget.
        if token_count <= effective_max:
            result.append(chunk)
            continue
        if token_count <= effective_max * OVERSIZE_WARN_RATIO:
            # Within the rounding margin; keep the chunk as-is.
            logger.debug(
                "Chunker piece in %s lines %d-%d lands %d tokens, "
                "slightly over the %d-token budget; keeping as-is",
                file_path, chunk.start_line, chunk.end_line,
                token_count, effective_max,
            )
            result.append(chunk)
            continue
        # Genuinely oversized. Anything above the single OVERSIZE_WARN_RATIO
        # threshold is a WARNING; the safety net will re-split it.
        logger.warning(
            "Chunker left an oversized chunk in %s lines %d-%d "
            "(%d tokens > %d), splitting at token boundaries",
            file_path, chunk.start_line, chunk.end_line,
            token_count, effective_max,
        )
        result.extend(build_chunks(
            chunk.content,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            language=chunk.language,
            chunk_type=chunk.chunk_type,
            max_tokens=effective_max,
            min_tokens=config.chunking.min_tokens,
            symbol_name=chunk.symbol_name,
            metadata=chunk.metadata,
            tokens=tokens,
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
    than the effective budget — the configured ``chunking.max_tokens``
    when the real per-model tokenizer is loaded, or
    ``chunking.max_tokens / token_safety_factor`` when the chunker is
    counting with the tiktoken fallback (because ``tokenizers`` is not
    installed, the model's HF repo is not reachable, or the user picked
    a model without one).

    Args:
        file_path: Relative path to the file (from project root).
        project_dir: Absolute path to the project root.
        config: Loaded configuration.

    Returns:
        List of Chunk objects. May be empty if the file is too small.
    """
    abs_path = os.path.join(project_dir, file_path)
    language = detect_language(file_path)
    tokens = resolver_for_config(config)
    effective_max = _effective_chunk_max_tokens(config, tokens)

    # Binary formats: delegate to office chunker (handles its own file I/O)
    if language in BINARY_FORMATS:
        from .chunkers.office import chunk_office
        return _enforce_token_budget(
            chunk_office(abs_path, file_path, language, config, tokens, effective_max),
            file_path, config, tokens, effective_max,
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
        chunks = chunk_markdown(content, file_path, config, tokens, effective_max)
    elif language in ("dita", "ditamap"):
        from .chunkers.dita import chunk_dita
        chunks = chunk_dita(content, file_path, language, config, tokens, effective_max)
    elif language in TREESITTER_LANGUAGES:
        from .chunkers.code import chunk_code_with_treesitter
        chunks = chunk_code_with_treesitter(
            content, file_path, language, config, tokens, effective_max,
        )
    else:
        chunks = chunk_text_fallback(
            content, file_path, language, config,
            tokens=tokens, effective_max=effective_max,
        )

    return _enforce_token_budget(
        chunks, file_path, config, tokens, effective_max,
    )
