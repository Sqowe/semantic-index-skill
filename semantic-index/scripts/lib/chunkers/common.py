"""Shared utilities for all chunking strategies.

This module is the single source of truth for helpers used across
chunker dispatch (lib/chunker.py) and strategy submodules
(chunkers/code.py, chunkers/markdown.py). No circular imports —
this module only depends on lib/config, lib/models, and
lib/tokenizer_resolver.
"""

import hashlib
import logging
import os
import re
from typing import Optional

import tiktoken

from ..config import Config
from ..models import Chunk, ChunkType
from ..tokenizer_resolver import TokenizerWrapper, get_resolver

logger = logging.getLogger(__name__)

# Re-export from the lightweight constants module so existing chunker
# code that imports from common.py keeps working.
from ..constants import BINARY_FORMATS, OFFICE_EXTENSIONS  # noqa: E402,F401

# Lazy-loaded fallback tokenizer (tiktoken cl100k_base). Kept for callers
# that still rely on ``get_tokenizer()`` directly (notably the DITA
# chunker's character-level last resort) and for the default ``tokens``
# argument used by every helper in this module.
_tokenizer: Optional[tiktoken.Encoding] = None


def get_tokenizer() -> tiktoken.Encoding:
    """Get or create the tiktoken tokenizer (cl100k_base).

    Prefer ``get_resolver(config.embedding.model)`` for new code so the
    tokenizer matches the active embedding model. This is the tiktoken
    fallback used when no real tokenizer is loaded.
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _default_tokens() -> TokenizerWrapper:
    """Return a resolver-backed tokenizer wrapping tiktoken.

    Internal helper so the public ``count_tokens`` /
    ``hard_split_by_tokens`` / ``split_text_with_lines`` / ``build_chunks``
    signatures stay parameterless for callers that have not been
    converted to pass an explicit resolver.
    """
    # The "tiktoken" wrapper keeps the existing semantics for legacy
    # callers: cl100k counts, no safety factor applied (those callers
    # already pass their own budget that is sized to cl100k).
    return get_resolver("cl100k_base")


def count_tokens(text: str, tokens: Optional[TokenizerWrapper] = None) -> int:
    """Count tokens in *text* using the provided or default tokenizer.

    Args:
        text: Text to count tokens for.
        tokens: Tokenizer wrapper to use. Defaults to a tiktoken
            ``cl100k_base`` wrapper, which matches the historical
            behaviour of every chunker before the embedding-model
            tokenizer was introduced.
    """
    wrapper = tokens if tokens is not None else _default_tokens()
    return len(wrapper.encode(text))


def hard_split_by_tokens(
    text: str,
    max_tokens: int,
    tokens: Optional[TokenizerWrapper] = None,
) -> list[str]:
    """Split *text* into consecutive pieces of at most *max_tokens* tokens.

    This is the last-resort splitter, used when no structural boundary is
    available: a minified source line, a base64 blob, a long data literal,
    or a file written without blank lines. Without it such content becomes
    a single chunk far larger than the configured budget, which the
    embedding API then rejects.

    Splitting happens at exact token boundaries. Two failure modes have
    to be handled:

    * Mid-byte corruption (BPE-style tokenizers like tiktoken). Decoding
      a slice that ends mid-byte produces a U+FFFD replacement
      character; a round-trip check (``encode(decode(slice)) == slice``)
      catches it and shrinks the window.

    * Drift across consecutive windows (SentencePiece-style tokenizers
      like BAAI/bge-m3). The leading-space marker token that SentencePiece
      prepends on the *first* encode is not reproduced when a decoded
      substring is re-encoded, so successive slices systematically fail
      the round-trip check. The check is therefore skipped for real
      tokenizers — they never split mid-character, so corruption is not
      the concern — and the produced pieces are *approximate*: a 60-token
      slice may re-encode to 61 or 62 tokens. ``_enforce_token_budget``
      catches and re-splits any such over-budget pieces, so the net
      guarantee still holds.

    Args:
        text: The text to split.
        max_tokens: Maximum tokens per piece. Values below 1 are raised to 1.
        tokens: Tokenizer wrapper to use. Defaults to a tiktoken
            ``cl100k_base`` wrapper.

    Returns:
        The pieces in order. A text already within the budget is returned
        unchanged as a single-element list.
    """
    if max_tokens < 1:
        max_tokens = 1

    wrapper = tokens if tokens is not None else _default_tokens()
    token_ids = wrapper.encode(text)
    if len(token_ids) <= max_tokens:
        return [text]

    # Only BPE-style tokenizers (tiktoken) need the round-trip check.
    # SentencePiece-style real tokenizers are safe and the check would
    # make every window shrink to almost nothing.
    do_round_trip = wrapper.kind != "real"

    pieces: list[str] = []
    i = 0
    while i < len(token_ids):
        end = min(i + max_tokens, len(token_ids))
        token_slice = token_ids[i:end]
        decoded = wrapper.decode(token_slice)

        if do_round_trip and wrapper.encode(decoded) != token_slice:
            # Shrink the window until the round trip is clean.
            while end > i + 1:
                end -= 1
                token_slice = token_ids[i:end]
                decoded = wrapper.decode(token_slice)
                if wrapper.encode(decoded) == token_slice:
                    break

        if decoded:
            pieces.append(decoded)
        i = end

    return pieces if pieces else [text]


def split_text_with_lines(
    text: str,
    start_line: int,
    max_tokens: int,
    tokens: Optional[TokenizerWrapper] = None,
) -> list[tuple[str, int, int, int]]:
    """Hard-split *text* and report where every piece sits.

    Wraps :func:`hard_split_by_tokens` and tracks how many newlines each
    piece consumes, so callers can build chunks with correct line numbers.
    A split does not have to land on a line boundary, so a piece may start
    on the same line where the previous one ended — the reported column
    tells them apart, which is what keeps their chunk IDs distinct.

    Args:
        text: The text to split.
        start_line: 1-based line number where *text* begins in its file.
        max_tokens: Maximum tokens per piece.
        tokens: Tokenizer wrapper to use. Defaults to a tiktoken
            ``cl100k_base`` wrapper.

    Returns:
        ``(piece, start_line, end_line, start_column)`` tuples in order,
        where the column is a 0-based character offset within
        ``start_line``.
    """
    result: list[tuple[str, int, int, int]] = []
    line = start_line
    column = 0
    for piece in hard_split_by_tokens(text, max_tokens, tokens=tokens):
        end_line = line + piece.count("\n")
        result.append((piece, line, end_line, column))
        if "\n" in piece:
            column = len(piece) - piece.rindex("\n") - 1
        else:
            column += len(piece)
        line = end_line
    return result


def build_chunks(
    text: str,
    *,
    file_path: str,
    start_line: int,
    language: Optional[str],
    chunk_type: ChunkType,
    max_tokens: int,
    min_tokens: int,
    symbol_name: Optional[str] = None,
    metadata: Optional[dict] = None,
    tokens: Optional[TokenizerWrapper] = None,
) -> list[Chunk]:
    """Turn one accumulated piece of text into chunks that fit the budget.

    Chunkers accumulate text up to *max_tokens* at structural boundaries
    (blank lines, paragraphs, source lines), but a single indivisible unit
    can already be larger than the whole budget. Routing every emission
    through this function guarantees no chunk leaves the chunker over
    *max_tokens*, which is what the embedding API enforces.

    Text at or under the budget produces one chunk. Oversized text is cut
    at token boundaries by :func:`split_text_with_lines`. Pieces below
    *min_tokens* are dropped, matching the behaviour elsewhere.

    Args:
        text: The chunk text.
        file_path: Project-relative path, used for the chunk ID.
        start_line: 1-based line number where *text* begins.
        language: Detected language, or None.
        chunk_type: Type recorded on every produced chunk.
        max_tokens: Maximum tokens per chunk.
        min_tokens: Pieces below this are dropped.
        symbol_name: Symbol the text belongs to, or None.
        metadata: Copied onto each chunk. Defaults to an empty dict.
        tokens: Tokenizer wrapper to use. Defaults to a tiktoken
            ``cl100k_base`` wrapper. Thread the resolver for the active
            embedding model so chunks are budgeted in the same units
            the API will count.

    Returns:
        Zero or more chunks, in order.
    """
    token_count = count_tokens(text, tokens=tokens)
    if token_count <= max_tokens:
        pieces = [(text, start_line, start_line + text.count("\n"), 0)]
    else:
        logger.debug(
            "Hard-splitting oversized %s chunk at %s:%d (%d tokens > %d)",
            chunk_type.value, file_path, start_line, token_count, max_tokens,
        )
        pieces = split_text_with_lines(text, start_line, max_tokens, tokens=tokens)

    base_metadata = metadata or {}
    chunks: list[Chunk] = []
    for piece, piece_start, piece_end, piece_column in pieces:
        piece_tokens = count_tokens(piece, tokens=tokens)
        if piece_tokens < min_tokens:
            continue
        chunks.append(Chunk(
            id=make_chunk_id(file_path, piece, piece_start, piece_column),
            file_path=file_path,
            start_line=piece_start,
            end_line=piece_end,
            content=piece,
            chunk_type=chunk_type,
            language=language,
            symbol_name=symbol_name,
            token_count=piece_tokens,
            metadata=base_metadata.copy(),
        ))
    return chunks


def make_chunk_id(
    file_path: str,
    content: str,
    start_line: int = 0,
    offset: int = 0,
) -> str:
    """Generate a deterministic chunk ID from file path, content, and position.

    Including start_line prevents ID collisions when identical content
    appears multiple times in the same file. That is not enough when a
    single long line is hard-split: every piece then starts on the same
    line, and repetitive content (a data literal, a base64 blob) produces
    identical pieces. *offset* — the character position of the piece
    within its line — separates them.

    The offset only enters the hash when it is non-zero, so IDs for
    content split at line boundaries stay the same as before this
    parameter existed and existing indexes need no rewrite.

    Args:
        file_path: Project-relative path.
        content: The chunk text.
        start_line: 1-based line where the chunk begins.
        offset: Character position within *start_line*, for pieces of a
            hard-split line.

    Returns:
        A ``sha256:<hex>`` identifier.
    """
    position = f"{start_line}" if offset == 0 else f"{start_line}+{offset}"
    raw = f"{file_path}:{position}:{content}"
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def detect_language(file_path: str) -> Optional[str]:
    """Detect language from file extension."""
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
        ".md": "markdown",
        ".mdx": "markdown",
        ".txt": "text",
        ".rst": "rst",
        ".dita": "dita",
        ".ditamap": "ditamap",
        ".pdf": "pdf",
        ".docx": "docx",
        ".pptx": "pptx",
    }
    ext = os.path.splitext(file_path)[1].lower()
    return ext_map.get(ext)


def chunk_text_fallback(
    content: str,
    file_path: str,
    language: Optional[str],
    config: Config,
    tokens: Optional[TokenizerWrapper] = None,
    effective_max: Optional[int] = None,
) -> list[Chunk]:
    """Split text at blank-line boundaries for unsupported languages.

    Uses offset-based line tracking for accurate line numbers.

    Args:
        content: The file's text content.
        file_path: Project-relative path.
        language: Detected language, or None.
        config: Loaded configuration.
        tokens: Tokenizer wrapper to use. Defaults to a tiktoken
            ``cl100k_base`` wrapper. Thread the resolver for the active
            embedding model so chunk budgets match the API's count.
        effective_max: Safety-factor-adjusted chunk budget. When None,
            ``chunking.max_tokens`` is used (legacy behaviour).
    """
    max_tokens = effective_max if effective_max is not None else config.chunking.max_tokens
    min_tokens = config.chunking.min_tokens

    # Find paragraphs with their character offsets
    gap_pattern = re.compile(r"\n\n+")
    blocks_with_offsets: list[tuple[str, int]] = []
    prev_end = 0
    for m in gap_pattern.finditer(content):
        block = content[prev_end:m.start()]
        if block.strip():
            blocks_with_offsets.append((block, prev_end))
        prev_end = m.end()
    if prev_end < len(content):
        trailing = content[prev_end:]
        if trailing.strip():
            blocks_with_offsets.append((trailing, prev_end))

    if not blocks_with_offsets:
        return []

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    current_start_offset = blocks_with_offsets[0][1]

    def emit(chunk_text: str, start_offset: int) -> None:
        """Append chunks for one accumulated text, hard-splitting if oversized."""
        s_line = 1 + content[:start_offset].count("\n")
        chunks.extend(build_chunks(
            chunk_text,
            file_path=file_path,
            start_line=s_line,
            language=language,
            chunk_type=ChunkType.UNKNOWN,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            tokens=tokens,
        ))

    for block, offset in blocks_with_offsets:
        block_tokens = count_tokens(block, tokens=tokens)

        if current_tokens + block_tokens > max_tokens and current_parts:
            emit("\n\n".join(current_parts), current_start_offset)
            current_start_offset = offset
            current_parts = []
            current_tokens = 0

        current_parts.append(block)
        current_tokens += block_tokens

    if current_parts:
        emit("\n\n".join(current_parts), current_start_offset)

    return chunks
