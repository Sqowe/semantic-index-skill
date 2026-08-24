"""Shared utilities for all chunking strategies.

This module is the single source of truth for helpers used across
chunker dispatch (lib/chunker.py) and strategy submodules
(chunkers/code.py, chunkers/markdown.py). No circular imports —
this module only depends on lib/config and lib/models.
"""

import hashlib
import logging
import os
import re
from typing import Optional

import tiktoken

from ..config import Config
from ..models import Chunk, ChunkType

logger = logging.getLogger(__name__)

# Re-export from the lightweight constants module so existing chunker
# code that imports from common.py keeps working.
from ..constants import BINARY_FORMATS, OFFICE_EXTENSIONS  # noqa: E402,F401

# Lazy-loaded tokenizer
_tokenizer: Optional[tiktoken.Encoding] = None


def get_tokenizer() -> tiktoken.Encoding:
    """Get or create the tiktoken tokenizer (cl100k_base)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(get_tokenizer().encode(text))


def hard_split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Split *text* into consecutive pieces of at most *max_tokens* tokens.

    This is the last-resort splitter, used when no structural boundary is
    available: a minified source line, a base64 blob, a long data literal,
    or a file written without blank lines. Without it such content becomes
    a single chunk far larger than the configured budget, which the
    embedding API then rejects.

    Splitting happens at exact tiktoken boundaries. A round-trip check
    guards against corruption: when a token slice does not re-encode to
    itself — possible if the slice ends mid-character — the window shrinks
    until the round trip is clean.

    Args:
        text: The text to split.
        max_tokens: Maximum tokens per piece. Values below 1 are raised to 1.

    Returns:
        The pieces in order. A text already within the budget is returned
        unchanged as a single-element list.
    """
    if max_tokens < 1:
        max_tokens = 1

    enc = get_tokenizer()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    pieces: list[str] = []
    i = 0
    while i < len(tokens):
        end = min(i + max_tokens, len(tokens))
        token_slice = tokens[i:end]
        decoded = enc.decode(token_slice)

        if enc.encode(decoded) != token_slice:
            # Shrink the window until the round trip is clean
            while end > i + 1:
                end -= 1
                token_slice = tokens[i:end]
                decoded = enc.decode(token_slice)
                if enc.encode(decoded) == token_slice:
                    break

        if decoded:
            pieces.append(decoded)
        i = end

    return pieces if pieces else [text]


def split_text_with_lines(
    text: str,
    start_line: int,
    max_tokens: int,
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

    Returns:
        ``(piece, start_line, end_line, start_column)`` tuples in order,
        where the column is a 0-based character offset within
        ``start_line``.
    """
    result: list[tuple[str, int, int, int]] = []
    line = start_line
    column = 0
    for piece in hard_split_by_tokens(text, max_tokens):
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

    Returns:
        Zero or more chunks, in order.
    """
    token_count = count_tokens(text)
    if token_count <= max_tokens:
        pieces = [(text, start_line, start_line + text.count("\n"), 0)]
    else:
        logger.debug(
            "Hard-splitting oversized %s chunk at %s:%d (%d tokens > %d)",
            chunk_type.value, file_path, start_line, token_count, max_tokens,
        )
        pieces = split_text_with_lines(text, start_line, max_tokens)

    base_metadata = metadata or {}
    chunks: list[Chunk] = []
    for piece, piece_start, piece_end, piece_column in pieces:
        piece_tokens = count_tokens(piece)
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
) -> list[Chunk]:
    """Split text at blank-line boundaries for unsupported languages.

    Uses offset-based line tracking for accurate line numbers.
    """
    max_tokens = config.chunking.max_tokens
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
        ))

    for block, offset in blocks_with_offsets:
        block_tokens = count_tokens(block)

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
