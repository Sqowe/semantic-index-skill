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


def make_chunk_id(file_path: str, content: str, start_line: int = 0) -> str:
    """Generate a deterministic chunk ID from file path, content, and position.

    Including start_line prevents ID collisions when identical content
    appears multiple times in the same file.
    """
    raw = f"{file_path}:{start_line}:{content}"
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

    for block, offset in blocks_with_offsets:
        block_tokens = count_tokens(block)

        if current_tokens + block_tokens > max_tokens and current_parts:
            chunk_text = "\n\n".join(current_parts)
            tc = count_tokens(chunk_text)
            s_line = 1 + content[:current_start_offset].count("\n")
            e_line = s_line + chunk_text.count("\n")
            if tc >= min_tokens:
                chunks.append(Chunk(
                    id=make_chunk_id(file_path, chunk_text, s_line),
                    file_path=file_path,
                    start_line=s_line,
                    end_line=e_line,
                    content=chunk_text,
                    chunk_type=ChunkType.UNKNOWN,
                    language=language,
                    token_count=tc,
                    metadata={},
                ))
            current_start_offset = offset
            current_parts = []
            current_tokens = 0

        current_parts.append(block)
        current_tokens += block_tokens

    if current_parts:
        chunk_text = "\n\n".join(current_parts)
        tc = count_tokens(chunk_text)
        s_line = 1 + content[:current_start_offset].count("\n")
        e_line = s_line + chunk_text.count("\n")
        if tc >= min_tokens:
            chunks.append(Chunk(
                id=make_chunk_id(file_path, chunk_text, s_line),
                file_path=file_path,
                start_line=s_line,
                end_line=e_line,
                content=chunk_text,
                chunk_type=ChunkType.UNKNOWN,
                language=language,
                token_count=tc,
                metadata={},
            ))

    return chunks
