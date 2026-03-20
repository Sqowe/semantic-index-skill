"""Header-based markdown chunking.

Splits markdown files into chunks by header hierarchy.
Frontmatter (YAML between --- delimiters) becomes its own chunk.
Sections exceeding max_tokens are split at paragraph boundaries.
"""

import re
from bisect import bisect_right

from .common import count_tokens, make_chunk_id
from ..config import Config
from ..models import Chunk, ChunkType

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def chunk_markdown(
    content: str,
    file_path: str,
    config: Config,
) -> list[Chunk]:
    """Split a markdown file into chunks by headers.

    - Frontmatter (YAML between --- delimiters) becomes its own chunk.
    - Each header section becomes a chunk.
    - Sections exceeding max_tokens are split at paragraph boundaries.
    - Each chunk preserves the header hierarchy in metadata.
    """
    chunks: list[Chunk] = []
    lines = content.split("\n")
    max_tokens = config.chunking.max_tokens
    min_tokens = config.chunking.min_tokens

    # Extract frontmatter
    fm_match = _FRONTMATTER_RE.match(content)
    fm_end_line = 0
    if fm_match:
        fm_text = fm_match.group(0)
        fm_lines = fm_text.count("\n")
        fm_end_line = fm_lines
        token_count = count_tokens(fm_text)
        if token_count >= min_tokens:
            chunks.append(Chunk(
                id=make_chunk_id(file_path, fm_text, 1),
                file_path=file_path,
                start_line=1,
                end_line=fm_lines,
                content=fm_text.strip(),
                chunk_type=ChunkType.MARKDOWN_FRONTMATTER,
                language="markdown",
                token_count=token_count,
                metadata={"header_path": [], "header_level": 0},
            ))

    # Find all headers and their positions
    header_positions: list[tuple[int, int, str]] = []  # (line_idx, level, title)
    for i, line in enumerate(lines):
        if i < fm_end_line:
            continue
        m = _HEADER_RE.match(line)
        if m:
            header_positions.append((i, len(m.group(1)), m.group(2).strip()))

    # Build sections
    sections: list[tuple[int, int, list[tuple[int, str]]]] = []
    # Each section: (start_line, end_line, header_path)

    if not header_positions:
        # No headers — entire content (after frontmatter) is one section
        if fm_end_line < len(lines):
            sections.append((fm_end_line, len(lines) - 1, []))
    else:
        # Content before first header
        if header_positions[0][0] > fm_end_line:
            pre_content = "\n".join(lines[fm_end_line:header_positions[0][0]]).strip()
            if pre_content and count_tokens(pre_content) >= min_tokens:
                sections.append((fm_end_line, header_positions[0][0] - 1, []))

        # Each header section
        header_stack: list[tuple[int, str]] = []  # (level, title)
        for idx, (line_idx, level, title) in enumerate(header_positions):
            # Update header stack
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, title))

            # Section end is the line before the next header, or EOF
            if idx + 1 < len(header_positions):
                end_line = header_positions[idx + 1][0] - 1
            else:
                end_line = len(lines) - 1

            header_path = [(lvl, t) for lvl, t in header_stack]
            sections.append((line_idx, end_line, header_path))

    # Convert sections to chunks, splitting large ones
    for start, end, header_path in sections:
        section_text = "\n".join(lines[start:end + 1]).strip()
        if not section_text:
            continue

        token_count = count_tokens(section_text)
        path_titles = [t for _, t in header_path]
        h_level = header_path[-1][0] if header_path else 0
        meta = {"header_path": path_titles, "header_level": h_level}

        if token_count <= max_tokens:
            if token_count >= min_tokens:
                chunks.append(Chunk(
                    id=make_chunk_id(file_path, section_text, start + 1),
                    file_path=file_path,
                    start_line=start + 1,
                    end_line=end + 1,
                    content=section_text,
                    chunk_type=ChunkType.MARKDOWN_SECTION,
                    language="markdown",
                    token_count=token_count,
                    metadata=meta,
                ))
        else:
            # Split at paragraph boundaries (double newline)
            sub_chunks = _split_text_by_paragraphs(
                section_text, start + 1, file_path, max_tokens, min_tokens, meta,
            )
            chunks.extend(sub_chunks)

    return chunks


def _split_text_by_paragraphs(
    text: str,
    base_line: int,
    file_path: str,
    max_tokens: int,
    min_tokens: int,
    metadata: dict,
) -> list[Chunk]:
    """Split text into chunks at paragraph boundaries (double newlines).

    Uses character offset tracking to compute accurate line numbers,
    accounting for variable numbers of blank lines between paragraphs.
    """
    # Find paragraph boundaries with their positions in the original text
    gap_pattern = re.compile(r"\n\n+")
    paragraphs: list[tuple[str, int]] = []  # (text, char_offset)
    prev_end = 0
    for m in gap_pattern.finditer(text):
        para_text = text[prev_end:m.start()]
        if para_text:
            paragraphs.append((para_text, prev_end))
        prev_end = m.end()
    # Trailing paragraph
    if prev_end < len(text):
        para_text = text[prev_end:]
        if para_text.strip():
            paragraphs.append((para_text, prev_end))

    if not paragraphs:
        return []

    # Precompute newline positions for O(log n) offset-to-line conversion
    _newline_offsets = [i for i, ch in enumerate(text) if ch == "\n"]

    def _offset_to_line(offset: int) -> int:
        return base_line + bisect_right(_newline_offsets, offset - 1)

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    current_start_offset = paragraphs[0][1] if paragraphs else 0

    for para_text, para_offset in paragraphs:
        para_tokens = count_tokens(para_text)

        if current_tokens + para_tokens > max_tokens and current_parts:
            # Flush current accumulation
            chunk_text = "\n\n".join(current_parts)
            tc = count_tokens(chunk_text)
            start_ln = _offset_to_line(current_start_offset)
            end_ln = start_ln + chunk_text.count("\n")
            if tc >= min_tokens:
                chunks.append(Chunk(
                    id=make_chunk_id(file_path, chunk_text, start_ln),
                    file_path=file_path,
                    start_line=start_ln,
                    end_line=end_ln,
                    content=chunk_text,
                    chunk_type=ChunkType.MARKDOWN_SECTION,
                    language="markdown",
                    token_count=tc,
                    metadata=metadata.copy(),
                ))
            current_start_offset = para_offset
            current_parts = []
            current_tokens = 0

        current_parts.append(para_text)
        current_tokens += para_tokens

    # Flush remaining
    if current_parts:
        chunk_text = "\n\n".join(current_parts)
        tc = count_tokens(chunk_text)
        start_ln = _offset_to_line(current_start_offset)
        end_ln = start_ln + chunk_text.count("\n")
        if tc >= min_tokens:
            chunks.append(Chunk(
                id=make_chunk_id(file_path, chunk_text, start_ln),
                file_path=file_path,
                start_line=start_ln,
                end_line=end_ln,
                content=chunk_text,
                chunk_type=ChunkType.MARKDOWN_SECTION,
                language="markdown",
                token_count=tc,
                metadata=metadata.copy(),
            ))

    return chunks
