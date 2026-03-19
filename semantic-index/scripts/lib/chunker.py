"""AST-aware code chunking and header-based markdown chunking.

Splits source files into semantically meaningful chunks for embedding.
Uses Tree-sitter for code files (Python, JS, TS) and header-based
splitting for markdown. Falls back to blank-line splitting for
unsupported languages.
"""

import hashlib
import logging
import os
import re
from bisect import bisect_right
from pathlib import Path
from typing import Optional

import tiktoken

from .config import Config
from .models import Chunk, ChunkType

logger = logging.getLogger(__name__)

# Lazy-loaded tokenizer
_tokenizer: Optional[tiktoken.Encoding] = None


def _get_tokenizer() -> tiktoken.Encoding:
    """Get or create the tiktoken tokenizer (cl100k_base)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_get_tokenizer().encode(text))


def _make_chunk_id(file_path: str, content: str, start_line: int = 0) -> str:
    """Generate a deterministic chunk ID from file path, content, and position.

    Including start_line prevents ID collisions when identical content
    appears multiple times in the same file.
    """
    raw = f"{file_path}:{start_line}:{content}"
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _detect_language(file_path: str) -> Optional[str]:
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
    }
    ext = os.path.splitext(file_path)[1].lower()
    return ext_map.get(ext)


# ---------------------------------------------------------------------------
# Markdown chunking (Step 1.5)
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _chunk_markdown(
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
        token_count = _count_tokens(fm_text)
        if token_count >= min_tokens:
            chunks.append(Chunk(
                id=_make_chunk_id(file_path, fm_text, 1),
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
            if pre_content and _count_tokens(pre_content) >= min_tokens:
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

        token_count = _count_tokens(section_text)
        path_titles = [t for _, t in header_path]
        h_level = header_path[-1][0] if header_path else 0
        meta = {"header_path": path_titles, "header_level": h_level}

        if token_count <= max_tokens:
            if token_count >= min_tokens:
                chunks.append(Chunk(
                    id=_make_chunk_id(file_path, section_text, start + 1),
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
        para_tokens = _count_tokens(para_text)

        if current_tokens + para_tokens > max_tokens and current_parts:
            # Flush current accumulation
            chunk_text = "\n\n".join(current_parts)
            tc = _count_tokens(chunk_text)
            start_ln = _offset_to_line(current_start_offset)
            end_ln = start_ln + chunk_text.count("\n")
            if tc >= min_tokens:
                chunks.append(Chunk(
                    id=_make_chunk_id(file_path, chunk_text, start_ln),
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
        tc = _count_tokens(chunk_text)
        start_ln = _offset_to_line(current_start_offset)
        end_ln = start_ln + chunk_text.count("\n")
        if tc >= min_tokens:
            chunks.append(Chunk(
                id=_make_chunk_id(file_path, chunk_text, start_ln),
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


# ---------------------------------------------------------------------------
# Tree-sitter code chunking (Step 1.6)
# ---------------------------------------------------------------------------

# Lazy-loaded parsers cache
_parsers: dict[str, object] = {}

# Tree-sitter node types that represent top-level definitions
EXTRACTABLE_NODES: dict[str, list[str]] = {
    "python": [
        "function_definition",
        "class_definition",
        "decorated_definition",
    ],
    "javascript": [
        "function_declaration",
        "class_declaration",
        "lexical_declaration",
        "export_statement",
    ],
    "typescript": [
        "function_declaration",
        "class_declaration",
        "lexical_declaration",
        "export_statement",
        "interface_declaration",
        "type_alias_declaration",
    ],
}

# Node types that represent methods inside classes
METHOD_NODES: dict[str, list[str]] = {
    "python": ["function_definition"],
    "javascript": ["method_definition"],
    "typescript": ["method_definition", "public_field_definition"],
}


def _get_ts_language(language: str):
    """Load a Tree-sitter Language object for the given language.

    Each grammar is imported independently so a missing grammar for one
    language doesn't disable AST parsing for others.

    Returns None if the grammar is not available.
    """
    if language == "python":
        try:
            import tree_sitter_python
            return tree_sitter_python.language()
        except ImportError:
            logger.warning("tree-sitter-python not installed, falling back to text splitting for Python")
            return None
        except Exception as exc:
            logger.warning("Failed to load Python grammar: %s", exc)
            return None

    elif language == "javascript":
        try:
            import tree_sitter_javascript
            return tree_sitter_javascript.language()
        except ImportError:
            logger.warning("tree-sitter-javascript not installed, falling back to text splitting for JS")
            return None
        except Exception as exc:
            logger.warning("Failed to load JavaScript grammar: %s", exc)
            return None

    elif language == "typescript":
        try:
            import tree_sitter_typescript
            return tree_sitter_typescript.language_typescript()
        except ImportError:
            logger.warning("tree-sitter-typescript not installed, falling back to text splitting for TS")
            return None
        except Exception as exc:
            logger.warning("Failed to load TypeScript grammar: %s", exc)
            return None

    return None


def _get_parser(language: str):
    """Get or create a Tree-sitter parser for the given language.

    Handles API differences across tree-sitter versions:
    - v0.22+: Parser(language) constructor
    - v0.21: Parser() + parser.language = Language(language)
    Falls back to text splitting if initialization fails.
    """
    if language in _parsers:
        return _parsers[language]

    try:
        import tree_sitter
    except ImportError:
        logger.warning("tree-sitter not installed")
        return None

    ts_lang = _get_ts_language(language)
    if ts_lang is None:
        _parsers[language] = None
        return None

    try:
        # New API (tree-sitter >= 0.22): Parser accepts language directly
        parser = tree_sitter.Parser(ts_lang)
    except TypeError:
        try:
            # Older API (tree-sitter 0.21.x): Parser() + set language
            parser = tree_sitter.Parser()
            parser.language = tree_sitter.Language(ts_lang)
        except (TypeError, AttributeError):
            try:
                # Fallback: older set_language method
                parser = tree_sitter.Parser()
                parser.set_language(ts_lang)
            except Exception as exc:
                logger.warning(
                    "Failed to initialize Tree-sitter parser for %s: %s. "
                    "Falling back to text splitting.",
                    language, exc,
                )
                _parsers[language] = None
                return None
    except Exception as exc:
        logger.warning(
            "Unexpected error creating Tree-sitter parser for %s: %s. "
            "Falling back to text splitting.",
            language, exc,
        )
        _parsers[language] = None
        return None

    _parsers[language] = parser
    return parser


def _extract_symbol_name(node, language: str) -> Optional[str]:
    """Extract the name of a function/class/method from an AST node."""
    # For decorated definitions (Python), look at the inner definition
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _extract_symbol_name(child, language)
        return None

    # For export statements, look at the inner declaration
    if node.type == "export_statement":
        for child in node.children:
            name = _extract_symbol_name(child, language)
            if name:
                return name
        return None

    # Look for a 'name' or 'identifier' child
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "type_identifier"):
            return child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text

    return None


def _node_to_chunk_type(node_type: str, is_method: bool = False) -> ChunkType:
    """Map a Tree-sitter node type to a ChunkType."""
    if is_method:
        return ChunkType.METHOD
    if "class" in node_type:
        return ChunkType.CLASS
    if "function" in node_type:
        return ChunkType.FUNCTION
    # Declarations that aren't functions or classes
    if node_type in (
        "lexical_declaration", "export_statement",
        "interface_declaration", "type_alias_declaration",
    ):
        return ChunkType.MODULE_LEVEL
    return ChunkType.UNKNOWN


def _split_oversized_chunk(
    content: str,
    file_path: str,
    start_line: int,
    language: str,
    chunk_type: ChunkType,
    symbol_name: Optional[str],
    max_tokens: int,
    min_tokens: int,
    metadata: dict,
) -> list[Chunk]:
    """Split an oversized code chunk at blank lines, then hard-split as last resort.

    Uses offset-based line tracking for accurate line numbers.
    """
    # Try splitting at blank lines first
    gap_pattern = re.compile(r"\n\n+")
    parts_with_offsets: list[tuple[str, int]] = []
    prev_end = 0
    for m in gap_pattern.finditer(content):
        part = content[prev_end:m.start()]
        if part:
            parts_with_offsets.append((part, prev_end))
        prev_end = m.end()
    if prev_end < len(content):
        trailing = content[prev_end:]
        if trailing.strip():
            parts_with_offsets.append((trailing, prev_end))

    if len(parts_with_offsets) > 1:
        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_tokens = 0
        current_start_offset = parts_with_offsets[0][1]

        for part, offset in parts_with_offsets:
            part_tokens = _count_tokens(part)
            if current_tokens + part_tokens > max_tokens and current_parts:
                chunk_text = "\n\n".join(current_parts)
                tc = _count_tokens(chunk_text)
                s_line = start_line + content[:current_start_offset].count("\n")
                e_line = s_line + chunk_text.count("\n")
                if tc >= min_tokens:
                    chunks.append(Chunk(
                        id=_make_chunk_id(file_path, chunk_text, s_line),
                        file_path=file_path,
                        start_line=s_line,
                        end_line=e_line,
                        content=chunk_text,
                        chunk_type=chunk_type,
                        language=language,
                        symbol_name=symbol_name,
                        token_count=tc,
                        metadata=metadata.copy(),
                    ))
                current_start_offset = offset
                current_parts = []
                current_tokens = 0

            current_parts.append(part)
            current_tokens += part_tokens

        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            tc = _count_tokens(chunk_text)
            s_line = start_line + content[:current_start_offset].count("\n")
            e_line = s_line + chunk_text.count("\n")
            if tc >= min_tokens:
                chunks.append(Chunk(
                    id=_make_chunk_id(file_path, chunk_text, s_line),
                    file_path=file_path,
                    start_line=s_line,
                    end_line=e_line,
                    content=chunk_text,
                    chunk_type=chunk_type,
                    language=language,
                    symbol_name=symbol_name,
                    token_count=tc,
                    metadata=metadata.copy(),
                ))

        if chunks:
            return chunks

    # Last resort: hard split by lines
    lines = content.split("\n")
    chunks = []
    current_lines: list[str] = []
    current_tokens = 0
    current_line = start_line

    for line in lines:
        line_tokens = _count_tokens(line)
        if current_tokens + line_tokens > max_tokens and current_lines:
            chunk_text = "\n".join(current_lines)
            tc = _count_tokens(chunk_text)
            if tc >= min_tokens:
                chunks.append(Chunk(
                    id=_make_chunk_id(file_path, chunk_text, current_line),
                    file_path=file_path,
                    start_line=current_line,
                    end_line=current_line + len(current_lines) - 1,
                    content=chunk_text,
                    chunk_type=chunk_type,
                    language=language,
                    symbol_name=symbol_name,
                    token_count=tc,
                    metadata=metadata.copy(),
                ))
            current_line = current_line + len(current_lines)
            current_lines = []
            current_tokens = 0

        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        chunk_text = "\n".join(current_lines)
        tc = _count_tokens(chunk_text)
        if tc >= min_tokens:
            chunks.append(Chunk(
                id=_make_chunk_id(file_path, chunk_text, current_line),
                file_path=file_path,
                start_line=current_line,
                end_line=current_line + len(current_lines) - 1,
                content=chunk_text,
                chunk_type=chunk_type,
                language=language,
                symbol_name=symbol_name,
                token_count=tc,
                metadata=metadata.copy(),
            ))

    return chunks


def _chunk_code_with_treesitter(
    content: str,
    file_path: str,
    language: str,
    config: Config,
) -> list[Chunk]:
    """Chunk a code file using Tree-sitter AST parsing.

    Extracts top-level functions, classes, and methods as individual chunks.
    Module-level code (imports, constants) becomes a separate chunk.
    Oversized nodes are split at logical boundaries.
    """
    parser = _get_parser(language)
    if parser is None:
        return _chunk_text_fallback(content, file_path, language, config)

    source_bytes = content.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    max_tokens = config.chunking.max_tokens
    min_tokens = config.chunking.min_tokens
    extractable = set(EXTRACTABLE_NODES.get(language, []))
    method_types = set(METHOD_NODES.get(language, []))

    chunks: list[Chunk] = []
    covered_ranges: list[tuple[int, int]] = []  # (start_byte, end_byte)

    for node in root.children:
        if node.type not in extractable:
            continue

        node_text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        symbol = _extract_symbol_name(node, language)
        is_class = "class" in node.type

        # For classes, try to extract methods as separate chunks
        if is_class and _count_tokens(node_text) > max_tokens:
            class_chunks = _chunk_class_node(
                node, source_bytes, file_path, language, symbol, config,
            )
            if class_chunks:
                chunks.extend(class_chunks)
                covered_ranges.append((node.start_byte, node.end_byte))
                continue

        token_count = _count_tokens(node_text)
        chunk_type = _node_to_chunk_type(node.type)

        if token_count <= max_tokens:
            if token_count >= min_tokens:
                chunks.append(Chunk(
                    id=_make_chunk_id(file_path, node_text, start_line),
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=node_text,
                    chunk_type=chunk_type,
                    language=language,
                    symbol_name=symbol,
                    token_count=token_count,
                    metadata={},
                ))
        else:
            # Oversized — split it
            sub_chunks = _split_oversized_chunk(
                node_text, file_path, start_line, language,
                chunk_type, symbol, max_tokens, min_tokens, {},
            )
            chunks.extend(sub_chunks)

        covered_ranges.append((node.start_byte, node.end_byte))

    # Collect module-level code (everything not covered by extracted nodes)
    # Precompute newline byte positions for O(log n) byte-offset-to-line conversion
    _nl_byte_offsets = [i for i, b in enumerate(source_bytes) if b == ord(b"\n")]

    def _byte_offset_to_line(offset: int) -> int:
        return bisect_right(_nl_byte_offsets, offset - 1) + 1

    # Track each gap segment with its real source line range
    module_segments: list[tuple[str, int, int]] = []  # (text, start_line, end_line)
    prev_end = 0

    for start_byte, end_byte in sorted(covered_ranges):
        gap = source_bytes[prev_end:start_byte].decode("utf-8", errors="replace")
        if gap.strip():  # check emptiness without mutating
            gap_start_line = _byte_offset_to_line(prev_end)
            gap_end_line = gap_start_line + gap.count("\n")
            module_segments.append((gap, gap_start_line, gap_end_line))
        prev_end = end_byte

    # Trailing module-level code
    trailing = source_bytes[prev_end:].decode("utf-8", errors="replace")
    if trailing.strip():
        trail_start_line = _byte_offset_to_line(prev_end)
        trail_end_line = trail_start_line + trailing.count("\n")
        module_segments.append((trailing, trail_start_line, trail_end_line))

    if module_segments:
        # Emit each segment (or join small ones) with real line ranges
        module_text = "\n\n".join(seg[0] for seg in module_segments)
        tc = _count_tokens(module_text)
        first_line = module_segments[0][1]
        last_line = module_segments[-1][2]

        if tc >= min_tokens:
            if tc <= max_tokens:
                chunks.append(Chunk(
                    id=_make_chunk_id(file_path, module_text, first_line),
                    file_path=file_path,
                    start_line=first_line,
                    end_line=last_line,
                    content=module_text,
                    chunk_type=ChunkType.MODULE_LEVEL,
                    language=language,
                    token_count=tc,
                    metadata={},
                ))
            else:
                sub = _split_oversized_chunk(
                    module_text, file_path, first_line, language,
                    ChunkType.MODULE_LEVEL, None, max_tokens, min_tokens, {},
                )
                chunks.extend(sub)

    return chunks


def _chunk_class_node(
    class_node,
    source_bytes: bytes,
    file_path: str,
    language: str,
    class_name: Optional[str],
    config: Config,
) -> list[Chunk]:
    """Extract methods from a class node as individual chunks.

    If the class is too large, each method becomes its own chunk.
    The class signature (without method bodies) becomes a separate chunk.
    """
    max_tokens = config.chunking.max_tokens
    min_tokens = config.chunking.min_tokens
    method_types = set(METHOD_NODES.get(language, []))
    chunks: list[Chunk] = []

    # Find the class body node
    body_node = None
    for child in class_node.children:
        if child.type in ("block", "class_body", "statement_block"):
            body_node = child
            break

    if body_node is None:
        return []

    method_ranges: list[tuple[int, int]] = []

    for child in body_node.children:
        # Handle decorated methods in Python:
        # Check the inner node type for method matching,
        # but use the outer decorated_definition range for content extraction.
        is_method = False
        if child.type == "decorated_definition":
            for sub in child.children:
                if sub.type in method_types:
                    is_method = True
                    break
        elif child.type in method_types:
            is_method = True

        if is_method:
            method_text = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
            method_name = _extract_symbol_name(child, language)
            start_line = child.start_point[0] + 1
            end_line = child.end_point[0] + 1
            tc = _count_tokens(method_text)

            if tc <= max_tokens:
                if tc >= min_tokens:
                    chunks.append(Chunk(
                        id=_make_chunk_id(file_path, method_text, start_line),
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        content=method_text,
                        chunk_type=ChunkType.METHOD,
                        language=language,
                        symbol_name=method_name,
                        token_count=tc,
                        metadata={"parent_class": class_name},
                    ))
            else:
                sub = _split_oversized_chunk(
                    method_text, file_path, start_line, language,
                    ChunkType.METHOD, method_name, max_tokens, min_tokens,
                    {"parent_class": class_name},
                )
                chunks.extend(sub)

            method_ranges.append((child.start_byte, child.end_byte))

    # Class signature: everything in the class that isn't a method body
    sig_parts: list[str] = []
    prev_end = class_node.start_byte
    for m_start, m_end in sorted(method_ranges):
        gap = source_bytes[prev_end:m_start].decode("utf-8", errors="replace").strip()
        if gap:
            sig_parts.append(gap)
        prev_end = m_end
    trailing = source_bytes[prev_end:class_node.end_byte].decode("utf-8", errors="replace").strip()
    if trailing:
        sig_parts.append(trailing)

    if sig_parts:
        sig_text = "\n".join(sig_parts)
        tc = _count_tokens(sig_text)
        if tc >= min_tokens:
            chunks.insert(0, Chunk(
                id=_make_chunk_id(file_path, sig_text, class_node.start_point[0] + 1),
                file_path=file_path,
                start_line=class_node.start_point[0] + 1,
                end_line=class_node.end_point[0] + 1,
                content=sig_text,
                chunk_type=ChunkType.CLASS,
                language=language,
                symbol_name=class_name,
                token_count=tc,
                metadata={},
            ))

    return chunks


# ---------------------------------------------------------------------------
# Fallback text chunking (unsupported languages)
# ---------------------------------------------------------------------------

def _chunk_text_fallback(
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
        block_tokens = _count_tokens(block)

        if current_tokens + block_tokens > max_tokens and current_parts:
            chunk_text = "\n\n".join(current_parts)
            tc = _count_tokens(chunk_text)
            s_line = 1 + content[:current_start_offset].count("\n")
            e_line = s_line + chunk_text.count("\n")
            if tc >= min_tokens:
                chunks.append(Chunk(
                    id=_make_chunk_id(file_path, chunk_text, s_line),
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
        tc = _count_tokens(chunk_text)
        s_line = 1 + content[:current_start_offset].count("\n")
        e_line = s_line + chunk_text.count("\n")
        if tc >= min_tokens:
            chunks.append(Chunk(
                id=_make_chunk_id(file_path, chunk_text, s_line),
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Languages supported by Tree-sitter in Phase 1
TREESITTER_LANGUAGES = {"python", "javascript", "typescript"}


def chunk_file(
    file_path: str,
    project_dir: str,
    config: Config,
) -> list[Chunk]:
    """Chunk a single file into semantically meaningful pieces.

    Dispatches to the appropriate chunking strategy based on language:
    - Markdown files: header-based splitting
    - Python/JS/TS: Tree-sitter AST-aware splitting
    - Other: blank-line fallback splitting

    Args:
        file_path: Relative path to the file (from project root).
        project_dir: Absolute path to the project root.
        config: Loaded configuration.

    Returns:
        List of Chunk objects. May be empty if the file is too small.
    """
    abs_path = os.path.join(project_dir, file_path)

    try:
        content = Path(abs_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Failed to read file %s: %s", file_path, exc)
        return []

    if not content.strip():
        return []

    language = _detect_language(file_path)

    if language == "markdown":
        return _chunk_markdown(content, file_path, config)
    elif language in TREESITTER_LANGUAGES:
        return _chunk_code_with_treesitter(content, file_path, language, config)
    elif language is not None:
        return _chunk_text_fallback(content, file_path, language, config)
    else:
        return _chunk_text_fallback(content, file_path, None, config)
