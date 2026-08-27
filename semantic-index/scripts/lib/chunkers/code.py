"""Tree-sitter AST-aware code chunking.

Parses code files using Tree-sitter grammars and extracts top-level
functions, classes, and methods as individual chunks. Module-level
code (imports, constants) becomes a separate chunk. Oversized nodes
are split at logical boundaries.

Supported languages: Python, JavaScript, TypeScript, Go, Rust, Java,
C, C++, Ruby, PHP.
"""

import logging
import re
from bisect import bisect_right
from typing import Optional

from .common import build_chunks, count_tokens as _base_count_tokens, make_chunk_id, chunk_text_fallback
from ..tokenizer_resolver import TokenizerWrapper
from ..config import Config
from ..models import Chunk, ChunkType

logger = logging.getLogger(__name__)

# Lazy-loaded parsers cache
_parsers: dict[str, object] = {}

# Grammars to try when the one picked from the file extension cannot parse
# the file cleanly.
#
# An extension does not always settle which grammar fits. A ".h" holds C in
# a C project and C++ in a C++ one, and both arrive here as "c". Guessing
# from the extension gets one of the two wrong: measured on one C++
# codebase, the C grammar failed to parse 98% of its ".h" files where the
# C++ grammar failed on 3%. Guessing from the content is no better — the
# markers that distinguish the two (``class``, ``namespace``, ``template``)
# appear inside comments and string literals as well.
#
# So the file itself decides. The declared grammar is tried first and an
# alternate is consulted only when that parse comes back with errors, which
# means a project whose headers really are C pays nothing.
GRAMMAR_ALTERNATES: dict[str, tuple[str, ...]] = {
    "c": ("cpp",),
}

# Node types treated as class-like containers for oversized splitting.
# Shared between chunk_code_with_treesitter (is_class) and _node_to_chunk_type
# to prevent drift.
CLASS_LIKE_NODES: set[str] = {
    "impl_item", "trait_item", "struct_item", "enum_item",
    "struct_specifier", "class_specifier", "enum_specifier",
    "interface_declaration", "enum_declaration",
    "record_declaration", "annotation_type_declaration",
    "trait_declaration", "module", "namespace_definition",
}

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
    "go": [
        "function_declaration",
        "method_declaration",
        "type_declaration",
    ],
    "rust": [
        "function_item",
        "struct_item",
        "enum_item",
        "impl_item",
        "trait_item",
        "type_item",
        "const_item",
        "static_item",
        "macro_definition",
    ],
    "java": [
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
        "annotation_type_declaration",
    ],
    "c": [
        "function_definition",
        "struct_specifier",
        "enum_specifier",
        "type_definition",
        "declaration",
    ],
    "cpp": [
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "enum_specifier",
        "namespace_definition",
        "template_declaration",
        "type_definition",
        "declaration",
    ],
    "ruby": [
        "method",
        "class",
        "module",
        "singleton_method",
    ],
    "php": [
        "function_definition",
        "class_declaration",
        "interface_declaration",
        "trait_declaration",
        "enum_declaration",
    ],
}

# Node types that represent methods inside classes
METHOD_NODES: dict[str, list[str]] = {
    "python": ["function_definition"],
    "javascript": ["method_definition"],
    "typescript": ["method_definition", "public_field_definition"],
    "go": [],  # Go methods are top-level (method_declaration), not nested
    "rust": ["function_item"],  # Inside impl blocks
    "java": ["method_declaration", "constructor_declaration"],
    "c": [],  # C has no classes
    "cpp": ["function_definition"],
    "ruby": ["method", "singleton_method"],
    "php": ["method_declaration"],
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

    elif language == "go":
        try:
            import tree_sitter_go
            return tree_sitter_go.language()
        except ImportError:
            logger.warning("tree-sitter-go not installed, falling back to text splitting for Go")
            return None
        except Exception as exc:
            logger.warning("Failed to load Go grammar: %s", exc)
            return None

    elif language == "rust":
        try:
            import tree_sitter_rust
            return tree_sitter_rust.language()
        except ImportError:
            logger.warning("tree-sitter-rust not installed, falling back to text splitting for Rust")
            return None
        except Exception as exc:
            logger.warning("Failed to load Rust grammar: %s", exc)
            return None

    elif language == "java":
        try:
            import tree_sitter_java
            return tree_sitter_java.language()
        except ImportError:
            logger.warning("tree-sitter-java not installed, falling back to text splitting for Java")
            return None
        except Exception as exc:
            logger.warning("Failed to load Java grammar: %s", exc)
            return None

    elif language == "c":
        try:
            import tree_sitter_c
            return tree_sitter_c.language()
        except ImportError:
            logger.warning("tree-sitter-c not installed, falling back to text splitting for C")
            return None
        except Exception as exc:
            logger.warning("Failed to load C grammar: %s", exc)
            return None

    elif language == "cpp":
        try:
            import tree_sitter_cpp
            return tree_sitter_cpp.language()
        except ImportError:
            logger.warning("tree-sitter-cpp not installed, falling back to text splitting for C++")
            return None
        except Exception as exc:
            logger.warning("Failed to load C++ grammar: %s", exc)
            return None

    elif language == "ruby":
        try:
            import tree_sitter_ruby
            return tree_sitter_ruby.language()
        except ImportError:
            logger.warning("tree-sitter-ruby not installed, falling back to text splitting for Ruby")
            return None
        except Exception as exc:
            logger.warning("Failed to load Ruby grammar: %s", exc)
            return None

    elif language == "php":
        try:
            import tree_sitter_php
            return tree_sitter_php.language_php()
        except (ImportError, AttributeError):
            try:
                # Some versions expose language() instead of language_php()
                import tree_sitter_php
                return tree_sitter_php.language()
            except ImportError:
                logger.warning("tree-sitter-php not installed, falling back to text splitting for PHP")
                return None
            except Exception as exc:
                logger.warning("Failed to load PHP grammar: %s", exc)
                return None
        except Exception as exc:
            logger.warning("Failed to load PHP grammar: %s", exc)
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


def _count_parse_errors(root) -> int:
    """Count the ERROR and MISSING nodes in a parse tree.

    Used to compare two candidate grammars when neither parses a file
    cleanly, so the closer fit still wins. Subtrees that parsed without
    errors are pruned via ``has_error``, so the walk only descends into
    the parts of the tree that actually went wrong.

    Args:
        root: Root node of a Tree-sitter parse tree.

    Returns:
        The number of error and missing nodes.
    """
    total = 0
    stack = [root]
    while stack:
        node = stack.pop()
        if not node.has_error:
            continue
        if node.type == "ERROR" or node.is_missing:
            total += 1
        stack.extend(node.children)
    return total


def _parse_with_best_grammar(source_bytes: bytes, language: str):
    """Parse *source_bytes*, retrying with an alternate grammar on failure.

    The grammar named by the file extension is tried first. If it parses
    cleanly — the common case — nothing else happens and the cost is one
    parse. Only when it reports errors are the alternates in
    :data:`GRAMMAR_ALTERNATES` tried: the first one to parse cleanly wins,
    and if none does, the one with the fewest errors wins. A file no
    grammar can handle keeps the declared grammar's tree, so behaviour is
    unchanged from before this fallback existed.

    Args:
        source_bytes: The file's UTF-8 encoded content.
        language: Language identifier detected from the file extension.

    Returns:
        A ``(tree, language)`` pair, where the language is the grammar that
        actually produced the tree. Callers must use the returned language
        for node-type lookups and symbol extraction, not the one they
        passed in. Returns ``(None, language)`` when no parser is available.
    """
    parser = _get_parser(language)
    if parser is None:
        return None, language

    tree = parser.parse(source_bytes)
    if not tree.root_node.has_error:
        return tree, language

    best_tree = tree
    best_language = language
    # Counted only if some alternate also fails and the two need comparing.
    # Walking a badly broken tree is far more expensive than parsing, and
    # the usual outcome is an alternate that parses cleanly, so this stays
    # unpaid in the common case.
    best_errors: Optional[int] = None

    for alternate in GRAMMAR_ALTERNATES.get(language, ()):
        alt_parser = _get_parser(alternate)
        if alt_parser is None:
            continue
        alt_tree = alt_parser.parse(source_bytes)
        if not alt_tree.root_node.has_error:
            return alt_tree, alternate
        if best_errors is None:
            best_errors = _count_parse_errors(best_tree.root_node)
        alt_errors = _count_parse_errors(alt_tree.root_node)
        if alt_errors < best_errors:
            best_tree, best_language, best_errors = alt_tree, alternate, alt_errors

    return best_tree, best_language


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

    # Rust impl blocks: extract the type being implemented
    # For `impl Type`, return Type.
    # For `impl Trait for Type`, skip past `for` and return Type.
    if node.type == "impl_item":
        children = list(node.children)
        # Check for `for` keyword indicating trait impl
        for_index = None
        for i, child in enumerate(children):
            if child.type == "for":
                for_index = i
                break
        if for_index is not None:
            # impl Trait for Type — return the type after `for`
            for child in children[for_index + 1:]:
                if child.type in ("type_identifier", "generic_type",
                                  "scoped_type_identifier"):
                    text = child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
                    return text
        else:
            # impl Type — return the first type identifier
            for child in children:
                if child.type in ("type_identifier", "generic_type",
                                  "scoped_type_identifier"):
                    text = child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
                    return text
        return None

    # Go type declarations: type Name struct/interface
    if node.type == "type_declaration":
        for child in node.children:
            if child.type == "type_spec":
                # type_spec contains the type name as type_identifier
                for spec_child in child.children:
                    if spec_child.type == "type_identifier":
                        return spec_child.text.decode("utf-8") if isinstance(spec_child.text, bytes) else spec_child.text
                return _extract_symbol_name(child, language)
        return None

    # C/C++ template declarations: look at the inner declaration
    if node.type == "template_declaration":
        for child in node.children:
            name = _extract_symbol_name(child, language)
            if name:
                return name
        return None

    # C/C++ struct/enum specifiers and class specifiers
    if node.type in ("struct_specifier", "enum_specifier", "class_specifier",
                      "namespace_definition"):
        for child in node.children:
            if child.type in ("type_identifier", "name", "identifier",
                              "namespace_identifier"):
                return child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
        return None

    # C type_definition: typedef ... Name;
    if node.type == "type_definition":
        for child in node.children:
            if child.type == "type_identifier":
                return child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
        return None

    # C/C++ function_definition: name is inside function_declarator child
    if node.type == "function_definition":
        for child in node.children:
            if child.type == "function_declarator":
                for fc_child in child.children:
                    if fc_child.type in ("identifier", "field_identifier",
                                          "qualified_identifier", "destructor_name"):
                        return fc_child.text.decode("utf-8") if isinstance(fc_child.text, bytes) else fc_child.text
                # function_declarator found but no identifier inside — fall through
                break

    # Look for a 'name' or 'identifier' child
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "type_identifier",
                          "name", "field_identifier", "constant"):
            return child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text

    return None


def _node_to_chunk_type(node_type: str, is_method: bool = False) -> ChunkType:
    """Map a Tree-sitter node type to a ChunkType."""
    if is_method:
        return ChunkType.METHOD
    if "class" in node_type:
        return ChunkType.CLASS
    if "function" in node_type or node_type in ("method", "singleton_method",
                                                  "method_declaration"):
        return ChunkType.FUNCTION
    if node_type in CLASS_LIKE_NODES:
        return ChunkType.CLASS
    if node_type in (
        "lexical_declaration", "export_statement",
        "type_alias_declaration", "type_declaration",
        "type_item", "const_item", "static_item", "macro_definition",
        "type_definition", "declaration", "template_declaration",
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
    tokens: Optional["TokenizerWrapper"] = None,
) -> list[Chunk]:
    """Split an oversized code chunk at blank lines, then hard-split as last resort.

    Uses offset-based line tracking for accurate line numbers.
    """
    def count_tokens(text: str) -> int:
        return _base_count_tokens(text, tokens=tokens)

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

    def emit(chunk_text: str, s_line: int) -> list[Chunk]:
        """Build chunks for one accumulated text, hard-splitting if oversized."""
        return build_chunks(
            chunk_text,
            file_path=file_path,
            start_line=s_line,
            language=language,
            chunk_type=chunk_type,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            symbol_name=symbol_name,
            metadata=metadata,
            tokens=tokens,
        )

    if len(parts_with_offsets) > 1:
        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_tokens = 0
        current_start_offset = parts_with_offsets[0][1]

        for part, offset in parts_with_offsets:
            part_tokens = count_tokens(part)
            if current_tokens + part_tokens > max_tokens and current_parts:
                s_line = start_line + content[:current_start_offset].count("\n")
                chunks.extend(emit("\n\n".join(current_parts), s_line))
                current_start_offset = offset
                current_parts = []
                current_tokens = 0

            current_parts.append(part)
            current_tokens += part_tokens

        if current_parts:
            s_line = start_line + content[:current_start_offset].count("\n")
            chunks.extend(emit("\n\n".join(current_parts), s_line))

        if chunks:
            return chunks

    # Last resort: hard split by lines
    lines = content.split("\n")
    chunks = []
    current_lines: list[str] = []
    current_tokens = 0
    current_line = start_line

    for line in lines:
        line_tokens = count_tokens(line)
        if current_tokens + line_tokens > max_tokens and current_lines:
            chunks.extend(emit("\n".join(current_lines), current_line))
            current_line = current_line + len(current_lines)
            current_lines = []
            current_tokens = 0

        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        chunks.extend(emit("\n".join(current_lines), current_line))

    return chunks


def chunk_code_with_treesitter(
    content: str,
    file_path: str,
    language: str,
    config: Config,
    tokens: Optional["TokenizerWrapper"] = None,
    effective_max: Optional[int] = None,
) -> list[Chunk]:
    """Chunk a code file using Tree-sitter AST parsing.

    Extracts top-level functions, classes, and methods as individual chunks.
    Module-level code (imports, constants) becomes a separate chunk.
    Oversized nodes are split at logical boundaries.

    Falls back to text splitting if Tree-sitter is unavailable.

    *language* is what the file extension suggests, which is not always
    what the file contains — a ``.h`` arrives here as C whether it holds C
    or C++. When that grammar cannot parse the file cleanly, the
    alternates in :data:`GRAMMAR_ALTERNATES` are tried and the best fit is
    used instead, for node types, symbol extraction and the language
    recorded on each chunk alike.

    Args:
        content: File text.
        file_path: Project-relative path.
        language: Language detected from the file extension. Treated as a
            starting point, not a verdict — see above.
        config: Loaded configuration.
        tokens: Tokenizer wrapper. Defaults to the chunkers.common
            resolver's tiktoken fallback.
        effective_max: Safety-factor-adjusted chunk budget. When None,
            ``chunking.max_tokens`` is used (legacy behaviour).
    """
    def count_tokens(text: str) -> int:
        return _base_count_tokens(text, tokens=tokens)

    source_bytes = content.encode("utf-8")
    tree, grammar = _parse_with_best_grammar(source_bytes, language)
    if tree is None:
        return chunk_text_fallback(
            content, file_path, language, config, tokens=tokens,
        )
    if grammar != language:
        logger.debug(
            "%s parses cleaner as %s than as %s; using the %s grammar",
            file_path, grammar, language, grammar,
        )
        # Everything below keys off the grammar that actually parsed the
        # file: node types, symbol extraction, and the language recorded
        # on each chunk.
        language = grammar
    root = tree.root_node

    max_tokens = effective_max if effective_max is not None else config.chunking.max_tokens
    min_tokens = config.chunking.min_tokens
    extractable = set(EXTRACTABLE_NODES.get(language, []))

    chunks: list[Chunk] = []
    covered_ranges: list[tuple[int, int]] = []  # (start_byte, end_byte)

    for node in root.children:
        if node.type not in extractable:
            continue

        node_text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        symbol = _extract_symbol_name(node, language)
        is_class = "class" in node.type or node.type in CLASS_LIKE_NODES

        # For classes, try to extract methods as separate chunks
        if is_class and count_tokens(node_text) > max_tokens:
            class_chunks = _chunk_class_node(
                node, source_bytes, file_path, language, symbol, config,
                tokens, max_tokens,
            )
            if class_chunks:
                chunks.extend(class_chunks)
                covered_ranges.append((node.start_byte, node.end_byte))
                continue

        token_count = count_tokens(node_text)
        chunk_type = _node_to_chunk_type(node.type)

        if token_count <= max_tokens:
            if token_count >= min_tokens:
                chunks.append(Chunk(
                    id=make_chunk_id(file_path, node_text, start_line),
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
                chunk_type, symbol, max_tokens, min_tokens, {}, tokens,
            )
            chunks.extend(sub_chunks)

        covered_ranges.append((node.start_byte, node.end_byte))

    # Collect module-level code (everything not covered by extracted nodes)
    _nl_byte_offsets = [i for i, b in enumerate(source_bytes) if b == ord(b"\n")]

    def _byte_offset_to_line(offset: int) -> int:
        return bisect_right(_nl_byte_offsets, offset - 1) + 1

    module_segments: list[tuple[str, int, int]] = []  # (text, start_line, end_line)
    prev_end = 0

    for start_byte, end_byte in sorted(covered_ranges):
        gap = source_bytes[prev_end:start_byte].decode("utf-8", errors="replace")
        if gap.strip():
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
        module_text = "\n\n".join(seg[0] for seg in module_segments)
        tc = count_tokens(module_text)
        first_line = module_segments[0][1]
        last_line = module_segments[-1][2]

        if tc >= min_tokens:
            if tc <= max_tokens:
                chunks.append(Chunk(
                    id=make_chunk_id(file_path, module_text, first_line),
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
                    tokens,
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
    tokens: Optional["TokenizerWrapper"] = None,
    effective_max: Optional[int] = None,
) -> list[Chunk]:
    """Extract methods from a class node as individual chunks.

    If the class is too large, each method becomes its own chunk.
    The class signature (without method bodies) becomes a separate chunk.

    Args:
        class_node: Tree-sitter class node.
        source_bytes: Raw file bytes for offset-based extraction.
        file_path: Project-relative path.
        language: Detected language.
        class_name: Class symbol name.
        config: Loaded configuration.
        tokens: Tokenizer wrapper. Defaults to the chunkers.common
            resolver's tiktoken fallback.
        effective_max: Safety-factor-adjusted chunk budget. When None,
            ``chunking.max_tokens`` is used (legacy behaviour).
    """
    def count_tokens(text: str) -> int:
        return _base_count_tokens(text, tokens=tokens)

    max_tokens = effective_max if effective_max is not None else config.chunking.max_tokens
    min_tokens = config.chunking.min_tokens
    method_types = set(METHOD_NODES.get(language, []))
    chunks: list[Chunk] = []

    # Find the class body node
    body_node = None
    for child in class_node.children:
        if child.type in ("block", "class_body", "statement_block",
                          "field_declaration_list", "declaration_list",
                          "body", "body_statement", "enum_body"):
            body_node = child
            break

    if body_node is None:
        return []

    method_ranges: list[tuple[int, int]] = []

    for child in body_node.children:
        # A container nested directly inside this one. C++ in particular is
        # written as ``namespace A { namespace B { ... } }``, so the
        # functions are two levels down and a scan of this body alone finds
        # nothing — leaving the whole outer namespace as one chunk. Descend
        # so the definitions inside are reached.
        if child is not body_node and (
            child.type in CLASS_LIKE_NODES or "class" in child.type
        ):
            nested_text = source_bytes[child.start_byte:child.end_byte].decode(
                "utf-8", errors="replace",
            )
            nested_name = _extract_symbol_name(child, language)
            if count_tokens(nested_text) > max_tokens:
                nested_chunks = _chunk_class_node(
                    child, source_bytes, file_path, language, nested_name,
                    config, tokens, max_tokens,
                )
                if nested_chunks:
                    chunks.extend(nested_chunks)
                    method_ranges.append((child.start_byte, child.end_byte))
                    continue
            elif count_tokens(nested_text) >= min_tokens:
                chunks.append(Chunk(
                    id=make_chunk_id(file_path, nested_text, child.start_point[0] + 1),
                    file_path=file_path,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    content=nested_text,
                    chunk_type=_node_to_chunk_type(child.type),
                    language=language,
                    symbol_name=nested_name,
                    token_count=count_tokens(nested_text),
                    metadata={"parent_class": class_name} if class_name else {},
                ))
                method_ranges.append((child.start_byte, child.end_byte))
                continue

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
        elif language == "cpp" and child.type == "field_declaration":
            # C++ field_declaration covers both data members and method
            # declarations. Only treat it as a method if it contains a
            # function_declarator (i.e., it's a method signature, not a
            # variable like `int id_;`).
            for sub in child.children:
                if sub.type == "function_declarator":
                    is_method = True
                    break

        if is_method:
            method_text = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
            method_name = _extract_symbol_name(child, language)
            start_line = child.start_point[0] + 1
            end_line = child.end_point[0] + 1
            tc = count_tokens(method_text)

            if tc <= max_tokens:
                if tc >= min_tokens:
                    chunks.append(Chunk(
                        id=make_chunk_id(file_path, method_text, start_line),
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
                    {"parent_class": class_name}, tokens,
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
        tc = count_tokens(sig_text)
        if tc < min_tokens:
            pass
        elif tc <= max_tokens:
            chunks.insert(0, Chunk(
                id=make_chunk_id(file_path, sig_text, class_node.start_point[0] + 1),
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
        else:
            # When no method or nested container was found, "everything not
            # covered by a method" is the whole class. Emitting that
            # unchecked hands the caller an over-budget chunk which it
            # trusts, and the dispatch-level safety net then cuts it at
            # token boundaries — losing the line numbers, which collapse to
            # the chunk's first line. Split it here instead, where the
            # blank-line boundaries are still available.
            chunks[0:0] = _split_oversized_chunk(
                sig_text, file_path, class_node.start_point[0] + 1, language,
                ChunkType.CLASS, class_name, max_tokens, min_tokens, {}, tokens,
            )

    return chunks
