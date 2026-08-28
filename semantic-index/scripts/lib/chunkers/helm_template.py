"""Definition-aware chunking for Helm template libraries (``.tpl``).

A ``.tpl`` file is a library of named templates — ``{{- define "chart.name"
-}} ... {{- end -}}`` — that the chart's manifests call by name. Each
definition is a self-contained unit doing one job, which makes it exactly
the right size and shape for a chunk, and gives the chunk a real name to
be found by.

The blank-line fallback usually lands close to these boundaries, because
authors separate definitions with blank lines. Usually is not always: a
definition holding a blank line is split down the middle, and a file
written without blank lines between definitions becomes one blob. Reading
the ``define``/``end`` structure removes the guesswork and, unlike blank
lines, names what it finds.

Nothing here renders the template. The nesting of ``{{ if }}``,
``{{ range }}`` and ``{{ with }}`` is tracked only so the ``{{ end }}``
that closes a definition can be told from the ones that close a
conditional inside it.
"""

import re
from typing import NamedTuple, Optional

from .common import build_chunks, count_tokens as _base_count_tokens
from ..config import Config
from ..models import Chunk, ChunkType
from ..tokenizer_resolver import TokenizerWrapper

# Any template action, capturing the keyword that opens it. ``{{-`` and
# ``{{`` are both matched, as is a leading ``/*`` comment marker.
_ACTION_RE = re.compile(r"\{\{-?\s*(?P<keyword>/\*|\w+)")

# The name a definition is registered under: {{- define "chart.name" -}}
_DEFINE_NAME_RE = re.compile(r"\{\{-?\s*(?:define|block)\s+\"(?P<name>[^\"]+)\"")

# Keywords that open a block needing a matching {{ end }}.
_BLOCK_OPENERS = frozenset({"define", "if", "range", "with", "block"})


class _Definition(NamedTuple):
    """One named template and the lines it spans."""

    start: int          # 0-based index of the ``define`` line
    end: int            # 0-based index one past the closing ``end`` line
    name: str


def chunk_helm_template(
    content: str,
    file_path: str,
    language: str,
    config: Config,
    tokens: Optional[TokenizerWrapper] = None,
    effective_max: Optional[int] = None,
) -> list[Chunk]:
    """Split a Helm template library into one chunk per named definition.

    Text outside any definition — file comments, stray directives — is
    gathered into chunks of its own rather than dropped.

    Args:
        content: File text.
        file_path: Project-relative path.
        language: Language identifier recorded on each chunk.
        config: Loaded configuration.
        tokens: Tokenizer wrapper. Defaults to the resolver's tiktoken
            fallback.
        effective_max: Safety-factor-adjusted chunk budget. When None,
            ``chunking.max_tokens`` is used.

    Returns:
        Chunks in file order. Definition chunks carry the template name as
        ``symbol_name`` and as ``template_name`` in their metadata.
    """
    max_tokens = effective_max if effective_max is not None else config.chunking.max_tokens
    min_tokens = config.chunking.min_tokens
    lines = content.split("\n")

    definitions = _find_definitions(lines)
    if not definitions:
        # No named templates: a partial, or a plain text file that happens
        # to carry the extension. Treat the whole file as one unit and let
        # build_chunks divide it if it does not fit.
        return _build(lines, 0, len(lines), None, file_path, language,
                      max_tokens, min_tokens, tokens)

    chunks: list[Chunk] = []
    cursor = 0
    for definition in definitions:
        if definition.start > cursor:
            chunks.extend(_build(
                lines, cursor, definition.start, None, file_path, language,
                max_tokens, min_tokens, tokens,
            ))
        chunks.extend(_build(
            lines, definition.start, definition.end, definition.name,
            file_path, language, max_tokens, min_tokens, tokens,
        ))
        cursor = definition.end

    if cursor < len(lines):
        chunks.extend(_build(
            lines, cursor, len(lines), None, file_path, language,
            max_tokens, min_tokens, tokens,
        ))
    return chunks


def _find_definitions(lines: list[str]) -> list[_Definition]:
    """Locate every top-level ``define`` block and where it closes.

    Depth is counted across the whole file, so an ``end`` is attributed to
    the construct that actually opened it. A definition left unterminated
    — a truncated file, or a template built by string concatenation —
    runs to the end of the file rather than swallowing the ones after it,
    because there is no honest way to tell where the author meant it to
    stop.
    """
    definitions: list[_Definition] = []
    depth = 0
    open_start: Optional[int] = None
    open_name: Optional[str] = None

    for i, line in enumerate(lines):
        for match in _ACTION_RE.finditer(line):
            keyword = match.group("keyword")
            if keyword in _BLOCK_OPENERS:
                if depth == 0 and keyword in ("define", "block"):
                    name_match = _DEFINE_NAME_RE.search(line[match.start():])
                    open_start = i
                    open_name = name_match.group("name") if name_match else None
                depth += 1
            elif keyword == "end":
                depth = max(0, depth - 1)
                if depth == 0 and open_start is not None:
                    definitions.append(_Definition(
                        open_start, i + 1, open_name or f"line {open_start + 1}",
                    ))
                    open_start = None
                    open_name = None

    if open_start is not None:
        definitions.append(_Definition(
            open_start, len(lines), open_name or f"line {open_start + 1}",
        ))
    return definitions


def _build(
    lines: list[str],
    start: int,
    end: int,
    name: Optional[str],
    file_path: str,
    language: str,
    max_tokens: int,
    min_tokens: int,
    tokens: Optional[TokenizerWrapper],
) -> list[Chunk]:
    """Emit one span as chunks, or nothing when it is only whitespace."""
    text = "\n".join(lines[start:end]).strip("\n")
    if not text.strip():
        return []
    if _base_count_tokens(text, tokens=tokens) < min_tokens:
        return []

    return build_chunks(
        text,
        file_path=file_path,
        start_line=start + 1,
        language=language,
        chunk_type=ChunkType.FUNCTION if name else ChunkType.MODULE_LEVEL,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        symbol_name=name,
        metadata={"template_name": name} if name else {},
        tokens=tokens,
    )
