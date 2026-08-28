"""Indentation-aware chunking for YAML configuration.

YAML carries its structure in indentation rather than in blank lines, and
most of it in practice has almost no blank lines at all — measured across
one estate of Helm charts and Kubernetes manifests, the median file was
2.6% blank. The generic blank-line fallback therefore cuts such a file at
arbitrary token boundaries, mid-key and sometimes mid-word, producing
chunks that match nothing.

This chunker splits on the structure instead. A file is divided into
documents at ``---``, each document into its top-level keys, and any key
whose block is still over budget is divided again into its own children,
as deep as it takes. A chunk that is not a whole top-level key carries a
breadcrumb comment naming the path it came from, so ``enabled: true``
arrives as ``# spec.tls.enabled`` rather than as two anonymous words.

Reading the structure out of the text is ``yaml_structure``'s job; this
module decides where to cut.
"""

from typing import Callable, NamedTuple, Optional

from .common import build_chunks, count_tokens as _base_count_tokens
from .yaml_structure import entries, render_path, split_documents
from ..config import Config
from ..models import Chunk, ChunkType
from ..tokenizer_resolver import TokenizerWrapper

# How deep the recursive split goes before giving up and hard-splitting.
# Real configuration nests perhaps six levels; anything deeper is either
# generated or pathological, and the token-boundary split is the honest
# answer for both.
MAX_SPLIT_DEPTH = 8


class _Context(NamedTuple):
    """Everything the recursive splitter needs but does not decide."""

    file_path: str
    language: str
    max_tokens: int
    min_tokens: int
    tokens: Optional[TokenizerWrapper]
    count: Callable[[str], int]


def chunk_yaml(
    content: str,
    file_path: str,
    language: str,
    config: Config,
    tokens: Optional[TokenizerWrapper] = None,
    effective_max: Optional[int] = None,
) -> list[Chunk]:
    """Split a YAML file into chunks along its own structure.

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
        Chunks in file order. Each carries ``key_path`` in its metadata —
        the sequence of keys leading to it, empty for a whole document —
        and ``doc_index`` when the file holds several documents.
    """
    ctx = _Context(
        file_path=file_path,
        language=language,
        max_tokens=effective_max if effective_max is not None else config.chunking.max_tokens,
        min_tokens=config.chunking.min_tokens,
        tokens=tokens,
        count=lambda text: _base_count_tokens(text, tokens=tokens),
    )

    lines = content.split("\n")
    documents = split_documents(lines)
    multi_document = len(documents) > 1

    chunks: list[Chunk] = []
    for doc_index, (doc_start, doc_end) in enumerate(documents):
        extra = {"doc_index": doc_index} if multi_document else {}
        chunks.extend(_chunk_region(lines, doc_start, doc_end, [], ctx, 0, extra))
    return chunks


def _breadcrumb(key_path: list[str]) -> str:
    """Render a key path as the comment prepended to a nested chunk."""
    return "# " + render_path(key_path)


def _build(
    lines: list[str],
    start: int,
    end: int,
    key_path: list[str],
    ctx: _Context,
    extra_metadata: dict,
) -> list[Chunk]:
    """Emit one region as chunks, prepending a breadcrumb when nested.

    A chunk that is a whole top-level key needs no breadcrumb: its own
    first line names it. A deeper one does, which is why the file's text
    and the chunk's text are not identical for nested chunks —
    ``start_line`` still points at the first real line, not the
    breadcrumb.
    """
    text = "\n".join(lines[start:end]).strip("\n")
    if not text.strip():
        return []

    body = f"{_breadcrumb(key_path)}\n{text}" if key_path else text
    metadata = {"key_path": list(key_path), **extra_metadata}
    return build_chunks(
        body,
        file_path=ctx.file_path,
        start_line=start + 1,
        language=ctx.language,
        chunk_type=ChunkType.CONFIG_BLOCK,
        max_tokens=ctx.max_tokens,
        min_tokens=ctx.min_tokens,
        symbol_name=render_path(key_path) if key_path else None,
        metadata=metadata,
        tokens=ctx.tokens,
    )


def _chunk_region(
    lines: list[str],
    start: int,
    end: int,
    key_path: list[str],
    ctx: _Context,
    depth: int,
    extra_metadata: dict,
) -> list[Chunk]:
    """Chunk one region, dividing it by its own structure when oversized.

    A region within budget becomes a single chunk. An oversized one is
    divided into the entries directly inside it; consecutive entries are
    grouped up to the budget, and any single entry still too large is
    divided again one level deeper.

    The region's own first line joins the first group rather than being
    dropped, because on a sequence item it carries real content
    (``- name: proxy``) and not just a key.
    """
    text = "\n".join(lines[start:end])
    if not text.strip():
        return []

    if ctx.count(text) <= ctx.max_tokens or depth >= MAX_SPLIT_DEPTH:
        return _build(lines, start, end, key_path, ctx, extra_metadata)

    children = entries(lines, start + 1, end)
    if not children:
        # Nothing to divide on — a long scalar or a block literal.
        # build_chunks falls back to token boundaries, which is the best
        # available answer for text with no structure left in it.
        return _build(lines, start, end, key_path, ctx, extra_metadata)

    if len(children) == 1:
        # One child holding the whole region: a sequence with a single
        # long item (``containers:`` with one container is the common
        # case), or a chain of single-key nesting. Descend into it —
        # the region loses a line each time, and MAX_SPLIT_DEPTH stops
        # the walk if the file is nested pathologically deep.
        return _chunk_region(
            lines, children[0].start, children[0].end,
            key_path + [children[0].key], ctx, depth + 1, extra_metadata,
        )

    chunks: list[Chunk] = []
    group_start: Optional[int] = None
    group_end = 0
    group_tokens = 0
    # The region's own opening lines, which no child claims. They join
    # whatever is emitted first so nothing is lost.
    header_start: Optional[int] = start if children[0].start > start else None

    def flush() -> None:
        nonlocal group_start, group_tokens
        if group_start is not None:
            chunks.extend(_build(
                lines, group_start, group_end, key_path, ctx, extra_metadata,
            ))
        group_start = None
        group_tokens = 0

    def claim_header(default: int) -> int:
        """Start a group at the unclaimed header if there is one."""
        nonlocal header_start
        if header_start is None:
            return default
        start_line, header_start = header_start, None
        return start_line

    for entry in children:
        entry_text = "\n".join(lines[entry.start:entry.end])
        entry_tokens = ctx.count(entry_text)

        if entry_tokens > ctx.max_tokens:
            flush()
            if header_start is not None:
                # Emit the header on its own rather than folding it into a
                # child it does not belong to. Usually a lone ``key:`` line,
                # which min_tokens then drops.
                chunks.extend(_build(
                    lines, claim_header(entry.start), entry.start,
                    key_path, ctx, extra_metadata,
                ))
            chunks.extend(_chunk_region(
                lines, entry.start, entry.end, key_path + [entry.key],
                ctx, depth + 1, extra_metadata,
            ))
            continue

        if group_start is not None and group_tokens + entry_tokens > ctx.max_tokens:
            flush()
        if group_start is None:
            group_start = claim_header(entry.start)
        group_end = entry.end
        group_tokens += entry_tokens

    flush()
    return chunks
