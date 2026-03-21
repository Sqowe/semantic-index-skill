"""XML-aware DITA documentation chunking.

Parses DITA XML files using Python's built-in xml.etree.ElementTree.
Supports topic types: topic, concept, task, reference, glossentry,
troubleshooting, plus specializations via the DITA class attribute.

For .ditamap files, extracts the topicref navigation hierarchy as a
single map overview chunk.

No external dependencies required.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Optional

from .common import count_tokens, get_tokenizer, make_chunk_id
from ..config import Config
from ..models import Chunk, ChunkType

logger = logging.getLogger(__name__)

_TOPIC_ELEMENTS = frozenset({
    "topic", "concept", "task", "reference",
    "glossentry", "troubleshooting",
})
_SKIP_ELEMENTS = frozenset({
    "prolog", "related-links", "link", "topicmeta",
    "navref", "anchor", "data", "data-about", "foreign", "unknown",
})
_BODY_ELEMENTS = frozenset({
    "body", "conbody", "taskbody", "refbody", "glossBody", "troublebody",
})

# DITA class tokens for structural element resolution (specializations)
_TITLE_CLASS = "topic/title"
_SHORTDESC_CLASS = "topic/shortdesc"
_BODY_CLASSES = frozenset({
    "topic/body", "concept/conbody", "task/taskbody",
    "reference/refbody", "glossentry/glossBody", "troubleshooting/troublebody",
})
_SECTION_CLASS = "topic/section"
_PROLOG_CLASS = "topic/prolog"

# Safety limit: reject XML content larger than 10 MB to mitigate
# entity-expansion / billion-laughs DoS from untrusted repos.
_MAX_XML_BYTES = 10 * 1024 * 1024


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix from a tag name."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def _find_child(parent: ET.Element, tag_name: str, class_token: str) -> Optional[ET.Element]:
    """Find a direct child by tag name OR DITA class attribute.

    Handles both standard DITA elements and specializations. Also
    namespace-agnostic — compares stripped tag names.
    """
    for child in parent:
        if _strip_ns(child.tag) == tag_name:
            return child
        if class_token and class_token in child.get("class", ""):
            return child
    return None


def _find_child_by_classes(
    parent: ET.Element, tag_names: frozenset[str], class_tokens: frozenset[str],
) -> Optional[ET.Element]:
    """Find a direct child matching any of the tag names or class tokens."""
    for child in parent:
        if _strip_ns(child.tag) in tag_names:
            return child
        cls = child.get("class", "")
        if any(tok in cls for tok in class_tokens):
            return child
    return None


def _is_topic(elem: ET.Element) -> bool:
    """Check if element is a DITA topic (by tag or class attribute)."""
    if _strip_ns(elem.tag) in _TOPIC_ELEMENTS:
        return True
    return "topic/topic" in elem.get("class", "")


def _is_section(elem: ET.Element) -> bool:
    """Check if element is a DITA section (by tag or class attribute)."""
    if _strip_ns(elem.tag) == "section":
        return True
    return _SECTION_CLASS in elem.get("class", "")


def _get_lang(elem: ET.Element) -> Optional[str]:
    """Extract xml:lang, checking the standard namespace form first."""
    return (elem.get("{http://www.w3.org/XML/1998/namespace}lang")
            or elem.get("xml:lang"))


def _extract_text(elem: ET.Element) -> str:
    """Recursively extract readable text, skipping structural elements."""
    tag = _strip_ns(elem.tag)
    cls = elem.get("class", "")
    if tag in _SKIP_ELEMENTS or _PROLOG_CLASS in cls:
        return ""
    parts: list[str] = []
    if elem.text and elem.text.strip():
        parts.append(elem.text.strip())
    for child in elem:
        child_tag = _strip_ns(child.tag)
        child_cls = child.get("class", "")
        if child_tag in _SKIP_ELEMENTS or _PROLOG_CLASS in child_cls:
            continue
        # Skip nested topics — they are chunked separately
        if _is_topic(child):
            continue
        child_text = _extract_text(child)
        if child_text:
            parts.append(child_text)
        if child.tail and child.tail.strip():
            parts.append(child.tail.strip())
    return "\n".join(parts)


def _extract_prolog(topic: ET.Element) -> dict:
    """Extract metadata from <prolog>: keywords, audience, category, author."""
    meta: dict = {}
    prolog = _find_child(topic, "prolog", _PROLOG_CLASS)
    if prolog is None:
        return meta
    keywords: list[str] = []
    for el in prolog.iter():
        tag = _strip_ns(el.tag)
        txt = (el.text or "").strip()
        if tag == "keyword" and txt:
            keywords.append(txt)
        elif tag == "audience":
            aud = el.get("type", "") or txt
            if aud:
                meta["audience"] = aud
        elif tag == "category" and txt:
            meta["category"] = txt
        elif tag == "author" and txt:
            meta["author"] = txt
    if keywords:
        meta["keywords"] = keywords
    return meta


def _has_conrefs(topic: ET.Element) -> bool:
    """Check if any element uses conref or conkeyref."""
    return any(el.get("conref") or el.get("conkeyref") for el in topic.iter())


def _prolog_prefix(meta: dict) -> str:
    """Build a text prefix from prolog metadata for embedding enrichment."""
    parts: list[str] = []
    if "audience" in meta:
        parts.append(f"[audience: {meta['audience']}]")
    if "keywords" in meta:
        parts.append(f"[keywords: {', '.join(meta['keywords'])}]")
    if "category" in meta:
        parts.append(f"[category: {meta['category']}]")
    return " ".join(parts)


def _make_chunk(
    file_path: str, text: str, chunk_type: ChunkType,
    language: str, metadata: dict, title: Optional[str] = None,
    line_hint: int = 1,
) -> Chunk:
    """Build a Chunk with standard field computation."""
    meta = {**metadata, "line_approximate": True}
    return Chunk(
        id=make_chunk_id(file_path, text, line_hint),
        file_path=file_path,
        start_line=line_hint,
        end_line=line_hint + text.count("\n"),
        content=text,
        chunk_type=chunk_type,
        language=language,
        symbol_name=title or None,
        token_count=count_tokens(text),
        metadata=meta,
    )


def _chunk_topic(
    topic: ET.Element, file_path: str, config: Config,
    inherited_lang: Optional[str] = None,
) -> list[Chunk]:
    """Chunk a single DITA topic (not its nested child topics).

    Produces one chunk if it fits within max_tokens, otherwise splits
    at <section> boundaries within the body.
    """
    max_tokens = config.chunking.max_tokens
    min_tokens = config.chunking.min_tokens
    topic_tag = _strip_ns(topic.tag)
    lang = _get_lang(topic) or inherited_lang
    prolog_meta = _extract_prolog(topic)

    # Resolve title and shortdesc via class-aware lookup
    title_el = _find_child(topic, "title", _TITLE_CLASS)
    title = ""
    if title_el is not None:
        title = _extract_text(title_el) if len(title_el) > 0 else (title_el.text or "").strip()

    sd_el = _find_child(topic, "shortdesc", _SHORTDESC_CLASS)
    shortdesc = _extract_text(sd_el) if sd_el is not None else ""

    # Resolve body via class-aware lookup
    body_elem = _find_child_by_classes(topic, _BODY_ELEMENTS, _BODY_CLASSES)
    body_text = _extract_text(body_elem) if body_elem is not None else ""

    # Assemble full text
    prefix = _prolog_prefix(prolog_meta)
    full_text = "\n".join(p for p in [prefix, title, shortdesc, body_text] if p)
    if not full_text.strip():
        return []

    meta: dict = {"topic_type": topic_tag, "prolog": prolog_meta}
    if lang:
        meta["xml_lang"] = lang
    if _has_conrefs(topic):
        meta["has_conrefs"] = True

    tc = count_tokens(full_text)
    if tc <= max_tokens:
        if tc < min_tokens:
            return []
        return [_make_chunk(file_path, full_text, ChunkType.DITA_TOPIC, "dita", meta, title)]

    if body_elem is None:
        return [_make_chunk(file_path, full_text, ChunkType.DITA_TOPIC, "dita", meta, title)]

    return _split_by_sections(body_elem, title, shortdesc, prefix, file_path, config, meta)


def _truncate_text(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate text to fit within max_tokens. Returns (text, was_truncated)."""
    if count_tokens(text) <= max_tokens:
        return text, False
    # Try line-level truncation first
    lines = text.split("\n")
    kept: list[str] = []
    tokens = 0
    for line in lines:
        line_tokens = count_tokens(line)
        if tokens + line_tokens > max_tokens and kept:
            break
        kept.append(line)
        tokens += line_tokens
    result = "\n".join(kept)
    # If still over (single long line), hard-truncate by tokens
    if count_tokens(result) > max_tokens:
        enc = get_tokenizer()
        token_ids = enc.encode(result)[:max_tokens]
        result = enc.decode(token_ids)
    return result, True


def _split_by_sections(
    body: ET.Element, title: str, shortdesc: str, prefix: str,
    file_path: str, config: Config, metadata: dict,
) -> list[Chunk]:
    """Split an oversized topic body at <section> boundaries."""
    max_tokens = config.chunking.max_tokens
    min_tokens = config.chunking.min_tokens
    ctx = "\n".join(p for p in [prefix, title] if p)
    chunks: list[Chunk] = []
    intro_parts: list[str] = [shortdesc] if shortdesc else []
    sec_idx = 0

    def _flush(parts: list[str]) -> None:
        if not parts:
            return
        text = "\n".join([ctx] + parts) if ctx else "\n".join(parts)
        if count_tokens(text) < min_tokens:
            return
        text, truncated = _truncate_text(text, max_tokens)
        meta = {**metadata, "truncated": True} if truncated else metadata
        chunks.append(_make_chunk(
            file_path, text, ChunkType.DITA_TOPIC, "dita", meta, title,
        ))

    for child in body:
        child_tag = _strip_ns(child.tag)
        child_cls = child.get("class", "")
        if child_tag in _SKIP_ELEMENTS or _PROLOG_CLASS in child_cls:
            continue
        # Skip nested topics
        if _is_topic(child):
            continue
        if _is_section(child):
            _flush(intro_parts)
            intro_parts = []
            sec_text = _extract_text(child)
            if not sec_text.strip():
                continue
            sec_idx += 1
            full = f"{ctx}\n{sec_text}" if ctx else sec_text
            full, truncated = _truncate_text(full, max_tokens)
            if count_tokens(full) >= min_tokens:
                sec_meta = {**metadata, "section_index": sec_idx}
                if truncated:
                    sec_meta["truncated"] = True
                chunks.append(_make_chunk(
                    file_path, full, ChunkType.DITA_TOPIC, "dita",
                    sec_meta, title, sec_idx,
                ))
        else:
            child_text = _extract_text(child)
            if child_text.strip():
                intro_parts.append(child_text)

    _flush(intro_parts)
    return chunks


def _chunk_ditamap(
    root: ET.Element, file_path: str, config: Config,
) -> list[Chunk]:
    """Parse a .ditamap into a single map overview chunk."""
    min_tokens = config.chunking.min_tokens
    # Namespace-agnostic title lookup
    title_el = _find_child(root, "title", _TITLE_CLASS)
    map_title = ""
    if title_el is not None:
        map_title = (title_el.text or "").strip()

    lines: list[str] = []
    if map_title:
        lines.extend([f"Map: {map_title}", ""])

    def _walk(parent: ET.Element, depth: int = 0) -> None:
        for child in parent:
            if _strip_ns(child.tag) != "topicref":
                continue
            href = child.get("href", "")
            navtitle = child.get("navtitle", "")
            keys = child.get("keys", "")
            # Try nested topicmeta/navtitle (namespace-agnostic)
            if not navtitle:
                tm = _find_child(child, "topicmeta", "")
                if tm is not None:
                    nt = _find_child(tm, "navtitle", "")
                    if nt is not None and nt.text:
                        navtitle = nt.text.strip()
            parts = []
            if navtitle:
                parts.append(navtitle)
            if href:
                parts.append(f"({href})")
            if keys:
                parts.append(f"[keys: {keys}]")
            if parts:
                lines.append(f"{'  ' * depth}- {' '.join(parts)}")
            _walk(child, depth + 1)

    _walk(root)
    if not lines:
        return []

    full_text = "\n".join(lines)
    if count_tokens(full_text) < min_tokens:
        return []

    max_tokens = config.chunking.max_tokens
    full_text, truncated = _truncate_text(full_text, max_tokens)

    lang = _get_lang(root)
    meta: dict = {"topic_type": "map"}
    if lang:
        meta["xml_lang"] = lang
    if truncated:
        meta["truncated"] = True

    return [_make_chunk(
        file_path, full_text, ChunkType.DITA_MAP, "ditamap", meta, map_title,
    )]


def _collect_topics(root: ET.Element) -> list[ET.Element]:
    """Collect all topic elements in document order, handling nesting.

    Walks the tree depth-first. Every element that is a DITA topic
    is collected, regardless of nesting depth. This ensures nested
    topics inside a parent topic are not silently dropped.
    """
    topics: list[ET.Element] = []
    for elem in root.iter():
        if _is_topic(elem):
            topics.append(elem)
    return topics


def chunk_dita(
    content: str, file_path: str, language: str, config: Config,
) -> list[Chunk]:
    """Chunk a DITA XML file into semantically meaningful pieces.

    For .dita files: extracts all topics (including nested) and chunks each.
    For .ditamap files: extracts navigation structure as a map overview.

    Args:
        content: Raw XML content of the file.
        file_path: Relative path from project root.
        language: Detected language ("dita" or "ditamap").
        config: Loaded configuration.

    Returns:
        List of Chunk objects. May be empty if the file has no content.
    """
    if not content or not content.strip():
        return []
    if len(content.encode("utf-8", errors="replace")) > _MAX_XML_BYTES:
        logger.warning(
            "DITA file %s exceeds %d byte safety limit, skipping",
            file_path, _MAX_XML_BYTES,
        )
        return []
    # Reject entity declarations to prevent entity expansion attacks.
    # DOCTYPE is safe with ElementTree (it ignores DTDs), and virtually
    # every valid DITA file has one, so we only block <!ENTITY>.
    if "<!ENTITY" in content:
        logger.warning(
            "DITA file %s contains entity declarations, skipping for safety",
            file_path,
        )
        return []
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        logger.warning("Failed to parse DITA XML %s: %s", file_path, exc)
        return []

    if language == "ditamap" or _strip_ns(root.tag) in ("map", "bookmap"):
        return _chunk_ditamap(root, file_path, config)

    inherited_lang = _get_lang(root)
    chunks: list[Chunk] = []

    # Collect ALL topics (root + nested) in document order
    for topic_elem in _collect_topics(root):
        topic_lang = _get_lang(topic_elem) or inherited_lang
        chunks.extend(_chunk_topic(topic_elem, file_path, config, topic_lang))

    if not chunks:
        logger.debug("No DITA topics found in %s", file_path)
    return chunks
