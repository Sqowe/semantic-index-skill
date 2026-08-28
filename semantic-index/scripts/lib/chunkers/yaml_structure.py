"""Reading the structure out of a YAML file without parsing it.

Helm charts are the bulk of the ``.yaml`` in most repositories, and they
are Go templates that are not valid YAML until rendered — a real parser
would reject exactly the files the chunker most needs to handle. So the
structure is read straight from the text: which lines open a key or a
sequence entry, how deeply they sit, and where one entry ends and the
next begins.

Kept apart from the chunking in ``yaml_config.py``: this module answers
what the structure *is*, that one decides where to cut it.
"""

import re
from typing import NamedTuple, Optional


# A mapping key: ``name:``, ``name: value``, ``"quoted name": value``.
# The key must not contain a ``{`` so a Go template line such as
# ``{{- if .Values.x }}`` is never mistaken for one.
_MAP_KEY_RE = re.compile(r"^(?P<key>[^#\s{][^:{]*):(?:\s|$)")

# A sequence entry: ``- value`` or a bare ``-``.
_LIST_ITEM_RE = re.compile(r"^-(?:\s|$)")

# A document separator: ``---`` alone, or followed by a comment/directive.
_DOC_SEPARATOR_RE = re.compile(r"^---\s*(?:#.*)?$")


class Entry(NamedTuple):
    """One structural entry: a mapping key or a sequence item."""

    start: int   # 0-based index of the entry's first line, comments included
    end: int     # 0-based index one past the entry's last line
    key: str     # the key name, or "-" for a sequence item


def split_documents(lines: list[str]) -> list[tuple[int, int]]:
    """Find the ``---`` separated documents in a file.

    Returns:
        ``(start, end)`` line index pairs, one per document with content.
        A file with no separator yields a single pair spanning it.
    """
    boundaries = [
        i for i, line in enumerate(lines) if _DOC_SEPARATOR_RE.match(line)
    ]
    if not boundaries:
        return [(0, len(lines))]

    documents: list[tuple[int, int]] = []
    starts = [0] + [b + 1 for b in boundaries]
    ends = boundaries + [len(lines)]
    for start, end in zip(starts, ends):
        if any(lines[i].strip() for i in range(start, end)):
            documents.append((start, end))
    return documents or [(0, len(lines))]


def entry_key(stripped: str) -> Optional[str]:
    """Return the key a line opens, or None if it opens nothing.

    A sequence item reports ``"-"``: the position in the list is not a
    name, and numbering it would make chunk identity depend on unrelated
    edits earlier in the same list.
    """
    # Sequence first: ``- name: proxy`` opens a list entry, not a key
    # called "- name". The keys inside it are found one level down.
    if _LIST_ITEM_RE.match(stripped):
        return "-"
    match = _MAP_KEY_RE.match(stripped)
    if match:
        return match.group("key").strip().strip("'\"")
    return None


def is_structural(line: str) -> bool:
    """Whether a line's indentation says anything about the structure.

    Blank lines and comments do not. Neither do Go template directives:
    Helm charts routinely write ``{{- include "chart.labels" . | nindent
    4 }}`` flush against the left margin while it stands for content
    nested several levels deep, so its column is not evidence of
    anything. Counting those would make every block look top-level and
    defeat the split entirely.
    """
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith(("#", "{{"))


def base_indent(lines: list[str], start: int, end: int) -> Optional[int]:
    """Smallest indentation among the structural lines of a region."""
    indents = [
        len(line) - len(line.lstrip())
        for line in lines[start:end]
        if is_structural(line)
    ]
    return min(indents) if indents else None


def entries(lines: list[str], start: int, end: int) -> list[Entry]:
    """Find the entries at the outermost indentation of a region.

    Comment lines directly above an entry belong to it — they are almost
    always its documentation, and splitting them off strands both halves.
    """
    base = base_indent(lines, start, end)
    if base is None:
        return []

    found: list[Entry] = []
    open_start: Optional[int] = None
    open_key: Optional[str] = None
    comment_start: Optional[int] = None

    for i in range(start, end):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if comment_start is None:
                comment_start = i
            continue

        indent = len(lines[i]) - len(lines[i].lstrip())
        key = (
            entry_key(stripped)
            if indent == base and is_structural(lines[i])
            else None
        )
        if key is not None:
            boundary = comment_start if comment_start is not None else i
            if open_start is not None and open_key is not None:
                found.append(Entry(open_start, boundary, open_key))
            open_start = boundary
            open_key = key
        comment_start = None

    if open_start is not None and open_key is not None:
        found.append(Entry(open_start, end, open_key))
    return found


def render_path(key_path: list[str]) -> str:
    """Render a key path for humans: sequence steps show as ``[]``."""
    rendered = ""
    for key in key_path:
        if key == "-":
            rendered += "[]"
        else:
            rendered += f".{key}" if rendered else key
    return rendered
