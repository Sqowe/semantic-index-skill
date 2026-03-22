"""Lightweight shared constants for the semantic-index pipeline.

This module has NO third-party dependencies — only stdlib. It is safe
to import from any module, including the migration script and hasher,
without pulling in heavy packages like tiktoken or tree-sitter.
"""

# File extensions for binary office formats (with leading dot).
OFFICE_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".pptx"})

# Language identifiers for binary formats (dot-stripped).
BINARY_FORMATS: frozenset[str] = frozenset({"pdf", "docx", "pptx"})
