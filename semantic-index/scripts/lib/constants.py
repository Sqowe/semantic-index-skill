"""Lightweight shared constants for the semantic-index pipeline.

This module has NO third-party dependencies — only stdlib. It is safe
to import from any module, including the migration script and hasher,
without pulling in heavy packages like tiktoken or tree-sitter.
"""

from pathlib import PurePosixPath

# File extensions for binary office formats (with leading dot).
OFFICE_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".pptx"})

# Language identifiers for binary formats (dot-stripped).
BINARY_FORMATS: frozenset[str] = frozenset({"pdf", "docx", "pptx"})

# C++ source and header suffixes beyond the original ".cpp"/".hpp" pair.
# Kept here so the default config, the migration script, and the language
# map cannot drift apart. ".txx" is a template implementation header —
# C++ definitions that a ".h" includes.
CPP_EXTENSIONS: tuple[str, ...] = (".cc", ".cxx", ".hh", ".hxx", ".txx")

# Structured-configuration suffixes. YAML is chunked by indentation and
# ".tpl" by Helm ``define`` blocks; see chunkers/yaml_config.py and
# chunkers/helm_template.py.
CONFIG_EXTENSIONS: tuple[str, ...] = (".yaml", ".yml", ".tpl")


def path_matches_glob(file_path: str, glob: str) -> bool:
    """Match a project-relative file path against a glob pattern.

    Uses proper globstar semantics (``pathlib.PurePosixPath.full_match``):
      * ``**`` matches any number of path segments (including zero), so
        ``src/**`` matches everything under ``src/``.
      * ``*`` matches within a single segment only and does NOT cross ``/``,
        so ``src/*`` matches direct children of ``src/`` but not nested files.

    Stored paths are relative to the project root and use OS separators;
    both the path and the glob are normalised to forward slashes so the
    matcher behaves identically on Windows and POSIX.

    Args:
        file_path: Project-relative file path (e.g. ``src/auth/login.py``).
        glob: Glob pattern (e.g. ``src/**``, ``src/*``, ``**/*.py``).

    Returns:
        True if the path matches the glob, False otherwise.
    """
    if not glob:
        return True
    normalised_path = file_path.replace("\\", "/")
    normalised_glob = glob.replace("\\", "/")
    return PurePosixPath(normalised_path).full_match(normalised_glob)
