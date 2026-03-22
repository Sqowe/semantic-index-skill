"""Chunking dispatch module.

Routes files to the appropriate chunking strategy based on language:
- Markdown files: header-based splitting (chunkers/markdown.py)
- Python/JS/TS: Tree-sitter AST-aware splitting (chunkers/code.py)
- Other: blank-line fallback splitting (chunkers/common.py)

Shared helpers live in chunkers/common.py to avoid circular imports.
"""

import logging
import os
from pathlib import Path

from .chunkers.common import chunk_text_fallback, detect_language
from .config import Config
from .models import Chunk

logger = logging.getLogger(__name__)

# Languages supported by Tree-sitter (Phase 1 + Phase 2)
TREESITTER_LANGUAGES = {
    "python", "javascript", "typescript",
    "go", "rust", "java", "c", "cpp", "ruby", "php",
}


# Binary formats that require specialized extraction (not UTF-8 text)
BINARY_FORMATS = {"pdf", "docx", "pptx"}


def chunk_file(
    file_path: str,
    project_dir: str,
    config: Config,
) -> list[Chunk]:
    """Chunk a single file into semantically meaningful pieces.

    Dispatches to the appropriate chunking strategy based on language:
    - Office documents (PDF/DOCX/PPTX): binary extraction via office chunker
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
    language = detect_language(file_path)

    # Binary formats: delegate to office chunker (handles its own file I/O)
    if language in BINARY_FORMATS:
        from .chunkers.office import chunk_office
        return chunk_office(abs_path, file_path, language, config)

    try:
        content = Path(abs_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Failed to read file %s: %s", file_path, exc)
        return []

    if not content.strip():
        return []

    if language == "markdown":
        from .chunkers.markdown import chunk_markdown
        return chunk_markdown(content, file_path, config)
    elif language in ("dita", "ditamap"):
        from .chunkers.dita import chunk_dita
        return chunk_dita(content, file_path, language, config)
    elif language in TREESITTER_LANGUAGES:
        from .chunkers.code import chunk_code_with_treesitter
        return chunk_code_with_treesitter(content, file_path, language, config)
    else:
        return chunk_text_fallback(content, file_path, language, config)
