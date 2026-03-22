"""Integration tests for office document support across modules.

Covers MR review gaps:
- hasher.walk_project_files() applying max_office_file_size_kb only to office extensions
- chunk_file() path for binary formats when office deps are missing
- migrate_config.py extension deduplication and office migration logic
- Centralized OFFICE_EXTENSIONS / BINARY_FORMATS constants consistency
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib.chunker import chunk_file
from lib.constants import BINARY_FORMATS, OFFICE_EXTENSIONS
from lib.config import Config, IndexingConfig
from lib.hasher import walk_project_files
from migrate_config import analyze_config, apply_migrations


# ---------------------------------------------------------------------------
# Constants consistency
# ---------------------------------------------------------------------------

class TestConstantsConsistency:
    """Verify centralized constants match across modules."""

    def test_binary_formats_match_office_extensions(self):
        """BINARY_FORMATS languages correspond to OFFICE_EXTENSIONS file exts."""
        expected_languages = {ext.lstrip(".") for ext in OFFICE_EXTENSIONS}
        assert BINARY_FORMATS == expected_languages

    def test_office_extensions_are_dotted(self):
        """All OFFICE_EXTENSIONS start with a dot."""
        for ext in OFFICE_EXTENSIONS:
            assert ext.startswith("."), f"{ext} missing leading dot"


# ---------------------------------------------------------------------------
# hasher.walk_project_files — office size limit
# ---------------------------------------------------------------------------

class TestWalkProjectFilesOfficeSize:
    """walk_project_files() uses max_office_file_size_kb for office files only."""

    def _setup_project(self, tmp_path: Path) -> Config:
        """Create a minimal project with one .py and one .pdf file."""
        (tmp_path / "small.py").write_text("x = 1")
        (tmp_path / "report.pdf").write_bytes(b"%PDF-" + b"A" * 600 * 1024)
        (tmp_path / "big_code.py").write_text("y = 2\n" * 100_000)

        config = Config()
        config.indexing = IndexingConfig(
            file_extensions=[".py", ".pdf"],
            exclude_patterns=[],
            max_file_size_kb=500,          # 500 KB for code
            max_office_file_size_kb=50000,  # 50 MB for office
            respect_gitignore=False,
        )
        return config

    def test_large_pdf_under_office_limit_included(self, tmp_path):
        """A 600 KB PDF exceeds max_file_size_kb but is under office limit."""
        config = self._setup_project(tmp_path)
        files = walk_project_files(str(tmp_path), config)
        assert "report.pdf" in files

    def test_large_code_file_excluded(self, tmp_path):
        """A code file exceeding max_file_size_kb is excluded."""
        config = self._setup_project(tmp_path)
        files = walk_project_files(str(tmp_path), config)
        assert "big_code.py" not in files

    def test_small_code_file_included(self, tmp_path):
        config = self._setup_project(tmp_path)
        files = walk_project_files(str(tmp_path), config)
        assert "small.py" in files

    def test_pdf_over_office_limit_excluded(self, tmp_path):
        """A PDF exceeding max_office_file_size_kb is excluded."""
        config = self._setup_project(tmp_path)
        config.indexing.max_office_file_size_kb = 0  # 0 KB limit
        files = walk_project_files(str(tmp_path), config)
        assert "report.pdf" not in files


# ---------------------------------------------------------------------------
# chunk_file — binary format with missing deps
# ---------------------------------------------------------------------------

class TestChunkFileMissingOfficeDeps:
    """chunk_file() returns empty list (no crash) when office deps missing."""

    def test_pdf_missing_fitz_returns_empty(self, tmp_path):
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-fake")
        config = Config()
        with patch.dict(sys.modules, {"fitz": None}):
            result = chunk_file("doc.pdf", str(tmp_path), config)
        assert result == []

    def test_docx_missing_docx_returns_empty(self, tmp_path):
        (tmp_path / "doc.docx").write_bytes(b"PK\x03\x04fake")
        config = Config()
        with patch.dict(sys.modules, {"docx": None}):
            result = chunk_file("doc.docx", str(tmp_path), config)
        assert result == []

    def test_pptx_missing_pptx_returns_empty(self, tmp_path):
        (tmp_path / "deck.pptx").write_bytes(b"PK\x03\x04fake")
        config = Config()
        with patch.dict(sys.modules, {"pptx": None}):
            result = chunk_file("deck.pptx", str(tmp_path), config)
        assert result == []


# ---------------------------------------------------------------------------
# migrate_config — office extensions and deduplication
# ---------------------------------------------------------------------------

class TestMigrateConfigOffice:
    """Migration adds office extensions and max_office_file_size_kb correctly."""

    def _base_config(self) -> dict:
        """Config that already has DITA but no office support."""
        return {
            "schema_version": "1.0",
            "embedding": {"device": None, "trust_remote_code": False},
            "search": {
                "default_top_k": 10,
                "default_threshold": 0.3,
                "mode": "hybrid",
                "hybrid_alpha": 0.7,
                "rerank_enabled": False,
                "rerank_model": "BAAI/bge-reranker-v2-m3",
                "rerank_top_n": 10,
            },
            "indexing": {
                "file_extensions": [
                    ".py", ".js", ".ts", ".md",
                    ".dita", ".ditamap",
                ],
            },
        }

    def test_adds_office_extensions(self):
        config = self._base_config()
        migrations = analyze_config(config)
        updated = apply_migrations(config, migrations)
        exts = updated["indexing"]["file_extensions"]
        for ext in [".pdf", ".docx", ".pptx"]:
            assert ext in exts

    def test_adds_max_office_file_size_kb(self):
        config = self._base_config()
        migrations = analyze_config(config)
        updated = apply_migrations(config, migrations)
        assert updated["indexing"]["max_office_file_size_kb"] == 50000

    def test_no_duplicate_extensions_after_migration(self):
        """Running migration on config that already has some office exts."""
        config = self._base_config()
        config["indexing"]["file_extensions"].append(".pdf")  # already has PDF
        migrations = analyze_config(config)
        updated = apply_migrations(config, migrations)
        exts = updated["indexing"]["file_extensions"]
        assert exts.count(".pdf") == 1

    def test_double_migration_no_duplicates(self):
        """Applying migrations twice doesn't create duplicate extensions."""
        config = self._base_config()
        migrations = analyze_config(config)
        updated = apply_migrations(config, migrations)
        # Analyze again on the already-migrated config
        migrations2 = analyze_config(updated)
        updated2 = apply_migrations(updated, migrations2)
        exts = updated2["indexing"]["file_extensions"]
        for ext in exts:
            assert exts.count(ext) == 1, f"Duplicate extension: {ext}"

    def test_skips_office_size_if_already_present(self):
        config = self._base_config()
        config["indexing"]["max_office_file_size_kb"] = 30000
        migrations = analyze_config(config)
        size_migrations = [m for m in migrations if "max_office" in m["field"]]
        assert len(size_migrations) == 0


# ---------------------------------------------------------------------------
# Regression: migrate_config must not depend on chunking third-party deps
# ---------------------------------------------------------------------------

class TestMigrateConfigNoCunkingDeps:
    """migrate_config.py must work even when tiktoken (and other chunking
    deps) are unavailable — it only needs lib.constants (stdlib-only)."""

    def test_analyze_config_without_tiktoken(self):
        """Simulate an environment where tiktoken is not installed."""
        import importlib

        # Temporarily hide tiktoken from the import system
        original = sys.modules.get("tiktoken")
        sys.modules["tiktoken"] = None  # type: ignore[assignment]
        try:
            # Re-import to ensure no transitive tiktoken pull
            import lib.constants as const_mod
            importlib.reload(const_mod)

            # analyze_config should work fine — it only needs constants
            config = {
                "schema_version": "1.0",
                "embedding": {"device": None, "trust_remote_code": False},
                "search": {
                    "default_top_k": 10,
                    "default_threshold": 0.3,
                    "mode": "hybrid",
                    "hybrid_alpha": 0.7,
                    "rerank_enabled": False,
                    "rerank_model": "BAAI/bge-reranker-v2-m3",
                    "rerank_top_n": 10,
                },
                "indexing": {
                    "file_extensions": [".py", ".md"],
                },
            }
            migrations = analyze_config(config)
            assert any("office" in m["reason"].lower() for m in migrations)
        finally:
            # Restore tiktoken
            if original is not None:
                sys.modules["tiktoken"] = original
            else:
                sys.modules.pop("tiktoken", None)
