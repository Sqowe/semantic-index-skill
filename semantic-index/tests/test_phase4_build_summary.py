"""Tests for Phase 4 — visibility of chunk truncation in the build summary.

Phase 4 surfaces the truncation count that the Embedder's pre-embed
token cap produces: ``build_index.py`` and ``mcp_server.py`` read
``embedder.truncation_stats`` after each batch, accumulate the
total, and add three fields to the success summary:

* ``truncated_chunks`` — int, always present
* ``truncated_files`` — sorted list of project-relative paths
* ``truncation_message`` — only when ``truncated_chunks > 0``,
  a human-readable explanation that the chunk endings are not
  searchable

Affected file paths are also emitted at INFO on the build's
stderr stream so operators can confirm in CI that the count is
zero when expected, and surface the paths without polluting
per-file progress output.
"""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib.chunker import chunk_file
from lib.config import Config, EmbeddingConfig, ChunkingConfig
from lib.embedder import Embedder
from lib.models import Chunk, ChunkType, TruncationRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str, idx: int, file_path: str = "src/x.py") -> Chunk:
    """Create a minimal Chunk for tests."""
    return Chunk(
        id=f"chunk-{idx}",
        file_path=file_path,
        start_line=1,
        end_line=10,
        content=content,
        chunk_type=ChunkType.FUNCTION,
        language="python",
        token_count=len(content.split()),
    )


def _make_config() -> Config:
    """Create a Config with a small max_embed_tokens so we can
    reliably trigger truncation in tests."""
    config = Config()
    config.embedding = EmbeddingConfig(
        provider="openrouter",
        batch_size=4,
        model="test-model",
        dimensions=3,
        max_embed_tokens=10,  # very small to force truncation
    )
    config.chunking = ChunkingConfig(max_tokens=10, overlap_tokens=0, min_tokens=2)
    return config


# ---------------------------------------------------------------------------
# build_index.py — truncation fields in success summary
# ---------------------------------------------------------------------------


class TestBuildIndexTruncationFields:
    """build_index.py surfaces truncation count and affected files."""

    def _run_build(
        self,
        files_to_index: list[str],
        chunks_per_file: dict[str, list[Chunk]],
        truncation_stats: list[TruncationRecord],
        caplog: pytest.LogCaptureFixture,
    ) -> dict:
        """Run main() with mocks and capture the JSON output."""
        from lib.models import FileChange
        from scripts import build_index

        config = _make_config()

        # Mock every external call. We focus on the JSON output and
        # the INFO log stream, so all the heavy machinery is stubbed.
        with patch.object(build_index, "load_config", return_value=config), \
             patch.object(build_index, "ensure_index_dir"), \
             patch.object(
                 build_index, "detect_changes",
                 return_value=FileChange(
                     to_index=files_to_index, to_delete=[], unchanged=0,
                 ),
             ), \
             patch.object(build_index, "BM25Index") as mock_bm25_cls, \
             patch.object(build_index, "VectorStore") as mock_store_cls, \
             patch.object(
                 build_index, "chunk_file",
                 side_effect=lambda fp, *args, **kwargs: chunks_per_file.get(
                     fp, [],
                 ),
             ), \
             patch.object(build_index, "update_manifest"), \
             patch.object(
                 build_index, "Embedder",
             ) as mock_embedder_cls, \
             patch.object(
                 sys, "argv",
                 ["build_index.py", "--project-dir", "/tmp/test"],
             ):
            mock_bm25 = MagicMock()
            mock_bm25.load.return_value = True
            mock_bm25_cls.return_value = mock_bm25
            mock_store = MagicMock()
            mock_store.has_index.return_value = True
            mock_store_cls.return_value = mock_store

            # The mock Embedder's ``embed_chunks`` returns the api
            # call count and seeds ``truncation_stats`` so build_index
            # sees the records.
            mock_embedder = MagicMock()
            mock_embedder.embed_chunks.return_value = 1
            mock_embedder.truncation_stats = list(truncation_stats)
            mock_embedder_cls.return_value = mock_embedder

            with caplog.at_level(logging.INFO, logger="scripts.build_index"):
                # Capture the JSON block printed to stdout.
                import io
                from contextlib import redirect_stdout
                buf = io.StringIO()
                with redirect_stdout(buf):
                    build_index.main()
                # The result is ``json.dumps(result, indent=2)`` — a
                # multi-line block. Find the first ``{`` and parse
                # from there.
                output = buf.getvalue()
                json_start = output.find("{")
                assert json_start != -1, "no JSON output captured"
                return json.loads(output[json_start:])

    def test_truncated_chunks_present_even_when_zero(self, caplog):
        """``truncated_chunks`` is always present, even when nothing
        was truncated. Callers can rely on the field existing."""
        result = self._run_build(
            files_to_index=["src/a.py"],
            chunks_per_file={"src/a.py": [_make_chunk("def f(): pass", 0)]},
            truncation_stats=[],
            caplog=caplog,
        )
        assert "truncated_chunks" in result
        assert result["truncated_chunks"] == 0
        assert "truncated_files" in result
        assert result["truncated_files"] == []
        # No human-readable message when there were no truncations.
        assert "truncation_message" not in result

    def test_truncation_count_appears_in_summary(self, caplog):
        """A single TruncationRecord surfaces as ``truncated_chunks=1``."""
        record = TruncationRecord(
            file_path="src/big_data.py",
            chunk_id="sha256:abc",
            original_tokens=100,
            final_tokens=8,
        )
        result = self._run_build(
            files_to_index=["src/big_data.py"],
            chunks_per_file={
                "src/big_data.py": [_make_chunk("x" * 100, 0, "src/big_data.py")],
            },
            truncation_stats=[record],
            caplog=caplog,
        )
        assert result["truncated_chunks"] == 1
        assert result["truncated_files"] == ["src/big_data.py"]
        # The human-readable message names the count and the consequence.
        assert "1 chunk(s) were shortened" in result["truncation_message"]
        assert "their endings are not searchable" in result["truncation_message"]

    def test_multiple_files_aggregate_correctly(self, caplog):
        """Truncations from multiple files all appear in the summary."""
        records = [
            TruncationRecord(
                file_path=f"src/{name}.py",
                chunk_id=f"sha256:{name}",
                original_tokens=200,
                final_tokens=8,
            )
            for name in ("a", "b", "c")
        ]
        result = self._run_build(
            files_to_index=[r.file_path for r in records],
            chunks_per_file={
                r.file_path: [_make_chunk("x" * 200, i, r.file_path)]
                for i, r in enumerate(records)
            },
            truncation_stats=records,
            caplog=caplog,
        )
        assert result["truncated_chunks"] == 3
        # Files are sorted for deterministic output.
        assert result["truncated_files"] == [
            "src/a.py", "src/b.py", "src/c.py",
        ]

    def test_duplicate_file_paths_deduplicated(self, caplog):
        """Two truncations in the same file produce one entry in
        ``truncated_files``, not two."""
        records = [
            TruncationRecord(
                file_path="src/a.py",
                chunk_id="sha256:1",
                original_tokens=100,
                final_tokens=8,
            ),
            TruncationRecord(
                file_path="src/a.py",
                chunk_id="sha256:2",
                original_tokens=120,
                final_tokens=8,
            ),
        ]
        result = self._run_build(
            files_to_index=["src/a.py"],
            chunks_per_file={
                "src/a.py": [_make_chunk("x" * 200, 0, "src/a.py")],
            },
            truncation_stats=records,
            caplog=caplog,
        )
        assert result["truncated_chunks"] == 2
        assert result["truncated_files"] == ["src/a.py"]

    def test_truncated_files_logged_at_info(self, caplog):
        """Affected file paths are emitted at INFO on the build's
        stderr stream so operators can confirm in CI without
        polluting the per-file progress output. INFO is the right
        level: the build's ``basicConfig(level=INFO)`` filter
        admits these records, but a clean build produces no lines
        (so the stream stays quiet for the common case)."""
        record = TruncationRecord(
            file_path="src/some_big_file.py",
            chunk_id="sha256:abc",
            original_tokens=200,
            final_tokens=8,
        )
        # The shared ``log_truncated_files`` helper uses the
        # ``lib.truncation_visibility`` logger. The build's own
        # logger (``scripts.build_index``) and the helper's logger
        # both end up in the same stderr stream.
        with caplog.at_level(
            logging.INFO, logger="lib.truncation_visibility",
        ):
            self._run_build(
                files_to_index=["src/some_big_file.py"],
                chunks_per_file={
                    "src/some_big_file.py": [
                        _make_chunk("x" * 200, 0, "src/some_big_file.py"),
                    ],
                },
                truncation_stats=[record],
                caplog=caplog,
            )
        # The INFO log line mentions the file and warns about the
        # search-coverage consequence.
        info_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and "src/some_big_file.py" in r.message
            and "truncated" in r.message.lower()
        ]
        assert info_records, (
            "expected an INFO log line about the truncated file"
        )


# ---------------------------------------------------------------------------
# Embedder — truncation_stats is the public read-after-call surface
# ---------------------------------------------------------------------------


class TestEmbedderTruncationStats:
    """The Embedder's ``truncation_stats`` is the public read-after-call
    surface that build_index.py / mcp_server.py consume. Pin the
    behaviour so future refactors don't quietly break Phase 4's
    visibility contract."""

    def test_truncation_stats_empty_initially(self):
        """A fresh Embedder has an empty ``truncation_stats`` list."""
        config = _make_config()
        with patch("lib.embedder.create_provider"):
            embedder = Embedder(config, project_dir=None)
        assert embedder.truncation_stats == []

    def test_truncation_stats_records_after_pretuncate(self):
        """A chunk above the token budget produces a TruncationRecord
        on ``truncation_stats`` with the original and final token
        counts recorded."""
        config = _make_config()
        with patch("lib.embedder.create_provider") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.embed_texts = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
            mock_factory.return_value = mock_provider
            embedder = Embedder(config, project_dir=None)

        # Build a chunk that is over the 10-token budget by a wide
        # margin so truncation fires.
        long_content = " ".join(["word"] * 100)
        chunk = _make_chunk(long_content, 0)
        embedder.embed_chunks([chunk])

        assert len(embedder.truncation_stats) == 1
        record = embedder.truncation_stats[0]
        assert record.file_path == "src/x.py"
        assert record.original_tokens > 10
        assert record.final_tokens <= 10
        # Chunk-level metadata advertises the loss.
        assert chunk.metadata.get("truncated") is True
        assert chunk.metadata.get("original_token_count") == record.original_tokens
        # Full text preserved in metadata.
        assert chunk.metadata.get("original_content") == long_content
