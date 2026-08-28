"""Tests for ``lib.truncation_visibility`` — the shared helper that
build_index.py and mcp_server.py both call to surface truncation
visibility. Centralising this logic eliminates the duplication
flagged in the Phase 4 review (a fix in one place must propagate to
both call sites) and gives us a single place to unit-test the
delta math, the result-dict shape, and the DEBUG log output.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib.embedder import Embedder
from lib.models import TruncationRecord
from lib.truncation_visibility import (
    accumulate_truncation,
    build_truncation_summary,
    log_truncated_files,
)


# ---------------------------------------------------------------------------
# accumulate_truncation — the per-batch delta + file-set update
# ---------------------------------------------------------------------------


class TestAccumulateTruncation:
    """The delta math must be correct across multiple batches and
    must guard against a non-positive delta (a future refactor that
    resets or shrinks ``truncation_stats`` must not corrupt the
    running totals)."""

    def _make_embedder(self, stats: list[TruncationRecord]) -> Embedder:
        """Build an Embedder mock with the given truncation_stats."""
        embedder = MagicMock(spec=Embedder)
        embedder.truncation_stats = list(stats)
        return embedder

    def test_zero_truncations_no_op(self):
        """Empty truncation_stats leaves totals unchanged."""
        embedder = self._make_embedder([])
        total = accumulate_truncation(embedder, 0, set())
        assert total == 0

    def test_single_batch_delta(self):
        """A batch that produces N records increments the total by N
        and adds the affected file paths to the set."""
        records = [
            TruncationRecord(
                file_path="src/a.py",
                chunk_id=f"sha256:{i}",
                original_tokens=100,
                final_tokens=8,
            )
            for i in range(3)
        ]
        embedder = self._make_embedder(records)
        files: set[str] = set()
        total = accumulate_truncation(embedder, 0, files)
        assert total == 3
        assert files == {"src/a.py"}

    def test_multiple_batches_accumulate_correctly(self):
        """When ``truncation_stats`` grows across batches, the delta
        is computed correctly for each batch and the file set
        accumulates the union."""
        # Batch 1: 2 records, file a.py
        records_batch1 = [
            TruncationRecord(
                file_path="src/a.py",
                chunk_id=f"sha256:1-{i}",
                original_tokens=100,
                final_tokens=8,
            )
            for i in range(2)
        ]
        embedder = self._make_embedder(records_batch1)
        files: set[str] = set()
        total = accumulate_truncation(embedder, 0, files)
        assert total == 2
        assert files == {"src/a.py"}

        # Batch 2: 3 more records, files a.py (duplicate) and b.py
        records_batch2 = records_batch1 + [
            TruncationRecord(
                file_path="src/a.py",
                chunk_id="sha256:2-0",
                original_tokens=100,
                final_tokens=8,
            ),
            TruncationRecord(
                file_path="src/b.py",
                chunk_id="sha256:2-1",
                original_tokens=100,
                final_tokens=8,
            ),
            TruncationRecord(
                file_path="src/b.py",
                chunk_id="sha256:2-2",
                original_tokens=100,
                final_tokens=8,
            ),
        ]
        embedder.truncation_stats = list(records_batch2)
        total = accumulate_truncation(embedder, total, files)
        assert total == 5
        # a.py is deduplicated; b.py is new.
        assert files == {"src/a.py", "src/b.py"}

    def test_negative_delta_is_a_noop(self):
        """A future refactor that shrinks ``truncation_stats``
        between calls must not corrupt the running total. The guard
        returns the existing total unchanged and adds no files."""
        records_batch1 = [
            TruncationRecord(
                file_path="src/a.py",
                chunk_id="sha256:1",
                original_tokens=100,
                final_tokens=8,
            )
        ]
        embedder = self._make_embedder(records_batch1)
        files: set[str] = {"src/a.py"}
        total = accumulate_truncation(embedder, 0, files)
        assert total == 1

        # Now imagine a future refactor that resets the stats list.
        embedder.truncation_stats = []
        total = accumulate_truncation(embedder, total, files)
        # Running total is preserved, files unchanged.
        assert total == 1
        assert files == {"src/a.py"}

    def test_zero_delta_is_a_noop(self):
        """When the per-batch delta is zero, the running totals are
        preserved and no slice is taken."""
        records = [
            TruncationRecord(
                file_path="src/a.py",
                chunk_id="sha256:1",
                original_tokens=100,
                final_tokens=8,
            )
        ]
        embedder = self._make_embedder(records)
        files: set[str] = {"src/a.py"}
        # First call accumulates.
        total = accumulate_truncation(embedder, 0, files)
        assert total == 1
        # Second call sees the same stats — delta is 0, no change.
        total = accumulate_truncation(embedder, total, files)
        assert total == 1
        assert files == {"src/a.py"}


# ---------------------------------------------------------------------------
# build_truncation_summary — the result-dict shape
# ---------------------------------------------------------------------------


class TestBuildTruncationSummary:
    """The summary dict has three fields; ``truncation_message`` is
    only added when ``total_truncated > 0``."""

    def test_zero_count_has_no_message(self):
        """When no chunks were truncated, ``truncated_chunks`` and
        ``truncated_files`` are present (callers can rely on them)
        but ``truncation_message`` is not."""
        summary = build_truncation_summary(0, set())
        assert summary == {
            "truncated_chunks": 0,
            "truncated_files": [],
        }
        assert "truncation_message" not in summary

    def test_nonzero_count_includes_message(self):
        """A non-zero count adds the human-readable message that names
        the count and the search-coverage consequence."""
        summary = build_truncation_summary(3, {"src/a.py", "src/b.py"})
        assert summary["truncated_chunks"] == 3
        # Files are sorted for deterministic output.
        assert summary["truncated_files"] == ["src/a.py", "src/b.py"]
        assert "3 chunk(s) were shortened" in summary["truncation_message"]
        assert "their endings are not searchable" in summary["truncation_message"]

    def test_files_sorted_alphabetically(self):
        """Files in the summary are always sorted, regardless of
        insertion order. Callers can rely on deterministic output."""
        summary = build_truncation_summary(
            2, {"src/zzz.py", "src/aaa.py", "src/mmm.py"},
        )
        assert summary["truncated_files"] == [
            "src/aaa.py", "src/mmm.py", "src/zzz.py",
        ]


# ---------------------------------------------------------------------------
# log_truncated_files — the INFO log emission
# ---------------------------------------------------------------------------


class TestLogTruncatedFiles:
    """The INFO log lines let operators confirm the truncation count
    and the affected files in CI without enabling DEBUG mode.

    INFO is the right level here: it is always visible at the build
    CLI's default ``basicConfig(level=INFO)`` setting, but emits one
    line per file only when truncations actually occurred, so it
    does not pollute the per-file progress output for a clean build.
    """

    def test_one_log_line_per_file(self, caplog):
        """Each affected file produces one INFO log line that names
        the file and warns about the search-coverage consequence."""
        with caplog.at_level(
            logging.INFO, logger="lib.truncation_visibility",
        ):
            log_truncated_files({"src/a.py", "src/b.py"})
        log_lines = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and r.name == "lib.truncation_visibility"
        ]
        # Two files, two log lines. Order is sorted for determinism.
        assert len(log_lines) == 2
        assert any("src/a.py" in r.message for r in log_lines)
        assert any("src/b.py" in r.message for r in log_lines)
        # The message names the search-coverage consequence.
        for r in log_lines:
            assert "search hits may miss content" in r.message

    def test_empty_set_emits_no_lines(self, caplog):
        """An empty set produces no log lines. Operators can confirm
        in CI that the count is zero by the absence of lines."""
        with caplog.at_level(
            logging.INFO, logger="lib.truncation_visibility",
        ):
            log_truncated_files(set())
        assert not any(
            r.name == "lib.truncation_visibility"
            for r in caplog.records
        )
