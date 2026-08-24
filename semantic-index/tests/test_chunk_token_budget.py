"""Tests for the chunk size guarantee (Phase 1).

Before this, two chunkers could emit a chunk far larger than
``chunking.max_tokens``: the code chunker's last-resort splitter worked
line by line, and ``chunk_text_fallback`` worked block by block. Content
with no such boundary — a minified source line, a base64 blob, a file
written without blank lines — passed through whole. The embedding API
then rejected it, and the retry logic halved the batch repeatedly to
find the offending chunk, costing one request per level.

These tests pin down the guarantee: nothing leaves ``chunk_file`` over
budget, whatever the input looks like.
"""

import pytest

from lib.chunker import chunk_file, _enforce_token_budget
from lib.chunkers.common import (
    build_chunks,
    chunk_text_fallback,
    count_tokens,
    hard_split_by_tokens,
    split_text_with_lines,
)
from lib.models import Chunk, ChunkType


# ---------------------------------------------------------------------------
# hard_split_by_tokens
# ---------------------------------------------------------------------------

class TestHardSplitByTokens:
    """The shared last-resort splitter."""

    def test_short_text_returned_unchanged(self):
        assert hard_split_by_tokens("a short line", max_tokens=100) == [
            "a short line"
        ]

    def test_empty_text_returned_unchanged(self):
        assert hard_split_by_tokens("", max_tokens=100) == [""]

    def test_every_piece_fits_the_budget(self):
        text = "word " * 2000
        pieces = hard_split_by_tokens(text, max_tokens=50)
        assert len(pieces) > 1
        assert all(count_tokens(piece) <= 50 for piece in pieces)

    def test_pieces_reassemble_into_the_original(self):
        text = "".join(f"item_{i}=value_{i};" for i in range(500))
        assert "".join(hard_split_by_tokens(text, max_tokens=30)) == text

    def test_no_boundary_needed(self):
        """A single unbroken token run still splits — no whitespace required."""
        text = "A1b2C3d4" * 1000
        pieces = hard_split_by_tokens(text, max_tokens=40)
        assert len(pieces) > 1
        assert all(count_tokens(piece) <= 40 for piece in pieces)
        assert "".join(pieces) == text

    @pytest.mark.parametrize("sample", [
        "Проверка многобайтового текста без пробелов и переносов. ",
        "日本語のテキストを分割するときの往復チェック。",
        "🙂🎉🚀 emoji sequences that encode across token boundaries ",
    ])
    def test_multibyte_text_survives_the_round_trip(self, sample):
        """Slices must never end mid-character."""
        text = sample * 200
        pieces = hard_split_by_tokens(text, max_tokens=25)
        assert all(count_tokens(piece) <= 25 for piece in pieces)
        assert "".join(pieces) == text
        assert "�" not in "".join(pieces)

    def test_max_tokens_below_one_is_raised_to_one(self):
        pieces = hard_split_by_tokens("some text here", max_tokens=0)
        assert all(count_tokens(piece) <= 1 for piece in pieces)


# ---------------------------------------------------------------------------
# split_text_with_lines
# ---------------------------------------------------------------------------

class TestSplitTextWithLines:
    """Line numbers reported alongside the pieces."""

    def test_single_piece_spans_its_own_lines(self):
        text = "line one\nline two\nline three"
        result = split_text_with_lines(text, start_line=10, max_tokens=100)
        assert result == [(text, 10, 12, 0)]

    def test_consecutive_pieces_chain_line_numbers(self):
        text = "\n".join(f"line {i} with some filler words" for i in range(60))
        result = split_text_with_lines(text, start_line=1, max_tokens=40)
        assert len(result) > 1
        for (_, _, prev_end, _), (_, next_start, _, _) in zip(result, result[1:]):
            # A split need not land on a line boundary, so the next piece
            # may continue on the line the previous one ended on.
            assert next_start == prev_end

    def test_last_line_matches_the_whole_text(self):
        text = "\n".join(f"line {i} with some filler words" for i in range(60))
        result = split_text_with_lines(text, start_line=1, max_tokens=40)
        assert result[-1][2] == 1 + text.count("\n")

    def test_column_advances_within_an_unbroken_line(self):
        """Pieces of one long line are told apart by their column."""
        text = "abcdefghij" * 400
        result = split_text_with_lines(text, start_line=5, max_tokens=20)
        assert len(result) > 1
        assert all(start == 5 and end == 5 for _, start, end, _ in result)
        columns = [column for *_, column in result]
        assert columns[0] == 0
        assert columns == sorted(set(columns))

    def test_column_resets_after_a_newline(self):
        text = "\n".join("payload " * 30 for _ in range(40))
        result = split_text_with_lines(text, start_line=1, max_tokens=25)
        for piece, _, _, column in result:
            assert column >= 0
        # Every piece here contains a newline, so each one starts at the
        # offset left over from the previous piece's final line.
        assert result[0][3] == 0


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------

class TestBuildChunks:
    """The shared emission path used by the chunkers."""

    def test_text_within_budget_produces_one_chunk(self):
        chunks = build_chunks(
            "def f():\n    return 1\n" * 5,
            file_path="a.py",
            start_line=1,
            language="python",
            chunk_type=ChunkType.FUNCTION,
            max_tokens=512,
            min_tokens=5,
        )
        assert len(chunks) == 1
        assert chunks[0].chunk_type is ChunkType.FUNCTION

    def test_oversized_text_is_split(self):
        chunks = build_chunks(
            "x = " + "1234567890" * 2000,
            file_path="a.py",
            start_line=1,
            language="python",
            chunk_type=ChunkType.MODULE_LEVEL,
            max_tokens=100,
            min_tokens=5,
        )
        assert len(chunks) > 1
        assert all(chunk.token_count <= 100 for chunk in chunks)

    def test_metadata_is_copied_not_shared(self):
        metadata = {"parent": "Widget"}
        chunks = build_chunks(
            "value = " + "abcdefghij" * 2000,
            file_path="a.py",
            start_line=1,
            language="python",
            chunk_type=ChunkType.METHOD,
            max_tokens=100,
            min_tokens=5,
            symbol_name="render",
            metadata=metadata,
        )
        assert len(chunks) > 1
        chunks[0].metadata["parent"] = "Other"
        assert chunks[1].metadata["parent"] == "Widget"
        assert metadata["parent"] == "Widget"

    def test_pieces_below_min_tokens_are_dropped(self):
        chunks = build_chunks(
            "tiny",
            file_path="a.py",
            start_line=1,
            language="python",
            chunk_type=ChunkType.MODULE_LEVEL,
            max_tokens=100,
            min_tokens=50,
        )
        assert chunks == []

    def test_ids_are_distinct_across_pieces(self):
        chunks = build_chunks(
            "q = " + "zyxwvutsrq" * 2000,
            file_path="a.py",
            start_line=1,
            language="python",
            chunk_type=ChunkType.MODULE_LEVEL,
            max_tokens=100,
            min_tokens=5,
        )
        assert len({chunk.id for chunk in chunks}) == len(chunks)


# ---------------------------------------------------------------------------
# chunk_text_fallback
# ---------------------------------------------------------------------------

class TestTextFallbackBudget:
    """The blank-line fallback used for unsupported languages."""

    def test_file_without_blank_lines_is_split(self, small_config):
        """Previously the whole file became one chunk."""
        content = "\n".join(
            f"record {i}: some payload text that keeps going" for i in range(400)
        )
        chunks = chunk_text_fallback(content, "data.log", "text", small_config)
        assert len(chunks) > 1
        assert all(
            chunk.token_count <= small_config.chunking.max_tokens
            for chunk in chunks
        )

    def test_single_oversized_block_is_split(self, small_config):
        content = "short intro\n\n" + "payload " * 4000 + "\n\nshort outro"
        chunks = chunk_text_fallback(content, "notes.txt", "text", small_config)
        assert all(
            chunk.token_count <= small_config.chunking.max_tokens
            for chunk in chunks
        )

    def test_normal_content_still_chunks_at_blank_lines(self, small_config):
        content = "\n\n".join(f"Paragraph {i}. " * 5 for i in range(10))
        chunks = chunk_text_fallback(content, "doc.txt", "text", small_config)
        assert len(chunks) > 1
        assert all(
            chunk.token_count <= small_config.chunking.max_tokens
            for chunk in chunks
        )


# ---------------------------------------------------------------------------
# _enforce_token_budget
# ---------------------------------------------------------------------------

class TestEnforceTokenBudget:
    """The safety net in the dispatcher."""

    @staticmethod
    def _chunk(content: str, token_count: int) -> Chunk:
        return Chunk(
            id="sha256:test",
            file_path="a.txt",
            start_line=1,
            end_line=1 + content.count("\n"),
            content=content,
            chunk_type=ChunkType.UNKNOWN,
            language="text",
            token_count=token_count,
        )

    def test_chunks_within_budget_pass_through_untouched(self, small_config):
        chunks = [self._chunk("hello world", 2)]
        assert _enforce_token_budget(chunks, "a.txt", small_config) is chunks

    def test_oversized_chunk_is_split(self, small_config):
        content = "payload " * 500
        chunks = [self._chunk(content, count_tokens(content))]
        result = _enforce_token_budget(chunks, "a.txt", small_config)
        assert len(result) > 1
        assert all(
            chunk.token_count <= small_config.chunking.max_tokens
            for chunk in result
        )

    def test_grossly_oversized_chunk_warns(self, small_config, caplog):
        content = "payload " * 500
        chunks = [self._chunk(content, count_tokens(content))]
        with caplog.at_level("WARNING"):
            _enforce_token_budget(chunks, "a.txt", small_config)
        assert "oversized chunk" in caplog.text

    def test_slight_overshoot_is_split_without_warning(self, small_config, caplog):
        """A few tokens over is rounding in an accumulating chunker, not a gap."""
        max_tokens = small_config.chunking.max_tokens
        content = "word " * 2000
        pieces = hard_split_by_tokens(content, max_tokens + 2)
        just_over = next(p for p in pieces if count_tokens(p) > max_tokens)
        chunks = [self._chunk(just_over, count_tokens(just_over))]

        with caplog.at_level("WARNING"):
            result = _enforce_token_budget(chunks, "a.txt", small_config)

        assert "oversized chunk" not in caplog.text
        assert all(chunk.token_count <= max_tokens for chunk in result)

    def test_unset_token_count_is_recounted(self, small_config):
        """A chunker that forgets token_count must not slip past the net."""
        content = "payload " * 500
        chunks = [self._chunk(content, 0)]
        result = _enforce_token_budget(chunks, "a.txt", small_config)
        assert len(result) > 1
        assert all(
            chunk.token_count <= small_config.chunking.max_tokens
            for chunk in result
        )


# ---------------------------------------------------------------------------
# End-to-end through chunk_file
# ---------------------------------------------------------------------------

class TestChunkFileBudget:
    """Nothing leaves the dispatcher over budget, whatever the input."""

    @pytest.mark.parametrize("name,content", [
        # Minified JavaScript: one line, no blank-line or newline boundary.
        (
            "bundle.js",
            "!function(){" + "".join(
                f"var v{i}=function(a,b){{return a+b+{i}}};" for i in range(800)
            ) + "}();",
        ),
        # A long data literal inside otherwise ordinary Python.
        (
            "data.py",
            "import os\n\n\nTABLE = ["
            + ",".join(f'"row_{i}_value"' for i in range(3000))
            + "]\n",
        ),
        # A log file with no blank lines at all.
        (
            "server.log",
            "\n".join(
                f"2026-08-24 11:31:{i % 60:02d} [INFO] handled request {i}"
                for i in range(600)
            ),
        ),
        # A single base64-like blob with no separators whatsoever.
        (
            "blob.txt",
            "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVph" * 900,
        ),
        # Markdown whose single paragraph has no internal boundary.
        (
            "notes.md",
            "# Title\n\n" + "x" * 40000 + "\n",
        ),
    ])
    def test_no_chunk_exceeds_max_tokens(
        self, tmp_path, default_config, name, content
    ):
        (tmp_path / name).write_text(content, encoding="utf-8")
        chunks = chunk_file(name, str(tmp_path), default_config)
        assert chunks, f"{name} produced no chunks"
        max_tokens = default_config.chunking.max_tokens
        oversized = [c for c in chunks if c.token_count > max_tokens]
        assert not oversized, (
            f"{name}: {len(oversized)} chunk(s) over budget, "
            f"largest {max(c.token_count for c in oversized)} > {max_tokens}"
        )

    def test_recorded_token_count_matches_the_content(
        self, tmp_path, default_config
    ):
        """The stored count must describe the stored text, post-split."""
        content = "K = [" + ",".join(str(i) for i in range(8000)) + "]\n"
        (tmp_path / "big.py").write_text(content, encoding="utf-8")
        chunks = chunk_file("big.py", str(tmp_path), default_config)
        for chunk in chunks:
            assert chunk.token_count == count_tokens(chunk.content)

    def test_line_numbers_stay_within_the_file(self, tmp_path, default_config):
        content = "\n".join(f"value_{i} = {i}" for i in range(400))
        (tmp_path / "consts.py").write_text(content, encoding="utf-8")
        chunks = chunk_file("consts.py", str(tmp_path), default_config)
        total_lines = 1 + content.count("\n")
        for chunk in chunks:
            assert 1 <= chunk.start_line <= total_lines
            assert chunk.start_line <= chunk.end_line <= total_lines

    def test_ordinary_python_file_is_unaffected(self, tmp_path, default_config):
        """The guarantee must not change chunking of normal source.

        Bodies are padded past ``min_tokens`` (20 by default), otherwise the
        chunkers drop them and the test would prove nothing.
        """
        def body(name: str) -> str:
            lines = "\n".join(
                f"    step_{i} = compute_{name}({i}) + offset_{i}"
                for i in range(8)
            )
            return f"def {name}(value, offset_0=0):\n{lines}\n    return value\n"

        content = "import os\nimport sys\n\n\n" + "\n\n".join(
            body(name) for name in ("alpha", "beta")
        )
        (tmp_path / "mod.py").write_text(content, encoding="utf-8")
        chunks = chunk_file("mod.py", str(tmp_path), default_config)
        symbols = {chunk.symbol_name for chunk in chunks}
        assert "alpha" in symbols and "beta" in symbols
        assert all(
            chunk.token_count <= default_config.chunking.max_tokens
            for chunk in chunks
        )
