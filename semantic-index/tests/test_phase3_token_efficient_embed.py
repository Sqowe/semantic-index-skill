"""Tests for Phase 3 — stop wasting API calls.

Phase 3 covers three changes that together make the embed path use far
fewer round trips and surface visibility of what it had to shorten:

1. **Token-based pre-truncation.** The OpenRouter and HuggingFace
   providers cap by ``max_embed_tokens`` (counted with the resolver
   bound to the active embedding model), not by characters. The hard
   ``max_embed_chars`` ceiling stays as defense in depth.

2. **Direct-computation progressive truncate.** ``_progressive_truncate``
   measures the token count once, computes the right char cut from
   the text's own chars/token ratio, and tries it once. Only if the
   API still rejects does it fall back to a finer stepping loop. No
   more 25% steps blindly from the original length.

3. **Largest-out isolation.** When a batch errors with OOM or
   context-length, the Embedder pulls the largest chunk out and
   retries the remainder as one batch — instead of halving blindly.

4. **First WARNING per build, then DEBUG.** Recoverable splits and
   truncations log at WARNING the first time they fire and at DEBUG
   thereafter. Per-build scope is achieved by hosting the sentinels
   on the provider / embedder instance (both fresh per build).
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib.config import Config, EmbeddingConfig
from lib.embedder import Embedder
from lib.models import Chunk, ChunkType, EmbeddingError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str, idx: int = 0, token_count: int | None = None) -> Chunk:
    """Create a minimal Chunk for testing."""
    return Chunk(
        id=f"chunk-{idx}",
        file_path=f"test_{idx}.py",
        start_line=1,
        end_line=10,
        content=content,
        chunk_type=ChunkType.FUNCTION,
        language="python",
        token_count=token_count if token_count is not None else len(content.split()),
    )


def _make_config(
    batch_size: int = 4,
    max_embed_tokens: int = 8192,
    max_embed_chars: int = 20000,
) -> Config:
    """Create a Config with the given embed settings."""
    config = Config()
    config.embedding = EmbeddingConfig(
        provider="openrouter",
        batch_size=batch_size,
        model="test-model",
        dimensions=3,
        max_embed_tokens=max_embed_tokens,
        max_embed_chars=max_embed_chars,
    )
    return config


def _mock_response(status_code: int, json_body):
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json.return_value = json_body
    resp.headers = {}
    return resp


def _ok_response(embedding=(0.1, 0.2, 0.3)):
    """Build a 200 success response with one embedding."""
    return _mock_response(
        200, {"data": [{"index": 0, "embedding": list(embedding)}]},
    )


# ---------------------------------------------------------------------------
# 3a — Token-based pre-truncation in OpenRouterProvider
# ---------------------------------------------------------------------------


class TestOpenRouterTokenCap:
    """The primary cap in ``embed_texts`` is the configured token budget."""

    def _make_provider(self, max_embed_tokens: int, max_embed_chars: int):
        config = Config()
        config.embedding = EmbeddingConfig(
            provider="openrouter",
            api_key="test-key",
            max_retries=1,
            retry_delay_seconds=0.01,
            max_embed_tokens=max_embed_tokens,
            max_embed_chars=max_embed_chars,
        )
        from lib.providers.openrouter import OpenRouterProvider
        return OpenRouterProvider(config)

    @patch("lib.providers.openrouter.requests.post")
    def test_text_under_token_budget_not_truncated(self, mock_post):
        """Text below the token budget is sent unmodified."""
        mock_post.return_value = _ok_response()
        # 10 tokens worth of content; budget is 100 tokens; nothing
        # should be touched.
        provider = self._make_provider(max_embed_tokens=100, max_embed_chars=20000)
        provider.embed_texts(["hello world this is a test sentence with some filler"])

        sent_body = mock_post.call_args[1]["json"]
        assert sent_body["input"] == ["hello world this is a test sentence with some filler"]

    @patch("lib.providers.openrouter.requests.post")
    def test_text_over_token_budget_truncated(self, mock_post):
        """Text exceeding the token budget is shortened before sending."""
        mock_post.return_value = _ok_response()
        # Budget is 5 tokens; this text is way over.
        provider = self._make_provider(max_embed_tokens=5, max_embed_chars=20000)
        # 100 tokens worth of content.
        long_text = " ".join(["word"] * 100)
        provider.embed_texts([long_text])

        sent_body = mock_post.call_args[1]["json"]
        # The sent text should be substantially shorter than the input.
        assert len(sent_body["input"][0]) < len(long_text)

    @patch("lib.providers.openrouter.requests.post")
    def test_hard_char_ceiling_still_applies(self, mock_post):
        """``max_embed_chars`` is the hard ceiling even when token count is fine.

        A pathological blob with a high chars/token ratio could pass the
        token check at huge length. The char ceiling catches it.
        """
        mock_post.return_value = _ok_response()
        # Token budget is huge (will fit anything), char ceiling is 100.
        provider = self._make_provider(max_embed_tokens=100000, max_embed_chars=100)
        provider.embed_texts(["x" * 500])

        sent_body = mock_post.call_args[1]["json"]
        # Char ceiling should cut this to 100 chars regardless.
        assert len(sent_body["input"][0]) <= 100

    @patch("lib.providers.openrouter.requests.post")
    def test_truncation_count_logged_at_warning_first_time(self, mock_post, caplog):
        """First pre-truncation in a build logs at WARNING level."""
        mock_post.return_value = _ok_response()
        provider = self._make_provider(max_embed_tokens=5, max_embed_chars=20000)
        with caplog.at_level(logging.WARNING, logger="lib.providers.openrouter"):
            provider.embed_texts([" ".join(["word"] * 100)])
        assert "Pre-truncated 1/1 texts" in caplog.text
        assert "fit 5-token budget" in caplog.text

    @patch("lib.providers.openrouter.requests.post")
    def test_subsequent_truncations_logged_at_debug(
        self, mock_post, caplog,
    ):
        """Truncations after the first one in a build log at DEBUG."""
        mock_post.return_value = _ok_response()
        provider = self._make_provider(max_embed_tokens=5, max_embed_chars=20000)
        # Two calls on the same provider. First WARNING, second DEBUG.
        with caplog.at_level(logging.DEBUG, logger="lib.providers.openrouter"):
            provider.embed_texts([" ".join(["word"] * 100)])
            provider.embed_texts([" ".join(["word"] * 100)])

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(warnings) == 1
        assert any("Pre-truncated" in r.message for r in debugs)


# ---------------------------------------------------------------------------
# 3b — Direct-computation progressive truncate
# ---------------------------------------------------------------------------


class TestOpenRouterProgressiveTruncate:
    """``_progressive_truncate`` uses a direct computation first."""

    def _make_provider(self, max_embed_tokens: int = 8192, max_embed_chars: int = 20000):
        config = Config()
        config.embedding = EmbeddingConfig(
            provider="openrouter",
            api_key="test-key",
            max_retries=1,
            retry_delay_seconds=0.01,
            max_embed_tokens=max_embed_tokens,
            max_embed_chars=max_embed_chars,
        )
        from lib.providers.openrouter import OpenRouterProvider
        return OpenRouterProvider(config)

    @patch("lib.providers.openrouter.requests.post")
    def test_direct_truncation_succeeds_in_one_step(self, mock_post):
        """When the direct computation lands below the API limit, no
        finer stepping is needed. This is the common case.

        The test forces the resolver to tiktoken cl100k so the token
        counts are reproducible regardless of which tokenizer is
        actually cached locally (bge-m3 produces different counts).
        """
        from lib.tokenizer_resolver import get_resolver

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            body = kwargs.get("json", {})
            text_len = len(body["input"][0]) if body.get("input") else 0
            # API fails on texts > 100 chars, succeeds otherwise. The
            # direct computation will produce ~8 chars, well under
            # this threshold — one retry is all it should take.
            if text_len > 100:
                return _mock_response(
                    400,
                    {"error": {"message": "maximum context length is 8192", "code": 400}},
                )
            return _ok_response((0.5, 0.6))

        mock_post.side_effect = side_effect
        # Use a config that the resolver falls back to tiktoken for:
        # "test-model" is not on the HF hub.
        provider = self._make_provider(max_embed_tokens=30, max_embed_chars=10000)
        # Force the resolver to tiktoken cl100k so token counts are
        # predictable regardless of which tokenizer is cached.
        provider._tokens = get_resolver("cl100k_base")

        # 200 chars at cl100k = 25 tokens, under 30 → no pre-truncation.
        # 200 < 10000 → no char ceiling. First call fails (200 > 100
        # threshold). Direct computation: target_tokens = max(1, 30-50)
        # = 1, chars_per_token = 200/25 = 8, target_chars = 8 → success.
        provider.embed_texts(["x" * 200])

        assert call_count[0] == 2

    @patch("lib.providers.openrouter.requests.post")
    def test_fallback_loop_used_when_direct_truncation_insufficient(
        self, mock_post,
    ):
        """When the direct computation is still over budget, fall back
        to a finer stepping loop until it fits or hits the floor."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            body = kwargs.get("json", {})
            text_len = len(body["input"][0]) if body.get("input") else 0
            # API rejects anything > 100 chars. Direct computation will
            # produce ~600 chars (well over the threshold) so we have
            # to step down.
            if text_len > 100:
                return _mock_response(
                    400,
                    {"error": {"message": "maximum context length is 8192", "code": 400}},
                )
            return _ok_response((0.5,))

        mock_post.side_effect = side_effect
        provider = self._make_provider(max_embed_tokens=200, max_embed_chars=2000)
        result = provider.embed_texts(["x" * 800])
        assert result == [[0.5]]
        # Should have stepped down from ~600 chars through the loop.
        assert call_count[0] > 2

    @patch("lib.providers.openrouter.requests.post")
    def test_exhaustion_raises_runtime_error_not_embedding_error(
        self, mock_post,
    ):
        """When the inner loop exhausts, the re-raise must be a
        RuntimeError, not the wrapped EmbeddingError from a wrapped
        200 response. The Embedder's split path only catches
        RuntimeError; an EmbeddingError would escape unhandled and
        crash the build.
        """
        # Every call returns the context-length error wrapped in a
        # 200 response, so the path goes through _call_api -> 200 ->
        # length detected -> EmbeddingError caught -> progressive
        # truncate path -> inner loop exhausts.
        def side_effect(*args, **kwargs):
            return _mock_response(
                200,
                {
                    "error": {
                        "message": "maximum context length is 8192",
                        "code": 400,
                    },
                },
            )
        mock_post.side_effect = side_effect
        provider = self._make_provider(max_embed_tokens=50, max_embed_chars=100)
        # The text is short enough that direct truncation cannot
        # reduce it below the API threshold; the inner loop will
        # exhaust the floor.
        with pytest.raises(RuntimeError, match="context length"):
            provider.embed_texts(["x" * 200])
        # The error must NOT be an EmbeddingError subclass — the
        # Embedder's batch-split path catches RuntimeError, not the
        # provider's EmbeddingError.
        try:
            provider.embed_texts(["x" * 200])
        except RuntimeError as exc:
            from lib.models import EmbeddingError
            assert not isinstance(exc, EmbeddingError), (
                "Exhausted progressive truncate must surface as plain "
                "RuntimeError, not EmbeddingError — the Embedder only "
                "catches RuntimeError for batch splitting."
            )

    @patch("lib.providers.openrouter.requests.post")
    def test_was_n_log_message_uses_previous_length(self, mock_post, caplog):
        """The 'was N' log message reports the previous length, not the
        original — that was a long-standing bug in the 25% stepping loop."""
        # Force the direct computation to land above the API threshold
        # so the fallback loop runs.
        def side_effect(*args, **kwargs):
            body = kwargs.get("json", {})
            text_len = len(body["input"][0]) if body.get("input") else 0
            # API rejects anything > 100 chars.
            if text_len > 100:
                return _mock_response(
                    400,
                    {"error": {"message": "maximum context length is 8192", "code": 400}},
                )
            return _ok_response((0.5,))

        mock_post.side_effect = side_effect
        # Use a long text (10000 chars) and a generous char ceiling
        # (20000) so the direct-computed target is well below the
        # original length. With cl100k_base, 10000 'x' characters
        # tokenise to 1250 tokens; max_embed_tokens=200 → target_tokens
        # = 150 → target_chars ~ 1200. The fallback loop's first
        # "was N" reports the *previous* limit (1200), not the
        # original text length (10000).
        provider = self._make_provider(max_embed_tokens=200, max_embed_chars=20000)
        # Force the resolver to tiktoken cl100k_base so token counts
        # are deterministic. Without this, the test would depend on
        # whichever tokenizer is cached (BAAI/bge-m3 if available),
        # whose BPE merges can change the measured count.
        from lib.tokenizer_resolver import get_resolver
        provider._tokens = get_resolver("cl100k_base")
        with caplog.at_level(logging.DEBUG, logger="lib.providers.openrouter"):
            provider.embed_texts(["x" * 10000])

        # The DEBUG log "Direct truncation insufficient; retrying with
        # N chars (was M)" must report M as the *previous* iteration's
        # limit, never the original text length. The original code
        # printed the original length every time, which was a bug.
        fallback_logs = [
            r for r in caplog.records
            if "Direct truncation insufficient" in r.message
        ]
        assert fallback_logs, "no fallback log emitted"
        # The original length is 10000; the previous limits are 1200,
        # 900, 675, etc. None of the fallback logs should mention
        # 10000.
        for rec in fallback_logs:
            msg = rec.message
            assert "was 10000" not in msg, (
                "fallback log should report the previous limit, "
                f"not the original length; got: {msg!r}"
            )
        # And the first fallback log should report a 'was' that is
        # strictly less than the original length — confirms the log
        # is the previous limit, not the original.
        first_was = fallback_logs[0].message
        assert "was 1200" in first_was, (
            f"first fallback log should report 'was 1200' (the previous "
            f"limit), got: {first_was!r}"
        )

    @patch("lib.providers.openrouter.requests.post")
    def test_target_chars_capped_by_text_length(self, mock_post, caplog):
        """When the computed target_chars exceeds the actual text
        length, the first attempt must send ``text[:len(text)]`` —
        i.e. the full text — rather than a slice that is past the
        end and equals the full text anyway. The fix is to cap
        target_chars by len(text).

        With text=60 chars and a target that would compute to ~4000
        chars (huge max_embed_tokens × chars_per_token), the cap
        forces target_chars=60. The first attempt sends text[:60] =
        60 chars (the full text); that call gets rejected; the inner
        loop then exits immediately because ``limit < _MIN_TRUNCATE_CHARS``,
        and we raise. The cap is what makes the inner loop's exit
        condition reachable in this regime.
        """
        # Reject everything > 50 chars.
        def side_effect(*args, **kwargs):
            return _mock_response(
                400,
                {"error": {"message": "maximum context length is 8192", "code": 400}},
            )

        mock_post.side_effect = side_effect
        # max_embed_tokens huge → computed target ~ 4000 chars; max_embed_chars
        # huge too. Without the len(text) cap, the first attempt would
        # try text[:4000] which equals the full text anyway (Python
        # slicing past the end clamps) — wasted iteration. With the
        # cap, target_chars = 60 = len(text).
        provider = self._make_provider(max_embed_tokens=1000, max_embed_chars=10000)
        with caplog.at_level(logging.WARNING, logger="lib.providers.openrouter"):
            with pytest.raises(RuntimeError, match="context length"):
                provider.embed_texts(["x" * 60])
        # The WARNING log records the (capped) target_chars, which
        # must equal the text length.
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "truncating to" in r.message
        ]
        assert warnings, "expected a 'truncating to N chars' WARNING"
        assert "truncating to 60 chars" in warnings[0].message


# ---------------------------------------------------------------------------
# 3c — Per-build first-WARNING-then-DEBUG for progressive truncate
# ---------------------------------------------------------------------------


class TestOpenRouterProgressiveTruncateLogLevels:
    """Progressive truncate logs WARNING once per provider, then DEBUG."""

    def _make_provider(self, max_embed_tokens: int = 8192, max_embed_chars: int = 20000):
        config = Config()
        config.embedding = EmbeddingConfig(
            provider="openrouter",
            api_key="test-key",
            max_retries=1,
            retry_delay_seconds=0.01,
            max_embed_tokens=max_embed_tokens,
            max_embed_chars=max_embed_chars,
        )
        from lib.providers.openrouter import OpenRouterProvider
        return OpenRouterProvider(config)

    @patch("lib.providers.openrouter.requests.post")
    def test_first_progressive_truncate_warning_then_debug(self, mock_post, caplog):
        """The first time _progressive_truncate fires, log WARNING.
        Subsequent fires log at DEBUG."""
        # Always return context-length error so _progressive_truncate
        # always runs.
        mock_post.return_value = _mock_response(
            400,
            {"error": {"message": "context length is 8192", "code": 400}},
        )
        provider = self._make_provider(max_embed_tokens=50, max_embed_chars=50)

        with caplog.at_level(logging.DEBUG, logger="lib.providers.openrouter"):
            # Two single-text calls. The first WARNING, the second DEBUG.
            with pytest.raises(RuntimeError):
                provider.embed_texts(["x" * 200])
            with pytest.raises(RuntimeError):
                provider.embed_texts(["x" * 200])

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "Single text" in r.message
        ]
        debugs = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG and "Single text" in r.message
        ]
        assert len(warnings) == 1, f"expected 1 WARNING, got {len(warnings)}"
        assert len(debugs) >= 1, f"expected at least 1 DEBUG, got {len(debugs)}"


# ---------------------------------------------------------------------------
# 3d — Embedder largest-out isolation and per-build first-WARNING-then-DEBUG
# ---------------------------------------------------------------------------


class TestEmbedderLargestOutIsolation:
    """Embedder.embed_chunks uses largest-out to isolate recoverable errors."""

    def _make_embedder(self, config: Config, mock_embed_texts):
        with patch("lib.embedder.create_provider") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.embed_texts = mock_embed_texts
            mock_factory.return_value = mock_provider
            return Embedder(config, project_dir=None)

    def test_one_bad_chunk_in_32_costs_three_calls(self):
        """Largest-out isolates one bad chunk in 32 with 3 calls total.

        With halving, this scenario costs 5+ extra calls. Largest-out
        saves the difference.
        """
        config = _make_config(batch_size=32)
        bad_content = "BAD_CHUNK" + " padding" * 50  # ~50 tokens, largest
        call_count = 0

        def mock_embed_texts(texts: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            if any(bad_content in t for t in texts):
                raise RuntimeError("context length exceeded")
            return [[0.1, 0.2, 0.3]] * len(texts)

        embedder = self._make_embedder(config, mock_embed_texts)
        chunks = (
            [_make_chunk(f"good chunk {i}", i) for i in range(31)]
            + [_make_chunk(bad_content, 31)]
        )

        with pytest.raises(EmbeddingError):
            embedder.embed_chunks(chunks)

        # 1: full 32 batch OOMs.
        # 2: 31 good chunks embed.
        # 3: bad chunk alone OOMs; Embedder raises EmbeddingError.
        assert call_count == 3
        assert sum(1 for c in chunks if "vector" in c.metadata) == 31

    def test_first_split_warning_then_debug(self, caplog):
        """The first recoverable batch split logs at WARNING; subsequent
        splits log at DEBUG. Per-build scope is achieved via instance
        state on the Embedder."""
        config = _make_config(batch_size=4)

        # Every batch will trigger a split because of the OOM condition.
        call_count = 0

        def mock_embed_texts(texts: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            if len(texts) > 2:
                raise RuntimeError("context length exceeded")
            return [[0.1, 0.2, 0.3]] * len(texts)

        embedder = self._make_embedder(config, mock_embed_texts)

        # Build 3 batches of 4 chunks each. Each batch will trigger
        # splits (since 4 > 2).
        all_chunks = []
        for batch_idx in range(3):
            all_chunks.extend(
                [_make_chunk(f"b{batch_idx} chunk {i}", batch_idx * 4 + i)
                 for i in range(4)]
            )

        with caplog.at_level(logging.DEBUG, logger="lib.embedder"):
            embedder.embed_chunks(all_chunks)

        # Count WARNING-level logs about batch splits vs DEBUG-level.
        split_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "pulling the largest" in r.message
        ]
        split_debugs = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG
            and "pulling the largest" in r.message
        ]
        assert len(split_warnings) == 1, (
            f"expected 1 split WARNING, got {len(split_warnings)}"
        )
        assert len(split_debugs) >= 1, (
            f"expected at least 1 split DEBUG, got {len(split_debugs)}"
        )


# ---------------------------------------------------------------------------
# 3e — Embedder pre-truncation marks chunks and populates truncation_stats
# ---------------------------------------------------------------------------


class TestEmbedderPreTruncate:
    """The Embedder pre-truncates chunks that exceed the token budget."""

    def _make_embedder(self, config: Config, mock_embed_texts):
        with patch("lib.embedder.create_provider") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.embed_texts = mock_embed_texts
            mock_factory.return_value = mock_provider
            return Embedder(config, project_dir=None)

    def test_oversized_chunk_is_pre_truncated(self):
        """A chunk that exceeds the token budget is shortened before the
        provider sees it. ``truncation_stats`` records the original and
        final token counts; the chunk's metadata flags it as truncated."""
        config = _make_config(batch_size=2, max_embed_tokens=10)
        # Build a chunk that is way over the 10-token budget.
        long_content = " ".join(["word"] * 100)

        embedder = self._make_embedder(config, lambda texts: [[0.1, 0.2, 0.3]] * len(texts))
        chunk = _make_chunk(long_content, 0)
        embedder.embed_chunks([chunk])

        assert len(embedder.truncation_stats) == 1
        record = embedder.truncation_stats[0]
        assert record.file_path == "test_0.py"
        assert record.original_tokens > 10  # way over budget
        assert record.final_tokens <= 10
        # Chunk metadata flag is set so search / inspection can see the loss.
        assert chunk.metadata.get("truncated") is True
        assert chunk.metadata.get("original_token_count") == record.original_tokens
        # The full original text is preserved on metadata so search
        # results can still show it; only the API payload is truncated.
        assert chunk.metadata.get("original_content") == long_content
        # The chunk's content has been shortened for embedding.
        assert chunk.content != long_content

    def test_chunk_under_budget_not_truncated(self):
        """A chunk already within the token budget passes through unchanged."""
        config = _make_config(batch_size=2, max_embed_tokens=100)
        embedder = self._make_embedder(config, lambda texts: [[0.1, 0.2, 0.3]] * len(texts))
        chunk = _make_chunk("short content", 0)
        embedder.embed_chunks([chunk])

        assert embedder.truncation_stats == []
        assert chunk.metadata.get("truncated") is not True

    def test_first_pre_truncate_warning_then_debug(self, caplog):
        """The first pre-truncation in a build logs WARNING; subsequent
        ones log DEBUG."""
        config = _make_config(batch_size=4, max_embed_tokens=10)
        embedder = self._make_embedder(config, lambda texts: [[0.1, 0.2, 0.3]] * len(texts))

        # Build chunks that all exceed the budget.
        chunks = [
            _make_chunk(" ".join(["word"] * 100), i) for i in range(4)
        ]

        with caplog.at_level(logging.DEBUG, logger="lib.embedder"):
            embedder.embed_chunks(chunks)

        pre_truncate_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "Pre-truncated" in r.message
        ]
        # All chunks fit in one batch, so we get one WARNING.
        assert len(pre_truncate_warnings) == 1

        # Now run a second batch — this should be DEBUG (not WARNING).
        chunks2 = [
            _make_chunk(" ".join(["word"] * 100), 10 + i) for i in range(4)
        ]
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="lib.embedder"):
            embedder.embed_chunks(chunks2)

        pre_truncate_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "Pre-truncated" in r.message
        ]
        pre_truncate_debugs = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG
            and "Pre-truncated" in r.message
        ]
        assert len(pre_truncate_warnings) == 0, (
            "second batch should not re-trigger the WARNING"
        )
        assert len(pre_truncate_debugs) >= 1, (
            "second batch should log at DEBUG"
        )

    def test_truncation_stats_accumulate_across_calls(self):
        """``truncation_stats`` accumulates across multiple
        ``embed_chunks`` calls so ``build_index.py`` can read the total
        at the end of a build."""
        config = _make_config(batch_size=4, max_embed_tokens=10)
        embedder = self._make_embedder(config, lambda texts: [[0.1, 0.2, 0.3]] * len(texts))

        chunks1 = [_make_chunk(" ".join(["word"] * 100), i) for i in range(2)]
        chunks2 = [_make_chunk(" ".join(["word"] * 100), 10 + i) for i in range(2)]
        embedder.embed_chunks(chunks1)
        embedder.embed_chunks(chunks2)
        assert len(embedder.truncation_stats) == 4

    def test_final_tokens_reserves_room_for_doc_prefix(self):
        """The Embedder's target budget subtracts the doc_prefix token
        count so the provider's secondary cap is never tighter than
        the Embedder's truncation. ``final_tokens`` in the
        TruncationRecord must reflect what the API actually sees —
        the chunk's content, not the prefixed text.
        """
        from lib.config import EmbeddingConfig

        # Use a non-empty doc_prefix so the reservation kicks in.
        # "DOC:" tokenises to 2 tokens with tiktoken cl100k (one for
        # the letters, one for the colon).
        config = _make_config(batch_size=1, max_embed_tokens=20)
        config.embedding.document_prefix = "DOC:"
        embedder = self._make_embedder(config, lambda texts: [[0.1, 0.2, 0.3]] * len(texts))
        prefix_tokens = embedder._doc_prefix_tokens
        assert prefix_tokens >= 1, (
            f"expected doc_prefix to be at least 1 token, got {prefix_tokens}"
        )

        # Build a chunk that is over budget. With max_embed_tokens=20
        # and prefix overhead, target_tokens = max(1, 20 - 50 - prefix).
        # For a positive prefix that is, target_tokens = 1.
        long_content = " ".join(["word"] * 100)
        chunk = _make_chunk(long_content, 0)
        embedder.embed_chunks([chunk])

        record = embedder.truncation_stats[0]
        # final_tokens = tokens of the truncated content (not prefixed).
        # The provider sees final_tokens + prefix_tokens, which is
        # bounded by max_embed_tokens = 20.
        assert record.final_tokens + prefix_tokens <= 20, (
            f"provider would see {record.final_tokens + prefix_tokens} "
            f"tokens, exceeding max_embed_tokens=20"
        )


# ---------------------------------------------------------------------------
# 3f — HuggingFace token-based cap parity with OpenRouter
# ---------------------------------------------------------------------------


class TestHuggingFaceTokenCap:
    """Direct test coverage for ``HuggingFaceProvider._cap_to_token_budget``.

    The OpenRouter provider has dedicated tests in
    :class:`TestOpenRouterTokenCap`; this class mirrors them for the
    HuggingFace provider so both implementations stay in sync. We
    build a provider with ``__init__`` bypassed (sentence-transformers
    is heavy) and exercise the cap logic on the bare instance.
    """

    def _make_provider(self, max_embed_tokens: int, max_embed_chars: int):
        """Build a HuggingFaceProvider with mocked model + resolver."""
        from lib.providers.huggingface import HuggingFaceProvider
        from lib.tokenizer_resolver import get_resolver

        config = Config()
        config.embedding = EmbeddingConfig(
            provider="huggingface",
            model="test-model",
            dimensions=3,
            batch_size=1,
            max_embed_chars=max_embed_chars,
            max_embed_tokens=max_embed_tokens,
            trust_remote_code=False,
        )

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_model.device = "cpu"

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            with patch.object(
                HuggingFaceProvider, "__init__", lambda self, cfg: None,
            ):
                provider = HuggingFaceProvider.__new__(HuggingFaceProvider)

        provider._model = mock_model
        provider._batch_size = config.embedding.batch_size
        provider._doc_prefix = config.embedding.document_prefix
        provider._query_prefix = config.embedding.query_prefix
        provider._dimensions = config.embedding.dimensions
        provider._max_embed_chars = config.embedding.max_embed_chars
        provider._max_embed_tokens = config.embedding.max_embed_tokens
        # ``test-model`` is not on the HF hub, so the resolver falls
        # back to tiktoken cl100k_base — predictable counts for tests.
        provider._tokens = get_resolver("cl100k_base")
        provider._first_embed_truncate_warning_logged = False
        provider._first_halving_warning_logged = False
        provider._first_progressive_truncate_warning_logged = False
        return provider

    def test_text_under_token_budget_not_truncated(self):
        """Text below the token budget is returned unchanged."""
        provider = self._make_provider(
            max_embed_tokens=100, max_embed_chars=20000,
        )
        text = "hello world this is a test sentence with some filler"
        out = provider._cap_to_token_budget(text)
        assert out is text  # identity check confirms no truncation

    def test_text_over_token_budget_truncated(self):
        """Text exceeding the token budget is shortened."""
        provider = self._make_provider(
            max_embed_tokens=5, max_embed_chars=20000,
        )
        text = " ".join(["word"] * 100)
        out = provider._cap_to_token_budget(text)
        assert len(out) < len(text)

    def test_hard_char_ceiling_still_applies(self):
        """``max_embed_chars`` is the hard ceiling even when the token
        count is fine. A high chars/token blob would pass the token
        check at huge length without the char ceiling."""
        provider = self._make_provider(
            max_embed_tokens=100000, max_embed_chars=100,
        )
        out = provider._cap_to_token_budget("x" * 500)
        assert len(out) <= 100

    def test_cap_to_token_budget_matches_openrouter_shape(self):
        """Sanity check that the HF cap mirrors OpenRouter's for the
        same input + config."""
        from lib.providers.openrouter import OpenRouterProvider

        config = Config()
        config.embedding = EmbeddingConfig(
            provider="openrouter",
            api_key="test-key",
            max_retries=1,
            retry_delay_seconds=0.01,
            max_embed_tokens=5,
            max_embed_chars=20000,
        )
        or_provider = OpenRouterProvider(config)
        # Same resolver on both so the token math is identical.
        or_provider._tokens = self._make_provider(
            max_embed_tokens=5, max_embed_chars=20000,
        )._tokens

        text = " ".join(["word"] * 100)
        assert (
            or_provider._cap_to_token_budget(text)
            == self._make_provider(
                max_embed_tokens=5, max_embed_chars=20000,
            )._cap_to_token_budget(text)
        )
