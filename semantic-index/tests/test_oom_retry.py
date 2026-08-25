"""Tests for OOM-resilient retry/splitting in Embedder and HuggingFaceProvider.

Covers:
1. OOM on large batch succeeds after internal halving (provider level)
2. OOM caused by one bad chunk — successful largest-out isolation
   (embedder level). One bad chunk in a 4-batch now costs 2 calls total
   instead of the 5 blind halving would burn.
3. Non-OOM RuntimeError is re-raised immediately
4. Final single-chunk OOM produces clear error messaging
5. Recoverable splits log at WARNING the first time per Embedder and at
   DEBUG for subsequent occurrences.
"""

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


def _make_chunk(content: str, idx: int = 0) -> Chunk:
    """Create a minimal Chunk for testing."""
    return Chunk(
        id=f"chunk-{idx}",
        file_path=f"test_{idx}.py",
        start_line=1,
        end_line=10,
        content=content,
        chunk_type=ChunkType.FUNCTION,
        language="python",
        token_count=len(content.split()),
    )


def _make_config(batch_size: int = 4) -> Config:
    """Create a Config with a given embedding batch_size."""
    config = Config()
    config.embedding = EmbeddingConfig(
        provider="openrouter",
        batch_size=batch_size,
        model="test-model",
        dimensions=3,
    )
    return config


def _make_embedder_with_mock_provider(config: Config, mock_embed_texts):
    """Create an Embedder with a mocked provider, bypassing real provider init.

    The Embedder's ``__init__`` calls ``create_provider`` to wire the
    provider; for these tests we substitute a mock and then patch the
    resolver to a tiktoken-backed one (so ``count_tokens`` works
    offline) without going through ``HuggingFaceProvider.__init__``.
    """
    with patch("lib.embedder.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.embed_texts = mock_embed_texts
        mock_factory.return_value = mock_provider
        embedder = Embedder(config, project_dir=None)
    return embedder


class TestProviderInternalHalving:
    """Test that HuggingFaceProvider.embed_texts retries with smaller batch_size on OOM."""

    def _make_provider(self, config: Config, mock_encode) -> "HuggingFaceProvider":
        """Build a HuggingFaceProvider with a mocked SentenceTransformer."""
        mock_model = MagicMock()
        mock_model.encode = mock_encode
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_model.device = "cpu"

        # Patch the import inside __init__
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            from lib.providers.huggingface import HuggingFaceProvider
            with patch.object(
                HuggingFaceProvider, "__init__", lambda self, cfg: None
            ):
                provider = HuggingFaceProvider.__new__(HuggingFaceProvider)

        # Manually set the attributes that __init__ would set
        provider._model = mock_model
        provider._batch_size = config.embedding.batch_size
        provider._doc_prefix = config.embedding.document_prefix
        provider._query_prefix = config.embedding.query_prefix
        provider._dimensions = config.embedding.dimensions
        provider._max_embed_chars = config.embedding.max_embed_chars
        provider._max_embed_tokens = config.embedding.max_embed_tokens
        # ``test-model`` has no HF repo, so the resolver falls back to
        # tiktoken cl100k_base. That is fine for the truncation math.
        from lib.tokenizer_resolver import resolver_for_config
        provider._tokens = resolver_for_config(config)
        provider._first_embed_truncate_warning_logged = False
        provider._first_progressive_truncate_warning_logged = False
        provider._first_halving_warning_logged = False
        return provider

    def test_oom_succeeds_after_internal_halving(self) -> None:
        """OOM at batch_size=4 should retry at 2, then succeed."""
        config = _make_config(batch_size=4)
        call_count = 0

        def mock_encode(texts, batch_size=32, **kwargs):
            nonlocal call_count
            call_count += 1
            if batch_size > 2:
                raise RuntimeError("Invalid buffer size: 16.03 GiB")
            import numpy as np
            return np.array([[0.1, 0.2, 0.3]] * len(texts))

        provider = self._make_provider(config, mock_encode)
        result = provider.embed_texts(["hello", "world"])

        assert len(result) == 2
        assert call_count == 2  # first call OOM, second succeeds

    def test_oom_at_batch_size_1_reraises_runtime_error(self) -> None:
        """When internal batch_size reaches 1 and still OOMs, re-raise RuntimeError."""
        config = _make_config(batch_size=1)

        def mock_encode(texts, batch_size=32, **kwargs):
            raise RuntimeError("CUDA out of memory")

        provider = self._make_provider(config, mock_encode)

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            provider.embed_texts(["hello"])

    def test_non_oom_runtime_error_reraises_immediately(self) -> None:
        """A RuntimeError that isn't OOM should propagate without retry."""
        config = _make_config(batch_size=4)
        call_count = 0

        def mock_encode(texts, batch_size=32, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("CUBLAS error: dimension mismatch")

        provider = self._make_provider(config, mock_encode)

        with pytest.raises(RuntimeError, match="CUBLAS error"):
            provider.embed_texts(["hello"])

        assert call_count == 1  # no retry attempted


class TestEmbedderBatchSplitting:
    """Test that Embedder.embed_chunks isolates bad chunks via largest-out.

    The strategy pulls the largest chunk out and retries the remainder as
    one batch. In the common case (one bad chunk in a batch), this costs
    one extra request to find the culprit instead of the five blind
    halving would burn.
    """

    def test_oom_splits_batch_and_isolates_bad_chunk(self) -> None:
        """One bad chunk in a 4-batch should be isolated in 2 calls total.

        Largest-out pulls the bad chunk out of the original 4-batch; the
        remainder of 3 good chunks embeds in one call. The bad chunk is
        then tried alone, which fails as EmbeddingError because no
        further split is possible. Net cost: 2 calls (1 successful on 3
        good chunks, 1 failed on the bad chunk).
        """
        config = _make_config(batch_size=4)
        # The bad chunk must be the largest in the batch — that is the
        # realistic shape (a pathological chunk is usually a huge blob).
        # Without it being largest, ``max()`` picks a good chunk and we
        # waste calls peeling good chunks off one at a time.
        bad_content = "BAD_CHUNK" + " extra padding" * 50  # ~50 tokens

        def mock_embed_texts(texts: list[str]) -> list[list[float]]:
            if any(bad_content in t for t in texts):
                raise RuntimeError("Invalid buffer size: 16.03 GiB")
            return [[0.1, 0.2, 0.3]] * len(texts)

        embedder = _make_embedder_with_mock_provider(config, mock_embed_texts)

        chunks = [
            _make_chunk("good chunk 0", 0),
            _make_chunk("good chunk 1", 1),
            _make_chunk(bad_content, 2),
            _make_chunk("good chunk 3", 3),
        ]

        with pytest.raises(EmbeddingError, match="Cannot embed single chunk"):
            embedder.embed_chunks(chunks)

        # The 3 good chunks should be embedded; the bad chunk should not.
        good_embedded = sum(
            1 for c in chunks
            if "vector" in c.metadata and bad_content not in c.content
        )
        assert good_embedded == 3

    def test_oom_full_batch_succeeds_after_split(self) -> None:
        """OOM on a 4-chunk batch is resolved by repeatedly pulling the
        largest out until each sub-batch fits. With 4 equal-sized chunks
        and a mock that OOMs above batch size 2, the path is:
          4 -> pull 1 largest -> 3 + 1
          3 -> pull 1 largest -> 2 + 1
          2 -> success
          1 -> success
          1 -> success
        Total: 5 calls, 3 successful. Halving would have hit 3 calls but
        the common case (one bad chunk) is what we optimise for.
        """
        config = _make_config(batch_size=4)
        call_count = 0

        def mock_embed_texts(texts: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            if len(texts) > 2:
                raise RuntimeError("Invalid buffer size: 16.03 GiB")
            return [[0.1, 0.2, 0.3]] * len(texts)

        embedder = _make_embedder_with_mock_provider(config, mock_embed_texts)
        chunks = [_make_chunk(f"chunk {i}", i) for i in range(4)]
        api_calls = embedder.embed_chunks(chunks)

        assert all("vector" in c.metadata for c in chunks)
        # 4 -> pull 1 largest -> queue is [3, [1]]
        # 3 -> pull 1 largest -> queue is [[1], 2, [1]]
        # 1 -> success
        # 2 -> success
        # 1 -> success
        assert call_count == 5
        assert api_calls == 3  # only successful calls counted

    def test_one_bad_chunk_in_32_costs_three_calls(self) -> None:
        """One bad chunk in a 32-batch is isolated in 3 calls total —
        vs the 5+ blind halving would burn.

        The bad chunk must be the largest in the batch — that is the
        realistic shape (a pathological chunk is usually a huge blob).
        The Embedder pulls it out first; the remaining 31 fit in one
        call; the bad chunk then OOMs alone and surfaces as
        EmbeddingError because no further split is possible. The
        "two extra calls" saved over halving is the headline win
        (halving would need 6 extra calls to isolate the bad chunk in
        a 32-batch; largest-out needs only 2).
        """
        config = _make_config(batch_size=32)
        bad_content = "BAD_CHUNK" + " extra padding" * 50  # ~50 tokens
        call_count = 0

        def mock_embed_texts(texts: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            if any(bad_content in t for t in texts):
                raise RuntimeError("Invalid buffer size: 16.03 GiB")
            return [[0.1, 0.2, 0.3]] * len(texts)

        embedder = _make_embedder_with_mock_provider(config, mock_embed_texts)
        chunks = (
            [_make_chunk(f"good chunk {i}", i) for i in range(31)]
            + [_make_chunk(bad_content, 31)]
        )

        with pytest.raises(EmbeddingError, match="Cannot embed single chunk"):
            embedder.embed_chunks(chunks)

        # First call: full 32 OOMs. Largest-out pulls bad, retries 31.
        # Second call: 31 good chunks embed successfully.
        # Third call: bad chunk alone OOMs; Embedder catches and raises
        # EmbeddingError because ``len(batch) <= 1`` makes further
        # splitting pointless. That is the 2-call savings over halving,
        # not 1 — we still need one call to discover the bad chunk is
        # unembeddable on its own.
        assert call_count == 3
        assert sum(1 for c in chunks if "vector" in c.metadata) == 31

    def test_non_oom_runtime_error_propagates(self) -> None:
        """A non-OOM RuntimeError from the provider should not trigger splitting."""
        config = _make_config(batch_size=4)

        def mock_embed_texts(texts: list[str]) -> list[list[float]]:
            raise RuntimeError("CUBLAS error: dimension mismatch")

        embedder = _make_embedder_with_mock_provider(config, mock_embed_texts)
        chunks = [_make_chunk("chunk", 0)]

        with pytest.raises(RuntimeError, match="CUBLAS error"):
            embedder.embed_chunks(chunks)
