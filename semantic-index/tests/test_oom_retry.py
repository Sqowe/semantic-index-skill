"""Tests for OOM-resilient retry/splitting in Embedder and HuggingFaceProvider.

Covers:
1. OOM on large batch succeeds after internal halving (provider level)
2. OOM caused by one bad chunk — successful recursive isolation (embedder level)
3. Non-OOM RuntimeError is re-raised immediately
4. Final single-chunk OOM produces clear error messaging
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
    """Create an Embedder with a mocked provider, bypassing real provider init."""
    with patch("lib.embedder.create_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_provider.embed_texts = mock_embed_texts
        mock_factory.return_value = mock_provider
        return Embedder(config, project_dir=None)


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
    """Test that Embedder.embed_chunks splits batches on OOM to isolate bad chunks."""

    def test_oom_splits_batch_and_isolates_bad_chunk(self) -> None:
        """A 4-chunk batch where chunk #2 causes OOM should split recursively
        until the bad chunk is isolated, and the other 3 succeed."""
        config = _make_config(batch_size=4)
        bad_content = "BAD_CHUNK"

        def mock_embed_texts(texts: list[str]) -> list[list[float]]:
            if any(bad_content in t for t in texts):
                if len(texts) == 1:
                    raise RuntimeError("Invalid buffer size: 16.03 GiB")
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

        # Good chunks in batches without the bad chunk should be embedded
        good_embedded = sum(1 for c in chunks if "vector" in c.metadata)
        assert good_embedded >= 2

    def test_oom_full_batch_succeeds_after_split(self) -> None:
        """OOM on a 4-chunk batch should split into 2+2 and succeed if
        smaller batches fit in memory."""
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
        # First call OOMs (4 chunks), then two successful calls (2+2)
        assert call_count == 3
        assert api_calls == 2  # only successful calls counted

    def test_non_oom_runtime_error_propagates(self) -> None:
        """A non-OOM RuntimeError from the provider should not trigger splitting."""
        config = _make_config(batch_size=4)

        def mock_embed_texts(texts: list[str]) -> list[list[float]]:
            raise RuntimeError("CUBLAS error: dimension mismatch")

        embedder = _make_embedder_with_mock_provider(config, mock_embed_texts)
        chunks = [_make_chunk("chunk", 0)]

        with pytest.raises(RuntimeError, match="CUBLAS error"):
            embedder.embed_chunks(chunks)
