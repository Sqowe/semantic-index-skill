"""HuggingFace local embedding provider.

Uses sentence-transformers for local inference. No API key needed.
Model is downloaded on first use to ~/.cache/huggingface/hub (~274MB
for Nomic, ~600MB for larger models). Subsequent runs load from cache.

Device auto-detection: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
Override with the `device` config field or leave null for auto.
"""

import logging
import sys
from typing import Optional

from ..models import EmbeddingError

logger = logging.getLogger(__name__)


class HuggingFaceProvider:
    """Local embedding provider using sentence-transformers.

    Implements the EmbeddingProvider interface for local inference.
    Dependencies (sentence-transformers, torch) are imported lazily
    at instantiation time — they are never loaded if this provider
    is not selected.

    Args:
        config: Config object with embedding settings.

    Raises:
        EmbeddingError: If sentence-transformers is not installed.
    """

    def __init__(self, config) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "HuggingFace provider requires sentence-transformers. "
                "Install with: pip install -r requirements-huggingface.txt "
                "or run: bash setup.sh --with-huggingface"
            ) from exc

        self._model_id = config.embedding.model
        self._dimensions = config.embedding.dimensions
        self._doc_prefix = config.embedding.document_prefix
        self._query_prefix = config.embedding.query_prefix
        self._batch_size = config.embedding.batch_size
        self._device: Optional[str] = config.embedding.device

        print(
            f"Loading embedding model {self._model_id}...",
            file=sys.stderr,
        )
        try:
            self._model = SentenceTransformer(
                self._model_id,
                trust_remote_code=True,
                device=self._device,  # None → auto (CUDA > MPS > CPU)
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load model {self._model_id}: {exc}"
            ) from exc

        actual_device = str(self._model.device)
        logger.info(
            "Loaded %s on device: %s", self._model_id, actual_device,
        )
        print(
            f"Model loaded on device: {actual_device}",
            file=sys.stderr,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts locally.

        Adds the document prefix and encodes via sentence-transformers.

        Args:
            texts: Raw text strings to embed.

        Returns:
            List of embedding vectors.
        """
        prefixed = [self._doc_prefix + t for t in texts]
        embeddings = self._model.encode(
            prefixed,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query locally.

        Adds the query prefix and encodes via sentence-transformers.

        Args:
            query: Natural language search query.

        Returns:
            Single embedding vector.
        """
        prefixed = self._query_prefix + query
        embedding = self._model.encode(
            [prefixed],
            convert_to_numpy=True,
        )
        return embedding[0].tolist()

    def get_dimensions(self) -> int:
        """Return the configured embedding dimensions."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Return the model identifier string."""
        return self._model_id
