"""HuggingFace local embedding provider.

Uses sentence-transformers for local inference. No API key needed.
Model is downloaded on first use to ~/.cache/huggingface/hub (~274MB
for Nomic, ~600MB for larger models). Subsequent runs load from cache.

Device auto-detection: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
Override with the `device` config field or leave null for auto.
"""

import logging
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
        self._max_embed_chars = config.embedding.max_embed_chars
        self._device: Optional[str] = config.embedding.device
        self._trust_remote_code: bool = getattr(
            config.embedding, "trust_remote_code", False,
        )

        if self._trust_remote_code:
            logger.warning(
                "trust_remote_code is enabled for model %s. "
                "This allows the model repository to execute arbitrary code.",
                self._model_id,
            )

        logger.info("Loading embedding model %s...", self._model_id)
        try:
            self._model = SentenceTransformer(
                self._model_id,
                trust_remote_code=self._trust_remote_code,
                device=self._device,  # None → auto (CUDA > MPS > CPU)
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load model {self._model_id}: {exc}"
            ) from exc

        # Validate dimensions: model's actual output must match config
        actual_dim = self._model.get_sentence_embedding_dimension()
        if actual_dim != self._dimensions:
            raise EmbeddingError(
                f"Dimension mismatch: model {self._model_id} produces "
                f"{actual_dim}-d vectors, but config specifies "
                f"embedding.dimensions={self._dimensions}. "
                f"Update config to match the model's native dimensions."
            )

        actual_device = str(self._model.device)
        logger.info(
            "Loaded %s on device: %s", self._model_id, actual_device,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts locally.

        Adds the document prefix and encodes via sentence-transformers.
        Truncates texts exceeding max_embed_chars to prevent OOM/errors.
        On OOM (RuntimeError from PyTorch), automatically halves the
        internal batch size and retries until batch_size reaches 1.

        Args:
            texts: Raw text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If encoding fails even at batch_size=1.
        """
        prefixed = []
        truncated_count = 0
        for t in texts:
            full = self._doc_prefix + t
            if len(full) > self._max_embed_chars:
                truncated_count += 1
                full = full[: self._max_embed_chars]
            prefixed.append(full)
        if truncated_count:
            logger.warning(
                "Truncated %d/%d texts to %d chars for embedding",
                truncated_count,
                len(texts),
                self._max_embed_chars,
            )
        batch_size = self._batch_size

        while batch_size >= 1:
            try:
                embeddings = self._model.encode(
                    prefixed,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                return embeddings.tolist()
            except RuntimeError as exc:
                if "Invalid buffer size" not in str(exc) and "out of memory" not in str(exc).lower():
                    raise
                new_batch_size = max(1, batch_size // 2)
                if new_batch_size == batch_size:
                    # Single text OOM — try progressive truncation
                    if len(prefixed) == 1:
                        text = prefixed[0]
                        limit = len(text)
                        while limit > 100:
                            limit = limit * 3 // 4
                            logger.warning(
                                "Single text OOM, retrying with %d chars (was %d)",
                                limit,
                                len(text),
                            )
                            try:
                                embeddings = self._model.encode(
                                    [text[:limit]],
                                    batch_size=1,
                                    show_progress_bar=False,
                                    convert_to_numpy=True,
                                )
                                return embeddings.tolist()
                            except RuntimeError as retry_exc:
                                retry_msg = str(retry_exc).lower()
                                if "invalid buffer size" not in retry_msg and "out of memory" not in retry_msg:
                                    raise
                                continue
                    # Re-raise the original RuntimeError so the caller
                    # (Embedder) can split the chunk batch to isolate
                    # the pathological chunk(s).
                    raise
                logger.warning(
                    "OOM at internal batch_size=%d, retrying with %d: %s",
                    batch_size, new_batch_size, exc,
                )
                batch_size = new_batch_size

        # Unreachable, but satisfies type checkers
        raise EmbeddingError("Embedding failed after all retries")

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query locally.

        Adds the query prefix and encodes via sentence-transformers.
        Truncates if exceeding max_embed_chars.

        Args:
            query: Natural language search query.

        Returns:
            Single embedding vector.
        """
        prefixed = self._query_prefix + query
        if len(prefixed) > self._max_embed_chars:
            logger.warning(
                "Truncating query from %d to %d chars for embedding",
                len(prefixed),
                self._max_embed_chars,
            )
            prefixed = prefixed[: self._max_embed_chars]
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
