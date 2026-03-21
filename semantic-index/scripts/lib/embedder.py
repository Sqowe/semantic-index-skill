"""Embedding provider abstraction, factory, and caching layer.

Defines the EmbeddingProvider ABC, a factory function to instantiate
the correct provider based on config, and the EmbeddingCache for
on-disk caching of content-hash → vector mappings.

The Embedder class wraps a provider with caching and batch orchestration,
providing the same public API used by build_index.py and semantic_search.py.
"""

import hashlib
import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .config import Config, INDEX_DIR_NAME
from .models import Chunk, EmbeddingError

logger = logging.getLogger(__name__)

CACHE_FILENAME = "embedding_cache.json"


# ---------------------------------------------------------------------------
# Abstract provider interface
# ---------------------------------------------------------------------------

class EmbeddingProvider(ABC):
    """Base class for all embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts. Returns list of vectors.

        Provider handles prefixing (e.g., 'search_document:' for Nomic).
        """

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query. Returns one vector.

        Provider handles prefixing (e.g., 'search_query:' for Nomic).
        """

    @abstractmethod
    def get_dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def create_embedder(config: Config) -> EmbeddingProvider:
    """Factory: instantiate the right provider based on config.embedding.provider.

    Supported providers:
        - "openrouter": REST API via OpenRouter (requires API key)
        - "huggingface": Local inference via sentence-transformers (no API key)

    Provider imports are lazy — only the selected provider's dependencies
    are imported. This means sentence-transformers/torch are never imported
    if the user uses OpenRouter, and requests is never imported if the user
    uses HuggingFace.

    Args:
        config: Validated Config object.

    Returns:
        An EmbeddingProvider instance.

    Raises:
        EmbeddingError: If the provider is unknown or fails to initialize.
    """
    provider = config.embedding.provider

    if provider == "openrouter":
        from .providers.openrouter import OpenRouterProvider
        return OpenRouterProvider(config)
    elif provider == "huggingface":
        from .providers.huggingface import HuggingFaceProvider
        return HuggingFaceProvider(config)
    else:
        raise EmbeddingError(
            f"Unknown embedding provider: {provider!r}. "
            "Supported: 'openrouter', 'huggingface'"
        )


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """On-disk cache mapping content hashes to embedding vectors.

    The cache is invalidated if the model or dimensions change.
    """

    def __init__(self, project_dir: str, config: Config) -> None:
        self._path = Path(project_dir) / INDEX_DIR_NAME / CACHE_FILENAME
        self._model = config.embedding.model
        self._dimensions = config.embedding.dimensions
        self._entries: dict[str, list[float]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache unreadable, starting fresh: %s", exc)
            return

        # Invalidate if model or dimensions changed
        if raw.get("model") != self._model or raw.get("dimensions") != self._dimensions:
            logger.info(
                "Cache model/dimensions mismatch (cached: %s/%s, current: %s/%s), clearing",
                raw.get("model"), raw.get("dimensions"),
                self._model, self._dimensions,
            )
            return
        self._entries = raw.get("entries", {})
        logger.info("Loaded %d cached embeddings", len(self._entries))

    def has(self, content_hash: str) -> bool:
        return content_hash in self._entries

    def get(self, content_hash: str) -> Optional[list[float]]:
        return self._entries.get(content_hash)

    def set(self, content_hash: str, vector: list[float]) -> None:
        self._entries[content_hash] = vector
        self._dirty = True

    def save(self) -> None:
        """Persist cache to disk if modified."""
        if not self._dirty:
            return
        data = {
            "version": "1.0",
            "model": self._model,
            "dimensions": self._dimensions,
            "entries": self._entries,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data) + "\n", encoding="utf-8")
        logger.info("Saved %d embeddings to cache", len(self._entries))
        self._dirty = False


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------

def _content_hash(text: str) -> str:
    """SHA-256 hash of text content for cache keying."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


# ---------------------------------------------------------------------------
# Embedder wrapper (caching + batch orchestration)
# ---------------------------------------------------------------------------

class Embedder:
    """High-level embedding client with caching and batch orchestration.

    Wraps an EmbeddingProvider (created via factory) with the embedding
    cache and batch progress reporting. This is the class used by
    build_index.py and semantic_search.py.

    Args:
        config: Validated Config object.
        project_dir: Optional project directory for cache persistence.
            If None, caching is disabled.
    """

    def __init__(self, config: Config, project_dir: Optional[str] = None) -> None:
        self._config = config
        self._provider = create_embedder(config)
        self._cache: Optional[EmbeddingCache] = None

        if project_dir:
            self._cache = EmbeddingCache(project_dir, config)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via the underlying provider.

        Args:
            texts: Raw text strings to embed.

        Returns:
            List of embedding vectors.
        """
        return self._provider.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query via the underlying provider.

        Args:
            query: Natural language search query.

        Returns:
            Single embedding vector.
        """
        return self._provider.embed_query(query)

    def embed_chunks(self, chunks: list[Chunk]) -> int:
        """Embed a list of chunks, using cache where possible.

        Modifies chunks in-place by adding a 'vector' key to metadata.
        Returns the number of API/inference calls made.

        Args:
            chunks: List of Chunk objects to embed.

        Returns:
            Number of batch embedding calls made.
        """
        batch_size = self._config.embedding.batch_size
        api_calls = 0

        # Separate cached vs uncached
        uncached: list[tuple[int, Chunk]] = []
        for i, chunk in enumerate(chunks):
            ch = _content_hash(chunk.content)
            if self._cache and self._cache.has(ch):
                chunk.metadata["vector"] = self._cache.get(ch)
            else:
                uncached.append((i, chunk))

        if uncached:
            logger.info(
                "Embedding %d chunks (%d cached, %d to embed)",
                len(chunks), len(chunks) - len(uncached), len(uncached),
            )

        # Batch embed uncached chunks
        for batch_start in range(0, len(uncached), batch_size):
            batch = uncached[batch_start:batch_start + batch_size]
            texts = [chunk.content for _, chunk in batch]

            print(
                f"  Embedding batch {batch_start // batch_size + 1}"
                f"/{(len(uncached) + batch_size - 1) // batch_size}"
                f" ({len(texts)} chunks)...",
                file=sys.stderr,
            )

            vectors = self._provider.embed_texts(texts)
            api_calls += 1

            for (idx, chunk), vector in zip(batch, vectors):
                chunk.metadata["vector"] = vector
                if self._cache:
                    self._cache.set(_content_hash(chunk.content), vector)

        # Save cache
        if self._cache:
            self._cache.save()

        return api_calls
