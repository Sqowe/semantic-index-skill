"""Embedding API client for OpenRouter.

Handles batching, retry with exponential backoff, and local caching
of embeddings by content hash to minimize API costs on re-indexing.
"""

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import requests

from .config import Config, INDEX_DIR_NAME
from .models import Chunk, EmbeddingError

logger = logging.getLogger(__name__)

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
CACHE_FILENAME = "embedding_cache.json"


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


def _content_hash(text: str) -> str:
    """SHA-256 hash of text content for cache keying."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


class Embedder:
    """OpenRouter embedding API client with batching, retry, and caching."""

    def __init__(self, config: Config, project_dir: Optional[str] = None) -> None:
        self._config = config
        self._api_key = config.embedding.api_key
        self._model = config.embedding.model
        self._dimensions = config.embedding.dimensions
        self._batch_size = config.embedding.batch_size
        self._doc_prefix = config.embedding.document_prefix
        self._query_prefix = config.embedding.query_prefix
        self._max_retries = config.embedding.max_retries
        self._retry_delay = config.embedding.retry_delay_seconds
        self._cache: Optional[EmbeddingCache] = None

        if project_dir:
            self._cache = EmbeddingCache(project_dir, config)

        if not self._api_key:
            raise EmbeddingError(
                "No API key found. Set OPENROUTER_API_KEY environment variable "
                "or add api_key to .index/config.json"
            )

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call OpenRouter embeddings API with retry logic.

        Args:
            texts: List of texts to embed (already prefixed).

        Returns:
            List of embedding vectors in the same order as input.

        Raises:
            EmbeddingError: If all retries are exhausted.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "input": texts,
        }
        if self._dimensions:
            body["dimensions"] = self._dimensions

        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                resp = requests.post(
                    OPENROUTER_EMBEDDINGS_URL,
                    headers=headers,
                    json=body,
                    timeout=60,
                )

                # Handle rate limiting
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", self._retry_delay * (2 ** attempt)))
                    logger.warning("Rate limited, retrying in %.1fs", retry_after)
                    time.sleep(retry_after)
                    continue

                resp.raise_for_status()
                data = resp.json()

                # Sort by index to ensure correct ordering
                embeddings = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in embeddings]

            except requests.RequestException as exc:
                last_error = exc
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        "API call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self._max_retries, delay, exc,
                    )
                    time.sleep(delay)

        raise EmbeddingError(f"Embedding API failed after {self._max_retries} retries: {last_error}")


    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with document prefix.

        Args:
            texts: Raw text strings to embed.

        Returns:
            List of embedding vectors.
        """
        prefixed = [self._doc_prefix + t for t in texts]
        return self._call_api(prefixed)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query with query prefix.

        Args:
            query: Natural language search query.

        Returns:
            Single embedding vector.
        """
        prefixed = self._query_prefix + query
        vectors = self._call_api([prefixed])
        return vectors[0]

    def embed_chunks(self, chunks: list[Chunk]) -> int:
        """Embed a list of chunks, using cache where possible.

        Modifies chunks in-place by adding a 'vector' key to metadata.
        Returns the number of API calls made.

        Args:
            chunks: List of Chunk objects to embed.

        Returns:
            Number of API batch calls made.
        """
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
        for batch_start in range(0, len(uncached), self._batch_size):
            batch = uncached[batch_start:batch_start + self._batch_size]
            texts = [chunk.content for _, chunk in batch]

            print(
                f"  Embedding batch {batch_start // self._batch_size + 1}"
                f"/{(len(uncached) + self._batch_size - 1) // self._batch_size}"
                f" ({len(texts)} chunks)...",
                file=sys.stderr,
            )

            vectors = self.embed_texts(texts)
            api_calls += 1

            for (idx, chunk), vector in zip(batch, vectors):
                chunk.metadata["vector"] = vector
                if self._cache:
                    self._cache.set(_content_hash(chunk.content), vector)

        # Save cache
        if self._cache:
            self._cache.save()

        return api_calls
