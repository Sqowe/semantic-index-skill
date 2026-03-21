"""OpenRouter embedding provider.

REST API client for OpenRouter's embedding endpoint with batching,
retry with exponential backoff, and rate limit handling.
"""

import logging
import math
import time
from typing import Optional

import requests

from ..models import EmbeddingError

logger = logging.getLogger(__name__)

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


class OpenRouterProvider:
    """OpenRouter REST API embedding provider.

    Implements the EmbeddingProvider interface for remote embedding
    via the OpenRouter API.

    Args:
        config: Config object with embedding settings.

    Raises:
        EmbeddingError: If no API key is available.
    """

    def __init__(self, config) -> None:
        self._api_key = config.embedding.api_key
        self._model = config.embedding.model
        self._dimensions = config.embedding.dimensions
        self._batch_size = config.embedding.batch_size
        self._doc_prefix = config.embedding.document_prefix
        self._query_prefix = config.embedding.query_prefix
        self._max_retries = config.embedding.max_retries
        self._retry_delay = config.embedding.retry_delay_seconds

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
        body: dict = {
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
                    fallback_delay = self._retry_delay * (2 ** attempt)
                    raw_retry = resp.headers.get("Retry-After")
                    try:
                        retry_after = float(raw_retry) if raw_retry else fallback_delay
                    except (ValueError, TypeError):
                        logger.warning(
                            "Non-numeric Retry-After header: %r, using backoff %.1fs",
                            raw_retry, fallback_delay,
                        )
                        retry_after = fallback_delay
                    if not math.isfinite(retry_after) or retry_after <= 0:
                        logger.warning(
                            "Invalid Retry-After value: %r, using backoff %.1fs",
                            retry_after, fallback_delay,
                        )
                        retry_after = fallback_delay
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

        raise EmbeddingError(
            f"Embedding API failed after {self._max_retries} retries: {last_error}"
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts.

        Adds the document prefix before sending to the API.

        Args:
            texts: Raw text strings to embed.

        Returns:
            List of embedding vectors.
        """
        prefixed = [self._doc_prefix + t for t in texts]
        return self._call_api(prefixed)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query.

        Adds the query prefix before sending to the API.

        Args:
            query: Natural language search query.

        Returns:
            Single embedding vector.
        """
        prefixed = self._query_prefix + query
        vectors = self._call_api([prefixed])
        return vectors[0]

    def get_dimensions(self) -> int:
        """Return the configured embedding dimensions."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Return the model identifier string."""
        return self._model
