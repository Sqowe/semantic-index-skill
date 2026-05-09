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
        self._max_embed_chars = config.embedding.max_embed_chars

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

                # Handle context/input length exceeded (400/413/422) — raise as
                # RuntimeError so the batch-splitting logic in Embedder
                # can catch it and retry with smaller batches.
                #
                # 413 is always a payload size issue — trigger split unconditionally.
                if resp.status_code == 413:
                    try:
                        err_body = resp.json()
                    except Exception:
                        err_body = resp.text[:300] or "Payload Too Large"
                    logger.warning(
                        "Payload too large for batch of %d texts "
                        "(HTTP 413), signaling for batch split: %s",
                        len(texts),
                        str(err_body)[:300],
                    )
                    raise RuntimeError(
                        f"context length exceeded: HTTP 413 - {str(err_body)[:300]}"
                    )

                # 400/422 may be length errors — check message keywords.
                if resp.status_code in (400, 422):
                    try:
                        err_body = resp.json()
                    except Exception:
                        err_body = resp.text[:500]
                    err_str = str(err_body).lower()
                    is_length_error = (
                        "context length" in err_str
                        or "too many tokens" in err_str
                        or "input sequence" in err_str
                        or "input length" in err_str
                        or "maximum context" in err_str
                        or "token limit" in err_str
                        or "payload too large" in err_str
                    )
                    if is_length_error:
                        logger.warning(
                            "Input length exceeded for batch of %d texts "
                            "(HTTP %d), signaling for batch split: %s",
                            len(texts),
                            resp.status_code,
                            str(err_body)[:300],
                        )
                        raise RuntimeError(
                            f"context length exceeded: {str(err_body)[:300]}"
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

                # Type guard: ensure response is a dict
                if not isinstance(data, dict):
                    snippet = str(data)[:200]
                    logger.error(
                        "API returned non-dict response (type=%s): %s",
                        type(data).__name__,
                        snippet,
                    )
                    raise EmbeddingError(
                        f"Unexpected API response type "
                        f"({type(data).__name__}): {snippet}"
                    )

                # Log response keys for debugging
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("API response keys: %s", list(data.keys()))

                # Validate response structure
                if "data" not in data:
                    error_msg = data.get("error", {})
                    # Flatten nested error messages for detection.
                    # OpenRouter wraps upstream errors as:
                    #   {"error": {"message": "HTTP 4xx: {...}", "code": N}}
                    # or {"error": "string message"}
                    if isinstance(error_msg, dict):
                        sanitized = str(error_msg.get("message", error_msg))[:500]
                    else:
                        sanitized = str(error_msg)[:500]

                    # Check if this is a context/input length error
                    # wrapped in a 200 response (OpenRouter proxying upstream errors)
                    sanitized_lower = sanitized.lower()
                    is_length_error = (
                        "context length" in sanitized_lower
                        or "too many tokens" in sanitized_lower
                        or "input sequence" in sanitized_lower
                        or "input length" in sanitized_lower
                        or "maximum context" in sanitized_lower
                        or "token limit" in sanitized_lower
                        or "payload too large" in sanitized_lower
                        or "request entity too large" in sanitized_lower
                    )
                    if is_length_error:
                        logger.warning(
                            "Input length exceeded for batch of %d texts "
                            "(wrapped in 200 response), signaling for batch split: %s",
                            len(texts),
                            sanitized[:300],
                        )
                        raise RuntimeError(
                            f"context length exceeded: {sanitized[:300]}"
                        )

                    logger.error(
                        "Unexpected API response (no 'data' field). "
                        "Error payload: %s",
                        sanitized,
                    )
                    raise EmbeddingError(
                        f"Unexpected API response (no 'data' field): {sanitized}"
                    )

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
        Truncates texts exceeding max_embed_chars to prevent API errors.
        If a single text still exceeds the model's token limit after
        truncation, progressively reduces it by 25% until it fits.

        Args:
            texts: Raw text strings to embed.

        Returns:
            List of embedding vectors.
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

        try:
            return self._call_api(prefixed)
        except RuntimeError as exc:
            # Only retry truncation for context-length errors
            if "context length" not in str(exc).lower():
                raise
            # If batch has multiple texts, re-raise for batch splitting
            if len(prefixed) > 1:
                raise
            # Single text still too long — progressively truncate
            return self._progressive_truncate(prefixed[0])
        except EmbeddingError as exc:
            # Catch context-length errors that came through a different path
            exc_str = str(exc).lower()
            if "context length" not in exc_str and "input length" not in exc_str:
                raise
            if len(prefixed) > 1:
                # Re-raise as RuntimeError for batch splitting
                raise RuntimeError(f"context length exceeded: {exc}") from exc
            return self._progressive_truncate(prefixed[0])

    def _progressive_truncate(self, text: str) -> list[list[float]]:
        """Progressively reduce text length by 25% until it fits the model.

        Args:
            text: The prefixed text that exceeded the token limit.

        Returns:
            List containing a single embedding vector.

        Raises:
            RuntimeError: If text cannot be embedded even at 100 chars.
        """
        limit = len(text)
        while limit > 100:
            limit = limit * 3 // 4  # reduce by 25% each iteration
            logger.warning(
                "Single text still exceeds token limit, "
                "retrying with %d chars (was %d)",
                limit,
                len(text),
            )
            try:
                return self._call_api([text[:limit]])
            except (RuntimeError, EmbeddingError) as retry_exc:
                exc_str = str(retry_exc).lower()
                if "context length" not in exc_str and "input length" not in exc_str:
                    raise
                continue
        # If we get here, even 100 chars fails — give up
        raise RuntimeError(
            f"context length exceeded: text cannot be embedded even at 100 chars"
        )

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query.

        Adds the query prefix before sending to the API.
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
        vectors = self._call_api([prefixed])
        return vectors[0]

    def get_dimensions(self) -> int:
        """Return the configured embedding dimensions."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Return the model identifier string."""
        return self._model
