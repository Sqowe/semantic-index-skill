"""OpenRouter embedding provider.

REST API client for OpenRouter's embedding endpoint with batching,
retry with exponential backoff, and rate limit handling.
"""

import logging
import math
import time
from typing import Optional

import requests

from ..chunkers.common import count_tokens
from ..models import EmbeddingError
from ..tokenizer_resolver import (
    TokenizerWrapper,
    resolver_for_config,
)

logger = logging.getLogger(__name__)

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

# Token margin kept inside ``max_embed_tokens`` whenever we truncate. The
# embedding API counts tokens with its own tokenizer; ours is at best the
# same model (so the count matches exactly) and at worst tiktoken cl100k
# (which under-counts bge-m3 by 1.30x median / 2.13x worst case). Fifty
# tokens of slack absorbs normal drift between local and remote tokenizers
# so the truncated prefix still lands under the model's context window.
_TRUNCATION_TOKEN_MARGIN = 50

# Floor on the per-step length in ``_progressive_truncate``. Below this
# the API request cannot meaningfully embed anything and we give up.
_MIN_TRUNCATE_CHARS = 100


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
        self._max_embed_tokens = config.embedding.max_embed_tokens
        # Resolver bound to the active embedding model. Used to count
        # tokens for the embed-time cap and for the direct-computation
        # progressive truncate. OpenRouter-only installs fall back to
        # tiktoken cl100k_base (see tokenizer_resolver.py).
        self._tokens: TokenizerWrapper = resolver_for_config(config)
        # First WARNING per build for pre-embed token-based truncation;
        # subsequent occurrences log at DEBUG. The Embedder is created
        # fresh per build, so this attribute is naturally per-build with
        # no reset needed.
        self._first_embed_truncate_warning_logged = False
        # First WARNING per build for single-text progressive truncate;
        # subsequent occurrences log at DEBUG. Same lifetime as above.
        self._first_progressive_truncate_warning_logged = False

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

        Adds the document prefix before sending to the API. The primary
        cap is the configured ``max_embed_tokens`` (counted with the
        resolver bound to the active embedding model); ``max_embed_chars``
        stays as a hard safety net for tokenizer drift — if our local
        count is significantly under the API's actual count, the char
        ceiling still prevents sending a payload the API will reject.

        A single text that still exceeds the model's token limit after
        this pre-truncation falls through to ``_progressive_truncate``,
        which computes the right cut directly from the measured token
        count instead of stepping down 25% at a time.

        Args:
            texts: Raw text strings to embed.

        Returns:
            List of embedding vectors.
        """
        prefixed = []
        truncated_count = 0
        for t in texts:
            full = self._doc_prefix + t
            truncated = self._cap_to_token_budget(full)
            if truncated is not full:
                truncated_count += 1
                full = truncated
            prefixed.append(full)
        if truncated_count:
            if not self._first_embed_truncate_warning_logged:
                logger.warning(
                    "Pre-truncated %d/%d texts to fit %d-token budget "
                    "(hard char ceiling: %d)",
                    truncated_count, len(texts),
                    self._max_embed_tokens, self._max_embed_chars,
                )
                self._first_embed_truncate_warning_logged = True
            else:
                logger.debug(
                    "Pre-truncated %d/%d texts to fit %d-token budget",
                    truncated_count, len(texts), self._max_embed_tokens,
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

    def _cap_to_token_budget(self, text: str) -> str:
        """Return *text* shortened to fit ``max_embed_tokens`` if needed.

        The token count uses the resolver bound to the active embedding
        model (the model's own tokenizer when reachable, tiktoken
        ``cl100k_base`` otherwise). When a shorten is required, the
        char cut is computed from the text's measured chars/token ratio
        with a ``_TRUNCATION_TOKEN_MARGIN`` slack so normal
        local-vs-remote tokenizer drift does not push the truncated
        payload back over the model's context window.

        ``max_embed_chars`` is applied after the token-based cut as a
        hard ceiling — a defense-in-depth measure for tokenizer drift
        that was not absorbed by the margin (a base64-like blob can
        have a wildly different chars/token ratio than the rest of the
        corpus).

        Returns *text* unchanged when it already fits; the identity
        check is intentional, callers compare with ``is`` to detect
        whether truncation happened.
        """
        measured_tokens = count_tokens(text, tokens=self._tokens)
        if measured_tokens <= self._max_embed_tokens:
            # Even if token-count is fine, the char ceiling is the hard
            # guard — tokenizer drift could still mean the API sees more
            # tokens than we measured.
            if len(text) <= self._max_embed_chars:
                return text
            return text[: self._max_embed_chars]

        # Token budget exceeded. Compute a char cut that targets
        # ``max_embed_tokens - margin`` tokens using the text's own
        # chars/token ratio.
        target_tokens = max(
            1, self._max_embed_tokens - _TRUNCATION_TOKEN_MARGIN,
        )
        chars_per_token = len(text) / measured_tokens
        target_chars = max(1, int(target_tokens * chars_per_token))
        truncated = text[:target_chars]

        # Hard char ceiling: if the token-based cut still exceeds the
        # chars ceiling, cut further. This is the defense-in-depth path
        # for the case where our local token count was wildly optimistic.
        if len(truncated) > self._max_embed_chars:
            truncated = truncated[: self._max_embed_chars]
        return truncated

    def _progressive_truncate(self, text: str) -> list[list[float]]:
        """Shorten a single text until it fits the model's context window.

        The first attempt uses a direct computation from the measured
        token count, targeting ``max_embed_tokens - margin`` tokens and
        deriving the char cut from the text's own chars/token ratio. In
        the common case this lands on the first try and no further
        round trips are made. If the API still rejects (e.g., the local
        count was wildly off because of a tokenizer mismatch), we fall
        back to a finer stepping loop that drops the limit by 25% per
        step until it fits or hits the ``_MIN_TRUNCATE_CHARS`` floor.

        Args:
            text: The prefixed text that exceeded the token limit.

        Returns:
            List containing a single embedding vector.

        Raises:
            RuntimeError: If text cannot be embedded even at
                ``_MIN_TRUNCATE_CHARS`` chars.
        """
        measured_tokens = count_tokens(text, tokens=self._tokens)
        target_tokens = max(
            1, self._max_embed_tokens - _TRUNCATION_TOKEN_MARGIN,
        )
        chars_per_token = len(text) / max(1, measured_tokens)
        target_chars = max(1, int(target_tokens * chars_per_token))
        # The hard char ceiling may sit below the computed target.
        # Also cap by the actual text length so we never send more
        # than we have — sending the full text again is exactly the
        # call that just failed, which would waste the iteration.
        target_chars = min(target_chars, len(text), self._max_embed_chars)

        if not self._first_progressive_truncate_warning_logged:
            logger.warning(
                "Single text exceeds token limit (%d tokens > %d); "
                "truncating to %d chars (was %d)",
                measured_tokens, self._max_embed_tokens,
                target_chars, len(text),
            )
            self._first_progressive_truncate_warning_logged = True
        else:
            logger.debug(
                "Single text exceeds token limit (%d tokens > %d); "
                "truncating to %d chars",
                measured_tokens, self._max_embed_tokens, target_chars,
            )

        try:
            return self._call_api([text[:target_chars]])
        except (RuntimeError, EmbeddingError) as retry_exc:
            exc_str = str(retry_exc).lower()
            if "context length" not in exc_str and "input length" not in exc_str:
                raise
            # Direct computation was insufficient — the local count was
            # under-counting. Fall back to a finer stepping loop from
            # ``target_chars`` downward. Track the last context-length
            # error so we can re-raise it (preserving the original
            # message text the Embedder matches on) if the loop
            # exhausts — that lets the Embedder split the batch.
            last_length_error = retry_exc
            limit = target_chars
            while limit > _MIN_TRUNCATE_CHARS:
                previous_chars = limit
                limit = limit * 3 // 4
                logger.debug(
                    "Direct truncation insufficient; retrying with %d chars "
                    "(was %d)", limit, previous_chars,
                )
                try:
                    return self._call_api([text[:limit]])
                except (RuntimeError, EmbeddingError) as retry_exc2:
                    exc_str = str(retry_exc2).lower()
                    if (
                        "context length" not in exc_str
                        and "input length" not in exc_str
                    ):
                        raise
                    last_length_error = retry_exc2
                    continue
            # If we get here, even the floor was rejected — re-raise
            # the last context-length error so the caller can split.
            # Wrap as RuntimeError because the Embedder's split path
            # only catches RuntimeError; an EmbeddingError instance
            # would propagate up unhandled and crash the build.
            raise RuntimeError(str(last_length_error)) from last_length_error

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
