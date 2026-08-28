"""HuggingFace local embedding provider.

Uses sentence-transformers for local inference. No API key needed.
Model is downloaded on first use to ~/.cache/huggingface/hub (~274MB
for Nomic, ~600MB for larger models). Subsequent runs load from cache.

Device auto-detection: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
Override with the `device` config field or leave null for auto.
"""

import logging
from typing import Optional

from ..chunkers.common import count_tokens
from ..models import EmbeddingError
from ..tokenizer_resolver import (
    TokenizerWrapper,
    resolver_for_config,
)

logger = logging.getLogger(__name__)

# Token margin kept inside ``max_embed_tokens`` whenever we truncate. The
# tokenizer running inside sentence-transformers is the model's own; ours
# is the same when ``tokenizers`` + the HF repo are reachable, tiktoken
# ``cl100k_base`` otherwise. Fifty tokens of slack absorbs drift between
# our local count and the model's internal count.
_TRUNCATION_TOKEN_MARGIN = 50

# Floor on the per-step length in the single-text OOM retry. Below this
# the model cannot meaningfully embed anything and we give up.
_MIN_TRUNCATE_CHARS = 100


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
        self._max_embed_tokens = config.embedding.max_embed_tokens
        # Resolver bound to the active embedding model. Used to count
        # tokens for the embed-time cap and for the direct-computation
        # OOM retry. ``sentence-transformers`` already pulls the model's
        # own tokenizer for inference; this resolver gives us a matching
        # count locally.
        self._tokens: TokenizerWrapper = resolver_for_config(config)
        # First WARNING per build for pre-embed token-based truncation;
        # subsequent occurrences log at DEBUG. The provider is created
        # fresh per build via Embedder / create_provider, so the
        # sentinel is naturally per-build.
        self._first_embed_truncate_warning_logged = False
        # First WARNING per build for the internal batch_size halving
        # path (``OOM at internal batch_size=N, retrying with M``).
        # Separate from the progressive-truncate sentinel below:
        # halving always fires before progressive truncate, so a
        # shared sentinel would suppress the progressive-truncate
        # WARNING to DEBUG on every build.
        self._first_halving_warning_logged = False
        # First WARNING per build for single-text OOM-driven progressive
        # truncate. Same lifetime as above.
        self._first_progressive_truncate_warning_logged = False
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
        The primary cap is the configured ``max_embed_tokens`` (counted
        with the resolver bound to the active embedding model);
        ``max_embed_chars`` stays as a hard safety net for tokenizer
        drift. On OOM (RuntimeError from PyTorch), automatically halves
        the internal batch size and retries until batch_size reaches 1;
        at that point the single text is shortened by a direct
        computation from the measured token count, with a finer
        stepping loop as a fallback.

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
                        return self._progressive_truncate_oom(prefixed[0])
                    # Re-raise the original RuntimeError so the caller
                    # (Embedder) can split the chunk batch to isolate
                    # the pathological chunk(s).
                    raise
                if not self._first_halving_warning_logged:
                    logger.warning(
                        "OOM at internal batch_size=%d, retrying with %d: %s",
                        batch_size, new_batch_size, exc,
                    )
                    self._first_halving_warning_logged = True
                else:
                    logger.debug(
                        "OOM at internal batch_size=%d, retrying with %d",
                        batch_size, new_batch_size,
                    )
                batch_size = new_batch_size

        # Unreachable, but satisfies type checkers
        raise EmbeddingError("Embedding failed after all retries")

    def _cap_to_token_budget(self, text: str) -> str:
        """Return *text* shortened to fit ``max_embed_tokens`` if needed.

        Mirrors ``OpenRouterProvider._cap_to_token_budget``. Uses the
        resolver bound to the active embedding model so the token count
        matches what sentence-transformers' internal tokenizer will see
        when the model actually runs. The hard char ceiling is applied
        last as a defense-in-depth measure for tokenizer drift.
        """
        measured_tokens = count_tokens(text, tokens=self._tokens)
        if measured_tokens <= self._max_embed_tokens:
            if len(text) <= self._max_embed_chars:
                return text
            return text[: self._max_embed_chars]

        target_tokens = max(
            1, self._max_embed_tokens - _TRUNCATION_TOKEN_MARGIN,
        )
        chars_per_token = len(text) / measured_tokens
        target_chars = max(1, int(target_tokens * chars_per_token))
        truncated = text[:target_chars]

        if len(truncated) > self._max_embed_chars:
            truncated = truncated[: self._max_embed_chars]
        return truncated

    def _progressive_truncate_oom(self, text: str) -> list[list[float]]:
        """Shorten a single text until sentence-transformers stops OOMing.

        First attempt is a direct computation from the measured token
        count, targeting ``max_embed_tokens - margin`` tokens and
        deriving the char cut from the text's own chars/token ratio. In
        the common case this lands on the first try and no further
        round trips are made. If OOM persists (e.g., the local count
        was wildly off), a finer stepping loop drops the limit by 25%
        per step until it fits or hits ``_MIN_TRUNCATE_CHARS``.

        Args:
            text: The prefixed text that exhausted internal batch_size
                halving and still OOMed at batch_size=1.

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
        # Cap by the actual text length so we never send more than we
        # have — sending the full text again is exactly the call that
        # just failed, which would waste the iteration.
        target_chars = min(target_chars, len(text), self._max_embed_chars)

        if not self._first_progressive_truncate_warning_logged:
            logger.warning(
                "Single text OOM (%d tokens > %d); truncating to %d chars "
                "(was %d)",
                measured_tokens, self._max_embed_tokens,
                target_chars, len(text),
            )
            self._first_progressive_truncate_warning_logged = True
        else:
            logger.debug(
                "Single text OOM (%d tokens > %d); truncating to %d chars",
                measured_tokens, self._max_embed_tokens, target_chars,
            )

        try:
            embeddings = self._model.encode(
                [text[:target_chars]],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.tolist()
        except RuntimeError as retry_exc:
            retry_msg = str(retry_exc).lower()
            if (
                "invalid buffer size" not in retry_msg
                and "out of memory" not in retry_msg
            ):
                raise
            # Direct computation was insufficient — fall back to a finer
            # stepping loop from ``target_chars`` downward. Track the
            # last OOM we saw so we can re-raise it (preserving the
            # "invalid buffer size" / "out of memory" markers) if the
            # loop exhausts. That lets the Embedder recognise the
            # error as recoverable and split the batch.
            last_oom = retry_exc
            limit = target_chars
            while limit > _MIN_TRUNCATE_CHARS:
                previous_chars = limit
                limit = limit * 3 // 4
                logger.debug(
                    "Direct truncation insufficient; retrying with %d chars "
                    "(was %d)", limit, previous_chars,
                )
                try:
                    embeddings = self._model.encode(
                        [text[:limit]],
                        batch_size=1,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
                    return embeddings.tolist()
                except RuntimeError as retry_exc2:
                    retry_msg = str(retry_exc2).lower()
                    if (
                        "invalid buffer size" not in retry_msg
                        and "out of memory" not in retry_msg
                    ):
                        raise
                    last_oom = retry_exc2
                    continue
            # If we get here, even the floor was rejected — re-raise
            # the last OOM so the caller (Embedder) can split the batch.
            # ``last_oom`` is already a RuntimeError caught from the
            # inner ``except RuntimeError`` clauses above, so it does
            # not need wrapping for the Embedder's split path — but
            # re-raising the original instance preserves the message
            # text the Embedder matches on ("Invalid buffer size" or
            # "out of memory").
            raise last_oom

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
