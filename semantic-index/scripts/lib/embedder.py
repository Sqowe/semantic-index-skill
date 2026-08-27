"""Embedding provider abstraction, factory, and caching layer.

Defines the EmbeddingProvider ABC, a factory function to instantiate
the correct provider based on config, and the EmbeddingCache for
on-disk caching of content-hash → vector mappings.

The Embedder class wraps a provider with caching and batch orchestration,
providing the same public API used by build_index.py and semantic_search.py.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Optional

from .chunkers.common import count_tokens, hard_split_by_tokens
from .config import Config
from .embedding_cache import EmbeddingCache
from .models import Chunk, EmbeddingError, TruncationRecord
from .tokenizer_resolver import TokenizerWrapper, resolver_for_config

logger = logging.getLogger(__name__)


# Margin kept inside ``max_embed_tokens`` whenever the Embedder pre-truncates
# a chunk. The provider (and the API) count tokens with their own tokenizer;
# our count is exact only when the resolver loads the model's own tokenizer.
# ``token_safety_factor`` already shrinks the chunker-side budget for the
# tiktoken fallback; this margin absorbs the smaller, per-call local-vs-remote
# drift that even a real tokenizer can exhibit on adversarial input.
_EMBED_TOKEN_MARGIN = 50


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

def create_provider(config: Config) -> EmbeddingProvider:
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


def create_embedder(config: Config) -> EmbeddingProvider:
    """Deprecated: use create_provider() instead."""
    import warnings
    warnings.warn(
        "create_embedder() is deprecated, use create_provider()",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_provider(config)


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
        self._provider = create_provider(config)
        self._cache: Optional[EmbeddingCache] = None
        # Resolver bound to the active embedding model. Used for the
        # pre-embed token cap and for the largest-out isolation. Cheap to
        # construct — the underlying tokenizer is process-cached.
        self._tokens: TokenizerWrapper = resolver_for_config(config)
        self._max_embed_tokens = config.embedding.max_embed_tokens
        # The provider prefixes every document with ``document_prefix``
        # before sending. Measure its token count once at init so the
        # pre-truncate budget accounts for the prefix overhead — that
        # way the provider's own token cap is never tighter than the
        # Embedder's truncation result, and ``TruncationRecord.final_tokens``
        # accurately reflects what was actually embedded.
        self._doc_prefix_tokens = count_tokens(
            config.embedding.document_prefix, tokens=self._tokens,
        )
        # First WARNING per build for pre-embed token-based truncation;
        # subsequent occurrences log at DEBUG. The Embedder is created
        # fresh per build (one per ``build_index.py`` invocation), so
        # this attribute is naturally per-build without explicit reset.
        self._first_pre_truncate_warning_logged = False
        # First WARNING per build for batch splits on recoverable errors
        # (OOM / context length). The same lifetime as above.
        self._first_batch_split_warning_logged = False
        # Public read-after-call attribute populated by ``embed_chunks``
        # with every chunk the pre-embed cap had to shorten. Accumulates
        # across calls within a single Embedder lifetime; ``build_index``
        # sums them at the end of the build. The Embedder is created
        # fresh per ``build_index.py`` invocation, so per-build scoping
        # is implicit — there is no explicit reset.
        self.truncation_stats: list[TruncationRecord] = []

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

        Pre-truncates oversized chunks using the embedding model's
        tokenizer before sending them to the provider. The provider's
        own cap (character or token based) is the defense-in-depth net
        for any drift between local and remote tokenizers. Records every
        truncated chunk on ``self.truncation_stats`` and marks the chunk
        in place with ``metadata["truncated"] = True`` plus
        ``metadata["original_token_count"] = N`` so the build summary
        can report the count and the stored chunk advertises the loss.

        On a recoverable embed-time error (OOM or context-length), the
        batch is split by isolating the largest chunk and retrying the
        remainder as one batch. This costs one extra request to find
        one bad chunk in a 32-batch, instead of the five a blind halving
        would burn. The largest chunk is then retried alone; if it
        still fails, the recursion only continues on the remainder (which
        usually succeeded at the first try).

        Modifies chunks in-place by adding a 'vector' key to metadata.
        Returns the number of successful API/inference calls made.
        ``truncation_stats`` accumulates across calls within a build
        (the Embedder is created fresh per build, so its sentinels are
        naturally per-build without an explicit reset).

        Args:
            chunks: List of Chunk objects to embed.

        Returns:
            Number of successful batch embedding calls made.
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

        # Pre-truncate chunks whose token count exceeds the embedding
        # budget. The provider still has its own char ceiling as a hard
        # safety net, but doing this here means we get chunk-level
        # visibility (metadata marker + TruncationRecord) instead of
        # silently chopping content inside the provider. Cache each
        # chunk's measured token count so the largest-out isolation
        # below can find the biggest without re-tokenizing.
        pre_truncated = 0
        chunk_token_counts: dict[int, int] = {}
        for orig_idx, chunk in uncached:
            chunk_tokens = count_tokens(chunk.content, tokens=self._tokens)
            chunk_token_counts[orig_idx] = chunk_tokens
            if self._pre_truncate_chunk(chunk):
                pre_truncated += 1
                # The chunk's content has changed; re-measure.
                chunk_token_counts[orig_idx] = count_tokens(
                    chunk.content, tokens=self._tokens,
                )
        if pre_truncated:
            if not self._first_pre_truncate_warning_logged:
                logger.warning(
                    "Pre-truncated %d chunk(s) to fit %d-token budget "
                    "(originals stored in full; only the surviving prefix is "
                    "covered by the embedding vector)",
                    pre_truncated, self._max_embed_tokens,
                )
                self._first_pre_truncate_warning_logged = True
            else:
                logger.debug(
                    "Pre-truncated %d chunk(s) to fit %d-token budget",
                    pre_truncated, self._max_embed_tokens,
                )

        # Batch embed uncached chunks with recoverable-error isolation.
        # On an OOM / context-length RuntimeError the largest chunk is
        # pulled out and the remainder retried as one batch. If the
        # remainder itself fails, the same logic recurses on it.
        pending_batches: deque[list[tuple[int, Chunk]]] = deque()
        for batch_start in range(0, len(uncached), batch_size):
            pending_batches.append(uncached[batch_start:batch_start + batch_size])

        total_batches = len(pending_batches)
        batch_num = 0

        while pending_batches:
            batch = pending_batches.popleft()
            batch_num += 1
            texts = [chunk.content for _, chunk in batch]

            logger.info(
                "Embedding batch %d/%d (%d chunks)...",
                batch_num, total_batches, len(texts),
            )

            try:
                vectors = self._provider.embed_texts(texts)
            except RuntimeError as exc:
                err_msg = str(exc).lower()
                is_oom = (
                    "invalid buffer size" in err_msg
                    or "out of memory" in err_msg
                )
                is_context_length = "context length" in err_msg
                if not is_oom and not is_context_length:
                    raise
                if len(batch) <= 1:
                    raise EmbeddingError(
                        f"Cannot embed single chunk "
                        f"({len(texts[0])} chars, "
                        f"~{count_tokens(texts[0], tokens=self._tokens)} "
                        f"tokens): exceeds model limits. Reduce "
                        f"chunking.max_tokens or use a model with a larger "
                        f"context window. Original error: {exc}"
                    ) from exc
                # Largest-out isolation: pull the biggest chunk out and
                # retry the remainder as one batch. This costs one extra
                # request to find one bad chunk in a 32-batch, instead of
                # the five blind halving would burn. Token counts were
                # cached during pre-truncation so we do not re-tokenize
                # every chunk on the error path.
                largest_idx = max(
                    range(len(batch)),
                    key=lambda i: chunk_token_counts[batch[i][0]],
                )
                largest = batch[largest_idx]
                remainder = batch[:largest_idx] + batch[largest_idx + 1:]
                reason = "OOM" if is_oom else "context length exceeded"
                if not self._first_batch_split_warning_logged:
                    logger.warning(
                        "%s on batch of %d chunks; pulling the largest "
                        "(%d tokens) out and retrying %d as one batch: %s",
                        reason, len(batch),
                        chunk_token_counts[batch[largest_idx][0]],
                        len(remainder), exc,
                    )
                    self._first_batch_split_warning_logged = True
                else:
                    logger.debug(
                        "%s on batch of %d chunks; pulling the largest "
                        "(%d tokens) out and retrying %d as one batch",
                        reason, len(batch),
                        chunk_token_counts[batch[largest_idx][0]],
                        len(remainder),
                    )
                # Process the remainder first (one batch), then the
                # largest alone. If the remainder itself fails, the same
                # isolation runs on it. The order matters: pushing the
                # largest first and the remainder second means popleft
                # returns the remainder first (we want the bigger work
                # done up front so the failure case is the cheap retry).
                # ``remainder`` is always non-empty here: the
                # ``len(batch) <= 1`` guard above already raised
                # EmbeddingError, and removing one element from a
                # batch of size >= 2 leaves at least one element.
                pending_batches.appendleft([largest])
                pending_batches.appendleft(remainder)
                total_batches += 1  # one batch became two
                batch_num -= 1  # re-count since we didn't finish this one
                continue

            api_calls += 1

            for (idx, chunk), vector in zip(batch, vectors):
                chunk.metadata["vector"] = vector
                if self._cache:
                    self._cache.set(_content_hash(chunk.content), vector)

        # Save cache
        if self._cache:
            self._cache.save()

        return api_calls

    def _pre_truncate_chunk(self, chunk: Chunk) -> bool:
        """Shorten a chunk whose token count exceeds the embed budget.

        Uses the resolver bound to the active embedding model so the
        count matches what the API will see. The char cut is derived
        from the chunk's own measured chars/token ratio, with a small
        margin of slack so tokenizer drift does not push the truncated
        payload back over the budget.

        On truncation:

        * ``chunk.metadata["truncated"] = True`` and
          ``chunk.metadata["original_token_count"] = N`` flag the loss.
        * The full original text is preserved as
          ``chunk.metadata["original_content"]`` so search results and
          later inspection can read the whole chunk; only the
          embedding vector covers the surviving prefix.
        * A ``TruncationRecord`` is appended to ``self.truncation_stats``
          so the build summary can report the count.

        The chunk's ``content`` field is replaced with the truncated
        prefix because that is what gets sent to the API for embedding;
        the metadata carries the original for display.

        Storage note: ``metadata["original_content"]`` duplicates text
        that is also on disk in the source file. For pathological
        chunks this can inflate the index proportionally to the volume
        of truncated content. Acceptable today because truncation is
        rare (a 10k-token chunk that gets cut to 8k tokens adds ~2k
        chars per chunk to the index), but if truncation becomes
        frequent in a corpus, switch to a content hash here and look
        the original up from disk in ``index_status.py`` /
        ``semantic_search.py`` when a truncated chunk is surfaced.

        Returns True if a truncation happened, False otherwise.
        """
        text = chunk.content
        measured_tokens = count_tokens(text, tokens=self._tokens)
        if measured_tokens <= self._max_embed_tokens:
            return False

        # Reserve room for the provider's document_prefix so the
        # Embedder's truncation result lands below the provider's
        # token cap. Without this, the provider's _cap_to_token_budget
        # may shorten the text further, making ``final_tokens`` here
        # overstate what was actually embedded. ``_doc_prefix_tokens``
        # is measured once at Embedder init.
        target_tokens = max(
            1,
            self._max_embed_tokens
            - _EMBED_TOKEN_MARGIN
            - self._doc_prefix_tokens,
        )
        chars_per_token = len(text) / max(1, measured_tokens)
        target_chars = max(1, int(target_tokens * chars_per_token))
        truncated_text = text[:target_chars]

        # If the resolver's chunk-side hard_split lands within the
        # margin, prefer it: it guarantees the truncated text is at or
        # below the budget by token boundary, not by an approximate char
        # cut that may still be a token over. This is the same
        # machinery ``chunkers.common.build_chunks`` uses. Pass
        # ``target_tokens`` (already clamped to >= 1) rather than the
        # raw subtraction so a tiny ``max_embed_tokens`` budget can't
        # produce a negative max_tokens argument.
        pieces = hard_split_by_tokens(
            truncated_text,
            target_tokens,
            tokens=self._tokens,
        )
        if pieces:
            candidate = pieces[0]
            # Only switch to the boundary-clean piece if it is not
            # materially shorter than the char cut (the hard split can
            # drop to a small fraction when the round-trip check shrinks
            # the window; for SentencePiece-style tokenizers that step
            # is skipped, so the candidate should be close to the char
            # cut already).
            if len(candidate) >= int(len(truncated_text) * 0.5):
                truncated_text = candidate

        final_tokens = count_tokens(truncated_text, tokens=self._tokens)
        self.truncation_stats.append(TruncationRecord(
            file_path=chunk.file_path,
            chunk_id=chunk.id,
            original_tokens=measured_tokens,
            final_tokens=final_tokens,
        ))
        chunk.metadata["truncated"] = True
        chunk.metadata["original_token_count"] = measured_tokens
        # Preserve the full original text in metadata so search results
        # and later inspection can still read the whole chunk. Only the
        # embedding vector covers the surviving prefix; the marker above
        # advertises the loss.
        chunk.metadata["original_content"] = text
        chunk.content = truncated_text
        return True
