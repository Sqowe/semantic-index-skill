"""Tokenizer resolver: pick the right tokenizer for the active embedding model.

Embedding models tokenize differently: tiktoken's ``cl100k_base`` (the default
in :mod:`lib.chunkers.common`) counts roughly 1.30x fewer tokens than
BAAI/bge-m3 for this project's content, with a worst-case ratio of 2.13x.
A chunk that passes the chunker budget at 512 cl100k tokens may be ~772
bge-m3 tokens at the API — well past the model's 8192-token limit when
many such chunks stack up, and over the limit for an individual chunk
when the ratio is at its worst.

When ``tokenizers`` is installed and the model's Hugging Face repo is
reachable (locally cached or online), the resolver returns a thin wrapper
around ``tokenizers.Tokenizer.from_pretrained(...)`` and counts tokens
exactly the way the embedding API will. When the real tokenizer is not
available — OpenRouter-only installs that did not pull in
``sentence-transformers``, offline runs whose cache misses, custom model
names that are not on the HF hub — the resolver falls back to tiktoken
``cl100k_base`` and the caller is expected to apply a safety factor
(``config.embedding.token_safety_factor``) to its budget before trusting
the count.

The resolver caches one tokenizer per model. It is safe to construct
freshly for each chunking run; the underlying tokenizer objects are
process-global via the cache.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

from .config import Config

logger = logging.getLogger(__name__)


class TokenizerWrapper:
    """Tiny shim around a tokenizer with the methods the codebase calls.

    Wraps either a ``tokenizers.Tokenizer`` (real model tokenizer) or a
    ``tiktoken.Encoding`` (fallback) behind the same ``encode``/``decode``
    interface. ``hard_split_by_tokens`` and friends talk only to this
    wrapper, so swapping tokenizers does not propagate through the
    chunker module.
    """

    def __init__(self, kind: str, inner) -> None:
        """Wrap a real tokenizer or a tiktoken encoding.

        Args:
            kind: ``"real"`` for ``tokenizers.Tokenizer`` (no_batch=True
                friendly, encoding-only fast path), ``"tiktoken"`` for
                ``tiktoken.Encoding``.
            inner: The wrapped object.
        """
        self._kind = kind
        self._inner = inner

    def encode(self, text: str) -> list[int]:
        """Encode *text* into a list of token ids.

        ``hard_split_by_tokens`` slices token windows out of this
        encoding. For real (SentencePiece-style) tokenizers the splitter
        skips the round-trip check entirely — they never split
        mid-character — so the presence of BOS/EOS markers from
        ``encode(text)`` does not matter for downstream slicing. The
        model's BOS/EOS handling at inference is its own concern.
        """
        if self._kind == "real":
            return self._inner.encode(text).ids
        return self._inner.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to text."""
        return self._inner.decode(ids)

    @property
    def kind(self) -> str:
        """Return ``"real"`` or ``"tiktoken"``."""
        return self._kind


_CACHE: Dict[str, TokenizerWrapper] = {}
_RESOLVE_ATTEMPTED: Dict[str, bool] = {}


def _try_load_real_tokenizer(model: str) -> Optional[TokenizerWrapper]:
    """Try to load the model's own tokenizer. Returns None on any failure.

    Failure paths are silent except at DEBUG, because OpenRouter-only
    installs are expected: ``tokenizers`` may not be installed, the
    model name may not be a HF repo, and the model may not be in the
    local HF cache when running offline. None of those are errors —
    the caller will fall back to tiktoken × safety_factor.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError:
        logger.debug(
            "tokenizers package not installed; using tiktoken fallback "
            "for model %s", model,
        )
        return None

    try:
        inner = Tokenizer.from_pretrained(model)
        return TokenizerWrapper("real", inner)
    except Exception as exc:
        logger.debug(
            "Could not load real tokenizer for %s (%s); using "
            "tiktoken fallback", model, exc,
        )
        return None


def get_resolver(model: str) -> TokenizerWrapper:
    """Return the cached tokenizer wrapper for *model*.

    Real-tokenizer load is attempted at most once per model per process;
    failures are cached so we do not pay the import cost on every call.
    """
    cached = _CACHE.get(model)
    if cached is not None:
        return cached

    if not _RESOLVE_ATTEMPTED.get(model, False):
        real = _try_load_real_tokenizer(model)
        _RESOLVE_ATTEMPTED[model] = True
        if real is not None:
            _CACHE[model] = real
            logger.debug("Using real tokenizer for %s", model)
            return real

    import tiktoken
    inner = tiktoken.get_encoding("cl100k_base")
    fallback = TokenizerWrapper("tiktoken", inner)
    _CACHE[model] = fallback
    logger.debug("Using tiktoken fallback for model %s", model)
    return fallback


def resolver_for_config(config: Config) -> TokenizerWrapper:
    """Convenience: return the resolver for ``config.embedding.model``."""
    return get_resolver(config.embedding.model)


def effective_max_tokens(
    embedding_max_tokens: int,
    safety_factor: float,
    is_real_tokenizer: bool,
) -> int:
    """Compute the embed-time token budget from the configured values.

    When the real tokenizer is loaded, the configured budget is used as
    is — the count is exact. When falling back to tiktoken, the safety
    factor shrinks the budget so a chunk that fits under
    ``embedding_max_tokens / safety_factor`` tiktoken tokens still fits
    under ``embedding_max_tokens`` real tokens in the worst observed
    case (1.6x covers the measured 1.30x median with room for the 2.13x
    worst case plus a small margin).

    Args:
        embedding_max_tokens: ``config.embedding.max_embed_tokens``.
        safety_factor: ``config.embedding.token_safety_factor`` (ignored
            when the real tokenizer is used).
        is_real_tokenizer: True if the active tokenizer is the model's
            own (e.g. bge-m3's), False for tiktoken fallback.

    Returns:
        The integer budget that callers should enforce.
    """
    if is_real_tokenizer:
        return embedding_max_tokens
    if safety_factor <= 0:
        return embedding_max_tokens
    return int(embedding_max_tokens / safety_factor)
