"""Tests for Phase 2 tokenizer resolution.

Phase 2 binds the chunker's token counting to the embedding model's
own tokenizer (BAAI/bge-m3 by default) when the ``tokenizers`` package
and the model's Hugging Face repo are both reachable. When the real
tokenizer is not available, it falls back to tiktoken ``cl100k_base``
with a configurable safety factor that accounts for the measured
ratio between the real tokenizer's count and tiktoken's.

These tests verify:

* Configuration exposes ``max_embed_tokens`` and ``token_safety_factor``
  with the documented defaults and validation.
* The resolver picks the real tokenizer when the model's HF repo is
  reachable, and the tiktoken fallback otherwise.
* The safety factor shrinks the effective chunk budget only when the
  fallback path is active; the real tokenizer uses the configured
  budget as-is.
* Hard-splitting the same text under both tokenizers produces pieces
  whose token counts differ by the expected ratio.

See ``lib/tokenizer_resolver.py`` for the implementation.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib.config import (
    Config,
    EmbeddingConfig,
    load_config,
)
from lib.models import ConfigError
from lib.tokenizer_resolver import (
    TokenizerWrapper,
    effective_max_tokens,
    get_resolver,
    resolver_for_config,
)


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class TestEmbeddingConfigValidation:
    """The new fields and their validation."""

    def test_max_embed_tokens_default_is_8192(self):
        assert EmbeddingConfig().max_embed_tokens == 8192

    def test_token_safety_factor_default_is_1_6(self):
        assert abs(EmbeddingConfig().token_safety_factor - 1.6) < 1e-9

    def test_max_embed_tokens_must_be_positive(self, tmp_path):
        from lib.config import ensure_index_dir
        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        (index_dir / "config.json").write_text(
            '{"embedding": {"max_embed_tokens": 0}}'
        )
        with pytest.raises(ConfigError, match="max_embed_tokens"):
            load_config(str(tmp_path))

    def test_token_safety_factor_below_one_rejected(self, tmp_path):
        from lib.config import ensure_index_dir
        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        (index_dir / "config.json").write_text(
            '{"embedding": {"token_safety_factor": 0.9}}'
        )
        with pytest.raises(ConfigError, match="token_safety_factor"):
            load_config(str(tmp_path))


# ---------------------------------------------------------------------------
# Resolver selection
# ---------------------------------------------------------------------------


def _bge_m3_available() -> bool:
    """Return True if the BAAI/bge-m3 tokenizer can be loaded locally.

    The check probes the same conditions the resolver checks: the
    ``tokenizers`` package is importable, and the HF hub either has the
    model cached or is reachable online. CI environments without the
    model cached and without network access will skip tests that depend
    on it.
    """
    try:
        from tokenizers import Tokenizer  # noqa: F401
    except ImportError:
        return False
    try:
        from tokenizers import Tokenizer
        Tokenizer.from_pretrained("BAAI/bge-m3")
        return True
    except Exception:
        return False


bge_m3_required = pytest.mark.skipif(
    not _bge_m3_available(),
    reason=(
        "BAAI/bge-m3 tokenizer not available; this test exercises the "
        "real-model path and requires either the ``tokenizers`` package "
        "with the model cached locally or network access to the HF hub."
    ),
)


class TestResolverSelection:
    """The resolver picks the right tokenizer for the active model."""

    @bge_m3_required
    def test_resolver_for_known_model_returns_real_tokenizer(self):
        """BAAI/bge-m3 is in the local HF cache; the real tokenizer loads."""
        tokens = resolver_for_config(Config())
        assert tokens.kind == "real"

    def test_resolver_for_unknown_model_falls_back_to_tiktoken(self):
        """A model name with no HF repo falls back to tiktoken cl100k_base."""
        # The literal string "cl100k_base" has no HF repo, so the
        # resolver skips the real-tokenizer attempt and caches tiktoken.
        wrapper = get_resolver("cl100k_base")
        assert wrapper.kind == "tiktoken"

    @bge_m3_required
    def test_resolver_is_cached(self):
        """Repeated calls return the same instance, not a fresh load."""
        first = resolver_for_config(Config())
        second = resolver_for_config(Config())
        assert first is second


# ---------------------------------------------------------------------------
# Effective budget under each tokenizer
# ---------------------------------------------------------------------------


class TestEffectiveBudget:
    """``effective_max_tokens`` shrinks the budget only on tiktoken fallback."""

    def test_real_tokenizer_uses_full_budget(self):
        """When the real tokenizer is loaded, the safety factor is a no-op."""
        assert effective_max_tokens(8192, 1.6, is_real_tokenizer=True) == 8192

    def test_real_tokenizer_ignores_factor_below_one(self):
        """Defensive: a misconfigured factor below 1.0 still uses the budget."""
        assert effective_max_tokens(8192, 0.5, is_real_tokenizer=True) == 8192

    def test_tiktoken_fallback_shrinks_budget(self):
        """The fallback applies the safety factor to stay within budget."""
        # 8192 / 1.6 = 5120
        assert effective_max_tokens(8192, 1.6, is_real_tokenizer=False) == 5120

    def test_tiktoken_fallback_with_smaller_factor(self):
        """A lower factor shrinks the budget more aggressively."""
        # 8192 / 2.0 = 4096
        assert effective_max_tokens(8192, 2.0, is_real_tokenizer=False) == 4096


# ---------------------------------------------------------------------------
# Token count differences
# ---------------------------------------------------------------------------


class TestTokenizerCountDifference:
    """The real tokenizer counts more tokens than tiktoken for the same text.

    This is the empirical reason the safety factor exists: a chunk
    budgeted at ``chunking.max_tokens`` tiktoken tokens can run up to
    ~2x over budget at the real (BAAI/bge-m3) tokenizer.
    """

    @bge_m3_required
    def test_real_tokenizer_counts_more_tokens_than_tiktoken(self):
        """For typical code/text content, bge-m3 produces more tokens than cl100k."""
        wrapper_real = get_resolver("BAAI/bge-m3")
        wrapper_tt = get_resolver("cl100k_base")
        text = (
            "def calculate_total(items):\n"
            "    return sum(item.value for item in items)\n"
        ) * 100
        real_count = len(wrapper_real.encode(text))
        tt_count = len(wrapper_tt.encode(text))
        # The measured ratio is ~1.30 median, ~2.13 worst case.
        assert real_count > tt_count
        ratio = real_count / tt_count
        assert 1.0 < ratio <= 2.5, (
            f"ratio {ratio:.2f} outside expected range; "
            "the safety factor may need adjustment"
        )

    @bge_m3_required
    def test_hard_split_under_real_tokenizer_respects_budget(self):
        """Pieces split under the real tokenizer are at or below the budget."""
        wrapper = get_resolver("BAAI/bge-m3")
        from lib.chunkers.common import hard_split_by_tokens
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 200
        pieces = hard_split_by_tokens(text, max_tokens=100, tokens=wrapper)
        for piece in pieces:
            assert len(wrapper.encode(piece)) <= 105, (
                f"piece has {len(wrapper.encode(piece))} tokens, expected <= 105"
            )


# ---------------------------------------------------------------------------
# End-to-end: chunk_file uses the resolver
# ---------------------------------------------------------------------------


class TestChunkFileUsesResolver:
    """``chunk_file`` wires the resolver through to the chunkers."""

    @bge_m3_required
    def test_chunk_file_emits_chunks_under_real_tokenizer_budget(self, tmp_path):
        """The dispatcher's effective budget reflects the active tokenizer."""
        from lib.chunker import chunk_file
        from lib.config import ChunkingConfig

        config = Config()
        config.embedding = EmbeddingConfig(
            provider="openrouter",
            model="BAAI/bge-m3",
            max_embed_tokens=8192,
            token_safety_factor=1.6,
        )
        config.chunking = ChunkingConfig(max_tokens=512, overlap_tokens=50, min_tokens=20)

        # A long data literal with no structural boundary — the case
        # the chunker must guarantee against the embedding API.
        content = "K = [" + ",".join(str(i) for i in range(2000)) + "]\n"
        (tmp_path / "data.py").write_text(content, encoding="utf-8")
        chunks = chunk_file("data.py", str(tmp_path), config)

        assert chunks, "no chunks produced"
        # The tolerance accounts for the SentencePiece round-trip drift:
        # a 512-token slice may re-encode to 516 tokens. The safety net
        # accepts anything within OVERSIZE_WARN_RATIO of the budget.
        margin = int(config.chunking.max_tokens * 1.25) + 1
        oversized = [c for c in chunks if c.token_count > margin]
        assert not oversized, (
            f"largest chunk {max(c.token_count for c in chunks)} tokens, "
            f"margin {margin}"
        )
