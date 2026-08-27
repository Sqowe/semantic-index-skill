"""The last-resort splitter must return the source text unchanged.

``decode(encode(x))`` does not round-trip for a SentencePiece-style
tokenizer such as BAAI/bge-m3: whitespace runs collapse, newlines become
spaces, and word boundaries can disappear. Rebuilding chunks that way put
text in the index that did not match the file and flattened every piece
onto a single line number.
"""

import pytest

from lib.chunkers.common import count_tokens, hard_split_by_tokens
from lib.config import Config
from lib.tokenizer_resolver import get_resolver, resolver_for_config

# The rounding margin lib/chunker.py already allows for accumulating chunkers.
OVERSHOOT_MARGIN = 1.25

WHITESPACE_HEAVY = "line one\nline two   with   spaces\n\nline four\n\ttabbed\n"

CODE = (
    "namespace Telemetry {\n\n"
    "class SessionStore {\n"
    "  public:\n"
    "    void insert(const Entry& e) {\n"
    "        entries_.push_back(e);\n"
    "    }\n"
    "};\n\n"
    "}  // namespace Telemetry\n"
) * 12


@pytest.fixture(params=["real", "tiktoken"])
def tokenizer(request):
    """Both tokenizer kinds, skipping 'real' when it cannot be loaded."""
    if request.param == "tiktoken":
        return get_resolver("cl100k_base")
    wrapper = resolver_for_config(Config())
    if wrapper.kind != "real":
        pytest.skip("real tokenizer unavailable (offline or tokenizers missing)")
    return wrapper


class TestTextIsPreserved:
    """The pieces must concatenate back to exactly the input."""

    @pytest.mark.parametrize("text", [
        WHITESPACE_HEAVY,
        CODE,
        "日本語のテキスト\nと英語 mixed\n\n" * 40,
        "x" * 5000,
        "a\n\n\n\n\nb\n\t\t\tc\n",
    ])
    @pytest.mark.parametrize("budget", [60, 320, 512])
    def test_pieces_rejoin_to_the_source(self, tokenizer, text, budget) -> None:
        pieces = hard_split_by_tokens(text, budget, tokens=tokenizer)
        assert "".join(pieces) == text

    def test_newlines_survive(self, tokenizer) -> None:
        """Line numbers downstream are counted from these newlines."""
        text = WHITESPACE_HEAVY * 40
        pieces = hard_split_by_tokens(text, 60, tokens=tokenizer)
        assert sum(p.count("\n") for p in pieces) == text.count("\n")

    def test_a_short_text_is_returned_untouched(self, tokenizer) -> None:
        assert hard_split_by_tokens(WHITESPACE_HEAVY, 4096, tokens=tokenizer) == [
            WHITESPACE_HEAVY
        ]


class TestPiecesRespectTheBudget:
    """Sizes stay within the margin the dispatch layer tolerates."""

    @pytest.mark.parametrize("budget", [60, 320, 512])
    def test_no_piece_runs_far_over(self, tokenizer, budget) -> None:
        pieces = hard_split_by_tokens(CODE, budget, tokens=tokenizer)
        for piece in pieces:
            assert count_tokens(piece, tokens=tokenizer) <= budget * OVERSHOOT_MARGIN

    def test_splitting_makes_progress(self, tokenizer) -> None:
        """Every piece must be non-empty, or the loop would not terminate."""
        pieces = hard_split_by_tokens(CODE, 60, tokens=tokenizer)
        assert len(pieces) > 1
        assert all(pieces)


class TestOffsetSupport:
    """The wrapper reports offsets only where they are meaningful."""

    def test_tiktoken_reports_none(self) -> None:
        assert get_resolver("cl100k_base").offsets("some text") is None

    def test_real_tokenizer_reports_spans(self) -> None:
        wrapper = resolver_for_config(Config())
        if wrapper.kind != "real":
            pytest.skip("real tokenizer unavailable")
        offsets = wrapper.offsets(CODE)
        assert offsets is not None
        assert len(offsets) == len(wrapper.encode(CODE))
        assert all(0 <= s <= e <= len(CODE) for s, e in offsets)
