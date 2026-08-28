"""Rate limiting must not consume the error-retry budget.

A 429 is the server asking for a wait, not a failed request. Counting it
against ``embedding.max_retries`` meant three rate-limit responses in a
row ended a build regardless of connection health — which on a job
embedding tens of thousands of chunks happened routinely.
"""

from unittest.mock import patch

import pytest
import requests

from lib.config import ChunkingConfig, Config
from lib.models import EmbeddingError
from lib.providers.openrouter import (
    MAX_RATE_LIMIT_WAITS,
    OpenRouterProvider,
)


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code: int, payload: dict | None = None,
                 headers: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = ""

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _ok(n: int = 1) -> FakeResponse:
    return FakeResponse(200, {"data": [
        {"index": i, "embedding": [0.0] * 4} for i in range(n)
    ]})


@pytest.fixture
def provider() -> OpenRouterProvider:
    config = Config()
    config.chunking = ChunkingConfig()
    config.embedding.api_key = "test-key"
    config.embedding.max_retries = 3
    config.embedding.retry_delay_seconds = 0.0
    config.embedding.dimensions = None
    return OpenRouterProvider(config)


class TestRateLimitBudget:
    """429 responses get their own allowance."""

    def test_rate_limits_do_not_end_the_request(self, provider) -> None:
        """Two 429s then success — under the old budget this raised."""
        responses = [FakeResponse(429), FakeResponse(429), _ok()]
        with patch("requests.post", side_effect=responses), \
                patch("time.sleep"):
            assert provider.embed_texts(["hello"]) == [[0.0] * 4]

    def test_a_timeout_then_rate_limits_still_succeeds(self, provider) -> None:
        """The exact shape of the failed build: timeout, then two 429s."""
        responses = [
            requests.Timeout("Read timed out"),
            FakeResponse(429),
            FakeResponse(429),
            _ok(),
        ]
        with patch("requests.post", side_effect=responses), \
                patch("time.sleep"):
            assert provider.embed_texts(["hello"]) == [[0.0] * 4]

    def test_more_rate_limits_than_allowed_gives_up(self, provider) -> None:
        responses = [FakeResponse(429)] * (MAX_RATE_LIMIT_WAITS + 2)
        with patch("requests.post", side_effect=responses), \
                patch("time.sleep"):
            with pytest.raises(EmbeddingError) as excinfo:
                provider.embed_texts(["hello"])
        assert "rate limited" in str(excinfo.value).lower()

    def test_retry_after_header_is_honoured(self, provider) -> None:
        responses = [FakeResponse(429, headers={"Retry-After": "7"}), _ok()]
        with patch("requests.post", side_effect=responses), \
                patch("time.sleep") as sleep:
            provider.embed_texts(["hello"])
        assert sleep.call_args_list[0].args[0] == 7.0


class TestErrorBudget:
    """Transport failures still consume max_retries, as before."""

    def test_transport_errors_are_retried_then_raise(self, provider) -> None:
        responses = [requests.Timeout("boom")] * 5
        with patch("requests.post", side_effect=responses) as post, \
                patch("time.sleep"):
            with pytest.raises(EmbeddingError):
                provider.embed_texts(["hello"])
        assert post.call_count == 3, "max_retries should bound transport attempts"

    def test_a_transient_error_recovers(self, provider) -> None:
        responses = [requests.Timeout("boom"), _ok()]
        with patch("requests.post", side_effect=responses), \
                patch("time.sleep"):
            assert provider.embed_texts(["hello"]) == [[0.0] * 4]

    def test_the_error_names_what_actually_stopped_it(self, provider) -> None:
        """A run ending on rate limits must not blame an earlier timeout."""
        responses = [requests.Timeout("Read timed out")] + \
                    [FakeResponse(429)] * (MAX_RATE_LIMIT_WAITS + 1)
        with patch("requests.post", side_effect=responses), \
                patch("time.sleep"):
            with pytest.raises(EmbeddingError) as excinfo:
                provider.embed_texts(["hello"])
        assert "rate limited" in str(excinfo.value).lower()
