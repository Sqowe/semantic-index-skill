"""Tests for OpenRouter provider response validation.

Covers:
- Successful response sorted by index
- Response JSON missing 'data' key raises EmbeddingError
- Non-dict JSON payload (list, string) raises EmbeddingError
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure lib is importable
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib.config import Config, EmbeddingConfig
from lib.models import EmbeddingError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider():
    """Create an OpenRouterProvider with test config."""
    config = Config()
    config.embedding = EmbeddingConfig(
        provider="openrouter",
        api_key="test-key",
        max_retries=1,
        retry_delay_seconds=0.01,
    )
    from lib.providers.openrouter import OpenRouterProvider

    return OpenRouterProvider(config)


def _mock_response(status_code: int, json_body):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json.return_value = json_body
    resp.headers = {}
    return resp


# ---------------------------------------------------------------------------
# Successful response tests
# ---------------------------------------------------------------------------


class TestSuccessfulResponse:
    """Tests for valid API responses."""

    @patch("lib.providers.openrouter.requests.post")
    def test_embeddings_sorted_by_index(self, mock_post):
        """Embeddings are returned in index order regardless of API order."""
        mock_post.return_value = _mock_response(
            200,
            {
                "data": [
                    {"index": 2, "embedding": [0.3, 0.3]},
                    {"index": 0, "embedding": [0.1, 0.1]},
                    {"index": 1, "embedding": [0.2, 0.2]},
                ],
            },
        )

        provider = _make_provider()
        result = provider.embed_texts(["a", "b", "c"])

        assert result == [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]

    @patch("lib.providers.openrouter.requests.post")
    def test_single_embedding(self, mock_post):
        """Single text embedding works correctly."""
        mock_post.return_value = _mock_response(
            200,
            {"data": [{"index": 0, "embedding": [0.5, 0.6, 0.7]}]},
        )

        provider = _make_provider()
        result = provider.embed_query("test query")

        assert result == [0.5, 0.6, 0.7]


# ---------------------------------------------------------------------------
# Missing 'data' key tests
# ---------------------------------------------------------------------------


class TestMissingDataKey:
    """Tests for API responses missing the 'data' field."""

    @patch("lib.providers.openrouter.requests.post")
    def test_error_object_response(self, mock_post):
        """API returning error object (no 'data') raises EmbeddingError."""
        mock_post.return_value = _mock_response(
            200,
            {"error": {"message": "Quota exceeded", "code": 429}},
        )

        provider = _make_provider()
        with pytest.raises(EmbeddingError, match="no 'data' field"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_empty_dict_response(self, mock_post):
        """API returning empty dict raises EmbeddingError."""
        mock_post.return_value = _mock_response(200, {})

        provider = _make_provider()
        with pytest.raises(EmbeddingError, match="no 'data' field"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_error_message_included_in_exception(self, mock_post):
        """The error payload from the API is included in the exception message."""
        mock_post.return_value = _mock_response(
            200,
            {"error": {"message": "Model not found", "code": 404}},
        )

        provider = _make_provider()
        with pytest.raises(EmbeddingError, match="Model not found"):
            provider.embed_texts(["test"])


# ---------------------------------------------------------------------------
# Non-dict JSON payload tests
# ---------------------------------------------------------------------------


class TestNonDictResponse:
    """Tests for API responses that are not dicts (list, string, etc.)."""

    @patch("lib.providers.openrouter.requests.post")
    def test_list_response(self, mock_post):
        """API returning a JSON list raises EmbeddingError."""
        mock_post.return_value = _mock_response(200, [1, 2, 3])

        provider = _make_provider()
        with pytest.raises(EmbeddingError, match="Unexpected API response type"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_string_response(self, mock_post):
        """API returning a JSON string raises EmbeddingError."""
        mock_post.return_value = _mock_response(200, "Internal Server Error")

        provider = _make_provider()
        with pytest.raises(EmbeddingError, match="Unexpected API response type"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_null_response(self, mock_post):
        """API returning JSON null raises EmbeddingError."""
        mock_post.return_value = _mock_response(200, None)

        provider = _make_provider()
        with pytest.raises(EmbeddingError, match="Unexpected API response type"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_nested_list_response(self, mock_post):
        """API returning a nested list (e.g. raw embeddings) raises EmbeddingError."""
        mock_post.return_value = _mock_response(200, [[0.1, 0.2], [0.3, 0.4]])

        provider = _make_provider()
        with pytest.raises(EmbeddingError, match="Unexpected API response type"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_type_name_in_error_message(self, mock_post):
        """Error message includes the actual type name for debugging."""
        mock_post.return_value = _mock_response(200, [1, 2, 3])

        provider = _make_provider()
        with pytest.raises(EmbeddingError, match="list"):
            provider.embed_texts(["test"])



# ---------------------------------------------------------------------------
# Context length error detection (triggers batch splitting)
# ---------------------------------------------------------------------------


class TestContextLengthDetection:
    """Tests for context/input length errors that trigger batch splitting."""

    @patch("lib.providers.openrouter.requests.post")
    def test_400_context_length_raises_runtime_error(self, mock_post):
        """HTTP 400 with 'context length' triggers RuntimeError for splitting."""
        resp = MagicMock()
        resp.status_code = 400
        resp.json.return_value = {
            "error": {
                "message": "This model's maximum context length is 8192 tokens.",
                "code": 400,
            }
        }
        resp.headers = {}
        mock_post.return_value = resp

        provider = _make_provider()
        with pytest.raises(RuntimeError, match="context length exceeded"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_422_input_sequence_raises_runtime_error(self, mock_post):
        """HTTP 422 with 'input sequence' triggers RuntimeError for splitting."""
        resp = MagicMock()
        resp.status_code = 422
        resp.json.return_value = {
            "detail": [{
                "msg": "Value error, The input sequence should have less than 131072 characters.",
            }]
        }
        resp.headers = {}
        mock_post.return_value = resp

        provider = _make_provider()
        with pytest.raises(RuntimeError, match="context length exceeded"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_200_wrapped_context_length_raises_runtime_error(self, mock_post):
        """HTTP 200 wrapping a context length error triggers RuntimeError."""
        mock_post.return_value = _mock_response(
            200,
            {
                "error": {
                    "message": "HTTP 400: {\"error\":{\"message\":\"This model's "
                    "maximum context length is 8192 tokens.\",\"code\":400}}",
                    "code": 400,
                }
            },
        )

        provider = _make_provider()
        with pytest.raises(RuntimeError, match="context length exceeded"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_200_wrapped_input_length_raises_runtime_error(self, mock_post):
        """HTTP 200 wrapping an input length error triggers RuntimeError."""
        mock_post.return_value = _mock_response(
            200,
            {
                "error": {
                    "message": "HTTP 422: Input length: 177508 exceeds limit",
                    "code": 422,
                }
            },
        )

        provider = _make_provider()
        with pytest.raises(RuntimeError, match="context length exceeded"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_413_payload_too_large_raises_runtime_error(self, mock_post):
        """HTTP 413 with 'payload too large' triggers RuntimeError."""
        resp = MagicMock()
        resp.status_code = 413
        resp.json.return_value = {"error": "Payload too large"}
        resp.headers = {}
        mock_post.return_value = resp

        provider = _make_provider()
        with pytest.raises(RuntimeError, match="context length exceeded"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_413_empty_body_raises_runtime_error(self, mock_post):
        """HTTP 413 with empty/non-JSON body still triggers RuntimeError."""
        resp = MagicMock()
        resp.status_code = 413
        resp.json.side_effect = ValueError("No JSON")
        resp.text = ""
        resp.headers = {}
        mock_post.return_value = resp

        provider = _make_provider()
        with pytest.raises(RuntimeError, match="context length exceeded"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_413_html_body_raises_runtime_error(self, mock_post):
        """HTTP 413 with HTML body (nginx default) still triggers RuntimeError."""
        resp = MagicMock()
        resp.status_code = 413
        resp.json.side_effect = ValueError("No JSON")
        resp.text = "<html><body><h1>413 Request Entity Too Large</h1></body></html>"
        resp.headers = {}
        mock_post.return_value = resp

        provider = _make_provider()
        with pytest.raises(RuntimeError, match="context length exceeded"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_200_wrapped_payload_too_large_raises_runtime_error(self, mock_post):
        """HTTP 200 wrapping 'payload too large' triggers RuntimeError."""
        mock_post.return_value = _mock_response(
            200,
            {
                "error": {
                    "message": "HTTP 413: Payload too large",
                    "code": 413,
                }
            },
        )

        provider = _make_provider()
        with pytest.raises(RuntimeError, match="context length exceeded"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_200_wrapped_request_entity_too_large_raises_runtime_error(self, mock_post):
        """HTTP 200 wrapping 'request entity too large' triggers RuntimeError."""
        mock_post.return_value = _mock_response(
            200,
            {
                "error": {
                    "message": "Request Entity Too Large",
                    "code": 413,
                }
            },
        )

        provider = _make_provider()
        with pytest.raises(RuntimeError, match="context length exceeded"):
            provider.embed_texts(["test"])

    @patch("lib.providers.openrouter.requests.post")
    def test_200_non_length_error_raises_embedding_error(self, mock_post):
        """HTTP 200 with non-length error still raises EmbeddingError."""
        mock_post.return_value = _mock_response(
            200,
            {"error": {"message": "Model not available", "code": 503}},
        )

        provider = _make_provider()
        with pytest.raises(EmbeddingError, match="no 'data' field"):
            provider.embed_texts(["test"])



# ---------------------------------------------------------------------------
# max_embed_chars truncation tests (OpenRouter)
# ---------------------------------------------------------------------------


class TestOpenRouterTruncation:
    """Tests for max_embed_chars truncation in OpenRouterProvider."""

    def _make_provider_with_limit(self, max_chars: int):
        """Create an OpenRouterProvider with a specific max_embed_chars."""
        config = Config()
        config.embedding = EmbeddingConfig(
            provider="openrouter",
            api_key="test-key",
            max_retries=1,
            retry_delay_seconds=0.01,
            max_embed_chars=max_chars,
            document_prefix="doc: ",
            query_prefix="query: ",
        )
        from lib.providers.openrouter import OpenRouterProvider

        return OpenRouterProvider(config)

    @patch("lib.providers.openrouter.requests.post")
    def test_text_under_limit_not_truncated(self, mock_post):
        """Text under max_embed_chars is sent as-is (with prefix)."""
        mock_post.return_value = _mock_response(
            200, {"data": [{"index": 0, "embedding": [0.1]}]}
        )
        provider = self._make_provider_with_limit(100)
        provider.embed_texts(["short"])

        sent_body = mock_post.call_args[1]["json"]
        assert sent_body["input"] == ["doc: short"]

    @patch("lib.providers.openrouter.requests.post")
    def test_text_at_exact_limit_not_truncated(self, mock_post):
        """Text exactly at max_embed_chars boundary is not truncated."""
        mock_post.return_value = _mock_response(
            200, {"data": [{"index": 0, "embedding": [0.1]}]}
        )
        # prefix "doc: " is 5 chars, so text of 95 chars = 100 total
        text = "x" * 95
        provider = self._make_provider_with_limit(100)
        provider.embed_texts([text])

        sent_body = mock_post.call_args[1]["json"]
        assert sent_body["input"] == [f"doc: {text}"]
        assert len(sent_body["input"][0]) == 100

    @patch("lib.providers.openrouter.requests.post")
    def test_text_over_limit_truncated(self, mock_post):
        """Text exceeding max_embed_chars is truncated to the limit."""
        mock_post.return_value = _mock_response(
            200, {"data": [{"index": 0, "embedding": [0.1]}]}
        )
        # prefix "doc: " is 5 chars, text of 100 chars = 105 total > 100 limit
        text = "x" * 100
        provider = self._make_provider_with_limit(100)
        provider.embed_texts([text])

        sent_body = mock_post.call_args[1]["json"]
        assert len(sent_body["input"][0]) == 100  # truncated to limit

    @patch("lib.providers.openrouter.requests.post")
    def test_prefix_included_in_truncation_length(self, mock_post):
        """Truncation limit includes the prefix length."""
        mock_post.return_value = _mock_response(
            200, {"data": [{"index": 0, "embedding": [0.1]}]}
        )
        # prefix "doc: " = 5 chars, limit = 10, so only 10 chars total
        text = "abcdefghij"  # 10 chars + 5 prefix = 15 > 10
        provider = self._make_provider_with_limit(10)
        provider.embed_texts([text])

        sent_body = mock_post.call_args[1]["json"]
        assert sent_body["input"][0] == "doc: abcde"  # 10 chars total
        assert len(sent_body["input"][0]) == 10

    @patch("lib.providers.openrouter.requests.post")
    def test_query_over_limit_truncated(self, mock_post):
        """Query exceeding max_embed_chars is truncated."""
        mock_post.return_value = _mock_response(
            200, {"data": [{"index": 0, "embedding": [0.1]}]}
        )
        query = "y" * 100  # + "query: " prefix = 107 > 100
        provider = self._make_provider_with_limit(100)
        provider.embed_query(query)

        sent_body = mock_post.call_args[1]["json"]
        assert len(sent_body["input"][0]) == 100

    @patch("lib.providers.openrouter.requests.post")
    def test_batch_partial_truncation(self, mock_post):
        """Only texts over the limit are truncated; others are untouched."""
        mock_post.return_value = _mock_response(
            200,
            {
                "data": [
                    {"index": 0, "embedding": [0.1]},
                    {"index": 1, "embedding": [0.2]},
                ]
            },
        )
        short_text = "hi"  # "doc: hi" = 5 chars < 20
        long_text = "z" * 50  # "doc: " + 50 = 55 > 20
        provider = self._make_provider_with_limit(20)
        provider.embed_texts([short_text, long_text])

        sent_body = mock_post.call_args[1]["json"]
        assert sent_body["input"][0] == "doc: hi"  # not truncated
        assert len(sent_body["input"][1]) == 20  # truncated


# ---------------------------------------------------------------------------
# max_embed_chars config validation tests
# ---------------------------------------------------------------------------


class TestMaxEmbedCharsValidation:
    """Tests for max_embed_chars config validation."""

    def test_zero_raises_config_error(self, tmp_path):
        """max_embed_chars=0 raises ConfigError."""
        from lib.models import ConfigError as CfgErr

        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        cfg_data = {
            "schema_version": "1.0",
            "embedding": {"max_embed_chars": 0},
        }
        (index_dir / "config.json").write_text(json.dumps(cfg_data))

        from lib.config import load_config

        with pytest.raises(CfgErr, match="max_embed_chars"):
            load_config(str(tmp_path))

    def test_negative_raises_config_error(self, tmp_path):
        """max_embed_chars=-1 raises ConfigError."""
        from lib.models import ConfigError as CfgErr

        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        cfg_data = {
            "schema_version": "1.0",
            "embedding": {"max_embed_chars": -100},
        }
        (index_dir / "config.json").write_text(json.dumps(cfg_data))

        from lib.config import load_config

        with pytest.raises(CfgErr, match="max_embed_chars"):
            load_config(str(tmp_path))

    def test_positive_value_passes(self, tmp_path):
        """max_embed_chars=1000 passes validation."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        cfg_data = {
            "schema_version": "1.0",
            "embedding": {"max_embed_chars": 1000},
        }
        (index_dir / "config.json").write_text(json.dumps(cfg_data))

        from lib.config import load_config

        config = load_config(str(tmp_path))
        assert config.embedding.max_embed_chars == 1000

    def test_missing_uses_default(self, tmp_path):
        """Missing max_embed_chars falls back to default 30000."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        cfg_data = {
            "schema_version": "1.0",
            "embedding": {"provider": "openrouter"},
        }
        (index_dir / "config.json").write_text(json.dumps(cfg_data))

        from lib.config import load_config

        config = load_config(str(tmp_path))
        assert config.embedding.max_embed_chars == 20000
