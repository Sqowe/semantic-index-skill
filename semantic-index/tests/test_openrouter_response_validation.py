"""Tests for OpenRouter provider response validation.

Covers:
- Successful response sorted by index
- Response JSON missing 'data' key raises EmbeddingError
- Non-dict JSON payload (list, string) raises EmbeddingError
"""

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
