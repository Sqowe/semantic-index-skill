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



# ---------------------------------------------------------------------------
# Progressive truncation retry tests (OpenRouter)
# ---------------------------------------------------------------------------


class TestProgressiveTruncation:
    """Tests for progressive truncation on single-text token limit errors."""

    def _make_provider_with_limit(self, max_chars: int):
        config = Config()
        config.embedding = EmbeddingConfig(
            provider="openrouter",
            api_key="test-key",
            max_retries=1,
            retry_delay_seconds=0.01,
            max_embed_chars=max_chars,
        )
        from lib.providers.openrouter import OpenRouterProvider

        return OpenRouterProvider(config)

    @patch("lib.providers.openrouter.requests.post")
    def test_progressive_truncation_succeeds(self, mock_post):
        """Single text that fails at full length succeeds after truncation."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            body = kwargs.get("json", {})
            text_len = len(body["input"][0]) if body.get("input") else 0
            # Fail if text > 500 chars, succeed otherwise
            if text_len > 500:
                resp = MagicMock()
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {
                    "error": {"message": "maximum context length is 8192 tokens", "code": 400}
                }
                return resp
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
            return resp

        mock_post.side_effect = side_effect
        provider = self._make_provider_with_limit(1000)
        result = provider.embed_texts(["x" * 1000])

        assert result == [[0.1, 0.2]]
        assert call_count[0] > 1  # Had to retry

    @patch("lib.providers.openrouter.requests.post")
    def test_non_length_error_not_retried(self, mock_post):
        """Non-context-length RuntimeError is raised immediately, no truncation."""
        # Return a non-length error wrapped in 200
        mock_post.return_value = _mock_response(
            200,
            {"error": {"message": "Model not available", "code": 503}},
        )

        provider = self._make_provider_with_limit(1000)
        with pytest.raises(EmbeddingError, match="no 'data' field"):
            provider.embed_texts(["x" * 500])

        # Should only be called once — no retries
        assert mock_post.call_count == 1

    @patch("lib.providers.openrouter.requests.post")
    def test_multi_text_length_error_reraises_for_splitting(self, mock_post):
        """Multi-text batch with length error re-raises for batch splitting."""
        mock_post.return_value = _mock_response(
            200,
            {"error": {"message": "maximum context length exceeded", "code": 400}},
        )

        provider = self._make_provider_with_limit(1000)
        with pytest.raises(RuntimeError, match="context length"):
            provider.embed_texts(["text1", "text2"])

    @patch("lib.providers.openrouter.requests.post")
    def test_non_length_runtime_error_during_retry_raises(self, mock_post):
        """If a non-length error occurs during truncation retry, it raises."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if call_count[0] == 1:
                # First call: context length error
                resp.json.return_value = {
                    "error": {"message": "maximum context length is 8192", "code": 400}
                }
            else:
                # Retry: different error (not length-related)
                resp.json.return_value = {
                    "error": {"message": "Service unavailable", "code": 503}
                }
            return resp

        mock_post.side_effect = side_effect
        provider = self._make_provider_with_limit(1000)

        # Should raise EmbeddingError (from the non-length error on retry)
        with pytest.raises(EmbeddingError, match="no 'data' field"):
            provider.embed_texts(["x" * 500])


    @patch("lib.providers.openrouter.requests.post")
    def test_http_400_context_length_triggers_progressive_truncation(self, mock_post):
        """Direct HTTP 400 context-length error triggers progressive truncation."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            body = kwargs.get("json", {})
            text_len = len(body["input"][0]) if body.get("input") else 0
            resp = MagicMock()
            resp.headers = {}
            if text_len > 500:
                # Return 400 context-length error
                resp.status_code = 400
                resp.json.return_value = {
                    "error": {
                        "message": "This model's maximum context length is 8192 tokens.",
                        "code": 400,
                    }
                }
            else:
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {"data": [{"index": 0, "embedding": [0.5]}]}
            return resp

        mock_post.side_effect = side_effect
        provider = self._make_provider_with_limit(1000)
        result = provider.embed_texts(["y" * 1000])

        assert result == [[0.5]]
        assert call_count[0] > 1


# ---------------------------------------------------------------------------
# HuggingFace progressive truncation tests
# ---------------------------------------------------------------------------


class TestHuggingFaceProgressiveTruncation:
    """Tests for HuggingFace provider progressive truncation on OOM."""

    def _make_provider(self, max_chars: int = 1000):
        """Build a HuggingFaceProvider with mocked model."""
        from lib.providers.huggingface import HuggingFaceProvider

        config = Config()
        config.embedding = EmbeddingConfig(
            provider="huggingface",
            model="test-model",
            dimensions=3,
            batch_size=1,
            max_embed_chars=max_chars,
            trust_remote_code=False,
        )

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_model.device = "cpu"

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            with patch.object(HuggingFaceProvider, "__init__", lambda self, cfg: None):
                provider = HuggingFaceProvider.__new__(HuggingFaceProvider)

        provider._model = mock_model
        provider._batch_size = config.embedding.batch_size
        provider._doc_prefix = config.embedding.document_prefix
        provider._query_prefix = config.embedding.query_prefix
        provider._dimensions = config.embedding.dimensions
        provider._max_embed_chars = config.embedding.max_embed_chars
        return provider

    def test_single_text_oom_succeeds_after_truncation(self):
        """Single text OOM succeeds after progressive truncation."""
        import numpy as np

        provider = self._make_provider(max_chars=1000)
        call_count = [0]

        def mock_encode(texts, batch_size=32, **kwargs):
            call_count[0] += 1
            text_len = len(texts[0])
            if text_len > 500:
                raise RuntimeError("Invalid buffer size: 16.03 GiB")
            return np.array([[0.1, 0.2, 0.3]])

        provider._model.encode = mock_encode
        result = provider.embed_texts(["x" * 1000])

        assert result == [[0.1, 0.2, 0.3]]
        assert call_count[0] > 1

    def test_non_oom_error_during_retry_raises(self):
        """Non-OOM RuntimeError during truncation retry raises immediately."""
        provider = self._make_provider(max_chars=1000)
        call_count = [0]

        def mock_encode(texts, batch_size=32, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Invalid buffer size: 16.03 GiB")
            # Second call: non-OOM error
            raise RuntimeError("CUDA device not available")

        provider._model.encode = mock_encode

        with pytest.raises(RuntimeError, match="CUDA device not available"):
            provider.embed_texts(["x" * 1000])

    def test_truncation_exhaustion_reraises(self):
        """If truncation reaches minimum and still OOMs, re-raises."""
        provider = self._make_provider(max_chars=200)

        def mock_encode(texts, batch_size=32, **kwargs):
            raise RuntimeError("Invalid buffer size: 16.03 GiB")

        provider._model.encode = mock_encode

        with pytest.raises(RuntimeError, match="Invalid buffer size"):
            provider.embed_texts(["x" * 200])
