"""Tests for provider factory, trust_remote_code security, dimension validation,
and reranker config flow.

Covers review issues:
- Provider selection via create_provider()
- Invalid provider raises EmbeddingError
- trust_remote_code defaults to False and is config-driven
- HuggingFace dimension validation against model output
- Reranker enable/disable flow and trust_remote_code passthrough
- semantic_search wiring: config → Reranker trust flag propagation
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure lib is importable
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib.config import Config, EmbeddingConfig, SearchConfig, load_config
from lib.embedder import create_provider
from lib.models import EmbeddingError


# ---------------------------------------------------------------------------
# Fixtures for mocking optional sentence_transformers dependency
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_sentence_transformers():
    """Provide a mock sentence_transformers module and clean up after test.

    Injects a MagicMock into sys.modules for 'sentence_transformers',
    then removes it AND any cached lib submodules that imported it,
    ensuring no leaked state across tests.
    """
    mock_st = MagicMock()
    # Track which lib modules were absent before so we only clean those
    modules_before = set(sys.modules.keys())

    sys.modules["sentence_transformers"] = mock_st
    yield mock_st

    # Remove the mock module
    sys.modules.pop("sentence_transformers", None)
    # Remove any lib submodules that were imported during the test
    # (they may hold references to the mock)
    stale = [
        k for k in sys.modules
        if k not in modules_before
        and k.startswith(("lib.providers.huggingface", "lib.reranker"))
    ]
    for k in stale:
        sys.modules.pop(k, None)


# ---------------------------------------------------------------------------
# Provider factory tests
# ---------------------------------------------------------------------------

class TestCreateProvider:
    """Tests for the create_provider() factory function."""

    @patch("lib.providers.openrouter.OpenRouterProvider")
    def test_selects_openrouter(self, mock_cls):
        """Factory returns OpenRouterProvider when config says 'openrouter'."""
        config = Config()
        config.embedding = EmbeddingConfig(provider="openrouter", api_key="test-key")
        create_provider(config)
        mock_cls.assert_called_once_with(config)

    @patch("lib.providers.huggingface.HuggingFaceProvider")
    def test_selects_huggingface(self, mock_cls):
        """Factory returns HuggingFaceProvider when config says 'huggingface'."""
        config = Config()
        config.embedding = EmbeddingConfig(provider="huggingface")
        create_provider(config)
        mock_cls.assert_called_once_with(config)

    def test_unknown_provider_raises(self):
        """Unknown provider name raises EmbeddingError."""
        config = Config()
        config.embedding = EmbeddingConfig(provider="ollama")
        with pytest.raises(EmbeddingError, match="Unknown embedding provider"):
            create_provider(config)


# ---------------------------------------------------------------------------
# trust_remote_code config tests
# ---------------------------------------------------------------------------

class TestTrustRemoteCodeConfig:
    """Tests for trust_remote_code config field defaults and loading."""

    def test_default_is_false(self):
        """EmbeddingConfig defaults trust_remote_code to False."""
        cfg = EmbeddingConfig()
        assert cfg.trust_remote_code is False

    def test_loaded_from_json(self, tmp_path):
        """trust_remote_code is read from config.json."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        cfg_data = {
            "schema_version": "1.0",
            "embedding": {"trust_remote_code": True},
        }
        (index_dir / "config.json").write_text(json.dumps(cfg_data))

        config = load_config(str(tmp_path))
        assert config.embedding.trust_remote_code is True

    def test_missing_field_defaults_false(self, tmp_path):
        """Missing trust_remote_code in JSON defaults to False."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        cfg_data = {"schema_version": "1.0", "embedding": {"model": "test-model"}}
        (index_dir / "config.json").write_text(json.dumps(cfg_data))

        config = load_config(str(tmp_path))
        assert config.embedding.trust_remote_code is False


# ---------------------------------------------------------------------------
# HuggingFace dimension validation tests
# ---------------------------------------------------------------------------

class TestHuggingFaceDimensionValidation:
    """Tests for dimension mismatch detection in HuggingFaceProvider."""

    def _make_config(self, dimensions: int = 1024) -> Config:
        config = Config()
        config.embedding = EmbeddingConfig(
            provider="huggingface",
            model="test-model",
            dimensions=dimensions,
            trust_remote_code=False,
        )
        return config

    def test_matching_dimensions_ok(self, mock_sentence_transformers):
        """When model dimensions match config, no error is raised."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_model.device = "cpu"
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        # Fresh import — picks up the mock from sys.modules
        from lib.providers.huggingface import HuggingFaceProvider
        provider = HuggingFaceProvider(self._make_config(1024))
        assert provider.get_dimensions() == 1024

    def test_mismatched_dimensions_raises(self, mock_sentence_transformers):
        """When model dimensions differ from config, EmbeddingError is raised."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 512
        mock_model.device = "cpu"
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        from lib.providers.huggingface import HuggingFaceProvider
        with pytest.raises(EmbeddingError, match="Dimension mismatch"):
            HuggingFaceProvider(self._make_config(1024))

    def test_trust_remote_code_passed_to_model(self, mock_sentence_transformers):
        """trust_remote_code value from config is forwarded to SentenceTransformer."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_model.device = "cpu"
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        from lib.providers.huggingface import HuggingFaceProvider
        HuggingFaceProvider(self._make_config(1024))

        mock_sentence_transformers.SentenceTransformer.assert_called_once_with(
            "test-model",
            trust_remote_code=False,
            device=None,
        )


# ---------------------------------------------------------------------------
# Reranker config flow tests
# ---------------------------------------------------------------------------

class TestRerankerConfigFlow:
    """Tests for reranker enable/disable and trust_remote_code passthrough."""

    def test_rerank_disabled_by_default(self):
        """SearchConfig defaults rerank_enabled to False."""
        cfg = SearchConfig()
        assert cfg.rerank_enabled is False

    def test_reranker_accepts_trust_remote_code(self):
        """Reranker constructor accepts and stores trust_remote_code."""
        from lib.reranker import Reranker

        reranker = Reranker(
            model_name="test-model",
            trust_remote_code=False,
        )
        assert reranker._trust_remote_code is False

        reranker_trusted = Reranker(
            model_name="test-model",
            trust_remote_code=True,
        )
        assert reranker_trusted._trust_remote_code is True

    def test_reranker_default_trust_is_false(self):
        """Reranker defaults trust_remote_code to False."""
        from lib.reranker import Reranker

        reranker = Reranker()
        assert reranker._trust_remote_code is False

    def test_reranker_passes_trust_to_crossencoder(self, mock_sentence_transformers):
        """trust_remote_code is forwarded to CrossEncoder on model load."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9]
        mock_sentence_transformers.CrossEncoder.return_value = mock_model

        from lib.reranker import Reranker

        reranker = Reranker(
            model_name="test-reranker",
            trust_remote_code=False,
        )
        reranker.rerank("query", [{"content": "doc"}], top_n=1)

        mock_sentence_transformers.CrossEncoder.assert_called_once_with(
            "test-reranker",
            device=None,
            trust_remote_code=False,
        )

    def test_reranker_empty_results_noop(self):
        """Reranking empty results returns empty list without loading model."""
        from lib.reranker import Reranker

        reranker = Reranker()
        result = reranker.rerank("query", [], top_n=5)
        assert result == []
        assert reranker._model is None  # Model never loaded


# ---------------------------------------------------------------------------
# semantic_search wiring: config → Reranker trust flag propagation
# ---------------------------------------------------------------------------

class TestSemanticSearchRerankerWiring:
    """Tests that semantic_search passes trust_remote_code from config to Reranker."""

    def test_reranker_constructed_with_config_trust_flag(self, tmp_path):
        """When reranking is enabled, Reranker receives trust_remote_code from config."""
        # Create a minimal project with config and a dummy index
        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        cfg_data = {
            "schema_version": "1.0",
            "embedding": {
                "provider": "openrouter",
                "api_key": "test-key",
                "trust_remote_code": True,
                "device": "cpu",
            },
            "search": {
                "rerank_enabled": True,
                "rerank_model": "test-reranker",
                "rerank_top_n": 5,
                "mode": "vector",
            },
        }
        (index_dir / "config.json").write_text(json.dumps(cfg_data))

        # Mock all heavy dependencies so main() can run without real infra
        mock_store = MagicMock()
        mock_store.has_index.return_value = True
        mock_store.search.return_value = [
            {"id": "c1", "score": 0.9, "file_path": "a.py", "start_line": 1,
             "end_line": 10, "content": "test", "chunk_type": "function",
             "language": "python", "symbol_name": "f", "token_count": 10},
        ]

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 1024

        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = mock_store.search.return_value

        with (
            patch("sys.argv", [
                "semantic_search.py",
                "--project-dir", str(tmp_path),
                "--query", "test query",
            ]),
            patch("lib.store.VectorStore", return_value=mock_store),
            patch("lib.embedder.Embedder", return_value=mock_embedder),
            patch("lib.reranker.Reranker", return_value=mock_reranker_instance) as mock_reranker_cls,
        ):
            # Import and run main — it will sys.exit or print JSON
            import semantic_search
            try:
                semantic_search.main()
            except SystemExit:
                pass  # main() may exit after printing JSON

            mock_reranker_cls.assert_called_once_with(
                model_name="test-reranker",
                device="cpu",
                trust_remote_code=True,
            )


# ---------------------------------------------------------------------------
# Config migration: trust_remote_code coverage
# ---------------------------------------------------------------------------

class TestMigrationTrustRemoteCode:
    """Tests that migrate_config adds trust_remote_code to embedding section."""

    def test_migration_adds_trust_remote_code(self):
        """Config missing trust_remote_code gets a migration entry."""
        from migrate_config import analyze_config

        config = {
            "schema_version": "1.0",
            "embedding": {
                "provider": "openrouter",
                "model": "test-model",
                "device": None,
                # trust_remote_code intentionally missing
            },
            "search": {
                "default_top_k": 10,
                "default_threshold": 0.3,
                "mode": "hybrid",
                "hybrid_alpha": 0.7,
                "rerank_enabled": False,
                "rerank_model": "BAAI/bge-reranker-v2-m3",
                "rerank_top_n": 10,
            },
        }

        migrations = analyze_config(config)
        trust_migrations = [
            m for m in migrations if m["field"] == "embedding.trust_remote_code"
        ]
        assert len(trust_migrations) == 1
        assert trust_migrations[0]["action"] == "add"
        assert trust_migrations[0]["new_value"] is False

    def test_migration_adds_both_device_and_trust(self):
        """Config missing both device and trust_remote_code gets both."""
        from migrate_config import analyze_config, apply_migrations

        config = {
            "schema_version": "1.0",
            "embedding": {"provider": "openrouter"},
            "search": {
                "default_top_k": 10,
                "default_threshold": 0.3,
                "mode": "hybrid",
                "hybrid_alpha": 0.7,
                "rerank_enabled": False,
                "rerank_model": "BAAI/bge-reranker-v2-m3",
                "rerank_top_n": 10,
            },
        }

        migrations = analyze_config(config)
        embedding_fields = [m["field"] for m in migrations if m["field"].startswith("embedding.")]
        assert "embedding.device" in embedding_fields
        assert "embedding.trust_remote_code" in embedding_fields

        updated = apply_migrations(config, migrations)
        assert updated["embedding"]["device"] is None
        assert updated["embedding"]["trust_remote_code"] is False

    def test_migration_skips_when_trust_present(self):
        """Config that already has trust_remote_code gets no migration for it."""
        from migrate_config import analyze_config

        config = {
            "schema_version": "1.0",
            "embedding": {
                "provider": "openrouter",
                "device": None,
                "trust_remote_code": True,
            },
            "search": {
                "default_top_k": 10,
                "default_threshold": 0.3,
                "mode": "hybrid",
                "hybrid_alpha": 0.7,
                "rerank_enabled": False,
                "rerank_model": "BAAI/bge-reranker-v2-m3",
                "rerank_top_n": 10,
            },
        }

        migrations = analyze_config(config)
        trust_migrations = [
            m for m in migrations if m["field"] == "embedding.trust_remote_code"
        ]
        assert len(trust_migrations) == 0


# ---------------------------------------------------------------------------
# trust_remote_code warning emission tests
# ---------------------------------------------------------------------------

class TestTrustRemoteCodeWarning:
    """Tests that enabling trust_remote_code emits an explicit warning."""

    def test_trust_true_emits_warning(self, mock_sentence_transformers):
        """When trust_remote_code=True, a warning is logged."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_model.device = "cpu"
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        config = Config()
        config.embedding = EmbeddingConfig(
            provider="huggingface",
            model="test-model",
            dimensions=1024,
            trust_remote_code=True,
        )

        with patch("lib.providers.huggingface.logger") as mock_logger:
            from lib.providers.huggingface import HuggingFaceProvider
            HuggingFaceProvider(config)

            warning_calls = [
                call for call in mock_logger.warning.call_args_list
                if "trust_remote_code" in str(call)
            ]
            assert len(warning_calls) == 1

    def test_trust_false_no_warning(self, mock_sentence_transformers):
        """When trust_remote_code=False, no trust warning is logged."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_model.device = "cpu"
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        config = Config()
        config.embedding = EmbeddingConfig(
            provider="huggingface",
            model="test-model",
            dimensions=1024,
            trust_remote_code=False,
        )

        with patch("lib.providers.huggingface.logger") as mock_logger:
            from lib.providers.huggingface import HuggingFaceProvider
            HuggingFaceProvider(config)

            warning_calls = [
                call for call in mock_logger.warning.call_args_list
                if "trust_remote_code" in str(call)
            ]
            assert len(warning_calls) == 0


# ---------------------------------------------------------------------------
# OpenRouter Retry-After header edge cases
# ---------------------------------------------------------------------------

class TestOpenRouterRetryAfter:
    """Tests for Retry-After header parsing in OpenRouterProvider."""

    def _make_provider(self):
        """Create an OpenRouterProvider with test config."""
        config = Config()
        config.embedding = EmbeddingConfig(
            provider="openrouter",
            api_key="test-key",
            max_retries=2,
            retry_delay_seconds=0.01,
        )
        from lib.providers.openrouter import OpenRouterProvider
        return OpenRouterProvider(config)

    @patch("lib.providers.openrouter.requests.post")
    def test_numeric_retry_after_is_used(self, mock_post):
        """Numeric Retry-After header is parsed and used."""
        # First call: 429 with numeric Retry-After
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "2"}

        # Second call: success
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.raise_for_status = MagicMock()
        resp_ok.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
        }

        mock_post.side_effect = [resp_429, resp_ok]

        provider = self._make_provider()
        with patch("lib.providers.openrouter.time.sleep") as mock_sleep:
            result = provider.embed_texts(["test"])
            # Should have slept with the parsed Retry-After value
            mock_sleep.assert_called_with(2.0)
        assert result == [[0.1, 0.2]]

    @patch("lib.providers.openrouter.requests.post")
    def test_http_date_retry_after_falls_back(self, mock_post):
        """HTTP-date Retry-After falls back to exponential backoff."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "Sat, 21 Mar 2026 12:00:00 GMT"}

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.raise_for_status = MagicMock()
        resp_ok.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
        }

        mock_post.side_effect = [resp_429, resp_ok]

        provider = self._make_provider()
        with patch("lib.providers.openrouter.time.sleep") as mock_sleep:
            result = provider.embed_texts(["test"])
            # Should fall back to retry_delay * 2^0 = 0.01
            mock_sleep.assert_called_with(0.01)
        assert result == [[0.1, 0.2]]

    @patch("lib.providers.openrouter.requests.post")
    def test_missing_retry_after_uses_backoff(self, mock_post):
        """Missing Retry-After header uses exponential backoff."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {}

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.raise_for_status = MagicMock()
        resp_ok.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
        }

        mock_post.side_effect = [resp_429, resp_ok]

        provider = self._make_provider()
        with patch("lib.providers.openrouter.time.sleep") as mock_sleep:
            result = provider.embed_texts(["test"])
            mock_sleep.assert_called_with(0.01)
        assert result == [[0.1, 0.2]]

    @patch("lib.providers.openrouter.requests.post")
    def test_negative_retry_after_falls_back(self, mock_post):
        """Negative Retry-After value falls back to exponential backoff."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "-5"}

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.raise_for_status = MagicMock()
        resp_ok.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
        }

        mock_post.side_effect = [resp_429, resp_ok]

        provider = self._make_provider()
        with patch("lib.providers.openrouter.time.sleep") as mock_sleep:
            result = provider.embed_texts(["test"])
            mock_sleep.assert_called_with(0.01)
        assert result == [[0.1, 0.2]]

    @patch("lib.providers.openrouter.requests.post")
    def test_inf_retry_after_falls_back(self, mock_post):
        """Inf Retry-After value falls back to exponential backoff."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "inf"}

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.raise_for_status = MagicMock()
        resp_ok.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
        }

        mock_post.side_effect = [resp_429, resp_ok]

        provider = self._make_provider()
        with patch("lib.providers.openrouter.time.sleep") as mock_sleep:
            result = provider.embed_texts(["test"])
            mock_sleep.assert_called_with(0.01)
        assert result == [[0.1, 0.2]]

    @patch("lib.providers.openrouter.requests.post")
    def test_nan_retry_after_falls_back(self, mock_post):
        """NaN Retry-After value falls back to exponential backoff."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "nan"}

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.raise_for_status = MagicMock()
        resp_ok.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
        }

        mock_post.side_effect = [resp_429, resp_ok]

        provider = self._make_provider()
        with patch("lib.providers.openrouter.time.sleep") as mock_sleep:
            result = provider.embed_texts(["test"])
            mock_sleep.assert_called_with(0.01)
        assert result == [[0.1, 0.2]]


# ---------------------------------------------------------------------------
# Reranker: missing content handling
# ---------------------------------------------------------------------------

class TestRerankerMissingContent:
    """Tests for reranker behavior when results are missing content."""

    def test_missing_content_skipped(self, mock_sentence_transformers):
        """Results without 'content' are skipped, not crashed on."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9]
        mock_sentence_transformers.CrossEncoder.return_value = mock_model

        from lib.reranker import Reranker

        reranker = Reranker(model_name="test-reranker")
        results = [
            {"id": "c1", "content": "valid doc"},
            {"id": "c2"},  # missing content
            {"id": "c3", "content": ""},  # empty content
        ]
        reranked = reranker.rerank("query", results, top_n=5)

        # Only the result with actual content should be scored
        assert len(reranked) == 1
        assert reranked[0]["id"] == "c1"
        mock_model.predict.assert_called_once_with([("query", "valid doc")])

    def test_all_missing_content_returns_empty(self, mock_sentence_transformers):
        """If all results lack content, returns empty without loading model."""
        from lib.reranker import Reranker

        reranker = Reranker(model_name="test-reranker")
        results = [{"id": "c1"}, {"id": "c2", "content": ""}]
        reranked = reranker.rerank("query", results, top_n=5)
        assert reranked == []
        assert reranker._model is None  # Model never loaded
        mock_sentence_transformers.CrossEncoder.assert_not_called()


# ---------------------------------------------------------------------------
# SEMANTIC_INDEX_PROVIDER env var override
# ---------------------------------------------------------------------------

class TestProviderEnvOverride:
    """Tests for SEMANTIC_INDEX_PROVIDER environment variable override."""

    def test_env_overrides_config_provider(self, tmp_path):
        """SEMANTIC_INDEX_PROVIDER env var overrides config file provider."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        cfg_data = {
            "schema_version": "1.0",
            "embedding": {"provider": "openrouter"},
        }
        (index_dir / "config.json").write_text(json.dumps(cfg_data))

        with patch.dict(os.environ, {"SEMANTIC_INDEX_PROVIDER": "huggingface"}):
            config = load_config(str(tmp_path))
            assert config.embedding.provider == "huggingface"

    def test_invalid_provider_env_raises(self, tmp_path):
        """Invalid SEMANTIC_INDEX_PROVIDER raises ConfigError at validation."""
        from lib.models import ConfigError as CfgErr

        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        cfg_data = {"schema_version": "1.0", "embedding": {"provider": "openrouter"}}
        (index_dir / "config.json").write_text(json.dumps(cfg_data))

        with patch.dict(os.environ, {"SEMANTIC_INDEX_PROVIDER": "invalid_provider"}):
            with pytest.raises(CfgErr, match="Invalid embedding.provider"):
                load_config(str(tmp_path))


# ---------------------------------------------------------------------------
# Backward compatibility: create_embedder deprecation shim
# ---------------------------------------------------------------------------

class TestCreateEmbedderDeprecation:
    """Tests that the deprecated create_embedder() shim still works."""

    @patch("lib.providers.openrouter.OpenRouterProvider")
    def test_create_embedder_dispatches_correctly(self, mock_cls):
        """create_embedder() delegates to create_provider() and returns a provider."""
        import warnings
        from lib.embedder import create_embedder

        config = Config()
        config.embedding = EmbeddingConfig(provider="openrouter", api_key="test-key")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_embedder(config)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "create_provider" in str(w[0].message)

        mock_cls.assert_called_once_with(config)
