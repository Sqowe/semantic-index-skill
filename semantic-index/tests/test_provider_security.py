"""Tests for provider factory, trust_remote_code security, dimension validation,
and reranker config flow.

Covers review issues:
- Provider selection via create_embedder()
- Invalid provider raises EmbeddingError
- trust_remote_code defaults to False and is config-driven
- HuggingFace dimension validation against model output
- Reranker enable/disable flow and trust_remote_code passthrough
- semantic_search wiring: config → Reranker trust flag propagation
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

from lib.config import Config, EmbeddingConfig, SearchConfig, load_config
from lib.embedder import create_embedder
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

class TestCreateEmbedder:
    """Tests for the create_embedder() factory function."""

    @patch("lib.providers.openrouter.OpenRouterProvider")
    def test_selects_openrouter(self, mock_cls):
        """Factory returns OpenRouterProvider when config says 'openrouter'."""
        config = Config()
        config.embedding = EmbeddingConfig(provider="openrouter", api_key="test-key")
        create_embedder(config)
        mock_cls.assert_called_once_with(config)

    @patch("lib.providers.huggingface.HuggingFaceProvider")
    def test_selects_huggingface(self, mock_cls):
        """Factory returns HuggingFaceProvider when config says 'huggingface'."""
        config = Config()
        config.embedding = EmbeddingConfig(provider="huggingface")
        create_embedder(config)
        mock_cls.assert_called_once_with(config)

    def test_unknown_provider_raises(self):
        """Unknown provider name raises EmbeddingError."""
        config = Config()
        config.embedding = EmbeddingConfig(provider="ollama")
        with pytest.raises(EmbeddingError, match="Unknown embedding provider"):
            create_embedder(config)


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
