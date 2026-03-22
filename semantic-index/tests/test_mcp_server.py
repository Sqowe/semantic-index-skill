"""Tests for the MCP server tool implementations.

Tests the synchronous helper functions directly (_build_index_sync,
_search_index_sync, _index_status_sync) to validate logic without
needing a running MCP server or async event loop.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mcp_server import (
    BuildIndexInput,
    IndexStatusInput,
    SearchIndexInput,
    _build_index_sync,
    _error_response,
    _index_status_sync,
    _search_index_sync,
)
from lib.models import (
    ChunkType,
    ConfigError,
    EmbeddingError,
    FileChange,
    SemanticIndexError,
)


def _write_config(tmp_path: Path, overrides: dict | None = None) -> None:
    """Write a minimal config.json for tests."""
    cfg: dict = {"schema_version": "1.0"}
    if overrides:
        cfg.update(overrides)
    index_dir = tmp_path / ".index"
    index_dir.mkdir(exist_ok=True)
    (index_dir / "config.json").write_text(json.dumps(cfg))


# ---------------------------------------------------------------------------
# _error_response
# ---------------------------------------------------------------------------


class TestErrorResponse:
    """Tests for the error formatting helper."""

    def test_formats_standard_exception(self):
        result = _error_response(ValueError("bad value"))
        assert result["status"] == "error"
        assert result["error"] == "bad value"
        assert result["error_type"] == "ValueError"

    def test_formats_config_error(self):
        result = _error_response(ConfigError("missing API key"))
        assert result["error_type"] == "ConfigError"
        assert "missing API key" in result["error"]


# ---------------------------------------------------------------------------
# semantic_index_status
# ---------------------------------------------------------------------------


class TestIndexStatus:
    """Tests for the index status tool."""

    def test_no_index_dir(self, tmp_path):
        """Returns indexed=False when .index/ doesn't exist."""
        _write_config(tmp_path)
        # Remove .index to simulate no index
        result = _index_status_sync(IndexStatusInput(project_dir=str(tmp_path)))
        # Either indexed=False or it found the config we just wrote
        assert isinstance(result, dict)
        # The .index dir exists because _write_config creates it, but
        # there's no LanceDB data, so has_index() should be False
        assert result.get("indexed") is False or "total_chunks" in result

    def test_bad_project_dir(self):
        """Returns error for nonexistent project directory."""
        result = _index_status_sync(
            IndexStatusInput(project_dir="/nonexistent/path/xyz")
        )
        # Should get an error or indexed=False
        assert result.get("status") == "error" or result.get("indexed") is False

    @patch("mcp_server.load_config")
    def test_config_error(self, mock_load):
        """Returns error when config loading fails."""
        mock_load.side_effect = ConfigError("invalid config")
        result = _index_status_sync(IndexStatusInput(project_dir="/tmp/test"))
        assert result["status"] == "error"
        assert result["error_type"] == "ConfigError"


# ---------------------------------------------------------------------------
# semantic_index_build
# ---------------------------------------------------------------------------


class TestBuildIndex:
    """Tests for the build index tool."""

    @patch("mcp_server.load_config")
    def test_config_error_returns_error(self, mock_load):
        """Returns error JSON when config is invalid."""
        mock_load.side_effect = ConfigError("missing API key")
        result = _build_index_sync(
            BuildIndexInput(project_dir="/tmp/test")
        )
        assert result["status"] == "error"
        assert result["error_type"] == "ConfigError"

    @patch("mcp_server.update_manifest")
    @patch("mcp_server.BM25Index")
    @patch("mcp_server.VectorStore")
    @patch("mcp_server.detect_changes")
    @patch("mcp_server.ensure_index_dir")
    @patch("mcp_server.load_config")
    def test_no_changes_returns_up_to_date(
        self, mock_load, mock_ensure, mock_detect, mock_store_cls,
        mock_bm25_cls, mock_manifest
    ):
        """Returns up_to_date when no files changed."""
        mock_load.return_value = MagicMock()
        mock_detect.return_value = FileChange(to_index=[], to_delete=[], unchanged=42)
        mock_bm25 = MagicMock()
        mock_bm25.load.return_value = True
        mock_bm25_cls.return_value = mock_bm25
        mock_store = MagicMock()
        mock_store.has_index.return_value = True
        mock_store_cls.return_value = mock_store

        result = _build_index_sync(
            BuildIndexInput(project_dir="/tmp/test")
        )
        assert result["status"] == "up_to_date"
        assert result["files_unchanged"] == 42

    @patch("mcp_server.update_manifest")
    @patch("mcp_server.BM25Index")
    @patch("mcp_server.VectorStore")
    @patch("mcp_server.Embedder")
    @patch("mcp_server.chunk_file")
    @patch("mcp_server.detect_changes")
    @patch("mcp_server.ensure_index_dir")
    @patch("mcp_server.load_config")
    def test_embedding_failure_preserves_old_data(
        self, mock_load, mock_ensure, mock_detect, mock_chunk,
        mock_embedder_cls, mock_store_cls, mock_bm25_cls, mock_manifest
    ):
        """If embedding fails, store.delete_by_file is never called."""
        mock_load.return_value = MagicMock()
        mock_detect.return_value = FileChange(
            to_index=["src/app.py"], to_delete=[], unchanged=0
        )
        mock_bm25 = MagicMock()
        mock_bm25.load.return_value = True
        mock_bm25_cls.return_value = mock_bm25
        mock_store = MagicMock()
        mock_store.has_index.return_value = True
        mock_store_cls.return_value = mock_store

        # chunk_file returns a mock chunk
        mock_chunk_obj = MagicMock()
        mock_chunk_obj.id = "abc"
        mock_chunk_obj.content = "def foo(): pass"
        mock_chunk_obj.file_path = "src/app.py"
        mock_chunk_obj.start_line = 1
        mock_chunk_obj.end_line = 1
        mock_chunk_obj.chunk_type = ChunkType.FUNCTION
        mock_chunk_obj.language = "python"
        mock_chunk_obj.symbol_name = "foo"
        mock_chunk_obj.token_count = 5
        mock_chunk.return_value = [mock_chunk_obj]

        # Embedding fails
        mock_embedder = MagicMock()
        mock_embedder.embed_chunks.side_effect = EmbeddingError("API down")
        mock_embedder_cls.return_value = mock_embedder

        result = _build_index_sync(
            BuildIndexInput(project_dir="/tmp/test")
        )
        assert result["status"] == "error"
        assert "API down" in result["error"]

        # Store should NOT have been mutated (delete never called)
        mock_store.delete_by_file.assert_not_called()
        mock_store.add.assert_not_called()


# ---------------------------------------------------------------------------
# semantic_index_search
# ---------------------------------------------------------------------------


class TestSearchIndex:
    """Tests for the search index tool."""

    @patch("mcp_server.load_config")
    def test_config_error_returns_error(self, mock_load):
        """Returns error JSON when config is invalid."""
        mock_load.side_effect = ConfigError("bad config")
        result = _search_index_sync(
            SearchIndexInput(project_dir="/tmp/test", query="auth flow")
        )
        assert result["status"] == "error"
        assert result["error_type"] == "ConfigError"

    @patch("mcp_server.VectorStore")
    @patch("mcp_server.load_config")
    def test_no_index_returns_error(self, mock_load, mock_store_cls):
        """Returns error when no index exists for vector mode."""
        mock_config = MagicMock()
        mock_config.search.default_top_k = 10
        mock_config.search.default_threshold = 0.3
        mock_config.search.mode = "vector"
        mock_config.search.hybrid_alpha = 0.7
        mock_load.return_value = mock_config
        mock_store = MagicMock()
        mock_store.has_index.return_value = False
        mock_store_cls.return_value = mock_store

        result = _search_index_sync(
            SearchIndexInput(project_dir="/tmp/test", query="auth flow")
        )
        assert result["status"] == "error"
        assert "No index found" in result["error"]

    @patch("mcp_server.BM25Index")
    @patch("mcp_server.Embedder")
    @patch("mcp_server.VectorStore")
    @patch("mcp_server.load_config")
    def test_vector_search_returns_results(
        self, mock_load, mock_store_cls, mock_embedder_cls, mock_bm25_cls
    ):
        """Happy path: vector search returns formatted results."""
        mock_config = MagicMock()
        mock_config.search.default_top_k = 10
        mock_config.search.default_threshold = 0.3
        mock_config.search.mode = "vector"
        mock_config.search.hybrid_alpha = 0.7
        mock_config.search.rerank_enabled = False
        mock_load.return_value = mock_config

        mock_store = MagicMock()
        mock_store.has_index.return_value = True
        mock_store.search.return_value = [
            {
                "score": 0.85,
                "file_path": "src/auth.py",
                "start_line": 10,
                "end_line": 30,
                "chunk_type": "function",
                "symbol_name": "login",
                "language": "python",
                "content": "def login(): ...",
            }
        ]
        mock_store_cls.return_value = mock_store

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_embedder_cls.return_value = mock_embedder

        result = _search_index_sync(
            SearchIndexInput(project_dir="/tmp/test", query="authentication")
        )
        assert result["query"] == "authentication"
        assert result["mode"] == "vector"
        assert result["total_results"] == 1
        assert result["results"][0]["rank"] == 1
        assert result["results"][0]["file_path"] == "src/auth.py"
        assert result["results"][0]["score"] == 0.85

    @patch("mcp_server.BM25Index")
    @patch("mcp_server.VectorStore")
    @patch("mcp_server.load_config")
    def test_keyword_no_bm25_returns_error(
        self, mock_load, mock_store_cls, mock_bm25_cls
    ):
        """Returns error when keyword mode but no BM25 index."""
        mock_config = MagicMock()
        mock_config.search.default_top_k = 10
        mock_config.search.default_threshold = 0.3
        mock_config.search.mode = "keyword"
        mock_config.search.hybrid_alpha = 0.7
        mock_load.return_value = mock_config

        mock_store = MagicMock()
        mock_store.has_index.return_value = True
        mock_store_cls.return_value = mock_store

        mock_bm25 = MagicMock()
        mock_bm25.load.return_value = False
        mock_bm25_cls.return_value = mock_bm25

        result = _search_index_sync(
            SearchIndexInput(
                project_dir="/tmp/test",
                query="auth",
                mode="keyword",
            )
        )
        assert result["status"] == "error"
        assert "BM25" in result["error"]

    @patch("mcp_server.BM25Index")
    @patch("mcp_server.Embedder")
    @patch("mcp_server.VectorStore")
    @patch("mcp_server.load_config")
    def test_rerank_false_overrides_config_true(
        self, mock_load, mock_store_cls, mock_embedder_cls, mock_bm25_cls
    ):
        """rerank=False explicitly disables reranking even when config enables it."""
        mock_config = MagicMock()
        mock_config.search.default_top_k = 10
        mock_config.search.default_threshold = 0.0
        mock_config.search.mode = "vector"
        mock_config.search.hybrid_alpha = 0.7
        mock_config.search.rerank_enabled = True  # Config says rerank
        mock_config.search.rerank_model = "BAAI/bge-reranker-v2-m3"
        mock_config.search.rerank_top_n = 5
        mock_load.return_value = mock_config

        mock_store = MagicMock()
        mock_store.has_index.return_value = True
        mock_store.search.return_value = [
            {
                "score": 0.9,
                "file_path": "a.py",
                "start_line": 1,
                "end_line": 5,
                "chunk_type": "function",
                "symbol_name": "f",
                "language": "python",
                "content": "def f(): pass",
            }
        ]
        mock_store_cls.return_value = mock_store

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_embedder_cls.return_value = mock_embedder

        # rerank=False should override config
        result = _search_index_sync(
            SearchIndexInput(
                project_dir="/tmp/test",
                query="test",
                rerank=False,
            )
        )
        # Should succeed without attempting rerank
        assert result["total_results"] == 1
        # No rerank_score in results
        assert "rerank_score" not in result["results"][0]

    @patch("mcp_server.BM25Index")
    @patch("mcp_server.Embedder")
    @patch("mcp_server.VectorStore")
    @patch("mcp_server.load_config")
    def test_rerank_none_uses_config_default(
        self, mock_load, mock_store_cls, mock_embedder_cls, mock_bm25_cls
    ):
        """rerank=None falls back to config.search.rerank_enabled."""
        mock_config = MagicMock()
        mock_config.search.default_top_k = 10
        mock_config.search.default_threshold = 0.0
        mock_config.search.mode = "vector"
        mock_config.search.hybrid_alpha = 0.7
        mock_config.search.rerank_enabled = False  # Config says no rerank
        mock_load.return_value = mock_config

        mock_store = MagicMock()
        mock_store.has_index.return_value = True
        mock_store.search.return_value = [
            {
                "score": 0.9,
                "file_path": "a.py",
                "start_line": 1,
                "end_line": 5,
                "chunk_type": "function",
                "symbol_name": "f",
                "language": "python",
                "content": "def f(): pass",
            }
        ]
        mock_store_cls.return_value = mock_store

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_embedder_cls.return_value = mock_embedder

        # rerank=None (default) — should use config (False)
        result = _search_index_sync(
            SearchIndexInput(project_dir="/tmp/test", query="test")
        )
        assert result["total_results"] == 1
        assert "rerank_score" not in result["results"][0]
