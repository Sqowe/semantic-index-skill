"""Tests for the MCP server tool implementations.

Tests the synchronous helper functions directly (_build_index_sync,
_search_index_sync, _index_status_sync) to validate logic without
needing a running MCP server or async event loop.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mcp_server import (
    BuildIndexInput,
    IndexStatusInput,
    SearchIndexInput,
    SearchMode,
    _build_index_sync,
    _error_response,
    _index_status_sync,
    _search_index_sync,
    mcp,
    semantic_index_build,
    semantic_index_search,
    semantic_index_status,
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

    @patch("mcp_server.detect_changes")
    @patch("mcp_server.VectorStore")
    @patch("mcp_server.load_config")
    def test_no_index_dir(self, mock_load, mock_store_cls, mock_detect, tmp_path):
        """Returns indexed=False when no LanceDB data exists."""
        mock_load.return_value = MagicMock()
        mock_store = MagicMock()
        mock_store.has_index.return_value = False
        mock_store.get_stats.return_value = {"total_chunks": 0}
        mock_store_cls.return_value = mock_store
        mock_detect.return_value = FileChange(to_index=[], to_delete=[], unchanged=0)

        result = _index_status_sync(IndexStatusInput(project_dir=str(tmp_path)))
        assert result.get("indexed") is False

    @patch("mcp_server.load_config")
    def test_bad_project_dir(self, mock_load):
        """Returns error for nonexistent project directory."""
        mock_load.side_effect = ConfigError("Project directory not found")
        result = _index_status_sync(
            IndexStatusInput(project_dir="/nonexistent/path/xyz")
        )
        assert result["status"] == "error"
        assert result["error_type"] == "ConfigError"

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


# ---------------------------------------------------------------------------
# Tool registration and annotations
# ---------------------------------------------------------------------------


class TestToolRegistration:
    """Tests that MCP tools are registered with correct names and annotations."""

    def _get_tool_map(self) -> dict:
        """Build a name → tool dict via the public FastMCP list_tools() API."""
        tools = asyncio.run(mcp.list_tools())
        return {tool.name: tool for tool in tools}

    def test_all_three_tools_registered(self):
        """All three tools are registered on the mcp server."""
        tools = self._get_tool_map()
        assert "semantic_index_build" in tools
        assert "semantic_index_search" in tools
        assert "semantic_index_status" in tools

    def test_build_annotations(self):
        """Build tool is not read-only, not destructive, is idempotent."""
        tool = self._get_tool_map()["semantic_index_build"]
        ann = tool.annotations
        assert ann.readOnlyHint is False
        assert ann.destructiveHint is False
        assert ann.idempotentHint is True
        assert ann.openWorldHint is True

    def test_search_annotations(self):
        """Search tool is read-only, not destructive, is idempotent."""
        tool = self._get_tool_map()["semantic_index_search"]
        ann = tool.annotations
        assert ann.readOnlyHint is True
        assert ann.destructiveHint is False
        assert ann.idempotentHint is True

    def test_status_annotations(self):
        """Status tool is read-only, not destructive, not open-world."""
        tool = self._get_tool_map()["semantic_index_status"]
        ann = tool.annotations
        assert ann.readOnlyHint is True
        assert ann.destructiveHint is False
        assert ann.idempotentHint is True
        assert ann.openWorldHint is False


# ---------------------------------------------------------------------------
# Pydantic input validation
# ---------------------------------------------------------------------------


class TestPydanticValidation:
    """Tests that Pydantic Field() constraints reject invalid inputs."""

    def test_build_rejects_empty_project_dir(self):
        """project_dir must be non-empty."""
        with pytest.raises(ValidationError):
            BuildIndexInput(project_dir="")

    def test_build_rejects_whitespace_project_dir(self):
        """project_dir is stripped then validated as min_length=1."""
        with pytest.raises(ValidationError):
            BuildIndexInput(project_dir="   ")

    def test_build_rejects_batch_size_zero(self):
        """batch_size must be >= 1."""
        with pytest.raises(ValidationError):
            BuildIndexInput(project_dir="/tmp/test", batch_size=0)

    def test_build_rejects_batch_size_over_max(self):
        """batch_size must be <= 500."""
        with pytest.raises(ValidationError):
            BuildIndexInput(project_dir="/tmp/test", batch_size=501)

    def test_search_rejects_empty_query(self):
        """query must be non-empty."""
        with pytest.raises(ValidationError):
            SearchIndexInput(project_dir="/tmp/test", query="")

    def test_search_rejects_query_too_long(self):
        """query must be <= 2000 characters."""
        with pytest.raises(ValidationError):
            SearchIndexInput(project_dir="/tmp/test", query="x" * 2001)

    def test_search_rejects_top_k_zero(self):
        """top_k must be >= 1."""
        with pytest.raises(ValidationError):
            SearchIndexInput(project_dir="/tmp/test", query="test", top_k=0)

    def test_search_rejects_threshold_out_of_range(self):
        """threshold must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            SearchIndexInput(project_dir="/tmp/test", query="test", threshold=1.5)

    def test_search_rejects_extra_fields(self):
        """extra='forbid' rejects unknown fields."""
        with pytest.raises(ValidationError):
            SearchIndexInput(project_dir="/tmp/test", query="test", unknown_field="x")

    def test_search_accepts_valid_mode(self):
        """SearchMode enum values are accepted."""
        params = SearchIndexInput(
            project_dir="/tmp/test", query="test", mode="hybrid"
        )
        assert params.mode == SearchMode.HYBRID

    def test_search_rejects_invalid_mode(self):
        """Invalid mode string is rejected."""
        with pytest.raises(ValidationError):
            SearchIndexInput(project_dir="/tmp/test", query="test", mode="invalid")


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


class TestAsyncWrappers:
    """Tests that async tool functions delegate to sync helpers via to_thread."""

    @patch("mcp_server.asyncio.to_thread")
    def test_build_delegates_via_to_thread(self, mock_to_thread):
        """semantic_index_build calls asyncio.to_thread with _build_index_sync."""
        mock_to_thread.return_value = {"status": "up_to_date"}
        params = BuildIndexInput(project_dir="/tmp/test")
        result = asyncio.run(semantic_index_build(params))
        mock_to_thread.assert_called_once_with(_build_index_sync, params)
        assert result["status"] == "up_to_date"

    @patch("mcp_server.asyncio.to_thread")
    def test_search_delegates_via_to_thread(self, mock_to_thread):
        """semantic_index_search calls asyncio.to_thread with _search_index_sync."""
        mock_to_thread.return_value = {"query": "test", "total_results": 0, "results": []}
        params = SearchIndexInput(project_dir="/tmp/test", query="test")
        result = asyncio.run(semantic_index_search(params))
        mock_to_thread.assert_called_once_with(_search_index_sync, params)
        assert result["total_results"] == 0

    @patch("mcp_server.asyncio.to_thread")
    def test_status_delegates_via_to_thread(self, mock_to_thread):
        """semantic_index_status calls asyncio.to_thread with _index_status_sync."""
        mock_to_thread.return_value = {"indexed": False}
        params = IndexStatusInput(project_dir="/tmp/test")
        result = asyncio.run(semantic_index_status(params))
        mock_to_thread.assert_called_once_with(_index_status_sync, params)
        assert result["indexed"] is False
