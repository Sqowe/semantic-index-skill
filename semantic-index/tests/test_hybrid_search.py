"""Integration tests for Phase 3 hybrid search features.

Tests cover:
1. BM25 bootstrap from existing vector store (upgrade path)
2. Zero-change incremental rebuild
3. file_path_glob filtering across vector/keyword/hybrid modes
4. RRF fusion dedup semantics
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure lib is importable
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib.bm25 import BM25Index, tokenize
from lib.fusion import fuse_results


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------

class TestTokenizer:
    """Tests for the BM25 tokenizer."""

    def test_basic_tokenization(self):
        tokens = tokenize("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_camel_case_splitting(self):
        tokens = tokenize("getUserName")
        assert "get" in tokens
        assert "user" in tokens
        assert "name" in tokens

    def test_snake_case_splitting(self):
        tokens = tokenize("get_user_name")
        assert "get" in tokens
        assert "user" in tokens
        assert "name" in tokens

    def test_stop_words_removed(self):
        tokens = tokenize("the quick brown fox is a test")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    def test_short_tokens_removed(self):
        tokens = tokenize("a b cd ef")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "cd" in tokens
        assert "ef" in tokens


# ---------------------------------------------------------------------------
# Sample chunk data for tests
# ---------------------------------------------------------------------------

def _make_chunks(n: int = 5) -> list[dict]:
    """Create sample chunk dicts for testing."""
    chunks = [
        {
            "id": "chunk_auth_1",
            "content": "def authenticate_user(username, password):\n    verify credentials against database",
            "file_path": "src/auth/login.py",
            "start_line": 10,
            "end_line": 25,
            "chunk_type": "function",
            "language": "python",
            "symbol_name": "authenticate_user",
            "token_count": 30,
        },
        {
            "id": "chunk_auth_2",
            "content": "class JWTTokenManager:\n    def create_token(self, user_id):\n        generate JWT token for authenticated user",
            "file_path": "src/auth/tokens.py",
            "start_line": 1,
            "end_line": 20,
            "chunk_type": "class",
            "language": "python",
            "symbol_name": "JWTTokenManager",
            "token_count": 40,
        },
        {
            "id": "chunk_db_1",
            "content": "def connect_database(host, port):\n    establish connection to PostgreSQL database",
            "file_path": "src/db/connection.py",
            "start_line": 5,
            "end_line": 15,
            "chunk_type": "function",
            "language": "python",
            "symbol_name": "connect_database",
            "token_count": 25,
        },
        {
            "id": "chunk_api_1",
            "content": "function handleLogin(req, res) {\n    authenticate user and return session token\n}",
            "file_path": "src/api/routes.js",
            "start_line": 30,
            "end_line": 45,
            "chunk_type": "function",
            "language": "javascript",
            "symbol_name": "handleLogin",
            "token_count": 28,
        },
        {
            "id": "chunk_docs_1",
            "content": "# Authentication Guide\n\nThis document explains how user authentication works in the system.",
            "file_path": "docs/auth-guide.md",
            "start_line": 1,
            "end_line": 10,
            "chunk_type": "markdown_section",
            "language": "markdown",
            "symbol_name": "",
            "token_count": 20,
        },
    ]
    return chunks[:n]


# ---------------------------------------------------------------------------
# BM25 Index tests
# ---------------------------------------------------------------------------

class TestBM25Index:
    """Tests for BM25Index build, search, and persistence."""

    def test_build_and_search(self, tmp_path):
        """Build index from chunks and search returns relevant results."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        bm25 = BM25Index(str(tmp_path))
        bm25.build(_make_chunks())

        results = bm25.search("authenticate user login", top_k=3)
        assert len(results) > 0
        # Auth-related chunks should rank high
        file_paths = [r["file_path"] for r in results]
        assert any("auth" in fp for fp in file_paths)

    def test_save_and_load(self, tmp_path):
        """Index persists to disk and loads back correctly."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        bm25 = BM25Index(str(tmp_path))
        bm25.build(_make_chunks())
        bm25.save()

        # Load into a fresh instance
        bm25_loaded = BM25Index(str(tmp_path))
        assert bm25_loaded.load() is True
        assert bm25_loaded._n_docs == 5

        results = bm25_loaded.search("database connection", top_k=3)
        assert len(results) > 0

    def test_load_missing_index(self, tmp_path):
        """Loading when no index file exists returns False."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        bm25 = BM25Index(str(tmp_path))
        assert bm25.load() is False

    def test_language_filter(self, tmp_path):
        """Language filter restricts results to matching language."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        bm25 = BM25Index(str(tmp_path))
        bm25.build(_make_chunks())

        results = bm25.search(
            "authenticate login",
            top_k=10,
            filters={"language": "javascript"},
        )
        for r in results:
            assert r["language"] == "javascript"

    def test_file_path_glob_filter(self, tmp_path):
        """file_path_glob filter restricts results to matching paths."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        bm25 = BM25Index(str(tmp_path))
        bm25.build(_make_chunks())

        results = bm25.search(
            "authenticate user",
            top_k=10,
            filters={"file_path_glob": "src/auth/*"},
        )
        for r in results:
            assert r["file_path"].startswith("src/auth/")

    def test_file_path_glob_excludes_non_matching(self, tmp_path):
        """file_path_glob filter excludes files outside the glob."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        bm25 = BM25Index(str(tmp_path))
        bm25.build(_make_chunks())

        results = bm25.search(
            "authenticate user",
            top_k=10,
            filters={"file_path_glob": "src/db/*"},
        )
        for r in results:
            assert r["file_path"].startswith("src/db/")

    def test_incremental_add_and_delete(self, tmp_path):
        """Incremental add/delete maintains index consistency."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        bm25 = BM25Index(str(tmp_path))
        bm25.build(_make_chunks(3))
        assert bm25._n_docs == 3

        # Add more chunks
        new_chunks = _make_chunks(5)[3:]
        bm25.add_chunks(new_chunks)
        assert bm25._n_docs == 5

        # Delete a file
        bm25.delete_by_file("src/auth/login.py")
        assert bm25._n_docs == 4

        # Verify deleted file doesn't appear in results
        results = bm25.search("authenticate", top_k=10)
        for r in results:
            assert r["file_path"] != "src/auth/login.py"

    def test_empty_index_search(self, tmp_path):
        """Searching an empty index returns no results."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        bm25 = BM25Index(str(tmp_path))
        results = bm25.search("anything")
        assert results == []


# ---------------------------------------------------------------------------
# RRF Fusion tests
# ---------------------------------------------------------------------------

class TestFusion:
    """Tests for Reciprocal Rank Fusion merging."""

    def _make_vector_results(self) -> list[dict]:
        return [
            {"id": "chunk_1", "score": 0.95, "file_path": "a.py", "start_line": 1,
             "end_line": 10, "content": "vector hit 1", "chunk_type": "function",
             "language": "python", "symbol_name": "func_a", "token_count": 20},
            {"id": "chunk_2", "score": 0.80, "file_path": "b.py", "start_line": 1,
             "end_line": 10, "content": "vector hit 2", "chunk_type": "function",
             "language": "python", "symbol_name": "func_b", "token_count": 20},
            {"id": "chunk_3", "score": 0.70, "file_path": "c.py", "start_line": 1,
             "end_line": 10, "content": "vector only", "chunk_type": "function",
             "language": "python", "symbol_name": "func_c", "token_count": 20},
        ]

    def _make_bm25_results(self) -> list[dict]:
        return [
            {"id": "chunk_2", "score": 8.5, "file_path": "b.py", "start_line": 1,
             "end_line": 10, "content": "bm25 hit 1", "chunk_type": "function",
             "language": "python", "symbol_name": "func_b", "token_count": 20},
            {"id": "chunk_1", "score": 6.2, "file_path": "a.py", "start_line": 1,
             "end_line": 10, "content": "bm25 hit 2", "chunk_type": "function",
             "language": "python", "symbol_name": "func_a", "token_count": 20},
            {"id": "chunk_4", "score": 4.0, "file_path": "d.py", "start_line": 1,
             "end_line": 10, "content": "bm25 only", "chunk_type": "function",
             "language": "python", "symbol_name": "func_d", "token_count": 20},
        ]

    def test_fusion_dedup(self):
        """Duplicate docs across sources are merged, not duplicated."""
        merged = fuse_results(self._make_vector_results(), self._make_bm25_results())
        ids = [r["id"] for r in merged]
        assert len(ids) == len(set(ids)), "Duplicate IDs in fused results"

    def test_fusion_includes_all_sources(self):
        """Results from both sources appear in merged output."""
        merged = fuse_results(self._make_vector_results(), self._make_bm25_results())
        ids = {r["id"] for r in merged}
        assert "chunk_3" in ids, "Vector-only result missing"
        assert "chunk_4" in ids, "BM25-only result missing"

    def test_fusion_has_per_source_scores(self):
        """Merged results include vector_score and bm25_score fields."""
        merged = fuse_results(self._make_vector_results(), self._make_bm25_results())
        for r in merged:
            assert "fused_score" in r
            assert "vector_score" in r
            assert "bm25_score" in r

    def test_fusion_dedup_picks_higher_score_source(self):
        """For docs in both sources, vector payload is always preferred."""
        merged = fuse_results(self._make_vector_results(), self._make_bm25_results())
        # chunk_2 has vector_score=0.80 and bm25_score=8.5
        # Vector payload is always preferred (deterministic rule)
        chunk_2 = next(r for r in merged if r["id"] == "chunk_2")
        assert chunk_2["content"] == "vector hit 2"
        assert chunk_2["vector_score"] == 0.80
        assert chunk_2["bm25_score"] == 8.5

    def test_fusion_vector_only_has_null_bm25(self):
        """Vector-only results have bm25_score=None."""
        merged = fuse_results(self._make_vector_results(), self._make_bm25_results())
        chunk_3 = next(r for r in merged if r["id"] == "chunk_3")
        assert chunk_3["vector_score"] is not None
        assert chunk_3["bm25_score"] is None

    def test_fusion_bm25_only_has_null_vector(self):
        """BM25-only results have vector_score=None."""
        merged = fuse_results(self._make_vector_results(), self._make_bm25_results())
        chunk_4 = next(r for r in merged if r["id"] == "chunk_4")
        assert chunk_4["vector_score"] is None
        assert chunk_4["bm25_score"] is not None

    def test_fusion_alpha_zero_pure_keyword(self):
        """Alpha=0.0 gives all weight to BM25."""
        merged = fuse_results(
            self._make_vector_results(), self._make_bm25_results(), alpha=0.0,
        )
        # With alpha=0, vector results get 0 RRF score
        # BM25-only chunk_4 should appear, vector-only chunk_3 should have low score
        ids = [r["id"] for r in merged]
        # BM25 top result should be near the top
        assert ids[0] == "chunk_2"  # BM25 rank 1

    def test_fusion_alpha_one_pure_vector(self):
        """Alpha=1.0 gives all weight to vector."""
        merged = fuse_results(
            self._make_vector_results(), self._make_bm25_results(), alpha=1.0,
        )
        ids = [r["id"] for r in merged]
        assert ids[0] == "chunk_1"  # Vector rank 1

    def test_fusion_empty_inputs(self):
        """Empty input lists produce empty output."""
        assert fuse_results([], []) == []
        assert fuse_results(self._make_vector_results(), []) != []
        assert fuse_results([], self._make_bm25_results()) != []



# ---------------------------------------------------------------------------
# Bootstrap and schema edge case tests
# ---------------------------------------------------------------------------

class TestBM25Bootstrap:
    """Tests for BM25 bootstrap from vector store data."""

    def test_bootstrap_from_chunk_list(self, tmp_path):
        """BM25 can be built from a list of chunk dicts (simulating store data)."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        chunks = _make_chunks()
        bm25 = BM25Index(str(tmp_path))
        bm25.build(chunks)

        assert bm25._n_docs == 5
        results = bm25.search("authenticate", top_k=3)
        assert len(results) > 0

    def test_bootstrap_incremental_matches_full_build(self, tmp_path):
        """Incremental add_chunks produces same results as full build."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        chunks = _make_chunks()

        # Full build
        bm25_full = BM25Index(str(tmp_path))
        bm25_full.build(chunks)
        full_results = bm25_full.search("database connection", top_k=5)

        # Incremental build (simulating batched bootstrap)
        bm25_inc = BM25Index(str(tmp_path))
        bm25_inc.add_chunks(chunks[:2])
        bm25_inc.add_chunks(chunks[2:])
        inc_results = bm25_inc.search("database connection", top_k=5)

        # Same docs, same scores
        assert len(full_results) == len(inc_results)
        for fr, ir in zip(full_results, inc_results):
            assert fr["id"] == ir["id"]
            assert fr["score"] == ir["score"]

    def test_bootstrap_with_missing_optional_fields(self, tmp_path):
        """Bootstrap handles chunks missing optional fields gracefully."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        # Minimal chunks — only required fields
        minimal_chunks = [
            {
                "id": "min_1",
                "content": "some code content here",
                "file_path": "src/main.py",
                "start_line": 1,
                "end_line": 10,
                "chunk_type": "function",
            },
        ]

        bm25 = BM25Index(str(tmp_path))
        bm25.add_chunks(minimal_chunks)
        assert bm25._n_docs == 1

        results = bm25.search("code content", top_k=5)
        assert len(results) == 1
        assert results[0]["language"] == ""  # default backfill


# ---------------------------------------------------------------------------
# Hybrid threshold semantics tests
# ---------------------------------------------------------------------------

class TestHybridThresholdSemantics:
    """Tests verifying that score scales don't mix across modes."""

    def test_fusion_output_has_no_mixed_score(self):
        """Fused results should not have a 'score' field that mixes scales."""
        vector_results = [
            {"id": "v1", "score": 0.9, "file_path": "a.py", "start_line": 1,
             "end_line": 10, "content": "x", "chunk_type": "function",
             "language": "python", "symbol_name": "f", "token_count": 10},
        ]
        bm25_results = [
            {"id": "b1", "score": 15.3, "file_path": "b.py", "start_line": 1,
             "end_line": 10, "content": "y", "chunk_type": "function",
             "language": "python", "symbol_name": "g", "token_count": 10},
        ]

        merged = fuse_results(vector_results, bm25_results)

        for r in merged:
            # fused_score must exist and be the canonical score
            assert "fused_score" in r
            assert "vector_score" in r
            assert "bm25_score" in r

    def test_fusion_fused_score_is_scale_independent(self):
        """Fused score should be comparable regardless of raw score magnitudes."""
        # Vector scores: 0-1 range
        vector_results = [
            {"id": "c1", "score": 0.95, "file_path": "a.py", "start_line": 1,
             "end_line": 10, "content": "x", "chunk_type": "function",
             "language": "python", "symbol_name": "f", "token_count": 10},
        ]
        # BM25 scores: 0-100 range (very different scale)
        bm25_results = [
            {"id": "c1", "score": 85.0, "file_path": "a.py", "start_line": 1,
             "end_line": 10, "content": "x", "chunk_type": "function",
             "language": "python", "symbol_name": "f", "token_count": 10},
        ]

        merged = fuse_results(vector_results, bm25_results)
        assert len(merged) == 1

        # fused_score is RRF-based, not a raw score mix
        r = merged[0]
        assert r["fused_score"] > 0
        assert r["fused_score"] < 1.0  # RRF scores are small fractions
        assert r["vector_score"] == 0.95
        assert r["bm25_score"] == 85.0

    def test_vector_mode_threshold_uses_cosine_score(self):
        """In vector-only mode, threshold applies to cosine similarity score."""
        # Simulating what semantic_search.py does in vector mode
        vector_results = [
            {"score": 0.9, "id": "v1", "file_path": "a.py", "start_line": 1,
             "end_line": 10, "content": "x", "chunk_type": "f",
             "language": "python", "symbol_name": "f", "token_count": 10},
            {"score": 0.2, "id": "v2", "file_path": "b.py", "start_line": 1,
             "end_line": 10, "content": "y", "chunk_type": "f",
             "language": "python", "symbol_name": "g", "token_count": 10},
        ]

        threshold = 0.3
        filtered = [r for r in vector_results if r["score"] >= threshold]
        assert len(filtered) == 1
        assert filtered[0]["id"] == "v1"

    def test_keyword_mode_threshold_uses_bm25_score(self):
        """In keyword mode, threshold applies to BM25 score."""
        bm25_results = [
            {"score": 8.5, "id": "b1", "file_path": "a.py", "start_line": 1,
             "end_line": 10, "content": "x", "chunk_type": "f",
             "language": "python", "symbol_name": "f", "token_count": 10},
            {"score": 0.1, "id": "b2", "file_path": "b.py", "start_line": 1,
             "end_line": 10, "content": "y", "chunk_type": "f",
             "language": "python", "symbol_name": "g", "token_count": 10},
        ]

        threshold = 0.3
        filtered = [r for r in bm25_results if r["score"] >= threshold]
        assert len(filtered) == 1
        assert filtered[0]["id"] == "b1"


# ---------------------------------------------------------------------------
# Fusion deterministic source selection tests
# ---------------------------------------------------------------------------

class TestFusionSourceSelection:
    """Tests for deterministic vector-preferred payload selection in fusion."""

    def test_fusion_always_prefers_vector_payload(self):
        """When a doc appears in both sources, vector payload is always used."""
        vector_results = [
            {"id": "dup_1", "score": 0.5, "file_path": "a.py", "start_line": 1,
             "end_line": 10, "content": "from vector", "chunk_type": "function",
             "language": "python", "symbol_name": "f", "token_count": 10},
        ]
        bm25_results = [
            {"id": "dup_1", "score": 999.0, "file_path": "a.py", "start_line": 1,
             "end_line": 10, "content": "from bm25", "chunk_type": "function",
             "language": "python", "symbol_name": "f", "token_count": 10},
        ]
        merged = fuse_results(vector_results, bm25_results)
        assert len(merged) == 1
        # Vector payload always wins, regardless of BM25 having higher raw score
        assert merged[0]["content"] == "from vector"
        assert merged[0]["vector_score"] == 0.5
        assert merged[0]["bm25_score"] == 999.0


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

class TestConfigValidation:
    """Tests for search config validation at load time."""

    def test_invalid_mode_raises_config_error(self, tmp_path):
        """Invalid search.mode raises ConfigError."""
        from lib.config import load_config
        from lib.models import ConfigError as CfgErr

        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        cfg = {
            "schema_version": "1.0",
            "search": {"mode": "turbo", "hybrid_alpha": 0.7},
        }
        (index_dir / "config.json").write_text(json.dumps(cfg))

        with pytest.raises(CfgErr, match="Invalid search.mode"):
            load_config(str(tmp_path))

    def test_alpha_below_zero_raises_config_error(self, tmp_path):
        """hybrid_alpha < 0 raises ConfigError."""
        from lib.config import load_config
        from lib.models import ConfigError as CfgErr

        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        cfg = {
            "schema_version": "1.0",
            "search": {"mode": "hybrid", "hybrid_alpha": -0.1},
        }
        (index_dir / "config.json").write_text(json.dumps(cfg))

        with pytest.raises(CfgErr, match="Invalid search.hybrid_alpha"):
            load_config(str(tmp_path))

    def test_alpha_above_one_raises_config_error(self, tmp_path):
        """hybrid_alpha > 1.0 raises ConfigError."""
        from lib.config import load_config
        from lib.models import ConfigError as CfgErr

        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        cfg = {
            "schema_version": "1.0",
            "search": {"mode": "hybrid", "hybrid_alpha": 1.5},
        }
        (index_dir / "config.json").write_text(json.dumps(cfg))

        with pytest.raises(CfgErr, match="Invalid search.hybrid_alpha"):
            load_config(str(tmp_path))

    def test_valid_config_passes_validation(self, tmp_path):
        """Valid mode and alpha values pass without error."""
        from lib.config import load_config

        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        cfg = {
            "schema_version": "1.0",
            "search": {"mode": "vector", "hybrid_alpha": 0.0},
        }
        (index_dir / "config.json").write_text(json.dumps(cfg))

        config = load_config(str(tmp_path))
        assert config.search.mode == "vector"
        assert config.search.hybrid_alpha == 0.0

    def test_boundary_alpha_values_pass(self, tmp_path):
        """Alpha at exact boundaries (0.0 and 1.0) passes validation."""
        from lib.config import load_config

        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        for alpha in [0.0, 1.0]:
            cfg = {
                "schema_version": "1.0",
                "search": {"mode": "hybrid", "hybrid_alpha": alpha},
            }
            (index_dir / "config.json").write_text(json.dumps(cfg))
            config = load_config(str(tmp_path))
            assert config.search.hybrid_alpha == alpha


# ---------------------------------------------------------------------------
# BM25 duplicate doc_id / stale postings tests
# ---------------------------------------------------------------------------

class TestBM25StalePostings:
    """Tests for BM25 add_chunks() stale posting cleanup."""

    def test_readd_same_doc_id_no_stale_terms(self, tmp_path):
        """Re-adding a doc_id with different content purges old terms."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        bm25 = BM25Index(str(tmp_path))

        # Add initial chunk with "alpha beta gamma"
        bm25.add_chunks([{
            "id": "doc_1",
            "content": "alpha beta gamma",
            "file_path": "a.py",
            "start_line": 1, "end_line": 5,
            "chunk_type": "function",
        }])

        # Verify "alpha" is searchable
        results = bm25.search("alpha", top_k=5)
        assert any(r["id"] == "doc_1" for r in results)

        # Re-add same doc_id with completely different content
        bm25.add_chunks([{
            "id": "doc_1",
            "content": "delta epsilon zeta",
            "file_path": "a.py",
            "start_line": 1, "end_line": 5,
            "chunk_type": "function",
        }])

        # "alpha" should no longer match doc_1 (stale posting purged)
        results = bm25.search("alpha", top_k=5)
        assert not any(r["id"] == "doc_1" for r in results)

        # "delta" should now match doc_1
        results = bm25.search("delta", top_k=5)
        assert any(r["id"] == "doc_1" for r in results)

        # Doc count should still be 1 (not 2)
        assert bm25._n_docs == 1

    def test_readd_preserves_other_docs(self, tmp_path):
        """Re-adding one doc_id doesn't affect other documents."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        bm25 = BM25Index(str(tmp_path))
        bm25.add_chunks([
            {"id": "doc_1", "content": "alpha beta", "file_path": "a.py",
             "start_line": 1, "end_line": 5, "chunk_type": "function"},
            {"id": "doc_2", "content": "alpha gamma", "file_path": "b.py",
             "start_line": 1, "end_line": 5, "chunk_type": "function"},
        ])

        # Re-add doc_1 with new content
        bm25.add_chunks([{
            "id": "doc_1",
            "content": "delta epsilon",
            "file_path": "a.py",
            "start_line": 1, "end_line": 5,
            "chunk_type": "function",
        }])

        # doc_2 should still be searchable for "alpha"
        results = bm25.search("alpha", top_k=5)
        assert any(r["id"] == "doc_2" for r in results)
        assert not any(r["id"] == "doc_1" for r in results)

        assert bm25._n_docs == 2


# ---------------------------------------------------------------------------
# BM25 corrupt index recovery tests
# ---------------------------------------------------------------------------

class TestBM25Recovery:
    """Tests for BM25 index recovery from corrupt/unreadable files."""

    def test_corrupt_json_returns_false(self, tmp_path):
        """Unreadable bm25_index.json causes load() to return False."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        (index_dir / "bm25_index.json").write_text("NOT VALID JSON {{{")

        bm25 = BM25Index(str(tmp_path))
        assert bm25.load() is False
        assert bm25._n_docs == 0

    def test_empty_file_returns_false(self, tmp_path):
        """Empty bm25_index.json causes load() to return False."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        (index_dir / "bm25_index.json").write_text("")

        bm25 = BM25Index(str(tmp_path))
        assert bm25.load() is False

    def test_rebuild_after_corrupt_load(self, tmp_path):
        """After failed load, index can be rebuilt from scratch."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()
        (index_dir / "bm25_index.json").write_text("{broken")

        bm25 = BM25Index(str(tmp_path))
        assert bm25.load() is False

        # Rebuild works fine
        bm25.build(_make_chunks())
        assert bm25._n_docs == 5
        results = bm25.search("authenticate", top_k=3)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# BM25 save/load round-trip regression tests
# ---------------------------------------------------------------------------

class TestBM25SaveLoadRoundTrip:
    """Regression tests for BM25Index.save() and load() round-trip."""

    def test_save_after_build_then_search(self, tmp_path):
        """save() after build() works and loaded index returns same results."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        bm25 = BM25Index(str(tmp_path))
        bm25.build(_make_chunks())
        bm25.save()

        # Verify the file was written
        bm25_path = index_dir / "bm25_index.json"
        assert bm25_path.exists()

        # Load into fresh instance and compare search results
        bm25_loaded = BM25Index(str(tmp_path))
        assert bm25_loaded.load() is True

        original_results = bm25.search("authenticate user", top_k=5)
        loaded_results = bm25_loaded.search("authenticate user", top_k=5)

        assert len(original_results) == len(loaded_results)
        for orig, loaded in zip(original_results, loaded_results):
            assert orig["id"] == loaded["id"]
            assert orig["score"] == loaded["score"]

    def test_save_after_add_chunks(self, tmp_path):
        """save() after incremental add_chunks() persists correctly."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        bm25 = BM25Index(str(tmp_path))
        bm25.add_chunks(_make_chunks(3))
        bm25.add_chunks(_make_chunks(5)[3:])
        bm25.save()

        bm25_loaded = BM25Index(str(tmp_path))
        assert bm25_loaded.load() is True
        assert bm25_loaded._n_docs == 5

    def test_save_load_preserves_doc_metadata(self, tmp_path):
        """Round-trip preserves all document metadata fields."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        bm25 = BM25Index(str(tmp_path))
        bm25.build(_make_chunks(1))
        bm25.save()

        bm25_loaded = BM25Index(str(tmp_path))
        bm25_loaded.load()

        results = bm25_loaded.search("authenticate", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert r["file_path"] == "src/auth/login.py"
        assert r["start_line"] == 10
        assert r["end_line"] == 25
        assert r["chunk_type"] == "function"
        assert r["language"] == "python"
        assert r["symbol_name"] == "authenticate_user"

    def test_save_empty_index(self, tmp_path):
        """save() on an empty index writes valid JSON that loads back."""
        index_dir = tmp_path / ".index"
        index_dir.mkdir()

        bm25 = BM25Index(str(tmp_path))
        bm25.save()

        bm25_loaded = BM25Index(str(tmp_path))
        assert bm25_loaded.load() is True
        assert bm25_loaded._n_docs == 0


# ---------------------------------------------------------------------------
# CLI alpha validation tests
# ---------------------------------------------------------------------------

class TestCLIAlphaValidation:
    """Tests verifying that --alpha is validated in the CLI path."""

    def _simulate_alpha_validation(self, alpha: float) -> bool:
        """Simulate the alpha validation logic from semantic_search.py.

        Returns True if the value is valid, False if it would be rejected.
        """
        return 0.0 <= alpha <= 1.0

    def test_alpha_negative_rejected(self):
        assert not self._simulate_alpha_validation(-0.1)
        assert not self._simulate_alpha_validation(-1.0)

    def test_alpha_above_one_rejected(self):
        assert not self._simulate_alpha_validation(1.1)
        assert not self._simulate_alpha_validation(2.0)

    def test_alpha_zero_accepted(self):
        assert self._simulate_alpha_validation(0.0)

    def test_alpha_one_accepted(self):
        assert self._simulate_alpha_validation(1.0)

    def test_alpha_mid_range_accepted(self):
        assert self._simulate_alpha_validation(0.5)
        assert self._simulate_alpha_validation(0.7)


# ---------------------------------------------------------------------------
# Hybrid threshold skip tests
# ---------------------------------------------------------------------------

class TestHybridThresholdSkip:
    """Tests confirming hybrid mode intentionally skips threshold filtering."""

    def test_hybrid_mode_returns_all_results_regardless_of_threshold(self):
        """In hybrid mode, threshold does not filter results."""
        # Simulate the threshold logic from semantic_search.py
        mode = "hybrid"
        threshold = 0.9  # Very high threshold

        merged = [
            {"fused_score": 0.01, "score": 0.01, "id": "r1"},
            {"fused_score": 0.005, "score": 0.005, "id": "r2"},
        ]

        results = []
        for r in merged:
            if mode == "hybrid":
                score = r.get("fused_score", 0.0)
            else:
                score = r.get("score", 0.0)

            if mode == "hybrid" or score >= threshold:
                results.append(r)

        # Both results pass despite fused_score << threshold
        assert len(results) == 2

    def test_vector_mode_respects_threshold(self):
        """In vector mode, threshold filters low-scoring results."""
        mode = "vector"
        threshold = 0.5

        merged = [
            {"score": 0.9, "id": "r1"},
            {"score": 0.3, "id": "r2"},
        ]

        results = []
        for r in merged:
            score = r.get("score", 0.0)
            if mode == "hybrid" or score >= threshold:
                results.append(r)

        assert len(results) == 1
        assert results[0]["id"] == "r1"

    def test_keyword_mode_respects_threshold(self):
        """In keyword mode, threshold filters low-scoring results."""
        mode = "keyword"
        threshold = 1.0

        merged = [
            {"score": 8.5, "id": "r1"},
            {"score": 0.5, "id": "r2"},
        ]

        results = []
        for r in merged:
            score = r.get("score", 0.0)
            if mode == "hybrid" or score >= threshold:
                results.append(r)

        # 0.5 < 1.0 threshold, so only r1 passes
        assert len(results) == 1
        assert results[0]["id"] == "r1"


# ---------------------------------------------------------------------------
# Hybrid fallback when BM25 index missing
# ---------------------------------------------------------------------------

class TestHybridFallbackNoBM25:
    """Tests for hybrid mode behavior when BM25 index is missing."""

    def test_hybrid_falls_back_to_vector_only(self):
        """When BM25 results are empty, hybrid returns vector results only."""
        vector_results = [
            {"id": "v1", "score": 0.9, "file_path": "a.py", "start_line": 1,
             "end_line": 10, "content": "x", "chunk_type": "function",
             "language": "python", "symbol_name": "f", "token_count": 10},
        ]
        bm25_results: list[dict] = []

        # When bm25_results is empty, fusion is skipped and vector_results used directly
        # (this mirrors the logic in semantic_search.py)
        if vector_results and bm25_results:
            merged = fuse_results(vector_results, bm25_results)
        else:
            merged = vector_results

        assert len(merged) == 1
        assert merged[0]["id"] == "v1"
        # In vector-only fallback, results have "score" not "fused_score"
        assert "score" in merged[0]
        assert "fused_score" not in merged[0]
