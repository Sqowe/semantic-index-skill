"""Removing many files must cost one pass, not one pass per file.

Both indexes had a delete whose work was the size of the whole index
rather than of the removal, called once per file. On a 200,000-document
index, clearing 1,672 files took half an hour.
"""

import pytest

from lib.bm25 import BM25Index
from lib.store import _batched, _sql_string


def _chunk(doc_id: str, file_path: str, content: str) -> dict:
    return {
        "id": doc_id,
        "content": content,
        "file_path": file_path,
        "start_line": 1,
        "end_line": 5,
        "chunk_type": "unknown",
        "language": "python",
        "symbol_name": "",
        "token_count": 10,
    }


@pytest.fixture
def index(tmp_path) -> BM25Index:
    """A BM25 index over six files, each with two distinctive chunks."""
    idx = BM25Index(str(tmp_path))
    idx.build([
        _chunk(f"d{f}{c}", f"src/file{f}.py", f"alpha beta file{f} chunk{c} gamma")
        for f in range(6) for c in range(2)
    ])
    return idx


class TestBm25BulkDelete:
    """delete_by_files removes exactly the named files, in one sweep."""

    def test_removes_every_named_file(self, index) -> None:
        index.delete_by_files(["src/file0.py", "src/file1.py", "src/file2.py"])
        remaining = {d["file_path"] for d in index._docs.values()}
        assert remaining == {"src/file3.py", "src/file4.py", "src/file5.py"}

    def test_leaves_other_files_alone(self, index) -> None:
        before = len([d for d in index._docs.values()
                      if d["file_path"] == "src/file5.py"])
        index.delete_by_files(["src/file0.py"])
        after = len([d for d in index._docs.values()
                     if d["file_path"] == "src/file5.py"])
        assert before == after == 2

    def test_postings_lose_the_deleted_docs(self, index) -> None:
        index.delete_by_files(["src/file0.py"])
        for term, postings in index._postings.items():
            assert not any(d.startswith("d0") for d in postings), term

    def test_terms_only_that_file_had_are_dropped(self, index) -> None:
        assert "file0" in index._postings
        index.delete_by_files(["src/file0.py"])
        assert "file0" not in index._postings
        assert "alpha" in index._postings, "shared terms must survive"

    def test_statistics_are_recomputed(self, index) -> None:
        index.delete_by_files(["src/file0.py", "src/file1.py"])
        assert index._n_docs == len(index._docs) == 8
        assert index._avg_dl > 0

    def test_deleting_everything_leaves_a_valid_index(self, index) -> None:
        index.delete_by_files([f"src/file{i}.py" for i in range(6)])
        assert index._n_docs == 0
        assert index._postings == {}
        assert index._avg_dl == 0.0

    def test_an_empty_request_is_a_no_op(self, index) -> None:
        index.delete_by_files([])
        assert index._n_docs == 12

    def test_unknown_paths_are_ignored(self, index) -> None:
        index.delete_by_files(["src/never_indexed.py"])
        assert index._n_docs == 12

    def test_single_file_helper_still_works(self, index) -> None:
        index.delete_by_file("src/file0.py")
        assert {d["file_path"] for d in index._docs.values()} == {
            f"src/file{i}.py" for i in range(1, 6)
        }

    def test_search_after_deletion_never_returns_removed_files(self, index) -> None:
        index.delete_by_files(["src/file0.py", "src/file3.py"])
        hits = index.search("alpha beta gamma", top_k=20)
        paths = {h["file_path"] for h in hits}
        assert "src/file0.py" not in paths
        assert "src/file3.py" not in paths
        assert paths


class TestDeletePredicate:
    """Paths are data; they must not be able to change the predicate."""

    @pytest.mark.parametrize("path,expected", [
        ("src/a.py", "'src/a.py'"),
        ("it's/a.py", "'it''s/a.py'"),
        ("a'; DROP TABLE chunks; --", "'a''; DROP TABLE chunks; --'"),
        ('with"double.py', "'with\"double.py'"),
    ])
    def test_quoting(self, path: str, expected: str) -> None:
        assert _sql_string(path) == expected

    def test_batching_covers_every_item_once(self) -> None:
        items = [f"f{i}" for i in range(450)]
        groups = list(_batched(items, 200))
        assert [len(g) for g in groups] == [200, 200, 50]
        assert [x for g in groups for x in g] == items

    def test_batching_an_empty_list_yields_nothing(self) -> None:
        assert list(_batched([], 200)) == []


class TestStoreCompaction:
    """Compaction must be safe to call, and never lose rows."""

    def _store(self, tmp_path):
        from lib.config import Config
        from lib.models import Chunk, ChunkType
        from lib.store import VectorStore

        config = Config()
        config.embedding.dimensions = 4
        store = VectorStore(str(tmp_path), config)
        chunks = []
        for i in range(20):
            chunk = Chunk(
                id=f"sha256:{i}",
                file_path=f"src/file{i % 5}.py",
                start_line=1,
                end_line=3,
                content=f"body {i}",
                chunk_type=ChunkType.FUNCTION,
                language="python",
                token_count=5,
            )
            chunk.metadata["vector"] = [float(i), 0.5, 0.25, 1.0]
            chunks.append(chunk)
        store.add(chunks)
        return store

    def test_compaction_preserves_every_row(self, tmp_path) -> None:
        store = self._store(tmp_path)
        before = store._get_table().count_rows()
        store.compact()
        assert store._get_table().count_rows() == before

    def test_compaction_after_a_bulk_delete(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.delete_by_files(["src/file0.py", "src/file1.py"])
        remaining = store._get_table().count_rows()
        store.compact()
        assert store._get_table().count_rows() == remaining
        paths = set(store._get_table().search().select(["file_path"])
                    .limit(100).to_arrow().column("file_path").to_pylist())
        assert paths == {"src/file2.py", "src/file3.py", "src/file4.py"}

    def test_compaction_on_a_missing_table_is_a_no_op(self, tmp_path) -> None:
        from lib.config import Config
        from lib.store import VectorStore

        config = Config()
        config.embedding.dimensions = 4
        VectorStore(str(tmp_path), config).compact()  # must not raise

    def test_a_compaction_failure_does_not_propagate(self, tmp_path, monkeypatch) -> None:
        """A build that produced a good index must not fail on cleanup."""
        store = self._store(tmp_path)
        table = store._get_table()
        monkeypatch.setattr(
            type(table), "optimize",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")),
        )
        store.compact()  # must swallow


class TestStoreBulkDelete:
    """delete_by_files removes exactly the named files."""

    def _store(self, tmp_path):
        return TestStoreCompaction()._store(tmp_path)

    def test_removes_only_the_named_files(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.delete_by_files(["src/file0.py", "src/file3.py"])
        paths = set(store._get_table().search().select(["file_path"])
                    .limit(100).to_arrow().column("file_path").to_pylist())
        assert paths == {"src/file1.py", "src/file2.py", "src/file4.py"}

    def test_an_empty_list_deletes_nothing(self, tmp_path) -> None:
        store = self._store(tmp_path)
        before = store._get_table().count_rows()
        assert store.delete_by_files([]) == 0
        assert store._get_table().count_rows() == before

    def test_duplicate_paths_are_collapsed(self, tmp_path) -> None:
        store = self._store(tmp_path)
        deleted = store.delete_by_files(["src/file0.py"] * 5)
        assert deleted == 4

    def test_a_path_with_a_quote_is_handled(self, tmp_path) -> None:
        """A quote in a path must not change what the predicate means."""
        store = self._store(tmp_path)
        before = store._get_table().count_rows()
        store.delete_by_files(["src/it's.py"])
        assert store._get_table().count_rows() == before
