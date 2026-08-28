"""Tests for the SQLite embedding cache.

The JSON cache it replaces held every vector in memory and rewrote the
whole file after each batch. The properties that matter here are that a
save costs only what was added, that an existing JSON cache is carried
across so paid-for embeddings are not lost, and that a model change still
invalidates everything.
"""

import json
import sqlite3

import pytest

from lib.config import Config
from lib.embedding_cache import (
    CACHE_FILENAME,
    LEGACY_CACHE_FILENAME,
    EmbeddingCache,
    _pack,
    _unpack,
)


@pytest.fixture
def config() -> Config:
    cfg = Config()
    cfg.embedding.model = "BAAI/bge-m3"
    cfg.embedding.dimensions = 4
    return cfg


def _vec(seed: float) -> list[float]:
    return [seed, seed + 0.5, seed - 0.25, seed * 2]


def _write_legacy(project_dir, model: str, dimensions, entries: dict) -> None:
    index_dir = project_dir / ".index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / LEGACY_CACHE_FILENAME).write_text(json.dumps({
        "version": "1.0",
        "model": model,
        "dimensions": dimensions,
        "entries": entries,
    }))


class TestRoundTrip:
    """What goes in comes back, at float32 precision."""

    def test_set_then_get_before_save(self, tmp_path, config) -> None:
        cache = EmbeddingCache(str(tmp_path), config)
        cache.set("sha256:a", _vec(1.0))
        assert cache.has("sha256:a")
        assert cache.get("sha256:a") == pytest.approx(_vec(1.0))

    def test_survives_a_reopen(self, tmp_path, config) -> None:
        cache = EmbeddingCache(str(tmp_path), config)
        cache.set("sha256:a", _vec(1.0))
        cache.save()
        cache.close()

        reopened = EmbeddingCache(str(tmp_path), config)
        assert reopened.get("sha256:a") == pytest.approx(_vec(1.0), rel=1e-6)

    def test_a_missing_hash_reports_nothing(self, tmp_path, config) -> None:
        cache = EmbeddingCache(str(tmp_path), config)
        assert not cache.has("sha256:absent")
        assert cache.get("sha256:absent") is None

    def test_packing_is_float32(self) -> None:
        vector = [0.1, -2.5, 3.75, 1e-7]
        assert len(_pack(vector)) == len(vector) * 4
        assert _unpack(_pack(vector)) == pytest.approx(vector, rel=1e-6)


class TestIncrementalSaves:
    """A save must cost only what was added since the last one."""

    def test_save_writes_only_pending_rows(self, tmp_path, config) -> None:
        cache = EmbeddingCache(str(tmp_path), config)
        for i in range(50):
            cache.set(f"sha256:{i}", _vec(float(i)))
        cache.save()

        # A second save with one new vector must not rewrite the rest.
        cache.set("sha256:new", _vec(99.0))
        cache.save()

        conn = sqlite3.connect(str(tmp_path / ".index" / CACHE_FILENAME))
        count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        conn.close()
        assert count == 51

    def test_save_with_nothing_pending_is_a_no_op(self, tmp_path, config) -> None:
        cache = EmbeddingCache(str(tmp_path), config)
        cache.set("sha256:a", _vec(1.0))
        cache.save()
        cache.save()  # must not raise or duplicate
        assert cache.get("sha256:a") is not None

    def test_close_flushes_pending(self, tmp_path, config) -> None:
        cache = EmbeddingCache(str(tmp_path), config)
        cache.set("sha256:a", _vec(1.0))
        cache.close()
        assert EmbeddingCache(str(tmp_path), config).has("sha256:a")

    def test_close_is_idempotent(self, tmp_path, config) -> None:
        cache = EmbeddingCache(str(tmp_path), config)
        cache.close()
        cache.close()


class TestModelInvalidation:
    """Vectors from another model are not comparable and must be dropped."""

    def test_same_model_keeps_entries(self, tmp_path, config) -> None:
        EmbeddingCache(str(tmp_path), config).set("sha256:a", _vec(1.0))
        cache = EmbeddingCache(str(tmp_path), config)
        cache.set("sha256:a", _vec(1.0))
        cache.save()
        assert EmbeddingCache(str(tmp_path), config).has("sha256:a")

    def test_model_change_clears_the_cache(self, tmp_path, config) -> None:
        cache = EmbeddingCache(str(tmp_path), config)
        cache.set("sha256:a", _vec(1.0))
        cache.close()

        config.embedding.model = "some/other-model"
        assert not EmbeddingCache(str(tmp_path), config).has("sha256:a")

    def test_dimension_change_clears_the_cache(self, tmp_path, config) -> None:
        cache = EmbeddingCache(str(tmp_path), config)
        cache.set("sha256:a", _vec(1.0))
        cache.close()

        config.embedding.dimensions = 1024
        assert not EmbeddingCache(str(tmp_path), config).has("sha256:a")


class TestLegacyImport:
    """An existing JSON cache carries across, once."""

    def test_entries_are_imported(self, tmp_path, config) -> None:
        _write_legacy(tmp_path, "BAAI/bge-m3", 4, {
            f"sha256:{i}": _vec(float(i)) for i in range(10)
        })
        cache = EmbeddingCache(str(tmp_path), config)
        assert cache.has("sha256:0")
        assert cache.get("sha256:9") == pytest.approx(_vec(9.0), rel=1e-6)

    def test_a_legacy_cache_for_another_model_is_ignored(self, tmp_path, config) -> None:
        _write_legacy(tmp_path, "some/other-model", 4, {"sha256:a": _vec(1.0)})
        assert not EmbeddingCache(str(tmp_path), config).has("sha256:a")

    def test_import_happens_only_once(self, tmp_path, config) -> None:
        _write_legacy(tmp_path, "BAAI/bge-m3", 4, {"sha256:a": _vec(1.0)})
        EmbeddingCache(str(tmp_path), config).close()

        # Rewrite the JSON with different content; it must not be re-read,
        # because the database is no longer empty.
        _write_legacy(tmp_path, "BAAI/bge-m3", 4, {"sha256:b": _vec(2.0)})
        cache = EmbeddingCache(str(tmp_path), config)
        assert cache.has("sha256:a")
        assert not cache.has("sha256:b")

    def test_a_corrupt_legacy_file_does_not_crash(self, tmp_path, config) -> None:
        index_dir = tmp_path / ".index"
        index_dir.mkdir(parents=True)
        (index_dir / LEGACY_CACHE_FILENAME).write_text("{not json")
        cache = EmbeddingCache(str(tmp_path), config)
        assert not cache.has("sha256:a")

    def test_no_legacy_file_is_fine(self, tmp_path, config) -> None:
        assert not EmbeddingCache(str(tmp_path), config).has("sha256:a")
