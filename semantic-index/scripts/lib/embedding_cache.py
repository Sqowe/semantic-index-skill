"""On-disk cache mapping content hashes to embedding vectors.

Kept in SQLite rather than a JSON file. The JSON version held every
vector in memory and rewrote the whole file after each batch, which on a
large project meant loading a multi-gigabyte document at startup and
serialising it hundreds of times during a build — measured on one
23,000-file project, 30 seconds of writing for every 18 seconds of real
work, and growing as the cache grew.

SQLite makes both costs proportional to what actually changed: lookups
hit an index instead of a dictionary held in memory, and a save appends
only the vectors added since the last one. An existing JSON cache is
imported once on first use, so the embeddings already paid for are kept.

Vectors are stored as raw little-endian float32. That is the precision
LanceDB keeps in the index itself, so nothing is lost that would have
survived anyway, and it halves the file against float64.
"""

import json
import logging
import sqlite3
import struct
from pathlib import Path
from typing import Optional

from .config import Config, INDEX_DIR_NAME

logger = logging.getLogger(__name__)

CACHE_FILENAME = "embedding_cache.db"
LEGACY_CACHE_FILENAME = "embedding_cache.json"

# Rows written per transaction when importing a legacy JSON cache.
_IMPORT_BATCH = 2000


def _pack(vector: list[float]) -> bytes:
    """Pack a vector into little-endian float32 bytes."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    """Unpack little-endian float32 bytes back into a vector."""
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class EmbeddingCache:
    """Content hash → embedding vector, persisted in SQLite.

    The cache is dropped and rebuilt if the model or the dimension count
    changes, because vectors from a different model are not comparable
    with the ones already in the index.
    """

    def __init__(self, project_dir: str, config: Config) -> None:
        self._dir = Path(project_dir) / INDEX_DIR_NAME
        self._path = self._dir / CACHE_FILENAME
        self._legacy_path = self._dir / LEGACY_CACHE_FILENAME
        self._model = config.embedding.model
        self._dimensions = config.embedding.dimensions
        # Vectors added since the last save. Kept small — save() drains it.
        self._pending: dict[str, list[float]] = {}
        self._conn: Optional[sqlite3.Connection] = None
        self._open()

    # -- setup ------------------------------------------------------------

    def _open(self) -> None:
        """Open the database, creating or resetting it as needed."""
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self._path))
        except sqlite3.Error as exc:
            logger.warning("Embedding cache unavailable (%s); running without it", exc)
            self._conn = None
            return

        # WAL keeps readers working during a write and survives a crash
        # mid-build without corrupting what was already stored.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "  content_hash TEXT PRIMARY KEY,"
            "  vector BLOB NOT NULL"
            ")"
        )
        self._conn.commit()

        if not self._model_matches():
            self._reset()
        if self._count() == 0 and self._legacy_path.exists():
            self._import_legacy()

        logger.info("Embedding cache holds %d vectors", self._count())

    def _model_matches(self) -> bool:
        """Whether the stored model and dimensions match the current config.

        A database with no stamp counts as not matching, so a fresh one
        goes through :meth:`_reset` and gets stamped. Without that the
        stamp would never be written and a later model change would go
        unnoticed.
        """
        assert self._conn is not None
        rows = dict(self._conn.execute("SELECT key, value FROM meta").fetchall())
        if not rows:
            return False
        stored_dims = rows.get("dimensions")
        current_dims = "" if self._dimensions is None else str(self._dimensions)
        if rows.get("model") == self._model and stored_dims == current_dims:
            return True
        logger.info(
            "Cache model/dimensions mismatch (cached: %s/%s, current: %s/%s), clearing",
            rows.get("model"), stored_dims or None, self._model, self._dimensions,
        )
        return False

    def _reset(self) -> None:
        """Drop every cached vector and stamp the current model."""
        assert self._conn is not None
        self._conn.execute("DELETE FROM embeddings")
        self._conn.execute("DELETE FROM meta")
        self._conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("model", self._model),
                ("dimensions", "" if self._dimensions is None else str(self._dimensions)),
                ("version", "2.0"),
            ],
        )
        self._conn.commit()

    def _count(self) -> int:
        assert self._conn is not None
        return self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    # -- legacy import ----------------------------------------------------

    def _import_legacy(self) -> None:
        """Bring across an ``embedding_cache.json`` written by an older build.

        Read once and then never again — the JSON file is left in place
        but stops being consulted, so it can be deleted whenever the
        owner is satisfied the database is good.

        The whole document has to be parsed at once (adding a streaming
        JSON parser would mean a new dependency, which this project
        deliberately avoids), but entries are drained into SQLite in
        batches so the peak is the parse itself and not parse plus copy.
        """
        assert self._conn is not None
        size_mb = self._legacy_path.stat().st_size / 1e6
        logger.info(
            "Importing legacy JSON cache (%.0f MB) — one time only", size_mb,
        )
        try:
            raw = json.loads(self._legacy_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, MemoryError) as exc:
            logger.warning(
                "Legacy cache unreadable (%s); continuing with an empty cache", exc,
            )
            return

        if raw.get("model") != self._model or raw.get("dimensions") != self._dimensions:
            logger.info(
                "Legacy cache was built for %s/%s, current is %s/%s; not importing",
                raw.get("model"), raw.get("dimensions"),
                self._model, self._dimensions,
            )
            return

        entries = raw.get("entries") or {}
        imported = 0
        batch: list[tuple[str, bytes]] = []
        while entries:
            content_hash, vector = entries.popitem()
            try:
                batch.append((content_hash, _pack(vector)))
            except (struct.error, TypeError):
                continue
            if len(batch) >= _IMPORT_BATCH:
                self._write(batch)
                imported += len(batch)
                batch = []
        if batch:
            self._write(batch)
            imported += len(batch)
        logger.info("Imported %d embeddings from the legacy cache", imported)

    def _write(self, rows: list[tuple[str, bytes]]) -> None:
        """Insert or replace a batch of rows in one transaction."""
        assert self._conn is not None
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (content_hash, vector) VALUES (?, ?)",
            rows,
        )
        self._conn.commit()

    # -- public API -------------------------------------------------------

    def has(self, content_hash: str) -> bool:
        """Whether a vector is cached for *content_hash*."""
        if content_hash in self._pending:
            return True
        if self._conn is None:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM embeddings WHERE content_hash = ?", (content_hash,),
        ).fetchone()
        return row is not None

    def get(self, content_hash: str) -> Optional[list[float]]:
        """Return the cached vector, or None if it is not held."""
        pending = self._pending.get(content_hash)
        if pending is not None:
            return pending
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT vector FROM embeddings WHERE content_hash = ?", (content_hash,),
        ).fetchone()
        return _unpack(row[0]) if row else None

    def set(self, content_hash: str, vector: list[float]) -> None:
        """Record a vector, to be persisted by the next :meth:`save`."""
        self._pending[content_hash] = vector

    def save(self) -> None:
        """Persist vectors added since the last call.

        Unlike the JSON cache this writes only what is new, so the cost
        is proportional to the batch rather than to the size of the whole
        cache.
        """
        if not self._pending or self._conn is None:
            self._pending.clear()
            return
        rows = [(h, _pack(v)) for h, v in self._pending.items()]
        added = len(rows)
        self._write(rows)
        self._pending.clear()
        logger.info("Cached %d new embeddings (%d total)", added, self._count())

    def close(self) -> None:
        """Flush and close. Safe to call more than once."""
        if self._conn is None:
            return
        self.save()
        self._conn.close()
        self._conn = None
