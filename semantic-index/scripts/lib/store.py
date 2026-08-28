"""LanceDB vector store wrapper.

Manages the chunks table in a file-based LanceDB database stored
in .index/lancedb/. Supports add, search, delete, and stats operations.
"""

import logging
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import lancedb
import pyarrow as pa

from .config import Config, INDEX_DIR_NAME
from .constants import path_matches_glob
from .models import Chunk, IndexingError

logger = logging.getLogger(__name__)

LANCEDB_DIR = "lancedb"
TABLE_NAME = "chunks"

# Over-fetch bounds for the file_path_glob post-filter. A glob can exclude
# most candidates, so we fetch more than top_k and filter in Python. The
# floor guarantees a useful pool for tiny top_k; the cap bounds memory/latency.
_GLOB_OVERFETCH_FLOOR = 200
_GLOB_OVERFETCH_CAP = 2000

# How many file paths go into a single delete predicate. Large enough that
# the per-transaction cost is amortised away, small enough that the engine
# is never handed a predicate with tens of thousands of terms.
_DELETE_BATCH = 200

# How much version history a build leaves behind. Anything older is
# dropped at the end of a build; a window rather than zero so a search
# running at that moment is not pulled out from under. Searches take
# milliseconds, so minutes is already generous — and the window has to
# stay well below how often builds run, or a build's own history is
# always too young to clean and the table only ever grows.
DEFAULT_VERSION_RETENTION = timedelta(minutes=10)


def _sql_string(value: str) -> str:
    """Quote *value* as a SQL string literal, escaping embedded quotes.

    File paths are data, not code. A path containing an apostrophe would
    otherwise end the literal early and change the meaning of the
    predicate — on a delete, that is rows disappearing that should not.
    """
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _batched(items: list[str], size: int):
    """Yield *items* in consecutive lists of at most *size*."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _build_schema(embedding_dim: int) -> pa.Schema:
    """Build the PyArrow schema for the chunks table."""
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("start_line", pa.int32()),
        pa.field("end_line", pa.int32()),
        pa.field("content", pa.string()),
        pa.field("chunk_type", pa.string()),
        pa.field("language", pa.string()),
        pa.field("symbol_name", pa.string()),
        pa.field("token_count", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), list_size=embedding_dim)),
    ])


class VectorStore:
    """LanceDB-backed vector store for chunk embeddings."""

    def __init__(self, project_dir: str, config: Config) -> None:
        self._project_dir = project_dir
        self._config = config
        self._dim = config.embedding.dimensions
        self._db_path = os.path.join(project_dir, INDEX_DIR_NAME, LANCEDB_DIR)
        self._db: Optional[lancedb.DBConnection] = None
        self._table = None

    def _get_db(self) -> lancedb.DBConnection:
        """Open or return the LanceDB connection."""
        if self._db is None:
            Path(self._db_path).mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(self._db_path)
        return self._db

    def _get_table(self):
        """Open or return the chunks table. Returns None if it doesn't exist."""
        if self._table is not None:
            return self._table
        db = self._get_db()
        if TABLE_NAME in db.table_names():
            self._table = db.open_table(TABLE_NAME)
        return self._table

    def _ensure_table(self):
        """Get or create the chunks table."""
        table = self._get_table()
        if table is not None:
            return table
        db = self._get_db()
        schema = _build_schema(self._dim)
        self._table = db.create_table(TABLE_NAME, schema=schema)
        logger.info("Created chunks table with %d-dim vectors", self._dim)
        return self._table

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks with their embedding vectors to the store.

        Expects each chunk to have a 'vector' key in metadata
        (set by Embedder.embed_chunks).

        Args:
            chunks: List of Chunk objects with vectors in metadata.

        Raises:
            IndexingError: If any chunk is missing its vector.
        """
        if not chunks:
            return

        table = self._ensure_table()
        records: list[dict[str, Any]] = []

        for chunk in chunks:
            vector = chunk.metadata.get("vector")
            if vector is None:
                raise IndexingError(f"Chunk {chunk.id} has no embedding vector")

            records.append({
                "id": chunk.id,
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content": chunk.content,
                "chunk_type": chunk.chunk_type.value,
                "language": chunk.language or "",
                "symbol_name": chunk.symbol_name or "",
                "token_count": chunk.token_count,
                "vector": vector,
            })

        table.add(records)
        logger.info("Added %d chunks to store", len(records))


    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks belonging to a specific file.

        Prefer :meth:`delete_by_files` when removing more than one file —
        each call here is a separate transaction, and the cost of a
        transaction does not depend on how many rows it removes.

        Args:
            file_path: Relative file path to remove chunks for.

        Returns:
            Number of chunks deleted (approximate).
        """
        return self.delete_by_files([file_path])

    def delete_by_files(self, file_paths: list[str]) -> int:
        """Delete all chunks belonging to any of *file_paths*.

        Every ``table.delete()`` writes a version manifest, a deletion
        file and a transaction record whatever it removes, and each one
        is a durable write into directories that already hold every such
        file from every previous build. Deleting a thousand files one at
        a time therefore costs a thousand transactions: measured on a
        200,000-row table, 224 ms per file against 2.2 ms when the same
        files go out in batches of 100.

        Paths go out in groups rather than one enormous predicate, so a
        very large removal does not build a query string the engine has
        to parse in one piece.

        Args:
            file_paths: Relative file paths to remove chunks for.

        Returns:
            Number of chunks deleted (approximate).
        """
        table = self._get_table()
        if table is None or not file_paths:
            return 0

        before = table.count_rows()
        for group in _batched(list(dict.fromkeys(file_paths)), _DELETE_BATCH):
            predicate = "file_path IN ({})".format(
                ", ".join(_sql_string(path) for path in group)
            )
            try:
                table.delete(predicate)
            except Exception as exc:
                logger.warning(
                    "Failed to delete chunks for %d files (first: %s): %s",
                    len(group), group[0], exc,
                )
        deleted = before - table.count_rows()
        if deleted > 0:
            logger.debug(
                "Deleted %d chunks across %d files", deleted, len(file_paths),
            )
        return deleted

    def search(
        self,
        vector: list[float],
        top_k: int = 20,
        filters: Optional[dict[str, Optional[str]]] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks using cosine similarity.

        Args:
            vector: Query embedding vector.
            top_k: Maximum number of results to return.
            filters: Optional filters (language, file_path_glob).

        Returns:
            List of result dicts with score, file_path, content, etc.
            Sorted by descending similarity score.
        """
        table = self._get_table()
        if table is None:
            return []

        path_glob = filters.get("file_path_glob") if filters else None

        # LanceDB's .where() is SQL and has no glob operator, so file_path_glob
        # is applied as a Python post-filter below. When a glob is active we
        # over-fetch candidates so that, after filtering, top_k can still be
        # filled from the allowed set rather than starved by excluded rows.
        fetch_limit = top_k
        if path_glob:
            fetch_limit = min(max(top_k * 10, _GLOB_OVERFETCH_FLOOR), _GLOB_OVERFETCH_CAP)

        query = table.search(vector).metric("cosine").limit(fetch_limit)

        # Apply language filter if specified (pushed down to LanceDB)
        if filters:
            lang = filters.get("language")
            if lang:
                query = query.where(f'language = "{lang}"')

        try:
            results = query.to_list()
        except Exception as exc:
            logger.warning("Search failed: %s", exc)
            return []

        # Convert LanceDB distance to similarity score (cosine distance → similarity)
        output: list[dict[str, Any]] = []
        for row in results:
            file_path = row.get("file_path", "")

            # Apply file path glob filter (post-filter; see fetch_limit above)
            if path_glob and not path_matches_glob(file_path, path_glob):
                continue

            # LanceDB returns _distance (cosine distance), convert to similarity
            distance = row.get("_distance", 0.0)
            score = 1.0 - distance

            output.append({
                "score": round(score, 4),
                "file_path": file_path,
                "start_line": row.get("start_line", 0),
                "end_line": row.get("end_line", 0),
                "content": row.get("content", ""),
                "chunk_type": row.get("chunk_type", ""),
                "language": row.get("language", ""),
                "symbol_name": row.get("symbol_name", ""),
                "token_count": row.get("token_count", 0),
                "id": row.get("id", ""),
            })

            if len(output) >= top_k:
                break

        return output


    def get_stats(self) -> dict[str, Any]:
        """Get index statistics.

        Returns:
            Dict with total_chunks, languages breakdown, and index size.
        """
        table = self._get_table()
        if table is None:
            return {
                "total_chunks": 0,
                "languages": {},
                "index_size_bytes": 0,
            }

        total = table.count_rows()

        # Get language breakdown using PyArrow (no pandas dependency)
        languages: dict[str, int] = {}
        try:
            arrow_table = table.to_arrow().select(["language"])
            lang_col = arrow_table.column("language")
            for lang in lang_col.to_pylist():
                if lang:
                    languages[lang] = languages.get(lang, 0) + 1
        except Exception as exc:
            logger.warning("Failed to compute language stats: %s", exc)

        # Compute index size on disk
        index_size = 0
        db_path = Path(self._db_path)
        if db_path.exists():
            for f in db_path.rglob("*"):
                if f.is_file():
                    index_size += f.stat().st_size

        return {
            "total_chunks": total,
            "languages": languages,
            "index_size_bytes": index_size,
        }

    def compact(self, retain: timedelta = DEFAULT_VERSION_RETENTION) -> None:
        """Merge small data fragments and drop superseded versions.

        Every write to the table — each batch added, each delete — leaves
        behind a version manifest, a transaction record and, for deletes,
        a deletion file. Nothing removes them, so they accumulate across
        every build the project has ever run: one 23,000-file project had
        83,958 such files, and because each new write reads the manifest
        chain and syncs into those directories, every build made the next
        one slower.

        Measured on one such table: 3.1 GB across 31,947 versions and
        897 data files became 1.9 GB across 2 versions and 446 data
        files, row count unchanged, in 35 seconds. Running it again
        immediately costs nothing and changes nothing, so the price is
        paid once on a neglected table and never again.

        Compaction is best-effort. A failure here leaves a working but
        unoptimised table, which is not worth failing a completed build
        over — it is logged and swallowed.

        Args:
            retain: Versions younger than this are kept, so a concurrent
                reader holding an older snapshot is not pulled out from
                under. History older than this is not recoverable
                afterwards, which is the point.
        """
        table = self._get_table()
        if table is None:
            return

        started = time.monotonic()
        try:
            # ``optimize`` merges fragments and drops old versions in one
            # step. The older ``compact_files`` / ``cleanup_old_versions``
            # pair is deprecated and, unlike this, needs the separate
            # ``pylance`` package that a normal install does not pull in.
            table.optimize(cleanup_older_than=retain)
        except Exception as exc:
            # An unoptimised table still answers every query correctly, so
            # this is not worth failing a completed build over.
            logger.warning("Store compaction skipped: %s", exc)
            return
        logger.info(
            "Compacted the vector store in %.1fs", time.monotonic() - started,
        )

    def has_index(self) -> bool:
        """Check if the index table exists and has data."""
        table = self._get_table()
        if table is None:
            return False
        return table.count_rows() > 0

    def get_all_chunks(self) -> list[dict[str, Any]]:
        """Retrieve all chunks from the store (without vectors).

        Used for bootstrapping the BM25 index from an existing vector store
        when upgrading from a pre-hybrid index. Schema-tolerant: inspects
        available columns and backfills missing fields with defaults.

        Returns:
            List of chunk dicts with id, file_path, content, etc.
            Empty list if no index exists.

        Raises:
            IndexingError: If the store has rows but essential columns
                (id, content, file_path) are missing.
        """
        table = self._get_table()
        if table is None:
            return []

        try:
            arrow_table = table.to_arrow()
        except Exception as exc:
            logger.warning("Failed to read arrow table from store: %s", exc)
            return []

        available = set(arrow_table.column_names)
        required = {"id", "content", "file_path"}
        missing_required = required - available
        if missing_required:
            from .models import IndexingError as IdxErr
            raise IdxErr(
                f"Vector store schema missing essential columns: {missing_required}. "
                "Run 'build_index.py --full' to rebuild the index."
            )

        # Select available columns, skip vector
        desired = [
            "id", "file_path", "start_line", "end_line",
            "content", "chunk_type", "language", "symbol_name", "token_count",
        ]
        select_cols = [c for c in desired if c in available]

        try:
            subset = arrow_table.select(select_cols)
            rows = subset.to_pylist()
        except Exception as exc:
            logger.warning("Failed to select columns from store: %s", exc)
            return []

        # Backfill missing optional fields with defaults
        defaults = {
            "start_line": 0, "end_line": 0, "chunk_type": "unknown",
            "language": "", "symbol_name": "", "token_count": 0,
        }
        for row in rows:
            for field, default in defaults.items():
                if field not in row:
                    row[field] = default

        return rows

    def iter_all_chunks(self, batch_size: int = 500) -> Any:
        """Yield chunks from the store in batches (without vectors).

        Attempts to use LanceDB's scanner API for column projection to
        avoid materializing the full vector column. Falls back to loading
        the full table if the scanner API is unavailable.

        Emits a warning when the table exceeds a size threshold, since
        the fallback path can cause memory spikes on large indexes.

        Args:
            batch_size: Number of rows per batch.

        Yields:
            Lists of chunk dicts, each list up to batch_size items.

        Raises:
            IndexingError: If essential columns are missing.
        """
        table = self._get_table()
        if table is None:
            return

        # Check schema without loading data
        try:
            schema = table.schema
        except Exception as exc:
            logger.warning("Failed to read table schema: %s", exc)
            return

        available = set(schema.names)
        required = {"id", "content", "file_path"}
        missing_required = required - available
        if missing_required:
            raise IndexingError(
                f"Vector store schema missing essential columns: {missing_required}. "
                "Run 'build_index.py --full' to rebuild the index."
            )

        desired = [
            "id", "file_path", "start_line", "end_line",
            "content", "chunk_type", "language", "symbol_name", "token_count",
        ]
        select_cols = [c for c in desired if c in available]

        defaults = {
            "start_line": 0, "end_line": 0, "chunk_type": "unknown",
            "language": "", "symbol_name": "", "token_count": 0,
        }

        # Warn on large tables — the fallback path materializes vectors
        _LARGE_TABLE_THRESHOLD = 50_000
        try:
            row_count = table.count_rows()
            if row_count > _LARGE_TABLE_THRESHOLD:
                logger.warning(
                    "Large index detected (%d rows). Bootstrap may use "
                    "significant memory. Consider running "
                    "'build_index.py --full' instead.",
                    row_count,
                )
        except Exception:
            row_count = None

        # Try scanner-based projection first (avoids loading vectors)
        try:
            scanner = table.scanner(columns=select_cols, batch_size=batch_size)
            for record_batch in scanner.to_batches():
                rows = record_batch.to_pylist()
                for row in rows:
                    for field, default in defaults.items():
                        if field not in row:
                            row[field] = default
                yield rows
            return  # Scanner path succeeded — skip fallback
        except (AttributeError, TypeError):
            # LanceDB version doesn't support scanner() with columns kwarg
            logger.debug("Scanner API unavailable, falling back to to_arrow()")
        except Exception as exc:
            logger.debug("Scanner failed (%s), falling back to to_arrow()", exc)

        # Fallback: load full arrow table, then select non-vector columns.
        # NOTE: table.to_arrow() materialises the entire table (including
        # vectors) briefly, causing a peak memory spike on large indexes.
        try:
            full_arrow = table.to_arrow()
            subset = full_arrow.select(select_cols)
            del full_arrow
        except Exception as exc:
            logger.warning("Failed to read arrow table from store: %s", exc)
            return

        for record_batch in subset.to_batches(max_chunksize=batch_size):
            rows = record_batch.to_pylist()
            for row in rows:
                for field, default in defaults.items():
                    if field not in row:
                        row[field] = default
            yield rows
