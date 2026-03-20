"""LanceDB vector store wrapper.

Manages the chunks table in a file-based LanceDB database stored
in .index/lancedb/. Supports add, search, delete, and stats operations.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import lancedb
import pyarrow as pa

from .config import Config, INDEX_DIR_NAME
from .models import Chunk, IndexingError

logger = logging.getLogger(__name__)

LANCEDB_DIR = "lancedb"
TABLE_NAME = "chunks"


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

        Args:
            file_path: Relative file path to remove chunks for.

        Returns:
            Number of chunks deleted (approximate).
        """
        table = self._get_table()
        if table is None:
            return 0

        try:
            # Get count before deletion for logging
            before = table.count_rows()
            table.delete(f'file_path = "{file_path}"')
            after = table.count_rows()
            deleted = before - after
            if deleted > 0:
                logger.debug("Deleted %d chunks for file: %s", deleted, file_path)
            return deleted
        except Exception as exc:
            logger.warning("Failed to delete chunks for %s: %s", file_path, exc)
            return 0

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

        query = table.search(vector).metric("cosine").limit(top_k)

        # Apply language filter if specified
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
            # LanceDB returns _distance (cosine distance), convert to similarity
            distance = row.get("_distance", 0.0)
            score = 1.0 - distance

            output.append({
                "score": round(score, 4),
                "file_path": row.get("file_path", ""),
                "start_line": row.get("start_line", 0),
                "end_line": row.get("end_line", 0),
                "content": row.get("content", ""),
                "chunk_type": row.get("chunk_type", ""),
                "language": row.get("language", ""),
                "symbol_name": row.get("symbol_name", ""),
                "token_count": row.get("token_count", 0),
                "id": row.get("id", ""),
            })

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
            arrow_table = table.to_arrow()
            lang_col = arrow_table.column("language")
            for val in lang_col:
                lang = val.as_py()
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

    def has_index(self) -> bool:
        """Check if the index table exists and has data."""
        table = self._get_table()
        if table is None:
            return False
        return table.count_rows() > 0
