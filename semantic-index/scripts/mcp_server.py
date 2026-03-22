#!/usr/bin/env python3
"""MCP server for semantic-index: exposes indexing, search, and status as MCP tools.

This is an alternative transport for the same functionality provided by the
CLI scripts (build_index.py, semantic_search.py, index_status.py). The SKILL
remains the primary interface; this MCP server is optional.

Usage:
    python mcp_server.py                    # stdio transport (default)
    python mcp_server.py --transport http   # streamable HTTP transport

Requires: pip install -r requirements-mcp.txt
"""

import asyncio
import json
import logging
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from lib.bm25 import BM25Index
from lib.chunker import chunk_file
from lib.config import INDEX_DIR_NAME, ensure_index_dir, load_config
from lib.embedder import Embedder
from lib.fusion import fuse_results
from lib.hasher import detect_changes, update_manifest
from lib.models import ConfigError, EmbeddingError, IndexingError, SemanticIndexError
from lib.store import VectorStore

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP("semantic_index_mcp", json_response=True)

DEFAULT_FILE_BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------


class BuildIndexInput(BaseModel):
    """Input parameters for building/rebuilding the semantic index."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project_dir: str = Field(
        ...,
        description="Absolute path to the project root directory to index.",
        min_length=1,
    )
    config_path: Optional[str] = Field(
        default=None,
        description="Path to config.json. Defaults to <project_dir>/.index/config.json.",
    )
    full_reindex: bool = Field(
        default=False,
        description="Force full re-index, ignoring the file manifest.",
    )
    batch_size: int = Field(
        default=DEFAULT_FILE_BATCH_SIZE,
        description="Number of files to process per batch. Smaller = less memory.",
        ge=1,
        le=500,
    )


class SearchMode(str, Enum):
    """Search mode selection."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SearchIndexInput(BaseModel):
    """Input parameters for semantic search."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project_dir: str = Field(
        ...,
        description="Absolute path to the project root directory.",
        min_length=1,
    )
    query: str = Field(
        ...,
        description="Natural language search query (e.g., 'how does authentication work?').",
        min_length=1,
        max_length=2000,
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Max results to return. Defaults to config value (usually 10).",
        ge=1,
        le=100,
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Min similarity score 0.0-1.0. Ignored in hybrid mode.",
        ge=0.0,
        le=1.0,
    )
    filter_lang: Optional[str] = Field(
        default=None,
        description="Filter results by language (e.g., 'python', 'typescript').",
    )
    filter_path: Optional[str] = Field(
        default=None,
        description="Filter results by file path glob (e.g., 'src/**').",
    )
    mode: Optional[SearchMode] = Field(
        default=None,
        description="Search mode: vector, keyword, or hybrid. Defaults to config value.",
    )
    alpha: Optional[float] = Field(
        default=None,
        description="Hybrid alpha: 0.0 = pure keyword, 1.0 = pure vector.",
        ge=0.0,
        le=1.0,
    )
    rerank: Optional[bool] = Field(
        default=None,
        description=(
            "Re-rank results using a cross-encoder model (requires HuggingFace deps). "
            "True to force on, False to force off, null to use config default."
        ),
    )


class IndexStatusInput(BaseModel):
    """Input parameters for checking index health."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project_dir: str = Field(
        ...,
        description="Absolute path to the project root directory.",
        min_length=1,
    )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _batched(items: list, size: int):
    """Yield successive batches of *size* items from *items*."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _error_response(error: Exception) -> dict[str, Any]:
    """Format an error as a JSON-serialisable dict."""
    return {
        "status": "error",
        "error": str(error),
        "error_type": type(error).__name__,
    }


# ---------------------------------------------------------------------------
# Blocking implementations (run via asyncio.to_thread from async tools)
# ---------------------------------------------------------------------------


def _build_index_sync(params: BuildIndexInput) -> dict[str, Any]:
    """Synchronous implementation of index building."""
    start_time = time.time()

    try:
        config = load_config(params.project_dir, params.config_path)
        ensure_index_dir(params.project_dir)
    except ConfigError as exc:
        return _error_response(exc)

    try:
        changes = detect_changes(
            params.project_dir, config, force_full=params.full_reindex
        )

        store = VectorStore(params.project_dir, config)
        bm25 = BM25Index(params.project_dir)

        # Bootstrap BM25 from vector store if missing
        bm25_loaded = bm25.load()
        bm25_bootstrapped = False
        if not bm25_loaded and store.has_index():
            logger.info("BM25 index missing — bootstrapping from vector store")
            for chunk_batch in store.iter_all_chunks(batch_size=500):
                bm25.add_chunks(chunk_batch)
            bm25_bootstrapped = True

        if not changes.to_index and not changes.to_delete:
            if bm25_bootstrapped:
                bm25.save()
            return {
                "status": "up_to_date",
                "message": "No changes detected",
                "files_unchanged": changes.unchanged,
                "bm25_bootstrapped": bm25_bootstrapped,
            }

        embedder = Embedder(config, project_dir=params.project_dir)

        total_chunks_created = 0
        total_api_calls = 0
        chunk_counts: dict[str, int] = {}
        file_batch_size = max(1, params.batch_size)

        for file_batch in _batched(changes.to_index, file_batch_size):
            batch_chunks = []
            for file_path in file_batch:
                chunks = chunk_file(file_path, params.project_dir, config)
                chunk_counts[file_path] = len(chunks)
                batch_chunks.extend(chunks)

            batch_api_calls = 0
            if batch_chunks:
                # Embed first — this is the expensive/fallible step.
                # If embedding fails, old index data is preserved intact.
                batch_api_calls = embedder.embed_chunks(batch_chunks)

            # Only mutate the store after embedding succeeds.
            # Delete old + add new as a tight pair.
            if batch_chunks:
                bm25_records = [
                    {
                        "id": c.id,
                        "content": c.content,
                        "file_path": c.file_path,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "chunk_type": c.chunk_type.value,
                        "language": c.language or "",
                        "symbol_name": c.symbol_name or "",
                        "token_count": c.token_count,
                    }
                    for c in batch_chunks
                ]
                for file_path in file_batch:
                    store.delete_by_file(file_path)
                    bm25.delete_by_file(file_path)
                store.add(batch_chunks)
                bm25.add_chunks(bm25_records)
            else:
                for file_path in file_batch:
                    store.delete_by_file(file_path)
                    bm25.delete_by_file(file_path)

            total_chunks_created += len(batch_chunks)
            total_api_calls += batch_api_calls

        # Handle deletions
        for file_path in changes.to_delete:
            store.delete_by_file(file_path)
            bm25.delete_by_file(file_path)

        bm25.save()
        update_manifest(params.project_dir, changes, chunk_counts)

        duration = time.time() - start_time
        return {
            "status": "success",
            "files_indexed": len(changes.to_index),
            "files_skipped": changes.unchanged,
            "files_deleted": len(changes.to_delete),
            "chunks_created": total_chunks_created,
            "duration_seconds": round(duration, 1),
            "embedding_api_calls": total_api_calls,
        }

    except (ConfigError, EmbeddingError, IndexingError, SemanticIndexError) as exc:
        return _error_response(exc)
    except Exception as exc:
        logger.exception("Unexpected error during indexing")
        return _error_response(exc)


def _search_index_sync(params: SearchIndexInput) -> dict[str, Any]:
    """Synchronous implementation of semantic search."""
    try:
        config = load_config(params.project_dir)
    except ConfigError as exc:
        return _error_response(exc)

    top_k = params.top_k or config.search.default_top_k
    threshold = (
        params.threshold if params.threshold is not None else config.search.default_threshold
    )
    mode = params.mode.value if params.mode else config.search.mode
    alpha = params.alpha if params.alpha is not None else config.search.hybrid_alpha

    try:
        store = VectorStore(params.project_dir, config)
        if not store.has_index() and mode in ("vector", "hybrid"):
            return _error_response(
                SemanticIndexError(
                    "No index found. Run semantic_index_build first to create the index."
                )
            )

        start_time = time.time()

        filters: Optional[dict[str, str]] = None
        if params.filter_lang or params.filter_path:
            filters = {}
            if params.filter_lang:
                filters["language"] = params.filter_lang
            if params.filter_path:
                filters["file_path_glob"] = params.filter_path

        vector_results: list[dict] = []
        bm25_results: list[dict] = []

        # Vector search
        if mode in ("vector", "hybrid"):
            embedder = Embedder(config)
            query_vector = embedder.embed_query(params.query)
            vector_results = store.search(
                vector=query_vector,
                top_k=top_k * 2,
                filters=filters,
            )

        # BM25 keyword search
        if mode in ("keyword", "hybrid"):
            bm25 = BM25Index(params.project_dir)
            if bm25.load():
                bm25_results = bm25.search(
                    query=params.query,
                    top_k=top_k * 2,
                    filters=filters,
                )
            elif mode == "keyword":
                return _error_response(
                    SemanticIndexError(
                        "No BM25 index found. Run semantic_index_build first."
                    )
                )
            else:
                logger.warning("No BM25 index found, falling back to vector-only")

        # Merge results
        if mode == "hybrid" and vector_results and bm25_results:
            merged = fuse_results(
                vector_results=vector_results,
                bm25_results=bm25_results,
                alpha=alpha,
            )
        elif mode == "keyword":
            merged = bm25_results
        else:
            merged = vector_results

        # Apply threshold and truncate
        results: list[dict] = []
        for r in merged:
            score = (
                r.get("fused_score", 0.0) if mode == "hybrid" else r.get("score", 0.0)
            )
            if mode == "hybrid" or score >= threshold:
                results.append(r)
            if len(results) >= top_k:
                break

        # Optional reranking — explicit param overrides config default
        rerank_enabled = (
            params.rerank if params.rerank is not None else config.search.rerank_enabled
        )
        if rerank_enabled and results:
            try:
                from lib.reranker import Reranker

                reranker = Reranker(
                    model_name=config.search.rerank_model,
                    device=config.embedding.device,
                    trust_remote_code=config.embedding.trust_remote_code,
                )
                results = reranker.rerank(
                    params.query, results, top_n=config.search.rerank_top_n
                )
            except EmbeddingError as exc:
                logger.warning("Reranking unavailable, skipping: %s", exc)

        duration_ms = (time.time() - start_time) * 1000

        return {
            "query": params.query,
            "mode": mode,
            "results": [
                {
                    "rank": i + 1,
                    "score": r.get(
                        "rerank_score", r.get("fused_score", r.get("score", 0.0))
                    ),
                    "file_path": r["file_path"],
                    "start_line": r["start_line"],
                    "end_line": r["end_line"],
                    "chunk_type": r["chunk_type"],
                    "symbol_name": r["symbol_name"],
                    "language": r["language"],
                    "content": r["content"],
                    **(
                        {"rerank_score": r["rerank_score"]}
                        if "rerank_score" in r
                        else {}
                    ),
                    **(
                        {
                            "vector_score": r["vector_score"],
                            "bm25_score": r["bm25_score"],
                        }
                        if mode == "hybrid" and "vector_score" in r
                        else {}
                    ),
                }
                for i, r in enumerate(results)
            ],
            "total_results": len(results),
            "search_duration_ms": round(duration_ms),
        }

    except (EmbeddingError, SemanticIndexError) as exc:
        return _error_response(exc)
    except Exception as exc:
        logger.exception("Unexpected error during search")
        return _error_response(exc)


def _index_status_sync(params: IndexStatusInput) -> dict[str, Any]:
    """Synchronous implementation of index status check."""
    try:
        config = load_config(params.project_dir)
    except ConfigError as exc:
        return _error_response(exc)

    try:
        index_dir = Path(params.project_dir) / INDEX_DIR_NAME

        if not index_dir.exists():
            return {
                "indexed": False,
                "message": "No .index/ directory found. Run semantic_index_build first.",
            }

        # Load manifest
        manifest_path = index_dir / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read manifest: %s", exc)

        manifest_files = manifest.get("files", {})
        last_indexed = manifest.get("last_indexed", "never")

        # Store stats
        store = VectorStore(params.project_dir, config)
        stats = store.get_stats()

        # Stale file detection
        changes = detect_changes(params.project_dir, config)
        stale_files = len(changes.to_index)

        # Index size on disk
        total_size = 0
        for f in index_dir.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size

        return {
            "indexed": store.has_index(),
            "total_files": len(manifest_files),
            "total_chunks": stats["total_chunks"],
            "last_indexed": last_indexed,
            "stale_files": stale_files,
            "embedding_model": config.embedding.model,
            "embedding_dimensions": config.embedding.dimensions,
            "index_size_mb": round(total_size / (1024 * 1024), 1),
            "languages": stats.get("languages", {}),
        }

    except SemanticIndexError as exc:
        return _error_response(exc)
    except Exception as exc:
        logger.exception("Unexpected error reading index status")
        return _error_response(exc)


# ---------------------------------------------------------------------------
# MCP Tools (thin async wrappers that offload blocking work to a thread)
# ---------------------------------------------------------------------------


@mcp.tool(
    name="semantic_index_build",
    annotations={
        "title": "Build Semantic Index",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def semantic_index_build(params: BuildIndexInput) -> dict[str, Any]:
    """Build or incrementally update the semantic index for a project.

    Scans the project for supported files (code, markdown, DITA, office docs),
    detects changes via SHA-256 manifest, chunks files using AST-aware
    splitting, embeds chunks via the configured provider, and stores
    embeddings in a local LanceDB vector store.

    On re-run, only changed/new files are re-indexed (incremental).
    Use full_reindex=True to force a complete rebuild.

    Args:
        params: Validated BuildIndexInput with project_dir, config_path,
            full_reindex, and batch_size.

    Returns:
        dict with keys: status, files_indexed, files_skipped, files_deleted,
        chunks_created, duration_seconds, embedding_api_calls.
        On error: status="error", error, error_type.
    """
    return await asyncio.to_thread(_build_index_sync, params)


@mcp.tool(
    name="semantic_index_search",
    annotations={
        "title": "Semantic Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def semantic_index_search(params: SearchIndexInput) -> dict[str, Any]:
    """Search the semantic index by meaning using natural language queries.

    Supports three modes: vector (embedding similarity), keyword (BM25),
    and hybrid (Reciprocal Rank Fusion of both). Optionally re-ranks
    results with a cross-encoder model for higher precision.

    Args:
        params: Validated SearchIndexInput with project_dir, query,
            and optional top_k, threshold, filter_lang, filter_path,
            mode, alpha, rerank.

    Returns:
        dict with keys: query, mode, results (list of ranked hits),
        total_results, search_duration_ms.
        Each result has: rank, score, file_path, start_line, end_line,
        chunk_type, symbol_name, language, content.
        On error: status="error", error, error_type.
    """
    return await asyncio.to_thread(_search_index_sync, params)


@mcp.tool(
    name="semantic_index_status",
    annotations={
        "title": "Index Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def semantic_index_status(params: IndexStatusInput) -> dict[str, Any]:
    """Check the health and statistics of the semantic index.

    Reports total files indexed, chunk count, last index time, stale
    files, embedding model, index size, and language breakdown.

    Args:
        params: Validated IndexStatusInput with project_dir.

    Returns:
        dict with keys: indexed, total_files, total_chunks, last_indexed,
        stale_files, embedding_model, embedding_dimensions, index_size_mb,
        languages.
        On error: status="error", error, error_type.
    """
    return await asyncio.to_thread(_index_status_sync, params)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
