#!/usr/bin/env python3
"""Search the semantic index by meaning.

Usage:
    python semantic_search.py --project-dir <path> --query <text>
        [--top-k N] [--threshold F] [--filter-lang <lang>] [--filter-path <glob>]
        [--mode vector|keyword|hybrid] [--alpha F]

Exit codes:
    0  Success
    1  Configuration error
    2  Runtime error (API failure, no index)
"""

import argparse
import json
import logging
import sys
import time

from lib.bm25 import BM25Index
from lib.config import load_config
from lib.embedder import Embedder
from lib.fusion import fuse_results
from lib.models import ConfigError, EmbeddingError, SemanticIndexError
from lib.store import VectorStore

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _handle_error(error: Exception, exit_code: int = 2) -> None:
    """Output error as JSON to stdout and exit."""
    print(json.dumps({
        "status": "error",
        "error": str(error),
        "error_type": type(error).__name__,
    }, indent=2))
    sys.exit(exit_code)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the semantic index by meaning")
    parser.add_argument("--project-dir", required=True, help="Project root directory")
    parser.add_argument("--query", required=True, help="Natural language search query")
    parser.add_argument("--top-k", type=int, default=None, help="Max results to return")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Min similarity score (0.0-1.0). Applies to vector and keyword modes. "
                             "Ignored in hybrid mode (RRF scores are not comparable to cosine similarity).")
    parser.add_argument("--filter-lang", default=None, help="Filter by language (e.g., 'python')")
    parser.add_argument("--filter-path", default=None, help="Filter by file path glob (e.g., 'src/**')")
    parser.add_argument(
        "--mode", default=None,
        choices=["vector", "keyword", "hybrid"],
        help="Search mode: vector, keyword, or hybrid (default from config)",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Hybrid alpha: 0.0 = pure keyword, 1.0 = pure vector (default from config)",
    )
    parser.add_argument(
        "--rerank", action="store_true", default=None,
        help="Re-rank results using a cross-encoder model (requires HuggingFace deps)",
    )
    parser.add_argument(
        "--no-rerank", action="store_true", default=False,
        help="Disable re-ranking even if enabled in config",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.project_dir)
    except ConfigError as exc:
        _handle_error(exc, exit_code=1)

    top_k = args.top_k or config.search.default_top_k
    threshold = args.threshold if args.threshold is not None else config.search.default_threshold
    mode = args.mode or config.search.mode
    alpha = args.alpha if args.alpha is not None else config.search.hybrid_alpha

    # Validate alpha (config-loaded values are checked by _validate_config,
    # but CLI --alpha bypasses that path).
    if not (0.0 <= alpha <= 1.0):
        _handle_error(
            ConfigError(
                f"Invalid --alpha value: {alpha}. Must be between 0.0 and 1.0 (inclusive)."
            ),
            exit_code=1,
        )

    try:
        # Check vector index exists
        store = VectorStore(args.project_dir, config)
        if not store.has_index() and mode in ("vector", "hybrid"):
            _handle_error(
                SemanticIndexError("No index found. Run build_index.py first to create the index."),
                exit_code=2,
            )

        start_time = time.time()

        # Build filters dict
        filters = {}
        if args.filter_lang:
            filters["language"] = args.filter_lang
        if args.filter_path:
            filters["file_path_glob"] = args.filter_path
        active_filters = filters if filters else None

        vector_results: list[dict] = []
        bm25_results: list[dict] = []

        # --- Vector search ---
        if mode in ("vector", "hybrid"):
            embedder = Embedder(config)
            query_vector = embedder.embed_query(args.query)
            vector_results = store.search(
                vector=query_vector,
                top_k=top_k * 2,
                filters=active_filters,
            )

        # --- BM25 keyword search ---
        if mode in ("keyword", "hybrid"):
            bm25 = BM25Index(args.project_dir)
            if bm25.load():
                bm25_results = bm25.search(
                    query=args.query,
                    top_k=top_k * 2,
                    filters=active_filters,
                )
            elif mode == "keyword":
                _handle_error(
                    SemanticIndexError(
                        "No BM25 index found. Run build_index.py first to create the index."
                    ),
                    exit_code=2,
                )
            else:
                # Hybrid mode but no BM25 index — fall back to vector only
                logger.warning("No BM25 index found, falling back to vector-only search")

        # --- Merge results ---
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

        # --- Apply threshold and truncate ---
        # Each mode uses its own score for thresholding:
        #   vector mode  -> cosine similarity score (0.0-1.0)
        #   keyword mode -> BM25 score (0.0-∞, not comparable to vector)
        #   hybrid mode  -> fused_score from RRF (scale-independent)
        #
        # Hybrid mode skips thresholding because RRF fused_score values
        # are on a completely different scale from cosine similarity.
        # The default threshold (0.3) is calibrated for vector cosine
        # scores and would incorrectly filter most hybrid results.
        # Hybrid relies on top_k truncation instead.
        if mode == "hybrid" and args.threshold is not None:
            logger.warning(
                "--threshold is ignored in hybrid mode (RRF scores are not "
                "comparable to cosine similarity). Results are limited by --top-k only."
            )

        results = []
        for r in merged:
            if mode == "hybrid":
                score = r.get("fused_score", 0.0)
            else:
                score = r.get("score", 0.0)

            if mode == "hybrid" or score >= threshold:
                results.append(r)
            if len(results) >= top_k:
                break

        # --- Optional reranking ---
        rerank_enabled = config.search.rerank_enabled
        if args.rerank:
            rerank_enabled = True
        if args.no_rerank:
            rerank_enabled = False

        if rerank_enabled and results:
            try:
                from lib.reranker import Reranker
                reranker = Reranker(
                    model_name=config.search.rerank_model,
                    device=config.embedding.device,
                    trust_remote_code=config.embedding.trust_remote_code,
                )
                rerank_top_n = config.search.rerank_top_n
                results = reranker.rerank(args.query, results, top_n=rerank_top_n)
            except EmbeddingError as exc:
                logger.warning("Reranking unavailable, skipping: %s", exc)

        duration_ms = (time.time() - start_time) * 1000

        output = {
            "query": args.query,
            "mode": mode,
            "results": [
                {
                    "rank": i + 1,
                    "score": r.get("rerank_score", r.get("fused_score", r.get("score", 0.0))),
                    "file_path": r["file_path"],
                    "start_line": r["start_line"],
                    "end_line": r["end_line"],
                    "chunk_type": r["chunk_type"],
                    "symbol_name": r["symbol_name"],
                    "language": r["language"],
                    "content": r["content"],
                    **({"rerank_score": r["rerank_score"]}
                       if "rerank_score" in r else {}),
                    **({"vector_score": r["vector_score"], "bm25_score": r["bm25_score"]}
                       if mode == "hybrid" and "vector_score" in r else {}),
                }
                for i, r in enumerate(results)
            ],
            "total_results": len(results),
            "search_duration_ms": round(duration_ms),
        }
        print(json.dumps(output, indent=2))

    except EmbeddingError as exc:
        _handle_error(exc, exit_code=2)
    except SemanticIndexError as exc:
        _handle_error(exc, exit_code=2)
    except Exception as exc:
        logger.exception("Unexpected error during search")
        _handle_error(exc, exit_code=2)


if __name__ == "__main__":
    main()
