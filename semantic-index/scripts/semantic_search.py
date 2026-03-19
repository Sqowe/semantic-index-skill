#!/usr/bin/env python3
"""Search the semantic index by meaning.

Usage:
    python semantic_search.py --project-dir <path> --query <text>
        [--top-k N] [--threshold F] [--filter-lang <lang>] [--filter-path <glob>]

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

from lib.config import load_config
from lib.embedder import Embedder
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
    parser.add_argument("--threshold", type=float, default=None, help="Min similarity score (0.0-1.0)")
    parser.add_argument("--filter-lang", default=None, help="Filter by language (e.g., 'python')")
    parser.add_argument("--filter-path", default=None, help="Filter by file path glob (e.g., 'src/**')")
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.project_dir)
    except ConfigError as exc:
        _handle_error(exc, exit_code=1)

    top_k = args.top_k or config.search.default_top_k
    threshold = args.threshold if args.threshold is not None else config.search.default_threshold

    try:
        # Check index exists
        store = VectorStore(args.project_dir, config)
        if not store.has_index():
            _handle_error(
                SemanticIndexError("No index found. Run build_index.py first to create the index."),
                exit_code=2,
            )

        # Embed the query
        embedder = Embedder(config)
        start_time = time.time()
        query_vector = embedder.embed_query(args.query)

        # Search with over-fetch for filtering
        filters = {}
        if args.filter_lang:
            filters["language"] = args.filter_lang
        if args.filter_path:
            filters["file_path_glob"] = args.filter_path

        raw_results = store.search(
            vector=query_vector,
            top_k=top_k * 2,
            filters=filters if filters else None,
        )

        # Apply threshold and truncate
        results = [r for r in raw_results if r["score"] >= threshold][:top_k]

        duration_ms = (time.time() - start_time) * 1000

        output = {
            "query": args.query,
            "results": [
                {
                    "rank": i + 1,
                    "score": r["score"],
                    "file_path": r["file_path"],
                    "start_line": r["start_line"],
                    "end_line": r["end_line"],
                    "chunk_type": r["chunk_type"],
                    "symbol_name": r["symbol_name"],
                    "language": r["language"],
                    "content": r["content"],
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
