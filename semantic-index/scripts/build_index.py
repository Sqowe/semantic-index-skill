#!/usr/bin/env python3
"""Build or rebuild the semantic index for a project.

Usage:
    python build_index.py --project-dir <path> [--config <config.json>] [--full]

Exit codes:
    0  Success (or no changes detected)
    1  Configuration error
    2  Runtime error (API failure, I/O error)
"""

import argparse
import json
import logging
import sys
import time

from lib.config import load_config, ensure_index_dir
from lib.hasher import detect_changes, update_manifest
from lib.chunker import chunk_file
from lib.embedder import Embedder
from lib.models import ConfigError, EmbeddingError, IndexingError, SemanticIndexError
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
    parser = argparse.ArgumentParser(description="Build the semantic index for a project")
    parser.add_argument("--project-dir", required=True, help="Project root directory")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--full", action="store_true", help="Force full re-index")
    args = parser.parse_args()

    start_time = time.time()

    # Load configuration
    try:
        config = load_config(args.project_dir, args.config)
        ensure_index_dir(args.project_dir)
    except ConfigError as exc:
        _handle_error(exc, exit_code=1)

    try:
        # Detect changes
        changes = detect_changes(args.project_dir, config, force_full=args.full)

        if not changes.to_index and not changes.to_delete:
            print(json.dumps({
                "status": "up_to_date",
                "message": "No changes detected",
                "files_unchanged": changes.unchanged,
            }, indent=2))
            sys.exit(0)

        print(
            f"Found {len(changes.to_index)} files to index, "
            f"{len(changes.to_delete)} to delete, "
            f"{changes.unchanged} unchanged",
            file=sys.stderr,
        )

        # Initialize components
        embedder = Embedder(config, project_dir=args.project_dir)
        store = VectorStore(args.project_dir, config)

        # Chunk new/changed files
        all_chunks = []
        chunk_counts: dict[str, int] = {}

        for i, file_path in enumerate(changes.to_index, 1):
            print(f"  Chunking [{i}/{len(changes.to_index)}] {file_path}...", file=sys.stderr)
            chunks = chunk_file(file_path, args.project_dir, config)
            chunk_counts[file_path] = len(chunks)
            all_chunks.extend(chunks)

        # Embed first — if this fails, the old index stays completely intact
        api_calls = 0
        if all_chunks:
            api_calls = embedder.embed_chunks(all_chunks)

        # === Commit phase: all mutations happen after successful embedding ===
        try:
            # Remove chunks for files deleted from disk
            for file_path in changes.to_delete:
                store.delete_by_file(file_path)
                logger.info("Deleted chunks for removed file: %s", file_path)

            # Remove old chunks for ALL changed files (including those that
            # now produce zero chunks, e.g. shrunk below min_tokens)
            for file_path in changes.to_index:
                store.delete_by_file(file_path)

            # Store new chunks
            if all_chunks:
                store.add(all_chunks)
        except Exception as exc:
            logger.error(
                "Store commit failed after embedding. Index may be in a partial state. "
                "Run 'build_index.py --full' to rebuild. Error: %s", exc,
            )
            raise

        # Update manifest with chunk counts
        update_manifest(args.project_dir, changes, chunk_counts)

        duration = time.time() - start_time
        result = {
            "status": "success",
            "files_indexed": len(changes.to_index),
            "files_skipped": changes.unchanged,
            "files_deleted": len(changes.to_delete),
            "chunks_created": len(all_chunks),
            "duration_seconds": round(duration, 1),
            "embedding_api_calls": api_calls,
        }
        print(json.dumps(result, indent=2))

    except (EmbeddingError, IndexingError) as exc:
        _handle_error(exc, exit_code=2)
    except SemanticIndexError as exc:
        _handle_error(exc, exit_code=2)
    except Exception as exc:
        logger.exception("Unexpected error during indexing")
        _handle_error(exc, exit_code=2)


if __name__ == "__main__":
    main()
