#!/usr/bin/env python3
"""Build or rebuild the semantic index for a project.

Usage:
    python build_index.py --project-dir <path> [--config <config.json>] [--full]
                          [--batch-size N]

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

DEFAULT_FILE_BATCH_SIZE = 50


def _handle_error(error: Exception, exit_code: int = 2) -> None:
    """Output error as JSON to stdout and exit."""
    print(json.dumps({
        "status": "error",
        "error": str(error),
        "error_type": type(error).__name__,
    }, indent=2))
    sys.exit(exit_code)


def _batched(items: list, size: int):
    """Yield successive batches of *size* items from *items*."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the semantic index for a project")
    parser.add_argument("--project-dir", required=True, help="Project root directory")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--full", action="store_true", help="Force full re-index")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_FILE_BATCH_SIZE,
        help=(
            f"Number of files to process per batch (default: {DEFAULT_FILE_BATCH_SIZE}). "
            "Smaller batches use less memory; larger batches reduce store commit overhead."
        ),
    )
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

        # --- Phase 1: process new/changed files in batches ---
        total_chunks_created = 0
        total_api_calls = 0
        chunk_counts: dict[str, int] = {}
        file_batch_size = max(1, args.batch_size)
        files_to_index = changes.to_index
        total_files = len(files_to_index)
        files_processed = 0

        for batch_num, file_batch in enumerate(_batched(files_to_index, file_batch_size), 1):
            batch_label = (
                f"Batch {batch_num}/"
                f"{(total_files + file_batch_size - 1) // file_batch_size}"
            )
            print(
                f"\n{batch_label}: processing {len(file_batch)} files...",
                file=sys.stderr,
            )

            # 1. Chunk the batch
            batch_chunks = []
            for file_path in file_batch:
                files_processed += 1
                print(
                    f"  Chunking [{files_processed}/{total_files}] {file_path}...",
                    file=sys.stderr,
                )
                chunks = chunk_file(file_path, args.project_dir, config)
                chunk_counts[file_path] = len(chunks)
                batch_chunks.extend(chunks)

            # 2. Embed the batch
            batch_api_calls = 0
            if batch_chunks:
                batch_api_calls = embedder.embed_chunks(batch_chunks)

            # 3. Commit to store: delete old, add new
            try:
                for file_path in file_batch:
                    store.delete_by_file(file_path)

                if batch_chunks:
                    store.add(batch_chunks)
            except Exception as exc:
                logger.error(
                    "Store commit failed at %s after embedding. "
                    "Index may be partially updated for files processed so far. "
                    "Run 'build_index.py --full' to rebuild. Error: %s",
                    batch_label, exc,
                )
                raise

            total_chunks_created += len(batch_chunks)
            total_api_calls += batch_api_calls

            print(
                f"  {batch_label} done: {len(batch_chunks)} chunks committed",
                file=sys.stderr,
            )

            # 4. Release batch memory
            del batch_chunks

        # --- Phase 2: handle deletions after all batches succeed ---
        for file_path in changes.to_delete:
            store.delete_by_file(file_path)
            logger.info("Deleted chunks for removed file: %s", file_path)

        # --- Phase 3: update manifest only after all batches succeed ---
        update_manifest(args.project_dir, changes, chunk_counts)

        duration = time.time() - start_time
        result = {
            "status": "success",
            "files_indexed": len(changes.to_index),
            "files_skipped": changes.unchanged,
            "files_deleted": len(changes.to_delete),
            "chunks_created": total_chunks_created,
            "duration_seconds": round(duration, 1),
            "embedding_api_calls": total_api_calls,
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
