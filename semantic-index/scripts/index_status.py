#!/usr/bin/env python3
"""Show index health and statistics.

Usage:
    python index_status.py --project-dir <path>

Exit codes:
    0  Success
    1  Configuration error
    2  Runtime error
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from lib.config import load_config, INDEX_DIR_NAME
from lib.hasher import detect_changes
from lib.models import ConfigError, SemanticIndexError
from lib.store import VectorStore

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"


def _handle_error(error: Exception, exit_code: int = 2) -> None:
    """Output error as JSON to stdout and exit."""
    print(json.dumps({
        "status": "error",
        "error": str(error),
        "error_type": type(error).__name__,
    }, indent=2))
    sys.exit(exit_code)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show semantic index health and stats")
    parser.add_argument("--project-dir", required=True, help="Project root directory")
    args = parser.parse_args()

    try:
        config = load_config(args.project_dir)
    except ConfigError as exc:
        _handle_error(exc, exit_code=1)

    try:
        index_dir = Path(args.project_dir) / INDEX_DIR_NAME

        # Check if index exists at all
        if not index_dir.exists():
            print(json.dumps({
                "indexed": False,
                "message": "No .index/ directory found. Run build_index.py first.",
            }, indent=2))
            sys.exit(0)

        # Load manifest for metadata
        manifest_path = index_dir / MANIFEST_FILENAME
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read manifest: %s", exc)

        manifest_files = manifest.get("files", {})
        last_indexed = manifest.get("last_indexed", "never")

        # Get store stats
        store = VectorStore(args.project_dir, config)
        stats = store.get_stats()

        # Detect stale files (changed since last index)
        changes = detect_changes(args.project_dir, config)
        stale_files = len(changes.to_index)

        # Compute total index size (entire .index/ directory)
        total_size = 0
        if index_dir.exists():
            for f in index_dir.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size

        output = {
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
        print(json.dumps(output, indent=2))

    except SemanticIndexError as exc:
        _handle_error(exc, exit_code=2)
    except Exception as exc:
        logger.exception("Unexpected error reading index status")
        _handle_error(exc, exit_code=2)


if __name__ == "__main__":
    main()
