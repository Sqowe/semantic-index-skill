"""File change detection using SHA-256 hashing.

Walks the project directory respecting .gitignore and .indexignore,
computes file hashes, and compares against a stored manifest to
determine which files need re-indexing.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pathspec

from .constants import OFFICE_EXTENSIONS
from .config import Config, INDEX_DIR_NAME
from .models import FileChange

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"


def _load_gitignore_spec(project_dir: Path) -> Optional[pathspec.PathSpec]:
    """Load .gitignore patterns if the file exists."""
    gitignore = project_dir / ".gitignore"
    if not gitignore.exists():
        return None
    lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def _load_indexignore_spec(project_dir: Path) -> Optional[pathspec.PathSpec]:
    """Load .indexignore patterns if the file exists."""
    indexignore = project_dir / ".indexignore"
    if not indexignore.exists():
        return None
    lines = indexignore.read_text(encoding="utf-8", errors="replace").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def _build_exclude_spec(config: Config) -> pathspec.PathSpec:
    """Build a PathSpec from config exclude_patterns."""
    return pathspec.PathSpec.from_lines("gitwildmatch", config.indexing.exclude_patterns)


def _sha256_file(file_path: Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _should_include(
    rel_path: str,
    config: Config,
    gitignore_spec: Optional[pathspec.PathSpec],
    indexignore_spec: Optional[pathspec.PathSpec],
    exclude_spec: pathspec.PathSpec,
) -> bool:
    """Check if a file should be included in indexing."""
    # Check file extension
    ext = os.path.splitext(rel_path)[1].lower()
    if ext not in config.indexing.file_extensions:
        return False

    # Check exclude patterns from config
    if exclude_spec.match_file(rel_path):
        return False

    # Check .gitignore
    if config.indexing.respect_gitignore and gitignore_spec:
        if gitignore_spec.match_file(rel_path):
            return False

    # Check .indexignore
    if indexignore_spec and indexignore_spec.match_file(rel_path):
        return False

    return True


def walk_project_files(project_dir: str, config: Config) -> list[str]:
    """Walk the project directory and return relative paths of indexable files.

    Respects .gitignore, .indexignore, and config exclude_patterns.
    Skips files exceeding max_file_size_kb.

    Returns:
        Sorted list of relative file paths.
    """
    project_path = Path(project_dir).resolve()
    gitignore_spec = _load_gitignore_spec(project_path)
    indexignore_spec = _load_indexignore_spec(project_path)
    exclude_spec = _build_exclude_spec(config)
    max_size = config.indexing.max_file_size_kb * 1024
    max_office_size = config.indexing.max_office_file_size_kb * 1024

    files: list[str] = []

    for root, dirs, filenames in os.walk(project_path):
        # Compute relative directory path for filtering
        rel_root = os.path.relpath(root, project_path)
        if rel_root == ".":
            rel_root = ""

        # Prune excluded directories in-place
        dirs[:] = [
            d for d in dirs
            if not exclude_spec.match_file(os.path.join(rel_root, d) + "/")
            and not (gitignore_spec and gitignore_spec.match_file(os.path.join(rel_root, d) + "/"))
            and not (indexignore_spec and indexignore_spec.match_file(os.path.join(rel_root, d) + "/"))
        ]

        for filename in filenames:
            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, project_path)

            if not _should_include(rel_path, config, gitignore_spec, indexignore_spec, exclude_spec):
                continue

            # Check file size (office files get a higher limit)
            ext = os.path.splitext(filename)[1].lower()
            size_limit = max_office_size if ext in OFFICE_EXTENSIONS else max_size
            try:
                if os.path.getsize(abs_path) > size_limit:
                    logger.debug("Skipping large file: %s", rel_path)
                    continue
            except OSError:
                continue

            files.append(rel_path)

    return sorted(files)


def _load_manifest(project_dir: str) -> dict:
    """Load the manifest file, returning empty structure if missing."""
    manifest_path = Path(project_dir) / INDEX_DIR_NAME / MANIFEST_FILENAME
    if not manifest_path.exists():
        return {"version": "1.0", "files": {}}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read manifest, treating as empty: %s", exc)
        return {"version": "1.0", "files": {}}


def detect_changes(
    project_dir: str,
    config: Config,
    force_full: bool = False,
) -> FileChange:
    """Compare current files against the manifest to find changes.

    Args:
        project_dir: Path to the project root.
        config: Loaded configuration.
        force_full: If True, mark all files for re-indexing.

    Returns:
        FileChange with lists of files to index, delete, and unchanged count.
    """
    current_files = walk_project_files(project_dir, config)
    project_path = Path(project_dir).resolve()

    if force_full:
        return FileChange(to_index=current_files, to_delete=[], unchanged=0)

    manifest = _load_manifest(project_dir)
    manifest_files = manifest.get("files", {})

    change = FileChange()
    current_set = set(current_files)
    manifest_set = set(manifest_files.keys())

    # Files to delete: in manifest but no longer on disk
    change.to_delete = sorted(manifest_set - current_set)

    # Check each current file against manifest
    for rel_path in current_files:
        abs_path = project_path / rel_path
        current_hash = _sha256_file(abs_path)

        if rel_path not in manifest_files:
            # New file
            change.to_index.append(rel_path)
        elif manifest_files[rel_path].get("hash") != current_hash:
            # Changed file
            change.to_index.append(rel_path)
        else:
            change.unchanged += 1

    logger.info(
        "Change detection: %d to index, %d to delete, %d unchanged",
        len(change.to_index),
        len(change.to_delete),
        change.unchanged,
    )
    return change


def update_manifest(
    project_dir: str,
    changes: FileChange,
    chunk_counts: Optional[dict[str, int]] = None,
) -> None:
    """Update the manifest after indexing.

    Args:
        project_dir: Path to the project root.
        changes: The FileChange that was processed.
        chunk_counts: Optional mapping of file_path -> chunk count.
    """
    manifest = _load_manifest(project_dir)
    files = manifest.get("files", {})
    project_path = Path(project_dir).resolve()
    now = datetime.now(timezone.utc).isoformat()

    # Remove deleted files
    for rel_path in changes.to_delete:
        files.pop(rel_path, None)

    # Update indexed files
    for rel_path in changes.to_index:
        abs_path = project_path / rel_path
        if not abs_path.exists():
            continue
        files[rel_path] = {
            "hash": _sha256_file(abs_path),
            "last_indexed": now,
            "chunk_count": (chunk_counts or {}).get(rel_path, 0),
            "file_size_bytes": abs_path.stat().st_size,
        }

    manifest["version"] = "1.0"
    manifest["last_indexed"] = now
    manifest["project_dir"] = str(project_path)
    manifest["files"] = files

    manifest_path = Path(project_dir) / INDEX_DIR_NAME / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Manifest updated: %d files tracked", len(files))
