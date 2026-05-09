#!/usr/bin/env python3
"""Migrate .index/config.json to the latest schema version.

Analyzes an existing project config and applies necessary upgrades
for new features (e.g., Phase 3 hybrid search fields). Reports what
changed and optionally writes the updated config.

Usage:
    python migrate_config.py --project-dir <path> [--dry-run] [--force]

Exit codes:
    0  Success (config up to date or migrated)
    1  Configuration error (file not found, invalid JSON)
    2  Runtime error
"""

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

# Add scripts/ to sys.path so we can import the canonical constants
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from lib.constants import OFFICE_EXTENSIONS  # noqa: E402

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

INDEX_DIR_NAME = ".index"
CONFIG_FILENAME = "config.json"

# Current schema defaults for the search section (Phase 3+4)
SEARCH_DEFAULTS = {
    "default_top_k": 10,
    "default_threshold": 0.3,
    "mode": "hybrid",
    "hybrid_alpha": 0.7,
    "rerank_enabled": False,
    "rerank_model": "BAAI/bge-reranker-v2-m3",
    "rerank_top_n": 10,
}

# Embedding fields added in Phase 4
EMBEDDING_DEFAULTS = {
    "device": None,
    "trust_remote_code": False,
    "max_embed_chars": 24000,
}

# Fields that should be removed (deprecated or moved to future phases)
DEPRECATED_SEARCH_FIELDS: list[str] = []


def _handle_error(error: Exception, exit_code: int = 1) -> None:
    """Output error as JSON to stdout and exit."""
    print(json.dumps({
        "status": "error",
        "error": str(error),
        "error_type": type(error).__name__,
    }, indent=2))
    sys.exit(exit_code)


def analyze_config(config: dict) -> list[dict]:
    """Analyze a config dict and return a list of suggested migrations.

    Each migration is a dict with:
        - field: dotted path to the field (e.g., "search.mode")
        - action: "add", "remove", or "update"
        - reason: human-readable explanation
        - old_value: current value (for remove/update) or None
        - new_value: suggested value (for add/update) or None

    Args:
        config: Parsed config.json dict.

    Returns:
        List of migration dicts.
    """
    migrations: list[dict] = []
    search = config.get("search", {})
    embedding = config.get("embedding", {})

    # Check for missing search fields (Phase 3 + Phase 4 rerank)
    for field, default_value in SEARCH_DEFAULTS.items():
        if field not in search:
            phase = "Phase 4 reranker" if field.startswith("rerank") else "Phase 3 hybrid search"
            migrations.append({
                "field": f"search.{field}",
                "action": "add",
                "reason": f"New {phase} field. Default: {default_value!r}",
                "old_value": None,
                "new_value": default_value,
            })

    # Check for missing embedding fields (Phase 4)
    for field, default_value in EMBEDDING_DEFAULTS.items():
        if field not in embedding:
            migrations.append({
                "field": f"embedding.{field}",
                "action": "add",
                "reason": f"New Phase 4 embedding field. Default: {default_value!r}",
                "old_value": None,
                "new_value": default_value,
            })

    # Check for deprecated fields
    for field in DEPRECATED_SEARCH_FIELDS:
        if field in search:
            migrations.append({
                "field": f"search.{field}",
                "action": "remove",
                "reason": "Deprecated field. This field has no effect and will be ignored.",
                "old_value": search[field],
                "new_value": None,
            })

    # Check schema_version
    version = config.get("schema_version", "")
    if version != "1.0":
        migrations.append({
            "field": "schema_version",
            "action": "update" if version else "add",
            "reason": "Ensure schema version is set to '1.0'",
            "old_value": version or None,
            "new_value": "1.0",
        })

    # Check for missing DITA file extensions (Phase 5)
    indexing = config.get("indexing", {})
    file_extensions = indexing.get("file_extensions")
    if not isinstance(file_extensions, list):
        if isinstance(file_extensions, str):
            file_extensions = [file_extensions]
        else:
            file_extensions = list(file_extensions) if file_extensions else []
    dita_extensions = [".dita", ".ditamap"]
    missing_dita = [ext for ext in dita_extensions if ext not in file_extensions]
    if missing_dita:
        migrations.append({
            "field": "indexing.file_extensions",
            "action": "update" if file_extensions else "add",
            "reason": (
                f"Phase 5 DITA support: add {', '.join(missing_dita)} "
                "to enable DITA documentation indexing."
            ),
            "old_value": file_extensions if file_extensions else None,
            "new_value": file_extensions + missing_dita,
        })

    # Check for missing office file extensions (Phase 9)
    # Re-read extensions in case DITA migration updated them
    current_exts = file_extensions + missing_dita
    office_extensions = sorted(OFFICE_EXTENSIONS)
    missing_office = [ext for ext in office_extensions if ext not in current_exts]
    if missing_office:
        migrations.append({
            "field": "indexing.file_extensions",
            "action": "update" if current_exts else "add",
            "reason": (
                f"Phase 9 office document support: add {', '.join(missing_office)} "
                "to enable PDF, DOCX, and PPTX indexing."
            ),
            "old_value": current_exts if current_exts else None,
            "new_value": current_exts + missing_office,
        })

    # Check for missing max_office_file_size_kb (Phase 9)
    if "max_office_file_size_kb" not in indexing:
        migrations.append({
            "field": "indexing.max_office_file_size_kb",
            "action": "add",
            "reason": (
                "Phase 9 office document support: separate size limit for "
                "binary office files (default 50MB). Office files are inherently "
                "larger than source code files."
            ),
            "old_value": None,
            "new_value": 50000,
        })

    return migrations


def apply_migrations(config: dict, migrations: list[dict]) -> dict:
    """Apply migrations to a config dict and return the updated config.

    Args:
        config: Original config dict (not modified in place).
        migrations: List of migration dicts from analyze_config().

    Returns:
        New config dict with migrations applied.
    """
    updated = deepcopy(config)

    for migration in migrations:
        field = migration["field"]
        action = migration["action"]
        parts = field.split(".")

        if action == "add":
            # Navigate to parent, set the field
            parent = updated
            for part in parts[:-1]:
                if part not in parent:
                    parent[part] = {}
                parent = parent[part]
            parent[parts[-1]] = migration["new_value"]

        elif action == "remove":
            # Navigate to parent, delete the field
            parent = updated
            for part in parts[:-1]:
                if part not in parent:
                    break
                parent = parent[part]
            else:
                parent.pop(parts[-1], None)

        elif action == "update":
            parent = updated
            for part in parts[:-1]:
                if part not in parent:
                    parent[part] = {}
                parent = parent[part]
            value = migration["new_value"]
            # Deduplicate list values (e.g. file_extensions) to prevent
            # accumulation across repeated migrations.
            if isinstance(value, list):
                seen: set = set()
                deduped: list = []
                for item in value:
                    if item not in seen:
                        seen.add(item)
                        deduped.append(item)
                value = deduped
            parent[parts[-1]] = value

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate .index/config.json to the latest schema"
    )
    parser.add_argument(
        "--project-dir", required=True,
        help="Project root directory containing .index/config.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing to disk",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Apply changes without prompting for confirmation",
    )
    args = parser.parse_args()

    config_path = Path(args.project_dir) / INDEX_DIR_NAME / CONFIG_FILENAME

    if not config_path.exists():
        _handle_error(
            FileNotFoundError(
                f"No config found at {config_path}. "
                "Run build_index.py first to create a default config."
            ),
            exit_code=1,
        )

    # Load existing config
    try:
        raw_text = config_path.read_text(encoding="utf-8")
        config = json.loads(raw_text)
    except (json.JSONDecodeError, OSError) as exc:
        _handle_error(exc, exit_code=1)

    # Analyze
    migrations = analyze_config(config)

    if not migrations:
        print(json.dumps({
            "status": "up_to_date",
            "message": "Config is already up to date. No migrations needed.",
        }, indent=2))
        sys.exit(0)

    # Report findings
    print("\nConfig migration analysis:", file=sys.stderr)
    print(f"  File: {config_path}", file=sys.stderr)
    print(f"  Migrations found: {len(migrations)}\n", file=sys.stderr)

    for i, m in enumerate(migrations, 1):
        action_label = {"add": "ADD", "remove": "REMOVE", "update": "UPDATE"}[m["action"]]
        print(f"  {i}. [{action_label}] {m['field']}", file=sys.stderr)
        print(f"     Reason: {m['reason']}", file=sys.stderr)
        if m["old_value"] is not None:
            print(f"     Current: {m['old_value']!r}", file=sys.stderr)
        if m["new_value"] is not None:
            print(f"     New:     {m['new_value']!r}", file=sys.stderr)
        print(file=sys.stderr)

    if args.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "migrations": migrations,
            "message": f"{len(migrations)} migration(s) would be applied.",
        }, indent=2))
        sys.exit(0)

    # Confirm unless --force
    if not args.force:
        print(
            f"Apply {len(migrations)} migration(s) to {config_path}? [y/N] ",
            end="", file=sys.stderr,
        )
        answer = input().strip().lower()
        if answer not in ("y", "yes"):
            print(json.dumps({
                "status": "cancelled",
                "message": "Migration cancelled by user.",
            }, indent=2))
            sys.exit(0)

    # Apply
    updated = apply_migrations(config, migrations)

    # Write back
    config_path.write_text(
        json.dumps(updated, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": "success",
        "migrations_applied": len(migrations),
        "message": f"Applied {len(migrations)} migration(s) to {config_path}",
    }, indent=2))


if __name__ == "__main__":
    main()
