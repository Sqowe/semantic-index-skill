"""Tests for the migrate_config Phase 2 additions.

Phase 2 adds two new embedding fields that align chunking budgets with
the embedding model's tokenizer:

* ``max_embed_tokens``: the model's context window in tokens
  (8192 for BAAI/bge-m3 by default).
* ``token_safety_factor``: the multiplier applied to chunking budgets
  when the chunker is using the tiktoken fallback (no real tokenizer
  available). Covers the measured 1.30x median ratio between bge-m3
  and cl100k tokens with room for the 2.13x worst case.

Existing indexes that pre-date Phase 2 are migrated by
``migrate_config.py``. This test verifies that the migration adds the
two fields with their documented defaults and does not touch
unrelated fields.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
MIGRATE_SCRIPT = SCRIPTS_DIR / "migrate_config.py"


def _run_migrate(project_dir: Path, *args: str) -> subprocess.CompletedProcess:
    """Invoke migrate_config.py with the given project_dir and flags."""
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "migrate_config.py"),
         "--project-dir", str(project_dir), *args],
        capture_output=True,
        text=True,
        check=False,
    )


class TestPhase2Migration:
    """``analyze_config`` adds Phase 2 fields to existing configs."""

    def test_max_embed_tokens_added_when_missing(self, tmp_path):
        (tmp_path / ".index").mkdir()
        (tmp_path / ".index" / "config.json").write_text(json.dumps({
            "schema_version": "1.0",
            "embedding": {"provider": "openrouter", "model": "BAAI/bge-m3"},
        }))

        result = _run_migrate(tmp_path, "--force")

        assert result.returncode == 0, result.stderr
        migrated = json.loads(
            (tmp_path / ".index" / "config.json").read_text()
        )
        assert migrated["embedding"]["max_embed_tokens"] == 8192

    def test_token_safety_factor_added_when_missing(self, tmp_path):
        (tmp_path / ".index").mkdir()
        (tmp_path / ".index" / "config.json").write_text(json.dumps({
            "schema_version": "1.0",
            "embedding": {"provider": "openrouter"},
        }))

        result = _run_migrate(tmp_path, "--force")

        assert result.returncode == 0
        migrated = json.loads(
            (tmp_path / ".index" / "config.json").read_text()
        )
        assert abs(migrated["embedding"]["token_safety_factor"] - 1.6) < 1e-9

    def test_existing_phase2_fields_not_overwritten(self, tmp_path):
        """If the user already set Phase 2 fields, the migration leaves them."""
        (tmp_path / ".index").mkdir()
        (tmp_path / ".index" / "config.json").write_text(json.dumps({
            "schema_version": "1.0",
            "embedding": {
                "provider": "openrouter",
                "max_embed_tokens": 32768,
                "token_safety_factor": 2.0,
            },
        }))

        result = _run_migrate(tmp_path, "--force")

        assert result.returncode == 0
        migrated = json.loads(
            (tmp_path / ".index" / "config.json").read_text()
        )
        assert migrated["embedding"]["max_embed_tokens"] == 32768
        assert abs(migrated["embedding"]["token_safety_factor"] - 2.0) < 1e-9

    def test_dry_run_reports_without_writing(self, tmp_path):
        (tmp_path / ".index").mkdir()
        original = {
            "schema_version": "1.0",
            "embedding": {"provider": "openrouter"},
        }
        (tmp_path / ".index" / "config.json").write_text(json.dumps(original))

        result = _run_migrate(tmp_path, "--dry-run")

        # File should be unchanged after a dry run.
        on_disk = json.loads(
            (tmp_path / ".index" / "config.json").read_text()
        )
        assert on_disk == original
        # stdout is JSON and includes both new fields in the migration list.
        stdout_payload = json.loads(result.stdout)
        migration_fields = {m["field"] for m in stdout_payload["migrations"]}
        assert "embedding.max_embed_tokens" in migration_fields
        assert "embedding.token_safety_factor" in migration_fields
