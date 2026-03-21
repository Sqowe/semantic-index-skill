"""Tests for numeric range validation in config loading.

Covers review issue: config validation is incomplete for numeric ranges.
Tests that invalid values for max_retries, batch_size, retry_delay_seconds,
default_top_k, and rerank_top_n are rejected at load time.
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib.config import load_config
from lib.models import ConfigError


def _write_config(tmp_path: Path, overrides: dict) -> None:
    """Write a config.json with the given section overrides."""
    cfg: dict = {"schema_version": "1.0"}
    cfg.update(overrides)
    index_dir = tmp_path / ".index"
    index_dir.mkdir(exist_ok=True)
    (index_dir / "config.json").write_text(json.dumps(cfg))


class TestEmbeddingNumericValidation:
    """Tests for embedding section numeric range validation."""

    def test_max_retries_zero_raises(self, tmp_path):
        _write_config(tmp_path, {"embedding": {"max_retries": 0}})
        with pytest.raises(ConfigError, match="max_retries"):
            load_config(str(tmp_path))

    def test_max_retries_negative_raises(self, tmp_path):
        _write_config(tmp_path, {"embedding": {"max_retries": -1}})
        with pytest.raises(ConfigError, match="max_retries"):
            load_config(str(tmp_path))

    def test_max_retries_one_passes(self, tmp_path):
        _write_config(tmp_path, {"embedding": {"max_retries": 1}})
        config = load_config(str(tmp_path))
        assert config.embedding.max_retries == 1

    def test_batch_size_zero_raises(self, tmp_path):
        _write_config(tmp_path, {"embedding": {"batch_size": 0}})
        with pytest.raises(ConfigError, match="batch_size"):
            load_config(str(tmp_path))

    def test_batch_size_negative_raises(self, tmp_path):
        _write_config(tmp_path, {"embedding": {"batch_size": -5}})
        with pytest.raises(ConfigError, match="batch_size"):
            load_config(str(tmp_path))

    def test_batch_size_one_passes(self, tmp_path):
        _write_config(tmp_path, {"embedding": {"batch_size": 1}})
        config = load_config(str(tmp_path))
        assert config.embedding.batch_size == 1

    def test_retry_delay_negative_raises(self, tmp_path):
        _write_config(tmp_path, {"embedding": {"retry_delay_seconds": -0.5}})
        with pytest.raises(ConfigError, match="retry_delay_seconds"):
            load_config(str(tmp_path))

    def test_retry_delay_zero_passes(self, tmp_path):
        _write_config(tmp_path, {"embedding": {"retry_delay_seconds": 0}})
        config = load_config(str(tmp_path))
        assert config.embedding.retry_delay_seconds == 0


class TestSearchNumericValidation:
    """Tests for search section numeric range validation."""

    def test_default_top_k_zero_raises(self, tmp_path):
        _write_config(tmp_path, {"search": {"default_top_k": 0}})
        with pytest.raises(ConfigError, match="default_top_k"):
            load_config(str(tmp_path))

    def test_default_top_k_negative_raises(self, tmp_path):
        _write_config(tmp_path, {"search": {"default_top_k": -1}})
        with pytest.raises(ConfigError, match="default_top_k"):
            load_config(str(tmp_path))

    def test_default_top_k_one_passes(self, tmp_path):
        _write_config(tmp_path, {"search": {"default_top_k": 1}})
        config = load_config(str(tmp_path))
        assert config.search.default_top_k == 1

    def test_rerank_top_n_zero_raises(self, tmp_path):
        _write_config(tmp_path, {"search": {"rerank_top_n": 0}})
        with pytest.raises(ConfigError, match="rerank_top_n"):
            load_config(str(tmp_path))

    def test_rerank_top_n_negative_raises(self, tmp_path):
        _write_config(tmp_path, {"search": {"rerank_top_n": -3}})
        with pytest.raises(ConfigError, match="rerank_top_n"):
            load_config(str(tmp_path))

    def test_rerank_top_n_one_passes(self, tmp_path):
        _write_config(tmp_path, {"search": {"rerank_top_n": 1}})
        config = load_config(str(tmp_path))
        assert config.search.rerank_top_n == 1
