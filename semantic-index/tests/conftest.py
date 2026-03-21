"""Shared test fixtures for semantic-index tests."""

import sys
from pathlib import Path

import pytest

# Add scripts/ to sys.path so `lib` is importable
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lib.config import Config, ChunkingConfig


@pytest.fixture
def default_config() -> Config:
    """Config with default chunking settings for tests."""
    config = Config()
    config.chunking = ChunkingConfig(max_tokens=512, overlap_tokens=50, min_tokens=20)
    return config


@pytest.fixture
def small_config() -> Config:
    """Config with small max_tokens to force oversized splitting."""
    config = Config()
    config.chunking = ChunkingConfig(max_tokens=60, overlap_tokens=10, min_tokens=5)
    return config
