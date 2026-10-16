"""Pytest configuration and shared test fixtures for LOKI."""

from __future__ import annotations

import logging
from pathlib import Path
import pytest

from loki.core.logging import LokiLoggerManager
from loki.utils.paths import ensure_app_dirs


@pytest.fixture
def temp_loki_dir(tmp_path: Path) -> Path:
    """Fixture providing an isolated temporary directory for LOKI app data."""
    base = tmp_path / "LOKI_TEST"
    ensure_app_dirs(base)
    return base


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logger state before and after each test."""
    LokiLoggerManager._initialized = False
    logger = logging.getLogger("loki")
    logger.handlers.clear()
    yield
    LokiLoggerManager._initialized = False
    logger.handlers.clear()
