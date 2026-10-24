"""Pytest configuration and shared test fixtures for LOKI."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
import pytest

from loki.core.logging import LokiLoggerManager
from loki.utils.paths import ensure_app_dirs

# Unconditionally isolate LOKI_HOME at module import time so collection-time
# imports can never access or pollute the user's real app directory (%LOCALAPPDATA%\LOKI),
# even if the parent shell environment already defined LOKI_HOME.
_ORIGINAL_LOKI_HOME: str | None = os.environ.get("LOKI_HOME")
_SESSION_TEST_ROOT = Path(tempfile.gettempdir()) / f"loki_pytest_session_{os.getpid()}"
ensure_app_dirs(_SESSION_TEST_ROOT)
os.environ["LOKI_HOME"] = str(_SESSION_TEST_ROOT)


def pytest_configure(config: pytest.Config) -> None:
    """Set an isolated baseline LOKI_HOME during pytest startup and collection.

    Prevents any collection-time or early initialization from touching
    the user's real app data directory (%LOCALAPPDATA%\\LOKI), even if
    the parent environment had LOKI_HOME defined.
    """
    ensure_app_dirs(_SESSION_TEST_ROOT)
    os.environ["LOKI_HOME"] = str(_SESSION_TEST_ROOT)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Restore parent environment LOKI_HOME if previously set."""
    if _ORIGINAL_LOKI_HOME is not None:
        os.environ["LOKI_HOME"] = _ORIGINAL_LOKI_HOME
    else:
        os.environ.pop("LOKI_HOME", None)


@pytest.fixture(autouse=True)
def isolate_loki_home_for_tests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Ensure every test uses a strictly isolated, clean temporary LOKI_HOME directory.

    Guarantees no test can read, create, or alter real conversations in
    %LOCALAPPDATA%\\LOKI\\data\\loki.db.
    """
    test_home = tmp_path / "loki_test_home"
    ensure_app_dirs(test_home)
    monkeypatch.setenv("LOKI_HOME", str(test_home))
    return test_home


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
