"""Tests for privacy-first offline logging."""

from pathlib import Path
import logging

from loki.core.logging import LokiLoggerManager, get_logger
from loki.utils.paths import get_logs_dir


def test_logger_file_creation(temp_loki_dir: Path):
    LokiLoggerManager.setup(
        log_level="DEBUG",
        base_dir=temp_loki_dir,
        console_output=False,
    )
    logger = get_logger("test_module")
    logger.info("Test event occurred")

    log_file = get_logs_dir(temp_loki_dir) / "loki.log"
    assert log_file.exists()

    content = log_file.read_text(encoding="utf-8")
    assert "Test event occurred" in content
    assert "[loki.test_module]" in content


def test_logger_privacy_scrubbing(temp_loki_dir: Path):
    LokiLoggerManager.setup(
        log_level="INFO",
        base_dir=temp_loki_dir,
        console_output=False,
    )
    logger = get_logger("privacy_test")
    
    home_dir = str(Path.home())
    sensitive_path = f"{home_dir}\\confidential\\file.pdf"
    logger.info("Accessing file at %s", sensitive_path)

    log_file = get_logs_dir(temp_loki_dir) / "loki.log"
    content = log_file.read_text(encoding="utf-8")

    assert home_dir not in content
    assert "<USER_HOME>\\confidential\\file.pdf" in content
