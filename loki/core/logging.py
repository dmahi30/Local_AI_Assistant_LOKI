"""Privacy-first, offline logging system for LOKI.

Ensures zero telemetry, zero cloud logging, and scrubs sensitive user paths
from local log files.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import ClassVar

from ..utils.paths import get_logs_dir, sanitize_path_for_logging


class PrivacyFilter(logging.Filter):
    """Log filter that automatically scrubs user names and absolute home paths."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = sanitize_path_for_logging(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: sanitize_path_for_logging(v) if isinstance(v, (str, Path)) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    sanitize_path_for_logging(a) if isinstance(a, (str, Path)) else a
                    for a in record.args
                )
        return True


class LokiLoggerManager:
    """Manages application-wide logging configuration."""

    _initialized: ClassVar[bool] = False
    _root_logger_name: ClassVar[str] = "loki"

    @classmethod
    def setup(
        cls,
        log_level: str = "INFO",
        base_dir: Path | str | None = None,
        console_output: bool = True,
        max_bytes: int = 5 * 1024 * 1024,  # 5 MB
        backup_count: int = 3,
    ) -> logging.Logger:
        """Initialize local file and console logging."""
        root_logger = logging.getLogger(cls._root_logger_name)
        root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        root_logger.handlers.clear()

        # Add Privacy Filter
        privacy_filter = PrivacyFilter()
        root_logger.addFilter(privacy_filter)

        log_format = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 1. Local Rotating File Handler
        logs_dir = get_logs_dir(base_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "loki.log"

        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(log_format)
        file_handler.addFilter(privacy_filter)
        root_logger.addHandler(file_handler)

        # 2. Console Handler (Optional for CLI/Dev)
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_format)
            console_handler.addFilter(privacy_filter)
            root_logger.addHandler(console_handler)

        cls._initialized = True
        return root_logger

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a namespaced logger under the LOKI hierarchy."""
        if not cls._initialized:
            cls.setup()
        if name.startswith("loki.") or name == "loki":
            return logging.getLogger(name)
        return logging.getLogger(f"loki.{name}")


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a namespaced LOKI logger."""
    return LokiLoggerManager.get_logger(name)
