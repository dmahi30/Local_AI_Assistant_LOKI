"""Path management and privacy-safe path sanitization for LOKI."""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "LOKI"


def get_default_app_dir() -> Path:
    """Get the default platform-appropriate application directory.
    
    On Windows: %LOCALAPPDATA%\\LOKI (e.g. C:\\Users\\<user>\\AppData\\Local\\LOKI)
    Fallback / Other OS: ~/.loki
    """
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_NAME
    return Path.home() / f".{APP_NAME.lower()}"


def get_app_dir(custom_base: Path | str | None = None) -> Path:
    """Return the base application directory."""
    if custom_base is not None:
        return Path(custom_base)
    env_override = os.environ.get("LOKI_HOME")
    if env_override:
        return Path(env_override)
    return get_default_app_dir()


def get_logs_dir(base_dir: Path | str | None = None) -> Path:
    """Return the directory where local logs are stored."""
    return get_app_dir(base_dir) / "logs"


def get_models_dir(base_dir: Path | str | None = None) -> Path:
    """Return the directory where local offline models/weights are stored."""
    return get_app_dir(base_dir) / "models"


def get_data_dir(base_dir: Path | str | None = None) -> Path:
    """Return the directory where local databases (SQLite, vector index) are stored."""
    return get_app_dir(base_dir) / "data"


def get_config_path(base_dir: Path | str | None = None) -> Path:
    """Return the path to the configuration JSON file."""
    return get_app_dir(base_dir) / "config.json"


def ensure_app_dirs(base_dir: Path | str | None = None) -> dict[str, Path]:
    """Create all required LOKI application directories if they do not exist.
    
    Returns a dictionary of created/verified directory paths.
    """
    base = get_app_dir(base_dir)
    dirs = {
        "app": base,
        "logs": get_logs_dir(base),
        "models": get_models_dir(base),
        "data": get_data_dir(base),
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def sanitize_path_for_logging(path_str: str | Path) -> str:
    """Sanitize paths for logs to protect user privacy.
    
    Replaces user home directories (e.g. C:\\Users\\johndoe) with <USER_HOME>
    so that sensitive user names are never leaked into logs.
    """
    raw = str(path_str)
    home = str(Path.home())
    if home and home in raw:
        return raw.replace(home, "<USER_HOME>")
    return raw
