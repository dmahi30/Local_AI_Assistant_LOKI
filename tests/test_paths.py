"""Tests for path management and privacy sanitization."""

from pathlib import Path
import os

from loki.utils.paths import (
    ensure_app_dirs,
    get_app_dir,
    get_config_path,
    get_data_dir,
    get_logs_dir,
    get_models_dir,
    sanitize_path_for_logging,
)


def test_custom_app_dir_hierarchy(tmp_path: Path):
    custom_base = tmp_path / "CustomLoki"
    assert get_app_dir(custom_base) == custom_base
    assert get_logs_dir(custom_base) == custom_base / "logs"
    assert get_models_dir(custom_base) == custom_base / "models"
    assert get_data_dir(custom_base) == custom_base / "data"
    assert get_config_path(custom_base) == custom_base / "config.json"


def test_ensure_app_dirs_creation(tmp_path: Path):
    custom_base = tmp_path / "ScaffoldTest"
    dirs = ensure_app_dirs(custom_base)

    assert dirs["app"].exists()
    assert dirs["logs"].exists()
    assert dirs["models"].exists()
    assert dirs["data"].exists()


def test_privacy_path_sanitization():
    home = str(Path.home())
    test_path = f"{home}\\Documents\\secret_notes.txt"
    sanitized = sanitize_path_for_logging(test_path)

    assert home not in sanitized
    assert "<USER_HOME>" in sanitized
    assert "secret_notes.txt" in sanitized


def test_env_override_path(monkeypatch, tmp_path: Path):
    override = tmp_path / "EnvLoki"
    monkeypatch.setenv("LOKI_HOME", str(override))
    assert get_app_dir() == override
