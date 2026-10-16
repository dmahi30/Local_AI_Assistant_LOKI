"""Tests for configuration persistence, defaults, and tier profiles."""

from pathlib import Path
import pytest

from loki.core.config import LokiConfig
from loki.core.exceptions import ConfigurationError
from loki.core.types import ExecutionDevice, HardwareTier


def test_default_config_properties(temp_loki_dir: Path):
    config = LokiConfig(base_dir=str(temp_loki_dir))
    assert config.app_name == "LOKI"
    assert config.offline_mode_enforced is True
    assert config.models.ollama_host == "http://127.0.0.1:11434"
    assert config.safety.require_confirmation_for_destructive is True
    assert config.safety.recycle_bin_deletions is True


def test_config_save_and_load_roundtrip(temp_loki_dir: Path):
    config = LokiConfig(base_dir=str(temp_loki_dir))
    config.models.active_model = "test-model:latest"
    config.hardware.ram_gb = 16.0
    config.hardware.tier = HardwareTier.BALANCED_GPU

    config_path = temp_loki_dir / "config.json"
    config.save(config_path)
    assert config_path.exists()

    loaded = LokiConfig.load(config_path, base_dir=temp_loki_dir)
    assert loaded.models.active_model == "test-model:latest"
    assert loaded.hardware.ram_gb == 16.0
    assert loaded.hardware.tier == HardwareTier.BALANCED_GPU


def test_config_create_for_eco_tier(temp_loki_dir: Path):
    config = LokiConfig.create_for_tier(HardwareTier.ECO, base_dir=temp_loki_dir)
    assert config.models.active_model == "qwen2.5:1.5b"
    assert config.models.execution_device == ExecutionDevice.CPU
    assert config.voice.whisper_model == "tiny.en"


def test_config_create_for_balanced_gpu_tier(temp_loki_dir: Path):
    config = LokiConfig.create_for_tier(HardwareTier.BALANCED_GPU, base_dir=temp_loki_dir)
    assert config.models.active_model == "qwen2.5:7b"
    assert config.models.execution_device == ExecutionDevice.GPU
    assert "qwen2.5:3b" in config.models.fallback_models


def test_corrupted_config_handling(temp_loki_dir: Path):
    bad_file = temp_loki_dir / "config.json"
    bad_file.write_text("{invalid_json_format", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        LokiConfig.load(bad_file, base_dir=temp_loki_dir)
