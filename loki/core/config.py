"""Configuration system for LOKI.

Handles hardware-adaptive defaults, persistence to local config.json,
and offline model definitions.
"""

from __future__ import annotations

import json
from pathlib import Path
from pydantic import BaseModel, Field

from ..utils.paths import (
    ensure_app_dirs,
    get_app_dir,
    get_config_path,
    get_data_dir,
    get_logs_dir,
    get_models_dir,
)
from .exceptions import ConfigurationError
from .types import ExecutionDevice, HardwareTier


class HardwareConfig(BaseModel):
    """Detected hardware profile and tier configuration."""
    tier: HardwareTier = HardwareTier.UNKNOWN
    ram_gb: float = Field(default=0.0, description="Total physical system RAM in GB")
    vram_gb: float = Field(default=0.0, description="Dedicated GPU VRAM in GB")
    has_dedicated_gpu: bool = Field(default=False, description="Whether a dedicated GPU is present")
    cpu_cores: int = Field(default=1, description="Physical CPU cores")


class ModelRuntimeConfig(BaseModel):
    """Local LLM and embedding configuration."""
    active_model: str = Field(default="qwen2.5:3b", description="Primary Ollama model name")
    fallback_models: list[str] = Field(default_factory=lambda: ["qwen2.5:1.5b"], description="Ordered fallbacks")
    embedding_model: str = Field(default="nomic-embed-text", description="Local embedding model")
    ollama_host: str = Field(default="http://127.0.0.1:11434", description="Local Ollama HTTP endpoint")
    context_length: int = Field(default=32768, description="Active context token window")
    execution_device: ExecutionDevice = Field(default=ExecutionDevice.CPU, description="Inference device")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=60, ge=5, le=300)


class SafetyConfig(BaseModel):
    """Safety and permissions settings."""
    require_confirmation_for_destructive: bool = Field(
        default=True, description="Strictly gate Tier 3 destructive operations"
    )
    recycle_bin_deletions: bool = Field(
        default=True, description="Move deleted files to Windows Recycle Bin"
    )
    allow_arbitrary_shell: bool = Field(
        default=False, description="Whether untrusted shell scripts are permitted"
    )


class VoiceConfig(BaseModel):
    """Offline voice input/output settings."""
    enabled: bool = Field(default=False, description="Enable voice I/O")
    whisper_model: str = Field(default="base.en", description="Local faster-whisper model name")
    piper_voice: str = Field(default="en_US-lessac-medium", description="Local Piper TTS voice")


class LokiConfig(BaseModel):
    """Root configuration object for LOKI."""
    app_name: str = "LOKI"
    version: str = "0.1.0"
    offline_mode_enforced: bool = True
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    models: ModelRuntimeConfig = Field(default_factory=ModelRuntimeConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    base_dir: str = Field(default="", description="Base storage directory")

    def model_post_init(self, __context: object) -> None:
        if not self.base_dir:
            self.base_dir = str(get_app_dir())

    @property
    def base_path(self) -> Path:
        return Path(self.base_dir)

    @property
    def logs_path(self) -> Path:
        return get_logs_dir(self.base_path)

    @property
    def data_path(self) -> Path:
        return get_data_dir(self.base_path)

    @property
    def models_path(self) -> Path:
        return get_models_dir(self.base_path)

    def save(self, target_path: Path | str | None = None) -> Path:
        """Persist current configuration to disk as JSON."""
        path = Path(target_path) if target_path else get_config_path(self.base_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.model_dump_json(indent=2))
            return path
        except OSError as e:
            raise ConfigurationError(f"Failed to save configuration to {path}: {e}") from e

    @classmethod
    def load(cls, config_path: Path | str | None = None, base_dir: Path | str | None = None) -> LokiConfig:
        """Load configuration from disk, creating default if file does not exist."""
        path = Path(config_path) if config_path else get_config_path(base_dir)
        if not path.exists():
            config = cls(base_dir=str(get_app_dir(base_dir)))
            ensure_app_dirs(config.base_path)
            config.save(path)
            return config

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.model_validate(data)
        except (json.JSONDecodeError, OSError) as e:
            raise ConfigurationError(f"Failed to parse configuration file {path}: {e}") from e

    @classmethod
    def create_for_tier(cls, tier: HardwareTier, base_dir: Path | str | None = None) -> LokiConfig:
        """Generate a pre-calibrated configuration matching a detected hardware tier."""
        config = cls(base_dir=str(get_app_dir(base_dir)))
        config.hardware.tier = tier

        if tier == HardwareTier.ECO:
            config.models.active_model = "qwen2.5:1.5b"
            config.models.fallback_models = []
            config.models.execution_device = ExecutionDevice.CPU
            config.models.context_length = 16384
            config.voice.whisper_model = "tiny.en"

        elif tier == HardwareTier.BALANCED_CPU:
            config.models.active_model = "qwen2.5:3b"
            config.models.fallback_models = ["qwen2.5:1.5b"]
            config.models.execution_device = ExecutionDevice.CPU
            config.models.context_length = 32768
            config.voice.whisper_model = "base.en"

        elif tier == HardwareTier.BALANCED_GPU:
            config.models.active_model = "qwen2.5:7b"
            config.models.fallback_models = ["qwen2.5:3b", "qwen2.5:1.5b"]
            config.models.execution_device = ExecutionDevice.GPU
            config.models.context_length = 32768
            config.voice.whisper_model = "base.en"

        elif tier == HardwareTier.POWER:
            config.models.active_model = "qwen2.5:7b"
            config.models.fallback_models = ["qwen2.5:3b"]
            config.models.execution_device = ExecutionDevice.GPU
            config.models.context_length = 65536
            config.voice.whisper_model = "small.en"

        return config
