"""Tests for ModelResolver with mocked OllamaClient."""

from unittest.mock import MagicMock
import pytest

from loki.core.config import ModelRuntimeConfig
from loki.core.types import HardwareTier
from loki.hardware.models import GPUInfo, HardwareSpec
from loki.llm.client import OllamaClient
from loki.llm.resolver import ModelResolutionResult, ModelResolver


def test_resolver_tier_recommendations():
    resolver = ModelResolver()
    assert resolver.get_tier_recommended_model(HardwareTier.ECO) == "qwen2.5:1.5b"
    assert resolver.get_tier_recommended_model(HardwareTier.BALANCED_CPU) == "qwen2.5:3b"
    assert resolver.get_tier_recommended_model(HardwareTier.BALANCED_GPU) == "qwen2.5:7b"
    assert resolver.get_tier_recommended_model(HardwareTier.POWER) == "qwen2.5:7b"


def test_resolver_ollama_offline():
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = False

    resolver = ModelResolver(client=mock_client)
    res = resolver.resolve()

    assert res.is_ready is False
    assert res.ollama_online is False
    assert res.effective_model is None
    assert "offline or unreachable" in res.message


def test_resolver_model_available():
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.check_health.return_value = True
    mock_client.list_local_models.return_value = ["qwen2.5:3b", "nomic-embed-text"]
    mock_client.is_model_available.side_effect = lambda m: m == "qwen2.5:3b"

    cfg = ModelRuntimeConfig(active_model="qwen2.5:3b")
    resolver = ModelResolver(config=cfg, client=mock_client)
    res = resolver.resolve()

    assert res.is_ready is True
    assert res.ollama_online is True
    assert res.effective_model == "qwen2.5:3b"
    assert res.using_fallback is False
    assert "available locally" in res.message


def test_resolver_model_missing_no_fallback():
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.check_health.return_value = True
    mock_client.list_local_models.return_value = ["qwen2.5:0.5b"]
    mock_client.is_model_available.return_value = False

    cfg = ModelRuntimeConfig(active_model="qwen2.5:1.5b", fallback_models=["qwen2.5:0.5b"])
    resolver = ModelResolver(config=cfg, client=mock_client)

    # By default, allow_fallback is False to prevent silent switching
    res = resolver.resolve(allow_fallback=False)

    assert res.is_ready is False
    assert res.effective_model is None
    assert res.ollama_online is True
    assert "Run 'ollama pull qwen2.5:1.5b'" in res.message
    assert "qwen2.5:0.5b" in res.installed_models


def test_resolver_model_missing_with_fallback_available():
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.check_health.return_value = True
    mock_client.list_local_models.return_value = ["qwen2.5:0.5b"]

    def fake_available(m: str) -> bool:
        return m == "qwen2.5:0.5b"

    mock_client.is_model_available.side_effect = fake_available

    cfg = ModelRuntimeConfig(active_model="qwen2.5:1.5b", fallback_models=["qwen2.5:0.5b"])
    resolver = ModelResolver(config=cfg, client=mock_client)

    res = resolver.resolve(allow_fallback=True)

    assert res.is_ready is True
    assert res.effective_model == "qwen2.5:0.5b"
    assert res.using_fallback is True
    assert "approved fallback" in res.message


def test_resolver_model_missing_with_fallback_also_missing():
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.check_health.return_value = True
    mock_client.list_local_models.return_value = ["unrelated:latest"]
    mock_client.is_model_available.return_value = False

    cfg = ModelRuntimeConfig(active_model="qwen2.5:1.5b", fallback_models=["qwen2.5:0.5b"])
    resolver = ModelResolver(config=cfg, client=mock_client)

    res = resolver.resolve(allow_fallback=True)

    assert res.is_ready is False
    assert res.effective_model is None
    assert "Run 'ollama pull qwen2.5:1.5b'" in res.message


def test_resolver_resolve_for_spec():
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.check_health.return_value = True
    mock_client.list_local_models.return_value = ["qwen2.5:7b"]
    mock_client.is_model_available.return_value = True

    spec = HardwareSpec(
        total_ram_gb=32.0,
        available_ram_gb=24.0,
        cpu_cores=16,
        cpu_threads=32,
        cpu_arch="AMD64",
        has_avx2=True,
        all_gpus=[GPUInfo(name="RTX 4080", vram_gb=16.0, is_dedicated=True)],
    )

    resolver = ModelResolver(client=mock_client)
    res = resolver.resolve_for_spec(spec)

    assert res.is_ready is True
    assert res.effective_model == "qwen2.5:7b"


def test_resolver_connection_failure_reported_as_offline():
    """Verify communication failure is reported as offline rather than missing model."""
    from loki.core.exceptions import ModelInferenceError

    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.list_local_models.side_effect = ModelInferenceError("Connection dropped")

    resolver = ModelResolver(client=mock_client)
    res = resolver.resolve()

    assert res.is_ready is False
    assert res.ollama_online is False
    assert "Failed to communicate with local Ollama daemon" in res.message
    assert "is not installed" not in res.message
