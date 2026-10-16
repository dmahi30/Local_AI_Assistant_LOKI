"""Tests for LOKI hardware detection, classification, and profiling."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from loki.core.config import LokiConfig
from loki.core.exceptions import HardwareDetectionError
from loki.core.types import ExecutionDevice, HardwareTier
from loki.hardware.classifier import HardwareTierClassifier
from loki.hardware.detector import (
    _is_dedicated_gpu,
    detect_avx2,
    detect_cpu,
    detect_gpus,
    detect_hardware,
    detect_ram,
)
from loki.hardware.models import GPUInfo, HardwareSpec, HardwareTierThresholds
from loki.hardware.profiler import HardwareProfiler


# =========================================================================
# Tier Classification Tests (Mocked Hardware Specs)
# =========================================================================

def test_classify_power_tier():
    spec = HardwareSpec(
        total_ram_gb=32.0,
        available_ram_gb=24.0,
        cpu_cores=8,
        cpu_threads=16,
        primary_gpu=GPUInfo(name="NVIDIA RTX 4070", vram_gb=12.0, is_dedicated=True),
        all_gpus=[GPUInfo(name="NVIDIA RTX 4070", vram_gb=12.0, is_dedicated=True)],
    )
    classifier = HardwareTierClassifier()
    result = classifier.classify(spec)

    assert result.tier == HardwareTier.POWER
    assert result.recommended_device == ExecutionDevice.GPU
    assert result.suggested_model == "qwen2.5:7b"
    assert "High-end" in result.reason


def test_classify_balanced_gpu_tier():
    spec = HardwareSpec(
        total_ram_gb=16.0,
        available_ram_gb=10.0,
        cpu_cores=6,
        cpu_threads=12,
        primary_gpu=GPUInfo(name="NVIDIA RTX 3050", vram_gb=4.0, is_dedicated=True),
        all_gpus=[GPUInfo(name="NVIDIA RTX 3050", vram_gb=4.0, is_dedicated=True)],
    )
    classifier = HardwareTierClassifier()
    result = classifier.classify(spec)

    assert result.tier == HardwareTier.BALANCED_GPU
    assert result.recommended_device == ExecutionDevice.GPU
    assert result.suggested_model == "qwen2.5:7b"
    assert "Mid-range GPU" in result.reason


def test_classify_balanced_cpu_tier():
    spec = HardwareSpec(
        total_ram_gb=16.0,
        available_ram_gb=11.0,
        cpu_cores=8,
        cpu_threads=16,
        primary_gpu=GPUInfo(name="Intel Iris Xe", vram_gb=0.0, is_dedicated=False),
        all_gpus=[GPUInfo(name="Intel Iris Xe", vram_gb=0.0, is_dedicated=False)],
    )
    classifier = HardwareTierClassifier()
    result = classifier.classify(spec)

    assert result.tier == HardwareTier.BALANCED_CPU
    assert result.recommended_device == ExecutionDevice.CPU
    assert result.suggested_model == "qwen2.5:3b"
    assert "Mid-range CPU" in result.reason


def test_classify_eco_tier_cpu_only():
    spec = HardwareSpec(
        total_ram_gb=8.0,
        available_ram_gb=3.5,
        cpu_cores=4,
        cpu_threads=4,
        primary_gpu=None,
        all_gpus=[],
    )
    classifier = HardwareTierClassifier()
    result = classifier.classify(spec)

    assert result.tier == HardwareTier.ECO
    assert result.recommended_device == ExecutionDevice.CPU
    assert result.suggested_model == "qwen2.5:1.5b"
    assert "Eco mode" in result.reason


def test_classify_eco_tier_with_gpu():
    spec = HardwareSpec(
        total_ram_gb=7.86,
        available_ram_gb=2.0,
        cpu_cores=4,
        cpu_threads=8,
        primary_gpu=GPUInfo(name="NVIDIA GTX 1050", vram_gb=4.0, is_dedicated=True),
        all_gpus=[GPUInfo(name="NVIDIA GTX 1050", vram_gb=4.0, is_dedicated=True)],
    )
    classifier = HardwareTierClassifier()
    result = classifier.classify(spec)

    assert result.tier == HardwareTier.ECO
    assert result.recommended_device == ExecutionDevice.GPU
    assert "GPU acceleration" in result.reason


def test_custom_thresholds():
    # Lower power tier requirement to 8GB RAM + 3GB VRAM
    custom_thresholds = HardwareTierThresholds(
        power_min_ram_gb=8.0,
        power_min_vram_gb=3.0,
    )
    spec = HardwareSpec(
        total_ram_gb=8.0,
        primary_gpu=GPUInfo(name="Test GPU", vram_gb=4.0, is_dedicated=True),
        all_gpus=[GPUInfo(name="Test GPU", vram_gb=4.0, is_dedicated=True)],
    )
    classifier = HardwareTierClassifier(thresholds=custom_thresholds)
    result = classifier.classify(spec)

    assert result.tier == HardwareTier.POWER


# =========================================================================
# Detector Unit & Graceful Failure Tests
# =========================================================================

def test_is_dedicated_gpu_heuristics():
    assert _is_dedicated_gpu("NVIDIA GeForce RTX 3060", 12.0) is True
    assert _is_dedicated_gpu("NVIDIA GeForce GTX 1650", 4.0) is True
    assert _is_dedicated_gpu("AMD Radeon RX 6600", 8.0) is True
    assert _is_dedicated_gpu("Intel(R) UHD Graphics 630", 0.0) is False
    assert _is_dedicated_gpu("Intel(R) Iris(R) Xe Graphics", 0.0) is False
    assert _is_dedicated_gpu("AMD Radeon Graphics", 0.5) is False


def test_detect_ram_mock_failure():
    with patch("psutil.virtual_memory", side_effect=RuntimeError("psutil error")):
        with patch("sys.platform", "linux"):
            with pytest.raises(HardwareDetectionError):
                detect_ram()


def test_detect_cpu_fallback():
    with patch("psutil.cpu_count", return_value=None):
        cores, threads, arch, _ = detect_cpu()
        assert cores >= 1
        assert threads >= 1
        assert isinstance(arch, str)


def test_detect_gpus_empty_graceful():
    with patch("loki.hardware.detector._detect_gpus_winreg", return_value=[]):
        with patch("loki.hardware.detector._detect_gpus_powershell", return_value=[]):
            with patch("loki.hardware.detector._detect_gpus_nvidia_smi", return_value=[]):
                gpus = detect_gpus()
                assert gpus == []


def test_detect_avx2_returns_bool():
    res = detect_avx2()
    assert isinstance(res, bool)


# =========================================================================
# Profiler & LokiConfig Integration Tests
# =========================================================================

def test_profiler_configure_loki(temp_loki_dir: Path):
    spec = HardwareSpec(
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        cpu_cores=8,
        cpu_threads=16,
        primary_gpu=GPUInfo(name="NVIDIA RTX 4060", vram_gb=8.0, is_dedicated=True),
        all_gpus=[GPUInfo(name="NVIDIA RTX 4060", vram_gb=8.0, is_dedicated=True)],
    )
    profiler = HardwareProfiler()
    config, result = profiler.configure_loki(mock_spec=spec, base_dir=str(temp_loki_dir))

    assert config.hardware.tier == HardwareTier.POWER
    assert config.hardware.ram_gb == 16.0
    assert config.hardware.vram_gb == 8.0
    assert config.hardware.has_dedicated_gpu is True
    assert config.models.execution_device == ExecutionDevice.GPU

    summary = profiler.format_summary(result)
    assert "LOKI Hardware Profile" in summary
    assert "16.0 GB" in summary
    assert "POWER" in summary


# =========================================================================
# Live Host Hardware Inspection Test
# =========================================================================

def test_live_hardware_detection():
    spec = detect_hardware()
    assert spec.total_ram_gb > 0.0
    assert spec.available_ram_gb >= 0.0
    assert spec.cpu_cores >= 1
    assert spec.cpu_threads >= spec.cpu_cores
    assert spec.cpu_arch != ""
    assert isinstance(spec.has_avx2, bool)
    assert isinstance(spec.all_gpus, list)
