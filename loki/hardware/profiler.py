"""Hardware profiler integrating detection, tier classification, and configuration."""

from __future__ import annotations

from typing import Any
from ..core.config import LokiConfig
from .classifier import ClassificationResult, HardwareTierClassifier
from .detector import detect_hardware
from .models import HardwareSpec, HardwareTierThresholds


class HardwareProfiler:
    """High-level service coordinating local hardware detection and profile setup."""

    def __init__(self, thresholds: HardwareTierThresholds | None = None) -> None:
        self.classifier = HardwareTierClassifier(thresholds)

    def profile_system(self, mock_spec: HardwareSpec | None = None) -> ClassificationResult:
        """Run inspection (or use mock spec) and classify the system."""
        spec = mock_spec if mock_spec is not None else detect_hardware()
        return self.classifier.classify(spec)

    def configure_loki(
        self,
        config: LokiConfig | None = None,
        mock_spec: HardwareSpec | None = None,
        base_dir: str | None = None,
    ) -> tuple[LokiConfig, ClassificationResult]:
        """Profile the hardware and apply calibrated settings to a LokiConfig instance.

        Returns the updated LokiConfig and the ClassificationResult.
        """
        result = self.profile_system(mock_spec=mock_spec)
        spec = result.spec

        # Create or update LokiConfig using the tier preset
        target_config = (
            config
            if config is not None
            else LokiConfig.create_for_tier(result.tier, base_dir=base_dir)
        )

        # Synchronize hardware properties
        target_config.hardware.tier = result.tier
        target_config.hardware.ram_gb = spec.total_ram_gb
        target_config.hardware.vram_gb = spec.dedicated_vram_gb
        target_config.hardware.has_dedicated_gpu = spec.has_dedicated_gpu
        target_config.hardware.cpu_cores = spec.cpu_cores
        target_config.models.execution_device = result.recommended_device

        # If a specific model was suggested and the config was freshly created
        if config is None:
            target_config.models.active_model = result.suggested_model

        return target_config, result

    @staticmethod
    def format_summary(result: ClassificationResult) -> str:
        """Generate a clean, human-readable summary formatted for setup or logs."""
        spec = result.spec
        gpu_str = (
            f"{spec.primary_gpu.name} ({spec.primary_gpu.vram_gb:.1f} GB VRAM)"
            if spec.primary_gpu
            else "None / Integrated"
        )
        avx2_str = "Supported" if spec.has_avx2 else "Not Supported"

        return (
            "=================== LOKI Hardware Profile ===================\n"
            f"Physical RAM     : {spec.total_ram_gb:.1f} GB (Available: {spec.available_ram_gb:.1f} GB)\n"
            f"CPU              : {spec.cpu_cores} Cores, {spec.cpu_threads} Threads ({spec.cpu_arch})\n"
            f"AVX2 Support     : {avx2_str}\n"
            f"GPU              : {gpu_str}\n"
            f"Hardware Tier    : {result.tier.value.upper()}\n"
            f"Execution Mode   : {result.recommended_device.value.upper()}\n"
            f"Recommended Model: {result.suggested_model}\n"
            f"Rationale        : {result.reason}\n"
            "============================================================="
        )
