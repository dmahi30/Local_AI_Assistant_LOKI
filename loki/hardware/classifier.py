"""Hardware tier classifier for LOKI.

Maps hardware specifications to optimal tiers (ECO, BALANCED_CPU, BALANCED_GPU, POWER)
using configurable thresholds and explains the assignment rationale.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.types import ExecutionDevice, HardwareTier
from .models import HardwareSpec, HardwareTierThresholds


class ClassificationResult(BaseModel):
    """Result of classifying a hardware specification into an operational tier."""
    tier: HardwareTier = Field(..., description="Assigned hardware tier")
    recommended_device: ExecutionDevice = Field(..., description="Recommended execution device (CPU or GPU)")
    reason: str = Field(..., description="Human-readable rationale for this tier selection")
    suggested_model: str = Field(..., description="Suggested primary model name for this tier")
    spec: HardwareSpec = Field(..., description="The hardware spec evaluated")
    thresholds: HardwareTierThresholds = Field(..., description="Thresholds used for classification")


class HardwareTierClassifier:
    """Configurable classifier that maps HardwareSpec to HardwareTier."""

    def __init__(self, thresholds: HardwareTierThresholds | None = None) -> None:
        self.thresholds = thresholds or HardwareTierThresholds()

    def classify(self, spec: HardwareSpec) -> ClassificationResult:
        """Evaluate a hardware specification and return the classification result."""
        ram = spec.total_ram_gb
        vram = spec.dedicated_vram_gb
        has_gpu = spec.has_dedicated_gpu
        t = self.thresholds

        # 1. Tier 3: Power (16-32+ GB RAM and 8+ GB VRAM)
        if ram >= t.power_min_ram_gb and vram >= t.power_min_vram_gb:
            return ClassificationResult(
                tier=HardwareTier.POWER,
                recommended_device=ExecutionDevice.GPU,
                reason=(
                    f"High-end hardware detected ({ram:.1f} GB RAM, {vram:.1f} GB dedicated VRAM). "
                    "System has ample headroom for full 7B/14B models in GPU memory."
                ),
                suggested_model="qwen2.5:7b",
                spec=spec,
                thresholds=t,
            )

        # 2. Tier 2B: Balanced GPU (16 GB RAM with 4-6 GB VRAM)
        if ram >= t.balanced_gpu_min_ram_gb and vram >= t.balanced_gpu_min_vram_gb:
            return ClassificationResult(
                tier=HardwareTier.BALANCED_GPU,
                recommended_device=ExecutionDevice.GPU,
                reason=(
                    f"Mid-range GPU system detected ({ram:.1f} GB RAM, {vram:.1f} GB dedicated VRAM). "
                    "Optimized for 7B models with GPU layer offload."
                ),
                suggested_model="qwen2.5:7b",
                spec=spec,
                thresholds=t,
            )

        # 3. Tier 2A: Balanced CPU (16 GB RAM, CPU-only or integrated graphics)
        if ram >= t.balanced_cpu_min_ram_gb:
            return ClassificationResult(
                tier=HardwareTier.BALANCED_CPU,
                recommended_device=ExecutionDevice.CPU,
                reason=(
                    f"Mid-range CPU system detected ({ram:.1f} GB RAM, integrated/low VRAM). "
                    "Optimized for fast 3B models to avoid CPU latency on 7B models."
                ),
                suggested_model="qwen2.5:3b",
                spec=spec,
                thresholds=t,
            )

        # 4. Tier 1: Eco (<= 8-10 GB RAM)
        # Check if an Eco machine still has a modest GPU for acceleration
        rec_dev = ExecutionDevice.GPU if (has_gpu and vram >= 2.0) else ExecutionDevice.CPU
        gpu_note = f" with {vram:.1f} GB GPU acceleration" if (has_gpu and vram >= 2.0) else ""

        return ClassificationResult(
            tier=HardwareTier.ECO,
            recommended_device=rec_dev,
            reason=(
                f"Budget or memory-constrained system detected ({ram:.1f} GB RAM{gpu_note}). "
                "Operating in Eco mode with compact 1.5B/3B models to preserve system stability."
            ),
            suggested_model="qwen2.5:1.5b",
            spec=spec,
            thresholds=t,
        )
