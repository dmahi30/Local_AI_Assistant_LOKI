"""Data models for hardware detection and classification."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from ..core.types import HardwareTier


class GPUInfo(BaseModel):
    """Information about a detected display adapter / GPU."""
    name: str = Field(..., description="GPU device or driver name")
    vram_gb: float = Field(default=0.0, description="Dedicated video memory in GB")
    is_dedicated: bool = Field(default=False, description="Whether this is a dedicated discrete GPU")
    detection_source: str = Field(default="unknown", description="Source method that detected this GPU")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Raw vendor or driver metadata")


class HardwareSpec(BaseModel):
    """Complete snapshot of local host hardware."""
    total_ram_gb: float = Field(..., ge=0.0, description="Total physical system RAM in GB")
    available_ram_gb: float = Field(default=0.0, ge=0.0, description="Available/free system RAM in GB")
    cpu_cores: int = Field(default=1, ge=1, description="Physical CPU cores")
    cpu_threads: int = Field(default=1, ge=1, description="Logical CPU threads/processors")
    cpu_arch: str = Field(default="x86_64", description="CPU Architecture (e.g. AMD64, ARM64)")
    cpu_model: str = Field(default="", description="CPU model name string")
    has_avx2: bool = Field(default=False, description="Whether AVX2 vector instructions are supported")
    primary_gpu: GPUInfo | None = Field(default=None, description="Primary dedicated or active GPU")
    all_gpus: list[GPUInfo] = Field(default_factory=list, description="All detected display adapters")

    @property
    def has_dedicated_gpu(self) -> bool:
        """Return True if any dedicated discrete GPU is found."""
        return any(gpu.is_dedicated for gpu in self.all_gpus)

    @property
    def dedicated_vram_gb(self) -> float:
        """Return the dedicated VRAM in GB of the best dedicated GPU found."""
        dedicated = [gpu.vram_gb for gpu in self.all_gpus if gpu.is_dedicated]
        return max(dedicated) if dedicated else 0.0


class HardwareTierThresholds(BaseModel):
    """Configurable memory and hardware thresholds for tier classification."""
    # Power Tier (Tier 3)
    power_min_ram_gb: float = Field(default=14.5, description="Min RAM for Tier 3 Power (nominally 16GB+)")
    power_min_vram_gb: float = Field(default=7.0, description="Min VRAM for Tier 3 Power (nominally 8GB+)")

    # Balanced GPU Tier (Tier 2B)
    balanced_gpu_min_ram_gb: float = Field(default=11.0, description="Min RAM for Tier 2B Balanced GPU (nominally 16GB)")
    balanced_gpu_min_vram_gb: float = Field(default=3.4, description="Min VRAM for Tier 2B (nominally 4-6GB)")

    # Balanced CPU Tier (Tier 2A)
    balanced_cpu_min_ram_gb: float = Field(default=11.0, description="Min RAM for Tier 2A Balanced CPU")

    # Eco Tier (Tier 1) ceiling
    eco_max_ram_gb: float = Field(default=10.5, description="Max RAM considered Eco tier (nominally 8GB)")
