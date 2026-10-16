"""Core types, enumerations, and data schemas for LOKI."""

from __future__ import annotations

from enum import Enum, IntEnum
import time
from typing import Any
from pydantic import BaseModel, Field


class HardwareTier(str, Enum):
    """Hardware capability tiers determining model assignments."""
    ECO = "tier_1_eco"                 # 8 GB RAM, CPU-only / integrated GPU
    BALANCED_CPU = "tier_2a_balanced"   # 16 GB RAM, CPU-only
    BALANCED_GPU = "tier_2b_balanced"   # 16 GB RAM, 4-6 GB dedicated VRAM
    POWER = "tier_3_power"              # 16-32+ GB RAM, 8+ GB dedicated VRAM
    UNKNOWN = "unknown"


class ExecutionDevice(str, Enum):
    """Device target for local model inference."""
    CPU = "cpu"
    GPU = "gpu"
    HYBRID = "hybrid"


class SafetyLevel(IntEnum):
    """Security classification for tool actions."""
    SAFE = 1        # Read-only actions (telemetry, search, opening apps)
    NOTIFY = 2      # Low-risk modifications (creating new file, organizing)
    DANGEROUS = 3   # High-risk / Destructive actions (deletions, terminations, scripts)


class IntentType(str, Enum):
    """Categorized user intent driving the hybrid router."""
    DETERMINISTIC_APP = "deterministic_app"
    DETERMINISTIC_SYSTEM = "deterministic_system"
    DETERMINISTIC_MEDIA = "deterministic_media"
    FILE_SEARCH = "file_search"
    DOCUMENT_QUERY = "document_query"
    CODE_ASSIST = "code_assist"
    MULTI_STEP_AUTOMATION = "multi_step_automation"
    GENERAL_CHAT = "general_chat"


class ToolResult(BaseModel):
    """Standard output envelope returned by all LOKI tools."""
    success: bool = Field(..., description="Whether the tool executed without error")
    data: Any = Field(default=None, description="Structured payload or output value")
    error: str | None = Field(default=None, description="Human-readable error message if failed")
    safety_level: SafetyLevel = Field(default=SafetyLevel.SAFE, description="Safety tier of the tool")
    execution_time_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Auxiliary execution metadata")


class ModelConfig(BaseModel):
    """Configuration for an Ollama-compatible local model."""
    model_name: str
    context_length: int = 32768
    quantization: str = "Q4_K_M"
    execution_device: ExecutionDevice = ExecutionDevice.CPU
    temperature: float = 0.2
    top_p: float = 0.9
    timeout_seconds: int = 60


class SafetyConfirmationRequest(BaseModel):
    """Payload representing a dangerous action awaiting explicit user confirmation."""
    action_id: str
    tool_name: str
    description: str
    targets: list[str]
    safety_level: SafetyLevel = SafetyLevel.DANGEROUS
    confirmed: bool = False
    created_at: float = Field(default_factory=time.time, description="Creation timestamp in epoch seconds")
    ttl_seconds: int = Field(default=300, description="Time to live in seconds before expiration")
