"""Tests for LOKI core types, enums, and models."""

import pytest
from pydantic import ValidationError

from loki.core.types import (
    ExecutionDevice,
    HardwareTier,
    IntentType,
    ModelConfig,
    SafetyConfirmationRequest,
    SafetyLevel,
    ToolResult,
)


def test_hardware_tier_values():
    assert HardwareTier.ECO.value == "tier_1_eco"
    assert HardwareTier.BALANCED_CPU.value == "tier_2a_balanced"
    assert HardwareTier.BALANCED_GPU.value == "tier_2b_balanced"
    assert HardwareTier.POWER.value == "tier_3_power"


def test_safety_level_ordering():
    assert SafetyLevel.SAFE < SafetyLevel.NOTIFY < SafetyLevel.DANGEROUS
    assert SafetyLevel.SAFE == 1
    assert SafetyLevel.DANGEROUS == 3


def test_tool_result_defaults():
    res = ToolResult(success=True, data={"output": "done"})
    assert res.success is True
    assert res.data["output"] == "done"
    assert res.error is None
    assert res.safety_level == SafetyLevel.SAFE
    assert res.execution_time_ms == 0.0


def test_tool_result_failure():
    res = ToolResult(
        success=False,
        error="Access denied",
        safety_level=SafetyLevel.DANGEROUS,
    )
    assert res.success is False
    assert res.error == "Access denied"
    assert res.safety_level == SafetyLevel.DANGEROUS


def test_model_config_validation():
    cfg = ModelConfig(
        model_name="qwen2.5:3b",
        execution_device=ExecutionDevice.CPU,
        temperature=0.3,
    )
    assert cfg.model_name == "qwen2.5:3b"
    assert cfg.context_length == 32768
    assert cfg.quantization == "Q4_K_M"


def test_safety_confirmation_request():
    req = SafetyConfirmationRequest(
        action_id="act_123",
        tool_name="delete_files",
        description="Delete duplicate files",
        targets=["file1.pdf", "file2.pdf"],
    )
    assert req.action_id == "act_123"
    assert req.confirmed is False
    assert req.safety_level == SafetyLevel.DANGEROUS
    assert len(req.targets) == 2
