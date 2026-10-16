"""Tests for LOKI exception hierarchy and serialization."""

from loki.core.exceptions import (
    ConfigurationError,
    HardwareDetectionError,
    LokiError,
    ModelError,
    ModelFallbackError,
    ModelInferenceError,
    SafetyViolationError,
    ToolError,
    ToolExecutionError,
)


def test_base_loki_error_serialization():
    err = LokiError("System fault", details={"code": 500, "component": "core"})
    assert str(err) == "System fault"
    assert err.message == "System fault"
    
    d = err.to_dict()
    assert d["error_type"] == "LokiError"
    assert d["message"] == "System fault"
    assert d["details"]["code"] == 500


def test_inheritance_hierarchy():
    assert issubclass(ConfigurationError, LokiError)
    assert issubclass(HardwareDetectionError, LokiError)
    assert issubclass(ModelInferenceError, ModelError)
    assert issubclass(ModelFallbackError, ModelError)
    assert issubclass(ModelError, LokiError)
    assert issubclass(ToolExecutionError, ToolError)
    assert issubclass(SafetyViolationError, ToolError)
    assert issubclass(ToolError, LokiError)


def test_specific_error_details():
    err = SafetyViolationError("Blocked deletion", details={"files": ["a.txt"]})
    assert err.details["files"] == ["a.txt"]
    d = err.to_dict()
    assert d["error_type"] == "SafetyViolationError"
