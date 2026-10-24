"""Structured domain exceptions for LOKI."""

from __future__ import annotations

from typing import Any


class LokiError(Exception):
    """Base exception for all LOKI system errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert error to a structured dictionary for logging and UI reporting."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


# Configuration & Environment
class ConfigurationError(LokiError):
    """Raised when configuration is invalid, corrupted, or missing required keys."""


class HardwareDetectionError(LokiError):
    """Raised when hardware inspection fails to detect memory, CPU, or GPU."""


# Model & Inference
class ModelError(LokiError):
    """Base exception for local model and Ollama operations."""


class ModelNotFoundError(ModelError):
    """Raised when the specified local model is not downloaded or accessible."""


class ModelInferenceError(ModelError):
    """Raised when local model inference fails, times out, or encounters OOM."""


class ModelOutputParseError(ModelError):
    """Raised when model returns invalid JSON or violates required tool schemas."""


class ModelFallbackError(ModelError):
    """Raised when the primary model and all available fallbacks have failed."""


# Tools & Safety
class ToolError(LokiError):
    """Base exception for computer tool operations."""


class ToolNotFoundError(ToolError):
    """Raised when an unregistered tool is requested."""


class ToolExecutionError(ToolError):
    """Raised when an OS or file tool operation fails during execution."""


class SafetyViolationError(ToolError):
    """Raised when a dangerous action is attempted without valid confirmation."""


# Storage & Memory
class StorageError(LokiError):
    """Raised when local SQLite or file storage fails."""


class ConversationNotFoundError(StorageError):
    """Raised when a requested conversation ID does not exist."""
