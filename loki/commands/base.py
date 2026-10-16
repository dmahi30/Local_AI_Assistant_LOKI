"""Base command definition for LOKI's deterministic execution engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
import time
from typing import Any

from ..core.exceptions import SafetyViolationError
from ..core.types import IntentType, SafetyLevel, ToolResult
from ..safety.gatekeeper import SafetyGatekeeper


class BaseCommand(ABC):
    """Abstract base class for all deterministic commands."""

    name: str
    description: str
    intent_type: IntentType
    safety_level: SafetyLevel = SafetyLevel.SAFE
    patterns: list[str] = []

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.patterns
        ]

    def can_handle(self, text: str) -> bool:
        """Check if this command matches the input text."""
        cleaned = text.strip()
        return any(p.search(cleaned) is not None for p in self._compiled_patterns)

    @abstractmethod
    def extract_args(self, text: str) -> dict[str, Any]:
        """Parse structured arguments from the natural language text."""
        ...

    @abstractmethod
    def run(
        self,
        args: dict[str, Any],
        gatekeeper: SafetyGatekeeper | None = None,
        action_id: str | None = None,
    ) -> ToolResult:
        """Execute the command logic and return a structured ToolResult."""
        ...

    def execute(
        self,
        args: dict[str, Any],
        gatekeeper: SafetyGatekeeper | None = None,
        action_id: str | None = None,
    ) -> ToolResult:
        """Execute with execution time tracking, fail-closed safety verification, and error preservation."""
        start_time = time.perf_counter()
        try:
            # Enforce fail-closed invariant for dangerous operations
            if self.safety_level == SafetyLevel.DANGEROUS and gatekeeper is None:
                raise SafetyViolationError(
                    f"Security invariant violated: Dangerous command '{self.name}' requires an active SafetyGatekeeper.",
                    details={"command": self.name, "safety_level": self.safety_level.value},
                )

            # Enforce gatekeeper permission if provided
            if gatekeeper is not None:
                targets = args.get("targets") or ([args["path"]] if "path" in args else None)
                gatekeeper.verify_execution_permitted(
                    tool_name=self.name,
                    safety_level=self.safety_level,
                    action_id=action_id,
                    targets=targets,
                )

            res = self.run(args, gatekeeper=gatekeeper, action_id=action_id)
            res.execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return res
        except SafetyViolationError as sve:
            duration = round((time.perf_counter() - start_time) * 1000, 2)
            return ToolResult(
                success=False,
                error=sve.message,
                safety_level=SafetyLevel.DANGEROUS,
                execution_time_ms=duration,
                metadata={
                    "command": self.name,
                    "safety_violation": sve.to_dict(),
                },
            )
        except Exception as e:
            duration = round((time.perf_counter() - start_time) * 1000, 2)
            return ToolResult(
                success=False,
                error=str(e),
                safety_level=self.safety_level,
                execution_time_ms=duration,
                metadata={"command": self.name},
            )
