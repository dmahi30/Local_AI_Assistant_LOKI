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
    allow_model_selection: bool = False

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.patterns
        ]

    def can_handle(self, text: str) -> bool:
        """Check if this command matches the input text."""
        cleaned = text.strip()
        return any(p.search(cleaned) is not None for p in self._compiled_patterns)

    def get_parameters_schema(self) -> dict[str, Any]:
        """Return JSON schema defining valid arguments for structured tool calls."""
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate input arguments before execution. Subclasses may override."""
        if not isinstance(args, dict):
            return False, f"Arguments must be a dictionary, got {type(args).__name__}"
        return True, None

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
            # Validate input arguments
            is_valid, validation_err = self.validate_args(args)
            if not is_valid:
                return ToolResult(
                    success=False,
                    error=validation_err or "Invalid command arguments.",
                    safety_level=self.safety_level,
                    execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
                    metadata={"command": self.name},
                )

            # Extract target paths from args
            raw_targets = args.get("targets")
            if raw_targets and isinstance(raw_targets, (list, tuple)):
                targets = [str(t) for t in raw_targets if t]
            elif raw_targets and isinstance(raw_targets, str) and raw_targets.strip():
                targets = [raw_targets.strip()]
            elif "path" in args and args["path"]:
                targets = [str(args["path"]).strip()]
            elif "target" in args and args["target"]:
                targets = [str(args["target"]).strip()]
            else:
                targets = None

            # Enforce fail-closed safety verification
            if gatekeeper is None:
                if self.safety_level == SafetyLevel.DANGEROUS:
                    raise SafetyViolationError(
                        f"Security invariant violated: Dangerous command '{self.name}' requires an active SafetyGatekeeper.",
                        details={"command": self.name, "safety_level": self.safety_level.value},
                    )
                if targets:
                    # Guard against direct execution bypassing gatekeeper risk escalation
                    default_gk = SafetyGatekeeper()
                    effective_safety = default_gk.evaluate_risk(
                        tool_name=self.name,
                        declared_level=self.safety_level,
                        targets=targets,
                    )
                    if effective_safety == SafetyLevel.DANGEROUS:
                        raise SafetyViolationError(
                            f"Security invariant violated: Dangerous operation for command '{self.name}' requires an active SafetyGatekeeper.",
                            details={"command": self.name, "effective_safety": effective_safety.value, "targets": targets},
                        )
            else:
                # Re-evaluate effective risk dynamically against targets
                effective_safety = gatekeeper.evaluate_risk(
                    tool_name=self.name,
                    declared_level=self.safety_level,
                    targets=targets,
                )
                gatekeeper.verify_execution_permitted(
                    tool_name=self.name,
                    safety_level=effective_safety,
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
