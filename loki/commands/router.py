"""Deterministic command router and dispatcher for LOKI."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from ..core.logging import get_logger
from ..core.types import SafetyConfirmationRequest, SafetyLevel, ToolResult
from ..safety.gatekeeper import SafetyGatekeeper
from .registry import CommandRegistry

logger = get_logger("commands.router")


class DeterministicRouteResult(BaseModel):
    """Result of routing a text query through the deterministic engine."""
    matched: bool = Field(..., description="Whether a registered deterministic command matched")
    command_name: str | None = Field(default=None, description="Name of the matched command")
    result: ToolResult | None = Field(default=None, description="Output if executed")
    pending_confirmation: SafetyConfirmationRequest | None = Field(
        default=None, description="Pending confirmation request if action is gated"
    )
    reason: str | None = Field(default=None, description="Diagnostic explanation or rejection reason")


class DeterministicRouter:
    """Routes user queries to deterministic commands with zero LLM overhead and safety gating."""

    def __init__(
        self,
        registry: CommandRegistry | None = None,
        gatekeeper: SafetyGatekeeper | None = None,
    ) -> None:
        self.registry = registry or CommandRegistry()
        self.gatekeeper = gatekeeper or SafetyGatekeeper()

    def route(self, text: str, action_id: str | None = None, conversation_id: str | None = None) -> DeterministicRouteResult:
        """Route input text to a deterministic command.

        If not matched, cleanly returns matched=False. Never executes arbitrary
        shell code or runs shell commands for unrecognized input.
        """
        match = self.registry.find_command(text)
        if not match:
            return DeterministicRouteResult(
                matched=False,
                reason="No matching deterministic command found. Input requires LLM reasoning or is unsupported.",
            )

        command, args = match
        targets = args.get("targets") or ([args["path"]] if "path" in args else None)
        effective_safety = self.gatekeeper.evaluate_risk(
            tool_name=command.name,
            declared_level=command.safety_level,
            targets=targets,
        )

        # If operation is dangerous and confirmation has not yet been approved
        if effective_safety == SafetyLevel.DANGEROUS and not action_id:
            pending_req = self.gatekeeper.request_confirmation(
                tool_name=command.name,
                description=f"Request to execute {command.name}",
                targets=targets or [],
                safety_level=effective_safety,
                conversation_id=conversation_id,
            )
            return DeterministicRouteResult(
                matched=True,
                command_name=command.name,
                pending_confirmation=pending_req,
                reason="Operation is classified as DANGEROUS and requires explicit user confirmation.",
            )

        # Execute command under gatekeeper supervision
        tool_result = command.execute(
            args=args,
            gatekeeper=self.gatekeeper,
            action_id=action_id,
        )

        return DeterministicRouteResult(
            matched=True,
            command_name=command.name,
            result=tool_result,
        )
