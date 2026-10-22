"""LOKI unified agent orchestrator.

Coordinates:
1. Deterministic Command Engine: Zero-overhead, zero-LLM command matching for OS operations.
2. Safety Gatekeeper: Fail-closed risk evaluation, path boundaries, and confirmation tokens.
3. Ollama Local LLM: Local-only, private text generation and conversational reasoning.
4. Model Resolver: Validates model availability locally; prevents unauthorized downloads or switches.

CRITICAL SECURITY INVARIANT:
LLM output is strictly treated as conversational text. LLM responses are NEVER
fed back into command execution, subprocess calls, file operations, or shell execution.
All system operations MUST originate directly from explicit deterministic user intents.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

from .commands.router import DeterministicRouter
from .core.config import LokiConfig
from .core.exceptions import (
    ModelFallbackError,
    ModelInferenceError,
    ModelNotFoundError,
    ModelOutputParseError,
)
from .core.logging import get_logger
from .core.types import SafetyConfirmationRequest, ToolResult
from .llm.client import OllamaClient
from .llm.resolver import ModelResolver
from .safety.gatekeeper import SafetyGatekeeper
from .utils.paths import get_config_path

logger = get_logger("agent")

DEFAULT_SYSTEM_PROMPT = (
    "You are LOKI, an offline, privacy-first local AI desktop assistant running entirely "
    "on the user's machine. Provide helpful, concise, and accurate answers."
)


class ResponseSource(str, Enum):
    """Source that generated the agent response."""
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    SYSTEM = "system"


class AgentResponse(BaseModel):
    """Structured response returned by the LokiAgent."""
    text: str = Field(..., description="Conversational text output shown to user")
    source: ResponseSource = Field(..., description="Origin of the response")
    matched_command: str | None = Field(default=None, description="Command name if matched deterministically")
    tool_result: ToolResult | None = Field(default=None, description="Raw tool result if executed")
    pending_confirmation: SafetyConfirmationRequest | None = Field(
        default=None, description="Pending confirmation request if action is gated"
    )
    model_used: str | None = Field(default=None, description="Name of LLM model used if LLM responded")
    success: bool = Field(default=True, description="Whether the interaction completed successfully")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context or diagnostics")


class LokiAgent:
    """Unified offline desktop assistant coordinating deterministic commands and local LLM."""

    def __init__(
        self,
        config: LokiConfig | None = None,
        router: DeterministicRouter | None = None,
        gatekeeper: SafetyGatekeeper | None = None,
        client: OllamaClient | None = None,
        resolver: ModelResolver | None = None,
        system_prompt: str | None = None,
        allow_fallback: bool = False,
    ) -> None:
        if config is None:
            config_file = get_config_path()
            if config_file.exists():
                try:
                    config = LokiConfig.load(config_file)
                except Exception:
                    config = LokiConfig()
            else:
                config = LokiConfig()
        self.config = config
        self.gatekeeper = gatekeeper or SafetyGatekeeper(config=self.config.safety)
        self.router = router or DeterministicRouter(gatekeeper=self.gatekeeper)
        self.client = client or OllamaClient(self.config.models)
        self.resolver = resolver or ModelResolver(config=self.config.models, client=self.client)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.allow_fallback = allow_fallback

        self._conversation_history: list[dict[str, str]] = []

    def confirm_action(
        self,
        action_id: str,
        expected_tool: str | None = None,
        expected_targets: list[str] | None = None,
    ) -> bool:
        """Approve a pending gated confirmation request."""
        return self.gatekeeper.confirm_action(
            action_id=action_id,
            expected_tool=expected_tool,
            expected_targets=expected_targets,
        )

    def cancel_action(self, action_id: str) -> bool:
        """Cancel a pending gated confirmation request."""
        return self.gatekeeper.cancel_action(action_id=action_id)

    def clear_history(self) -> None:
        """Reset conversational context history."""
        self._conversation_history.clear()

    def get_history(self) -> list[dict[str, str]]:
        """Return a copy of the current conversational context history."""
        return list(self._conversation_history)

    @staticmethod
    def _format_tool_result(res: ToolResult) -> str:
        """Format structured ToolResult.data into user-friendly text."""
        data = res.data
        if not data:
            return "Command executed successfully."
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            if "status" in data and data["status"] == "launched":
                return f"Launched {data.get('app', 'application')} successfully."
            if "deleted_file" in data:
                return f"File deleted successfully: {data.get('deleted_file')}"
            if "battery" in data:
                b = data["battery"]
                if "percent" in b:
                    plugged = "Plugged in" if b.get("power_plugged") else "On battery"
                    return f"Battery: {b['percent']}% ({plugged})"
                return str(b.get("status", "No battery detected."))
            if "ram" in data:
                r = data["ram"]
                return f"RAM Usage: {r.get('used_gb')} GB / {r.get('total_gb')} GB ({r.get('percent_used')}%)"
            if "cpu" in data:
                c = data["cpu"]
                return f"CPU Usage: {c.get('usage_percent')}% across {c.get('cores')} cores"
            if "storage" in data:
                s = data["storage"]
                return f"Storage ({s.get('drive')}): {s.get('free_gb')} GB free of {s.get('total_gb')} GB"
            return ", ".join(f"{k}: {v}" for k, v in data.items())
        if isinstance(data, list):
            if not data:
                return "No items found."
            return "\n".join(str(item) for item in data)
        return str(data)

    def handle_input(
        self,
        text: str,
        action_id: str | None = None,
        chat_mode: bool = True,
    ) -> AgentResponse:
        """Process user input through deterministic routing, falling back to local LLM.

        Security & Architecture Guarantees:
        - Deterministic commands execute with ZERO LLM involvement.
        - Destructive operations return pending confirmation before executing.
        - Non-deterministic queries escalate to local Ollama only after model verification.
        - If Ollama is offline or model is missing, returns clear diagnostics and NEVER
          triggers unapproved downloads or silent model switches.
        - LLM output is strictly treated as text and never executed as code or commands.
        """
        clean_input = text.strip()
        if not clean_input:
            return AgentResponse(
                text="Please provide a command or question.",
                source=ResponseSource.SYSTEM,
                success=False,
            )

        # =========================================================================
        # Stage 0: Deterministic Command Engine (LLM Bypass)
        # =========================================================================
        route_result = self.router.route(clean_input, action_id=action_id)

        if route_result.matched:
            # Case 1: Gated action requiring explicit user authorization
            if route_result.pending_confirmation is not None:
                req = route_result.pending_confirmation
                targets_str = ", ".join(req.targets) if req.targets else "none"
                confirm_msg = (
                    f"Action '{req.tool_name}' requires your confirmation:\n"
                    f"- Description: {req.description}\n"
                    f"- Targets: {targets_str}\n"
                    f"- Action ID: {req.action_id}\n"
                    f"To proceed, confirm action ID '{req.action_id}'."
                )
                return AgentResponse(
                    text=confirm_msg,
                    source=ResponseSource.SYSTEM,
                    matched_command=route_result.command_name,
                    pending_confirmation=req,
                    success=True,
                    metadata={"action_id": req.action_id},
                )

            # Case 2: Deterministic command executed
            if route_result.result is not None:
                res = route_result.result
                if res.success:
                    output_text = self._format_tool_result(res)
                else:
                    output_text = f"Command failed: {res.error}"

                return AgentResponse(
                    text=output_text,
                    source=ResponseSource.DETERMINISTIC,
                    matched_command=route_result.command_name,
                    tool_result=res,
                    success=res.success,
                    metadata={"data": res.data},
                )

        # =========================================================================
        # Stage 1: Fallthrough to Local AI LLM
        # =========================================================================
        logger.info("Input did not match deterministic commands. Escalating to local LLM.")

        # 1. Verify model readiness locally (fail-closed if offline or missing)
        resolution = self.resolver.resolve(allow_fallback=self.allow_fallback)
        if not resolution.is_ready:
            return AgentResponse(
                text=resolution.message,
                source=ResponseSource.SYSTEM,
                success=False,
                metadata={
                    "ollama_online": resolution.ollama_online,
                    "installed_models": resolution.installed_models,
                },
            )

        effective_model = resolution.effective_model

        # 2. Local AI Inference
        if chat_mode:
            self._conversation_history.append({"role": "user", "content": clean_input})
            messages = [{"role": "system", "content": self.system_prompt}] + self._conversation_history

            try:
                reply = self.client.chat(messages, model=effective_model)
                self._conversation_history.append({"role": "assistant", "content": reply})
                return AgentResponse(
                    text=reply,
                    source=ResponseSource.LLM,
                    model_used=effective_model,
                    success=True,
                    metadata={"using_fallback": resolution.using_fallback},
                )
            except (ModelNotFoundError, ModelInferenceError, ModelOutputParseError, ModelFallbackError) as e:
                # Rollback user message from context to prevent poisoned history
                self._conversation_history.pop()
                logger.error("Local LLM chat failed: %s", e)
                return AgentResponse(
                    text=f"AI inference error: {e}",
                    source=ResponseSource.SYSTEM,
                    model_used=effective_model,
                    success=False,
                    metadata={"error": str(e)},
                )
        else:
            try:
                reply = self.client.generate(
                    prompt=clean_input,
                    model=effective_model,
                    system=self.system_prompt,
                )
                return AgentResponse(
                    text=reply,
                    source=ResponseSource.LLM,
                    model_used=effective_model,
                    success=True,
                    metadata={"using_fallback": resolution.using_fallback},
                )
            except (ModelNotFoundError, ModelInferenceError, ModelOutputParseError, ModelFallbackError) as e:
                logger.error("Local LLM generation failed: %s", e)
                return AgentResponse(
                    text=f"AI inference error: {e}",
                    source=ResponseSource.SYSTEM,
                    model_used=effective_model,
                    success=False,
                    metadata={"error": str(e)},
                )
