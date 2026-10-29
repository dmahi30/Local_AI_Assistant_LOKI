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
from pathlib import Path
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
from .storage.conversation import ConversationStore
from .utils.paths import get_config_path

logger = get_logger("agent")

DEFAULT_SYSTEM_PROMPT = (
    "You are LOKI, an offline, privacy-first local AI desktop assistant running entirely "
    "on the user's machine.\n"
    "Guidelines:\n"
    "- Provide direct, accurate, and concise answers to the user's question.\n"
    "- Avoid conversational filler, introductory preambles, and unnecessary disclaimers.\n"
    "- Avoid repeating sentences, phrases, or the user's prompt.\n"
    "- If you do not know the answer or lack local data, state so concisely rather than guessing.\n"
    "- You are a conversational language model: do not claim to execute operating system commands, "
    "launch applications, or modify files. System tools are handled deterministically outside the LLM."
)


class ResponseSource(str, Enum):
    """Source that generated the agent response."""
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    SYSTEM = "system"
    TOOL = "tool"


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
    conversation_id: str | None = Field(default=None, description="Active conversation identifier if tracked")
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
        storage: ConversationStore | None = None,
        active_conversation_id: str | None = None,
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
        self.storage = storage
        self.active_conversation_id = active_conversation_id
        self.max_history_messages = (
            getattr(self.config.models, "max_history_messages", 20)
            if self.config and hasattr(self.config, "models")
            else 20
        )

        self._conversation_history: list[dict[str, str]] = []
        if self.storage and self.active_conversation_id:
            self.switch_conversation(self.active_conversation_id)

    def switch_conversation(self, conversation_id: str) -> bool:
        """Switch active conversation, strictly isolating conversational context.

        Invariants:
        - Clears existing in-memory context before loading new context.
        - Guarantees zero context bleed across distinct conversations.
        - Loads strictly the target conversation's chronological turns.
        """
        self._conversation_history.clear()
        self.active_conversation_id = conversation_id
        if self.storage:
            detail = self.storage.get_conversation(conversation_id)
            if not detail:
                return False
            self._conversation_history = self.storage.get_context_messages(
                conversation_id, max_messages=self.max_history_messages
            )
            return True
        return True

    def confirm_action(
        self,
        action_id: str,
        expected_tool: str | None = None,
        expected_targets: list[str] | None = None,
        expected_conversation_id: str | None = None,
    ) -> bool:
        """Approve a pending gated confirmation request."""
        return self.gatekeeper.confirm_action(
            action_id=action_id,
            expected_tool=expected_tool,
            expected_targets=expected_targets,
            expected_conversation_id=expected_conversation_id,
        )

    def cancel_action(self, action_id: str) -> bool:
        """Cancel a pending gated confirmation request."""
        return self.gatekeeper.cancel_action(action_id=action_id)

    def clear_history(self) -> None:
        """Reset conversational context history."""
        self._conversation_history.clear()
        if self.storage and self.active_conversation_id:
            self.storage.clear_messages(self.active_conversation_id)
            self.storage.update_title(self.active_conversation_id, "New Chat")

    def _maybe_auto_title(
        self,
        conversation_id: str,
        prompt: str,
        attachment: dict[str, Any] | None = None,
    ) -> None:
        """Auto-generate a meaningful conversation title on first message turn."""
        if not self.storage or not conversation_id:
            return
        detail = self.storage.get_conversation(conversation_id)
        if not detail:
            return
        if detail.title in ("New Chat", "Untitled Conversation") and len(detail.messages) == 0:
            att_filename = attachment.get("filename") if (attachment and isinstance(attachment, dict)) else None
            clean_prompt = prompt.strip() if prompt else ""

            if att_filename:
                lower_p = clean_prompt.lower()
                is_generic = (
                    not clean_prompt
                    or lower_p in (
                        "summarize this file",
                        "summarize this file.",
                        "summarize this document",
                        "summarize this document.",
                        "summarize this pdf",
                        "summarize this pdf.",
                        "summarize this",
                        "summarize",
                        "explain this",
                        "read this",
                    )
                    or lower_p.startswith("summarize this ")
                    or lower_p == f"summarize {att_filename.lower()}."
                    or lower_p == f"summarize {att_filename.lower()}"
                )
                if is_generic:
                    candidate = f"Summarize {att_filename}"
                else:
                    candidate = f"{clean_prompt} ({att_filename})"
                auto_title = self.storage.generate_title_from_prompt(candidate)
            elif clean_prompt:
                auto_title = self.storage.generate_title_from_prompt(clean_prompt)
            else:
                auto_title = "New Chat"

            self.storage.update_title(conversation_id, auto_title)

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
            if "created" in data and "type" in data:
                item_type = data["type"]
                name = data.get("name", "item")
                path = data["created"]
                if item_type in ("folder", "directory"):
                    return f"Folder '{name}' created successfully at {path}."
                size_str = f" ({data.get('size_bytes', 0)} bytes)" if "size_bytes" in data else ""
                return f"File '{name}' created successfully at {path}{size_str}."
            if "deleted_file" in data:
                return f"File deleted successfully: {data.get('deleted_file')}"
            if "analysis" in data and "document" in data:
                doc_name = data.get("document", "Document")
                task_desc = data.get("task", "Analysis").capitalize()
                pages_info = f" ({data.get('page_count')} page(s))" if data.get("page_count", 1) > 1 else ""
                return f"### {task_desc} of {doc_name}{pages_info}\n\n{data['analysis']}"
            if "document_text" in data and "document" in data:
                doc_name = data.get("document", "Document")
                pages_info = f" ({data.get('page_count')} page(s))" if data.get("page_count", 1) > 1 else ""
                return f"### Contents of {doc_name}{pages_info}\n\n{data['document_text']}"
            if "is_scanned" in data:
                return data.get("message", "Scanned PDF detected. OCR is required.")
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
            if "matches" in data and "matches_count" in data:
                count = data["matches_count"]
                if count == 0:
                    q = data.get("query")
                    query_desc = f" for '{q}'" if q and q != "*" else ""
                    return f"No items found{query_desc}."
                lines = [f"Found {count} item{'s' if count != 1 else ''}:"]
                for m in data["matches"][:15]:
                    item_type = m.get("type", "file")
                    name = m.get("name") or m.get("filename", "unknown")
                    if item_type == "directory":
                        lines.append(f"- [Folder] {name} ({m.get('path')})")
                    else:
                        size = f"{m.get('size_mb', 0.0)} MB"
                        lines.append(f"- [File] {name} ({size}, modified {m.get('modified')})")
                if count > 15:
                    lines.append(f"... and {count - 15} more items.")
                return "\n".join(lines)
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
        conversation_id: str | None = None,
        attachment: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Process user input through deterministic routing, falling back to local LLM.

        Security & Architecture Guarantees:
        - Deterministic commands execute with ZERO LLM involvement.
        - Destructive operations return pending confirmation before executing.
        - Non-deterministic queries escalate to local Ollama only after model verification.
        - If Ollama is offline or model is missing, returns clear diagnostics and NEVER
          triggers unapproved downloads or silent model switches.
        - LLM output is strictly treated as text and never executed as code or commands.
        - Conversations are strictly isolated: no turns bleed across conversations.
        """
        clean_input = text.strip()
        effective_conv_id = conversation_id or self.active_conversation_id
        if not effective_conv_id and self.storage:
            created = self.storage.create_conversation(title="New Chat")
            effective_conv_id = created.id
            self.active_conversation_id = effective_conv_id

        if not clean_input and not attachment:
            return AgentResponse(
                text="Please provide a command or question.",
                source=ResponseSource.SYSTEM,
                success=False,
                conversation_id=effective_conv_id,
            )

        if attachment and not clean_input:
            clean_input = f"Summarize {attachment.get('filename', 'this document')}."

        # Context isolation & conversation persistence setup
        if effective_conv_id and self.storage:
            if self.active_conversation_id != effective_conv_id:
                self.switch_conversation(effective_conv_id)

            detail = self.storage.get_conversation(effective_conv_id)
            if not detail:
                self.storage.create_conversation(title="New Chat", conversation_id=effective_conv_id)

        # =========================================================================
        # Stage -1: Attached Document Processing
        # =========================================================================
        if attachment and isinstance(attachment, dict):
            att_path_str = attachment.get("path")
            att_filename = attachment.get("filename", "document")
            if not att_path_str:
                return AgentResponse(
                    text="Invalid attachment: missing file path.",
                    source=ResponseSource.SYSTEM,
                    success=False,
                    conversation_id=effective_conv_id,
                )

            att_path = Path(att_path_str).resolve()
            if not att_path.exists() or not att_path.is_file():
                return AgentResponse(
                    text=f"Attachment file not found on disk: '{att_filename}'.",
                    source=ResponseSource.SYSTEM,
                    success=False,
                    conversation_id=effective_conv_id,
                )

            if not self.gatekeeper.is_path_safe_for_read(att_path):
                return AgentResponse(
                    text=f"Access blocked: Attachment '{att_path}' is outside approved directories.",
                    source=ResponseSource.SYSTEM,
                    success=False,
                    conversation_id=effective_conv_id,
                )

            task = "summarize"
            question = None
            clean_lower = clean_input.lower()
            if "simple words" in clean_lower or "explain" in clean_lower:
                task = "explain"
            elif any(k in clean_lower for k in ("main points", "key points", "important points", "dates and names", "main dates")):
                task = "key_points"
            elif any(k in clean_lower for k in ("read", "display", "show", "cat")):
                task = "read"
            elif any(q_word in clean_lower for q_word in ("what", "how", "why", "when", "who", "where", "is there", "does the", "can you tell", "?")):
                task = "qa"
                question = clean_input

            doc_cmd = self.router.registry.get("document_query")
            if not doc_cmd:
                from .commands.builtin.document_query import DocumentQueryCommand
                doc_cmd = DocumentQueryCommand()

            tool_res = doc_cmd.run(
                args={
                    "target": str(att_path),
                    "task": task,
                    "question": question,
                },
                gatekeeper=self.gatekeeper,
                client=self.client,
            )

            if tool_res.success:
                output_text = self._format_tool_result(tool_res)
            else:
                output_text = f"Document processing failed: {tool_res.error}"

            if self.storage and effective_conv_id:
                self._maybe_auto_title(effective_conv_id, clean_input, attachment=attachment)
                user_msg_text = f"{clean_input}\n[Attached: {att_filename}]" if clean_input else f"[Attached: {att_filename}]"
                self.storage.add_message(
                    effective_conv_id,
                    role="user",
                    content=user_msg_text,
                    source="user",
                    metadata={"attachment": attachment},
                )
                safe_data = tool_res.data if isinstance(tool_res.data, dict) else {"result": str(tool_res.data)}
                self.storage.add_message(
                    effective_conv_id,
                    role="assistant",
                    content=output_text,
                    source="tool",
                    metadata={"tool_name": "document_query", "data": safe_data, "attachment": attachment},
                )

            return AgentResponse(
                text=output_text,
                source=ResponseSource.TOOL,
                matched_command="document_query",
                tool_result=tool_res,
                success=tool_res.success,
                conversation_id=effective_conv_id,
                metadata={"attachment": attachment, "data": tool_res.data},
            )

        # =========================================================================
        # Stage 0: Deterministic Command Engine (LLM Bypass)
        # =========================================================================
        if effective_conv_id:
            route_result = self.router.route(clean_input, action_id=action_id, conversation_id=effective_conv_id)
        else:
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
                    conversation_id=effective_conv_id,
                    metadata={"action_id": req.action_id},
                )

            # Case 2: Deterministic command executed
            if route_result.result is not None:
                res = route_result.result
                if res.success:
                    output_text = self._format_tool_result(res)
                else:
                    output_text = f"Command failed: {res.error}"

                if self.storage and effective_conv_id:
                    self._maybe_auto_title(effective_conv_id, clean_input)
                    self.storage.add_message(effective_conv_id, role="user", content=clean_input, source="user")
                    safe_data = res.data if isinstance(res.data, dict) else {"result": str(res.data)}
                    self.storage.add_message(
                        effective_conv_id,
                        role="assistant",
                        content=output_text,
                        source="deterministic",
                        metadata=safe_data,
                    )

                return AgentResponse(
                    text=output_text,
                    source=ResponseSource.DETERMINISTIC,
                    matched_command=route_result.command_name,
                    tool_result=res,
                    success=res.success,
                    conversation_id=effective_conv_id,
                    metadata={"data": res.data},
                )

        # =========================================================================
        # Stage 0.5: Controlled Model-Assisted Tool Selection
        # =========================================================================
        selectable_tools = self.router.registry.get_model_selectable_commands()
        if chat_mode and selectable_tools:
            try:
                resolution = self.resolver.resolve(allow_fallback=self.allow_fallback)
                if resolution.is_ready and resolution.effective_model:
                    from .commands.model_selector import select_tool

                    proposal = select_tool(
                        user_input=clean_input,
                        client=self.client,
                        model=resolution.effective_model,
                        selectable_commands=selectable_tools,
                        gatekeeper=self.gatekeeper,
                        timeout=min(self.config.models.timeout_seconds, 15.0),
                    )
                    if proposal is not None:
                        tool_cmd = selectable_tools.get(proposal.tool_name)
                        if tool_cmd is not None:
                            tool_res = tool_cmd.execute(
                                args=proposal.arguments,
                                gatekeeper=self.gatekeeper,
                                action_id=action_id,
                            )
                            if tool_res.success:
                                output_text = self._format_tool_result(tool_res)
                            else:
                                output_text = f"Tool execution failed: {tool_res.error}"

                            if self.storage and effective_conv_id:
                                self._maybe_auto_title(effective_conv_id, clean_input)
                                self.storage.add_message(
                                    effective_conv_id, role="user", content=clean_input, source="user"
                                )
                                safe_data = (
                                    tool_res.data
                                    if isinstance(tool_res.data, dict)
                                    else {"result": str(tool_res.data)}
                                )
                                self.storage.add_message(
                                    effective_conv_id,
                                    role="assistant",
                                    content=output_text,
                                    source="tool",
                                    metadata={
                                        "tool_name": proposal.tool_name,
                                        "arguments": proposal.arguments,
                                        "data": safe_data,
                                    },
                                )

                            return AgentResponse(
                                text=output_text,
                                source=ResponseSource.TOOL,
                                matched_command=proposal.tool_name,
                                tool_result=tool_res,
                                model_used=resolution.effective_model,
                                success=tool_res.success,
                                conversation_id=effective_conv_id,
                                metadata={
                                    "model_assisted": True,
                                    "arguments": proposal.arguments,
                                    "data": tool_res.data,
                                },
                            )
            except Exception as e:
                logger.warning("Stage 0.5 tool selection encountered error: %s", e)

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
                conversation_id=effective_conv_id,
                metadata={
                    "ollama_online": resolution.ollama_online,
                    "installed_models": resolution.installed_models,
                },
            )

        effective_model = resolution.effective_model

        # 2. Local AI Inference
        if chat_mode:
            if self.storage and effective_conv_id:
                self._maybe_auto_title(effective_conv_id, clean_input)
                self.storage.add_message(effective_conv_id, role="user", content=clean_input, source="user")
                self._conversation_history = self.storage.get_context_messages(
                    effective_conv_id, max_messages=self.max_history_messages
                )
            else:
                self._conversation_history.append({"role": "user", "content": clean_input})
                if len(self._conversation_history) > self.max_history_messages:
                    self._conversation_history = self._conversation_history[-self.max_history_messages:]
                    while (
                        self._conversation_history
                        and self._conversation_history[0]["role"] == "assistant"
                        and len(self._conversation_history) > 1
                    ):
                        self._conversation_history = self._conversation_history[1:]

            messages = [{"role": "system", "content": self.system_prompt}] + self._conversation_history

            try:
                reply = self.client.chat(messages, model=effective_model)
                if self.storage and effective_conv_id:
                    self.storage.add_message(
                        effective_conv_id,
                        role="assistant",
                        content=reply,
                        source="llm",
                        model_used=effective_model,
                    )
                    self._conversation_history = self.storage.get_context_messages(
                        effective_conv_id, max_messages=self.max_history_messages
                    )
                else:
                    self._conversation_history.append({"role": "assistant", "content": reply})

                return AgentResponse(
                    text=reply,
                    source=ResponseSource.LLM,
                    model_used=effective_model,
                    success=True,
                    conversation_id=effective_conv_id,
                    metadata={"using_fallback": resolution.using_fallback},
                )
            except (ModelNotFoundError, ModelInferenceError, ModelOutputParseError, ModelFallbackError) as e:
                # Rollback or reload context to prevent poisoned history
                if self.storage and effective_conv_id:
                    self._conversation_history = self.storage.get_context_messages(
                        effective_conv_id, max_messages=self.max_history_messages
                    )
                else:
                    self._conversation_history.pop()
                logger.error("Local LLM chat failed: %s", e)
                return AgentResponse(
                    text=f"AI inference error: {e}",
                    source=ResponseSource.SYSTEM,
                    model_used=effective_model,
                    success=False,
                    conversation_id=effective_conv_id,
                    metadata={"error": str(e)},
                )
        else:
            try:
                reply = self.client.generate(
                    prompt=clean_input,
                    model=effective_model,
                    system=self.system_prompt,
                )
                if self.storage and effective_conv_id:
                    self._maybe_auto_title(effective_conv_id, clean_input)
                    self.storage.add_message(effective_conv_id, role="user", content=clean_input, source="user")
                    self.storage.add_message(
                        effective_conv_id,
                        role="assistant",
                        content=reply,
                        source="llm",
                        model_used=effective_model,
                    )

                return AgentResponse(
                    text=reply,
                    source=ResponseSource.LLM,
                    model_used=effective_model,
                    success=True,
                    conversation_id=effective_conv_id,
                    metadata={"using_fallback": resolution.using_fallback},
                )
            except (ModelNotFoundError, ModelInferenceError, ModelOutputParseError, ModelFallbackError) as e:
                logger.error("Local LLM generation failed: %s", e)
                return AgentResponse(
                    text=f"AI inference error: {e}",
                    source=ResponseSource.SYSTEM,
                    model_used=effective_model,
                    success=False,
                    conversation_id=effective_conv_id,
                    metadata={"error": str(e)},
                )
