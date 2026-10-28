"""Model-assisted tool selection engine for LOKI.

Provides Stage 0.5 controlled tool selection between deterministic regex routing
and standard conversational LLM generation.

Security Invariants:
1. ONLY registered commands with `allow_model_selection = True` and `safety_level = SAFE`
   are eligible for selection.
2. Destructive tools (e.g. file_delete) and application launchers are strictly forbidden
   from model selection.
3. Model output is treated as untrusted data; all proposed arguments are validated against
   strict schemas, lengths, and path boundaries.
4. Malformed JSON, unknown tools, invalid parameters, timeouts, or model errors fail
   closed without executing any tool.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from ..core.exceptions import ModelError
from ..core.logging import get_logger
from ..core.types import SafetyLevel
from ..llm.client import OllamaClient
from ..safety.gatekeeper import SafetyGatekeeper
from .base import BaseCommand

logger = get_logger("commands.model_selector")


class ModelToolProposal(BaseModel):
    """Validated structured tool proposal from the local model."""
    tool_name: str = Field(..., description="Allowlisted registered tool name")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Validated argument dictionary")


def build_tool_selection_prompt(selectable_commands: dict[str, BaseCommand]) -> str:
    """Build a constrained system prompt defining allowlisted tools and response format."""
    tool_descriptions: list[str] = []
    for name, cmd in sorted(selectable_commands.items()):
        schema = cmd.get_parameters_schema()
        props = schema.get("properties", {})
        param_desc = ", ".join(f"'{k}'" for k in props.keys()) or "none"
        tool_descriptions.append(
            f"- {name}: {cmd.description} (Parameters: {param_desc})"
        )

    tools_text = "\n".join(tool_descriptions)

    return (
        "You are LOKI's tool selector. Determine whether the user's request requires executing "
        "one of the available system tools, or if it is conversational.\n\n"
        f"AVAILABLE TOOLS:\n{tools_text}\n\n"
        "STRICT CRITERIA:\n"
        "- ONLY select 'system_info' if the user specifically asks about PC/hardware status: battery, RAM/memory, CPU, or storage/disk.\n"
        "- ONLY select 'file_search' if the user asks to find, locate, or search for files or documents on disk.\n"
        "- For ALL OTHER requests (including conversational chat, arithmetic/math like 'what is 25+17', personal questions like 'what is my favorite color', writing, or general knowledge), you MUST respond with: {\"tool\": \"none\"}.\n\n"
        "RESPONSE FORMAT:\n"
        "Respond with ONLY a raw JSON object:\n"
        "{\"tool\": \"<tool_name>\", \"arguments\": { ... }} OR {\"tool\": \"none\"}\n"
        "Do not include any other text."
    )


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Extract and parse a JSON dictionary from raw model text."""
    cleaned = raw_text.strip()
    if not cleaned:
        return None

    # Strip markdown code blocks if the model wrapped the output
    if cleaned.startswith("```"):
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()

    # Find outermost JSON object braces
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        return None

    candidate = cleaned[start_idx : end_idx + 1]
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def parse_and_validate_tool_call(
    raw_text: str,
    selectable_commands: dict[str, BaseCommand],
    gatekeeper: SafetyGatekeeper | None = None,
    user_input: str | None = None,
) -> tuple[ModelToolProposal | None, str | None]:
    """Parse and strictly validate a model-proposed tool invocation.

    Returns:
        (ModelToolProposal, None) on success.
        (None, None) if the model deliberately chose "none" (conversational).
        (None, error_reason) if parsing or validation failed.
    """
    data = extract_json_object(raw_text)
    if data is None:
        return None, "Output did not contain a valid JSON object."

    tool_raw = data.get("tool")
    if tool_raw is None or not isinstance(tool_raw, str):
        return None, "JSON missing required string field 'tool'."

    tool_name = tool_raw.strip().lower()
    if tool_name in ("none", "null", ""):
        return None, None

    # Invariant 1: Tool must be in allowlisted selectable commands
    if tool_name not in selectable_commands:
        return None, f"Tool '{tool_name}' is not in the allowlist of model-selectable tools."

    command = selectable_commands[tool_name]

    # Invariant 2: Command must be SAFE and explicitly allow selection
    if command.safety_level != SafetyLevel.SAFE or not getattr(command, "allow_model_selection", False):
        return None, f"Security violation: Tool '{tool_name}' is not permitted for model-assisted execution."

    args = data.get("arguments")
    if args is None:
        args = {}
    elif not isinstance(args, dict):
        return None, f"Tool arguments must be a dictionary, got {type(args).__name__}."

    # Invariant 3: Validate arguments via command's strict schema
    is_valid, validation_err = command.validate_args(args)
    if not is_valid:
        return None, f"Argument validation failed: {validation_err}"

    # Invariant 4: Semantic relevance validation to guard against small-model hallucinations
    if user_input:
        query_lower = user_input.lower()
        if tool_name == "system_info":
            system_keywords = (
                "battery", "power", "charge", "charging", "drain", "laptop", "energy", "plugged",
                "ram", "memory", "usage", "available", "mem",
                "cpu", "processor", "load", "core", "cores", "ghz",
                "storage", "disk", "space", "drive", "capacity", "ssd", "hdd",
                "system", "hardware", "telemetry", "status", "specs", "pc", "computer", "machine", "metrics", "device"
            )
            if not any(kw in query_lower for kw in system_keywords):
                return None, "Tool 'system_info' proposed but user query contains no relevant hardware keywords."

        elif tool_name == "file_search":
            search_keywords = (
                "file", "files", "document", "documents", "find", "search", "locate", "look for",
                "where is", "show me", "folder", "directory", ".txt", ".pdf", ".docx", ".log",
                ".csv", ".py", ".md", ".json", ".zip"
            )
            q_val = str(args.get("query", "")).lower()
            if not any(kw in query_lower for kw in search_keywords) and (q_val not in query_lower):
                return None, "Tool 'file_search' proposed but user query contains no search keywords."

    # Invariant 5: Additional path checks for file_search
    if tool_name == "file_search":
        location = args.get("location")
        if location:
            if "\x00" in location or "\n" in location or "\r" in location:
                return None, "File search location contains invalid characters."

            # Verify safe read root if gatekeeper is present
            if gatekeeper is not None:
                user_home = Path.home()
                alias_map = {
                    "downloads": user_home / "Downloads",
                    "download": user_home / "Downloads",
                    "documents": user_home / "Documents",
                    "docs": user_home / "Documents",
                    "desktop": user_home / "Desktop",
                    "pictures": user_home / "Pictures",
                    "videos": user_home / "Videos",
                }
                resolved_loc = alias_map.get(location.lower(), Path(location).resolve())
                if not gatekeeper.is_path_safe_for_read(resolved_loc):
                    return None, f"Location '{location}' is restricted or outside safe read roots."

    return ModelToolProposal(tool_name=tool_name, arguments=args), None


def select_tool(
    user_input: str,
    client: OllamaClient,
    model: str,
    selectable_commands: dict[str, BaseCommand],
    gatekeeper: SafetyGatekeeper | None = None,
    timeout: float = 10.0,
) -> ModelToolProposal | None:
    """Prompt the local LLM to select an allowlisted tool, validating output fail-closed.

    Returns:
        ModelToolProposal if a valid, authorized tool proposal was produced.
        None if conversational, invalid, timed out, or unallowlisted.
    """
    if not selectable_commands:
        return None

    clean_input = user_input.strip()
    if not clean_input:
        return None

    system_prompt = build_tool_selection_prompt(selectable_commands)

    try:
        # Request short, zero-temperature JSON tool classification
        raw_output = client.generate(
            prompt=clean_input,
            model=model,
            system=system_prompt,
            options={"temperature": 0.0, "num_predict": 128},
            timeout=timeout,
        )

        proposal, err = parse_and_validate_tool_call(
            raw_text=raw_output,
            selectable_commands=selectable_commands,
            gatekeeper=gatekeeper,
            user_input=clean_input,
        )

        if err:
            logger.info("Model tool selection discarded: %s", err)
            return None

        if proposal:
            logger.info("Model selected allowlisted tool: %s with args %s", proposal.tool_name, proposal.arguments)
        return proposal

    except ModelError as me:
        logger.warning("Local Ollama model error during tool selection: %s", me)
        return None
    except Exception as e:
        logger.warning("Tool selection failed or timed out: %s", e)
        return None
