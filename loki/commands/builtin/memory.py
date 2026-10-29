"""Deterministic memory management command for personal preferences and facts."""

from __future__ import annotations

import re
from typing import Any

from ...core.exceptions import InvalidMemoryError, SafetyViolationError
from ...core.types import IntentType, SafetyLevel, ToolResult
from ...safety.gatekeeper import SafetyGatekeeper
from ...storage.memory import MemoryCategory, MemoryRecord, MemorySourceType, MemoryStore
from ..base import BaseCommand


def _infer_category(content: str) -> str:
    """Deterministically categorize memory based on keyword presence."""
    lower = content.lower()
    preference_keywords = (
        "prefer",
        "preference",
        "favorite",
        "favourite",
        "like",
        "likes",
        "love",
        "dislike",
        "hate",
        "want",
        "always",
        "never",
    )
    instruction_keywords = (
        "always answer",
        "never answer",
        "format as",
        "respond in",
        "speak in",
        "keep answers",
    )
    if any(k in lower for k in instruction_keywords):
        return MemoryCategory.INSTRUCTION.value
    if any(k in lower for k in preference_keywords):
        return MemoryCategory.PREFERENCE.value
    return MemoryCategory.FACT.value


class MemoryCommand(BaseCommand):
    """Deterministic memory management command for personal preferences and facts."""

    name = "memory"
    description = "Deterministic storage, retrieval, and deletion of personal user memories and preferences."
    intent_type = IntentType.DETERMINISTIC_SYSTEM
    safety_level = SafetyLevel.SAFE
    allow_model_selection = False

    # Regex patterns for matching
    _CLEAR_ALL_PATTERN = (
        r"^(?:please\s+)?(?:forget\s+everything(?:\s+you\s+remember)?|"
        r"clear\s+(?:all\s+)?memories|delete\s+all\s+memories|"
        r"forget\s+all\s+memories|wipe\s+all\s+memories)$"
    )
    _LIST_PATTERN = (
        r"^(?:what\s+do\s+you\s+remember(?:\s+about\s+me)?\??|"
        r"list\s+(?:all\s+)?(?:my\s+)?memories|"
        r"show\s+(?:all\s+)?(?:my\s+)?memories|"
        r"what\s+memories\s+do\s+you\s+have(?:\s+about\s+me)?\??)$"
    )
    _REMEMBER_PATTERN = r"^(?:please\s+)?remember(?:\s+that|:)?(?:\s+(?P<content>.+))?$"
    _FORGET_SPECIFIC_PATTERN = r"^(?:please\s+)?forget(?:\s+that|:)?(?:\s+(?P<target>.+))?$"

    patterns = [
        _CLEAR_ALL_PATTERN,
        _LIST_PATTERN,
        _REMEMBER_PATTERN,
        _FORGET_SPECIFIC_PATTERN,
    ]

    def __init__(self, memory_store: MemoryStore | None = None) -> None:
        super().__init__()
        self.memory_store = memory_store or MemoryStore()
        self._clear_all_re = re.compile(self._CLEAR_ALL_PATTERN, re.IGNORECASE)
        self._list_re = re.compile(self._LIST_PATTERN, re.IGNORECASE)
        self._remember_re = re.compile(self._REMEMBER_PATTERN, re.IGNORECASE)
        self._forget_specific_re = re.compile(self._FORGET_SPECIFIC_PATTERN, re.IGNORECASE)

    def extract_args(self, text: str) -> dict[str, Any]:
        """Parse structured arguments from the natural language text."""
        cleaned = text.strip()

        # 1. Clear all memories (DANGEROUS) must be checked before specific forget
        if self._clear_all_re.search(cleaned):
            return {
                "action": "clear_all",
                "targets": ["all_memories"],
                "target": "all_memories",
                "description": "Clear all stored memories",
            }

        # 2. List memories query
        if self._list_re.search(cleaned):
            return {
                "action": "list",
            }

        # 3. Remember explicit fact/preference
        m_rem = self._remember_re.search(cleaned)
        if m_rem:
            content = m_rem.group("content")
            return {
                "action": "store",
                "content": content.strip() if content else "",
            }

        # 4. Forget specific memory
        m_forg = self._forget_specific_re.search(cleaned)
        if m_forg:
            target = m_forg.group("target")
            return {
                "action": "forget",
                "target": target.strip() if target else "",
            }

        return {}

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate input arguments before execution."""
        if not isinstance(args, dict):
            return False, "Arguments must be a dictionary."

        action = args.get("action")
        if not action or not isinstance(action, str):
            return False, "Missing memory action."

        action_clean = action.strip().lower()
        if action_clean not in ("store", "list", "forget", "clear_all"):
            return False, f"Unknown memory action '{action}'."

        if action_clean == "store":
            content = args.get("content")
            if not content or not isinstance(content, str) or not content.strip():
                return False, "No memory content provided to remember."

        if action_clean == "forget":
            target = args.get("target")
            if not target or not isinstance(target, str) or not target.strip():
                return False, "No memory target specified to forget."

        return True, None

    def run(
        self,
        args: dict[str, Any],
        gatekeeper: SafetyGatekeeper | None = None,
        action_id: str | None = None,
    ) -> ToolResult:
        """Execute the deterministic memory command."""
        action = args.get("action", "").strip().lower()

        if action == "store":
            content = args.get("content", "").strip()
            category = _infer_category(content)
            try:
                rec = self.memory_store.store_memory(
                    content=content,
                    category=category,
                    source_type=MemorySourceType.EXPLICIT_COMMAND.value,
                )
                return ToolResult(
                    success=True,
                    data={
                        "memory_action": "store",
                        "id": rec.id,
                        "content": rec.content,
                        "category": rec.category,
                        "message": f'I will remember: "{rec.content}"',
                    },
                    safety_level=self.safety_level,
                    metadata={"command": self.name, "memory_id": rec.id},
                )
            except InvalidMemoryError as ime:
                return ToolResult(
                    success=False,
                    error=str(ime),
                    safety_level=self.safety_level,
                    metadata={"command": self.name},
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Failed to store memory: {e}",
                    safety_level=self.safety_level,
                    metadata={"command": self.name},
                )

        if action == "list":
            try:
                memories = self.memory_store.list_memories(limit=50)
                if not memories:
                    msg = "I do not remember anything about you yet."
                    return ToolResult(
                        success=True,
                        data={
                            "memory_action": "list",
                            "count": 0,
                            "memories": [],
                            "message": msg,
                        },
                        safety_level=self.safety_level,
                        metadata={"command": self.name},
                    )

                lines = ["Here is what I remember:"]
                for m in memories:
                    lines.append(f"- {m.content}")
                msg = "\n".join(lines)
                return ToolResult(
                    success=True,
                    data={
                        "memory_action": "list",
                        "count": len(memories),
                        "memories": [
                            {"id": m.id, "content": m.content, "category": m.category}
                            for m in memories
                        ],
                        "message": msg,
                    },
                    safety_level=self.safety_level,
                    metadata={"command": self.name, "count": len(memories)},
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Failed to list memories: {e}",
                    safety_level=self.safety_level,
                    metadata={"command": self.name},
                )

        if action == "forget":
            target = args.get("target", "").strip()

            # Direct UUID lookup shortcut
            matches: list[MemoryRecord] = []
            if len(target) == 32 and all(c in "0123456789abcdefABCDEF" for c in target):
                mem = self.memory_store.get_memory(target)
                if mem:
                    matches = [mem]

            if not matches:
                # Search using FTS5 BM25 search
                matches = self.memory_store.search_memories(target, limit=20)

            # Substring fallback if FTS didn't match (e.g. stop-words or partial strings)
            if not matches:
                all_memories = self.memory_store.list_memories(limit=100)
                target_lower = target.lower()
                matches = [
                    m
                    for m in all_memories
                    if target_lower in m.content.lower()
                    or (m.key and target_lower in m.key.lower())
                ]

            if not matches:
                msg = f"No saved memory found matching '{target}'."
                return ToolResult(
                    success=True,
                    data={
                        "memory_action": "forget",
                        "status": "not_found",
                        "target": target,
                        "message": msg,
                    },
                    safety_level=self.safety_level,
                    metadata={"command": self.name, "target": target},
                )

            if len(matches) > 1:
                lines = [
                    f"Found multiple memories matching '{target}'. Please specify which one to forget:"
                ]
                for m in matches:
                    lines.append(f"- {m.content}")
                msg = "\n".join(lines)
                return ToolResult(
                    success=True,
                    data={
                        "memory_action": "forget",
                        "status": "ambiguous",
                        "matches_count": len(matches),
                        "matches": [
                            {"id": m.id, "content": m.content} for m in matches
                        ],
                        "message": msg,
                    },
                    safety_level=self.safety_level,
                    metadata={"command": self.name, "matches_count": len(matches)},
                )

            # Exactly one match: delete it safely
            single_match = matches[0]
            try:
                self.memory_store.delete_memory(single_match.id)
                msg = f'I have forgotten: "{single_match.content}"'
                return ToolResult(
                    success=True,
                    data={
                        "memory_action": "forget",
                        "status": "deleted",
                        "deleted_id": single_match.id,
                        "deleted_content": single_match.content,
                        "message": msg,
                    },
                    safety_level=self.safety_level,
                    metadata={"command": self.name, "deleted_id": single_match.id},
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Failed to delete memory: {e}",
                    safety_level=self.safety_level,
                    metadata={"command": self.name},
                )

        if action == "clear_all":
            if gatekeeper is None:
                raise SafetyViolationError(
                    "Execution prohibited: Clearing all memories requires an active SafetyGatekeeper.",
                    details={"command": self.name, "action": "clear_all"},
                )

            try:
                deleted_count = self.memory_store.clear_memories()
                msg = f"All memories have been cleared ({deleted_count} deleted)."
                return ToolResult(
                    success=True,
                    data={
                        "memory_action": "clear_all",
                        "deleted_count": deleted_count,
                        "message": msg,
                    },
                    safety_level=SafetyLevel.DANGEROUS,
                    metadata={"command": self.name, "action": "clear_all"},
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Failed to clear memories: {e}",
                    safety_level=SafetyLevel.DANGEROUS,
                    metadata={"command": self.name},
                )

        return ToolResult(
            success=False,
            error=f"Unsupported memory action '{action}'.",
            safety_level=self.safety_level,
            metadata={"command": self.name},
        )
