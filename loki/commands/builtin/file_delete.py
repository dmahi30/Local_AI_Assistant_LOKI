"""File deletion command demonstrating the SafetyGatekeeper's DANGEROUS tier."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from ...core.exceptions import SafetyViolationError
from ...core.types import IntentType, SafetyLevel, ToolResult
from ...safety.gatekeeper import SafetyGatekeeper
from ..base import BaseCommand


class FileDeleteCommand(BaseCommand):
    """Deletes a file safely; strictly gated by SafetyGatekeeper as DANGEROUS."""

    name = "file_delete"
    description = "Deletes a designated file from an approved user directory after explicit confirmation."
    intent_type = IntentType.DETERMINISTIC_SYSTEM
    safety_level = SafetyLevel.DANGEROUS

    patterns = [
        r"^(delete|remove)\s+(?:file\s+)?(.+)$",
    ]

    def extract_args(self, text: str) -> dict[str, Any]:
        match = re.search(r"^(?:delete|remove)\s+(?:file\s+)?(.+)$", text.strip(), re.IGNORECASE)
        target = match.group(1).strip() if match else ""
        return {"path": target, "targets": [target] if target else []}

    def run(
        self,
        args: dict[str, Any],
        gatekeeper: SafetyGatekeeper | None = None,
        action_id: str | None = None,
    ) -> ToolResult:
        raw_path = args.get("path", "").strip()
        if not raw_path:
            return ToolResult(
                success=False,
                error="No file path provided for deletion.",
                safety_level=self.safety_level,
            )

        raw_p = Path(raw_path)
        if not raw_p.exists():
            return ToolResult(
                success=False,
                error=f"Target file does not exist: '{raw_path}'",
                safety_level=self.safety_level,
            )

        target = raw_p.resolve()
        if not target.is_file():
            return ToolResult(
                success=False,
                error=f"Target is not a regular file: '{raw_path}'",
                safety_level=self.safety_level,
            )

        # Fail-closed: Gatekeeper is strictly mandatory
        if gatekeeper is None:
            raise SafetyViolationError(
                "Execution prohibited: FileDeleteCommand requires an active SafetyGatekeeper.",
                details={"target": str(target)},
            )

        # Both the link location and resolved target must be safe for write
        if not gatekeeper.is_path_safe_for_write(raw_p) or not gatekeeper.is_path_safe_for_write(target):
            raise SafetyViolationError(
                f"Deletion prohibited: Target '{target}' is in a protected or non-user location.",
                details={"target": str(target)},
            )

        try:
            target.unlink()
            return ToolResult(
                success=True,
                data={
                    "deleted_file": str(target),
                    "status": "deleted",
                },
                safety_level=self.safety_level,
                metadata={"action": "delete_file"},
            )
        except OSError as e:
            return ToolResult(
                success=False,
                error=f"Failed to delete file '{target}': {e}",
                safety_level=self.safety_level,
            )
