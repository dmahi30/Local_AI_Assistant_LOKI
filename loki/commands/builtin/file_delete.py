"""File deletion command demonstrating the SafetyGatekeeper's DANGEROUS tier."""

from __future__ import annotations

import ctypes
import os
import re
from ctypes import wintypes
from pathlib import Path
from typing import Any

from ...core.exceptions import SafetyViolationError
from ...core.types import IntentType, SafetyLevel, ToolResult
from ...safety.gatekeeper import SafetyGatekeeper
from ..base import BaseCommand


def _move_to_recycle_bin(path: Path) -> None:
    """Move a file to the Windows Recycle Bin, failing safely if unavailable."""
    if os.name != "nt":
        raise OSError("Recycle Bin deletion is supported only on Windows.")

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR),
            ("fFlags", wintypes.WORD),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", wintypes.LPVOID),
            ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    # pFrom must be double-null-terminated.
    source = str(path.resolve()) + "\0\0"

    operation = SHFILEOPSTRUCTW()
    operation.wFunc = 3  # FO_DELETE
    # FOF_ALLOWUNDO (0x40) | FOF_NOCONFIRMATION (0x10) | FOF_SILENT (0x4) | FOF_NOERRORUI (0x400)
    operation.fFlags = 0x40 | 0x10 | 0x4 | 0x400
    operation.pFrom = source

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    shell32.SHFileOperationW.argtypes = [ctypes.POINTER(SHFILEOPSTRUCTW)]
    shell32.SHFileOperationW.restype = ctypes.c_int

    result = shell32.SHFileOperationW(ctypes.byref(operation))
    if result != 0 or operation.fAnyOperationsAborted:
        raise OSError(result, f"Could not move file to the Recycle Bin: {path}")
    if path.exists():
        raise OSError(f"The file still exists after the Recycle Bin operation: {path}")


class FileDeleteCommand(BaseCommand):
    """Deletes a file safely; strictly gated by SafetyGatekeeper as DANGEROUS."""

    name = "file_delete"
    description = "Deletes a designated file from an approved user directory after explicit confirmation."
    intent_type = IntentType.DETERMINISTIC_SYSTEM
    safety_level = SafetyLevel.DANGEROUS
    allow_model_selection = False

    patterns = [
        r"^(delete|remove)\s+(?:file\s+)?(.+)$",
    ]

    def extract_args(self, text: str) -> dict[str, Any]:
        match = re.search(r"^(?:delete|remove)\s+(?:file\s+)?(.+)$", text.strip(), re.IGNORECASE)
        target = match.group(1).strip() if match else ""
        return {"path": target, "targets": [target] if target else []}

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        """Reject malformed args or a `path` that disagrees with `targets`.

        Returns (is_valid, error_message), matching BaseCommand's contract.
        """
        if not isinstance(args, dict):
            return False, "Arguments must be a dictionary."

        path = args.get("path")
        if not isinstance(path, str) or not path.strip():
            return False, "No file path provided for deletion."

        targets = args.get("targets")
        if targets is not None:
            if not isinstance(targets, (list, tuple)) or len(targets) != 1:
                return False, "Exactly one deletion target is required."
            target = targets[0]
            if not isinstance(target, str) or not target.strip():
                return False, "Invalid deletion target."
            try:
                if Path(target).resolve() != Path(path).resolve():
                    return False, "Invalid deletion arguments: 'path' and 'targets' must match."
            except (OSError, RuntimeError, ValueError):
                return False, "Invalid deletion arguments: could not resolve path."

        return True, None

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

        # Defense in depth: never act on a path that disagrees with the
        # targets that were (or would be) authorized.
        is_valid, validation_error = self.validate_args(args)
        if not is_valid:
            return ToolResult(
                success=False,
                error=validation_error or "Invalid deletion arguments.",
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

        if not getattr(gatekeeper.config, "recycle_bin_deletions", False):
            return ToolResult(
                success=False,
                error="Deletion blocked: Recycle Bin deletion is disabled.",
                safety_level=self.safety_level,
            )

        try:
            _move_to_recycle_bin(target)
            return ToolResult(
                success=True,
                data={
                    "deleted_file": str(target),
                    "status": "moved_to_recycle_bin",
                },
                safety_level=self.safety_level,
                metadata={"action": "move_to_recycle_bin"},
            )
        except (OSError, AttributeError) as e:
            return ToolResult(
                success=False,
                error=f"Could not move file to the Recycle Bin: {e}",
                safety_level=self.safety_level,
            )