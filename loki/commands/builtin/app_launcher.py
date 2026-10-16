"""Safe application launcher for LOKI."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

from ...core.types import IntentType, SafetyLevel, ToolResult
from ...safety.gatekeeper import SafetyGatekeeper
from ..base import BaseCommand

# Approved applications whitelist with their common executable candidates
APPROVED_APPLICATIONS: dict[str, list[str]] = {
    "chrome": ["chrome.exe", "chrome"],
    "google chrome": ["chrome.exe", "chrome"],
    "code": ["code.cmd", "code.exe", "code"],
    "vscode": ["code.cmd", "code.exe", "code"],
    "vs code": ["code.cmd", "code.exe", "code"],
    "visual studio code": ["code.cmd", "code.exe", "code"],
    "notepad": ["notepad.exe", "notepad"],
    "calculator": ["calc.exe", "calc"],
    "calc": ["calc.exe", "calc"],
    "explorer": ["explorer.exe"],
    "file explorer": ["explorer.exe"],
    "paint": ["mspaint.exe", "mspaint"],
    "mspaint": ["mspaint.exe"],
    "edge": ["msedge.exe", "msedge"],
    "microsoft edge": ["msedge.exe"],
    "task manager": ["taskmgr.exe"],
    "taskmgr": ["taskmgr.exe"],
}

# Forbidden command characters to strictly prevent shell injection and path tampering
DISALLOWED_CHARS = re.compile(r"[\x00-\x1f\x7f;&|`$<>{}\[\]\"'\n\r]")


class AppLauncherCommand(BaseCommand):
    """Safely launches whitelisted desktop applications without shell execution."""

    name = "app_launcher"
    description = "Opens approved desktop applications (Chrome, VS Code, Notepad, Calculator, etc.)."
    intent_type = IntentType.DETERMINISTIC_APP
    safety_level = SafetyLevel.SAFE

    patterns = [
        r"^(open|launch|start)\s+([a-zA-Z0-9_\-\.\s]+)$",
    ]

    def extract_args(self, text: str) -> dict[str, Any]:
        match = re.search(r"^(?:open|launch|start)\s+(.+)$", text.strip(), re.IGNORECASE)
        app_name = match.group(1).strip().lower() if match else ""
        return {"app_name": app_name}

    def _find_executable(self, candidate_names: list[str]) -> str | None:
        """Find executable strictly as a verified absolute path."""
        # 1. On Windows, check trusted system locations first (System32 and Windows)
        if sys.platform == "win32":
            sys_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
            sys32 = sys_root / "System32"
            for candidate in candidate_names:
                s32_file = sys32 / candidate
                if s32_file.exists() and s32_file.is_file():
                    return str(s32_file.resolve())
                win_file = sys_root / candidate
                if win_file.exists() and win_file.is_file():
                    return str(win_file.resolve())

        # 2. Check system PATH via shutil.which
        for candidate in candidate_names:
            resolved = shutil.which(candidate)
            if resolved:
                p = Path(resolved).resolve()
                if p.is_absolute() and p.exists() and p.is_file():
                    return str(p)

        # 3. Windows Registry App Paths fallback
        if sys.platform == "win32":
            try:
                import winreg

                for candidate in candidate_names:
                    reg_key_path = rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{candidate}"
                    for root_hkey in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                        try:
                            with winreg.OpenKey(root_hkey, reg_key_path) as k:
                                val, _ = winreg.QueryValueEx(k, "")
                                if val:
                                    clean_val = val.strip('"\t ')
                                    p = Path(clean_val).resolve()
                                    if p.is_absolute() and p.exists() and p.is_file():
                                        return str(p)
                        except OSError:
                            continue
            except Exception:
                pass

        return None

    def run(
        self,
        args: dict[str, Any],
        gatekeeper: SafetyGatekeeper | None = None,
        action_id: str | None = None,
    ) -> ToolResult:
        raw_name = args.get("app_name", "").strip().lower()

        # Strict security validation
        if not raw_name:
            return ToolResult(
                success=False,
                error="No application name specified.",
                safety_level=self.safety_level,
            )

        if DISALLOWED_CHARS.search(raw_name):
            return ToolResult(
                success=False,
                error=f"Security violation: Illegal characters detected in app request '{raw_name}'.",
                safety_level=SafetyLevel.DANGEROUS,
            )

        # Check against approved applications whitelist
        candidates = APPROVED_APPLICATIONS.get(raw_name)
        if not candidates:
            return ToolResult(
                success=False,
                error=(
                    f"Application '{raw_name}' is not in the approved application whitelist. "
                    f"Supported applications: {', '.join(sorted(APPROVED_APPLICATIONS.keys()))}"
                ),
                safety_level=self.safety_level,
            )

        # Locate executable strictly as an absolute path
        target_path = self._find_executable(candidates)
        if not target_path or not Path(target_path).is_absolute() or not Path(target_path).exists():
            return ToolResult(
                success=False,
                error=f"Could not locate a verified absolute executable for '{raw_name}'. Execution refused for safety.",
                safety_level=self.safety_level,
            )

        try:
            # Launch without shell=True to prevent injection
            if sys.platform == "win32":
                os.startfile(target_path)
            else:
                subprocess.Popen([target_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            return ToolResult(
                success=True,
                data={
                    "app_name": raw_name,
                    "target": target_path,
                    "status": "launched",
                },
                safety_level=self.safety_level,
                metadata={"action": "launch_app"},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to launch '{raw_name}': {e}",
                safety_level=self.safety_level,
            )
