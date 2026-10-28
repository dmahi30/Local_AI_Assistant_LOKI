"""System information and telemetry command for LOKI."""

from __future__ import annotations

import os
import sys
from typing import Any
import psutil

from ...core.types import IntentType, SafetyLevel, ToolResult
from ...safety.gatekeeper import SafetyGatekeeper
from ..base import BaseCommand


class SystemInfoCommand(BaseCommand):
    """Reports system metrics (battery, RAM, CPU, storage) with minimal latency."""

    name = "system_info"
    description = "Queries real-time system metrics such as battery, RAM, CPU, and storage."
    intent_type = IntentType.DETERMINISTIC_SYSTEM
    safety_level = SafetyLevel.SAFE
    allow_model_selection = True
    ALLOWED_METRICS = ("all", "battery", "ram", "cpu", "storage")

    def get_parameters_schema(self) -> dict[str, Any]:
        """Return JSON Schema for model-assisted tool calling."""
        return {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": list(self.ALLOWED_METRICS),
                    "description": "Specific telemetry metric to query: battery, ram, cpu, storage, or all.",
                }
            },
            "required": ["metric"],
            "additionalProperties": False,
        }

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate system_info arguments against schema and allowable metrics."""
        if not isinstance(args, dict):
            return False, "Arguments must be a dictionary."
        extra_keys = set(args.keys()) - {"metric"}
        if extra_keys:
            return False, f"Unexpected arguments for system_info: {', '.join(sorted(extra_keys))}"
        if "metric" not in args:
            return False, "Missing required argument 'metric' for system_info."
        metric = args["metric"]
        if not isinstance(metric, str) or metric.lower() not in self.ALLOWED_METRICS:
            return False, f"Invalid metric '{metric}'. Must be one of: {', '.join(self.ALLOWED_METRICS)}"
        return True, None

    patterns = [
        r"^(what('?s| is) my )?(battery|battery percentage|battery status|charging status)$",
        r"\b(battery status|battery percentage|is laptop charging)\b",
        r"^(what('?s| is) my )?(ram|memory|free ram|ram usage)$",
        r"\b(ram usage|memory usage|how much ram|free memory)\b",
        r"^(what('?s| is) my )?(cpu|cpu usage|processor usage)$",
        r"\b(cpu usage|processor usage)\b",
        r"^(what('?s| is) my )?(storage|disk space|free disk|drive space)$",
        r"\b(disk space|storage space|how much storage|free space)\b",
        r"\b(system info|system status|pc status|system telemetry)\b",
    ]

    def extract_args(self, text: str) -> dict[str, Any]:
        lower = text.lower()
        if "battery" in lower or "charging" in lower:
            metric = "battery"
        elif "ram" in lower or "memory" in lower:
            metric = "ram"
        elif "cpu" in lower or "processor" in lower:
            metric = "cpu"
        elif "storage" in lower or "disk" in lower or "drive" in lower:
            metric = "storage"
        else:
            metric = "all"
        return {"metric": metric}

    def run(
        self,
        args: dict[str, Any],
        gatekeeper: SafetyGatekeeper | None = None,
        action_id: str | None = None,
    ) -> ToolResult:
        metric = args.get("metric", "all")
        data: dict[str, Any] = {}

        if metric in ("battery", "all"):
            battery = psutil.sensors_battery()
            if battery is not None:
                data["battery"] = {
                    "percent": round(battery.percent, 1),
                    "power_plugged": battery.power_plugged,
                    "secs_left": battery.secsleft if battery.secsleft != -1 else "Calculating/Plugged in",
                }
            else:
                data["battery"] = {"status": "No battery detected (Desktop or AC power)"}

        if metric in ("ram", "all"):
            vm = psutil.virtual_memory()
            data["ram"] = {
                "total_gb": round(vm.total / (1024**3), 2),
                "used_gb": round(vm.used / (1024**3), 2),
                "available_gb": round(vm.available / (1024**3), 2),
                "percent_used": vm.percent,
            }

        if metric in ("cpu", "all"):
            data["cpu"] = {
                "usage_percent": psutil.cpu_percent(interval=None),
                "cores": psutil.cpu_count(logical=False) or 1,
                "threads": psutil.cpu_count(logical=True) or 1,
            }

        if metric in ("storage", "all"):
            root_drive = "C:\\" if sys.platform == "win32" else "/"
            usage = psutil.disk_usage(root_drive)
            data["storage"] = {
                "drive": root_drive,
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
                "percent_used": usage.percent,
            }

        return ToolResult(
            success=True,
            data=data,
            safety_level=self.safety_level,
            metadata={"metric": metric},
        )
