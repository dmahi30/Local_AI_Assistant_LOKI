"""Deterministic file search command restricted to safe user directories."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re
from typing import Any

from ...core.types import IntentType, SafetyLevel, ToolResult
from ...safety.gatekeeper import SafetyGatekeeper
from ..base import BaseCommand


class FileSearchCommand(BaseCommand):
    """Searches for files by name/extension strictly within allowed user directories."""

    name = "file_search"
    description = "Searches for files matching a keyword or pattern in safe user directories."
    intent_type = IntentType.FILE_SEARCH
    safety_level = SafetyLevel.SAFE
    allow_model_selection = True

    def get_parameters_schema(self) -> dict[str, Any]:
        """Return JSON Schema for model-assisted tool calling."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "File search pattern, filename, or keyword to find.",
                },
                "location": {
                    "type": ["string", "null"],
                    "maxLength": 260,
                    "description": "Optional safe folder alias (downloads, desktop, documents, pictures, videos) or directory path.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        """Strictly validate file_search arguments against length, characters, and schemas."""
        if not isinstance(args, dict):
            return False, "Arguments must be a dictionary."
        extra_keys = set(args.keys()) - {"query", "location"}
        if extra_keys:
            return False, f"Unexpected arguments for file_search: {', '.join(sorted(extra_keys))}"
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return False, "Query must be a non-empty string."
        if len(query) > 200:
            return False, "Query exceeds maximum allowed length of 200 characters."
        if any(ord(c) < 32 for c in query):
            return False, "Query contains invalid control characters."

        location = args.get("location")
        if location is not None:
            if not isinstance(location, str):
                return False, "Location must be a string or null."
            if len(location) > 260:
                return False, "Location exceeds maximum allowed path length of 260 characters."
            if any(ord(c) < 32 for c in location):
                return False, "Location contains invalid control characters."
        return True, None

    patterns = [
        r"^(find|search for|locate)\s+(?:file\s+)?(.+)$",
    ]

    def extract_args(self, text: str) -> dict[str, Any]:
        match = re.search(r"^(?:find|search for|locate)\s+(?:file\s+)?(.+)$", text.strip(), re.IGNORECASE)
        raw_query = match.group(1).strip() if match else ""

        # Check if user specified a location filter: "find notes in Downloads"
        in_match = re.search(r"^(.*?)\s+in\s+([a-zA-Z0-9_\-\.\:\\]+)$", raw_query, re.IGNORECASE)
        if in_match:
            query = in_match.group(1).strip()
            location = in_match.group(2).strip()
        else:
            query = raw_query
            location = None

        return {"query": query, "location": location}

    def _resolve_search_roots(
        self,
        location: str | None,
        gatekeeper: SafetyGatekeeper | None,
    ) -> list[Path]:
        """Resolve and validate the directories to search."""
        if gatekeeper and gatekeeper.allowed_search_roots:
            default_roots = gatekeeper.allowed_search_roots
        else:
            user_home = Path.home()
            default_roots = [
                user_home / "Documents",
                user_home / "Downloads",
                user_home / "Desktop",
            ]

        if not location:
            return [r for r in default_roots if r.exists()]

        # Try mapping common alias names (e.g. "downloads", "desktop", "documents")
        loc_lower = location.lower()
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

        target_dir = alias_map.get(loc_lower, Path(location).resolve())
        if not target_dir.exists():
            return []

        # Validate with gatekeeper whitelist and blacklist
        if gatekeeper is not None:
            if not gatekeeper.is_path_safe_for_read(target_dir):
                return []
        else:
            # Fail-safe containment to default roots if gatekeeper is omitted
            is_contained = any(r == target_dir or r in target_dir.parents for r in default_roots)
            if not is_contained:
                return []

        return [target_dir]

    @staticmethod
    def _is_symlink_or_junction(path: Path) -> bool:
        """Detect both symbolic links and Windows NTFS directory junctions."""
        if path.is_symlink():
            return True
        if hasattr(path, "is_junction") and path.is_junction():
            return True
        return False

    def run(
        self,
        args: dict[str, Any],
        gatekeeper: SafetyGatekeeper | None = None,
        action_id: str | None = None,
    ) -> ToolResult:
        query = args.get("query", "").strip()
        location = args.get("location")
        max_results = args.get("max_results", 25)
        max_depth = args.get("max_depth", 3)

        if not query:
            return ToolResult(
                success=False,
                error="Search query cannot be empty.",
                safety_level=self.safety_level,
            )

        roots = self._resolve_search_roots(location, gatekeeper)
        if not roots:
            return ToolResult(
                success=False,
                error=(
                    f"Invalid or disallowed search location: '{location}'. "
                    "Searches are restricted to safe user directories."
                ),
                safety_level=self.safety_level,
            )

        query_clean = query.strip("*").lower()
        matches: list[dict[str, Any]] = []

        for root in roots:
            root_depth = len(root.parts)
            try:
                for current_root, dirs, files in os.walk(root, followlinks=False):
                    curr_path = Path(current_root)
                    if len(curr_path.parts) - root_depth > max_depth:
                        dirs.clear()  # Do not recurse deeper
                        continue

                    # Security: Do not follow or traverse directory symlinks or NTFS junctions
                    dirs[:] = [d for d in dirs if not self._is_symlink_or_junction(curr_path / d)]

                    for file_name in files:
                        if query_clean in file_name.lower():
                            full_path = curr_path / file_name
                            try:
                                stat = full_path.stat()
                                size_mb = round(stat.st_size / (1024 * 1024), 2)
                                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                            except OSError:
                                size_mb = 0.0
                                modified = "unknown"

                            matches.append(
                                {
                                    "filename": file_name,
                                    "path": str(full_path),
                                    "size_mb": size_mb,
                                    "modified": modified,
                                }
                            )
                            if len(matches) >= max_results:
                                break
                    if len(matches) >= max_results:
                        break
            except (PermissionError, OSError):
                continue
            if len(matches) >= max_results:
                break

        return ToolResult(
            success=True,
            data={
                "query": query,
                "matches_count": len(matches),
                "matches": matches,
            },
            safety_level=self.safety_level,
            metadata={"searched_roots": [str(r) for r in roots]},
        )
