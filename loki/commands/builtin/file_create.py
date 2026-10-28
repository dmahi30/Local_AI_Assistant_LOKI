"""Deterministic safe file and folder creation command restricted to approved user directories."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
from typing import Any

from ...core.types import IntentType, SafetyLevel, ToolResult
from ...safety.gatekeeper import SafetyGatekeeper
from ..base import BaseCommand

# Maximum allowed initial plain-text content size: 1 MB
DEFAULT_MAX_CONTENT_BYTES: int = 1_048_576

# Prohibited executable and script extensions
PROHIBITED_EXTENSIONS: set[str] = {
    ".exe", ".bat", ".cmd", ".ps1", ".vbs", ".com",
    ".msi", ".msp", ".scr", ".pif", ".wsf", ".wsh",
    ".jar", ".sh", ".bash", ".csh", ".reg", ".inf",
    ".dll", ".sys", ".drv", ".appx", ".msix",
}

# Windows reserved device names that cannot be created as files or folders
WINDOWS_RESERVED_NAMES: set[str] = {
    "con", "prn", "aux", "nul",
    "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
}

# Invalid filename characters on Windows filesystems
INVALID_FILENAME_CHARS: set[str] = set('<>:"/\\|?*')

GENERIC_LOC_WORDS: set[str] = {
    "an approved directory", "approved directory", "approved directories",
    "safe directory", "safe directories", "safe folder", "safe folders",
    "an approved folder", "approved folder", "approved folders",
    "an approved location", "approved location",
    "an approved", "approved", "safe", "safe root", "safe roots",
    "system", "computer", "pc",
}


class FileCreateCommand(BaseCommand):
    """Safely creates a new plain-text file or folder strictly within approved user directories."""

    name = "file_create"
    description = "Safely creates a new plain-text file or folder in approved user directories."
    intent_type = IntentType.FILE_CREATE
    safety_level = SafetyLevel.SAFE
    allow_model_selection = True

    def get_parameters_schema(self) -> dict[str, Any]:
        """Return JSON Schema for model-assisted tool calling."""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 255,
                    "description": "Name of the file or folder to create (e.g. 'notes.txt' or 'Projects').",
                },
                "location": {
                    "type": "string",
                    "maxLength": 260,
                    "description": "Safe folder alias (desktop, documents, downloads, pictures, videos, music, project) or directory path.",
                },
                "item_type": {
                    "type": "string",
                    "enum": ["file", "folder", "directory"],
                    "description": "Type of item to create: 'file' (default) or 'folder'.",
                },
                "content": {
                    "type": ["string", "null"],
                    "maxLength": DEFAULT_MAX_CONTENT_BYTES,
                    "description": "Optional initial plain-text content for files.",
                },
                "create_parents": {
                    "type": "boolean",
                    "description": "Whether to create missing parent directories (default: false).",
                },
            },
            "required": ["name", "location"],
            "additionalProperties": False,
        }

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        """Strictly validate file_create arguments against length, characters, and security schemas."""
        if not isinstance(args, dict):
            return False, "Arguments must be a dictionary."

        allowed_keys = {
            "name",
            "location",
            "item_type",
            "content",
            "create_parents",
            "target",
            "targets",
        }
        extra_keys = set(args.keys()) - allowed_keys
        if extra_keys:
            return False, f"Unexpected arguments for file_create: {', '.join(sorted(extra_keys))}"

        name = args.get("name")
        if not isinstance(name, str) or not name.strip():
            return False, "File or folder name must be a non-empty string."
        name = name.strip()
        if len(name) > 255:
            return False, "Name exceeds maximum allowed length of 255 characters."
        if any(ord(c) < 32 for c in name):
            return False, "Name contains invalid control characters."
        if any(c in INVALID_FILENAME_CHARS for c in name):
            return False, f"Name contains invalid filename characters: {name}"

        # Reserved Windows device names
        stem = Path(name).stem.lower()
        if stem in WINDOWS_RESERVED_NAMES or name.lower() in WINDOWS_RESERVED_NAMES:
            return False, f"'{name}' is a reserved Windows system device name and cannot be created."

        location = args.get("location")
        if not isinstance(location, str) or not location.strip():
            return False, "Location must be a non-empty string."
        if len(location) > 260:
            return False, "Location exceeds maximum allowed path length of 260 characters."
        if any(ord(c) < 32 for c in location):
            return False, "Location contains invalid control characters."

        item_type = args.get("item_type", "file")
        if item_type not in ("file", "folder", "directory"):
            return False, f"Invalid item_type '{item_type}'. Must be 'file' or 'folder'."

        # Prohibited executable / script extensions
        ext = Path(name).suffix.lower()
        if ext in PROHIBITED_EXTENSIONS:
            return False, f"Creating files with prohibited executable/script extension '{ext}' is not permitted."

        content = args.get("content")
        if content is not None:
            if not isinstance(content, str):
                return False, "Content must be a string."
            if item_type in ("folder", "directory"):
                return False, "Content cannot be provided when creating a folder."
            if len(content.encode("utf-8")) > DEFAULT_MAX_CONTENT_BYTES:
                return False, f"Content exceeds maximum allowed size of {DEFAULT_MAX_CONTENT_BYTES} bytes."

        create_parents = args.get("create_parents")
        if create_parents is not None and not isinstance(create_parents, bool):
            return False, "create_parents must be a boolean."

        return True, None

    patterns = [
        r"^(?:create|make|new)\s+(?:a\s+)?(?:new\s+)?(?:folder|directory|dir)\s+(?:(?:named|called)\s+)?(?:['\"]?)([^'\"\s]+)(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+(.+))?$",
        r"^(?:create|make|new)\s+(?:a\s+)?(?:new\s+)?(?:(?:text\s+)?file)\s+(?:(?:named|called)\s+)?(?:['\"]?)([^'\"\s]+)?(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+(.+?))?(?:\s+with\s+(?:(?:initial\s+)?content\s+)?['\"]?(.*?)['\"]?)?$",
    ]

    @staticmethod
    def _clean_location_string(raw: str) -> str | None:
        loc = raw.strip().rstrip(".").strip()
        loc = re.sub(r"^(?:in|on|at|inside|under)\s+", "", loc, flags=re.IGNORECASE).strip()
        loc = re.sub(r"^(?:my|the)\s+", "", loc, flags=re.IGNORECASE).strip()
        loc = re.sub(r"\s+(?:folder|directory|dir)$", "", loc, flags=re.IGNORECASE).strip()
        if loc.lower() in GENERIC_LOC_WORDS or not loc:
            return None
        return loc

    def extract_args(self, text: str) -> dict[str, Any]:
        cleaned = text.strip().rstrip(".").strip()

        # Check for folder creation pattern
        folder_match = re.search(
            r"^(?:create|make|new)\s+(?:a\s+)?(?:new\s+)?(?:folder|directory|dir)\s+(?:(?:named|called)\s+)?(?:['\"]?)([^'\"\s]+)(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+(.+))?$",
            cleaned,
            re.IGNORECASE,
        )
        if folder_match:
            name = folder_match.group(1).strip()
            raw_loc = folder_match.group(2)
            location = self._clean_location_string(raw_loc) if raw_loc else None
            return {
                "name": name,
                "location": location,
                "item_type": "folder",
                "content": None,
                "target": f"{location}/{name}" if location else name,
            }

        # Check for file creation pattern
        file_match = re.search(
            r"^(?:create|make|new)\s+(?:a\s+)?(?:new\s+)?(?:(?:text\s+)?file)\s+(?:(?:named|called)\s+)?(?:['\"]?)([^'\"\s]+)?(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+(.+?))?(?:\s+with\s+(?:(?:initial\s+)?content\s+)?['\"]?(.*?)['\"]?)?$",
            cleaned,
            re.IGNORECASE,
        )
        if file_match:
            raw_name = file_match.group(1)
            raw_loc = file_match.group(2)
            raw_content = file_match.group(3)

            name = raw_name.strip() if raw_name else None
            location = self._clean_location_string(raw_loc) if raw_loc else None
            content = raw_content.strip() if raw_content else None

            return {
                "name": name,
                "location": location,
                "item_type": "file",
                "content": content,
                "target": f"{location}/{name}" if location and name else name,
            }

        return {}

    def _resolve_target_dir(
        self,
        location: str,
        gatekeeper: SafetyGatekeeper | None,
    ) -> tuple[Path | None, str | None]:
        """Resolve and validate the destination directory for creation.

        Returns:
            A tuple of (target_dir, error_message).
        """
        user_home = Path.home().resolve()
        project_root = Path(__file__).resolve().parents[3]

        loc_clean = self._clean_location_string(location) or location.strip()
        loc_lower = loc_clean.lower()

        alias_map = {
            "downloads": user_home / "Downloads",
            "download": user_home / "Downloads",
            "documents": user_home / "Documents",
            "docs": user_home / "Documents",
            "doc": user_home / "Documents",
            "desktop": user_home / "Desktop",
            "pictures": user_home / "Pictures",
            "pics": user_home / "Pictures",
            "images": user_home / "Pictures",
            "videos": user_home / "Videos",
            "vids": user_home / "Videos",
            "music": user_home / "Music",
            "project": project_root,
            "loki project": project_root,
            "loki": project_root,
            "workspace": project_root,
        }

        if loc_lower in alias_map:
            target_dir = alias_map[loc_lower]
        else:
            try:
                target_dir = Path(loc_clean).resolve()
            except Exception:
                return None, f"Invalid location path: '{location}'."

        if not target_dir.exists():
            return None, f"Target directory does not exist: '{location}'."

        if not target_dir.is_dir():
            return None, f"Target location is not a directory: '{location}'."

        # Validate directory is safe for writing
        if gatekeeper is not None:
            if not gatekeeper.is_path_safe_for_write(target_dir):
                return None, (
                    f"Creation blocked: Target directory '{target_dir}' is outside approved write roots."
                )
        else:
            allowed_defaults = [
                user_home / "Documents",
                user_home / "Downloads",
                user_home / "Desktop",
                user_home / "Pictures",
                user_home / "Videos",
                user_home / "Music",
                project_root,
            ]
            is_contained = any(
                r == target_dir or r in target_dir.parents
                for r in allowed_defaults
            )
            if not is_contained:
                return None, (
                    f"Creation blocked: Target directory '{target_dir}' is outside approved write roots."
                )

        return target_dir, None

    def run(
        self,
        args: dict[str, Any],
        gatekeeper: SafetyGatekeeper | None = None,
        action_id: str | None = None,
    ) -> ToolResult:
        name = args.get("name")
        location = args.get("location")
        item_type = str(args.get("item_type", "file")).lower()
        content = args.get("content")
        create_parents = bool(args.get("create_parents", False))

        if not name or not isinstance(name, str) or not name.strip():
            return ToolResult(
                success=False,
                error="Please specify a valid name for the new file or folder.",
                safety_level=self.safety_level,
            )

        if not location or not isinstance(location, str) or not location.strip():
            return ToolResult(
                success=False,
                error="Destination location is required. Please specify where to create the file or folder (e.g. 'Desktop' or 'Documents').",
                safety_level=self.safety_level,
            )

        name = name.strip()

        # Check prohibited executable extensions
        ext = Path(name).suffix.lower()
        if ext in PROHIBITED_EXTENSIONS:
            return ToolResult(
                success=False,
                error=f"Creating files with prohibited executable/script extension '{ext}' is not permitted for security reasons.",
                safety_level=self.safety_level,
            )

        target_dir, err = self._resolve_target_dir(location, gatekeeper)
        if not target_dir:
            return ToolResult(
                success=False,
                error=err or f"Invalid or disallowed creation directory: '{location}'.",
                safety_level=self.safety_level,
            )

        destination = (target_dir / name).resolve()

        # Path traversal guard: ensure destination resides strictly within target_dir
        if sys.platform == "win32":
            dest_parts = [p.lower() for p in destination.parts]
            parent_parts = [p.lower() for p in target_dir.parts]
            is_inside = len(dest_parts) > len(parent_parts) and dest_parts[:len(parent_parts)] == parent_parts
        else:
            is_inside = target_dir in destination.parents

        if not is_inside:
            return ToolResult(
                success=False,
                error=f"Path traversal detected: '{name}' escapes target directory '{target_dir}'.",
                safety_level=self.safety_level,
            )

        # Enforce write containment on resolved destination
        if gatekeeper is not None:
            if not gatekeeper.is_path_safe_for_write(destination):
                return ToolResult(
                    success=False,
                    error=f"Creation blocked: Destination path '{destination}' is outside approved write roots.",
                    safety_level=self.safety_level,
                )

        # Check existing destination (fail-closed, never overwrite)
        if destination.exists():
            item_desc = "Directory" if destination.is_dir() else "File"
            return ToolResult(
                success=False,
                error=f"{item_desc} already exists at '{destination}'. Overwriting is not permitted.",
                safety_level=self.safety_level,
            )

        # Handle parent directory creation
        if not destination.parent.exists():
            if not create_parents:
                return ToolResult(
                    success=False,
                    error=f"Parent directory does not exist: '{destination.parent}'. Set create_parents=True to create intermediate directories.",
                    safety_level=self.safety_level,
                )
            if gatekeeper and not gatekeeper.is_path_safe_for_write(destination.parent):
                return ToolResult(
                    success=False,
                    error=f"Creation blocked: Parent directory '{destination.parent}' is outside approved write roots.",
                    safety_level=self.safety_level,
                )
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=f"Failed to create parent directories: {e}",
                    safety_level=self.safety_level,
                )

        # Execute safe creation
        try:
            if item_type in ("folder", "directory"):
                destination.mkdir(parents=False, exist_ok=False)
                return ToolResult(
                    success=True,
                    data={
                        "created": str(destination),
                        "name": destination.name,
                        "type": "directory",
                        "size_bytes": 0,
                    },
                    safety_level=self.safety_level,
                    metadata={"target": str(destination)},
                )
            else:
                # Exclusive atomic creation ('x' mode) to prevent race conditions and overwrites
                with open(destination, mode="x", encoding="utf-8") as f:
                    if content:
                        f.write(content)

                size_bytes = destination.stat().st_size
                return ToolResult(
                    success=True,
                    data={
                        "created": str(destination),
                        "name": destination.name,
                        "type": "file",
                        "size_bytes": size_bytes,
                    },
                    safety_level=self.safety_level,
                    metadata={"target": str(destination)},
                )
        except FileExistsError:
            return ToolResult(
                success=False,
                error=f"Destination already exists at '{destination}'. Overwriting is not permitted.",
                safety_level=self.safety_level,
            )
        except PermissionError as e:
            return ToolResult(
                success=False,
                error=f"Permission denied creating '{destination.name}': {e}",
                safety_level=self.safety_level,
            )
        except OSError as e:
            return ToolResult(
                success=False,
                error=f"Filesystem error creating '{destination.name}': {e}",
                safety_level=self.safety_level,
            )


class FolderCreateCommand(FileCreateCommand):
    """Dedicated command for creating folders/directories in approved user directories."""

    name = "folder_create"
    description = "Safely creates a new folder/directory in approved user directories."
    intent_type = IntentType.FILE_CREATE
    safety_level = SafetyLevel.SAFE
    allow_model_selection = True

    patterns = [
        r"^(?:create|make|new)\s+(?:a\s+)?(?:new\s+)?(?:folder|directory|dir)\s+(?:(?:named|called)\s+)?(?:['\"]?)([^'\"\s]+)(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+(.+))?$",
    ]

    def extract_args(self, text: str) -> dict[str, Any]:
        args = super().extract_args(text)
        args["item_type"] = "folder"
        return args
