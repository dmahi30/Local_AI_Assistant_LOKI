"""Deterministic file search and location command restricted to safe user directories."""

from __future__ import annotations

from datetime import datetime
import fnmatch
import os
from pathlib import Path
import re
from typing import Any

from ...core.types import IntentType, SafetyLevel, ToolResult
from ...safety.gatekeeper import SafetyGatekeeper
from ..base import BaseCommand

COMMON_EXT_MAP = {
    "python": ".py",
    "py": ".py",
    "pdf": ".pdf",
    "text": ".txt",
    "txt": ".txt",
    "markdown": ".md",
    "md": ".md",
    "json": ".json",
    "csv": ".csv",
    "excel": ".xlsx",
    "word": ".docx",
    "doc": ".docx",
    "docx": ".docx",
    "image": ".png",
    "images": ".png",
    "log": ".log",
    "zip": ".zip",
}

GENERIC_EXT_WORDS = {"particular", "specific", "given", "certain", "any", "some", "custom"}
GENERIC_LOC_WORDS = {
    "an approved directory", "approved directory", "approved directories",
    "safe directory", "safe directories", "safe folder", "safe folders",
    "an approved folder", "approved folder", "approved folders",
    "an approved location", "approved location",
    "an approved", "approved", "safe", "safe root", "safe roots",
    "system", "computer", "pc",
}


class FileSearchCommand(BaseCommand):
    """Searches for files and folders by name/extension strictly within allowed user directories."""

    name = "file_search"
    description = "Searches for files and folders matching a keyword or pattern in safe user directories, or lists directory contents."
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
                    "maxLength": 200,
                    "description": "File search pattern, filename, keyword, or '*' to match/list all.",
                },
                "location": {
                    "type": ["string", "null"],
                    "maxLength": 260,
                    "description": "Optional safe folder alias (downloads, desktop, documents, pictures, videos, music, project) or directory path.",
                },
                "extension": {
                    "type": ["string", "null"],
                    "maxLength": 20,
                    "description": "Optional file extension filter (e.g. '.py', '.pdf', '.txt').",
                },
                "target_type": {
                    "type": ["string", "null"],
                    "enum": ["file", "directory", "both", None],
                    "description": "Search target: 'file' for files, 'directory' for folders, or 'both'.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        """Strictly validate file_search arguments against length, characters, and schemas."""
        if not isinstance(args, dict):
            return False, "Arguments must be a dictionary."

        allowed_keys = {
            "query",
            "location",
            "extension",
            "target_type",
            "max_results",
            "max_depth",
            "list_only",
        }
        extra_keys = set(args.keys()) - allowed_keys
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

        extension = args.get("extension")
        if extension is not None:
            if not isinstance(extension, str):
                return False, "Extension must be a string or null."
            if len(extension) > 20:
                return False, "Extension exceeds maximum allowed length of 20 characters."
            if any(ord(c) < 32 for c in extension) or "/" in extension or "\\" in extension:
                return False, "Extension contains invalid characters."

        target_type = args.get("target_type")
        if target_type is not None:
            if not isinstance(target_type, str):
                return False, "Target type must be a string."
            if target_type.lower() not in {
                "file", "files", "directory", "directories", "folder", "folders", "both", "all"
            }:
                return False, f"Invalid target_type '{target_type}'. Must be 'file', 'directory', or 'both'."

        max_results = args.get("max_results")
        if max_results is not None:
            if not isinstance(max_results, int) or max_results < 1:
                return False, "max_results must be a positive integer."

        max_depth = args.get("max_depth")
        if max_depth is not None:
            if not isinstance(max_depth, int) or max_depth < 1:
                return False, "max_depth must be a positive integer."

        return True, None

    patterns = [
        r"^(?:find|search(?:\s+for)?|locate)\s+(?:all\s+)?(?:the\s+)?(.+)$",
        r"^(?:list|show|dir|ls)\s+(?:all\s+)?(?:the\s+)?(?:files|folders|directories|contents)?(?:\s+(?:in|on|inside|of|under)\s+.+)?$",
    ]

    @staticmethod
    def _clean_location_string(raw: str) -> str | None:
        loc = raw.strip().rstrip(".").strip()
        loc = re.sub(r"^(?:in|on|inside|of|under)\s+", "", loc, flags=re.IGNORECASE).strip()
        loc = re.sub(r"^(?:my|the)\s+", "", loc, flags=re.IGNORECASE).strip()
        loc = re.sub(r"\s+(?:folder|directory|dir)$", "", loc, flags=re.IGNORECASE).strip()
        if loc.lower() in GENERIC_LOC_WORDS or not loc:
            return None
        return loc

    def extract_args(self, text: str) -> dict[str, Any]:
        cleaned = text.strip().rstrip(".").strip()

        # Check for directory listing pattern
        list_match = re.search(
            r"^(?:list|show|dir|ls)\s+(?:all\s+)?(?:the\s+)?(?:files|folders|directories|contents)?(?:\s+(?:in|on|inside|of|under)\s+(.+))?$",
            cleaned,
            re.IGNORECASE,
        )
        if list_match:
            raw_loc = list_match.group(1)
            location = self._clean_location_string(raw_loc) if raw_loc else None
            target_type = "both"
            if re.search(r"\bfiles\b", cleaned, re.IGNORECASE):
                target_type = "file"
            elif re.search(r"\b(?:folders|directories)\b", cleaned, re.IGNORECASE):
                target_type = "directory"
            return {
                "query": "*",
                "location": location,
                "target_type": target_type,
                "list_only": True,
            }

        # Check for search / find / locate pattern
        search_match = re.search(
            r"^(?:find|search(?:\s+for)?|locate)\s+(?:all\s+)?(?:the\s+)?(.+)$",
            cleaned,
            re.IGNORECASE,
        )
        if not search_match:
            return {"query": cleaned or "*", "location": None}

        remainder = search_match.group(1).strip()

        # Determine target_type from phrasing
        target_type = "both"
        if re.search(r"\b(?:folders?|directories?)\b", remainder, re.IGNORECASE):
            target_type = "directory"
        elif re.search(r"\bfiles?\b", remainder, re.IGNORECASE):
            target_type = "file"

        # Check for location filter at end
        loc_match = re.search(
            r"^(.*?)\s+(?:in|on|inside|under)\s+([a-zA-Z0-9_\-\.\:\\\s]+)$",
            remainder,
            re.IGNORECASE,
        )
        if loc_match:
            body = loc_match.group(1).strip()
            location = self._clean_location_string(loc_match.group(2))
        else:
            body = remainder
            location = None

        # Check for explicit extension pattern
        extension = None
        ext_match = re.search(
            r"(?:files?\s+)?with\s+(?:(?:a|an|the)\s+)?(?:(?:particular|specific|given)\s+)?(?:extension\s+([.\w]+)|([.\w]+)\s+extension)",
            body,
            re.IGNORECASE,
        )
        if ext_match:
            ext_val = ext_match.group(1) or ext_match.group(2)
            if ext_val:
                ext_val = ext_val.strip().lower()
                if ext_val not in GENERIC_EXT_WORDS:
                    if not ext_val.startswith("."):
                        ext_val = "." + ext_val
                    extension = ext_val
            body = re.sub(
                r"(?:files?\s+)?with\s+(?:(?:a|an|the)\s+)?(?:(?:particular|specific|given)\s+)?(?:extension\s+[.\w]+|[.\w]+\s+extension)",
                "",
                body,
                flags=re.IGNORECASE,
            ).strip()

        # Check for language type file pattern: e.g. "Python files", "PDF files", "text files"
        type_file_match = re.search(r"^(?:all\s+)?(\w+)\s+files?$", body, re.IGNORECASE)
        if type_file_match:
            type_word = type_file_match.group(1).lower()
            if type_word in COMMON_EXT_MAP:
                extension = COMMON_EXT_MAP[type_word]
                body = ""
            else:
                body = type_word

        # Clean redundant prefixes
        body = re.sub(r"^(?:folders?|directories?|files?)\s+named\s+", "", body, flags=re.IGNORECASE).strip()
        body = re.sub(r"^(?:folders?|directories?|files?)\s+", "", body, flags=re.IGNORECASE).strip()

        query = body.strip()
        if not query and extension:
            query = f"*{extension}"

        return {
            "query": query or "*",
            "location": location,
            "extension": extension,
            "target_type": target_type,
        }

    def _resolve_search_roots(
        self,
        location: str | None,
        gatekeeper: SafetyGatekeeper | None,
    ) -> tuple[list[Path], str | None]:
        """Resolve and validate the directories to search.

        Returns:
            A tuple of (roots, error_message).
        """
        user_home = Path.home().resolve()
        project_root = Path(__file__).resolve().parents[3]

        if not location:
            if gatekeeper and gatekeeper.allowed_search_roots:
                default_roots = [
                    r for r in gatekeeper.allowed_search_roots
                    if r != user_home
                ]
                if not default_roots:
                    default_roots = gatekeeper.allowed_search_roots
            else:
                default_roots = [
                    user_home / "Documents",
                    user_home / "Downloads",
                    user_home / "Desktop",
                ]
            roots = [r for r in default_roots if r.exists()]
            if not roots:
                return [], "No safe default search directories found."
            return roots, None

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
                return [], f"Invalid location path: '{location}'."

        if not target_dir.exists():
            return [], f"Specified search location does not exist: '{location}'."

        if not target_dir.is_dir():
            return [], f"Specified search location is not a directory: '{location}'."

        # Security: ensure target directory itself is not a symlink/junction escaping safe boundaries
        if self._is_symlink_or_junction(target_dir):
            try:
                resolved_junction = target_dir.resolve()
                if gatekeeper and not gatekeeper.is_path_safe_for_read(resolved_junction):
                    return [], f"Invalid or disallowed search location: '{location}'. Symlink/junction target is outside safe directories."
            except Exception:
                return [], f"Invalid search location: '{location}'."

        # Validate with gatekeeper whitelist and blacklist
        if gatekeeper is not None:
            if not gatekeeper.is_path_safe_for_read(target_dir):
                return [], (
                    f"Invalid or disallowed search location: '{location}'. "
                    "Searches are restricted to safe user directories."
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
                return [], (
                    f"Invalid or disallowed search location: '{location}'. "
                    "Searches are restricted to safe user directories."
                )

        return [target_dir], None

    @staticmethod
    def _is_symlink_or_junction(path: Path) -> bool:
        """Detect both symbolic links and Windows NTFS directory junctions."""
        if path.is_symlink():
            return True
        if hasattr(path, "is_junction") and path.is_junction():
            return True
        return False

    @staticmethod
    def _matches_item(name: str, raw_query: str, query_clean: str, is_dir: bool = False) -> bool:
        """Evaluate if an item matches the query."""
        if not raw_query or raw_query in ("*", "*.*"):
            return True
        name_lower = name.lower()
        if "*" in raw_query or "?" in raw_query:
            return fnmatch.fnmatch(name_lower, raw_query.lower())
        return query_clean in name_lower

    def run(
        self,
        args: dict[str, Any],
        gatekeeper: SafetyGatekeeper | None = None,
        action_id: str | None = None,
    ) -> ToolResult:
        query = str(args.get("query", "")).strip()
        location = args.get("location")
        is_list_only = bool(args.get("list_only", False))
        max_results = min(max(int(args.get("max_results", 25)), 1), 100)
        max_depth = 1 if is_list_only else min(max(int(args.get("max_depth", 3)), 1), 5)

        target_type = str(args.get("target_type", "both")).lower()
        if target_type in ("files", "file"):
            search_files, search_dirs = True, False
        elif target_type in ("directories", "directory", "folders", "folder"):
            search_files, search_dirs = False, True
        else:
            search_files, search_dirs = True, True

        extension = args.get("extension")
        if extension:
            extension = str(extension).strip().lower()
            if not extension.startswith("."):
                extension = "." + extension

        if not query and not extension and not is_list_only:
            return ToolResult(
                success=False,
                error="Search query cannot be empty.",
                safety_level=self.safety_level,
            )

        roots, err = self._resolve_search_roots(location, gatekeeper)
        if not roots:
            return ToolResult(
                success=False,
                error=err or (
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
                    depth_diff = len(curr_path.parts) - root_depth

                    if depth_diff >= max_depth:
                        dirs.clear()
                        continue

                    # Security: Do not follow or traverse directory symlinks or NTFS junctions
                    dirs[:] = [d for d in dirs if not self._is_symlink_or_junction(curr_path / d)]

                    if depth_diff >= max_depth - 1:
                        current_dirs = list(dirs)
                        dirs.clear()
                    else:
                        current_dirs = list(dirs)

                    # Check directories if requested (and no file extension filter is set)
                    if search_dirs and not extension:
                        for d_name in current_dirs:
                            if self._matches_item(d_name, query, query_clean, is_dir=True):
                                d_full = curr_path / d_name
                                try:
                                    d_res = d_full.resolve()
                                    if gatekeeper and not gatekeeper.is_path_safe_for_read(d_res):
                                        continue
                                    stat = d_full.stat()
                                    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                                except OSError:
                                    modified = "unknown"

                                matches.append(
                                    {
                                        "filename": d_name,
                                        "name": d_name,
                                        "path": str(d_full.resolve()),
                                        "type": "directory",
                                        "size_bytes": 0,
                                        "size_mb": 0.0,
                                        "modified": modified,
                                        "extension": "",
                                    }
                                )
                                if len(matches) >= max_results:
                                    break
                        if len(matches) >= max_results:
                            break

                    # Check files if requested
                    if search_files:
                        for file_name in files:
                            f_path = curr_path / file_name

                            # Check extension filter first if specified
                            if extension and f_path.suffix.lower() != extension:
                                continue

                            if self._matches_item(file_name, query, query_clean, is_dir=False):
                                # Security: If file is a symlink, verify resolved target stays within safe read roots
                                if f_path.is_symlink():
                                    try:
                                        target_res = f_path.resolve()
                                        if gatekeeper and not gatekeeper.is_path_safe_for_read(target_res):
                                            continue  # Escapes safe roots
                                    except Exception:
                                        continue

                                try:
                                    stat = f_path.stat()
                                    size_bytes = stat.st_size
                                    size_mb = round(size_bytes / (1024 * 1024), 2)
                                    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                                except OSError:
                                    size_bytes = 0
                                    size_mb = 0.0
                                    modified = "unknown"

                                matches.append(
                                    {
                                        "filename": file_name,
                                        "name": file_name,
                                        "path": str(f_path.resolve()),
                                        "type": "file",
                                        "size_bytes": size_bytes,
                                        "size_mb": size_mb,
                                        "modified": modified,
                                        "extension": f_path.suffix.lower(),
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
                "location": location,
                "target_type": target_type,
                "matches_count": len(matches),
                "matches": matches,
            },
            safety_level=self.safety_level,
            metadata={"searched_roots": [str(r) for r in roots]},
        )
