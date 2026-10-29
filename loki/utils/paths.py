"""Path management, Windows known-folder resolution, and privacy-safe path sanitization for LOKI."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
from typing import Any

APP_NAME = "LOKI"

# Windows Known Folder GUIDs (FOLDERID_* from Windows SDK)
KNOWN_FOLDER_GUIDS: dict[str, str] = {
    "desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "music": "{4BD8D570-50AB-4987-A2F4-53147C4F2A07}",
    "videos": "{18989B1D-99B5-455B-841C-AB7C77E1D4F2}",
}

# Windows Explorer User Shell Folders registry value names
KNOWN_FOLDER_REG_NAMES: dict[str, str] = {
    "desktop": "Desktop",
    "documents": "Personal",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "pictures": "My Pictures",
    "music": "My Music",
    "videos": "My Video",
}

# Supported natural language alias mappings to canonical names
CANONICAL_FOLDER_ALIASES: dict[str, str] = {
    "desktop": "desktop",
    "documents": "documents",
    "document": "documents",
    "docs": "documents",
    "doc": "documents",
    "downloads": "downloads",
    "download": "downloads",
    "pictures": "pictures",
    "picture": "pictures",
    "pics": "pictures",
    "images": "pictures",
    "photos": "pictures",
    "music": "music",
    "audio": "music",
    "videos": "videos",
    "video": "videos",
    "movies": "videos",
    "project": "project",
    "loki project": "project",
    "loki": "project",
    "workspace": "project",
}


def get_default_app_dir() -> Path:
    """Get the default platform-appropriate application directory.

    On Windows: %LOCALAPPDATA%\\LOKI (e.g. C:\\Users\\<user>\\AppData\\Local\\LOKI)
    Fallback / Other OS: ~/.loki
    """
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_NAME
    return Path.home() / f".{APP_NAME.lower()}"


def get_app_dir(custom_base: Path | str | None = None) -> Path:
    """Return the base application directory."""
    if custom_base is not None:
        return Path(custom_base)
    env_override = os.environ.get("LOKI_HOME")
    if env_override:
        return Path(env_override)
    return get_default_app_dir()


def get_logs_dir(base_dir: Path | str | None = None) -> Path:
    """Return the directory where local logs are stored."""
    return get_app_dir(base_dir) / "logs"


def get_models_dir(base_dir: Path | str | None = None) -> Path:
    """Return the directory where local offline models/weights are stored."""
    return get_app_dir(base_dir) / "models"


def get_data_dir(base_dir: Path | str | None = None) -> Path:
    """Return the directory where local databases (SQLite, vector index) are stored."""
    return get_app_dir(base_dir) / "data"


def get_config_path(base_dir: Path | str | None = None) -> Path:
    """Return the path to the configuration JSON file."""
    return get_app_dir(base_dir) / "config.json"


def ensure_app_dirs(base_dir: Path | str | None = None) -> dict[str, Path]:
    """Create all required LOKI application directories if they do not exist.

    Returns a dictionary of created/verified directory paths.
    """
    base = get_app_dir(base_dir)
    dirs = {
        "app": base,
        "logs": get_logs_dir(base),
        "models": get_models_dir(base),
        "data": get_data_dir(base),
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def sanitize_path_for_logging(path_str: str | Path) -> str:
    """Sanitize paths for logs to protect user privacy.

    Replaces user home directories (e.g. C:\\Users\\johndoe) with <USER_HOME>
    so that sensitive user names are never leaked into logs.
    """
    raw = str(path_str)
    home = str(Path.home())
    if home and home in raw:
        return raw.replace(home, "<USER_HOME>")
    return raw


def get_project_root() -> Path:
    """Return the LOKI project repository root directory."""
    return Path(__file__).resolve().parents[2]


def clean_location_alias(raw: str | Path | None) -> str | None:
    """Strip prepositions, articles, and 'folder/dir' suffixes from a location alias."""
    if raw is None:
        return None
    loc = str(raw).strip().rstrip(".").strip()
    loc = re.sub(r"^(?:in|on|at|inside|of|under)\s+", "", loc, flags=re.IGNORECASE).strip()
    loc = re.sub(r"^(?:my|the)\s+", "", loc, flags=re.IGNORECASE).strip()
    loc = re.sub(r"\s+(?:folder|directory|dir)$", "", loc, flags=re.IGNORECASE).strip()
    return loc or None


def _get_known_folder_via_win32(canon: str) -> Path | None:
    """Query Windows SHGetKnownFolderPath via ctypes."""
    if sys.platform != "win32":
        return None
    guid_str = KNOWN_FOLDER_GUIDS.get(canon)
    if not guid_str:
        return None

    try:
        import ctypes
        from ctypes import wintypes
        import uuid

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", wintypes.BYTE * 8),
            ]

        u = uuid.UUID(guid_str)
        guid = GUID(
            u.time_low,
            u.time_mid,
            u.time_hi_version,
            (wintypes.BYTE * 8)(*u.bytes[8:16]),
        )
        p_path = wintypes.LPWSTR()
        res = ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(guid), 0, None, ctypes.byref(p_path)
        )
        if res == 0 and p_path.value:
            val = p_path.value
            ctypes.windll.ole32.CoTaskMemFree(p_path)
            candidate = Path(val).resolve()
            if candidate.exists():
                return candidate
    except Exception:
        pass
    return None


def _get_known_folder_via_registry(canon: str) -> Path | None:
    """Query Windows User Shell Folders registry key."""
    if sys.platform != "win32":
        return None
    reg_name = KNOWN_FOLDER_REG_NAMES.get(canon)
    if not reg_name:
        return None

    try:
        import winreg

        reg_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path) as key:
            val, _ = winreg.QueryValueEx(key, reg_name)
            expanded = os.path.expandvars(val)
            candidate = Path(expanded).resolve()
            if candidate.exists():
                return candidate
    except Exception:
        pass
    return None


def _get_known_folder_via_onedrive(canon: str) -> Path | None:
    """Check OneDrive redirection environment variables."""
    for env_var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        od = os.environ.get(env_var)
        if od:
            candidate = (Path(od) / canon.capitalize()).resolve()
            if candidate.exists():
                return candidate
    return None


def resolve_known_folder(alias: str | Path | None) -> Path | None:
    """Resolve a known folder alias to its actual existing directory path on disk.

    Supports:
    - Desktop (including OneDrive redirection)
    - Documents (including OneDrive redirection)
    - Downloads
    - Pictures (including OneDrive redirection)
    - Music
    - Videos
    - LOKI Project / Workspace

    Prefers Windows Known Folder API and User Shell Folders registry before
    falling back to standard profile subdirectories.
    """
    cleaned = clean_location_alias(alias)
    if not cleaned:
        return None

    canon = CANONICAL_FOLDER_ALIASES.get(cleaned.lower())
    if not canon:
        return None

    if canon == "project":
        root = get_project_root()
        if root.exists():
            return root
        return None

    # 1. Windows Known Folder API (SHGetKnownFolderPath)
    path = _get_known_folder_via_win32(canon)
    if path and path.exists():
        return path

    # 2. Windows User Shell Folders Registry
    path = _get_known_folder_via_registry(canon)
    if path and path.exists():
        return path

    # 3. OneDrive redirection
    path = _get_known_folder_via_onedrive(canon)
    if path and path.exists():
        return path

    # 4. Standard User Home (~/Desktop, ~/Documents, etc.)
    user_home = Path.home().resolve()
    candidate = (user_home / canon.capitalize()).resolve()
    if candidate.exists():
        return candidate

    return None


def get_all_known_folder_paths() -> list[Path]:
    """Return all detected and resolved Windows known folder paths that currently exist on disk."""
    found: list[Path] = []
    for alias in ["desktop", "documents", "downloads", "pictures", "videos", "music"]:
        p = resolve_known_folder(alias)
        if p and p.exists() and p not in found:
            found.append(p)
    return found


def get_default_search_roots() -> list[Path]:
    """Return default safe user content roots for search when no location is specified."""
    roots: list[Path] = []
    # Prioritize active known folders (Desktop, Documents, Downloads, Pictures, etc.)
    for kf in get_all_known_folder_paths():
        if kf not in roots:
            roots.append(kf)

    user_home = Path.home().resolve()
    for sub in ["Documents", "Downloads", "Desktop", "Pictures"]:
        p = (user_home / sub).resolve()
        if p.exists() and p not in roots:
            roots.append(p)

    return roots


def resolve_safe_directory(
    location: str | Path | None,
    gatekeeper: Any = None,
    for_write: bool = False,
) -> tuple[Path | None, str | None]:
    """Resolve a directory alias or path and validate against SafetyGatekeeper.

    Handles:
    - Supported aliases (Desktop, Documents, Downloads, Pictures, Videos, Music, Project).
    - Windows Known Folder & OneDrive redirection.
    - Test isolation where gatekeeper specifies custom allowed roots.
    - Path existence, directory type verification, and gatekeeper read/write boundary checks.

    Returns:
        (resolved_dir, error_message). On success, error_message is None.
    """
    cleaned = clean_location_alias(location)
    if not cleaned:
        action_word = "Target" if for_write else "Search"
        return None, f"{action_word} location was not specified."

    user_home = Path.home().resolve()
    target_dir: Path | None = None

    norm_alias = cleaned.lower()
    if norm_alias in CANONICAL_FOLDER_ALIASES:
        canon = CANONICAL_FOLDER_ALIASES[norm_alias]

        # 1. In test environments where gatekeeper has custom allowed roots, check if any custom root contains the folder
        custom_roots: list[Path] = []
        if gatekeeper:
            if for_write and getattr(gatekeeper, "allowed_write_roots", None):
                custom_roots.extend(gatekeeper.allowed_write_roots)
            if getattr(gatekeeper, "allowed_search_roots", None):
                custom_roots.extend(gatekeeper.allowed_search_roots)
        for r in custom_roots:
            if r != user_home:
                for c_name in (canon.capitalize(), canon):
                    candidate = (r / c_name).resolve()
                    if candidate.exists() and candidate.is_dir():
                        target_dir = candidate
                        break
            if target_dir:
                break

        # 2. Query system known folders (handles Windows OneDrive redirection)
        if target_dir is None:
            target_dir = resolve_known_folder(cleaned)

        # 3. Fallback to user home
        if target_dir is None:
            fallback = (user_home / canon.capitalize()).resolve()
            if fallback.exists() and fallback.is_dir():
                target_dir = fallback
            else:
                action_word = "Target" if for_write else "Specified search"
                return None, f"{action_word} directory does not exist: '{location}'."
    else:
        # Literal filesystem path
        try:
            target_dir = Path(cleaned).resolve()
        except Exception:
            return None, f"Invalid location path: '{location}'."

    if not target_dir.exists():
        action_word = "Target" if for_write else "Specified search"
        return None, f"{action_word} directory does not exist: '{location}'."

    if not target_dir.is_dir():
        action_word = "Target" if for_write else "Specified search"
        return None, f"{action_word} location is not a directory: '{location}'."

    # Gatekeeper boundary validation
    if gatekeeper is not None:
        if for_write:
            if not gatekeeper.is_path_safe_for_write(target_dir):
                return None, f"Creation blocked: Target directory '{target_dir}' is outside approved write roots."
        else:
            if not gatekeeper.is_path_safe_for_read(target_dir):
                return None, f"Invalid or disallowed search location: '{location}'. Searches are restricted to safe user directories."
    else:
        # Standalone containment check
        allowed_defaults = [
            user_home / "Documents",
            user_home / "Downloads",
            user_home / "Desktop",
            user_home / "Pictures",
            user_home / "Videos",
            user_home / "Music",
            get_project_root(),
        ] + get_all_known_folder_paths()
        is_contained = any(
            r == target_dir or r in target_dir.parents
            for r in allowed_defaults
        )
        if not is_contained:
            if for_write:
                return None, f"Creation blocked: Target directory '{target_dir}' is outside approved write roots."
            else:
                return None, f"Invalid or disallowed search location: '{location}'. Searches are restricted to safe user directories."

    return target_dir, None
