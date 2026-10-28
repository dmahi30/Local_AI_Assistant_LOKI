"""Tests for LOKI Phase 9.2 safe deterministic file and folder creation."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from loki.agent import LokiAgent
from loki.commands.builtin.file_create import (
    DEFAULT_MAX_CONTENT_BYTES,
    PROHIBITED_EXTENSIONS,
    WINDOWS_RESERVED_NAMES,
    FileCreateCommand,
    FolderCreateCommand,
)
from loki.commands.model_selector import parse_and_validate_tool_call
from loki.commands.registry import CommandRegistry
from loki.commands.router import DeterministicRouter
from loki.core.types import SafetyLevel, ToolResult
from loki.safety.gatekeeper import SafetyGatekeeper


@pytest.fixture
def create_env(tmp_path: Path):
    """Fixture providing isolated temporary directories and Gatekeeper for safe creation."""
    safe_root = tmp_path / "SafeUserRoot"
    safe_root.mkdir()

    desktop = safe_root / "Desktop"
    desktop.mkdir()

    documents = safe_root / "Documents"
    documents.mkdir()

    protected_root = tmp_path / "ProtectedSystem"
    protected_root.mkdir()

    gk = SafetyGatekeeper(
        protected_paths=[protected_root],
        allowed_search_roots=[safe_root],
    )
    cmd = FileCreateCommand()

    return {
        "safe_root": safe_root,
        "desktop": desktop,
        "documents": documents,
        "protected_root": protected_root,
        "gk": gk,
        "cmd": cmd,
    }


def test_successful_file_creation(create_env):
    cmd = create_env["cmd"]
    gk = create_env["gk"]
    desktop = create_env["desktop"]

    res = cmd.execute(
        {
            "name": "notes.txt",
            "location": str(desktop),
            "content": "Meeting notes content",
            "item_type": "file",
        },
        gatekeeper=gk,
    )
    assert res.success is True
    target_file = desktop / "notes.txt"
    assert target_file.exists()
    assert target_file.is_file()
    assert target_file.read_text(encoding="utf-8") == "Meeting notes content"
    assert res.data["created"] == str(target_file.resolve())
    assert res.data["type"] == "file"
    assert res.data["size_bytes"] == len("Meeting notes content".encode("utf-8"))


def test_successful_folder_creation(create_env):
    cmd = create_env["cmd"]
    gk = create_env["gk"]
    desktop = create_env["desktop"]

    res = cmd.execute(
        {
            "name": "Projects",
            "location": str(desktop),
            "item_type": "folder",
        },
        gatekeeper=gk,
    )
    assert res.success is True
    target_dir = desktop / "Projects"
    assert target_dir.exists()
    assert target_dir.is_dir()
    assert res.data["created"] == str(target_dir.resolve())
    assert res.data["type"] == "directory"


def test_existing_file_collision_fails(create_env):
    cmd = create_env["cmd"]
    gk = create_env["gk"]
    desktop = create_env["desktop"]

    # Pre-existing file
    target_file = desktop / "already_exists.txt"
    target_file.write_text("original content", encoding="utf-8")

    # Attempt to create file at existing path
    res = cmd.execute(
        {
            "name": "already_exists.txt",
            "location": str(desktop),
            "content": "new malicious overwrite content",
        },
        gatekeeper=gk,
    )
    assert res.success is False
    assert "already exists" in res.error.lower()
    # Ensure original content was preserved and NOT overwritten
    assert target_file.read_text(encoding="utf-8") == "original content"


def test_existing_folder_collision_fails(create_env):
    cmd = create_env["cmd"]
    gk = create_env["gk"]
    desktop = create_env["desktop"]

    # Pre-existing directory
    target_dir = desktop / "ExistingFolder"
    target_dir.mkdir()

    res = cmd.execute(
        {
            "name": "ExistingFolder",
            "location": str(desktop),
            "item_type": "folder",
        },
        gatekeeper=gk,
    )
    assert res.success is False
    assert "already exists" in res.error.lower()


def test_path_traversal_prevention(create_env):
    cmd = create_env["cmd"]
    gk = create_env["gk"]
    desktop = create_env["desktop"]

    # Attempt directory traversal with .. to escape target_dir
    res = cmd.execute(
        {
            "name": "../escaped_note.txt",
            "location": str(desktop),
            "content": "escaped",
        },
        gatekeeper=gk,
    )
    assert res.success is False

    # Control characters in name
    valid, err = cmd.validate_args(
        {"name": "test\x00file.txt", "location": str(desktop)}
    )
    assert valid is False
    assert "control characters" in err

    # Control characters in location
    valid, err = cmd.validate_args(
        {"name": "test.txt", "location": "C:\\Safe\nFolder"}
    )
    assert valid is False
    assert "control characters" in err


def test_prohibited_executable_extensions(create_env):
    cmd = create_env["cmd"]
    gk = create_env["gk"]
    desktop = create_env["desktop"]

    for ext in [".exe", ".bat", ".cmd", ".ps1", ".vbs", ".com", ".msi", ".dll"]:
        res = cmd.execute(
            {
                "name": f"malicious{ext}",
                "location": str(desktop),
                "content": "echo bad",
            },
            gatekeeper=gk,
        )
        assert res.success is False
        assert "prohibited executable/script extension" in res.error
        assert not (desktop / f"malicious{ext}").exists()


def test_windows_reserved_device_names(create_env):
    cmd = create_env["cmd"]
    desktop = create_env["desktop"]

    for res_name in ["CON", "con.txt", "NUL", "prn.log", "COM1", "lpt2.dat"]:
        valid, err = cmd.validate_args(
            {"name": res_name, "location": str(desktop)}
        )
        assert valid is False
        assert "reserved Windows system device name" in err


def test_invalid_filename_characters(create_env):
    cmd = create_env["cmd"]
    desktop = create_env["desktop"]

    for bad_name in ["test<file.txt", "file>name.txt", 'file"name.txt', "pipe|file.txt", "star*file.txt", "quest?file.txt"]:
        valid, err = cmd.validate_args(
            {"name": bad_name, "location": str(desktop)}
        )
        assert valid is False
        assert "invalid filename characters" in err


def test_content_size_limit(create_env):
    cmd = create_env["cmd"]
    desktop = create_env["desktop"]

    oversized_content = "x" * (DEFAULT_MAX_CONTENT_BYTES + 10)
    valid, err = cmd.validate_args(
        {
            "name": "large.txt",
            "location": str(desktop),
            "content": oversized_content,
        }
    )
    assert valid is False
    assert "exceeds maximum allowed size" in err


def test_content_forbidden_for_folders(create_env):
    cmd = create_env["cmd"]
    desktop = create_env["desktop"]

    valid, err = cmd.validate_args(
        {
            "name": "NewFolder",
            "location": str(desktop),
            "item_type": "folder",
            "content": "some text",
        }
    )
    assert valid is False
    assert "Content cannot be provided when creating a folder" in err


def test_creation_outside_safe_roots_blocked(create_env):
    cmd = create_env["cmd"]
    gk = create_env["gk"]
    protected_root = create_env["protected_root"]

    res = cmd.execute(
        {
            "name": "injected.txt",
            "location": str(protected_root),
            "content": "kernel tweak",
        },
        gatekeeper=gk,
    )
    assert res.success is False
    assert "outside approved write roots" in res.error.lower()
    assert not (protected_root / "injected.txt").exists()


def test_parent_directory_creation_flag(create_env):
    cmd = create_env["cmd"]
    gk = create_env["gk"]
    desktop = create_env["desktop"]

    nested_loc = desktop / "Deep" / "SubDir"

    # Default create_parents=False fails when parents do not exist
    res_no_parents = cmd.execute(
        {
            "name": "deep_note.txt",
            "location": str(nested_loc),
            "create_parents": False,
        },
        gatekeeper=gk,
    )
    assert res_no_parents.success is False
    assert "Parent directory does not exist" in res_no_parents.error or "Target directory does not exist" in res_no_parents.error

    # When create_parents=True, parents are created safely
    res_with_parents = cmd.execute(
        {
            "name": "deep_note.txt",
            "location": str(desktop),
            "create_parents": True,
        },
        gatekeeper=gk,
    )
    assert res_with_parents.success is True
    assert (desktop / "deep_note.txt").exists()


def test_permission_error_gracefully_handled(create_env):
    cmd = create_env["cmd"]
    gk = create_env["gk"]
    desktop = create_env["desktop"]

    with patch("builtins.open", side_effect=PermissionError("Access denied by OS")):
        res = cmd.execute(
            {
                "name": "denied.txt",
                "location": str(desktop),
            },
            gatekeeper=gk,
        )
        assert res.success is False
        assert "permission denied" in res.error.lower()


def test_deterministic_routing_for_creation():
    router = DeterministicRouter()

    test_cases = [
        ("Create a folder named Projects on my Desktop.", "Projects", "Desktop", "folder", None),
        ("Create a file named notes.txt in my Documents folder.", "notes.txt", "Documents", "file", None),
        ("Create a new text file named todo.txt on my Desktop with content Buy milk.", "todo.txt", "Desktop", "file", "Buy milk"),
        ("Create directory Archives on Desktop", "Archives", "Desktop", "folder", None),
        ("Make a folder called Photos in Pictures", "Photos", "Pictures", "folder", None),
        ("Create file summary.txt in Documents with initial content Project is complete.", "summary.txt", "Documents", "file", "Project is complete"),
    ]

    for phrase, exp_name, exp_loc, exp_type, exp_content in test_cases:
        match = router.registry.find_command(phrase)
        assert match is not None, f"Phrase failed to match: '{phrase}'"
        cmd, args = match
        assert cmd.name in ("file_create", "folder_create")
        assert args.get("name") == exp_name
        if exp_loc:
            assert args.get("location").lower() == exp_loc.lower()
        assert args.get("item_type") in (exp_type, "directory" if exp_type == "folder" else "file")
        if exp_content:
            assert args.get("content") == exp_content


def test_command_registry_contains_creation_commands():
    reg = CommandRegistry()
    cmds = reg.list_commands()
    names = {c.name for c in cmds}

    assert "file_create" in names
    assert "folder_create" in names

    selectable = reg.get_model_selectable_commands()
    assert "file_create" in selectable
    assert "folder_create" in selectable


def test_model_selector_validation_for_creation(create_env):
    selectable = {"file_create": create_env["cmd"]}
    gk = create_env["gk"]
    desktop = create_env["desktop"]
    protected = create_env["protected_root"]

    # Valid file creation proposal
    valid_payload = {
        "tool": "file_create",
        "arguments": {
            "name": "new_project.txt",
            "location": str(desktop).replace("\\", "/"),
            "item_type": "file",
            "content": "hello world",
        },
    }
    proposal, err = parse_and_validate_tool_call(
        valid_payload, selectable, gk, user_input="create a new file new_project.txt on my desktop"
    )
    assert err is None
    assert proposal is not None
    assert proposal.tool_name == "file_create"
    assert proposal.arguments["name"] == "new_project.txt"

    # Reject prohibited extension
    bad_ext_payload = {
        "tool": "file_create",
        "arguments": {
            "name": "malicious.bat",
            "location": str(desktop).replace("\\", "/"),
        },
    }
    proposal, err = parse_and_validate_tool_call(
        bad_ext_payload, selectable, gk, user_input="create a file malicious.bat on desktop"
    )
    assert proposal is None
    assert "prohibited executable/script extension" in err.lower()

    # Reject unapproved location
    bad_loc_payload = {
        "tool": "file_create",
        "arguments": {
            "name": "test.txt",
            "location": str(protected).replace("\\", "/"),
        },
    }
    proposal, err = parse_and_validate_tool_call(
        bad_loc_payload, selectable, gk, user_input="create a file test.txt in system32"
    )
    assert proposal is None
    assert "restricted or outside safe write roots" in err.lower()


def test_agent_formatting_for_creation():
    # File created
    file_res = ToolResult(
        success=True,
        data={
            "created": "C:\\Users\\dmahi\\Desktop\\notes.txt",
            "name": "notes.txt",
            "type": "file",
            "size_bytes": 104,
        },
    )
    msg = LokiAgent._format_tool_result(file_res)
    assert "File 'notes.txt' created successfully at C:\\Users\\dmahi\\Desktop\\notes.txt (104 bytes)." == msg

    # Folder created
    folder_res = ToolResult(
        success=True,
        data={
            "created": "C:\\Users\\dmahi\\Desktop\\Projects",
            "name": "Projects",
            "type": "directory",
            "size_bytes": 0,
        },
    )
    msg_folder = LokiAgent._format_tool_result(folder_res)
    assert "Folder 'Projects' created successfully at C:\\Users\\dmahi\\Desktop\\Projects." == msg_folder
