"""Tests for LOKI Phase 9.1 safe deterministic file and folder search."""

from datetime import datetime
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from loki.agent import LokiAgent
from loki.commands.base import BaseCommand
from loki.commands.builtin.file_search import FileSearchCommand
from loki.commands.model_selector import parse_and_validate_tool_call
from loki.commands.registry import CommandRegistry
from loki.commands.router import DeterministicRouter
from loki.core.types import SafetyLevel, ToolResult
from loki.safety.gatekeeper import SafetyGatekeeper


@pytest.fixture
def search_env(tmp_path: Path):
    """Create a populated, isolated filesystem fixture for search tests."""
    safe_root = tmp_path / "SafeUserRoot"
    safe_root.mkdir()

    # Documents
    docs = safe_root / "Documents"
    docs.mkdir()
    (docs / "annual_report.pdf").write_text("Annual Report 2026", encoding="utf-8")
    (docs / "budget.xlsx").write_text("Budget Spreadsheet", encoding="utf-8")
    (docs / "notes.txt").write_text("Meeting notes", encoding="utf-8")

    # Subdirectory in documents
    projects_dir = docs / "Projects"
    projects_dir.mkdir()
    (projects_dir / "app.py").write_text("print('hello')", encoding="utf-8")
    (projects_dir / "test_app.py").write_text("assert True", encoding="utf-8")
    (projects_dir / "README.md").write_text("# Project Readme", encoding="utf-8")

    # Downloads
    downloads = safe_root / "Downloads"
    downloads.mkdir()
    (downloads / "setup.exe").write_text("installer", encoding="utf-8")
    (downloads / "archive.zip").write_text("zip data", encoding="utf-8")

    # Desktop
    desktop = safe_root / "Desktop"
    desktop.mkdir()
    (desktop / "report.pdf").write_text("Desktop Report", encoding="utf-8")
    (desktop / "todo.txt").write_text("TODO list", encoding="utf-8")

    # An isolated unapproved directory outside safe roots
    unapproved = tmp_path / "ProtectedSystem"
    unapproved.mkdir()
    (unapproved / "secret_kernel.sys").write_text("secret", encoding="utf-8")

    gk = SafetyGatekeeper(
        protected_paths=[unapproved],
        allowed_search_roots=[safe_root],
    )
    cmd = FileSearchCommand()

    return {
        "safe_root": safe_root,
        "docs": docs,
        "projects_dir": projects_dir,
        "downloads": downloads,
        "desktop": desktop,
        "unapproved": unapproved,
        "gk": gk,
        "cmd": cmd,
    }


def test_exact_filename_search(search_env):
    cmd = search_env["cmd"]
    gk = search_env["gk"]
    desktop = search_env["desktop"]

    res = cmd.execute(
        {"query": "report.pdf", "location": str(desktop)},
        gatekeeper=gk,
    )
    assert res.success is True
    assert res.data["matches_count"] == 1
    m = res.data["matches"][0]
    assert m["filename"] == "report.pdf"
    assert m["name"] == "report.pdf"
    assert m["type"] == "file"
    assert m["extension"] == ".pdf"
    assert m["size_bytes"] > 0
    assert Path(m["path"]).resolve() == (desktop / "report.pdf").resolve()


def test_extension_filtering(search_env):
    cmd = search_env["cmd"]
    gk = search_env["gk"]
    docs = search_env["docs"]

    # Filter python files via extension parameter
    res = cmd.execute(
        {"query": "*", "location": str(docs), "extension": ".py"},
        gatekeeper=gk,
    )
    assert res.success is True
    assert res.data["matches_count"] == 2
    filenames = {m["filename"] for m in res.data["matches"]}
    assert filenames == {"app.py", "test_app.py"}

    # Filter pdf files via wildcard query
    res_pdf = cmd.execute(
        {"query": "*.pdf", "location": str(docs)},
        gatekeeper=gk,
    )
    assert res_pdf.success is True
    assert res_pdf.data["matches_count"] == 1
    assert res_pdf.data["matches"][0]["filename"] == "annual_report.pdf"


def test_directory_listing_list_only(search_env):
    cmd = search_env["cmd"]
    gk = search_env["gk"]
    docs = search_env["docs"]

    # In list_only mode (max_depth=1), only immediate children of docs should be listed
    res = cmd.execute(
        {"query": "*", "location": str(docs), "list_only": True},
        gatekeeper=gk,
    )
    assert res.success is True
    names = {m["name"] for m in res.data["matches"]}
    assert "annual_report.pdf" in names
    assert "budget.xlsx" in names
    assert "notes.txt" in names
    assert "Projects" in names  # Directory is included

    # Deep children (app.py) should NOT be included in immediate listing
    assert "app.py" not in names


def test_locate_folders_vs_files(search_env):
    cmd = search_env["cmd"]
    gk = search_env["gk"]
    safe_root = search_env["safe_root"]

    # Locate folders only
    res_folders = cmd.execute(
        {"query": "Projects", "location": str(safe_root), "target_type": "directory"},
        gatekeeper=gk,
    )
    assert res_folders.success is True
    assert res_folders.data["matches_count"] >= 1
    for m in res_folders.data["matches"]:
        assert m["type"] == "directory"
        assert "Projects" in m["name"]

    # Locate files only
    res_files = cmd.execute(
        {"query": "Projects", "location": str(safe_root), "target_type": "file"},
        gatekeeper=gk,
    )
    assert res_files.success is True
    # "Projects" folder should NOT be in file-only search
    for m in res_files.data["matches"]:
        assert m["type"] == "file"


def test_bounded_result_limits(tmp_path: Path):
    safe_dir = tmp_path / "LotsOfFiles"
    safe_dir.mkdir()
    for i in range(30):
        (safe_dir / f"doc_{i:02d}.txt").write_text(f"content {i}", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_dir])
    cmd = FileSearchCommand()

    res = cmd.execute(
        {"query": "doc", "location": str(safe_dir), "max_results": 7},
        gatekeeper=gk,
    )
    assert res.success is True
    assert res.data["matches_count"] == 7
    assert len(res.data["matches"]) == 7


def test_missing_and_disallowed_paths(search_env):
    cmd = search_env["cmd"]
    gk = search_env["gk"]
    safe_root = search_env["safe_root"]
    unapproved = search_env["unapproved"]

    # Missing directory
    missing_dir = safe_root / "NonExistentFolder"
    res_missing = cmd.execute({"query": "test", "location": str(missing_dir)}, gatekeeper=gk)
    assert res_missing.success is False
    assert "does not exist" in res_missing.error

    # Disallowed protected location
    res_disallowed = cmd.execute({"query": "test", "location": str(unapproved)}, gatekeeper=gk)
    assert res_disallowed.success is False
    assert "restricted to safe user directories" in res_disallowed.error


def test_path_traversal_prevention(search_env):
    cmd = search_env["cmd"]
    gk = search_env["gk"]
    safe_root = search_env["safe_root"]
    unapproved = search_env["unapproved"]

    # Attempt directory traversal with .. escaping safe roots
    escaped_target = str(safe_root / ".." / unapproved.name)
    res = cmd.execute({"query": "*", "location": escaped_target}, gatekeeper=gk)
    assert res.success is False
    assert "restricted to safe user directories" in res.error

    # Control characters in query
    valid, err = cmd.validate_args({"query": "test\x00escape"})
    assert valid is False
    assert "control characters" in err

    # Control characters in location
    valid, err = cmd.validate_args({"query": "test", "location": "C:\\Safe\nFolder"})
    assert valid is False
    assert "control characters" in err

    # Control characters in extension
    valid, err = cmd.validate_args({"query": "test", "extension": ".py\x00evil"})
    assert valid is False
    assert "invalid characters" in err


def test_symlink_escaping_safe_roots(search_env):
    cmd = search_env["cmd"]
    gk = search_env["gk"]
    docs = search_env["docs"]
    unapproved = search_env["unapproved"]

    # 1. Detection of symlinks and NTFS junctions
    fake_symlink = MagicMock(spec=Path)
    fake_symlink.is_symlink.return_value = True
    assert cmd._is_symlink_or_junction(fake_symlink) is True

    fake_junction = MagicMock(spec=Path)
    fake_junction.is_symlink.return_value = False
    fake_junction.is_junction.return_value = True
    assert cmd._is_symlink_or_junction(fake_junction) is True

    # 2. File symlink pointing to an unapproved target is skipped
    symlink_stub = docs / "evil_symlink.txt"
    symlink_stub.write_text("stub", encoding="utf-8")

    real_is_symlink = Path.is_symlink
    real_resolve = Path.resolve

    def mock_is_symlink(self):
        if self.name == "evil_symlink.txt":
            return True
        return real_is_symlink(self)

    def mock_resolve(self):
        if self.name == "evil_symlink.txt":
            return unapproved / "secret_kernel.sys"
        return real_resolve(self)

    with patch.object(Path, "is_symlink", mock_is_symlink), \
         patch.object(Path, "resolve", mock_resolve):
        res = cmd.execute({"query": "evil_symlink", "location": str(docs)}, gatekeeper=gk)
        assert res.success is True
        assert res.data["matches_count"] == 0


def test_permission_error_resilience(search_env):
    cmd = search_env["cmd"]
    gk = search_env["gk"]
    docs = search_env["docs"]

    with patch("os.walk") as mock_walk:
        # Simulate PermissionError on walking
        mock_walk.side_effect = PermissionError("Access denied")
        res = cmd.execute({"query": "test", "location": str(docs)}, gatekeeper=gk)
        # Should gracefully return success with 0 matches instead of unhandled crash
        assert res.success is True
        assert res.data["matches_count"] == 0


def test_deterministic_routing_phrases():
    router = DeterministicRouter()

    test_cases = [
        ("Find report.pdf on my Desktop.", "file_search", "report.pdf", "desktop", "both"),
        ("Search for Python files in my LOKI project.", "file_search", "*.py", "loki project", "file"),
        ("Locate folders named downloads.", "file_search", "downloads", None, "directory"),
        ("List files in my Documents folder.", "file_search", "*", "documents", "file"),
        ("Find files with .pdf extension in Documents.", "file_search", "*.pdf", "documents", "file"),
        ("search for notes", "file_search", "notes", None, "both"),
        ("find notes in Downloads", "file_search", "notes", "downloads", "both"),
    ]

    for phrase, expected_cmd, expected_query, expected_loc, expected_type in test_cases:
        match = router.registry.find_command(phrase)
        assert match is not None, f"Phrase '{phrase}' failed to match deterministic router"
        cmd, args = match
        assert cmd.name == expected_cmd
        assert args.get("query") == expected_query
        if expected_loc:
            assert args.get("location").lower() == expected_loc.lower()
        if expected_type:
            assert args.get("target_type") == expected_type


def test_model_selector_validation(search_env):
    selectable = {"file_search": search_env["cmd"]}
    gk = search_env["gk"]
    safe_root = search_env["safe_root"]

    # Valid model proposal
    payload = {
        "tool": "file_search",
        "arguments": {
            "query": "report.pdf",
            "location": str(safe_root).replace("\\", "/"),
            "extension": ".pdf",
            "target_type": "file",
        },
    }
    proposal, err = parse_and_validate_tool_call(
        payload, selectable, gk, user_input="find report.pdf in safe directory"
    )
    assert err is None
    assert proposal is not None
    assert proposal.tool_name == "file_search"
    assert proposal.arguments["query"] == "report.pdf"

    # Rejection of unapproved location
    bad_payload = {
        "tool": "file_search",
        "arguments": {
            "query": "secret",
            "location": str(search_env["unapproved"]).replace("\\", "/"),
        },
    }
    proposal, err = parse_and_validate_tool_call(
        bad_payload, selectable, gk, user_input="search for secret in protected directory"
    )
    assert proposal is None
    assert "restricted or outside safe read roots" in err


def test_agent_search_formatting():
    # Empty result formatting
    empty_res = ToolResult(
        success=True,
        data={"query": "notes", "matches_count": 0, "matches": []},
    )
    msg = LokiAgent._format_tool_result(empty_res)
    assert msg == "No items found for 'notes'."

    # Multiple item result formatting
    items_res = ToolResult(
        success=True,
        data={
            "query": "*",
            "matches_count": 2,
            "matches": [
                {
                    "name": "Projects",
                    "path": "C:\\Safe\\Documents\\Projects",
                    "type": "directory",
                    "size_mb": 0.0,
                    "modified": "2026-10-24 12:00:00",
                },
                {
                    "name": "annual_report.pdf",
                    "path": "C:\\Safe\\Documents\\annual_report.pdf",
                    "type": "file",
                    "size_mb": 1.25,
                    "modified": "2026-10-24 14:00:00",
                },
            ],
        },
    )
    formatted = LokiAgent._format_tool_result(items_res)
    assert "Found 2 items:" in formatted
    assert "- [Folder] Projects (C:\\Safe\\Documents\\Projects)" in formatted
    assert "- [File] annual_report.pdf (1.25 MB, modified 2026-10-24 14:00:00)" in formatted
