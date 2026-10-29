"""Comprehensive tests for Windows known-folder and alias resolution, OneDrive redirection, and Gatekeeper integration."""

import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

from loki.commands.builtin.file_create import FileCreateCommand, FolderCreateCommand
from loki.commands.builtin.file_search import FileSearchCommand
from loki.commands.model_selector import parse_and_validate_tool_call
from loki.safety.gatekeeper import SafetyGatekeeper
from loki.utils.paths import (
    CANONICAL_FOLDER_ALIASES,
    clean_location_alias,
    get_all_known_folder_paths,
    get_default_search_roots,
    get_project_root,
    resolve_known_folder,
    resolve_safe_directory,
)


def test_clean_location_alias():
    assert clean_location_alias("on my Desktop") == "Desktop"
    assert clean_location_alias("in the Documents folder") == "Documents"
    assert clean_location_alias("at downloads") == "downloads"
    assert clean_location_alias("inside workspace") == "workspace"
    assert clean_location_alias("under Pictures directory") == "Pictures"
    assert clean_location_alias("of the Music dir.") == "Music"
    assert clean_location_alias("Desktop") == "Desktop"
    assert clean_location_alias(None) is None
    assert clean_location_alias("") is None


def test_canonical_folder_aliases_mapping():
    assert CANONICAL_FOLDER_ALIASES["desktop"] == "desktop"
    assert CANONICAL_FOLDER_ALIASES["documents"] == "documents"
    assert CANONICAL_FOLDER_ALIASES["docs"] == "documents"
    assert CANONICAL_FOLDER_ALIASES["doc"] == "documents"
    assert CANONICAL_FOLDER_ALIASES["downloads"] == "downloads"
    assert CANONICAL_FOLDER_ALIASES["download"] == "downloads"
    assert CANONICAL_FOLDER_ALIASES["pictures"] == "pictures"
    assert CANONICAL_FOLDER_ALIASES["pics"] == "pictures"
    assert CANONICAL_FOLDER_ALIASES["images"] == "pictures"
    assert CANONICAL_FOLDER_ALIASES["photos"] == "pictures"
    assert CANONICAL_FOLDER_ALIASES["music"] == "music"
    assert CANONICAL_FOLDER_ALIASES["audio"] == "music"
    assert CANONICAL_FOLDER_ALIASES["videos"] == "videos"
    assert CANONICAL_FOLDER_ALIASES["video"] == "videos"
    assert CANONICAL_FOLDER_ALIASES["movies"] == "videos"
    assert CANONICAL_FOLDER_ALIASES["project"] == "project"
    assert CANONICAL_FOLDER_ALIASES["loki project"] == "project"
    assert CANONICAL_FOLDER_ALIASES["loki"] == "project"
    assert CANONICAL_FOLDER_ALIASES["workspace"] == "project"


def test_project_root_resolution():
    proj = resolve_known_folder("project")
    assert proj is not None
    assert proj.exists()
    assert (proj / "loki").exists()
    assert resolve_known_folder("loki project") == proj
    assert resolve_known_folder("workspace") == proj


def test_onedrive_redirection_resolution(tmp_path: Path):
    fake_onedrive = tmp_path / "FakeOneDrive"
    fake_onedrive.mkdir()
    fake_desktop = fake_onedrive / "Desktop"
    fake_desktop.mkdir()
    fake_documents = fake_onedrive / "Documents"
    fake_documents.mkdir()

    # Disable win32 ctypes and winreg so resolution hits onedrive check
    with patch.dict(os.environ, {"OneDrive": str(fake_onedrive)}), \
         patch("loki.utils.paths._get_known_folder_via_win32", return_value=None), \
         patch("loki.utils.paths._get_known_folder_via_registry", return_value=None):

        res_desktop = resolve_known_folder("Desktop")
        assert res_desktop == fake_desktop.resolve()

        res_docs = resolve_known_folder("Documents")
        assert res_docs == fake_documents.resolve()


def test_registry_redirection_resolution(tmp_path: Path):
    reg_desktop = tmp_path / "CustomRegistryDesktop"
    reg_desktop.mkdir()

    with patch("loki.utils.paths._get_known_folder_via_win32", return_value=None), \
         patch("loki.utils.paths._get_known_folder_via_registry", return_value=reg_desktop):

        res = resolve_known_folder("Desktop")
        assert res == reg_desktop.resolve()


def test_win32_known_folder_resolution(tmp_path: Path):
    win32_desktop = tmp_path / "Win32Desktop"
    win32_desktop.mkdir()

    with patch("loki.utils.paths._get_known_folder_via_win32", return_value=win32_desktop):
        res = resolve_known_folder("Desktop")
        assert res == win32_desktop.resolve()


def test_standard_home_fallback(tmp_path: Path):
    fake_home = tmp_path / "FakeHome"
    fake_home.mkdir()
    fake_downloads = fake_home / "Downloads"
    fake_downloads.mkdir()

    with patch("loki.utils.paths._get_known_folder_via_win32", return_value=None), \
         patch("loki.utils.paths._get_known_folder_via_registry", return_value=None), \
         patch("loki.utils.paths._get_known_folder_via_onedrive", return_value=None), \
         patch("pathlib.Path.home", return_value=fake_home):

        res = resolve_known_folder("downloads")
        assert res == fake_downloads.resolve()


def test_resolve_safe_directory_test_isolation(tmp_path: Path):
    safe_root = tmp_path / "IsolatedSafeRoot"
    safe_root.mkdir()
    desktop = safe_root / "Desktop"
    desktop.mkdir()

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])

    # Search resolution
    resolved_read, err_read = resolve_safe_directory("Desktop", gatekeeper=gk, for_write=False)
    assert err_read is None
    assert resolved_read == desktop.resolve()

    # Write resolution
    resolved_write, err_write = resolve_safe_directory("on my Desktop", gatekeeper=gk, for_write=True)
    assert err_write is None
    assert resolved_write == desktop.resolve()


def test_resolve_safe_directory_missing_folder(tmp_path: Path):
    safe_root = tmp_path / "SafeRoot"
    safe_root.mkdir()
    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])

    # Non-existent alias
    with patch("loki.utils.paths.resolve_known_folder", return_value=None), \
         patch("pathlib.Path.home", return_value=safe_root):
        resolved, err = resolve_safe_directory("Pictures", gatekeeper=gk, for_write=False)
        assert resolved is None
        assert "Specified search directory does not exist: 'Pictures'." in err

        resolved_w, err_w = resolve_safe_directory("Pictures", gatekeeper=gk, for_write=True)
        assert resolved_w is None
        assert "Target directory does not exist: 'Pictures'." in err_w


def test_resolve_safe_directory_not_a_directory(tmp_path: Path):
    safe_root = tmp_path / "SafeRoot"
    safe_root.mkdir()
    dummy_file = safe_root / "not_a_dir.txt"
    dummy_file.write_text("hello", encoding="utf-8")
    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])

    resolved_read, err_read = resolve_safe_directory(str(dummy_file), gatekeeper=gk, for_write=False)
    assert resolved_read is None
    assert "Specified search location is not a directory:" in err_read

    resolved_write, err_write = resolve_safe_directory(str(dummy_file), gatekeeper=gk, for_write=True)
    assert resolved_write is None
    assert "Target location is not a directory:" in err_write


def test_resolve_safe_directory_outside_gatekeeper_roots(tmp_path: Path):
    safe_root = tmp_path / "SafeRoot"
    safe_root.mkdir()
    unapproved = tmp_path / "UnapprovedDir"
    unapproved.mkdir()

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])

    resolved_read, err_read = resolve_safe_directory(str(unapproved), gatekeeper=gk, for_write=False)
    assert resolved_read is None
    assert "Searches are restricted to safe user directories." in err_read

    resolved_write, err_write = resolve_safe_directory(str(unapproved), gatekeeper=gk, for_write=True)
    assert resolved_write is None
    assert "outside approved write roots" in err_write


def test_gatekeeper_default_search_roots_inclusion():
    roots = SafetyGatekeeper._get_default_search_roots()
    assert len(roots) > 0
    # User home should be included
    assert Path.home().resolve() in roots
    # Project root should be included
    assert get_project_root().resolve() in roots
    # All existing known folders should be included
    for kf in get_all_known_folder_paths():
        assert kf in roots


def test_file_search_command_with_desktop_alias(tmp_path: Path):
    safe_root = tmp_path / "SafeUserRoot"
    safe_root.mkdir()
    desktop = safe_root / "Desktop"
    desktop.mkdir()
    (desktop / "my_report.pdf").write_text("Confidential Report", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    cmd = FileSearchCommand()

    res = cmd.execute({"query": "my_report.pdf", "location": "Desktop"}, gatekeeper=gk)
    assert res.success is True
    assert res.data["matches_count"] == 1
    assert res.data["matches"][0]["name"] == "my_report.pdf"


def test_file_create_command_with_desktop_alias(tmp_path: Path):
    safe_root = tmp_path / "SafeUserRoot"
    safe_root.mkdir()
    desktop = safe_root / "Desktop"
    desktop.mkdir()

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    cmd = FileCreateCommand()

    res = cmd.execute(
        {
            "name": "from_alias.txt",
            "location": "on my Desktop",
            "content": "Created via alias resolution",
            "item_type": "file",
        },
        gatekeeper=gk,
    )
    assert res.success is True
    target = desktop / "from_alias.txt"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "Created via alias resolution"


def test_folder_create_command_with_documents_alias(tmp_path: Path):
    safe_root = tmp_path / "SafeUserRoot"
    safe_root.mkdir()
    documents = safe_root / "Documents"
    documents.mkdir()

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    cmd = FolderCreateCommand()

    res = cmd.execute(
        {
            "name": "MyNewFolder",
            "location": "Documents folder",
            "item_type": "folder",
        },
        gatekeeper=gk,
    )
    assert res.success is True
    target = documents / "MyNewFolder"
    assert target.exists()
    assert target.is_dir()


def test_model_selector_validation_with_aliases(tmp_path: Path):
    safe_root = tmp_path / "SafeUserRoot"
    safe_root.mkdir()
    desktop = safe_root / "Desktop"
    desktop.mkdir()

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    selectable = {
        "file_search": FileSearchCommand(),
        "file_create": FileCreateCommand(),
    }

    # Propose file search on Desktop
    payload_search = {
        "tool": "file_search",
        "arguments": {"query": "test.txt", "location": "Desktop"},
    }
    proposal_s, err_s = parse_and_validate_tool_call(
        payload_search, selectable, gk, user_input="search for test.txt on Desktop"
    )
    assert err_s is None
    assert proposal_s is not None
    assert proposal_s.tool_name == "file_search"

    # Propose file create on Desktop
    payload_create = {
        "tool": "file_create",
        "arguments": {"name": "created_via_model.txt", "location": "Desktop"},
    }
    proposal_c, err_c = parse_and_validate_tool_call(
        payload_create, selectable, gk, user_input="create file created_via_model.txt on Desktop"
    )
    assert err_c is None
    assert proposal_c is not None
    assert proposal_c.tool_name == "file_create"
