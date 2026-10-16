"""Unit tests for built-in deterministic commands."""

from pathlib import Path
from unittest.mock import patch
import pytest

from loki.commands.builtin.app_launcher import (
    APPROVED_APPLICATIONS,
    AppLauncherCommand,
)
from loki.commands.builtin.file_delete import FileDeleteCommand
from loki.commands.builtin.file_search import FileSearchCommand
from loki.commands.builtin.system_info import SystemInfoCommand
from loki.core.exceptions import SafetyViolationError
from loki.core.types import SafetyLevel
from loki.safety.gatekeeper import SafetyGatekeeper


# =========================================================================
# SystemInfoCommand Tests
# =========================================================================

def test_system_info_metrics():
    cmd = SystemInfoCommand()
    assert cmd.name == "system_info"
    assert cmd.safety_level == SafetyLevel.SAFE

    # Test all metrics
    res_all = cmd.execute({"metric": "all"})
    assert res_all.success is True
    assert "ram" in res_all.data
    assert "cpu" in res_all.data
    assert "storage" in res_all.data
    assert res_all.execution_time_ms >= 0.0

    # Test individual metrics
    res_battery = cmd.execute({"metric": "battery"})
    assert res_battery.success is True
    assert "battery" in res_battery.data

    res_ram = cmd.execute({"metric": "ram"})
    assert res_ram.success is True
    assert res_ram.data["ram"]["total_gb"] > 0


def test_system_info_pattern_matching():
    cmd = SystemInfoCommand()
    assert cmd.can_handle("what is my battery percentage") is True
    assert cmd.can_handle("ram usage") is True
    assert cmd.can_handle("cpu usage") is True
    assert cmd.can_handle("disk space") is True
    assert cmd.can_handle("system info") is True
    assert cmd.can_handle("play a song") is False


# =========================================================================
# AppLauncherCommand Tests
# =========================================================================

def test_app_launcher_pattern_matching():
    cmd = AppLauncherCommand()
    assert cmd.can_handle("open chrome") is True
    assert cmd.can_handle("launch vs code") is True
    assert cmd.can_handle("start notepad") is True
    assert cmd.can_handle("calculate 5 + 5") is False


def test_app_launcher_whitelisted_execution(tmp_path: Path):
    cmd = AppLauncherCommand()
    mock_exe = tmp_path / "notepad.exe"
    mock_exe.write_text("binary", encoding="utf-8")

    with patch("loki.commands.builtin.app_launcher.AppLauncherCommand._find_executable", return_value=str(mock_exe)):
        with patch("os.startfile", return_value=None):
            res = cmd.execute({"app_name": "notepad"})
            assert res.success is True
            assert res.data["app_name"] == "notepad"
            assert res.data["status"] == "launched"


def test_app_launcher_unapproved_rejection():
    cmd = AppLauncherCommand()
    res = cmd.execute({"app_name": "malicious_script"})
    assert res.success is False
    assert "not in the approved application whitelist" in res.error


def test_app_launcher_injection_prevention():
    cmd = AppLauncherCommand()
    res = cmd.execute({"app_name": "calc & format C:"})
    assert res.success is False
    assert "Security violation" in res.error
    assert res.safety_level == SafetyLevel.DANGEROUS


# =========================================================================
# FileSearchCommand Tests
# =========================================================================

def test_file_search_within_safe_root(tmp_path: Path):
    safe_root = tmp_path / "Documents"
    safe_root.mkdir()
    (safe_root / "dbms_notes.pdf").write_text("DBMS notes content", encoding="utf-8")
    (safe_root / "os_notes.txt").write_text("OS notes content", encoding="utf-8")
    (safe_root / "unrelated.docx").write_text("other", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    cmd = FileSearchCommand()

    res = cmd.execute(
        {"query": "notes", "location": str(safe_root)},
        gatekeeper=gk,
    )
    assert res.success is True
    assert res.data["matches_count"] == 2
    filenames = [m["filename"] for m in res.data["matches"]]
    assert "dbms_notes.pdf" in filenames
    assert "os_notes.txt" in filenames


def test_file_search_disallowed_location(tmp_path: Path):
    protected = tmp_path / "System32"
    protected.mkdir()

    gk = SafetyGatekeeper(
        protected_paths=[protected],
        allowed_search_roots=[tmp_path / "SafeUserDir"],
    )
    cmd = FileSearchCommand()

    res = cmd.execute({"query": "kernel", "location": str(protected)}, gatekeeper=gk)
    assert res.success is False
    assert "restricted to safe user directories" in res.error


# =========================================================================
# FileDeleteCommand Tests (Safety Level DANGEROUS)
# =========================================================================

def test_file_delete_requires_confirmation(tmp_path: Path):
    safe_root = tmp_path / "UserDir"
    safe_root.mkdir()
    target_file = safe_root / "temp.log"
    target_file.write_text("to delete", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    cmd = FileDeleteCommand()

    # 1. Unconfirmed attempt blocked by Gatekeeper
    res = cmd.execute({"path": str(target_file)}, gatekeeper=gk)
    assert res.success is False
    assert "requires explicit confirmation" in res.error
    assert target_file.exists()

    # 2. Confirm action and execute
    req = gk.request_confirmation("file_delete", "Delete temp log", [str(target_file)])
    gk.confirm_action(req.action_id)

    res_confirmed = cmd.execute(
        {"path": str(target_file)},
        gatekeeper=gk,
        action_id=req.action_id,
    )
    assert res_confirmed.success is True
    assert not target_file.exists()


def test_file_delete_blocked_on_protected_path(tmp_path: Path):
    protected = tmp_path / "ProtectedCore"
    protected.mkdir()
    sys_file = protected / "important.sys"
    sys_file.write_text("critical", encoding="utf-8")

    gk = SafetyGatekeeper(
        protected_paths=[protected],
        allowed_search_roots=[tmp_path / "SafeDir"],
    )
    cmd = FileDeleteCommand()

    req = gk.request_confirmation("file_delete", "desc", [str(sys_file)])
    gk.confirm_action(req.action_id)

    res = cmd.execute(
        {"path": str(sys_file)},
        gatekeeper=gk,
        action_id=req.action_id,
    )
    assert res.success is False
    assert "protected or non-user location" in res.error
    assert sys_file.exists()
