"""Tests for the DeterministicRouter and safe command rejection."""

from pathlib import Path
from unittest.mock import patch
import pytest

from loki.commands.builtin.app_launcher import AppLauncherCommand
from loki.commands.builtin.system_info import SystemInfoCommand
from loki.commands.registry import CommandRegistry
from loki.commands.router import DeterministicRouter
from loki.safety.gatekeeper import SafetyGatekeeper


def test_router_matches_system_info():
    router = DeterministicRouter()
    res = router.route("what is my battery percentage")
    assert res.matched is True
    assert res.command_name == "system_info"
    assert res.result is not None
    assert res.result.success is True
    assert "battery" in res.result.data


def test_router_matches_app_launcher(tmp_path: Path):
    router = DeterministicRouter()
    mock_exe = tmp_path / "notepad.exe"
    mock_exe.write_text("binary", encoding="utf-8")

    with patch("loki.commands.builtin.app_launcher.AppLauncherCommand._find_executable", return_value=str(mock_exe)):
        with patch("os.startfile", return_value=None):
            res = router.route("launch notepad")
            assert res.matched is True
            assert res.command_name == "app_launcher"
            assert res.result is not None
            assert res.result.success is True
            assert res.result.data["status"] == "launched"


def test_router_safely_rejects_unknown_command():
    router = DeterministicRouter()
    # Arbitrary natural language must NOT execute any shell command
    res = router.route("write a python script to scrape twitter")
    assert res.matched is False
    assert res.command_name is None
    assert res.result is None
    assert "No matching deterministic command found" in res.reason


def test_router_safely_rejects_dangerous_shell_injection():
    router = DeterministicRouter()
    # Shell injection attempt must be safely rejected
    res = router.route("rmdir /s /q C:\\Windows")
    assert res.matched is False
    assert res.result is None


def test_router_gated_confirmation_flow(tmp_path: Path):
    safe_root = tmp_path / "UserStorage"
    safe_root.mkdir()
    target_file = safe_root / "unwanted.tmp"
    target_file.write_text("dummy", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    router = DeterministicRouter(gatekeeper=gk)

    # 1. First pass without confirmation token -> returns pending confirmation request
    res1 = router.route(f"delete {target_file}")
    assert res1.matched is True
    assert res1.command_name == "file_delete"
    assert res1.result is None
    assert res1.pending_confirmation is not None
    assert res1.pending_confirmation.confirmed is False
    assert target_file.exists()

    action_id = res1.pending_confirmation.action_id

    # 2. Confirm the action via gatekeeper
    assert gk.confirm_action(action_id) is True

    # 3. Second pass with confirmed action_id -> executes successfully
    res2 = router.route(f"delete {target_file}", action_id=action_id)
    assert res2.matched is True
    assert res2.result is not None
    assert res2.result.success is True
    assert not target_file.exists()
