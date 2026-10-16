"""Adversarial regression test suite for Phase 3.1 Security Remediation."""

from pathlib import Path
import sys
import time
from unittest.mock import patch
import pytest

from loki.commands.builtin.app_launcher import AppLauncherCommand
from loki.commands.builtin.file_delete import FileDeleteCommand
from loki.commands.builtin.file_search import FileSearchCommand
from loki.commands.router import DeterministicRouter
from loki.core.config import SafetyConfig
from loki.core.exceptions import SafetyViolationError
from loki.core.types import SafetyLevel
from loki.safety.gatekeeper import SafetyGatekeeper


# =========================================================================
# 1 & 2. Confirmation Token Binding: Command Mismatch & Target Mismatch
# =========================================================================

def test_token_rejected_on_different_command(tmp_path: Path):
    """SEC-01: A confirmed token for file_delete must not authorize another command."""
    safe_root = tmp_path / "SafeUserDir"
    safe_root.mkdir()
    target_file = safe_root / "test.txt"
    target_file.write_text("content", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    req = gk.request_confirmation(
        tool_name="file_delete",
        description="Delete file",
        targets=[str(target_file)],
    )
    assert gk.confirm_action(req.action_id) is True

    # Attempt to use the confirmed token with a different tool_name
    with pytest.raises(SafetyViolationError) as exc_info:
        gk.verify_execution_permitted(
            tool_name="format_disk",
            safety_level=SafetyLevel.DANGEROUS,
            action_id=req.action_id,
            targets=[str(target_file)],
        )
    assert "was authorized for 'file_delete', not 'format_disk'" in str(exc_info.value)


def test_token_rejected_on_different_target(tmp_path: Path):
    """SEC-01: A confirmed token for target A must not authorize target B."""
    safe_root = tmp_path / "SafeUserDir"
    safe_root.mkdir()
    file_a = safe_root / "harmless.log"
    file_b = safe_root / "confidential.pdf"
    file_a.write_text("log data", encoding="utf-8")
    file_b.write_text("secret data", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    req = gk.request_confirmation(
        tool_name="file_delete",
        description="Delete harmless log",
        targets=[str(file_a)],
    )
    assert gk.confirm_action(req.action_id) is True

    # Attempt to apply confirmed token to target B
    with pytest.raises(SafetyViolationError) as exc_info:
        gk.verify_execution_permitted(
            tool_name="file_delete",
            safety_level=SafetyLevel.DANGEROUS,
            action_id=req.action_id,
            targets=[str(file_b)],
        )
    assert "Target mismatch" in str(exc_info.value)
    assert file_b.exists()


# =========================================================================
# 3. Token Lifecycle: Expired, Canceled, Missing, and Replayed Tokens
# =========================================================================

def test_token_rejected_when_expired(tmp_path: Path):
    """SEC-05: Expired tokens must be pruned and rejected."""
    safe_root = tmp_path / "SafeUserDir"
    safe_root.mkdir()
    target = safe_root / "temp.dat"
    target.write_text("data", encoding="utf-8")

    # Configure short TTL (100 ms)
    gk = SafetyGatekeeper(
        config=SafetyConfig(confirmation_ttl_seconds=1),
        allowed_search_roots=[safe_root],
    )
    req = gk.request_confirmation("file_delete", "desc", [str(target)])
    req.ttl_seconds = 0.05  # 50 ms TTL for test
    time.sleep(0.06)

    # Attempt confirmation after expiration
    assert gk.confirm_action(req.action_id) is False

    # Execution verification must fail
    with pytest.raises(SafetyViolationError) as exc_info:
        gk.verify_execution_permitted("file_delete", SafetyLevel.DANGEROUS, req.action_id, [str(target)])
    assert "invalid, expired, or already consumed" in str(exc_info.value)


def test_token_rejected_when_canceled(tmp_path: Path):
    safe_root = tmp_path / "SafeUserDir"
    safe_root.mkdir()
    target = safe_root / "temp.dat"
    target.write_text("data", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    req = gk.request_confirmation("file_delete", "desc", [str(target)])
    assert gk.cancel_action(req.action_id) is True

    with pytest.raises(SafetyViolationError) as exc_info:
        gk.verify_execution_permitted("file_delete", SafetyLevel.DANGEROUS, req.action_id, [str(target)])
    assert "invalid, expired, or already consumed" in str(exc_info.value)


def test_token_rejected_when_missing():
    gk = SafetyGatekeeper()
    with pytest.raises(SafetyViolationError) as exc_info:
        gk.verify_execution_permitted("file_delete", SafetyLevel.DANGEROUS, "act_nonexistent", [])
    assert "invalid, expired, or already consumed" in str(exc_info.value)


def test_token_replay_rejected(tmp_path: Path):
    safe_root = tmp_path / "SafeUserDir"
    safe_root.mkdir()
    target = safe_root / "file1.txt"
    target.write_text("data", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    req = gk.request_confirmation("file_delete", "desc", [str(target)])
    assert gk.confirm_action(req.action_id) is True

    # First consumption succeeds
    assert gk.verify_execution_permitted("file_delete", SafetyLevel.DANGEROUS, req.action_id, [str(target)]) is True

    # Replay attempt must fail
    with pytest.raises(SafetyViolationError) as exc_info:
        gk.verify_execution_permitted("file_delete", SafetyLevel.DANGEROUS, req.action_id, [str(target)])
    assert "invalid, expired, or already consumed" in str(exc_info.value)


# =========================================================================
# 4. Fail-Closed: Dangerous Command Execution Without Gatekeeper
# =========================================================================

def test_dangerous_command_fails_closed_without_gatekeeper(tmp_path: Path):
    """SEC-03: Dangerous commands must refuse execution if gatekeeper is None."""
    safe_file = tmp_path / "vital.txt"
    safe_file.write_text("vital data", encoding="utf-8")

    cmd = FileDeleteCommand()
    # Execute without passing gatekeeper
    res = cmd.execute({"path": str(safe_file)}, gatekeeper=None)

    assert res.success is False
    assert "requires an active SafetyGatekeeper" in res.error
    assert res.safety_level == SafetyLevel.DANGEROUS
    assert "safety_violation" in res.metadata
    assert safe_file.exists()


# =========================================================================
# 5 & 6. Search Roots Whitelist Containment & Case Insensitivity
# =========================================================================

def test_search_rejects_paths_outside_allowed_roots(tmp_path: Path):
    """SEC-02 & SEC-06: File search must reject paths outside allowed_search_roots."""
    allowed_root = tmp_path / "AllowedDocs"
    allowed_root.mkdir()
    outside_dir = tmp_path / "ExternalSecrets"
    outside_dir.mkdir()
    (outside_dir / "secret.key").write_text("key data", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[allowed_root])
    cmd = FileSearchCommand()

    res = cmd.execute({"query": "secret", "location": str(outside_dir)}, gatekeeper=gk)
    assert res.success is False
    assert "restricted to safe user directories" in res.error


def test_gatekeeper_case_insensitivity_and_traversal(tmp_path: Path):
    """Verify Windows case insensitivity and .. traversal normalization."""
    safe_root = tmp_path / "SafeDir"
    safe_root.mkdir()
    target = safe_root / "sub" / "file.txt"
    (safe_root / "sub").mkdir()
    target.write_text("test", encoding="utf-8")

    gk = SafetyGatekeeper(
        protected_paths=[tmp_path / "PROTECTED_CORE"],
        allowed_search_roots=[safe_root],
    )

    # Path traversal attempt outside safe root
    traversal_path = str(safe_root) + r"\sub\..\..\outside.txt"
    assert gk.is_path_safe_for_write(traversal_path) is False

    # Case variance check on safe path
    cased_path = str(target).upper() if sys.platform == "win32" else str(target)
    assert gk.is_path_safe_for_read(cased_path) is True
    assert gk.is_path_safe_for_write(cased_path) is True

    # Case variance check on protected path
    cased_protected = (str(tmp_path / "protected_core") + r"\sys.dat").lower()
    assert gk.is_path_safe_for_read(cased_protected) is False
    assert gk.is_path_safe_for_write(cased_protected) is False


# =========================================================================
# 7. Symlink Containment
# =========================================================================

def test_search_does_not_traverse_directory_symlinks(tmp_path: Path):
    """SEC-02: os.walk must prune directory symlinks/junctions pointing outside allowed roots."""
    allowed_root = tmp_path / "Allowed"
    allowed_root.mkdir()
    outside_dir = tmp_path / "SecretOutside"
    outside_dir.mkdir()
    (outside_dir / "hidden.txt").write_text("secret", encoding="utf-8")

    # Create directory junction on Windows or symlink on Unix
    junction_dir = allowed_root / "link_to_outside"
    if sys.platform == "win32":
        import _winapi
        _winapi.CreateJunction(str(outside_dir), str(junction_dir))
    else:
        try:
            junction_dir.symlink_to(outside_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("Symlink creation not permitted in this environment")

    gk = SafetyGatekeeper(allowed_search_roots=[allowed_root])
    cmd = FileSearchCommand()

    res = cmd.execute({"query": "hidden", "location": str(allowed_root)}, gatekeeper=gk)
    assert res.success is True
    # The hidden.txt inside the junction/symlinked outside directory must NOT be found
    assert res.data["matches_count"] == 0


# =========================================================================
# 8. App Launcher: Refuse Unresolved Relative Executables
# =========================================================================

def test_app_launcher_refuses_unresolved_relative_binary():
    """SEC-04: Must refuse execution if executable cannot be resolved to an absolute path."""
    cmd = AppLauncherCommand()

    # Simulate _find_executable failing to locate an absolute path
    with patch("loki.commands.builtin.app_launcher.AppLauncherCommand._find_executable", return_value=None):
        res = cmd.execute({"app_name": "calculator"})
        assert res.success is False
        assert "Could not locate a verified absolute executable" in res.error


def test_app_launcher_refuses_relative_path_even_if_returned():
    """SEC-04: A relative path returned by any lookup must be rejected."""
    cmd = AppLauncherCommand()

    with patch("loki.commands.builtin.app_launcher.AppLauncherCommand._find_executable", return_value="relative_calc.exe"):
        res = cmd.execute({"app_name": "calc"})
        assert res.success is False
        assert "Could not locate a verified absolute executable" in res.error


# =========================================================================
# 9 & 10. Injection & File Deletion Blocked on Authorization Failure
# =========================================================================

def test_app_launcher_rejects_null_bytes_and_quotes():
    cmd = AppLauncherCommand()
    assert cmd.execute({"app_name": "calc\x00malicious"}).success is False
    assert cmd.execute({"app_name": 'calc" & dir'}).success is False
    assert cmd.execute({"app_name": "calc; ls"}).success is False


def test_file_delete_blocked_when_authorization_fails(tmp_path: Path):
    safe_root = tmp_path / "SafeDir"
    safe_root.mkdir()
    target = safe_root / "precious.txt"
    target.write_text("do not delete", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    router = DeterministicRouter(gatekeeper=gk)

    # 1. Unconfirmed attempt: router requests confirmation, target remains untouched
    res1 = router.route(f"delete {target}")
    assert res1.matched is True
    assert res1.result is None
    assert target.exists()

    action_id = res1.pending_confirmation.action_id

    # 2. Caller tries to confirm with wrong expected tool name: confirmation fails
    assert gk.confirm_action(action_id, expected_tool="other_tool") is False

    # 3. Execution attempt with unconfirmed action_id: fails, target remains untouched
    res2 = router.route(f"delete {target}", action_id=action_id)
    assert res2.result.success is False
    assert target.exists()
