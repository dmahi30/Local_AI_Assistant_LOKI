"""Tests for SafetyGatekeeper path validation and confirmation gates."""

from pathlib import Path
import pytest

from loki.core.config import SafetyConfig
from loki.core.exceptions import SafetyViolationError
from loki.core.types import SafetyLevel
from loki.safety.gatekeeper import DEFAULT_PROTECTED_DIRECTORIES, SafetyGatekeeper


def test_gatekeeper_read_path_safety(tmp_path: Path):
    user_file = tmp_path / "notes.txt"
    user_file.write_text("my notes", encoding="utf-8")

    gk = SafetyGatekeeper(
        protected_paths=[tmp_path / "SystemSecret"],
        allowed_search_roots=[tmp_path],
    )

    assert gk.is_path_safe_for_read(user_file) is True
    assert gk.is_path_safe_for_read(tmp_path / "SystemSecret" / "secret.sys") is False


def test_gatekeeper_write_path_safety(tmp_path: Path):
    safe_root = tmp_path / "UserDocs"
    safe_root.mkdir()
    target_file = safe_root / "document.pdf"

    protected_root = tmp_path / "WindowsCore"
    protected_root.mkdir()
    system_file = protected_root / "kernel.sys"

    outside_file = tmp_path / "random.sys"

    gk = SafetyGatekeeper(
        protected_paths=[protected_root],
        allowed_search_roots=[safe_root],
    )

    assert gk.is_path_safe_for_write(target_file) is True
    assert gk.is_path_safe_for_write(system_file) is False
    assert gk.is_path_safe_for_write(outside_file) is False


def test_gatekeeper_confirmation_flow(tmp_path: Path):
    safe_root = tmp_path / "SafeDir"
    safe_root.mkdir()
    target = safe_root / "file_to_delete.txt"
    target.write_text("data", encoding="utf-8")

    gk = SafetyGatekeeper(
        config=SafetyConfig(require_confirmation_for_destructive=True),
        allowed_search_roots=[safe_root],
    )

    # 1. Unconfirmed execution should be blocked
    with pytest.raises(SafetyViolationError) as exc_info:
        gk.verify_execution_permitted(
            tool_name="delete_file",
            safety_level=SafetyLevel.DANGEROUS,
            targets=[str(target)],
        )
    assert "requires explicit confirmation" in str(exc_info.value)

    # 2. Request confirmation
    req = gk.request_confirmation(
        tool_name="delete_file",
        description="Delete temporary file",
        targets=[str(target)],
    )
    assert req.action_id.startswith("act_")
    assert req.confirmed is False

    # 3. Execution with unconfirmed token should be blocked
    with pytest.raises(SafetyViolationError) as exc_info:
        gk.verify_execution_permitted(
            tool_name="delete_file",
            safety_level=SafetyLevel.DANGEROUS,
            action_id=req.action_id,
            targets=[str(target)],
        )
    assert "is not confirmed" in str(exc_info.value)

    # 4. Confirm action
    assert gk.confirm_action(req.action_id) is True
    assert gk.get_pending_request(req.action_id).confirmed is True

    # 5. Execution now permitted
    assert (
        gk.verify_execution_permitted(
            tool_name="delete_file",
            safety_level=SafetyLevel.DANGEROUS,
            action_id=req.action_id,
            targets=[str(target)],
        )
        is True
    )

    # 6. Re-using the same action_id is blocked because it has been consumed
    with pytest.raises(SafetyViolationError):
        gk.verify_execution_permitted(
            tool_name="delete_file",
            safety_level=SafetyLevel.DANGEROUS,
            action_id=req.action_id,
            targets=[str(target)],
        )


def test_gatekeeper_cancel_action():
    gk = SafetyGatekeeper()
    req = gk.request_confirmation("tool_test", "desc", ["target1"])
    assert gk.get_pending_request(req.action_id) is not None

    assert gk.cancel_action(req.action_id) is True
    assert gk.get_pending_request(req.action_id) is None
    assert gk.cancel_action("non_existent_id") is False
