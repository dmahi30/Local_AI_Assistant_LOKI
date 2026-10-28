"""Unit and integration tests for LOKI Desktop UI backend."""

from pathlib import Path
import threading
import time
from unittest.mock import MagicMock, patch
import pytest
import requests
import uvicorn

import loki.commands.builtin.file_delete as file_delete_module
from loki.agent import LokiAgent, ResponseSource
from loki.core.config import LokiConfig
from loki.core.types import ToolResult
from loki.llm.client import OllamaClient
from loki.safety.gatekeeper import SafetyGatekeeper
from loki.ui.launcher import (
    find_free_port,
    launch_desktop_window,
    run_desktop_ui,
    wait_for_server_ready,
)
from loki.ui.server import (
    CancelRequest,
    ChatRequest,
    ConfirmRequest,
    create_ui_app,
)


@pytest.fixture
def mock_agent(tmp_path: Path) -> LokiAgent:
    """Fixture providing a mock LokiAgent for UI tests with an isolated test store."""
    from loki.storage import ConversationStore

    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.get_version.return_value = "0.40.5"
    mock_client.list_local_models.return_value = ["qwen2.5:0.5b"]
    mock_client.is_model_available.return_value = True
    mock_client.chat.return_value = "Mocked conversational response from LOKI."

    store = ConversationStore(db_path=tmp_path / "mock_agent_ui.db")
    agent = LokiAgent(client=mock_client, storage=store)
    return agent


def test_ui_status_returns_live_telemetry(mock_agent):
    """Verify /api/status returns genuine hardware and Ollama telemetry."""
    app = create_ui_app(agent=mock_agent)

    # Find the status route handler
    route_fn = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/status":
            route_fn = route.endpoint
            break

    assert route_fn is not None
    status = route_fn()

    assert status["ollama_online"] is True
    assert status["ollama_version"] == "0.40.5"
    assert status["ollama_host"] == "http://127.0.0.1:11434"
    assert status["active_model"] == "qwen2.5:0.5b"
    assert status["model_available"] is True
    assert status["effective_model"] == "qwen2.5:0.5b"
    assert status["installed_models"] == ["qwen2.5:0.5b"]
    assert status["ram_total_gb"] > 0
    assert status["ram_used_gb"] > 0
    assert 0 <= status["ram_percent"] <= 100
    assert status["cpu_cores"] >= 1
    assert status["offline_mode_enforced"] is True


def test_ui_chat_deterministic_command_bypass(mock_agent):
    """Verify deterministic commands bypass LLM with zero LLM invocations."""
    app = create_ui_app(agent=mock_agent)

    route_fn = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/chat":
            route_fn = route.endpoint
            break

    assert route_fn is not None
    req = ChatRequest(message="what is my battery percentage")
    res = route_fn(req)

    assert res["success"] is True
    assert res["source"] == "deterministic"
    assert res["matched_command"] == "system_info"
    assert res["pending_confirmation"] is None
    # LLM was never contacted
    mock_agent.client.chat.assert_not_called()


def test_ui_chat_conversational_llm(mock_agent):
    """Verify non-command messages cleanly invoke the LLM through LokiAgent."""
    app = create_ui_app(agent=mock_agent)

    route_fn = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/chat":
            route_fn = route.endpoint
            break

    assert route_fn is not None
    req = ChatRequest(message="Explain general relativity")
    res = route_fn(req)

    assert res["success"] is True
    assert res["source"] == "llm"
    assert "Mocked conversational response" in res["text"]
    assert res["model_used"] == "qwen2.5:0.5b"


def test_ui_gated_confirmation_and_approval_flow(tmp_path: Path, monkeypatch):
    """Verify UI handles pending confirmation and authorizes execution via /api/confirm."""
    safe_root = tmp_path / "UIDir"
    safe_root.mkdir()
    target_file = safe_root / "obsolete.tmp"
    target_file.write_text("delete me", encoding="utf-8")

    # Never move test files into the real Windows Recycle Bin.
    # Simulate the file leaving its original location instead.
    monkeypatch.setattr(
        file_delete_module,
        "_move_to_recycle_bin",
        lambda path: path.unlink(),
    )

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    gk.config.recycle_bin_deletions = True
    agent = LokiAgent(gatekeeper=gk)
    app = create_ui_app(agent=agent)

    chat_fn = None
    confirm_fn = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/chat":
            chat_fn = route.endpoint
        elif getattr(route, "path", None) == "/api/confirm":
            confirm_fn = route.endpoint

    # 1. First call: returns pending confirmation
    chat_res = chat_fn(ChatRequest(message=f"delete {target_file}"))
    assert chat_res["pending_confirmation"] is not None
    assert target_file.exists()
    action_id = chat_res["pending_confirmation"]["action_id"]

    # 2. Confirm call: executes with authorization
    confirm_res = confirm_fn(
        ConfirmRequest(
            action_id=action_id,
            original_message=f"delete {target_file}",
        )
    )
    assert confirm_res["success"] is True
    assert not target_file.exists()


def test_ui_confirmation_ignores_modified_browser_message(tmp_path: Path, monkeypatch):
    """A browser cannot redirect a confirmed deletion to a different file."""
    safe_root = tmp_path / "UIDir"
    safe_root.mkdir()
    approved_file = safe_root / "approved.tmp"
    other_file = safe_root / "other.tmp"
    approved_file.write_text("approved", encoding="utf-8")
    other_file.write_text("must remain", encoding="utf-8")

    moved = []

    def fake_recycle_bin(path: Path) -> None:
        moved.append(path)
        path.unlink()  # Simulate the file leaving its original location.

    monkeypatch.setattr(file_delete_module, "_move_to_recycle_bin", fake_recycle_bin)

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    gk.config.recycle_bin_deletions = True
    agent = LokiAgent(gatekeeper=gk)
    app = create_ui_app(agent=agent)

    chat_fn = None
    confirm_fn = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/chat":
            chat_fn = route.endpoint
        elif getattr(route, "path", None) == "/api/confirm":
            confirm_fn = route.endpoint

    chat_res = chat_fn(ChatRequest(message=f"delete {approved_file}"))
    action_id = chat_res["pending_confirmation"]["action_id"]

    # The browser tampers with original_message to point at a different file.
    confirm_res = confirm_fn(
        ConfirmRequest(
            action_id=action_id,
            original_message=f"delete {other_file}",
        )
    )

    assert confirm_res["success"] is True
    assert moved == [approved_file.resolve()]
    assert not approved_file.exists()
    assert other_file.exists()
    assert other_file.read_text(encoding="utf-8") == "must remain"


def test_ui_cancel_action(tmp_path: Path):
    """Verify /api/cancel cancels pending gated actions."""
    safe_root = tmp_path / "UIDir"
    safe_root.mkdir()
    target_file = safe_root / "keep.tmp"
    target_file.write_text("keep me", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    agent = LokiAgent(gatekeeper=gk)
    app = create_ui_app(agent=agent)

    chat_fn = None
    cancel_fn = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/chat":
            chat_fn = route.endpoint
        elif getattr(route, "path", None) == "/api/cancel":
            cancel_fn = route.endpoint

    chat_res = chat_fn(ChatRequest(message=f"delete {target_file}"))
    action_id = chat_res["pending_confirmation"]["action_id"]

    cancel_res = cancel_fn(CancelRequest(action_id=action_id))
    assert cancel_res["success"] is True
    assert "canceled" in cancel_res["message"]
    assert target_file.exists()


def test_ui_clear_history(mock_agent):
    """Verify /api/clear resets conversational context."""
    app = create_ui_app(agent=mock_agent)

    chat_fn = None
    clear_fn = None
    history_fn = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/chat":
            chat_fn = route.endpoint
        elif getattr(route, "path", None) == "/api/clear":
            clear_fn = route.endpoint
        elif getattr(route, "path", None) == "/api/history":
            history_fn = route.endpoint

    chat_fn(ChatRequest(message="Hello turn 1"))
    assert len(history_fn()["history"]) > 0

    clear_res = clear_fn()
    assert clear_res["success"] is True
    assert len(history_fn()["history"]) == 0


def test_ui_offline_ollama_graceful_handling():
    """Verify UI chat returns safe diagnostic error when Ollama is offline."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = False

    agent = LokiAgent(client=mock_client)
    app = create_ui_app(agent=agent)

    chat_fn = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/chat":
            chat_fn = route.endpoint
            break

    res = chat_fn(ChatRequest(message="Can you help me?"))
    assert res["success"] is False
    assert res["source"] == "system"
    assert "offline or unreachable" in res["text"]


def test_ui_http_server_endpoints(mock_agent):
    """Integration test running Uvicorn server on ephemeral port and querying over HTTP."""
    port = find_free_port(11499)
    app = create_ui_app(agent=mock_agent)

    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    base_url = f"http://127.0.0.1:{port}"

    # Wait for server ready
    ready = False
    for _ in range(30):
        try:
            r = requests.get(f"{base_url}/api/status", timeout=0.5)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.05)

    assert ready is True

    try:
        # 1. Test Status
        res_status = requests.get(f"{base_url}/api/status")
        assert res_status.status_code == 200
        assert res_status.json()["ollama_online"] is True

        # 2. Test Chat
        res_chat = requests.post(
            f"{base_url}/api/chat",
            json={"message": "what is my battery percentage"},
        )
        assert res_chat.status_code == 200
        assert res_chat.json()["source"] == "deterministic"

        # 3. Test Static index.html serving
        res_index = requests.get(f"{base_url}/")
        assert res_index.status_code == 200
        assert "LOKI" in res_index.text

        # 4. Test Static style.css serving
        res_css = requests.get(f"{base_url}/static/style.css")
        assert res_css.status_code == 200
        assert "gold-primary" in res_css.text

    finally:
        server.should_exit = True


def test_wait_for_server_ready_success(mock_agent):
    """Verify wait_for_server_ready detects responsive server within timeout."""
    port = find_free_port(11498)
    app = create_ui_app(agent=mock_agent)
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    url = f"http://127.0.0.1:{port}"
    try:
        ready = wait_for_server_ready(url, timeout=3.0, server_thread=server_thread)
        assert ready is True
    finally:
        server.should_exit = True
        server_thread.join(timeout=2.0)


def test_wait_for_server_ready_timeout():
    """Verify wait_for_server_ready returns False when target port is unreachable."""
    port = find_free_port(11497)
    ready = wait_for_server_ready(f"http://127.0.0.1:{port}", timeout=0.2, poll_interval=0.05)
    assert ready is False


def test_wait_for_server_ready_fails_fast_on_dead_server_thread():
    """Verify wait_for_server_ready immediately aborts if server thread died."""
    port = find_free_port(11496)
    dead_thread = threading.Thread(target=lambda: None)
    dead_thread.start()
    dead_thread.join()

    start = time.time()
    ready = wait_for_server_ready(
        f"http://127.0.0.1:{port}",
        timeout=5.0,
        server_thread=dead_thread,
    )
    elapsed = time.time() - start

    assert ready is False
    assert elapsed < 1.0  # Fails fast without waiting full 5 seconds


def test_run_desktop_ui_startup_failure_does_not_open_window(mock_agent):
    """Verify run_desktop_ui halts and refuses to open browser if readiness check fails."""
    with patch("loki.ui.launcher.wait_for_server_ready", return_value=False) as mock_ready:
        with patch("loki.ui.launcher.launch_desktop_window") as mock_launch:
            result = run_desktop_ui(
                agent=mock_agent,
                port=11495,
                open_window=True,
                startup_timeout=0.1,
            )

            assert result is False
            mock_ready.assert_called_once()
            mock_launch.assert_not_called()


def test_run_desktop_ui_delegated_browser_lifecycle(mock_agent):
    """Verify run_desktop_ui safely handles browser process delegation (Edge exit code 21)."""
    mock_proc = MagicMock()
    mock_proc.pid = 4321
    mock_proc.returncode = 21
    mock_proc.poll.return_value = 21  # Browser manager delegated window and exited immediately

    with patch("loki.ui.launcher.wait_for_server_ready", return_value=True):
        with patch("loki.ui.launcher.launch_desktop_window", return_value=mock_proc):
            with patch("time.sleep", return_value=None):
                # Mock server run and exit
                def mock_run_server():
                    pass

                with patch("uvicorn.Server.run", side_effect=mock_run_server):
                    result = run_desktop_ui(
                        agent=mock_agent,
                        port=11494,
                        open_window=True,
                        startup_timeout=0.1,
                    )
                    assert result is True


def test_ui_default_active_model_is_qwen_05b():
    """Verify that default UI app configuration uses qwen2.5:0.5b."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.get_version.return_value = "0.40.5"
    mock_client.list_local_models.return_value = ["qwen2.5:0.5b"]
    mock_client.is_model_available.return_value = True

    agent = LokiAgent(client=mock_client)
    app = create_ui_app(agent=agent)

    for route in app.routes:
        if getattr(route, "path", None) == "/api/status":
            status = route.endpoint()
            assert status["active_model"] == "qwen2.5:0.5b"
            assert status["model_available"] is True
            assert status["effective_model"] == "qwen2.5:0.5b"
            break


def test_ui_status_distinguishes_model_missing_from_ollama_offline():
    """Verify status endpoint distinguishes daemon connectivity from model missing."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.get_version.return_value = "0.40.5"
    mock_client.list_local_models.return_value = ["other-model:latest"]
    mock_client.is_model_available.return_value = False

    agent = LokiAgent(client=mock_client)
    app = create_ui_app(agent=agent)

    for route in app.routes:
        if getattr(route, "path", None) == "/api/status":
            status = route.endpoint()
            # Invariant: Daemon is online, but model is clearly reported as missing
            assert status["ollama_online"] is True
            assert status["active_model"] == "qwen2.5:0.5b"
            assert status["model_available"] is False
            assert status["effective_model"] is None
            assert status["using_fallback"] is False
            assert status["installed_models"] == ["other-model:latest"]
            break


def test_ui_status_detects_fallback_model_when_enabled():
    """Verify status endpoint reports approved fallback when active model is missing."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.get_version.return_value = "0.40.5"
    mock_client.list_local_models.return_value = ["qwen2.5:1.5b"]
    mock_client.is_model_available.side_effect = lambda m, **kwargs: m == "qwen2.5:1.5b"

    config = LokiConfig()
    config.models.active_model = "qwen2.5:0.5b"
    config.models.fallback_models = ["qwen2.5:1.5b"]

    agent = LokiAgent(config=config, client=mock_client, allow_fallback=True)
    app = create_ui_app(agent=agent)

    for route in app.routes:
        if getattr(route, "path", None) == "/api/status":
            status = route.endpoint()
            assert status["ollama_online"] is True
            assert status["active_model"] == "qwen2.5:0.5b"
            assert status["model_available"] is True
            assert status["using_fallback"] is True
            assert status["effective_model"] == "qwen2.5:1.5b"
            break


def test_create_ui_app_isolated_from_real_database(tmp_path: Path):
    """Regression test: verify create_ui_app cannot write conversations to the real user database."""
    from loki.storage import ConversationStore
    from loki.utils.paths import get_default_app_dir, get_data_dir

    real_db_path = get_default_app_dir() / "data" / "loki.db"

    # Capture real database state before test
    real_db_exists = real_db_path.exists()
    real_db_size = real_db_path.stat().st_size if real_db_exists else 0
    real_db_mtime = real_db_path.stat().st_mtime if real_db_exists else 0

    mock_client = MagicMock(spec=OllamaClient)
    mock_client.chat.return_value = "Isolated response."

    # Create app without explicit storage - should use isolated LOKI_HOME
    agent = LokiAgent(client=mock_client)
    app = create_ui_app(agent=agent)

    # Invariant: storage is NOT pointing to the real user database
    assert agent.storage is not None
    assert agent.storage.db_path != real_db_path
    assert str(get_data_dir()) in str(agent.storage.db_path)

    # Execute a chat turn
    for route in app.routes:
        if getattr(route, "path", None) == "/api/chat":
            route.endpoint(ChatRequest(message="Isolation verification test prompt"))
            break

    # Verify conversation was written to isolated test store, NOT the real db
    isolated_convs = agent.storage.list_conversations()
    assert len(isolated_convs) == 1
    assert isolated_convs[0].title == "Isolation verification test prompt"

    # Invariant: Real user database was completely untouched
    if real_db_exists:
        assert real_db_path.stat().st_size == real_db_size
        assert real_db_path.stat().st_mtime == real_db_mtime


def test_run_desktop_ui_reuses_existing_instance(mock_agent):
    """Verify run_desktop_ui detects already-healthy local instance and avoids duplicate server thread."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with patch("loki.ui.launcher.launch_desktop_window") as mock_launch:
            with patch("uvicorn.Server.run") as mock_server_run:
                result = run_desktop_ui(agent=mock_agent, port=None, open_window=True)
                assert result is True
                # Existing server was reused, new uvicorn server was NOT spawned
                mock_server_run.assert_not_called()
                mock_launch.assert_called_once_with("http://127.0.0.1:11435")


def test_loki_home_override_preserves_real_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Regression test: verify that deliberately setting LOKI_HOME points all storage to that directory and leaves the real database untouched."""
    from loki.storage import ConversationStore
    from loki.utils.paths import get_default_app_dir, get_data_dir

    real_db_path = get_default_app_dir() / "data" / "loki.db"
    real_db_exists = real_db_path.exists()
    real_db_size = real_db_path.stat().st_size if real_db_exists else 0
    real_db_mtime = real_db_path.stat().st_mtime if real_db_exists else 0

    custom_dir = tmp_path / "custom_loki_env"
    monkeypatch.setenv("LOKI_HOME", str(custom_dir))

    # Assert get_data_dir strictly follows the overridden LOKI_HOME
    expected_data_dir = custom_dir / "data"
    assert get_data_dir() == expected_data_dir

    # Default-initialized ConversationStore must write to custom_dir/data/loki.db
    store = ConversationStore()
    assert store.db_path == expected_data_dir / "loki.db"

    # Add conversations and messages
    conv = store.create_conversation("Override Verification Chat")
    store.add_message(conv.id, role="user", content="Test message in custom LOKI_HOME")
    store.add_message(conv.id, role="assistant", content="Response in custom LOKI_HOME")

    # Invariant: Data is present in custom directory
    detail = store.get_conversation(conv.id)
    assert detail is not None
    assert len(detail.messages) == 2

    # Invariant: Real user database at %LOCALAPPDATA%\LOKI\data\loki.db was NOT touched
    if real_db_exists:
        assert real_db_path.stat().st_size == real_db_size
        assert real_db_path.stat().st_mtime == real_db_mtime


def test_run_desktop_ui_reuses_instance_on_arbitrary_port_via_server_info(mock_agent):
    """Verify run_desktop_ui detects running instance on an arbitrary port via server.json metadata."""
    from loki.ui.launcher import _write_server_info

    # Simulate an existing server running on dynamic ephemeral port 54321
    _write_server_info("http://127.0.0.1:54321", 54321)

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with patch("loki.ui.launcher.launch_desktop_window") as mock_launch:
            with patch("uvicorn.Server.run") as mock_server_run:
                result = run_desktop_ui(agent=mock_agent, port=None, open_window=True)
                assert result is True
                # Reused dynamic port instance, avoided starting second server
                mock_server_run.assert_not_called()
                mock_launch.assert_called_once_with("http://127.0.0.1:54321")

# =========================================================================
# Phase 8: Hardened Confirmation UI Security Tests
# =========================================================================

def test_ui_confirm_rejects_unsupported_tools():
    """Verify /api/confirm rejects actions for tools not supported by UI confirmation."""
    from fastapi import HTTPException

    gk = SafetyGatekeeper()
    # Create a pending confirmation for an unsupported tool
    req = gk.request_confirmation("system_terminal_exec", "Run terminal script", ["C:\\script.bat"])

    agent = LokiAgent(gatekeeper=gk)
    app = create_ui_app(agent=agent)

    confirm_fn = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/confirm":
            confirm_fn = route.endpoint

    with pytest.raises(HTTPException) as exc_info:
        confirm_fn(ConfirmRequest(action_id=req.action_id, original_message="run script"))

    assert exc_info.value.status_code == 400
    assert "not supported by the UI" in exc_info.value.detail
    # Invariant: Action was never confirmed
    assert gk.get_pending_request(req.action_id).confirmed is False


def test_ui_confirm_rejects_expired_action():
    """Verify /api/confirm rejects action IDs that have exceeded their TTL."""
    from fastapi import HTTPException

    gk = SafetyGatekeeper()
    req = gk.request_confirmation("file_delete", "Delete file", ["C:\\temp.txt"])
    # Artificially expire the request
    req.ttl_seconds = 0.001
    time.sleep(0.01)

    agent = LokiAgent(gatekeeper=gk)
    app = create_ui_app(agent=agent)

    confirm_fn = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/confirm")

    with pytest.raises(HTTPException) as exc_info:
        confirm_fn(ConfirmRequest(action_id=req.action_id, original_message="delete C:\\temp.txt"))

    assert exc_info.value.status_code == 400
    assert "invalid, expired, or already consumed" in exc_info.value.detail


def test_ui_confirm_rejects_replay_attempts(tmp_path: Path, monkeypatch):
    """Verify a consumed action ID cannot be replayed a second time."""
    from fastapi import HTTPException

    safe_root = tmp_path / "UIDir"
    safe_root.mkdir()
    target_file = safe_root / "temp.log"
    target_file.write_text("to delete", encoding="utf-8")

    monkeypatch.setattr(
        file_delete_module,
        "_move_to_recycle_bin",
        lambda p: p.unlink(),
    )

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    agent = LokiAgent(gatekeeper=gk)
    app = create_ui_app(agent=agent)

    chat_fn = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/chat")
    confirm_fn = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/confirm")

    # Step 1: Request dangerous action
    chat_res = chat_fn(ChatRequest(message=f"delete {target_file}"))
    action_id = chat_res["pending_confirmation"]["action_id"]

    # Step 2: First confirmation succeeds
    confirm_res1 = confirm_fn(ConfirmRequest(action_id=action_id, original_message=f"delete {target_file}"))
    assert confirm_res1["success"] is True
    assert not target_file.exists()

    # Step 3: Replay with same action_id must be strictly rejected
    with pytest.raises(HTTPException) as exc_info:
        confirm_fn(ConfirmRequest(action_id=action_id, original_message=f"delete {target_file}"))

    assert exc_info.value.status_code == 400
    assert "invalid, expired, or already consumed" in exc_info.value.detail


def test_ui_confirm_rejects_conversation_id_tampering(tmp_path: Path, monkeypatch):
    """Verify /api/confirm rejects requests attempting to redirect approval to a different conversation."""
    from fastapi import HTTPException

    safe_root = tmp_path / "UIDir"
    safe_root.mkdir()
    target_file = safe_root / "test.txt"
    target_file.write_text("content", encoding="utf-8")

    monkeypatch.setattr(
        file_delete_module,
        "_move_to_recycle_bin",
        lambda p: p.unlink(),
    )

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    agent = LokiAgent(gatekeeper=gk)
    app = create_ui_app(agent=agent)

    chat_fn = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/chat")
    confirm_fn = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/confirm")

    # Create confirmation bound to conversation "conv_original"
    chat_res = chat_fn(ChatRequest(message=f"delete {target_file}", conversation_id="conv_original"))
    action_id = chat_res["pending_confirmation"]["action_id"]

    # Attempt to confirm from an unauthorized conversation "conv_hijacker"
    with pytest.raises(HTTPException) as exc_info:
        confirm_fn(
            ConfirmRequest(
                action_id=action_id,
                original_message=f"delete {target_file}",
                conversation_id="conv_hijacker",
            )
        )

    assert exc_info.value.status_code == 400
    assert "Conversation ID mismatch" in exc_info.value.detail
    # Invariant: target file was NOT deleted
    assert target_file.exists()


def test_ui_confirm_rejects_invalid_targets_count():
    """Verify /api/confirm rejects file_delete requests with multiple or zero targets."""
    from fastapi import HTTPException

    gk = SafetyGatekeeper()
    # file_delete requires exactly one target
    req = gk.request_confirmation("file_delete", "Multi delete", ["C:\\file1.txt", "C:\\file2.txt"])

    agent = LokiAgent(gatekeeper=gk)
    app = create_ui_app(agent=agent)

    confirm_fn = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/api/confirm")

    with pytest.raises(HTTPException) as exc_info:
        confirm_fn(ConfirmRequest(action_id=req.action_id, original_message="delete files"))

    assert exc_info.value.status_code == 400
    assert "Invalid file deletion confirmation targets" in exc_info.value.detail
