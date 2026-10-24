"""Integration tests for LOKI UI Persistent Chat History and Conversation Management."""

from pathlib import Path
import threading
import time
from unittest.mock import MagicMock
import pytest
import requests
import uvicorn
from fastapi import HTTPException

from loki.agent import LokiAgent
from loki.llm.client import OllamaClient
from loki.storage import ConversationStore
from loki.ui.launcher import find_free_port
from loki.ui.server import (
    ChatRequest,
    CreateConversationRequest,
    UpdateConversationRequest,
    create_ui_app,
)


@pytest.fixture
def mock_agent_and_store(tmp_path: Path):
    """Fixture providing a mock LokiAgent with an isolated temporary SQLite store."""
    db_file = tmp_path / "test_ui_loki.db"
    store = ConversationStore(db_path=db_file)

    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.get_version.return_value = "0.40.5"
    mock_client.list_local_models.return_value = ["qwen2.5:0.5b"]
    mock_client.is_model_available.return_value = True
    mock_client.chat.side_effect = lambda msgs, **kwargs: f"Echo: {msgs[-1]['content']}"

    agent = LokiAgent(client=mock_client, storage=store)
    return agent, store


def test_ui_conversations_empty_list(mock_agent_and_store):
    """Verify GET /api/conversations returns empty list when no conversations exist."""
    agent, store = mock_agent_and_store
    app = create_ui_app(agent=agent, storage=store)

    list_fn = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/conversations" and "GET" in route.methods:
            list_fn = route.endpoint
            break

    assert list_fn is not None
    res = list_fn()
    assert res == {"conversations": []}


def test_ui_conversations_create_and_get(mock_agent_and_store):
    """Verify POST /api/conversations creates a conversation and GET /api/conversations/{id} loads it."""
    agent, store = mock_agent_and_store
    app = create_ui_app(agent=agent, storage=store)

    create_fn = None
    get_fn = None
    list_fn = None
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", set())
        if path == "/api/conversations" and "POST" in methods:
            create_fn = route.endpoint
        elif path == "/api/conversations/{conversation_id}" and "GET" in methods:
            get_fn = route.endpoint
        elif path == "/api/conversations" and "GET" in methods:
            list_fn = route.endpoint

    # 1. Create with default title
    created = create_fn(CreateConversationRequest(title="Default Session"))
    conv_id = created["conversation"]["id"]
    assert created["conversation"]["title"] == "Default Session"
    assert created["conversation"]["message_count"] == 0

    # 2. Verify agent switched to this active conversation
    assert agent.active_conversation_id == conv_id

    # 3. Get detail
    detail = get_fn(conv_id)
    assert detail["conversation"]["id"] == conv_id
    assert detail["conversation"]["title"] == "Default Session"
    assert detail["conversation"]["messages"] == []

    # 4. Verify in list
    listed = list_fn()
    assert len(listed["conversations"]) == 1
    assert listed["conversations"][0]["id"] == conv_id


def test_ui_conversations_get_not_found(mock_agent_and_store):
    """Verify GET /api/conversations/{id} raises 404 for unknown conversation."""
    agent, store = mock_agent_and_store
    app = create_ui_app(agent=agent, storage=store)

    get_fn = None
    for route in app.routes:
        if getattr(route, "path", None) == "/api/conversations/{conversation_id}" and "GET" in route.methods:
            get_fn = route.endpoint
            break

    with pytest.raises(HTTPException) as exc_info:
        get_fn("non_existent_id")
    assert exc_info.value.status_code == 404


def test_ui_conversations_rename_success_and_validation(mock_agent_and_store):
    """Verify PATCH /api/conversations/{id} renames conversation and validates title."""
    agent, store = mock_agent_and_store
    app = create_ui_app(agent=agent, storage=store)

    create_fn = None
    patch_fn = None
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", set())
        if path == "/api/conversations" and "POST" in methods:
            create_fn = route.endpoint
        elif path == "/api/conversations/{conversation_id}" and "PATCH" in methods:
            patch_fn = route.endpoint

    created = create_fn(CreateConversationRequest(title="Old Name"))
    cid = created["conversation"]["id"]

    # Valid rename
    renamed = patch_fn(cid, UpdateConversationRequest(title="New Name"))
    assert renamed["success"] is True
    assert renamed["conversation"]["title"] == "New Name"

    # Empty whitespace title raises 400
    with pytest.raises(HTTPException) as exc_info:
        patch_fn(cid, UpdateConversationRequest(title="   "))
    assert exc_info.value.status_code == 400

    # Unknown ID raises 404
    with pytest.raises(HTTPException) as exc_info:
        patch_fn("fake_id", UpdateConversationRequest(title="Valid"))
    assert exc_info.value.status_code == 404


def test_ui_conversations_delete_cascade_and_active_reset(mock_agent_and_store):
    """Verify DELETE /api/conversations/{id} deletes conversation, cascades messages, and clears active context."""
    agent, store = mock_agent_and_store
    app = create_ui_app(agent=agent, storage=store)

    create_fn = None
    delete_fn = None
    chat_fn = None
    list_fn = None
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", set())
        if path == "/api/conversations" and "POST" in methods:
            create_fn = route.endpoint
        elif path == "/api/conversations/{conversation_id}" and "DELETE" in methods:
            delete_fn = route.endpoint
        elif path == "/api/chat" and "POST" in methods:
            chat_fn = route.endpoint
        elif path == "/api/conversations" and "GET" in methods:
            list_fn = route.endpoint

    created = create_fn(CreateConversationRequest(title="Delete Target"))
    cid = created["conversation"]["id"]
    assert agent.active_conversation_id == cid

    # Send a message to populate history
    chat_fn(ChatRequest(message="Message in conversation", conversation_id=cid))
    assert len(store.get_conversation(cid).messages) == 2

    # Delete conversation
    del_res = delete_fn(cid)
    assert del_res["success"] is True

    # Active conversation should be reset to None and history cleared
    assert agent.active_conversation_id is None
    assert agent.get_history() == []

    # Database record and messages should no longer exist
    assert store.get_conversation(cid) is None
    assert list_fn()["conversations"] == []

    # Deleting again raises 404
    with pytest.raises(HTTPException) as exc_info:
        delete_fn(cid)
    assert exc_info.value.status_code == 404


def test_ui_chat_auto_creates_conversation_and_auto_titles(mock_agent_and_store):
    """Verify /api/chat auto-creates conversation if omitted and auto-titles from prompt."""
    agent, store = mock_agent_and_store
    app = create_ui_app(agent=agent, storage=store)

    chat_fn = None
    get_fn = None
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", set())
        if path == "/api/chat" and "POST" in methods:
            chat_fn = route.endpoint
        elif path == "/api/conversations/{conversation_id}" and "GET" in methods:
            get_fn = route.endpoint

    # Chat with no conversation_id specified
    res = chat_fn(ChatRequest(message="What is the speed of light?"))
    cid = res["conversation_id"]
    assert cid is not None
    assert res["success"] is True

    # Conversation was automatically created and auto-titled from the first question
    detail = get_fn(cid)["conversation"]
    assert "What is the speed of light" in detail["title"]
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][0]["content"] == "What is the speed of light?"
    assert detail["messages"][1]["role"] == "assistant"
    assert "Echo: What is the speed of light?" in detail["messages"][1]["content"]


def test_ui_multi_conversation_isolation(mock_agent_and_store):
    """Verify strict conversational context isolation across multiple conversations through the UI."""
    agent, store = mock_agent_and_store
    app = create_ui_app(agent=agent, storage=store)

    create_fn = None
    chat_fn = None
    get_fn = None
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", set())
        if path == "/api/conversations" and "POST" in methods:
            create_fn = route.endpoint
        elif path == "/api/chat" and "POST" in methods:
            chat_fn = route.endpoint
        elif path == "/api/conversations/{conversation_id}" and "GET" in methods:
            get_fn = route.endpoint

    # 1. Create Conversation A (e.g. Physics)
    conv_a = create_fn(CreateConversationRequest(title="Physics"))["conversation"]["id"]
    chat_fn(ChatRequest(message="My secret topic is Quantum Gravity", conversation_id=conv_a))

    # Inspect call args for client.chat
    last_call = agent.client.chat.call_args[0][0]
    prompt_contents_a = [m["content"] for m in last_call]
    assert any("Quantum Gravity" in c for c in prompt_contents_a)

    # 2. Create Conversation B (e.g. Cooking)
    conv_b = create_fn(CreateConversationRequest(title="Cooking"))["conversation"]["id"]
    chat_fn(ChatRequest(message="How do I bake sourdough bread?", conversation_id=conv_b))

    # Invariant: Prompt for conversation B MUST NOT contain "Quantum Gravity" from conversation A
    last_call_b = agent.client.chat.call_args[0][0]
    prompt_contents_b = [m["content"] for m in last_call_b]
    assert any("sourdough bread" in c for c in prompt_contents_b)
    for c in prompt_contents_b:
        assert "Quantum Gravity" not in c

    # 3. Switch back to Conversation A and continue chatting
    chat_fn(ChatRequest(message="Can you elaborate further?", conversation_id=conv_a))

    last_call_a2 = agent.client.chat.call_args[0][0]
    prompt_contents_a2 = [m["content"] for m in last_call_a2]
    # Invariant: Must contain original Quantum Gravity, and MUST NOT contain sourdough bread
    assert any("Quantum Gravity" in c for c in prompt_contents_a2)
    assert any("elaborate further" in c for c in prompt_contents_a2)
    for c in prompt_contents_a2:
        assert "sourdough bread" not in c


def test_ui_http_server_conversation_endpoints_integration(mock_agent_and_store):
    """Full HTTP network integration test of conversation endpoints over loopback Uvicorn."""
    agent, store = mock_agent_and_store
    port = find_free_port(11550)
    app = create_ui_app(agent=agent, storage=store)

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
    for _ in range(40):
        try:
            r = requests.get(f"{base_url}/api/status", timeout=0.5)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.05)

    assert ready, "Server failed to start on loopback"

    try:
        # 1. Initially empty
        r = requests.get(f"{base_url}/api/conversations")
        assert r.status_code == 200
        assert r.json()["conversations"] == []

        # 2. POST create conversation
        r = requests.post(f"{base_url}/api/conversations", json={"title": "Session One"})
        assert r.status_code == 200
        cid = r.json()["conversation"]["id"]
        assert r.json()["conversation"]["title"] == "Session One"

        # 3. POST chat message associated to session
        r = requests.post(
            f"{base_url}/api/chat",
            json={"message": "What is Python?", "conversation_id": cid},
        )
        assert r.status_code == 200
        res = r.json()
        assert res["success"] is True
        assert res["conversation_id"] == cid

        # 4. GET conversation detail
        r = requests.get(f"{base_url}/api/conversations/{cid}")
        assert r.status_code == 200
        detail = r.json()["conversation"]
        assert len(detail["messages"]) == 2

        # 5. PATCH rename
        r = requests.patch(
            f"{base_url}/api/conversations/{cid}",
            json={"title": "Python Overview"},
        )
        assert r.status_code == 200
        assert r.json()["conversation"]["title"] == "Python Overview"

        # 6. DELETE conversation
        r = requests.delete(f"{base_url}/api/conversations/{cid}")
        assert r.status_code == 200
        assert r.json()["success"] is True

        # 7. GET returns 404
        r = requests.get(f"{base_url}/api/conversations/{cid}")
        assert r.status_code == 404

    finally:
        server.should_exit = True


def test_ui_serves_html_and_assets_with_no_cache_headers(mock_agent_and_store):
    """Regression test: verify HTML and JS assets are served with no-cache headers to prevent Edge stale caching."""
    agent, store = mock_agent_and_store
    port = find_free_port(11560)
    app = create_ui_app(agent=agent, storage=store)

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

    ready = False
    for _ in range(40):
        try:
            r = requests.get(f"{base_url}/api/status", timeout=0.5)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.05)

    assert ready, "Server failed to start on loopback"

    try:
        # 1. Root index.html check
        r_index = requests.get(f"{base_url}/")
        assert r_index.status_code == 200
        cache_hdr = r_index.headers.get("Cache-Control", "")
        assert "no-cache" in cache_hdr
        assert "no-store" in cache_hdr

        html_body = r_index.text
        # Invariant: hardcoded 'Current Session' must NEVER appear in served HTML
        assert "Current Session" not in html_body
        # Invariant: searchChatsBtn exists with button type and accessible label
        assert 'id="searchChatsBtn"' in html_body
        assert 'aria-label="Search chats"' in html_body
        assert 'type="button"' in html_body
        # Invariant: search bar and input exist
        assert 'id="recentSearchBar"' in html_body
        assert 'id="recentSearchInput"' in html_body
        # Invariant: version query string to bust cache
        assert "/static/app.js?v=" in html_body

        # 2. Static JS asset check
        r_js = requests.get(f"{base_url}/static/app.js")
        assert r_js.status_code == 200
        js_cache_hdr = r_js.headers.get("Cache-Control", "")
        assert "no-cache" in js_cache_hdr
        # Invariant: JS handles searchChatsBtn and loads conversations dynamically
        assert "searchChatsBtn" in r_js.text
        assert "loadConversations" in r_js.text

        # 3. Static CSS asset check
        r_css = requests.get(f"{base_url}/static/style.css")
        assert r_css.status_code == 200
        assert ".icon-btn-subtle" in r_css.text

    finally:
        server.should_exit = True


def test_ui_multiple_conversations_creation_and_listing_regression(mock_agent_and_store):
    """Regression test: verify creating multiple distinct chats lists all of them in sidebar."""
    agent, store = mock_agent_and_store
    app = create_ui_app(agent=agent, storage=store)

    create_fn = None
    list_fn = None
    chat_fn = None
    get_fn = None
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", set())
        if path == "/api/conversations" and "POST" in methods:
            create_fn = route.endpoint
        elif path == "/api/conversations" and "GET" in methods:
            list_fn = route.endpoint
        elif path == "/api/chat" and "POST" in methods:
            chat_fn = route.endpoint
        elif path == "/api/conversations/{conversation_id}" and "GET" in methods:
            get_fn = route.endpoint

    # 1. Create Conversation 1
    c1 = create_fn(CreateConversationRequest(title="Chat Alpha"))["conversation"]
    chat_fn(ChatRequest(message="Message in Alpha", conversation_id=c1["id"]))

    # 2. Create Conversation 2
    c2 = create_fn(CreateConversationRequest(title="Chat Beta"))["conversation"]
    chat_fn(ChatRequest(message="Message in Beta", conversation_id=c2["id"]))

    # 3. Create Conversation 3
    c3 = create_fn(CreateConversationRequest(title="Chat Gamma"))["conversation"]
    chat_fn(ChatRequest(message="Message in Gamma", conversation_id=c3["id"]))

    # 4. List conversations must return all 3
    all_convs = list_fn()["conversations"]
    assert len(all_convs) == 3
    conv_titles = [c["title"] for c in all_convs]
    assert "Chat Alpha" in conv_titles
    assert "Chat Beta" in conv_titles
    assert "Chat Gamma" in conv_titles

    # 5. Detail inspection: verify isolation between conversations
    detail_alpha = get_fn(c1["id"])["conversation"]
    detail_beta = get_fn(c2["id"])["conversation"]

    alpha_contents = [m["content"] for m in detail_alpha["messages"]]
    beta_contents = [m["content"] for m in detail_beta["messages"]]

    assert any("Message in Alpha" in c for c in alpha_contents)
    assert not any("Message in Beta" in c for c in alpha_contents)

    assert any("Message in Beta" in c for c in beta_contents)
    assert not any("Message in Alpha" in c for c in beta_contents)


def test_launcher_desktop_window_uses_isolated_user_data_dir(tmp_path: Path):
    """Verify launch_desktop_window isolates browser profile to prevent delegation and stale caching."""
    from unittest.mock import patch
    from loki.ui.launcher import launch_desktop_window

    with patch("loki.ui.launcher._find_app_browser", return_value=(r"C:\Windows\msedge.exe", "edge")):
        with patch("subprocess.Popen") as mock_popen:
            custom_profile = tmp_path / "custom_browser_profile"
            launch_desktop_window("http://127.0.0.1:11435", profile_dir=custom_profile)

            assert mock_popen.called
            args = mock_popen.call_args[0][0]
            assert r"C:\Windows\msedge.exe" == args[0]
            assert "--app=http://127.0.0.1:11435" in args
            assert f"--user-data-dir={custom_profile}" in args
            assert "--no-first-run" in args
            assert "--no-default-browser-check" in args
            assert "--new-window" in args


def test_app_js_readystatus_resilience_and_error_handling():
    """Verify app.js provides readyState fallback and diagnostic logging for conversation loading."""
    app_js_path = Path(__file__).resolve().parent.parent / "loki" / "ui" / "static" / "app.js"
    content = app_js_path.read_text(encoding="utf-8")

    # Invariant: Must not solely rely on future DOMContentLoaded; must check document.readyState
    assert "document.readyState" in content
    assert 'document.readyState === "loading"' in content
    assert "initLokiApp" in content

    # Invariant: Diagnostic logging present
    assert "[LOKI UI]" in content

    # Invariant: Error state recovery button present
    assert "retryLoadConvBtn" in content
    assert "Unable to load conversations" in content

    # Invariant: HTML escaping applied to dynamic badges and confirmation gate elements
    assert "escapeHtml(badgeLabel)" in content
    assert "safeDesc" in content
    assert "safeTargets" in content
    assert "safeActionId" in content
