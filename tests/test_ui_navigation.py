"""Tests for LOKI Explore, Library, and Settings Navigation & Endpoints."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from loki.agent import LokiAgent
from loki.core.config import LokiConfig
from loki.llm.client import OllamaClient
from loki.storage.conversation import ConversationStore
from loki.storage.memory import MemoryStore
from loki.ui.server import create_ui_app


@pytest.fixture
def isolated_ui_app(tmp_path: Path):
    """Fixture providing a test FastAPI app with isolated storage and mocked Ollama client."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.get_version.return_value = "0.40.5"
    mock_client.list_local_models.return_value = ["qwen2.5:0.5b"]
    mock_client.is_model_available.return_value = True
    mock_client.chat.return_value = "Mocked conversational response from LOKI."

    test_db = tmp_path / "test_ui_nav.db"
    conv_store = ConversationStore(db_path=test_db)
    mem_store = MemoryStore(db_path=test_db)

    agent = LokiAgent(client=mock_client, storage=conv_store, memory_store=mem_store)
    app = create_ui_app(agent=agent, storage=conv_store)
    return app, agent, tmp_path


# =============================================================================
# 1. HTML & Static Navigation Verification
# =============================================================================


def test_app_js_syntax_validity():
    """Verify app.js parses without any syntax errors using node.js."""
    import subprocess
    app_js_path = Path("loki/ui/static/app.js")
    assert app_js_path.exists()

    res = subprocess.run(["node", "-c", str(app_js_path)], capture_output=True, text=True)
    assert res.returncode == 0, f"app.js syntax error: {res.stderr}"


def test_index_html_script_tag_at_end_of_body():
    """Verify script tag is placed after all DOM elements and right before </body>."""
    html_text = Path("loki/ui/static/index.html").read_text(encoding="utf-8")
    modal_pos = html_text.find('id="settingsModal"')
    script_pos = html_text.find('<script src="/static/app.js')
    body_pos = html_text.find('</body>')

    assert modal_pos != -1
    assert script_pos != -1
    assert body_pos != -1
    assert modal_pos < script_pos, "settingsModal must be defined before app.js script tag"
    assert script_pos < body_pos, "app.js script tag must be inside <body> before </body>"

def test_html_sidebar_navigation_interactive():
    """Verify Explore and Library buttons in index.html are interactive and not marked planned."""
    index_path = Path("loki/ui/static/index.html")
    assert index_path.exists(), "index.html must exist"
    html_text = index_path.read_text(encoding="utf-8")

    # Explore must have data-tab="explore" and must not have class "planned"
    assert 'data-tab="explore"' in html_text
    assert '<button class="nav-item planned" data-tab="explore"' not in html_text

    # Library must have data-tab="library" and must not have class "planned"
    assert 'data-tab="library"' in html_text
    assert '<button class="nav-item planned" data-tab="library"' not in html_text

    # Views must be defined
    assert 'id="chatView"' in html_text
    assert 'id="exploreView"' in html_text
    assert 'id="libraryView"' in html_text

    # Settings modal with memory management must be defined
    assert 'id="settingsModal"' in html_text
    assert 'id="settingsTabMemory"' in html_text
    assert 'id="settingsMemoryPanel"' in html_text
    assert 'id="memoryEnabledToggle"' in html_text


# =============================================================================
# 2. Document Library Endpoints Tests
# =============================================================================

def test_documents_list_empty_when_no_attachments(isolated_ui_app, monkeypatch):
    """Verify /api/documents returns empty list when no documents staged."""
    app, agent, tmp_path = isolated_ui_app
    empty_att_dir = tmp_path / "empty_attachments"
    monkeypatch.setattr("loki.utils.paths.get_data_dir", lambda: tmp_path)

    client = TestClient(app)
    res = client.get("/api/documents")
    assert res.status_code == 200
    data = res.json()
    assert "documents" in data
    assert data["documents"] == []


def test_documents_list_and_delete_lifecycle(isolated_ui_app, monkeypatch):
    """Verify /api/documents lists staged files and /api/documents/{id} deletes them."""
    app, agent, tmp_path = isolated_ui_app
    monkeypatch.setattr("loki.utils.paths.get_data_dir", lambda: tmp_path)

    att_dir = tmp_path / "attachments"
    att_dir.mkdir(parents=True, exist_ok=True)

    doc_file = att_dir / "1234abcd_analysis_report.pdf"
    doc_file.write_bytes(b"%PDF-1.4 sample content")

    doc_txt = att_dir / "5678efgh_notes.txt"
    doc_txt.write_text("Hello text", encoding="utf-8")

    client = TestClient(app)

    # 1. List documents
    res = client.get("/api/documents")
    assert res.status_code == 200
    docs = res.json()["documents"]
    assert len(docs) == 2

    pdf_entry = next((d for d in docs if d["filename"] == "analysis_report.pdf"), None)
    assert pdf_entry is not None
    assert pdf_entry["id"] == "1234abcd_analysis_report.pdf"
    assert pdf_entry["extension"] == ".pdf"
    assert pdf_entry["size_bytes"] == len(b"%PDF-1.4 sample content")

    # 2. Delete document
    del_res = client.delete(f"/api/documents/{pdf_entry['id']}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True
    assert not doc_file.exists()

    # 3. List again: only notes.txt remains
    res2 = client.get("/api/documents")
    assert res2.status_code == 200
    docs2 = res2.json()["documents"]
    assert len(docs2) == 1
    assert docs2[0]["filename"] == "notes.txt"


def test_documents_delete_prevents_path_traversal(isolated_ui_app, monkeypatch):
    """Verify /api/documents/{id} rejects traversal attempts outside attachments folder."""
    app, agent, tmp_path = isolated_ui_app
    monkeypatch.setattr("loki.utils.paths.get_data_dir", lambda: tmp_path)

    client = TestClient(app)

    # Path traversal attempts
    res1 = client.delete("/api/documents/..%2F..%2Fsecret.txt")
    assert res1.status_code in (400, 404)

    res2 = client.delete("/api/documents/some_file/../../etc/passwd")
    assert res2.status_code in (400, 404)

    # 404 for non-existent document
    res3 = client.delete("/api/documents/non_existent_doc.pdf")
    assert res3.status_code == 404


# =============================================================================
# 3. Settings & Memory-Awareness Endpoints Tests
# =============================================================================

def test_settings_get_and_update(isolated_ui_app):
    """Verify /api/settings returns and updates memory_enabled configuration."""
    app, agent, _ = isolated_ui_app
    client = TestClient(app)

    # Initial state
    res = client.get("/api/settings")
    assert res.status_code == 200
    assert res.json()["memory_enabled"] is True

    # Disable memory
    patch_res = client.patch("/api/settings", json={"memory_enabled": False})
    assert patch_res.status_code == 200
    assert patch_res.json()["memory_enabled"] is False
    assert agent.memory_enabled is False

    # Check updated state
    res2 = client.get("/api/settings")
    assert res2.json()["memory_enabled"] is False

    # Re-enable memory
    patch_res2 = client.patch("/api/settings", json={"memory_enabled": True})
    assert patch_res2.status_code == 200
    assert patch_res2.json()["memory_enabled"] is True
    assert agent.memory_enabled is True


def test_agent_memory_enabled_toggle_disables_memory_injection(isolated_ui_app):
    """Verify turning off memory_enabled completely disables memory injection in agent reasoning."""
    app, agent, _ = isolated_ui_app

    # Store a memory
    agent.memory_store.store_memory(
        category="preference",
        key="favorite_language",
        content="My favorite programming language is Python",
    )

    # 1. When memory_enabled is True: memory is injected into system prompt
    agent.memory_enabled = True
    agent.client.chat.reset_mock()
    res1 = agent.handle_input("What is my favorite language?")
    assert res1.source.value == "llm"
    called_messages1 = agent.client.chat.call_args[0][0]
    sys_prompt1 = called_messages1[0]["content"]
    assert "--- BEGIN UNTRUSTED USER MEMORY CONTEXT ---" in sys_prompt1
    assert "favorite programming language is Python" in sys_prompt1

    # 2. When memory_enabled is False: memory is NOT injected
    agent.memory_enabled = False
    agent.client.chat.reset_mock()
    res2 = agent.handle_input("What is my favorite language?")
    assert res2.source.value == "llm"
    called_messages2 = agent.client.chat.call_args[0][0]
    sys_prompt2 = called_messages2[0]["content"]
    assert "--- BEGIN UNTRUSTED USER MEMORY CONTEXT ---" not in sys_prompt2
    assert "favorite programming language is Python" not in sys_prompt2


# =============================================================================
# 4. Memory Management CRUD Endpoints Tests
# =============================================================================

def test_memories_crud_lifecycle(isolated_ui_app):
    """Verify full CRUD lifecycle via /api/memories endpoints."""
    app, agent, _ = isolated_ui_app
    client = TestClient(app)

    # 1. Create a memory
    create_res = client.post(
        "/api/memories",
        json={
            "content": "User prefers concise bullet-point summaries",
            "category": "preference",
            "key": "summary_style",
        },
    )
    assert create_res.status_code == 200
    created = create_res.json()["memory"]
    memory_id = created["id"]
    assert created["content"] == "User prefers concise bullet-point summaries"
    assert created["category"] == "preference"
    assert created["key"] == "summary_style"

    # 2. List memories
    list_res = client.get("/api/memories")
    assert list_res.status_code == 200
    data = list_res.json()
    assert data["count"] == 1
    assert data["memories"][0]["id"] == memory_id

    # 3. Search memories
    search_res = client.get("/api/memories?query=bullet")
    assert search_res.status_code == 200
    assert search_res.json()["count"] >= 1

    # 4. Update memory
    patch_res = client.patch(
        f"/api/memories/{memory_id}",
        json={"content": "User strongly prefers ultra-short responses"},
    )
    assert patch_res.status_code == 200
    updated = patch_res.json()["memory"]
    assert updated["content"] == "User strongly prefers ultra-short responses"

    # 5. Delete memory
    del_res = client.delete(f"/api/memories/{memory_id}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # 6. Verify empty list
    list_res2 = client.get("/api/memories")
    assert list_res2.json()["count"] == 0


def test_memories_create_validates_and_rejects_credentials(isolated_ui_app):
    """Verify /api/memories rejects empty content and sensitive security tokens."""
    app, agent, _ = isolated_ui_app
    client = TestClient(app)

    # Empty content
    res_empty = client.post("/api/memories", json={"content": "   ", "category": "preference"})
    assert res_empty.status_code in (400, 422)

    # Sensitive API key
    res_key = client.post(
        "/api/memories",
        json={"content": "My API key is sk-123456789012345678901234567890", "category": "preference"},
    )
    assert res_key.status_code == 400
    assert "sensitive" in res_key.json()["detail"].lower()


def test_memories_clear_endpoint(isolated_ui_app):
    """Verify POST /api/memories/clear wipes stored memories cleanly."""
    app, agent, _ = isolated_ui_app
    client = TestClient(app)

    # Store 2 memories
    client.post("/api/memories", json={"content": "Fact 1", "category": "fact"})
    client.post("/api/memories", json={"content": "Fact 2", "category": "fact"})

    list_res = client.get("/api/memories")
    assert list_res.json()["count"] == 2

    # Clear all
    clear_res = client.post("/api/memories/clear")
    assert clear_res.status_code == 200
    assert clear_res.json()["success"] is True
    assert clear_res.json()["deleted_count"] == 2

    # Verify 0
    list_res2 = client.get("/api/memories")
    assert list_res2.json()["count"] == 0
