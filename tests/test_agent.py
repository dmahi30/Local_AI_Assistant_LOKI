"""Unit tests for LokiAgent with full mocking of external and local dependencies."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from loki.agent import AgentResponse, LokiAgent, ResponseSource
from loki.commands.router import DeterministicRouter
from loki.core.config import LokiConfig, ModelRuntimeConfig, SafetyConfig
from loki.core.exceptions import ModelInferenceError
from loki.core.types import SafetyLevel, ToolResult
from loki.llm.client import OllamaClient
from loki.llm.resolver import ModelResolutionResult, ModelResolver
from loki.safety.gatekeeper import SafetyGatekeeper


@pytest.fixture
def mock_ollama_client() -> MagicMock:
    client = MagicMock(spec=OllamaClient)
    client.host = "http://127.0.0.1:11434"
    client.check_health.return_value = True
    client.list_local_models.return_value = ["qwen2.5:3b", "qwen2.5:1.5b"]
    client.is_model_available.side_effect = lambda m: m in ["qwen2.5:3b", "qwen2.5:1.5b"]
    client.chat.return_value = "Mocked LLM conversational reply."
    client.generate.return_value = "Mocked LLM single-turn response."
    return client


def test_agent_empty_input(mock_ollama_client):
    agent = LokiAgent(client=mock_ollama_client)
    res = agent.handle_input("   ")
    assert res.success is False
    assert res.source == ResponseSource.SYSTEM
    assert "Please provide a command" in res.text
    mock_ollama_client.chat.assert_not_called()
    mock_ollama_client.generate.assert_not_called()


def test_agent_deterministic_bypass(mock_ollama_client):
    """Verify that supported OS commands bypass LLM completely with zero LLM calls."""
    agent = LokiAgent(client=mock_ollama_client)

    res = agent.handle_input("what is my battery percentage")

    assert res.success is True
    assert res.source == ResponseSource.DETERMINISTIC
    assert res.matched_command == "system_info"
    assert res.tool_result is not None
    assert "battery" in res.tool_result.data
    # Invariant: LLM was never contacted
    mock_ollama_client.chat.assert_not_called()
    mock_ollama_client.generate.assert_not_called()


def test_agent_gated_action_confirmation_flow(tmp_path: Path, mock_ollama_client):
    """Verify destructive command triggers pending confirmation and executes only after approval."""
    safe_root = tmp_path / "AgentStorage"
    safe_root.mkdir()
    target_file = safe_root / "obsolete.log"
    target_file.write_text("temporary log content", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    agent = LokiAgent(gatekeeper=gk, client=mock_ollama_client)

    # 1. Unconfirmed attempt: should return pending confirmation
    res1 = agent.handle_input(f"delete {target_file}")
    assert res1.source == ResponseSource.SYSTEM
    assert res1.pending_confirmation is not None
    assert res1.matched_command == "file_delete"
    assert target_file.exists()
    action_id = res1.pending_confirmation.action_id

    # Invariant: LLM was never contacted
    mock_ollama_client.chat.assert_not_called()

    # 2. Approve via agent
    assert agent.confirm_action(action_id) is True

    # 3. Execution with confirmed token
    res2 = agent.handle_input(f"delete {target_file}", action_id=action_id)
    assert res2.source == ResponseSource.DETERMINISTIC
    assert res2.success is True
    assert not target_file.exists()


def test_agent_conversational_chat(mock_ollama_client):
    """Non-deterministic conversational input escalates to local Ollama chat."""
    agent = LokiAgent(client=mock_ollama_client)

    res = agent.handle_input("Explain how photosynthesis works")

    assert res.success is True
    assert res.source == ResponseSource.LLM
    assert res.text == "Mocked LLM conversational reply."
    assert res.model_used == "qwen2.5:3b"

    # Context history should contain the user turn and the assistant turn
    history = agent.get_history()
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Explain how photosynthesis works"}
    assert history[1] == {"role": "assistant", "content": "Mocked LLM conversational reply."}


def test_agent_conversational_generate(mock_ollama_client):
    """Single-turn generation when chat_mode=False."""
    agent = LokiAgent(client=mock_ollama_client)

    res = agent.handle_input("Summarize the project goals", chat_mode=False)

    assert res.success is True
    assert res.source == ResponseSource.LLM
    assert res.text == "Mocked LLM single-turn response."
    mock_ollama_client.generate.assert_called_once()


def test_agent_ollama_offline_handling():
    """Agent returns clear diagnostic message when Ollama daemon is offline."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = False

    agent = LokiAgent(client=mock_client)
    res = agent.handle_input("Hello LOKI, how are you?")

    assert res.success is False
    assert res.source == ResponseSource.SYSTEM
    assert "offline or unreachable" in res.text
    mock_client.chat.assert_not_called()


def test_agent_model_missing_handling():
    """Agent notifies user when model is missing without attempting unapproved downloads."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.list_local_models.return_value = ["different-model:latest"]
    mock_client.is_model_available.return_value = False

    agent = LokiAgent(client=mock_client)
    res = agent.handle_input("Write a short story")

    assert res.success is False
    assert res.source == ResponseSource.SYSTEM
    assert "is not installed in local Ollama library" in res.text
    assert "ollama pull" in res.text
    mock_client.chat.assert_not_called()


def test_agent_strict_separation_no_llm_execution(mock_ollama_client, tmp_path: Path):
    """Security Invariant: LLM responses containing malicious command text are NEVER executed."""
    victim_file = tmp_path / "sensitive_passwords.txt"
    victim_file.write_text("super_secret", encoding="utf-8")

    # Mock the LLM to output what looks like malicious commands
    malicious_output = (
        f"Sure! I will run: delete {victim_file}\n"
        "And also: rmdir /s /q C:\\Windows\n"
        "And launch notepad.exe"
    )
    mock_ollama_client.chat.return_value = malicious_output

    with patch("loki.commands.router.DeterministicRouter.route") as mock_route:
        # First call for user input returns unmatched -> escalates to LLM
        mock_route.return_value = MagicMock(matched=False)

        agent = LokiAgent(client=mock_ollama_client)
        res = agent.handle_input("Can you clean up my system?")

        assert res.source == ResponseSource.LLM
        assert res.text == malicious_output
        # Invariant 1: victim file must NOT have been touched
        assert victim_file.exists()
        # Invariant 2: DeterministicRouter must ONLY have been called once for user input,
        # NEVER for the LLM's returned output
        assert mock_route.call_count == 1
        mock_route.assert_called_once_with("Can you clean up my system?", action_id=None)


def test_agent_llm_inference_error_handling(mock_ollama_client):
    """Inference errors are caught gracefully and do not corrupt conversation context."""
    mock_ollama_client.chat.side_effect = ModelInferenceError("Connection reset by peer")

    agent = LokiAgent(client=mock_ollama_client)
    res = agent.handle_input("Tell me a joke")

    assert res.success is False
    assert res.source == ResponseSource.SYSTEM
    assert "AI inference error" in res.text

    # Conversation history should not contain the failed user message
    assert len(agent.get_history()) == 0


def test_agent_model_fallback_when_enabled():
    """Agent uses approved fallback model when enabled."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.list_local_models.return_value = ["qwen2.5:1.5b"]

    def fake_available(m: str) -> bool:
        return m == "qwen2.5:1.5b"

    mock_client.is_model_available.side_effect = fake_available
    mock_client.chat.return_value = "Fallback response from 1.5b"

    cfg = LokiConfig()
    cfg.models.active_model = "qwen2.5:3b"
    cfg.models.fallback_models = ["qwen2.5:1.5b"]

    agent = LokiAgent(config=cfg, client=mock_client, allow_fallback=True)
    res = agent.handle_input("Help me write an essay")

    assert res.success is True
    assert res.source == ResponseSource.LLM
    assert res.model_used == "qwen2.5:1.5b"
    assert res.metadata.get("using_fallback") is True


def test_agent_clear_history(mock_ollama_client):
    agent = LokiAgent(client=mock_ollama_client)
    agent.handle_input("Question 1")
    assert len(agent.get_history()) == 2
    agent.clear_history()
    assert len(agent.get_history()) == 0
