"""Automated tests for Phase 5A: Conversational Quality & Security Evaluation."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from loki.agent import AgentResponse, LokiAgent, ResponseSource
from loki.commands.router import DeterministicRouter
from loki.core.config import LokiConfig, ModelRuntimeConfig, SafetyConfig
from loki.core.exceptions import SafetyViolationError
from loki.core.types import SafetyLevel, ToolResult
from loki.llm.client import OllamaClient
from loki.llm.resolver import ModelResolutionResult, ModelResolver
from loki.safety.gatekeeper import SafetyGatekeeper


def _is_local_qwen_available() -> bool:
    """Check if local Ollama daemon is reachable and has qwen2.5:0.5b installed."""
    try:
        client = OllamaClient()
        return client.check_health() and client.is_model_available("qwen2.5:0.5b")
    except Exception:
        return False


# =========================================================================
# Part A: Conversational Tests
# =========================================================================


def test_conversational_factual_question():
    """Evaluate simple factual query through LokiAgent."""
    config = LokiConfig()
    config.models.active_model = "qwen2.5:0.5b"
    config.models.fallback_models = []

    if _is_local_qwen_available():
        agent = LokiAgent(config=config, allow_fallback=False)
    else:
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.host = "http://127.0.0.1:11434"
        mock_client.check_health.return_value = True
        mock_client.list_local_models.return_value = ["qwen2.5:0.5b"]
        mock_client.is_model_available.return_value = True
        mock_client.chat.return_value = "The capital of France is Paris."
        agent = LokiAgent(config=config, client=mock_client, allow_fallback=False)

    res = agent.handle_input("What is the capital of France?")

    assert res.success is True
    assert res.source == ResponseSource.LLM
    assert res.model_used == "qwen2.5:0.5b"
    assert len(res.text.strip()) > 0
    assert "Paris" in res.text


def test_conversational_arithmetic_question():
    """Evaluate basic arithmetic query through LokiAgent."""
    config = LokiConfig()
    config.models.active_model = "qwen2.5:0.5b"
    config.models.fallback_models = []

    if _is_local_qwen_available():
        agent = LokiAgent(config=config, allow_fallback=False)
    else:
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.host = "http://127.0.0.1:11434"
        mock_client.check_health.return_value = True
        mock_client.list_local_models.return_value = ["qwen2.5:0.5b"]
        mock_client.is_model_available.return_value = True
        mock_client.chat.return_value = "25 + 17 equals 42."
        agent = LokiAgent(config=config, client=mock_client, allow_fallback=False)

    res = agent.handle_input("What is 25 + 17?")

    assert res.success is True
    assert res.source == ResponseSource.LLM
    assert res.model_used == "qwen2.5:0.5b"
    assert "42" in res.text


def test_conversational_code_explanation():
    """Evaluate code explanation prompt through LokiAgent."""
    config = LokiConfig()
    config.models.active_model = "qwen2.5:0.5b"
    config.models.fallback_models = []

    if _is_local_qwen_available():
        agent = LokiAgent(config=config, allow_fallback=False)
    else:
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.host = "http://127.0.0.1:11434"
        mock_client.check_health.return_value = True
        mock_client.list_local_models.return_value = ["qwen2.5:0.5b"]
        mock_client.is_model_available.return_value = True
        mock_client.chat.return_value = "This list comprehension multiplies each element by 2, resulting in [2, 4, 6]."
        agent = LokiAgent(config=config, client=mock_client, allow_fallback=False)

    res = agent.handle_input("Explain what this Python code does: [x * 2 for x in [1, 2, 3]]")

    assert res.success is True
    assert res.source == ResponseSource.LLM
    assert res.model_used == "qwen2.5:0.5b"
    assert len(res.text.strip()) > 0


def test_conversational_multiturn_context_retention():
    """Evaluate multi-turn context retention across conversation turns."""
    config = LokiConfig()
    config.models.active_model = "qwen2.5:0.5b"
    config.models.fallback_models = []

    if _is_local_qwen_available():
        agent = LokiAgent(config=config, allow_fallback=False)
    else:
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.host = "http://127.0.0.1:11434"
        mock_client.check_health.return_value = True
        mock_client.list_local_models.return_value = ["qwen2.5:0.5b"]
        mock_client.is_model_available.return_value = True
        mock_client.chat.side_effect = [
            "Emerald green is a lovely shade of green.",
            "Your favorite color is emerald green.",
        ]
        agent = LokiAgent(config=config, client=mock_client, allow_fallback=False)

    agent.clear_history()

    # Turn 1
    res1 = agent.handle_input("My favorite color is emerald green.")
    assert res1.success is True
    assert res1.source == ResponseSource.LLM

    # Turn 2
    res2 = agent.handle_input("What is my favorite color?")
    assert res2.success is True
    assert res2.source == ResponseSource.LLM

    history = agent.get_history()
    # 2 turns * 2 messages (user + assistant) = 4 messages
    assert len(history) == 4
    assert history[0]["content"] == "My favorite color is emerald green."
    assert history[2]["content"] == "What is my favorite color?"


def test_conversational_empty_whitespace_input():
    """Evaluate that empty and whitespace-only inputs are safely rejected."""
    agent = LokiAgent()

    for empty_prompt in ["", "   ", "\t\t\n  "]:
        res = agent.handle_input(empty_prompt)
        assert res.success is False
        assert res.source == ResponseSource.SYSTEM
        assert "Please provide a command or question" in res.text


def test_conversational_unavailable_model_mocked():
    """Verify missing model returns safe diagnostic response without silent downloads."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.list_local_models.return_value = ["different-model:latest"]
    mock_client.is_model_available.return_value = False

    config = LokiConfig()
    config.models.active_model = "uninstalled-model:latest"
    config.models.fallback_models = []

    agent = LokiAgent(config=config, client=mock_client, allow_fallback=False)
    res = agent.handle_input("Write a poem about the sunrise")

    assert res.success is False
    assert res.source == ResponseSource.SYSTEM
    assert "is not installed in local Ollama library" in res.text
    assert "ollama pull" in res.text
    mock_client.chat.assert_not_called()
    mock_client.generate.assert_not_called()


# =========================================================================
# Part B: Deterministic Routing and Security Tests
# =========================================================================


def test_security_deterministic_routing_bypasses_llm():
    """Verify registered commands bypass the LLM entirely with zero LLM interaction."""
    mock_client = MagicMock(spec=OllamaClient)

    agent = LokiAgent(client=mock_client)
    res = agent.handle_input("what is my battery percentage")

    assert res.success is True
    assert res.source == ResponseSource.DETERMINISTIC
    assert res.matched_command == "system_info"
    assert res.tool_result is not None
    # Verify zero LLM calls
    mock_client.chat.assert_not_called()
    mock_client.generate.assert_not_called()


def test_security_unmatched_prompt_escalates_to_llm():
    """Verify non-command queries escalate cleanly to local LLM."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.list_local_models.return_value = ["qwen2.5:3b"]
    mock_client.is_model_available.return_value = True
    mock_client.chat.return_value = "The ocean is vast and deep."

    agent = LokiAgent(client=mock_client)
    res = agent.handle_input("Tell me about the ocean")

    assert res.success is True
    assert res.source == ResponseSource.LLM
    assert res.matched_command is None
    mock_client.chat.assert_called_once()


def test_security_destructive_actions_gated_by_confirmation(tmp_path: Path):
    """Verify destructive actions fail closed until explicit user confirmation."""
    safe_root = tmp_path / "ControlledUserArea"
    safe_root.mkdir()
    target_file = safe_root / "temp_to_delete.txt"
    target_file.write_text("critical data", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    mock_client = MagicMock(spec=OllamaClient)
    agent = LokiAgent(gatekeeper=gk, client=mock_client)

    # 1. First execution attempt without token -> must return pending confirmation
    res1 = agent.handle_input(f"delete {target_file}")
    assert res1.source == ResponseSource.SYSTEM
    assert res1.pending_confirmation is not None
    assert res1.pending_confirmation.confirmed is False
    assert target_file.exists()  # File must remain untouched
    action_id = res1.pending_confirmation.action_id

    # 2. Approve confirmation via agent
    assert agent.confirm_action(action_id) is True

    # 3. Second execution attempt with valid confirmed token -> succeeds
    res2 = agent.handle_input(f"delete {target_file}", action_id=action_id)
    assert res2.source == ResponseSource.DETERMINISTIC
    assert res2.success is True
    assert not target_file.exists()


def test_security_confirmation_tokens_bound_to_command_and_target(tmp_path: Path):
    """Verify confirmation tokens cannot be transferred to different tools or different targets."""
    safe_root = tmp_path / "BoundArea"
    safe_root.mkdir()
    file_a = safe_root / "authorized.txt"
    file_b = safe_root / "unauthorized.txt"
    file_a.write_text("a", encoding="utf-8")
    file_b.write_text("b", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])
    req = gk.request_confirmation(
        tool_name="file_delete",
        description="Delete file A",
        targets=[str(file_a)],
    )
    assert gk.confirm_action(req.action_id) is True

    # Attempt to use file A's token for file B -> must be rejected
    with pytest.raises(SafetyViolationError) as exc_target:
        gk.verify_execution_permitted(
            tool_name="file_delete",
            safety_level=SafetyLevel.DANGEROUS,
            action_id=req.action_id,
            targets=[str(file_b)],
        )
    assert "Target mismatch" in str(exc_target.value)

    # Request new token for file A
    req2 = gk.request_confirmation(
        tool_name="file_delete",
        description="Delete file A",
        targets=[str(file_a)],
    )
    assert gk.confirm_action(req2.action_id) is True

    # Attempt to use file_delete token for format_drive -> must be rejected
    with pytest.raises(SafetyViolationError) as exc_tool:
        gk.verify_execution_permitted(
            tool_name="format_drive",
            safety_level=SafetyLevel.DANGEROUS,
            action_id=req2.action_id,
            targets=[str(file_a)],
        )
    assert "was authorized for 'file_delete', not 'format_drive'" in str(exc_tool.value)


def test_security_llm_text_never_executed_as_command_or_shell(tmp_path: Path):
    """Critical Invariant: Model responses containing shell commands or file operations are NEVER executed."""
    victim_file = tmp_path / "protected_passwords.txt"
    victim_file.write_text("secret_master_key", encoding="utf-8")

    # Mock the LLM to output what appears to be command strings
    malicious_llm_output = (
        f"Sure! Executing: delete {victim_file}\n"
        "rmdir /s /q C:\\Windows\n"
        "launch notepad.exe"
    )

    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.list_local_models.return_value = ["qwen2.5:3b"]
    mock_client.is_model_available.return_value = True
    mock_client.chat.return_value = malicious_llm_output

    with patch("loki.commands.router.DeterministicRouter.route") as mock_route:
        mock_route.return_value = MagicMock(matched=False)

        agent = LokiAgent(client=mock_client)
        res = agent.handle_input("Can you clean up temporary files?")

        assert res.source == ResponseSource.LLM
        assert res.text == malicious_llm_output
        # Invariant 1: File remains intact
        assert victim_file.exists()
        # Invariant 2: Router was only called for the initial user input, never for LLM output
        assert mock_route.call_count == 1
        mock_route.assert_called_once_with("Can you clean up temporary files?", action_id=None)


def test_security_unavailable_ollama_server_produces_safe_error():
    """Verify offline Ollama daemon produces clear diagnostic message without crashing."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = False

    agent = LokiAgent(client=mock_client)
    res = agent.handle_input("Hello, can you assist me?")

    assert res.success is False
    assert res.source == ResponseSource.SYSTEM
    assert "offline or unreachable" in res.text
    mock_client.chat.assert_not_called()
    mock_client.generate.assert_not_called()
