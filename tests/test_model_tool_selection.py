"""Tests for Phase 8 model-assisted tool selection (Stage 0.5).

Verifies:
1. Strict tool allowlisting (only system_info and file_search; never file_delete or app_launcher).
2. JSON parsing, schema validation, and fail-closed rejection of unknown/malformed output.
3. Path boundaries and control character rejection.
4. End-to-end LokiAgent execution through Stage 0.5.
5. Invariant: Model responses cannot execute destructive actions or bypass gatekeeper.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from loki.agent import LokiAgent, ResponseSource
from loki.commands.builtin.app_launcher import AppLauncherCommand
from loki.commands.builtin.file_delete import FileDeleteCommand
from loki.commands.builtin.file_search import FileSearchCommand
from loki.commands.builtin.system_info import SystemInfoCommand
from loki.commands.model_selector import (
    ModelToolProposal,
    build_tool_selection_prompt,
    extract_json_object,
    parse_and_validate_tool_call,
    select_tool,
)
from loki.commands.registry import CommandRegistry
from loki.core.config import LokiConfig
from loki.core.exceptions import ModelInferenceError
from loki.core.types import SafetyLevel
from loki.llm.client import OllamaClient
from loki.safety.gatekeeper import SafetyGatekeeper


# =========================================================================
# 1. Allowlist and Registry Invariants
# =========================================================================

def test_registry_model_selectable_commands_allowlist():
    """Only SAFE, explicitly approved read-only tools can be selected by the model."""
    registry = CommandRegistry()
    selectable = registry.get_model_selectable_commands()

    # Allowed read-only tools
    assert "system_info" in selectable
    assert "file_search" in selectable

    # Strictly forbidden dangerous or app launching tools
    assert "file_delete" not in selectable
    assert "app_launcher" not in selectable

    # Verify every selectable tool has SAFE level and allow_model_selection=True
    for name, cmd in selectable.items():
        assert cmd.safety_level == SafetyLevel.SAFE
        assert cmd.allow_model_selection is True

    # Check explicit flags on non-selectable commands
    del_cmd = registry.get("file_delete")
    assert del_cmd is not None
    assert del_cmd.allow_model_selection is False

    app_cmd = registry.get("app_launcher")
    assert app_cmd is not None
    assert app_cmd.allow_model_selection is False


# =========================================================================
# 2. Schema and Argument Validation
# =========================================================================

def test_system_info_schema_and_validation():
    cmd = SystemInfoCommand()
    schema = cmd.get_parameters_schema()

    assert schema["type"] == "object"
    assert "metric" in schema["properties"]
    assert "all" in schema["properties"]["metric"]["enum"]
    assert schema["additionalProperties"] is False

    # Valid metrics
    for m in ["battery", "ram", "cpu", "storage", "all"]:
        valid, err = cmd.validate_args({"metric": m})
        assert valid is True
        assert err is None

    # Invalid metric
    valid, err = cmd.validate_args({"metric": "gpu_overclock"})
    assert valid is False
    assert "Invalid metric" in err

    # Missing required metric
    valid, err = cmd.validate_args({})
    assert valid is False
    assert "Missing required argument 'metric'" in err

    # Unexpected extra arguments rejected
    valid, err = cmd.validate_args({"metric": "ram", "extra_param": "malicious"})
    assert valid is False
    assert "Unexpected arguments" in err

    # Non-dict rejected
    valid, err = cmd.validate_args(["metric", "ram"])
    assert valid is False


def test_file_search_schema_and_validation():
    cmd = FileSearchCommand()
    schema = cmd.get_parameters_schema()

    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert schema["additionalProperties"] is False

    # Valid query and location
    valid, err = cmd.validate_args({"query": "budget_report.pdf", "location": "documents"})
    assert valid is True

    valid, err = cmd.validate_args({"query": "notes"})
    assert valid is True

    # Empty query rejected
    valid, err = cmd.validate_args({"query": ""})
    assert valid is False
    assert "non-empty string" in err

    # Query over 200 characters rejected
    valid, err = cmd.validate_args({"query": "a" * 201})
    assert valid is False
    assert "exceeds maximum allowed length" in err

    # Control characters in query rejected
    valid, err = cmd.validate_args({"query": "notes\x00secret"})
    assert valid is False
    assert "invalid control characters" in err

    # Location over 260 characters rejected
    valid, err = cmd.validate_args({"query": "notes", "location": "C:\\" + "a" * 260})
    assert valid is False
    assert "exceeds maximum allowed path length" in err

    # Extra arguments rejected
    valid, err = cmd.validate_args({"query": "notes", "delete_after": True})
    assert valid is False
    assert "Unexpected arguments" in err


# =========================================================================
# 3. JSON Extraction and Tool Call Parsing
# =========================================================================

def test_extract_json_object():
    # Raw JSON
    assert extract_json_object('{"tool": "system_info", "arguments": {"metric": "cpu"}}') == {
        "tool": "system_info",
        "arguments": {"metric": "cpu"},
    }

    # Markdown code fences
    fenced = '```json\n{"tool": "none"}\n```'
    assert extract_json_object(fenced) == {"tool": "none"}

    # Text preamble and postamble
    wrapped = 'Here is the selection:\n{"tool": "system_info", "arguments": {"metric": "ram"}}\nHope this helps!'
    assert extract_json_object(wrapped) == {"tool": "system_info", "arguments": {"metric": "ram"}}

    # Malformed text
    assert extract_json_object("Just a normal text response without JSON.") is None
    assert extract_json_object("{malformed: json, missing quotes}") is None
    assert extract_json_object("") is None


def test_parse_and_validate_tool_call_valid():
    selectable = {
        "system_info": SystemInfoCommand(),
        "file_search": FileSearchCommand(),
    }

    # Valid system_info
    raw = '{"tool": "system_info", "arguments": {"metric": "battery"}}'
    proposal, err = parse_and_validate_tool_call(
        raw, selectable, user_input="How is the battery percentage?"
    )
    assert err is None
    assert proposal is not None
    assert proposal.tool_name == "system_info"
    assert proposal.arguments == {"metric": "battery"}

    # Valid file_search
    raw_search = '{"tool": "file_search", "arguments": {"query": "invoice", "location": "downloads"}}'
    proposal, err = parse_and_validate_tool_call(
        raw_search, selectable, user_input="Find my invoice file in downloads"
    )
    assert err is None
    assert proposal is not None
    assert proposal.tool_name == "file_search"
    assert proposal.arguments == {"query": "invoice", "location": "downloads"}


def test_parse_and_validate_tool_call_none_conversational():
    selectable = {"system_info": SystemInfoCommand()}
    raw = '{"tool": "none"}'
    proposal, err = parse_and_validate_tool_call(raw, selectable, user_input="Hello!")
    assert proposal is None
    assert err is None  # Deliberate conversational intent


def test_parse_and_validate_tool_call_rejects_unallowlisted_tools():
    selectable = {
        "system_info": SystemInfoCommand(),
        "file_search": FileSearchCommand(),
    }

    # Attempt to select file_delete
    raw_del = r'{"tool": "file_delete", "arguments": {"path": "C:\\Windows\\System32"}}'
    proposal, err = parse_and_validate_tool_call(raw_del, selectable, user_input="delete everything")
    assert proposal is None
    assert "not in the allowlist" in err

    # Attempt to select app_launcher
    raw_app = '{"tool": "app_launcher", "arguments": {"app_name": "cmd.exe"}}'
    proposal, err = parse_and_validate_tool_call(raw_app, selectable, user_input="launch cmd")
    assert proposal is None
    assert "not in the allowlist" in err

    # Arbitrary command
    raw_arb = '{"tool": "shell_exec", "arguments": {"command": "dir"}}'
    proposal, err = parse_and_validate_tool_call(raw_arb, selectable, user_input="run dir")
    assert proposal is None
    assert "not in the allowlist" in err


def test_parse_and_validate_tool_call_rejects_hallucinations_on_unrelated_queries():
    selectable = {"system_info": SystemInfoCommand()}

    # Arithmetic question should not accept hallucinated battery query
    raw = '{"tool": "system_info", "arguments": {"metric": "battery"}}'
    proposal, err = parse_and_validate_tool_call(
        raw, selectable, user_input="What is 25 + 17?"
    )
    assert proposal is None
    assert "contains no relevant hardware keywords" in err

    # Personal question should not accept hallucinated ram query
    raw_color = '{"tool": "system_info", "arguments": {"metric": "ram"}}'
    proposal, err = parse_and_validate_tool_call(
        raw_color, selectable, user_input="What is my favorite color?"
    )
    assert proposal is None
    assert "contains no relevant hardware keywords" in err


def test_parse_and_validate_tool_call_gatekeeper_path_checks(tmp_path: Path):
    safe_root = tmp_path / "SafeDir"
    safe_root.mkdir()
    protected_root = tmp_path / "ProtectedCore"
    protected_root.mkdir()

    gk = SafetyGatekeeper(
        protected_paths=[protected_root],
        allowed_search_roots=[safe_root],
    )
    selectable = {"file_search": FileSearchCommand()}

    # Safe search inside safe_root
    raw_safe = f'{{"tool": "file_search", "arguments": {{"query": "notes", "location": "{str(safe_root).replace(chr(92), "/")}"}}}}'
    proposal, err = parse_and_validate_tool_call(
        raw_safe, selectable, gatekeeper=gk, user_input="Find my notes file"
    )
    assert err is None
    assert proposal is not None

    # Search in protected directory rejected
    raw_bad = f'{{"tool": "file_search", "arguments": {{"query": "secrets", "location": "{str(protected_root).replace(chr(92), "/")}"}}}}'
    proposal, err = parse_and_validate_tool_call(
        raw_bad, selectable, gatekeeper=gk, user_input="Search for secrets file"
    )
    assert proposal is None
    assert "restricted or outside safe read roots" in err


# =========================================================================
# 4. select_tool Client Integration
# =========================================================================

def test_select_tool_successful_selection():
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.generate.return_value = '{"tool": "system_info", "arguments": {"metric": "cpu"}}'

    selectable = {"system_info": SystemInfoCommand()}
    proposal = select_tool(
        user_input="How heavily is the cpu loaded right now?",
        client=mock_client,
        model="qwen2.5:0.5b",
        selectable_commands=selectable,
    )

    assert proposal is not None
    assert proposal.tool_name == "system_info"
    assert proposal.arguments == {"metric": "cpu"}
    mock_client.generate.assert_called_once()


def test_select_tool_model_error_fails_closed():
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.generate.side_effect = ModelInferenceError("Ollama connection timed out")

    selectable = {"system_info": SystemInfoCommand()}
    proposal = select_tool(
        user_input="what is my battery level?",
        client=mock_client,
        model="qwen2.5:0.5b",
        selectable_commands=selectable,
    )

    # Must fail closed without crashing
    assert proposal is None


# =========================================================================
# 5. End-to-End LokiAgent Stage 0.5 Execution
# =========================================================================

def test_agent_stage05_executes_system_info():
    """Verify that input not matching regex routes through Stage 0.5 and executes system_info."""
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.list_local_models.return_value = ["qwen2.5:0.5b"]
    mock_client.is_model_available.return_value = True
    mock_client.generate.return_value = '{"tool": "system_info", "arguments": {"metric": "ram"}}'

    agent = LokiAgent(client=mock_client)

    # Input phrased naturally that does not match standard deterministic regex patterns
    res = agent.handle_input("Please inspect the current memory consumption of the system.")

    assert res.success is True
    assert res.source == ResponseSource.TOOL
    assert res.matched_command == "system_info"
    assert res.tool_result is not None
    assert "ram" in res.tool_result.data
    assert "RAM Usage:" in res.text
    assert res.metadata.get("model_assisted") is True


def test_agent_stage05_executes_file_search(tmp_path: Path):
    """Verify that file search requested in natural language routes through Stage 0.5."""
    safe_root = tmp_path / "UserDocuments"
    safe_root.mkdir()
    (safe_root / "quantum_mechanics.pdf").write_text("physics notes", encoding="utf-8")

    gk = SafetyGatekeeper(allowed_search_roots=[safe_root])

    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.list_local_models.return_value = ["qwen2.5:0.5b"]
    mock_client.is_model_available.return_value = True
    mock_client.generate.return_value = (
        f'{{"tool": "file_search", "arguments": {{"query": "quantum", "location": "{str(safe_root).replace(chr(92), "/")}"}}}}'
    )

    agent = LokiAgent(client=mock_client, gatekeeper=gk)

    # Input phrased naturally without regex trigger prefixes (e.g. without 'find', 'search for', 'locate')
    res = agent.handle_input("Can you retrieve the quantum mechanics document from my folder?")

    assert res.success is True
    assert res.source == ResponseSource.TOOL
    assert res.matched_command == "file_search"
    assert res.tool_result is not None
    assert res.tool_result.data["matches_count"] == 1
    assert "quantum_mechanics.pdf" in res.tool_result.data["matches"][0]["filename"]


def test_agent_stage05_rejects_destructive_proposal_and_falls_through_to_chat(tmp_path: Path):
    """Invariant: If model proposes a destructive tool, Stage 0.5 rejects it and falls through to LLM chat."""
    victim_file = tmp_path / "important_data.csv"
    victim_file.write_text("critical", encoding="utf-8")

    mock_client = MagicMock(spec=OllamaClient)
    mock_client.host = "http://127.0.0.1:11434"
    mock_client.check_health.return_value = True
    mock_client.list_local_models.return_value = ["qwen2.5:0.5b"]
    mock_client.is_model_available.return_value = True
    # Model attempts to select file_delete
    mock_client.generate.return_value = f'{{"tool": "file_delete", "arguments": {{"path": "{str(victim_file).replace(chr(92), "/")}"}}}}'
    mock_client.chat.return_value = "Here is a story about Thor and Loki."

    agent = LokiAgent(client=mock_client)

    res = agent.handle_input("Tell me an interesting story about ancient Norse mythology.")

    # Invariant: File must never have been deleted
    assert victim_file.exists()
    # Response fell through to ordinary chat
    assert res.source == ResponseSource.LLM
    assert res.text == "Here is a story about Thor and Loki."
