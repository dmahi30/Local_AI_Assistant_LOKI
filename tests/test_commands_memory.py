"""Tests for LOKI Phase 11.2: Deterministic Memory Management Commands.

Verifies:
- Command parsing, regex matching, argument extraction, and validation.
- Deterministic storage of preferences and facts without Ollama.
- Listing stored memories and empty-state messaging.
- Specific memory forget with single-match deletion, multiple-match disambiguation,
  and zero-match reporting.
- SafetyGatekeeper confirmation flow for 'Forget everything you remember' (DANGEROUS tier).
- Zero Ollama / LLM invocation across all memory commands.
- Cross-conversation access (user-scoped memory persistence).
- Security sanitization and credential/secret rejection.
- Model selection exclusion (allow_model_selection=False).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from loki.agent import AgentResponse, LokiAgent, ResponseSource
from loki.commands.builtin.memory import MemoryCommand, _infer_category
from loki.commands.registry import CommandRegistry
from loki.commands.router import DeterministicRouter
from loki.core.exceptions import SafetyViolationError
from loki.core.types import SafetyLevel
from loki.llm.client import OllamaClient
from loki.llm.resolver import ModelResolutionResult, ModelResolver
from loki.safety.gatekeeper import SafetyGatekeeper
from loki.storage.conversation import ConversationStore
from loki.storage.memory import MemoryCategory, MemorySourceType, MemoryStore


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Provide a path to a temporary loki.db."""
    return tmp_path / "loki_test.db"


@pytest.fixture
def memory_store(temp_db: Path) -> MemoryStore:
    """Provide an isolated MemoryStore."""
    return MemoryStore(db_path=temp_db)


@pytest.fixture
def memory_command(memory_store: MemoryStore) -> MemoryCommand:
    """Provide a MemoryCommand instance bound to an isolated store."""
    return MemoryCommand(memory_store=memory_store)


@pytest.fixture
def mock_client() -> MagicMock:
    """Provide a mock OllamaClient to verify zero LLM invocation."""
    client = MagicMock(spec=OllamaClient)
    client.chat.side_effect = AssertionError("OllamaClient.chat was unexpectedly called for a deterministic memory command!")
    client.generate.side_effect = AssertionError("OllamaClient.generate was unexpectedly called for a deterministic memory command!")
    client.list_local_models.return_value = ["qwen2.5:0.5b"]
    return client


@pytest.fixture
def mock_resolver() -> MagicMock:
    """Provide a mock ModelResolver."""
    resolver = MagicMock(spec=ModelResolver)
    resolver.resolve.return_value = ModelResolutionResult(
        is_ready=True,
        effective_model="qwen2.5:0.5b",
        installed_models=["qwen2.5:0.5b"],
        ollama_online=True,
        message="Model ready",
        using_fallback=False,
    )
    return resolver


@pytest.fixture
def agent(temp_db: Path, memory_store: MemoryStore, mock_client: MagicMock, mock_resolver: MagicMock) -> LokiAgent:
    """Provide a fully wired LokiAgent with mock LLM and isolated storage."""
    storage = ConversationStore(db_path=temp_db)
    gatekeeper = SafetyGatekeeper()
    registry = CommandRegistry(memory_store=memory_store)
    router = DeterministicRouter(registry=registry, gatekeeper=gatekeeper)
    return LokiAgent(
        router=router,
        gatekeeper=gatekeeper,
        client=mock_client,
        resolver=mock_resolver,
        storage=storage,
        memory_store=memory_store,
    )


# =============================================================================
# 1. Regex & Pattern Matching Tests
# =============================================================================


def test_command_metadata_and_security_invariants(memory_command: MemoryCommand):
    """Ensure command is classified as deterministic, safe, and not model-selectable."""
    assert memory_command.name == "memory"
    assert memory_command.safety_level == SafetyLevel.SAFE
    assert memory_command.allow_model_selection is False

    registry = CommandRegistry(load_defaults=True)
    selectable = registry.get_model_selectable_commands()
    assert "memory" not in selectable


def test_pattern_matching_remember_variants(memory_command: MemoryCommand):
    """Test matching various natural remember phrases."""
    cases = [
        ("Remember that my favorite language is Python", "my favorite language is Python"),
        ("remember that I prefer concise answers", "I prefer concise answers"),
        ("Please remember that I like dark mode", "I like dark mode"),
        ("Remember: my name is Alice", "my name is Alice"),
        ("Remember I work as a software engineer", "I work as a software engineer"),
    ]
    for text, expected_content in cases:
        assert memory_command.can_handle(text) is True, f"Failed to handle: {text}"
        args = memory_command.extract_args(text)
        assert args["action"] == "store"
        assert args["content"] == expected_content


def test_pattern_matching_list_variants(memory_command: MemoryCommand):
    """Test matching various natural memory retrieval queries."""
    cases = [
        "What do you remember about me?",
        "what do you remember about me",
        "What do you remember?",
        "what do you remember",
        "List my memories",
        "list my memories",
        "list memories",
        "Show my memories",
        "show memories",
        "List all memories",
        "What memories do you have about me?",
    ]
    for text in cases:
        assert memory_command.can_handle(text) is True, f"Failed to handle: {text}"
        args = memory_command.extract_args(text)
        assert args["action"] == "list"


def test_pattern_matching_forget_specific_variants(memory_command: MemoryCommand):
    """Test matching specific forget commands."""
    cases = [
        ("Forget my favorite language", "my favorite language"),
        ("forget that my favorite language is Python", "my favorite language is Python"),
        ("Please forget that I prefer concise answers", "I prefer concise answers"),
        ("Forget: my name is Alice", "my name is Alice"),
        ("Forget preference: dark mode", "preference: dark mode"),
    ]
    for text, expected_target in cases:
        assert memory_command.can_handle(text) is True, f"Failed to handle: {text}"
        args = memory_command.extract_args(text)
        assert args["action"] == "forget"
        assert args["target"] == expected_target


def test_pattern_matching_clear_all_variants(memory_command: MemoryCommand):
    """Test matching clear all memory commands."""
    cases = [
        "Forget everything you remember",
        "forget everything you remember",
        "Forget everything",
        "Clear all memories",
        "clear all memories",
        "clear memories",
        "delete all memories",
        "wipe all memories",
        "Please forget everything you remember",
    ]
    for text in cases:
        assert memory_command.can_handle(text) is True, f"Failed to handle: {text}"
        args = memory_command.extract_args(text)
        assert args["action"] == "clear_all"
        assert args["targets"] == ["all_memories"]


def test_category_inference():
    """Verify deterministic category inference."""
    assert _infer_category("My favorite language is Python") == MemoryCategory.PREFERENCE.value
    assert _infer_category("I prefer concise answers") == MemoryCategory.PREFERENCE.value
    assert _infer_category("I like dark mode themes") == MemoryCategory.PREFERENCE.value
    assert _infer_category("Always answer with bullet points") == MemoryCategory.INSTRUCTION.value
    assert _infer_category("My project directory is C:\\Dev") == MemoryCategory.FACT.value


# =============================================================================
# 2. Store & Remember Execution Tests
# =============================================================================


def test_remember_stores_in_database(memory_command: MemoryCommand, memory_store: MemoryStore):
    """Ensure explicit remember commands persist memory records with correct metadata."""
    res = memory_command.execute({"action": "store", "content": "My favorite language is Python"})
    assert res.success is True
    assert "I will remember" in res.data["message"]
    assert res.data["category"] == "preference"

    # Verify persisted in SQLite
    memories = memory_store.list_memories()
    assert len(memories) == 1
    assert memories[0].content == "My favorite language is Python"
    assert memories[0].category == "preference"
    assert memories[0].source_type == MemorySourceType.EXPLICIT_COMMAND.value


def test_remember_multiple_items(memory_command: MemoryCommand, memory_store: MemoryStore):
    """Store multiple distinct memories."""
    memory_command.execute({"action": "store", "content": "My favorite language is Python"})
    memory_command.execute({"action": "store", "content": "I prefer concise answers"})
    memory_command.execute({"action": "store", "content": "I live in Berlin"})

    memories = memory_store.list_memories()
    assert len(memories) == 3


def test_remember_rejects_empty_content(memory_command: MemoryCommand):
    """Validation rejects empty content."""
    res = memory_command.execute({"action": "store", "content": "   "})
    assert res.success is False
    assert "No memory content provided" in res.error


def test_remember_rejects_secrets_and_passwords(memory_command: MemoryCommand, memory_store: MemoryStore):
    """Security check strictly blocks storing passwords and API keys."""
    res1 = memory_command.execute({"action": "store", "content": "My password is supersecret123"})
    assert res1.success is False
    assert "sensitive credentials" in res1.error

    res2 = memory_command.execute({"action": "store", "content": "My API key is sk-abcdef1234567890abcdef123456"})
    assert res2.success is False
    assert "sensitive credentials" in res2.error

    # Ensure nothing was persisted
    assert len(memory_store.list_memories()) == 0


# =============================================================================
# 3. List Memories Execution Tests
# =============================================================================


def test_list_memories_empty(memory_command: MemoryCommand):
    """Listing with no stored memories returns clean message."""
    res = memory_command.execute({"action": "list"})
    assert res.success is True
    assert res.data["count"] == 0
    assert "do not remember anything" in res.data["message"]


def test_list_memories_populated(memory_command: MemoryCommand, memory_store: MemoryStore):
    """Listing returns formatted bullet points of stored items."""
    memory_store.store_memory("My favorite language is Python", category="preference")
    memory_store.store_memory("I prefer concise answers", category="preference")

    res = memory_command.execute({"action": "list"})
    assert res.success is True
    assert res.data["count"] == 2
    assert "- My favorite language is Python" in res.data["message"]
    assert "- I prefer concise answers" in res.data["message"]


# =============================================================================
# 4. Forget Specific Memory Tests (Disambiguation, Deletion, Not Found)
# =============================================================================


def test_forget_single_matching_memory(memory_command: MemoryCommand, memory_store: MemoryStore):
    """Single match is deleted cleanly."""
    memory_store.store_memory("My favorite language is Python", category="preference")
    memory_store.store_memory("I live in Seattle", category="fact")

    res = memory_command.execute({"action": "forget", "target": "favorite language"})
    assert res.success is True
    assert res.data["status"] == "deleted"
    assert 'I have forgotten: "My favorite language is Python"' in res.data["message"]

    remaining = memory_store.list_memories()
    assert len(remaining) == 1
    assert remaining[0].content == "I live in Seattle"


def test_forget_multiple_matches_disambiguation(memory_command: MemoryCommand, memory_store: MemoryStore):
    """Multiple matches prompt user for clarification and do NOT delete anything."""
    memory_store.store_memory("My favorite language is Python", category="preference")
    memory_store.store_memory("I use Python for machine learning", category="fact")
    memory_store.store_memory("I prefer concise answers", category="preference")

    res = memory_command.execute({"action": "forget", "target": "Python"})
    assert res.success is True
    assert res.data["status"] == "ambiguous"
    assert res.data["matches_count"] == 2
    assert "Found multiple memories matching 'Python'" in res.data["message"]
    assert "- My favorite language is Python" in res.data["message"]
    assert "- I use Python for machine learning" in res.data["message"]

    # Invariant: Neither memory was deleted!
    remaining = memory_store.list_memories()
    assert len(remaining) == 3


def test_forget_no_matching_memory(memory_command: MemoryCommand, memory_store: MemoryStore):
    """Non-existent memory target returns helpful feedback."""
    memory_store.store_memory("My favorite language is Python", category="preference")

    res = memory_command.execute({"action": "forget", "target": "favorite editor"})
    assert res.success is True
    assert res.data["status"] == "not_found"
    assert "No saved memory found matching 'favorite editor'" in res.data["message"]

    # Nothing deleted
    assert len(memory_store.list_memories()) == 1


def test_forget_by_exact_uuid(memory_command: MemoryCommand, memory_store: MemoryStore):
    """Specific forget by UUID hex works directly."""
    rec = memory_store.store_memory("Temporary note", category="fact")
    res = memory_command.execute({"action": "forget", "target": rec.id})
    assert res.success is True
    assert res.data["status"] == "deleted"
    assert len(memory_store.list_memories()) == 0


# =============================================================================
# 5. Clear All Memories Confirmation Flow (DANGEROUS Gate)
# =============================================================================


def test_clear_all_requires_gatekeeper_and_confirmation(temp_db: Path, memory_store: MemoryStore):
    """Clearing all memories requires SafetyGatekeeper and returns a confirmation request."""
    memory_store.store_memory("My favorite language is Python")
    memory_store.store_memory("I prefer concise answers")

    gk = SafetyGatekeeper()
    registry = CommandRegistry(memory_store=memory_store)
    router = DeterministicRouter(registry=registry, gatekeeper=gk)

    # 1. Unconfirmed attempt via router: must be intercepted as DANGEROUS
    route_res = router.route("Forget everything you remember")
    assert route_res.matched is True
    assert route_res.command_name == "memory"
    assert route_res.result is None
    assert route_res.pending_confirmation is not None
    assert route_res.pending_confirmation.safety_level == SafetyLevel.DANGEROUS
    assert route_res.pending_confirmation.tool_name == "memory"
    assert route_res.pending_confirmation.targets == ["all_memories"]

    # Verify memories were NOT deleted
    assert len(memory_store.list_memories()) == 2

    # 2. Confirm action via gatekeeper
    action_id = route_res.pending_confirmation.action_id
    assert gk.confirm_action(action_id, expected_tool="memory", expected_targets=["all_memories"]) is True

    # 3. Second pass with confirmed action_id executes safely
    route_res2 = router.route("Forget everything you remember", action_id=action_id)
    assert route_res2.matched is True
    assert route_res2.result is not None
    assert route_res2.result.success is True
    assert "All memories have been cleared" in route_res2.result.data["message"]
    assert route_res2.result.data["deleted_count"] == 2

    # Verify database is now empty
    assert len(memory_store.list_memories()) == 0


def test_clear_all_direct_execution_without_gatekeeper_fails(memory_command: MemoryCommand):
    """Direct execution of clear_all without gatekeeper violates security invariants."""
    res = memory_command.execute({"action": "clear_all", "targets": ["all_memories"]})
    assert res.success is False
    assert "Security invariant violated" in res.error or "requires an active SafetyGatekeeper" in res.error


def test_clear_all_reusing_consumed_action_id_fails(temp_db: Path, memory_store: MemoryStore):
    """Action IDs cannot be replayed or reused."""
    gk = SafetyGatekeeper()
    registry = CommandRegistry(memory_store=memory_store)
    router = DeterministicRouter(registry=registry, gatekeeper=gk)

    route_res = router.route("Clear all memories")
    action_id = route_res.pending_confirmation.action_id
    assert gk.confirm_action(action_id) is True

    # First execution succeeds and consumes token
    res1 = router.route("Clear all memories", action_id=action_id)
    assert res1.result.success is True

    # Second execution with same action_id must be rejected
    res2 = router.route("Clear all memories", action_id=action_id)
    assert res2.result.success is False
    assert "already consumed" in res2.result.error or "invalid" in res2.result.error


# =============================================================================
# 6. End-to-End Agent & Zero Ollama Invocation Tests
# =============================================================================


def test_agent_remember_zero_ollama(agent: LokiAgent, memory_store: MemoryStore, mock_client: MagicMock):
    """Verify remember turn executes deterministically with zero Ollama invocation."""
    resp = agent.handle_input("Remember that my favorite language is Python")
    assert resp.success is True
    assert resp.source == ResponseSource.DETERMINISTIC
    assert resp.matched_command == "memory"
    assert 'I will remember: "my favorite language is Python"' in resp.text

    # Assert Ollama was never invoked
    mock_client.chat.assert_not_called()
    mock_client.generate.assert_not_called()

    # Verify persisted
    memories = memory_store.list_memories()
    assert len(memories) == 1
    assert memories[0].content == "my favorite language is Python"


def test_agent_list_memories_zero_ollama(agent: LokiAgent, memory_store: MemoryStore, mock_client: MagicMock):
    """Verify list memories executes deterministically with zero Ollama invocation."""
    memory_store.store_memory("My favorite language is Python", category="preference")
    memory_store.store_memory("I prefer concise answers", category="preference")

    resp = agent.handle_input("What do you remember about me?")
    assert resp.success is True
    assert resp.source == ResponseSource.DETERMINISTIC
    assert resp.matched_command == "memory"
    assert "- My favorite language is Python" in resp.text
    assert "- I prefer concise answers" in resp.text

    mock_client.chat.assert_not_called()
    mock_client.generate.assert_not_called()


def test_agent_forget_specific_zero_ollama(agent: LokiAgent, memory_store: MemoryStore, mock_client: MagicMock):
    """Verify specific forget executes deterministically with zero Ollama invocation."""
    memory_store.store_memory("My favorite language is Python", category="preference")

    resp = agent.handle_input("Forget my favorite language")
    assert resp.success is True
    assert resp.source == ResponseSource.DETERMINISTIC
    assert resp.matched_command == "memory"
    assert 'I have forgotten: "My favorite language is Python"' in resp.text

    mock_client.chat.assert_not_called()
    mock_client.generate.assert_not_called()
    assert len(memory_store.list_memories()) == 0


def test_agent_forget_ambiguous_zero_ollama(agent: LokiAgent, memory_store: MemoryStore, mock_client: MagicMock):
    """Verify ambiguous forget asks for clarification with zero Ollama invocation."""
    memory_store.store_memory("My favorite language is Python")
    memory_store.store_memory("I use Python for backend")

    resp = agent.handle_input("Forget Python")
    assert resp.success is True
    assert resp.source == ResponseSource.DETERMINISTIC
    assert "Found multiple memories matching 'Python'" in resp.text
    assert "- My favorite language is Python" in resp.text
    assert "- I use Python for backend" in resp.text

    mock_client.chat.assert_not_called()
    mock_client.generate.assert_not_called()
    assert len(memory_store.list_memories()) == 2


def test_agent_clear_all_gated_flow_zero_ollama(agent: LokiAgent, memory_store: MemoryStore, mock_client: MagicMock):
    """Verify clear all requires confirmation and executes deterministically upon confirmation."""
    memory_store.store_memory("Fact 1")
    memory_store.store_memory("Fact 2")

    # Turn 1: Intercepted as dangerous
    resp1 = agent.handle_input("Forget everything you remember")
    assert resp1.pending_confirmation is not None
    assert "requires your confirmation" in resp1.text
    assert resp1.pending_confirmation.safety_level == SafetyLevel.DANGEROUS
    action_id = resp1.pending_confirmation.action_id

    # Confirm action via agent
    assert agent.confirm_action(action_id) is True

    # Turn 2: Executed with action ID
    resp2 = agent.handle_input("Forget everything you remember", action_id=action_id)
    assert resp2.success is True
    assert resp2.source == ResponseSource.DETERMINISTIC
    assert "All memories have been cleared (2 deleted)" in resp2.text

    mock_client.chat.assert_not_called()
    mock_client.generate.assert_not_called()
    assert len(memory_store.list_memories()) == 0


# =============================================================================
# 7. Cross-Conversation Persistence Tests
# =============================================================================


def test_cross_conversation_memory_access(agent: LokiAgent, memory_store: MemoryStore):
    """Memories stored in conversation A are visible and manageable in conversation B."""
    # Conversation 1: User stores preference
    agent.switch_conversation("conv_alpha")
    resp1 = agent.handle_input("Remember that my favorite language is Python", conversation_id="conv_alpha")
    assert resp1.success is True
    assert 'I will remember: "my favorite language is Python"' in resp1.text

    # Conversation 2: User queries memories in a completely different conversation
    agent.switch_conversation("conv_beta")
    resp2 = agent.handle_input("What do you remember about me?", conversation_id="conv_beta")
    assert resp2.success is True
    assert "- my favorite language is Python" in resp2.text

    # Conversation 3: User deletes the memory from a third conversation
    agent.switch_conversation("conv_gamma")
    resp3 = agent.handle_input("Forget my favorite language", conversation_id="conv_gamma")
    assert resp3.success is True
    assert 'I have forgotten: "my favorite language is Python"' in resp3.text

    # Verify gone everywhere
    resp4 = agent.handle_input("List my memories", conversation_id="conv_alpha")
    assert "do not remember anything" in resp4.text
