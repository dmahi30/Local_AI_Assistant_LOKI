"""Tests verifying strict context isolation and persistence across conversations in LokiAgent."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from loki.agent import LokiAgent, ResponseSource
from loki.llm.client import OllamaClient
from loki.storage.conversation import ConversationStore


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock(spec=OllamaClient)
    client.host = "http://127.0.0.1:11434"
    client.check_health.return_value = True
    client.list_local_models.return_value = ["qwen2.5:0.5b"]
    client.is_model_available.return_value = True
    client.chat.return_value = "Mocked answer."
    return client


@pytest.fixture
def temp_store(tmp_path: Path) -> ConversationStore:
    return ConversationStore(db_path=tmp_path / "isolation_test.db")


def test_conversation_context_isolation(mock_client, temp_store):
    """CRITICAL REQUIREMENT: Turns from Conversation A must NEVER leak into Conversation B's prompt."""
    conv_a = temp_store.create_conversation(title="Chat A")
    conv_b = temp_store.create_conversation(title="Chat B")

    agent = LokiAgent(client=mock_client, storage=temp_store)

    # 1. User talks in Conversation A
    mock_client.chat.return_value = "Apples are delicious fruits."
    agent.handle_input("Tell me about apples", conversation_id=conv_a.id)

    # Inspect call args sent to client.chat for Conversation A
    last_messages_a = mock_client.chat.call_args[0][0]
    prompt_contents_a = [m["content"] for m in last_messages_a]
    assert "Tell me about apples" in prompt_contents_a
    assert "Tell me about bananas" not in prompt_contents_a

    # 2. User switches to Conversation B
    mock_client.chat.reset_mock()
    mock_client.chat.return_value = "Bananas are yellow tropical fruits."
    agent.handle_input("Tell me about bananas", conversation_id=conv_b.id)

    # Invariant: Prompt for B must NOT contain any mention of apples!
    last_messages_b = mock_client.chat.call_args[0][0]
    prompt_contents_b = [m["content"] for m in last_messages_b]
    assert "Tell me about bananas" in prompt_contents_b
    assert "Tell me about apples" not in prompt_contents_b
    assert "Apples are delicious fruits." not in prompt_contents_b

    # 3. Switch back to Conversation A: must restore A's turns without B's turns
    mock_client.chat.reset_mock()
    mock_client.chat.return_value = "Apples can be red or green."
    agent.handle_input("What colors can they be?", conversation_id=conv_a.id)

    last_messages_a2 = mock_client.chat.call_args[0][0]
    prompt_contents_a2 = [m["content"] for m in last_messages_a2]
    assert "Tell me about apples" in prompt_contents_a2
    assert "Apples are delicious fruits." in prompt_contents_a2
    assert "What colors can they be?" in prompt_contents_a2
    # Invariant: Must still contain ZERO mention of bananas
    assert "Tell me about bananas" not in prompt_contents_a2
    assert "Bananas are yellow tropical fruits." not in prompt_contents_a2


def test_persistence_across_agent_reboots(mock_client, temp_store):
    """Verify conversations and message history persist when an agent reboots."""
    conv = temp_store.create_conversation(title="Reboot Test")

    # Session 1: Run with Agent 1
    agent1 = LokiAgent(client=mock_client, storage=temp_store, active_conversation_id=conv.id)
    mock_client.chat.return_value = "Paris is the capital of France."
    res1 = agent1.handle_input("What is the capital of France?")
    assert res1.success is True
    assert res1.conversation_id == conv.id

    # Session 2: Create brand new agent instance with the same storage
    agent2 = LokiAgent(client=mock_client, storage=temp_store, active_conversation_id=conv.id)

    # In-memory history of agent2 must have loaded Session 1's messages
    history = agent2.get_history()
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "What is the capital of France?"}
    assert history[1] == {"role": "assistant", "content": "Paris is the capital of France."}

    # Follow-up question preserves conversational context
    mock_client.chat.reset_mock()
    mock_client.chat.return_value = "Its population is roughly 2.1 million."
    res2 = agent2.handle_input("What is its population?")
    assert res2.success is True

    messages_sent = mock_client.chat.call_args[0][0]
    contents = [m["content"] for m in messages_sent]
    assert "What is the capital of France?" in contents
    assert "Paris is the capital of France." in contents
    assert "What is its population?" in contents


def test_auto_title_generation_on_first_turn(mock_client, temp_store):
    """Verify that an initial 'New Chat' is automatically titled from the user's first prompt."""
    conv = temp_store.create_conversation(title="New Chat")
    agent = LokiAgent(client=mock_client, storage=temp_store, active_conversation_id=conv.id)

    agent.handle_input("Explain how neural networks learn representations")

    detail = temp_store.get_conversation(conv.id)
    assert detail.title != "New Chat"
    assert "Explain how neural networks" in detail.title


def test_deterministic_commands_persisted_in_history(mock_client, temp_store):
    """Verify deterministic command execution is recorded to conversation history."""
    conv = temp_store.create_conversation(title="System Diagnostics")
    agent = LokiAgent(client=mock_client, storage=temp_store, active_conversation_id=conv.id)

    res = agent.handle_input("what is my battery percentage")
    assert res.source == ResponseSource.DETERMINISTIC
    assert res.success is True

    detail = temp_store.get_conversation(conv.id)
    assert len(detail.messages) == 2
    assert detail.messages[0].role == "user"
    assert detail.messages[0].content == "what is my battery percentage"
    assert detail.messages[1].role == "assistant"
    assert detail.messages[1].source == "deterministic"
    assert "Battery" in detail.messages[1].content
