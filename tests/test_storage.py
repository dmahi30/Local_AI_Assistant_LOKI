"""Unit tests for SQLite ConversationStore."""

from pathlib import Path
import pytest

from loki.core.exceptions import ConversationNotFoundError, StorageError
from loki.storage.conversation import (
    ConversationStore,
    sanitize_metadata,
)


@pytest.fixture
def temp_store(tmp_path: Path) -> ConversationStore:
    """Fixture providing a ConversationStore with a temporary SQLite database."""
    db_file = tmp_path / "test_conversations.db"
    return ConversationStore(db_path=db_file)


def test_create_and_get_conversation(temp_store: ConversationStore):
    """Verify creating a conversation stores summary and can be retrieved."""
    summary = temp_store.create_conversation(title="Researching Quantum Mechanics")

    assert summary.id is not None
    assert summary.title == "Researching Quantum Mechanics"
    assert summary.message_count == 0

    detail = temp_store.get_conversation(summary.id)
    assert detail is not None
    assert detail.id == summary.id
    assert detail.title == "Researching Quantum Mechanics"
    assert len(detail.messages) == 0


def test_list_conversations_ordering_and_counts(temp_store: ConversationStore):
    """Verify list_conversations orders by updated_at descending with accurate counts."""
    c1 = temp_store.create_conversation(title="Chat One")
    c2 = temp_store.create_conversation(title="Chat Two")

    # Add messages to c1 to update its timestamp
    temp_store.add_message(c1.id, role="user", content="Hello in C1")
    temp_store.add_message(c1.id, role="assistant", content="Reply in C1")

    convs = temp_store.list_conversations()
    assert len(convs) == 2
    # c1 was updated more recently than c2
    assert convs[0].id == c1.id
    assert convs[0].message_count == 2
    assert convs[1].id == c2.id
    assert convs[1].message_count == 0


def test_chronological_messages_persistence(temp_store: ConversationStore):
    """Verify messages are appended and retrieved in strict chronological order."""
    conv = temp_store.create_conversation(title="Chronological Test")

    m1 = temp_store.add_message(conv.id, role="user", content="First message")
    m2 = temp_store.add_message(conv.id, role="assistant", content="Second message", model_used="qwen2.5:0.5b")
    m3 = temp_store.add_message(conv.id, role="user", content="Third message")

    detail = temp_store.get_conversation(conv.id)
    assert detail is not None
    assert len(detail.messages) == 3

    assert [m.content for m in detail.messages] == ["First message", "Second message", "Third message"]
    assert detail.messages[0].id == m1.id
    assert detail.messages[1].model_used == "qwen2.5:0.5b"
    assert detail.messages[2].id == m3.id


def test_metadata_sanitization(temp_store: ConversationStore):
    """Security Invariant: Internal confirmation tokens and secrets must not be stored in metadata."""
    raw_meta = {
        "action_id": "act_sensitive_999",
        "confirmation_token": "secret_tok_123",
        "secret": "top_secret_key",
        "safe_flag": True,
        "model_runtime": "qwen2.5:0.5b",
        "nested": {
            "token": "nested_secret",
            "allowed_data": 42,
        },
    }

    sanitized = sanitize_metadata(raw_meta)
    assert "action_id" not in sanitized
    assert "confirmation_token" not in sanitized
    assert "secret" not in sanitized
    assert sanitized["safe_flag"] is True
    assert sanitized["nested"]["allowed_data"] == 42
    assert "token" not in sanitized["nested"]

    # Test via add_message
    conv = temp_store.create_conversation()
    msg = temp_store.add_message(
        conv.id,
        role="assistant",
        content="Action processed",
        metadata=raw_meta,
    )

    detail = temp_store.get_conversation(conv.id)
    saved_meta = detail.messages[0].metadata
    assert "action_id" not in saved_meta
    assert "secret" not in saved_meta
    assert saved_meta["safe_flag"] is True


def test_update_conversation_title(temp_store: ConversationStore):
    """Verify renaming a conversation updates title and timestamp."""
    conv = temp_store.create_conversation(title="Original Title")
    assert conv.title == "Original Title"

    updated = temp_store.update_title(conv.id, "Updated Custom Title")
    assert updated is True

    detail = temp_store.get_conversation(conv.id)
    assert detail.title == "Updated Custom Title"

    # Truncation of overly long title
    long_title = "A" * 200
    temp_store.update_title(conv.id, long_title)
    detail2 = temp_store.get_conversation(conv.id)
    assert len(detail2.title) == 100


def test_cascade_delete_conversation(temp_store: ConversationStore):
    """Verify deleting conversation deletes conversation and cascades to delete all messages."""
    conv = temp_store.create_conversation(title="To be deleted")
    temp_store.add_message(conv.id, role="user", content="Msg 1")
    temp_store.add_message(conv.id, role="assistant", content="Msg 2")

    deleted = temp_store.delete_conversation(conv.id)
    assert deleted is True

    # Conversation no longer exists
    assert temp_store.get_conversation(conv.id) is None
    assert len(temp_store.list_conversations()) == 0

    # Underlying messages table has zero rows left
    with temp_store._get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conv.id,)).fetchone()[0]
        assert count == 0


def test_invalid_conversation_id_handling(temp_store: ConversationStore):
    """Verify safe handling of non-existent conversation IDs."""
    fake_id = "non-existent-uuid-12345"

    assert temp_store.get_conversation(fake_id) is None
    assert temp_store.update_title(fake_id, "New Title") is False
    assert temp_store.delete_conversation(fake_id) is False

    with pytest.raises(ConversationNotFoundError):
        temp_store.add_message(fake_id, role="user", content="Orphan message")


def test_get_context_messages_isolation(temp_store: ConversationStore):
    """Verify context messages extraction only includes user and assistant messages for that conversation."""
    c1 = temp_store.create_conversation(title="Conversation 1")
    c2 = temp_store.create_conversation(title="Conversation 2")

    temp_store.add_message(c1.id, role="user", content="Hello from C1")
    temp_store.add_message(c1.id, role="assistant", content="Reply to C1")
    temp_store.add_message(c1.id, role="system", content="System notification (should be omitted from LLM prompt)")

    temp_store.add_message(c2.id, role="user", content="Hello from C2")

    c1_context = temp_store.get_context_messages(c1.id)
    c2_context = temp_store.get_context_messages(c2.id)

    assert len(c1_context) == 2
    assert c1_context[0] == {"role": "user", "content": "Hello from C1"}
    assert c1_context[1] == {"role": "assistant", "content": "Reply to C1"}

    assert len(c2_context) == 1
    assert c2_context[0] == {"role": "user", "content": "Hello from C2"}


def test_generate_title_from_prompt():
    """Verify smart title generation trims and handles length limits."""
    assert ConversationStore.generate_title_from_prompt("   ") == "New Chat"
    assert ConversationStore.generate_title_from_prompt("What is Python?") == "What is Python?"
    long_prompt = "Explain in great detail the mathematical theory behind general relativity and curved spacetime"
    title = ConversationStore.generate_title_from_prompt(long_prompt, max_length=35)
    assert len(title) <= 38  # 35 + ...
    assert title.endswith("...")
