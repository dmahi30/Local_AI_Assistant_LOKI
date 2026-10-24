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


def test_context_messages_sliding_window_and_turn_boundaries(temp_store: ConversationStore):
    """Verify sliding-window limits context size, preserves turn boundaries, and keeps SQLite records intact."""
    conv = temp_store.create_conversation(title="Sliding Window Test")

    # Add 4 conversational turns (8 messages)
    for i in range(1, 5):
        temp_store.add_message(conv.id, role="user", content=f"User question {i}")
        temp_store.add_message(conv.id, role="assistant", content=f"Assistant answer {i}")

    # Verify SQLite has all 8 messages
    detail_full = temp_store.get_conversation(conv.id)
    assert len(detail_full.messages) == 8

    # 1. Fetch with max_messages=4 -> Should return [U3, A3, U4, A4]
    ctx_4 = temp_store.get_context_messages(conv.id, max_messages=4)
    assert len(ctx_4) == 4
    assert ctx_4[0] == {"role": "user", "content": "User question 3"}
    assert ctx_4[1] == {"role": "assistant", "content": "Assistant answer 3"}
    assert ctx_4[2] == {"role": "user", "content": "User question 4"}
    assert ctx_4[3] == {"role": "assistant", "content": "Assistant answer 4"}

    # 2. Fetch with max_messages=3 -> Naive slice is [A3, U4, A4]; turn boundary trims A3 -> [U4, A4]
    ctx_3 = temp_store.get_context_messages(conv.id, max_messages=3)
    assert len(ctx_3) == 2
    assert ctx_3[0] == {"role": "user", "content": "User question 4"}
    assert ctx_3[1] == {"role": "assistant", "content": "Assistant answer 4"}

    # 3. CRITICAL INVARIANT: SQLite database records must be 100% preserved
    detail_after = temp_store.get_conversation(conv.id)
    assert len(detail_after.messages) == 8
    assert [m.content for m in detail_after.messages] == [
        "User question 1", "Assistant answer 1",
        "User question 2", "Assistant answer 2",
        "User question 3", "Assistant answer 3",
        "User question 4", "Assistant answer 4",
    ]


def test_context_messages_deterministic_tool_bounding(temp_store: ConversationStore):
    """Verify deterministic command output is bounded in LLM prompt context without altering SQLite storage."""
    conv = temp_store.create_conversation(title="Tool Output Test")

    long_tool_output = "Line 1: Found file C:\\path\\to\\file.txt\n" * 40  # ~1600 characters
    temp_store.add_message(conv.id, role="user", content="search files")
    temp_store.add_message(
        conv.id,
        role="assistant",
        content=long_tool_output,
        source="deterministic",
    )

    # Context prompt should truncate long tool output
    ctx = temp_store.get_context_messages(conv.id, max_tool_chars=150)
    assert len(ctx) == 2
    assert ctx[0]["content"] == "search files"
    assert len(ctx[1]["content"]) < 250
    assert "[Command output truncated for context]" in ctx[1]["content"]

    # Database record must preserve full raw output untouched
    detail = temp_store.get_conversation(conv.id)
    assert len(detail.messages[1].content) == len(long_tool_output)
    assert "[Command output truncated for context]" not in detail.messages[1].content


# =========================================================================
# Connection Lifecycle & Persistence Regression Tests
# =========================================================================

def test_connection_lifecycle_closes_on_success(tmp_path: Path):
    """Verify SQLite connection is closed after operation and does not lock file on Windows."""
    db_file = tmp_path / "lifecycle_success.db"
    store = ConversationStore(db_path=db_file)

    # Perform normal writes
    conv = store.create_conversation("Lock Test")
    store.add_message(conv.id, role="user", content="Hello lock check")
    store.add_message(conv.id, role="assistant", content="Echo lock check")

    # If connection was leaked, opening in exclusive mode or deleting/moving the file would fail on Windows
    assert db_file.exists()
    # Test file can be opened in write/exclusive mode or renamed
    renamed_file = tmp_path / "lifecycle_success_renamed.db"
    db_file.rename(renamed_file)
    assert renamed_file.exists()


def test_connection_lifecycle_rollback_on_exception(tmp_path: Path):
    """Verify transaction rolls back and connection is closed when an error occurs."""
    db_file = tmp_path / "lifecycle_rollback.db"
    store = ConversationStore(db_path=db_file)
    conv = store.create_conversation("Rollback Test")

    # Attempt an invalid operation inside transaction
    with pytest.raises(Exception):
        with store._get_connection() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, source, created_at) VALUES (?, ?, ?, ?, ?)",
                (conv.id, "user", "Rollback message", "user", "2026-01-01T00:00:00Z"),
            )
            # Deliberately raise error before block exits
            raise RuntimeError("Forced transaction failure")

    # Verify message was rolled back
    detail = store.get_conversation(conv.id)
    assert detail is not None
    assert len(detail.messages) == 0

    # Verify file is not locked and can be renamed
    renamed = tmp_path / "lifecycle_rollback_renamed.db"
    db_file.rename(renamed)
    assert renamed.exists()


def test_persistence_across_separate_store_instances(tmp_path: Path):
    """Verify conversation and messages persist cleanly across separate store instances."""
    db_file = tmp_path / "cross_instance.db"

    # Instance 1: Create conversation and messages
    store1 = ConversationStore(db_path=db_file)
    conv = store1.create_conversation(title="Multi-Instance Chat")
    store1.add_message(conv.id, role="user", content="First question")
    store1.add_message(conv.id, role="assistant", content="First answer")

    # Instance 2: Read conversation from same db file
    store2 = ConversationStore(db_path=db_file)
    convs = store2.list_conversations()
    assert len(convs) == 1
    assert convs[0].title == "Multi-Instance Chat"
    assert convs[0].message_count == 2

    detail = store2.get_conversation(conv.id)
    assert detail is not None
    assert len(detail.messages) == 2
    assert detail.messages[0].content == "First question"
    assert detail.messages[1].content == "First answer"

    # Instance 2: Delete conversation
    deleted = store2.delete_conversation(conv.id)
    assert deleted is True

    # Instance 3: Verify clean slate persisted after deletion
    store3 = ConversationStore(db_path=db_file)
    assert len(store3.list_conversations()) == 0
    assert store3.get_conversation(conv.id) is None
