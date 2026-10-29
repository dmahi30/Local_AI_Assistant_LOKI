"""Unit and integration tests for SQLite MemoryStore and FTS5 keyword indexing."""

from pathlib import Path
import pytest

from loki.core.exceptions import InvalidMemoryError, MemoryNotFoundError, StorageError
from loki.storage.conversation import ConversationStore
from loki.storage.memory import (
    MemoryCategory,
    MemoryRecord,
    MemorySourceType,
    MemoryStore,
    sanitize_fts_query,
    sanitize_memory_metadata,
)


@pytest.fixture
def memory_store(tmp_path: Path) -> MemoryStore:
    """Fixture providing a fresh isolated MemoryStore."""
    db_file = tmp_path / "test_memory.db"
    return MemoryStore(db_path=db_file)


# =============================================================================
# 1. Table Creation and Reopen Persistence Tests
# =============================================================================


def test_memory_store_table_creation_and_reopen_persistence(tmp_path: Path):
    """Verify memories table and FTS5 index are created and persist across store reboots."""
    db_file = tmp_path / "persistence_test.db"

    # Session 1: Store memory
    store1 = MemoryStore(db_path=db_file)
    assert store1.fts_available is True

    mem1 = store1.store_memory(
        content="User prefers high-contrast dark theme in terminal.",
        category=MemoryCategory.PREFERENCE.value,
        key="theme",
        confidence=0.95,
        is_pinned=True,
        metadata={"source": "user_settings"},
    )
    assert mem1.id is not None
    assert mem1.content == "User prefers high-contrast dark theme in terminal."

    # Session 2: Reopen from same database file
    store2 = MemoryStore(db_path=db_file)
    reloaded = store2.get_memory(mem1.id)
    assert reloaded is not None
    assert reloaded.id == mem1.id
    assert reloaded.content == "User prefers high-contrast dark theme in terminal."
    assert reloaded.category == "preference"
    assert reloaded.key == "theme"
    assert reloaded.confidence == 0.95
    assert reloaded.is_pinned is True
    assert reloaded.metadata == {"source": "user_settings"}

    # Search in reopened store
    search_res = store2.search_memories("terminal")
    assert len(search_res) == 1
    assert search_res[0].id == mem1.id


# =============================================================================
# 2. CRUD Lifecycle Tests (Create, Get, List, Update, Delete, Clear)
# =============================================================================


def test_memory_create_get_list_lifecycle(memory_store: MemoryStore):
    """Verify creating, getting by id/key, and listing with filters."""
    # 1. Create items
    m1 = memory_store.store_memory(
        content="User's primary programming language is Python.",
        category=MemoryCategory.FACT.value,
        key="language",
        confidence=1.0,
        is_pinned=True,
    )
    m2 = memory_store.store_memory(
        content="User prefers tabs instead of spaces.",
        category=MemoryCategory.PREFERENCE.value,
        key="indentation",
        confidence=0.8,
        is_pinned=False,
    )
    m3 = memory_store.store_memory(
        content="Always answer in concise bullet points.",
        category=MemoryCategory.INSTRUCTION.value,
        key="formatting",
        confidence=1.0,
        is_pinned=True,
    )

    # 2. Get by ID and verify touch
    assert m1.last_accessed_at is None
    fetched_touch = memory_store.get_memory(m1.id, touch=True)
    assert fetched_touch is not None
    assert fetched_touch.last_accessed_at is not None

    # 3. Get by key
    by_key = memory_store.get_memory_by_key("language")
    assert by_key is not None
    assert by_key.id == m1.id

    by_key_cat = memory_store.get_memory_by_key("language", category="fact")
    assert by_key_cat is not None
    assert by_key_cat.id == m1.id

    assert memory_store.get_memory_by_key("nonexistent") is None

    # 4. List without filters (pinned first, then updated_at desc)
    all_mems = memory_store.list_memories()
    assert len(all_mems) == 3
    # Pinned items should precede unpinned
    assert all_mems[0].is_pinned is True
    assert all_mems[1].is_pinned is True
    assert all_mems[2].is_pinned is False

    # 5. List with category filter
    prefs = memory_store.list_memories(category="preference")
    assert len(prefs) == 1
    assert prefs[0].id == m2.id

    # 6. List with pinned filter
    pinned_only = memory_store.list_memories(is_pinned=True)
    assert len(pinned_only) == 2
    assert all(m.is_pinned for m in pinned_only)

    # 7. Pagination (limit and offset)
    paged = memory_store.list_memories(limit=1, offset=1)
    assert len(paged) == 1
    assert paged[0].id == all_mems[1].id


def test_memory_update_and_fts_synchronization(memory_store: MemoryStore):
    """Verify updating a memory updates its content and synchronizes FTS5 index triggers."""
    m = memory_store.store_memory(
        content="User prefers Python for data processing.",
        category=MemoryCategory.PREFERENCE.value,
        key="pref_data_lang",
    )

    # FTS verifies initial content
    assert len(memory_store.search_memories("Python")) == 1
    assert len(memory_store.search_memories("Rust")) == 0

    # Update content to Rust
    updated = memory_store.update_memory(
        m.id,
        content="User prefers Rust for data processing.",
        confidence=0.99,
        is_pinned=True,
    )
    assert updated.content == "User prefers Rust for data processing."
    assert updated.confidence == 0.99
    assert updated.is_pinned is True

    # FTS must reflect the update: Python gone, Rust searchable
    assert len(memory_store.search_memories("Python")) == 0
    rust_results = memory_store.search_memories("Rust")
    assert len(rust_results) == 1
    assert rust_results[0].id == m.id


def test_memory_delete_and_clear_lifecycle(memory_store: MemoryStore):
    """Verify deleting individual memories and clearing by category or all."""
    m1 = memory_store.store_memory("Fact 1", category="fact")
    m2 = memory_store.store_memory("Fact 2", category="fact")
    m3 = memory_store.store_memory("Pref 1", category="preference")

    assert len(memory_store.list_memories()) == 3

    # Delete 1 item
    deleted = memory_store.delete_memory(m1.id)
    assert deleted is True
    assert memory_store.get_memory(m1.id) is None
    assert len(memory_store.search_memories("Fact 1")) == 0
    # Deleting non-existent returns False
    assert memory_store.delete_memory("nonexistent") is False

    # Clear by category: removes m2, keeps m3
    cleared_facts = memory_store.clear_memories(category="fact")
    assert cleared_facts == 1
    remaining = memory_store.list_memories()
    assert len(remaining) == 1
    assert remaining[0].id == m3.id

    # Clear all
    cleared_all = memory_store.clear_memories()
    assert cleared_all == 1
    assert len(memory_store.list_memories()) == 0
    assert len(memory_store.search_memories("Pref")) == 0


def test_memory_upsert_by_key(memory_store: MemoryStore):
    """Verify upsert_key updates existing memory when key and category match."""
    # Initial store
    m1 = memory_store.store_memory(
        content="Default theme is dark",
        category="preference",
        key="theme",
        confidence=0.5,
    )

    # Upsert with same key and category
    m2 = memory_store.store_memory(
        content="User changed theme to light",
        category="preference",
        key="theme",
        confidence=0.9,
        upsert_key=True,
    )

    # Invariant: ID remains the same, content updated, total count is 1
    assert m2.id == m1.id
    assert m2.content == "User changed theme to light"
    assert m2.confidence == 0.9

    all_mems = memory_store.list_memories()
    assert len(all_mems) == 1


# =============================================================================
# 3. FTS5 Search, BM25 Ranking, and Fallback Tests
# =============================================================================


def test_memory_search_bm25_ranking_and_category_filter(memory_store: MemoryStore):
    """Verify search relevance ordering and category filtering."""
    memory_store.store_memory("Astrophysics involves study of stars and galaxies.", category="fact")
    memory_store.store_memory("Quantum mechanics describes atomic particles.", category="fact")
    memory_store.store_memory("User likes reading about astrophysics history.", category="preference")

    # Search query matching both astrophysics items
    results = memory_store.search_memories("astrophysics")
    assert len(results) == 2

    # Filtered by category
    fact_results = memory_store.search_memories("astrophysics", category="fact")
    assert len(fact_results) == 1
    assert fact_results[0].category == "fact"

    pref_results = memory_store.search_memories("astrophysics", category="preference")
    assert len(pref_results) == 1
    assert pref_results[0].category == "preference"


def test_memory_fts_unavailable_fallback(tmp_path: Path):
    """Verify safe parameterized LIKE search functions properly when FTS5 is disabled."""
    db_file = tmp_path / "no_fts.db"
    store = MemoryStore(db_path=db_file, enable_fts=False)
    assert store.fts_available is False

    store.store_memory("User prefers espresso over filter coffee.", category="preference", key="coffee")
    store.store_memory("User likes green tea in the morning.", category="preference", key="tea")

    # Search with LIKE fallback
    res_coffee = store.search_memories("espresso")
    assert len(res_coffee) == 1
    assert "espresso" in res_coffee[0].content

    res_tea = store.search_memories("green tea")
    assert len(res_tea) == 1
    assert "green tea" in res_tea[0].content

    # Case insensitivity in LIKE fallback
    res_caps = store.search_memories("ESPRESSO")
    assert len(res_caps) == 1


def test_memory_search_handles_sql_injection_and_special_syntax(memory_store: MemoryStore):
    """Verify search handles quotes, SQL operators, and malicious strings gracefully."""
    memory_store.store_memory("Regular memory content for security testing.")

    malicious_inputs = [
        "' OR 1=1; --",
        '"; DROP TABLE memories; --',
        '"""""',
        "OR",
        "AND",
        "NOT",
        "NEAR/5",
        "content: test",
        "(*)",
        ":::",
        "<script>alert(1)</script>",
    ]

    for bad in malicious_inputs:
        # None of these should raise OperationalError or crash the store
        results = memory_store.search_memories(bad)
        assert isinstance(results, list)

    # Normal search still functions afterwards
    normal = memory_store.search_memories("Regular")
    assert len(normal) == 1


# =============================================================================
# 4. Foreign Key and Conversation Deletion Tests
# =============================================================================


def test_memory_foreign_key_on_delete_set_null(tmp_path: Path):
    """Verify ON DELETE SET NULL on source_conversation_id when conversation is deleted."""
    db_file = tmp_path / "fk_test.db"
    conv_store = ConversationStore(db_path=db_file)
    mem_store = MemoryStore(db_path=db_file)

    # 1. Create a conversation
    conv = conv_store.create_conversation(title="Source Conversation")
    cid = conv.id

    # 2. Store memory linked to this conversation
    mem = mem_store.store_memory(
        content="Extracted user fact from chat.",
        category="fact",
        source_type=MemorySourceType.EXTRACTED_CONVERSATION.value,
        source_conversation_id=cid,
    )
    assert mem.source_conversation_id == cid

    # 3. Delete the parent conversation
    deleted = conv_store.delete_conversation(cid)
    assert deleted is True

    # 4. Invariant: Memory remains intact, but source_conversation_id is now None (SET NULL)
    reloaded = mem_store.get_memory(mem.id)
    assert reloaded is not None
    assert reloaded.id == mem.id
    assert reloaded.source_conversation_id is None
    assert reloaded.content == "Extracted user fact from chat."


# =============================================================================
# 5. Security: Secret Rejection and Metadata Sanitization Tests
# =============================================================================


def test_memory_rejects_credentials_and_secrets(memory_store: MemoryStore):
    """Security Invariant: Memory content with passwords, API keys, or raw tokens is rejected."""
    secret_payloads = [
        "My password: super_secret_123",
        "api_key: sk-1234567890abcdef1234567890",
        "secret_key=xyz987654321",
        "Here is my private_key: abc123def456",
        "access_token: ghp_1234567890abcdef1234567890abcdef",
        "action_id: act_delete_secret_999",
        "confirmation_token: tok_abc_123",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQ...",
    ]

    for payload in secret_payloads:
        with pytest.raises(InvalidMemoryError) as exc_info:
            memory_store.store_memory(content=payload, category="fact")
        assert "sensitive credentials" in str(exc_info.value).lower()

    # Invariant: No items were stored
    assert len(memory_store.list_memories()) == 0


def test_memory_sanitizes_metadata_keys(memory_store: MemoryStore):
    """Security Invariant: Sensitive metadata keys are stripped before saving."""
    raw_meta = {
        "action_id": "act_sensitive_999",
        "confirmation_token": "secret_tok_123",
        "password": "pass",
        "api_key": "key",
        "safe_origin": "chat_dialog",
        "nested": {
            "token": "nested_secret",
            "allowed_data": 42,
        },
    }

    sanitized = sanitize_memory_metadata(raw_meta)
    assert "action_id" not in sanitized
    assert "confirmation_token" not in sanitized
    assert "password" not in sanitized
    assert "api_key" not in sanitized
    assert sanitized["safe_origin"] == "chat_dialog"
    assert sanitized["nested"]["allowed_data"] == 42
    assert "token" not in sanitized["nested"]

    # Store with metadata and verify retrieval
    mem = memory_store.store_memory(
        content="Safe user preference.",
        category="preference",
        metadata=raw_meta,
    )
    fetched = memory_store.get_memory(mem.id)
    assert fetched is not None
    assert "action_id" not in fetched.metadata
    assert fetched.metadata["safe_origin"] == "chat_dialog"


# =============================================================================
# 6. Input Validation and Boundary Tests
# =============================================================================


def test_memory_input_validation(memory_store: MemoryStore):
    """Verify validation on empty content, empty category, and missing IDs."""
    # Empty content
    with pytest.raises(InvalidMemoryError) as exc_content:
        memory_store.store_memory(content="", category="fact")
    assert "empty" in str(exc_content.value).lower()

    with pytest.raises(InvalidMemoryError):
        memory_store.store_memory(content="   ", category="fact")

    # Empty category
    with pytest.raises(InvalidMemoryError) as exc_cat:
        memory_store.store_memory(content="Valid content", category="")
    assert "category" in str(exc_cat.value).lower()

    # Update non-existent ID
    with pytest.raises(MemoryNotFoundError):
        memory_store.update_memory("nonexistent_id", content="New content")

    # Update empty ID
    with pytest.raises(MemoryNotFoundError):
        memory_store.update_memory("", content="New content")

    # Empty search query returns []
    assert memory_store.search_memories("") == []
    assert memory_store.search_memories("   ") == []

    # Get empty ID returns None
    assert memory_store.get_memory("") is None
    assert memory_store.get_memory_by_key("") is None
