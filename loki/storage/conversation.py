"""Persistent conversation and message storage using SQLite for LOKI.

Guarantees:
- Fully local and offline storage in local app data directory.
- ACID transactions with foreign-key cascading and parameterized queries.
- Strict isolation across distinct conversation histories.
- Sanitization to prevent storing internal confirmation tokens or secrets.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Generator
import uuid
from pydantic import BaseModel, Field

from ..core.exceptions import ConversationNotFoundError, StorageError
from ..core.logging import get_logger
from ..utils.paths import get_data_dir

logger = get_logger("storage.conversation")

# Keys that must never be persisted to history for security/safety
SENSITIVE_METADATA_KEYS = {
    "action_id",
    "token",
    "tokens",
    "secret",
    "secrets",
    "confirmation_token",
    "pending_confirmation",
}


def _utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def sanitize_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Strip internal confirmation tokens and secrets before saving metadata."""
    if not meta:
        return {}
    sanitized: dict[str, Any] = {}
    for k, v in meta.items():
        if k.lower() in SENSITIVE_METADATA_KEYS:
            continue
        # Also prevent nested dictionary leaks of action_id
        if isinstance(v, dict):
            sanitized[k] = sanitize_metadata(v)
        else:
            sanitized[k] = v
    return sanitized


class ConversationSummary(BaseModel):
    """Metadata summary of a conversation for list views and sidebars."""
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class MessageRecord(BaseModel):
    """A single stored message in a conversation."""
    id: int | None = None
    conversation_id: str
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str
    source: str = Field(default="llm", description="'llm', 'deterministic', or 'system'")
    model_used: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utc_now_iso)


class ConversationDetail(BaseModel):
    """Full conversation detail with all chronologically ordered messages."""
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageRecord] = Field(default_factory=list)


class ConversationStore:
    """Thread-safe SQLite store managing persistent conversation sessions and messages."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = get_data_dir() / "loki.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Create and configure a SQLite connection with foreign keys enabled.

        Guarantees that transactions commit on success or rollback on exception,
        and that every connection is explicitly closed upon exit to prevent file
        locks and resource leaks on Windows.
        """
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            raise StorageError(f"Failed to connect to SQLite database at {self.db_path}: {e}") from e

        try:
            with conn:
                yield conn
        finally:
            if conn is not None:
                conn.close()

    def _init_database(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    model_used TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
                """
            )

    @staticmethod
    def generate_title_from_prompt(prompt: str, max_length: int = 36) -> str:
        """Generate a clean, truncated conversation title from an initial prompt."""
        clean = " ".join(prompt.strip().split())
        if not clean:
            return "New Chat"
        if len(clean) <= max_length:
            return clean
        # Break nicely on whitespace if possible
        truncated = clean[:max_length]
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        return truncated + "..."

    def create_conversation(
        self,
        title: str = "New Chat",
        conversation_id: str | None = None,
    ) -> ConversationSummary:
        """Create and store a new conversation session."""
        cid = conversation_id or uuid.uuid4().hex
        clean_title = title.strip() or "New Chat"
        now = _utc_now_iso()

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO conversations (id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cid, clean_title, now, now),
                )
            return ConversationSummary(
                id=cid,
                title=clean_title,
                created_at=now,
                updated_at=now,
                message_count=0,
            )
        except sqlite3.Error as e:
            raise StorageError(f"Failed to create conversation {cid}: {e}") from e

    def list_conversations(self) -> list[ConversationSummary]:
        """List all conversations ordered from most recently active to oldest."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT c.id, c.title, c.created_at, c.updated_at,
                           COUNT(m.id) AS message_count
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    GROUP BY c.id
                    ORDER BY c.updated_at DESC
                    """
                )
                rows = cursor.fetchall()
                return [
                    ConversationSummary(
                        id=row["id"],
                        title=row["title"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        message_count=row["message_count"],
                    )
                    for row in rows
                ]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to list conversations: {e}") from e

    def get_conversation(self, conversation_id: str) -> ConversationDetail | None:
        """Fetch a conversation and all its messages in chronological order."""
        try:
            with self._get_connection() as conn:
                conv_row = conn.execute(
                    "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()

                if not conv_row:
                    return None

                msg_rows = conn.execute(
                    """
                    SELECT id, conversation_id, role, content, source, model_used, metadata, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id ASC
                    """,
                    (conversation_id,),
                ).fetchall()

                messages: list[MessageRecord] = []
                for m in msg_rows:
                    meta_dict = {}
                    if m["metadata"]:
                        try:
                            meta_dict = json.loads(m["metadata"])
                        except json.JSONDecodeError:
                            meta_dict = {}

                    messages.append(
                        MessageRecord(
                            id=m["id"],
                            conversation_id=m["conversation_id"],
                            role=m["role"],
                            content=m["content"],
                            source=m["source"],
                            model_used=m["model_used"],
                            metadata=meta_dict,
                            created_at=m["created_at"],
                        )
                    )

                return ConversationDetail(
                    id=conv_row["id"],
                    title=conv_row["title"],
                    created_at=conv_row["created_at"],
                    updated_at=conv_row["updated_at"],
                    messages=messages,
                )
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get conversation {conversation_id}: {e}") from e

    def update_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title."""
        clean_title = title.strip() or "Untitled Conversation"
        clean_title = clean_title[:100]  # Reasonable max length
        now = _utc_now_iso()

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (clean_title, now, conversation_id),
                )
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise StorageError(f"Failed to update conversation title {conversation_id}: {e}") from e

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and all its messages via foreign-key cascade."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE id = ?",
                    (conversation_id,),
                )
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise StorageError(f"Failed to delete conversation {conversation_id}: {e}") from e

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        source: str = "llm",
        model_used: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageRecord:
        """Append a message to a conversation and update its activity timestamp."""
        now = _utc_now_iso()
        safe_meta = sanitize_metadata(metadata)
        meta_str = json.dumps(safe_meta) if safe_meta else None

        try:
            with self._get_connection() as conn:
                # 1. Verify existence
                exists = conn.execute(
                    "SELECT 1 FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()

                if not exists:
                    raise ConversationNotFoundError(
                        f"Cannot append message to non-existent conversation: {conversation_id}"
                    )

                # 2. Insert message
                cursor = conn.execute(
                    """
                    INSERT INTO messages (conversation_id, role, content, source, model_used, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (conversation_id, role, content, source, model_used, meta_str, now),
                )
                msg_id = cursor.lastrowid

                # 3. Update conversation updated_at
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )

            return MessageRecord(
                id=msg_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
                source=source,
                model_used=model_used,
                metadata=safe_meta,
                created_at=now,
            )
        except ConversationNotFoundError:
            raise
        except sqlite3.Error as e:
            raise StorageError(f"Failed to add message to conversation {conversation_id}: {e}") from e

    def get_context_messages(
        self,
        conversation_id: str,
        max_messages: int | None = 20,
        max_tool_chars: int = 400,
    ) -> list[dict[str, str]]:
        """Return strictly scoped user and assistant messages for LLM context window.

        Invariants:
        - Scoped strictly to conversation_id (never leaks turns from other chats).
        - Excludes non-conversational system alerts or error notifications.
        - Preserves strict chronological order.
        - Enforces sliding-window limit (max_messages) without deleting historical SQLite records.
        - Preserves turn boundaries: begins on a user turn after system prompt where practical.
        - Applies safe prompt-only bounding for bulky deterministic command output.
        """
        detail = self.get_conversation(conversation_id)
        if not detail:
            return []

        # 1. Filter conversational roles
        candidate_messages: list[MessageRecord] = [
            m for m in detail.messages if m.role in ("user", "assistant")
        ]

        # 2. Enforce sliding window while preserving valid turn boundaries
        if max_messages is not None and max_messages > 0 and len(candidate_messages) > max_messages:
            candidate_messages = candidate_messages[-max_messages:]
            # Ensure context starts on a user message so the prompt follows system prompt naturally
            while candidate_messages and candidate_messages[0].role == "assistant" and len(candidate_messages) > 1:
                candidate_messages = candidate_messages[1:]

        # 3. Construct LLM context items with bounded tool output
        context: list[dict[str, str]] = []
        for msg in candidate_messages:
            content = msg.content
            # Safe prompt-only bounding for deterministic tool outputs to prevent context pollution
            if msg.source == "deterministic" and len(content) > max_tool_chars:
                content = content[:max_tool_chars].rstrip() + "\n... [Command output truncated for context]"
            context.append({"role": msg.role, "content": content})
        return context

    def clear_messages(self, conversation_id: str) -> bool:
        """Delete all messages in a conversation while keeping the conversation session."""
        now = _utc_now_iso()
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )
                return cursor.rowcount >= 0
        except sqlite3.Error as e:
            raise StorageError(f"Failed to clear messages for conversation {conversation_id}: {e}") from e
