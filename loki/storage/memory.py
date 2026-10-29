"""Persistent user memory and context storage using SQLite for LOKI.

Guarantees:
- Fully local and offline storage in local app data directory (loki.db).
- ACID transactions with foreign-key cascading and parameterized queries.
- Independent of the raw messages table; source_conversation_id uses ON DELETE SET NULL.
- Sensitive credentials, API keys, and confirmation tokens are strictly rejected/sanitized.
- SQLite FTS5 for offline keyword and BM25 search with reliable triggers.
- Safe parameterized LIKE fallback if FTS5 is unavailable.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Generator
import uuid
from pydantic import BaseModel, Field

from ..core.exceptions import InvalidMemoryError, MemoryNotFoundError, StorageError
from ..core.logging import get_logger
from ..utils.paths import get_data_dir

logger = get_logger("storage.memory")

# Sensitive metadata keys that must never be persisted in memory
SENSITIVE_METADATA_KEYS = {
    "action_id",
    "token",
    "tokens",
    "secret",
    "secrets",
    "confirmation_token",
    "pending_confirmation",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "private_key",
    "secret_key",
}

# Regex patterns matching security credentials, private keys, or raw tokens
SENSITIVE_CONTENT_PATTERNS = [
    re.compile(r"(?i)\b(api_key|apikey|secret_key|private_key|auth_token|access_token|confirmation_token|action_id)\s*(?:is|[:=])\s*\S+"),
    re.compile(r"(?i)\bpassword\s*(?:is|[:=])\s*\S+"),
    re.compile(r"(?i)\b(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,})"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def _utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def sanitize_memory_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Strip internal confirmation tokens and secrets before saving metadata."""
    if not meta:
        return {}
    sanitized: dict[str, Any] = {}
    for k, v in meta.items():
        if k.lower() in SENSITIVE_METADATA_KEYS:
            continue
        if isinstance(v, dict):
            sanitized[k] = sanitize_memory_metadata(v)
        else:
            sanitized[k] = v
    return sanitized


def validate_memory_content_safety(content: str) -> None:
    """Validate that content does not contain sensitive tokens, keys, or passwords.

    Raises:
        InvalidMemoryError: If sensitive patterns are detected in content.
    """
    if not content or not content.strip():
        raise InvalidMemoryError("Memory content cannot be empty.")

    for pattern in SENSITIVE_CONTENT_PATTERNS:
        if pattern.search(content):
            raise InvalidMemoryError(
                "Memory content contains sensitive credentials, API keys, or security tokens, which cannot be stored."
            )


def sanitize_fts_query(query: str) -> str:
    """Sanitize user search query for SQLite FTS5 matching.

    Extracts word tokens and wraps each in double quotes to prevent syntax errors
    from operators (AND, OR, NOT, NEAR), colons, or punctuation.
    """
    tokens = re.findall(r"\b[a-zA-Z0-9_\u0080-\uffff]+\b", query)
    if not tokens:
        return ""
    # Wrap each token in quotes to prevent FTS5 operator collisions
    return " ".join(f'"{t}"' for t in tokens)


class MemoryCategory(str, Enum):
    """Categorization for persistent memories."""
    PREFERENCE = "preference"
    FACT = "fact"
    ENTITY = "entity"
    INSTRUCTION = "instruction"
    OTHER = "other"


class MemorySourceType(str, Enum):
    """Origin source of the stored memory."""
    EXPLICIT_COMMAND = "explicit_command"
    EXTRACTED_CONVERSATION = "extracted_conversation"
    SYSTEM = "system"


class MemoryRecord(BaseModel):
    """A single persistent memory item."""
    id: str = Field(..., description="Unique memory identifier (UUID hex)")
    category: str = Field(..., description="Category: preference, fact, entity, instruction, other")
    key: str | None = Field(default=None, description="Optional normalized lookup key/topic")
    content: str = Field(..., description="Plain-text memory description")
    source_type: str = Field(default=MemorySourceType.EXPLICIT_COMMAND.value, description="Memory source")
    source_conversation_id: str | None = Field(default=None, description="Associated conversation ID if any")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    is_pinned: bool = Field(default=False, description="Whether memory is protected from pruning")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Sanitized metadata")
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)
    last_accessed_at: str | None = Field(default=None, description="Last access timestamp")


class MemoryStore:
    """Thread-safe SQLite store managing persistent user memories with FTS5 search."""

    def __init__(self, db_path: Path | str | None = None, enable_fts: bool = True) -> None:
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = get_data_dir() / "loki.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._enable_fts = enable_fts
        self._fts_available = False
        self._init_database()

    @property
    def fts_available(self) -> bool:
        """Whether SQLite FTS5 search is supported and active."""
        return self._fts_available

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Create and configure a SQLite connection with foreign keys enabled."""
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
        """Create memories table, indexes, and FTS5 virtual table if supported."""
        with self._get_connection() as conn:
            # 0. Ensure foreign-key target conversations table exists
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

            # 1. Base memories table
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    key TEXT,
                    content TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
                    confidence REAL DEFAULT 1.0,
                    is_pinned INTEGER DEFAULT 0,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
                CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
                CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC);
                """
            )

            # 2. FTS5 Virtual Table and Synchronizing Triggers (if enabled)
            if self._enable_fts:
                try:
                    conn.executescript(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                            content,
                            key,
                            category,
                            content='memories',
                            content_rowid='rowid'
                        );

                        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                            INSERT INTO memories_fts(rowid, content, key, category)
                            VALUES (new.rowid, new.content, new.key, new.category);
                        END;

                        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                            INSERT INTO memories_fts(memories_fts, rowid, content, key, category)
                            VALUES ('delete', old.rowid, old.content, old.key, old.category);
                        END;

                        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                            INSERT INTO memories_fts(memories_fts, rowid, content, key, category)
                            VALUES ('delete', old.rowid, old.content, old.key, old.category);
                            INSERT INTO memories_fts(rowid, content, key, category)
                            VALUES (new.rowid, new.content, new.key, new.category);
                        END;
                        """
                    )
                    self._fts_available = True
                except sqlite3.OperationalError as e:
                    logger.warning("SQLite FTS5 virtual table initialization failed (%s); using LIKE fallback.", e)
                    self._fts_available = False
            else:
                self._fts_available = False

    def store_memory(
        self,
        content: str,
        category: str = MemoryCategory.FACT.value,
        key: str | None = None,
        source_type: str = MemorySourceType.EXPLICIT_COMMAND.value,
        source_conversation_id: str | None = None,
        confidence: float = 1.0,
        is_pinned: bool = False,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
        upsert_key: bool = False,
    ) -> MemoryRecord:
        """Create or upsert a persistent memory item.

        Args:
            content: Meaningful memory text.
            category: Category name.
            key: Optional normalized topic or identifier.
            source_type: Origin type of memory.
            source_conversation_id: Optional reference to an originating conversation.
            confidence: Confidence score between 0.0 and 1.0.
            is_pinned: Protected from automatic pruning.
            metadata: Additional sanitized metadata dictionary.
            memory_id: Optional explicit UUID (generated if omitted).
            upsert_key: If True and a memory with the same key and category exists, update it.
        """
        clean_content = content.strip() if content else ""
        validate_memory_content_safety(clean_content)

        clean_cat = category.strip().lower() if category else ""
        if not clean_cat:
            raise InvalidMemoryError("Memory category cannot be empty.")

        clean_key = key.strip().lower() if key and key.strip() else None
        safe_meta = sanitize_memory_metadata(metadata)
        meta_str = json.dumps(safe_meta) if safe_meta else None
        now = _utc_now_iso()
        bounded_conf = max(0.0, min(float(confidence), 1.0))
        pinned_int = 1 if is_pinned else 0
        updated_target_id = None
        mid = memory_id or uuid.uuid4().hex

        try:
            with self._get_connection() as conn:
                # Upsert by key if requested
                if upsert_key and clean_key:
                    existing = conn.execute(
                        "SELECT id FROM memories WHERE key = ? AND category = ?",
                        (clean_key, clean_cat),
                    ).fetchone()
                    if existing:
                        updated_target_id = existing["id"]
                        conn.execute(
                            """
                            UPDATE memories
                            SET content = ?, source_type = ?, source_conversation_id = ?,
                                confidence = ?, is_pinned = ?, metadata = ?, updated_at = ?
                            WHERE id = ?
                            """,
                            (clean_content, source_type, source_conversation_id, bounded_conf, pinned_int, meta_str, now, updated_target_id),
                        )

                if updated_target_id is None:
                    conn.execute(
                        """
                        INSERT INTO memories (
                            id, category, key, content, source_type, source_conversation_id,
                            confidence, is_pinned, metadata, created_at, updated_at, last_accessed_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            mid,
                            clean_cat,
                            clean_key,
                            clean_content,
                            source_type,
                            source_conversation_id,
                            bounded_conf,
                            pinned_int,
                            meta_str,
                            now,
                            now,
                            None,
                        ),
                    )
        except sqlite3.Error as e:
            raise StorageError(f"Failed to store memory: {e}") from e

        if updated_target_id is not None:
            return self.get_memory(updated_target_id)  # type: ignore[return-value]

        return MemoryRecord(
            id=mid,
            category=clean_cat,
            key=clean_key,
            content=clean_content,
            source_type=source_type,
            source_conversation_id=source_conversation_id,
            confidence=bounded_conf,
            is_pinned=is_pinned,
            metadata=safe_meta,
            created_at=now,
            updated_at=now,
            last_accessed_at=None,
        )

    def get_memory(self, memory_id: str, touch: bool = False) -> MemoryRecord | None:
        """Fetch a single memory item by ID."""
        clean_id = memory_id.strip() if memory_id else ""
        if not clean_id:
            return None

        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT id, category, key, content, source_type, source_conversation_id,
                           confidence, is_pinned, metadata, created_at, updated_at, last_accessed_at
                    FROM memories
                    WHERE id = ?
                    """,
                    (clean_id,),
                ).fetchone()

                if not row:
                    return None

                accessed_at = row["last_accessed_at"]
                if touch:
                    accessed_at = _utc_now_iso()
                    conn.execute("UPDATE memories SET last_accessed_at = ? WHERE id = ?", (accessed_at, clean_id))

                meta_dict = {}
                if row["metadata"]:
                    try:
                        meta_dict = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        meta_dict = {}

                return MemoryRecord(
                    id=row["id"],
                    category=row["category"],
                    key=row["key"],
                    content=row["content"],
                    source_type=row["source_type"],
                    source_conversation_id=row["source_conversation_id"],
                    confidence=row["confidence"],
                    is_pinned=bool(row["is_pinned"]),
                    metadata=meta_dict,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    last_accessed_at=accessed_at,
                )
        except sqlite3.Error as e:
            raise StorageError(f"Failed to retrieve memory {clean_id}: {e}") from e

    def get_memory_by_key(self, key: str, category: str | None = None, touch: bool = False) -> MemoryRecord | None:
        """Fetch a memory by its normalized key."""
        clean_key = key.strip().lower() if key else ""
        if not clean_key:
            return None

        try:
            with self._get_connection() as conn:
                if category:
                    row = conn.execute(
                        "SELECT id FROM memories WHERE key = ? AND category = ? LIMIT 1",
                        (clean_key, category.strip().lower()),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT id FROM memories WHERE key = ? LIMIT 1",
                        (clean_key,),
                    ).fetchone()

                if not row:
                    return None
                return self.get_memory(row["id"], touch=touch)
        except sqlite3.Error as e:
            raise StorageError(f"Failed to query memory by key '{clean_key}': {e}") from e

    def list_memories(
        self,
        category: str | None = None,
        is_pinned: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        """List stored memories with optional category and pinned filters."""
        bounded_limit = max(1, min(limit, 500))
        bounded_offset = max(0, offset)

        query_clauses = ["1=1"]
        params: list[Any] = []

        if category:
            query_clauses.append("category = ?")
            params.append(category.strip().lower())

        if is_pinned is not None:
            query_clauses.append("is_pinned = ?")
            params.append(1 if is_pinned else 0)

        params.extend([bounded_limit, bounded_offset])
        sql = f"""
            SELECT id, category, key, content, source_type, source_conversation_id,
                   confidence, is_pinned, metadata, created_at, updated_at, last_accessed_at
            FROM memories
            WHERE {' AND '.join(query_clauses)}
            ORDER BY is_pinned DESC, updated_at DESC
            LIMIT ? OFFSET ?
        """

        try:
            with self._get_connection() as conn:
                rows = conn.execute(sql, params).fetchall()
                memories: list[MemoryRecord] = []
                for row in rows:
                    meta_dict = {}
                    if row["metadata"]:
                        try:
                            meta_dict = json.loads(row["metadata"])
                        except json.JSONDecodeError:
                            meta_dict = {}
                    memories.append(
                        MemoryRecord(
                            id=row["id"],
                            category=row["category"],
                            key=row["key"],
                            content=row["content"],
                            source_type=row["source_type"],
                            source_conversation_id=row["source_conversation_id"],
                            confidence=row["confidence"],
                            is_pinned=bool(row["is_pinned"]),
                            metadata=meta_dict,
                            created_at=row["created_at"],
                            updated_at=row["updated_at"],
                            last_accessed_at=row["last_accessed_at"],
                        )
                    )
                return memories
        except sqlite3.Error as e:
            raise StorageError(f"Failed to list memories: {e}") from e

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        key: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
        is_pinned: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Update an existing memory item."""
        clean_id = memory_id.strip() if memory_id else ""
        if not clean_id:
            raise MemoryNotFoundError("Memory ID cannot be empty.")

        existing = self.get_memory(clean_id)
        if not existing:
            raise MemoryNotFoundError(f"Memory with ID '{clean_id}' not found.")

        updates: list[str] = []
        params: list[Any] = []

        if content is not None:
            clean_content = content.strip()
            validate_memory_content_safety(clean_content)
            updates.append("content = ?")
            params.append(clean_content)

        if key is not None:
            clean_key = key.strip().lower() if key.strip() else None
            updates.append("key = ?")
            params.append(clean_key)

        if category is not None:
            clean_cat = category.strip().lower()
            if not clean_cat:
                raise InvalidMemoryError("Memory category cannot be empty.")
            updates.append("category = ?")
            params.append(clean_cat)

        if confidence is not None:
            bounded_conf = max(0.0, min(float(confidence), 1.0))
            updates.append("confidence = ?")
            params.append(bounded_conf)

        if is_pinned is not None:
            updates.append("is_pinned = ?")
            params.append(1 if is_pinned else 0)

        if metadata is not None:
            safe_meta = sanitize_memory_metadata(metadata)
            updates.append("metadata = ?")
            params.append(json.dumps(safe_meta) if safe_meta else None)

        if not updates:
            return existing

        now = _utc_now_iso()
        updates.append("updated_at = ?")
        params.append(now)
        params.append(clean_id)

        sql = f"UPDATE memories SET {', '.join(updates)} WHERE id = ?"
        try:
            with self._get_connection() as conn:
                conn.execute(sql, params)
            return self.get_memory(clean_id)  # type: ignore[return-value]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to update memory {clean_id}: {e}") from e

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory item by ID."""
        clean_id = memory_id.strip() if memory_id else ""
        if not clean_id:
            return False

        try:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM memories WHERE id = ?", (clean_id,))
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise StorageError(f"Failed to delete memory {clean_id}: {e}") from e

    def clear_memories(self, category: str | None = None) -> int:
        """Delete all memories (or memories of a specific category)."""
        try:
            with self._get_connection() as conn:
                if category:
                    cursor = conn.execute("DELETE FROM memories WHERE category = ?", (category.strip().lower(),))
                else:
                    cursor = conn.execute("DELETE FROM memories")
                deleted_count = cursor.rowcount

                # Ensure FTS5 external index is cleanly rebuilt if FTS is active
                if self._fts_available:
                    try:
                        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild');")
                    except sqlite3.OperationalError:
                        pass
                return deleted_count
        except sqlite3.Error as e:
            raise StorageError(f"Failed to clear memories: {e}") from e

    def search_memories(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
    ) -> list[MemoryRecord]:
        """Search memories using SQLite FTS5 with BM25 ranking, falling back to LIKE.

        Args:
            query: User search text or keyword.
            limit: Maximum results to return (bounded 1 to 100).
            category: Optional category filter.
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return []

        bounded_limit = max(1, min(limit, 100))

        # 1. Try SQLite FTS5 Match
        if self._fts_available:
            sanitized_fts = sanitize_fts_query(clean_query)
            if sanitized_fts:
                try:
                    params: list[Any] = [sanitized_fts]
                    cat_clause = ""
                    if category:
                        cat_clause = "AND m.category = ?"
                        params.append(category.strip().lower())
                    params.append(bounded_limit)

                    sql = f"""
                        SELECT m.id, m.category, m.key, m.content, m.source_type, m.source_conversation_id,
                               m.confidence, m.is_pinned, m.metadata, m.created_at, m.updated_at, m.last_accessed_at,
                               memories_fts.rank
                        FROM memories m
                        JOIN memories_fts ON m.rowid = memories_fts.rowid
                        WHERE memories_fts MATCH ?
                        {cat_clause}
                        ORDER BY memories_fts.rank ASC, m.is_pinned DESC, m.updated_at DESC
                        LIMIT ?
                    """
                    with self._get_connection() as conn:
                        rows = conn.execute(sql, params).fetchall()
                        results: list[MemoryRecord] = []
                        for row in rows:
                            meta_dict = {}
                            if row["metadata"]:
                                try:
                                    meta_dict = json.loads(row["metadata"])
                                except json.JSONDecodeError:
                                    meta_dict = {}
                            results.append(
                                MemoryRecord(
                                    id=row["id"],
                                    category=row["category"],
                                    key=row["key"],
                                    content=row["content"],
                                    source_type=row["source_type"],
                                    source_conversation_id=row["source_conversation_id"],
                                    confidence=row["confidence"],
                                    is_pinned=bool(row["is_pinned"]),
                                    metadata=meta_dict,
                                    created_at=row["created_at"],
                                    updated_at=row["updated_at"],
                                    last_accessed_at=row["last_accessed_at"],
                                )
                            )
                        return results
                except sqlite3.OperationalError as e:
                    logger.debug("FTS5 match failed (%s); falling back to LIKE query.", e)

        # 2. Safe Parameterized Fallback Search (LIKE %term%)
        tokens = re.findall(r"\b[a-zA-Z0-9_\u0080-\uffff]+\b", clean_query)
        if not tokens:
            tokens = [clean_query]

        like_clauses: list[str] = []
        params_like: list[Any] = []
        for token in tokens[:5]:  # Bound tokens to avoid query explosion
            like_clauses.append("(content LIKE ? OR key LIKE ?)")
            like_pattern = f"%{token}%"
            params_like.extend([like_pattern, like_pattern])

        cat_clause = ""
        if category:
            cat_clause = "AND category = ?"
            params_like.append(category.strip().lower())

        params_like.append(bounded_limit)
        sql_like = f"""
            SELECT id, category, key, content, source_type, source_conversation_id,
                   confidence, is_pinned, metadata, created_at, updated_at, last_accessed_at
            FROM memories
            WHERE ({' OR '.join(like_clauses)})
            {cat_clause}
            ORDER BY is_pinned DESC, updated_at DESC
            LIMIT ?
        """

        try:
            with self._get_connection() as conn:
                rows = conn.execute(sql_like, params_like).fetchall()
                results = []
                for row in rows:
                    meta_dict = {}
                    if row["metadata"]:
                        try:
                            meta_dict = json.loads(row["metadata"])
                        except json.JSONDecodeError:
                            meta_dict = {}
                    results.append(
                        MemoryRecord(
                            id=row["id"],
                            category=row["category"],
                            key=row["key"],
                            content=row["content"],
                            source_type=row["source_type"],
                            source_conversation_id=row["source_conversation_id"],
                            confidence=row["confidence"],
                            is_pinned=bool(row["is_pinned"]),
                            metadata=meta_dict,
                            created_at=row["created_at"],
                            updated_at=row["updated_at"],
                            last_accessed_at=row["last_accessed_at"],
                        )
                    )
                return results
        except sqlite3.Error as e:
            raise StorageError(f"Memory fallback search failed: {e}") from e
