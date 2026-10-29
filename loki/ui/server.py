"""Local HTTP and API server bridging LokiAgent to the desktop UI.

Strict invariants:
- Bound strictly to local loopback (127.0.0.1). Zero external network exposure.
- Directly reuses LokiAgent.handle_input() without duplicating routing or safety logic.
- Returns actual live system telemetry (RAM, CPU via psutil) and genuine Ollama status.
- Preserves safety gatekeeper confirmation workflows for gated actions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import psutil

from ..agent import AgentResponse, LokiAgent, ResponseSource
from ..core.config import LokiConfig
from ..core.exceptions import ConversationNotFoundError, StorageError
from ..core.logging import get_logger
from ..storage import (
    ConversationDetail,
    ConversationStore,
    ConversationSummary,
    MessageRecord,
)

logger = get_logger("ui.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class AttachmentPayload(BaseModel):
    """Uploaded attachment reference."""
    filename: str = Field(..., description="Original attachment file name")
    path: str = Field(..., description="Staged file path on disk")
    size_bytes: int = Field(default=0, description="File size in bytes")


class ChatRequest(BaseModel):
    """Payload for user interaction with LokiAgent."""
    message: str = Field(..., description="User prompt or command text")
    action_id: str | None = Field(default=None, description="Confirmation token if approving a gated action")
    chat_mode: bool = Field(default=True, description="Whether to retain multi-turn context")
    conversation_id: str | None = Field(default=None, description="Target conversation session ID")
    attachment: AttachmentPayload | None = Field(default=None, description="Optional attached document payload")


class ConfirmRequest(BaseModel):
    """Payload for approving a pending gated action."""
    action_id: str = Field(..., description="Confirmation token identifier")
    original_message: str = Field(..., description="Original command string to execute")
    conversation_id: str | None = Field(default=None, description="Target conversation session ID")


class CancelRequest(BaseModel):
    """Payload for canceling a pending gated action."""
    action_id: str = Field(..., description="Confirmation token identifier")


class CreateConversationRequest(BaseModel):
    """Payload for creating a new conversation."""
    title: str = Field(default="New Chat", description="Initial title of conversation")


class UpdateConversationRequest(BaseModel):
    """Payload for renaming a conversation."""
    title: str = Field(..., min_length=1, description="Updated conversation title")


def create_ui_app(
    agent: LokiAgent | None = None,
    storage: ConversationStore | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application for the LOKI desktop UI."""
    loki_agent = agent or LokiAgent()
    conv_store = storage or loki_agent.storage
    if conv_store is None:
        conv_store = ConversationStore()
    loki_agent.storage = conv_store

    app = FastAPI(
        title="LOKI Desktop Assistant",
        description="Local-only backend serving LOKI Desktop UI",
        version="0.1.0",
        docs_url=None,  # Disabled for offline desktop privacy
        redoc_url=None,
    )

    @app.middleware("http")
    async def add_no_cache_headers(request, call_next):
        """Prevent browser caching of desktop application UI assets."""
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/api/status")
    def get_status() -> dict[str, Any]:
        """Return real-time local system status and telemetry (never fabricated)."""
        # 1. Ollama status & model readiness
        ollama_online = loki_agent.client.check_health(timeout=1.5)
        ollama_version = loki_agent.client.get_version(timeout=1.5) if ollama_online else None

        active_model = loki_agent.config.models.active_model
        model_available = False
        effective_model = None
        installed_models: list[str] = []
        using_fallback = False

        if ollama_online:
            try:
                installed_models = loki_agent.client.list_local_models(timeout=1.5)
                model_available = loki_agent.client.is_model_available(active_model, timeout=1.5)
                if model_available:
                    effective_model = active_model
                elif loki_agent.allow_fallback:
                    for fb in loki_agent.config.models.fallback_models:
                        if loki_agent.client.is_model_available(fb, timeout=1.0):
                            effective_model = fb
                            using_fallback = True
                            model_available = True
                            break
            except Exception as e:
                logger.warning("Error resolving model status in /api/status: %s", e)

        # 2. Hardware telemetry
        vm = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=None)

        return {
            "ollama_online": ollama_online,
            "ollama_version": ollama_version,
            "ollama_host": loki_agent.client.host,
            "active_model": active_model,
            "model_available": model_available,
            "effective_model": effective_model,
            "installed_models": installed_models,
            "using_fallback": using_fallback,
            "hardware_tier": loki_agent.config.hardware.tier.value,
            "ram_total_gb": round(vm.total / (1024**3), 1),
            "ram_used_gb": round(vm.used / (1024**3), 1),
            "ram_percent": vm.percent,
            "cpu_cores": psutil.cpu_count(logical=False) or 1,
            "cpu_threads": psutil.cpu_count(logical=True) or 1,
            "cpu_percent": cpu_pct,
            "offline_mode_enforced": loki_agent.config.offline_mode_enforced,
            "history_length": len(loki_agent.get_history()),
            "active_conversation_id": loki_agent.active_conversation_id,
        }

    # =========================================================================
    # Conversation History API Endpoints
    # =========================================================================
    @app.get("/api/conversations")
    def list_conversations() -> dict[str, Any]:
        """List all conversations ordered from most recently active to oldest."""
        all_convs = conv_store.list_conversations()
        active_id = loki_agent.active_conversation_id
        filtered = [
            c for c in all_convs
            if c.message_count > 0 or (active_id is not None and c.id == active_id)
        ]
        return {"conversations": [c.model_dump() for c in filtered]}

    @app.post("/api/conversations")
    def create_conversation(req: CreateConversationRequest | None = None) -> dict[str, Any]:
        """Create a new conversation session and switch to it."""
        title = (req.title.strip() if req and req.title else None) or "New Chat"
        active_id = loki_agent.active_conversation_id
        if active_id and title == "New Chat":
            existing = conv_store.get_conversation(active_id)
            if existing and existing.title == "New Chat" and len(existing.messages) == 0:
                summary = ConversationSummary(
                    id=existing.id,
                    title=existing.title,
                    created_at=existing.created_at,
                    updated_at=existing.updated_at,
                    message_count=0,
                )
                return {"conversation": summary.model_dump()}

        summary = conv_store.create_conversation(title=title)
        loki_agent.switch_conversation(summary.id)
        return {"conversation": summary.model_dump()}

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation(conversation_id: str) -> dict[str, Any]:
        """Fetch detail and full message history for a conversation, switching active context."""
        detail = conv_store.get_conversation(conversation_id)
        if not detail:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation '{conversation_id}' not found.",
            )
        loki_agent.switch_conversation(conversation_id)
        return {"conversation": detail.model_dump()}

    @app.patch("/api/conversations/{conversation_id}")
    def update_conversation(
        conversation_id: str,
        req: UpdateConversationRequest,
    ) -> dict[str, Any]:
        """Rename an existing conversation."""
        clean_title = req.title.strip()
        if not clean_title:
            raise HTTPException(status_code=400, detail="Conversation title cannot be empty.")
        updated = conv_store.update_title(conversation_id, clean_title)
        if not updated:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation '{conversation_id}' not found.",
            )
        detail = conv_store.get_conversation(conversation_id)
        return {
            "conversation": detail.model_dump() if detail else None,
            "success": True,
        }

    @app.delete("/api/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str) -> dict[str, Any]:
        """Delete a conversation and all its messages."""
        deleted = conv_store.delete_conversation(conversation_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation '{conversation_id}' not found.",
            )
        if loki_agent.active_conversation_id == conversation_id:
            loki_agent.active_conversation_id = None
            loki_agent.clear_history()
        return {
            "success": True,
            "message": f"Conversation '{conversation_id}' deleted.",
        }

    # =========================================================================
    # Chat & Action Execution Endpoints
    # =========================================================================
    @app.post("/api/chat")
    def chat(req: ChatRequest) -> dict[str, Any]:
        """Process user message through LokiAgent and persist turn to history."""
        if req.conversation_id:
            target_conv_id = req.conversation_id
            existing = conv_store.get_conversation(target_conv_id)
            if not existing:
                created = conv_store.create_conversation(title="New Chat", conversation_id=target_conv_id)
                target_conv_id = created.id
        else:
            created = conv_store.create_conversation(title="New Chat")
            target_conv_id = created.id

        att_payload = req.attachment.model_dump() if req.attachment else None
        response = loki_agent.handle_input(
            text=req.message,
            action_id=req.action_id,
            chat_mode=req.chat_mode,
            conversation_id=target_conv_id,
            attachment=att_payload,
        )

        return {
            "text": response.text,
            "source": response.source.value,
            "matched_command": response.matched_command,
            "success": response.success,
            "model_used": response.model_used,
            "pending_confirmation": (
                response.pending_confirmation.model_dump()
                if response.pending_confirmation
                else None
            ),
            "metadata": response.metadata,
            "conversation_id": response.conversation_id or target_conv_id,
        }

    @app.post("/api/confirm")
    def confirm_action(req: ConfirmRequest) -> dict[str, Any]:
        """Approve and execute only the original server-side pending action."""
        pending = loki_agent.gatekeeper.get_pending_request(req.action_id)
        if pending is None:
            raise HTTPException(
                status_code=400,
                detail=f"Action ID '{req.action_id}' is invalid, expired, or already consumed.",
            )

        # Validate the tool BEFORE approving, so unsupported requests are never
        # marked as confirmed. Do not execute the browser-supplied original_message;
        # reconstruct the command from the server-authorized pending request.
        if pending.tool_name == "file_delete":
            if len(pending.targets) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file deletion confirmation targets.",
                )
            authorized_message = f"delete {pending.targets[0]}"
        else:
            raise HTTPException(
                status_code=400,
                detail="This confirmation action is not supported by the UI.",
            )

        # Verify that conversation IDs cannot redirect an approved action into an unintended context.
        if pending.conversation_id is not None and req.conversation_id is not None:
            if pending.conversation_id != req.conversation_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"Conversation ID mismatch: action '{req.action_id}' belongs to conversation '{pending.conversation_id}'.",
                )

        # Bind approval to the server's original tool, targets, and conversation.
        approved = loki_agent.confirm_action(
            req.action_id,
            expected_tool=pending.tool_name,
            expected_targets=pending.targets,
            expected_conversation_id=pending.conversation_id,
        )
        if not approved:
            raise HTTPException(
                status_code=400,
                detail="Confirmation could not be verified.",
            )

        target_conv_id = pending.conversation_id or req.conversation_id or loki_agent.active_conversation_id
        response = loki_agent.handle_input(
            text=authorized_message,
            action_id=req.action_id,
            conversation_id=target_conv_id,
        )

        return {
            "text": response.text,
            "source": response.source.value,
            "matched_command": response.matched_command,
            "success": response.success,
            "model_used": response.model_used,
            "pending_confirmation": None,
            "metadata": response.metadata,
            "conversation_id": response.conversation_id or target_conv_id,
        }

    @app.post("/api/cancel")
    def cancel_action(req: CancelRequest) -> dict[str, Any]:
        """Cancel a pending gated confirmation request."""
        canceled = loki_agent.cancel_action(req.action_id)
        return {
            "success": canceled,
            "message": f"Action '{req.action_id}' was {'canceled' if canceled else 'not found'}.",
        }

    @app.get("/api/history")
    def get_history() -> dict[str, Any]:
        """Return the multi-turn conversational history."""
        return {"history": loki_agent.get_history()}

    @app.post("/api/clear")
    def clear_history() -> dict[str, Any]:
        """Clear active conversation context."""
        loki_agent.clear_history()
        return {"success": True, "message": "Conversation history cleared."}

    @app.post("/api/upload")
    async def upload_attachment(file: UploadFile = File(...)) -> dict[str, Any]:
        """Safely stage an uploaded document attachment locally."""
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file missing filename.")

        raw_name = Path(file.filename).name.strip()
        if not raw_name or "\x00" in raw_name or "\n" in raw_name or "\r" in raw_name:
            raise HTTPException(status_code=400, detail="Invalid filename characters.")

        ext = Path(raw_name).suffix.lower()
        allowed_exts = {".txt", ".pdf", ".csv", ".md"}
        if ext not in allowed_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Supported formats: .txt, .pdf, .md, .csv.",
            )

        from ..documents.reader import DEFAULT_MAX_FILE_BYTES
        max_bytes = DEFAULT_MAX_FILE_BYTES
        content = bytearray()
        chunk_size = 1024 * 64

        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > max_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds maximum allowed limit of {max_bytes // (1024 * 1024)} MB.",
                )

        if len(content) == 0:
            raise HTTPException(status_code=400, detail=f"Attached file '{raw_name}' is empty (0 bytes).")

        from ..utils.paths import get_data_dir
        import uuid
        attachments_dir = get_data_dir() / "attachments"
        attachments_dir.mkdir(parents=True, exist_ok=True)

        unique_id = uuid.uuid4().hex[:8]
        safe_file_name = f"{unique_id}_{raw_name}"
        staged_path = attachments_dir / safe_file_name

        try:
            staged_path.write_bytes(content)
        except OSError as e:
            logger.error("Failed to write attachment to disk: %s", e)
            raise HTTPException(status_code=500, detail=f"Failed to save attachment: {e}")

        return {
            "success": True,
            "filename": raw_name,
            "path": str(staged_path.resolve()),
            "size_bytes": len(content),
        }

    # =========================================================================
    # Static Asset Serving (Zero-External-Network)
    # =========================================================================
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def serve_index() -> FileResponse:
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                return FileResponse(
                    index_path,
                    headers={
                        "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                        "Pragma": "no-cache",
                        "Expires": "0",
                    },
                )
            raise HTTPException(status_code=404, detail="index.html not found")

    return app