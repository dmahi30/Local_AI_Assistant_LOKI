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
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import psutil

from ..agent import AgentResponse, LokiAgent, ResponseSource
from ..core.config import LokiConfig
from ..core.logging import get_logger

logger = get_logger("ui.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    """Payload for user interaction with LokiAgent."""
    message: str = Field(..., description="User prompt or command text")
    action_id: str | None = Field(default=None, description="Confirmation token if approving a gated action")
    chat_mode: bool = Field(default=True, description="Whether to retain multi-turn context")


class ConfirmRequest(BaseModel):
    """Payload for approving a pending gated action."""
    action_id: str = Field(..., description="Confirmation token identifier")
    original_message: str = Field(..., description="Original command string to execute")


class CancelRequest(BaseModel):
    """Payload for canceling a pending gated action."""
    action_id: str = Field(..., description="Confirmation token identifier")


def create_ui_app(agent: LokiAgent | None = None) -> FastAPI:
    """Create and configure the FastAPI application for the LOKI desktop UI."""
    loki_agent = agent or LokiAgent()

    app = FastAPI(
        title="LOKI Desktop Assistant",
        description="Local-only backend serving LOKI Desktop UI",
        version="0.1.0",
        docs_url=None,  # Disabled for offline desktop privacy
        redoc_url=None,
    )

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
        }

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> dict[str, Any]:
        """Process user message through LokiAgent."""
        response = loki_agent.handle_input(
            text=req.message,
            action_id=req.action_id,
            chat_mode=req.chat_mode,
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
        }

    @app.post("/api/confirm")
    def confirm_action(req: ConfirmRequest) -> dict[str, Any]:
        """Approve a gated confirmation and execute the authorized command."""
        approved = loki_agent.confirm_action(req.action_id)
        if not approved:
            raise HTTPException(
                status_code=400,
                detail=f"Action ID '{req.action_id}' is invalid, expired, or already consumed.",
            )

        # Execute under gatekeeper with confirmed action_id
        response = loki_agent.handle_input(
            text=req.original_message,
            action_id=req.action_id,
        )

        return {
            "text": response.text,
            "source": response.source.value,
            "matched_command": response.matched_command,
            "success": response.success,
            "model_used": response.model_used,
            "pending_confirmation": None,
            "metadata": response.metadata,
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

    # Serve static assets
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def serve_index() -> FileResponse:
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            raise HTTPException(status_code=404, detail="index.html not found")

    return app
