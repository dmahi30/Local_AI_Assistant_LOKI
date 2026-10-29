"""Safety gatekeeper enforcing risk classification and confirmation gates for LOKI."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import uuid
from typing import Any

from ..core.config import SafetyConfig
from ..core.exceptions import SafetyViolationError
from ..core.logging import get_logger
from ..core.types import SafetyConfirmationRequest, SafetyLevel

logger = get_logger("safety.gatekeeper")

# Default Windows and generic system directories strictly protected from destructive actions
DEFAULT_PROTECTED_DIRECTORIES: list[str] = [
    r"C:\Windows",
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\Recovery",
    r"C:\Boot",
    r"C:\ProgramData\Microsoft",
    "/bin",
    "/sbin",
    "/usr",
    "/etc",
    "/lib",
    "/boot",
]


class SafetyGatekeeper:
    """Evaluates action safety, validates path boundaries, and manages user confirmations."""

    def __init__(
        self,
        config: SafetyConfig | None = None,
        protected_paths: list[str | Path] | None = None,
        allowed_search_roots: list[str | Path] | None = None,
    ) -> None:
        self.config = config or SafetyConfig()
        self.protected_paths = [
            Path(p).resolve() for p in (protected_paths or DEFAULT_PROTECTED_DIRECTORIES)
        ]
        self.allowed_search_roots = (
            [Path(p).resolve() for p in allowed_search_roots]
            if allowed_search_roots is not None
            else self._get_default_search_roots()
        )
        self._pending_confirmations: dict[str, SafetyConfirmationRequest] = {}

    @staticmethod
    def _is_subpath(child: Path, parent: Path) -> bool:
        """Check if child is equal to or located within parent directory.

        Handles Windows case-insensitivity and relative path normalization.
        """
        try:
            child_res = child.resolve()
            parent_res = parent.resolve()
            if sys.platform == "win32":
                child_parts = [p.lower() for p in child_res.parts]
                parent_parts = [p.lower() for p in parent_res.parts]
                return len(child_parts) >= len(parent_parts) and child_parts[:len(parent_parts)] == parent_parts
            return child_res == parent_res or parent_res in child_res.parents
        except Exception:
            return False

    @staticmethod
    def _get_default_search_roots() -> list[Path]:
        """Compute platform-appropriate safe user directories for search and file ops."""
        from ..utils.paths import get_all_known_folder_paths, get_data_dir, get_project_root

        user_home = Path.home().resolve()
        safe_roots = [user_home]

        # Add all resolved known folders (Desktop, Documents, Downloads, Pictures, etc., including OneDrive)
        for kf in get_all_known_folder_paths():
            if kf and kf.exists() and kf not in safe_roots:
                safe_roots.append(kf)

        for sub in ["Documents", "Downloads", "Desktop", "Pictures", "Videos", "Music"]:
            p = (user_home / sub).resolve()
            if p.exists() and p not in safe_roots:
                safe_roots.append(p)

        proj = get_project_root()
        if proj.exists() and proj not in safe_roots:
            safe_roots.append(proj)

        data_dir = get_data_dir()
        if data_dir.exists() and data_dir not in safe_roots:
            safe_roots.append(data_dir)

        # Include custom local app directory if set
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            safe_roots.append(Path(local_app).resolve() / "LOKI")
        return [r for r in safe_roots if r.exists()]

    def is_path_safe_for_read(self, target_path: str | Path) -> bool:
        """Verify that a path is safe to read/search.

        Must not be within protected system directories AND must reside under
        allowed search roots (whitelist containment).
        """
        try:
            resolved = Path(target_path).resolve()
        except Exception:
            return False

        # Strictly disallow reading protected Windows system cores
        for protected in self.protected_paths:
            if self._is_subpath(resolved, protected):
                logger.warning("Read blocked on protected system path: %s", resolved)
                return False

        # Enforce whitelist: must reside under at least one allowed search root
        if self.allowed_search_roots:
            is_under_allowed = any(
                self._is_subpath(resolved, root) for root in self.allowed_search_roots
            )
            if not is_under_allowed:
                logger.warning("Read blocked outside allowed search roots: %s", resolved)
                return False

        return True

    def is_path_safe_for_write(self, target_path: str | Path) -> bool:
        """Verify that a path is safe for modification or deletion.

        Must not be within protected system directories and must be located under
        allowed user locations.
        """
        try:
            resolved = Path(target_path).resolve()
        except Exception:
            return False

        # Disallow filesystem root directly (e.g. C:\ or /)
        if resolved == resolved.parent:
            logger.warning("Write blocked on filesystem root: %s", resolved)
            return False

        # Disallow protected directories
        for protected in self.protected_paths:
            if self._is_subpath(resolved, protected):
                logger.warning("Write blocked on protected path: %s", resolved)
                return False

        # Must reside under at least one allowed search/user root
        is_under_user_root = any(
            self._is_subpath(resolved, root) for root in self.allowed_search_roots
        )
        if not is_under_user_root:
            logger.warning("Write blocked outside allowed user directories: %s", resolved)
            return False

        return True

    def evaluate_risk(
        self,
        tool_name: str,
        declared_level: SafetyLevel,
        targets: list[str] | None = None,
    ) -> SafetyLevel:
        """Evaluate the effective safety level, dynamically escalating if targets are sensitive."""
        effective_level = declared_level

        # If any target path is outside safe areas, escalate to DANGEROUS or block
        if targets:
            for t in targets:
                if not self.is_path_safe_for_write(t):
                    effective_level = SafetyLevel.DANGEROUS
                    break

        return effective_level

    def _prune_expired_confirmations(self) -> None:
        """Remove any expired confirmation requests from memory."""
        now = time.time()
        expired_ids = [
            aid for aid, req in self._pending_confirmations.items()
            if now - req.created_at > req.ttl_seconds
        ]
        for aid in expired_ids:
            del self._pending_confirmations[aid]
            logger.info("Pruned expired confirmation action ID: %s", aid)

    def request_confirmation(
        self,
        tool_name: str,
        description: str,
        targets: list[str],
        safety_level: SafetyLevel = SafetyLevel.DANGEROUS,
        conversation_id: str | None = None,
    ) -> SafetyConfirmationRequest:
        """Create a pending confirmation request for a dangerous operation."""
        self._prune_expired_confirmations()
        action_id = f"act_{uuid.uuid4().hex[:8]}"
        # Canonicalize target paths
        canonical_targets = [str(Path(t).resolve()) for t in targets]
        ttl = getattr(self.config, "confirmation_ttl_seconds", 300)

        req = SafetyConfirmationRequest(
            action_id=action_id,
            tool_name=tool_name,
            description=description,
            targets=canonical_targets,
            safety_level=safety_level,
            confirmed=False,
            created_at=time.time(),
            ttl_seconds=ttl,
            conversation_id=conversation_id,
        )
        self._pending_confirmations[action_id] = req
        logger.warning(
            "Confirmation required for action [%s]: %s on %d target(s) (TTL: %ds)",
            action_id,
            description,
            len(targets),
            ttl,
        )
        return req

    def get_pending_request(self, action_id: str) -> SafetyConfirmationRequest | None:
        """Retrieve a pending confirmation request by its ID, pruning if expired."""
        self._prune_expired_confirmations()
        return self._pending_confirmations.get(action_id)

    def confirm_action(
        self,
        action_id: str,
        expected_tool: str | None = None,
        expected_targets: list[str] | None = None,
        expected_conversation_id: str | None = None,
    ) -> bool:
        """Approve a pending confirmation request with optional tool and target validation."""
        self._prune_expired_confirmations()
        req = self._pending_confirmations.get(action_id)
        if not req:
            logger.warning("Attempted to confirm unknown or expired action ID: %s", action_id)
            return False

        if expected_tool is not None and req.tool_name != expected_tool:
            logger.warning("Confirmation rejected: tool mismatch for %s (expected %s, got %s)", action_id, expected_tool, req.tool_name)
            return False

        if expected_targets is not None:
            req_targets = sorted([str(Path(t).resolve()).lower() for t in req.targets])
            exp_targets = sorted([str(Path(t).resolve()).lower() for t in expected_targets])
            if req_targets != exp_targets:
                logger.warning("Confirmation rejected: targets mismatch for %s", action_id)
                return False

        if (
            expected_conversation_id is not None
            and req.conversation_id is not None
            and req.conversation_id != expected_conversation_id
        ):
            logger.warning(
                "Confirmation rejected: conversation mismatch for %s (expected %s, got %s)",
                action_id,
                expected_conversation_id,
                req.conversation_id,
            )
            return False

        req.confirmed = True
        logger.info("Action [%s] approved by user", action_id)
        return True

    def cancel_action(self, action_id: str) -> bool:
        """Cancel and remove a pending confirmation request."""
        if action_id in self._pending_confirmations:
            del self._pending_confirmations[action_id]
            logger.info("Action [%s] canceled", action_id)
            return True
        return False

    def verify_execution_permitted(
        self,
        tool_name: str,
        safety_level: SafetyLevel,
        action_id: str | None = None,
        targets: list[str] | None = None,
    ) -> bool:
        """Verify whether an action is permitted to execute now.

        Strictly enforces:
        1. Action ID presence
        2. Action ID validity and non-consumption
        3. TTL expiration check
        4. User confirmation check
        5. Exact tool name binding
        6. Exact canonical target path binding (missing targets are rejected
           when the confirmation was approved for specific targets)
        """
        # If destructive operations require confirmation
        if safety_level == SafetyLevel.DANGEROUS and self.config.require_confirmation_for_destructive:
            if not action_id:
                raise SafetyViolationError(
                    f"Execution blocked: Action '{tool_name}' requires explicit confirmation.",
                    details={"tool_name": tool_name, "targets": targets or []},
                )

            req = self._pending_confirmations.get(action_id)
            if not req:
                raise SafetyViolationError(
                    f"Execution blocked: Action ID '{action_id}' is invalid, expired, or already consumed.",
                    details={"action_id": action_id, "tool_name": tool_name},
                )

            # Check TTL
            if time.time() - req.created_at > req.ttl_seconds:
                del self._pending_confirmations[action_id]
                raise SafetyViolationError(
                    f"Execution blocked: Action ID '{action_id}' has expired.",
                    details={"action_id": action_id, "tool_name": tool_name, "expired": True},
                )

            if not req.confirmed:
                raise SafetyViolationError(
                    f"Execution blocked: Action ID '{action_id}' is not confirmed.",
                    details={"action_id": action_id, "tool_name": tool_name, "confirmed": False},
                )

            # Check tool name binding
            if req.tool_name != tool_name:
                raise SafetyViolationError(
                    f"Execution blocked: Action ID '{action_id}' was authorized for '{req.tool_name}', not '{tool_name}'.",
                    details={
                        "action_id": action_id,
                        "expected_tool": req.tool_name,
                        "actual_tool": tool_name,
                    },
                )

            # Check targets binding. Missing targets must not bypass a
            # confirmation that was approved for specific targets.
            if req.targets and not targets:
                raise SafetyViolationError(
                    f"Execution blocked: Target information is missing for action '{action_id}'.",
                    details={
                        "action_id": action_id,
                        "expected_targets": req.targets,
                        "actual_targets": targets,
                    },
                )

            if targets is not None:
                req_targets = sorted([str(Path(t).resolve()).lower() for t in req.targets])
                actual_targets = sorted([str(Path(t).resolve()).lower() for t in targets])
                if req_targets != actual_targets:
                    raise SafetyViolationError(
                        f"Execution blocked: Target mismatch for action '{action_id}'.",
                        details={
                            "action_id": action_id,
                            "expected_targets": req.targets,
                            "actual_targets": targets,
                        },
                    )

            # Consume the confirmation once permitted so it cannot be re-used
            del self._pending_confirmations[action_id]

        return True