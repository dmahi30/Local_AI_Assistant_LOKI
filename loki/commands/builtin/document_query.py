"""Deterministic safe document reading and intelligence command for LOKI."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from ...core.types import IntentType, SafetyLevel, ToolResult
from ...documents.intelligence import run_document_analysis
from ...documents.reader import DocumentContent, DocumentReader
from ...llm.client import OllamaClient
from ...llm.resolver import ModelResolver
from ...safety.gatekeeper import SafetyGatekeeper
from ...utils.paths import clean_location_alias, get_default_search_roots, resolve_safe_directory
from ..base import BaseCommand

GENERIC_DOC_WORDS: set[str] = {
    "this", "that", "the", "a", "an", "file", "document", "pdf", "text", "image",
    "this document", "this file", "this pdf", "this text file", "the document", "the file", "the pdf",
}


class DocumentQueryCommand(BaseCommand):
    """Safely reads, summarizes, explains, or answers questions about local documents."""

    name = "document_query"
    description = "Safely reads, summarizes, explains, extracts key points, or answers questions about local documents (.txt, .md, .csv, .pdf, images)."
    intent_type = IntentType.DOCUMENT_QUERY
    safety_level = SafetyLevel.SAFE
    allow_model_selection = True

    patterns = [
        # 1. Summarize with explicit document/file indicator or file extension
        r"^(?:summarize|summary\s+of|give\s+me\s+a\s+summary\s+of)\s+(?:this\s+|the\s+)?(?:document|file|pdf|text\s+file|text|image)\s*(?:named|called)?\s*(?:['\"]?)([^\'\"\n]*?)(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+(.+))?$",
        r"^(?:summarize|summary\s+of|give\s+me\s+a\s+summary\s+of)\s+(?:this\s+|the\s+)?(?:['\"]?)([^\'\"\n\s]+\.[a-zA-Z0-9]+)(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+(.+))?$",

        # 2. Explain in simple words with explicit document/file indicator or file extension
        r"^(?:explain|describe)\s+(?:this\s+|the\s+)?(?:document|file|pdf|text\s+file|text|image)?\s*(?:in\s+simple\s+(?:words|terms))(?:\s*:\s*|\s+)(?:['\"]?)([^\'\"\n]+?)(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+(.+))?$",
        r"^(?:explain|describe)\s+(?:this\s+|the\s+)?(?:document|file|pdf|text\s+file|text|image)\s*(?:named|called)?\s*(?:['\"]?)([^\'\"\n]*?)(?:['\"]?)(?:\s+in\s+simple\s+(?:words|terms))?(?:\s+(?:in|on|at|inside|under)\s+(.+))?$",
        r"^(?:explain|describe)\s+(?:this\s+|the\s+)?(?:['\"]?)([^\'\"\n\s]+\.[a-zA-Z0-9]+)(?:['\"]?)(?:\s+in\s+simple\s+(?:words|terms))?(?:\s+(?:in|on|at|inside|under)\s+(.+))?$",

        # 3. Extract key points / dates and names with file extension or document indicator
        r"^(?:extract|get|find|list)\s+(?:the\s+)?(?:important|key|main)?\s*(?:points|dates|names|takeaways)\s+(?:from|in|of)\s+(?:this\s+|the\s+)?(?:document|file|pdf|text\s+file|text|image)\s*(?:named|called)?\s*(?:['\"]?)([^\'\"\n]*?)(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+(.+))?$",
        r"^(?:extract|get|find|list)\s+(?:the\s+)?(?:important|key|main)?\s*(?:points|dates|names|takeaways)\s+(?:from|in|of)\s+(?:this\s+|the\s+)?(?:['\"]?)([^\'\"\n\s]+\.[a-zA-Z0-9]+)(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+(.+))?$",
        r"^(?:what\s+are\s+the\s+(?:main|key|important)?\s*(?:dates\s+and\s+names|names\s+and\s+dates|dates|names|points|takeaways)\s+(?:in|from|of))\s+(?:this\s+|the\s+)?(?:document|file|pdf|text\s+file|text|image)\s*(?:named|called)?\s*(?:['\"]?)([^\'\"\n\?]+?)(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+([^\'\"\n\?]+))?\??$",
        r"^(?:what\s+are\s+the\s+(?:main|key|important)?\s*(?:dates\s+and\s+names|names\s+and\s+dates|dates|names|points|takeaways)\s+(?:in|from|of))\s+(?:this\s+|the\s+)?(?:['\"]?)([^\'\"\n\s\?]+\.[a-zA-Z0-9]+)(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+([^\'\"\n\?]+))?\??$",

        # 4. Image text extraction (OCR/Vision)
        r"^(?:read\s+this\s+image\s+and\s+extract\s+(?:the\s+)?text:?\s*)(?:['\"]?)([^\'\"\n]+?)(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+(.+))?$",
        r"^(?:read|extract\s+text\s+from)\s+(?:the\s+)?image\s*(?:named|called)?\s*(?:['\"]?)([^\'\"\n]+?)(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+(.+))?$",

        # 5. Direct read / display contents
        r"^(?:read|display|view|show|cat)\s+(?:this\s+|the\s+)?(?:document|file|pdf|text\s+file|text|image)\s*(?:named|called)?\s*(?:['\"]?)([^\'\"\n]+?)(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+(.+))?$",
        r"^(?:read|display|view|show|cat)\s+(?:this\s+|the\s+)?(?:['\"]?)([^\'\"\n\s]+\.[a-zA-Z0-9]+)(?:['\"]?)(?:\s+(?:in|on|at|inside|under)\s+(.+))?$",

        # 6. Q&A patterns: "Answer questions using this document: <file> : <question>"
        r"^(?:answer\s+questions?\s+using\s+(?:this\s+)?(?:document|file|pdf):?\s*)(?:['\"]?)([^\'\"\n:]+?)(?:['\"]?)(?:\s*[:\?-]\s*(.+))?$",
    ]

    def get_parameters_schema(self) -> dict[str, Any]:
        """Return JSON Schema for model-assisted tool calling."""
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 260,
                    "description": "Name or path of the document (e.g. 'report.pdf' or 'notes.txt').",
                },
                "location": {
                    "type": "string",
                    "maxLength": 260,
                    "description": "Safe folder alias (desktop, documents, downloads, project) or directory path.",
                },
                "task": {
                    "type": "string",
                    "enum": ["summarize", "explain", "key_points", "read", "qa"],
                    "description": "Operation to perform on document: summarize, explain, key_points, read, or qa.",
                },
                "question": {
                    "type": "string",
                    "maxLength": 1000,
                    "description": "Specific question to answer when task is 'qa'.",
                },
            },
            "required": ["target"],
            "additionalProperties": False,
        }

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        """Strictly validate document_query arguments against types and character safety."""
        if not isinstance(args, dict):
            return False, "Arguments must be a dictionary."

        allowed_keys = {"target", "location", "task", "question"}
        unexpected = set(args.keys()) - allowed_keys
        if unexpected:
            return False, f"Unexpected arguments provided: {', '.join(sorted(unexpected))}."

        target = args.get("target")
        if target is not None:
            if not isinstance(target, str):
                return False, "Target must be a string."
            if "\x00" in target or "\n" in target or "\r" in target:
                return False, "Target path contains invalid control characters."

        location = args.get("location")
        if location is not None:
            if not isinstance(location, str):
                return False, "Location must be a string."
            if "\x00" in location or "\n" in location or "\r" in location:
                return False, "Location contains invalid control characters."

        task = args.get("task", "summarize")
        if task not in ("summarize", "explain", "key_points", "read", "qa"):
            return False, f"Invalid task '{task}'. Supported tasks: summarize, explain, key_points, read, qa."

        return True, None

    def extract_args(self, text: str) -> dict[str, Any]:
        """Parse structured arguments from natural language input."""
        cleaned = text.strip().rstrip(".?").strip()

        # Determine task
        task = "summarize"
        cleaned_lower = cleaned.lower()
        if "simple words" in cleaned_lower or "explain" in cleaned_lower:
            task = "explain"
        elif any(k in cleaned_lower for k in ("important points", "key points", "dates and names", "main dates")):
            task = "key_points"
        elif any(k in cleaned_lower for k in ("read this image", "extract text from", "read document", "display", "show")):
            task = "read"
        elif "answer question" in cleaned_lower:
            task = "qa"

        for pattern in self._compiled_patterns:
            match = pattern.search(cleaned)
            if match:
                groups = match.groups()
                raw_target = groups[0].strip() if len(groups) > 0 and groups[0] else None
                raw_loc = groups[1].strip() if len(groups) > 1 and groups[1] else None

                # Clean generic words from target
                target = raw_target
                if target:
                    target_lower = target.lower()
                    if target_lower in GENERIC_DOC_WORDS:
                        target = None

                # Clean location
                location = clean_location_alias(raw_loc) if raw_loc else None

                question = None
                if task == "qa" and len(groups) > 1:
                    question = groups[1].strip()

                return {
                    "target": target,
                    "location": location,
                    "task": task,
                    "question": question,
                }

        # Fallback heuristic
        return {"target": None, "task": task}

    def execute(
        self,
        args: dict[str, Any],
        gatekeeper: SafetyGatekeeper | None = None,
        action_id: str | None = None,
    ) -> ToolResult:
        """Execute document query with execution time tracking and safe read boundary checks."""
        start_time = time.perf_counter()
        is_valid, validation_err = self.validate_args(args)
        if not is_valid:
            return ToolResult(
                success=False,
                error=validation_err or "Invalid command arguments.",
                safety_level=self.safety_level,
                execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
                metadata={"command": self.name},
            )

        try:
            res = self.run(args=args, gatekeeper=gatekeeper, action_id=action_id)
            res.execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return res
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Document query error: {e}",
                safety_level=self.safety_level,
                execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
                metadata={"command": self.name},
            )

    def _resolve_document_file(
        self,
        target: str,
        location: str | None,
        gatekeeper: SafetyGatekeeper | None,
    ) -> tuple[Path | None, str | None]:
        """Locate and resolve the target document, strictly enforcing safe read boundaries."""
        user_home = Path.home().resolve()

        # Case 1: Location explicitly specified
        if location:
            target_dir, err = resolve_safe_directory(location, gatekeeper=gatekeeper, for_write=False)
            if not target_dir:
                return None, err

            resolved_file = (target_dir / target).resolve()

            # Path traversal guard: ensure file resides inside target_dir
            if sys.platform == "win32":
                file_parts = [p.lower() for p in resolved_file.parts]
                dir_parts = [p.lower() for p in target_dir.parts]
                is_inside = len(file_parts) > len(dir_parts) and file_parts[: len(dir_parts)] == dir_parts
            else:
                is_inside = target_dir in resolved_file.parents

            if not is_inside:
                return None, f"Path traversal detected: '{target}' escapes allowed directory '{target_dir}'."

            if not resolved_file.exists():
                return None, f"Document does not exist: '{target}' in '{location}'."

            if gatekeeper and not gatekeeper.is_path_safe_for_read(resolved_file):
                return None, f"Access blocked: Document '{resolved_file}' is outside approved directories."

            return resolved_file, None

        # Case 2: Direct absolute or relative path provided
        if "/" in target or "\\" in target or Path(target).is_absolute():
            try:
                resolved_file = Path(target).resolve()
            except Exception:
                return None, f"Invalid document path: '{target}'."

            if not resolved_file.exists():
                return None, f"Document does not exist: '{target}'."

            if gatekeeper and not gatekeeper.is_path_safe_for_read(resolved_file):
                return None, f"Access blocked: Document '{resolved_file}' is outside approved directories."

            return resolved_file, None

        # Case 3: Bare filename provided without location. Search across allowed default roots and subdirs
        if gatekeeper and gatekeeper.allowed_search_roots:
            search_roots = gatekeeper.allowed_search_roots
        else:
            search_roots = get_default_search_roots()

        matched_files: list[Path] = []
        for root in search_roots:
            if not root.exists() or not root.is_dir():
                continue
            direct_candidate = (root / target).resolve()
            if direct_candidate.exists() and direct_candidate.is_file():
                if gatekeeper is None or gatekeeper.is_path_safe_for_read(direct_candidate):
                    if direct_candidate not in matched_files:
                        matched_files.append(direct_candidate)

            # Also search immediate subdirectories (e.g. Desktop, Documents, Downloads under safe root)
            try:
                for sub in root.iterdir():
                    if sub.is_dir():
                        sub_candidate = (sub / target).resolve()
                        if sub_candidate.exists() and sub_candidate.is_file():
                            if gatekeeper is None or gatekeeper.is_path_safe_for_read(sub_candidate):
                                if sub_candidate not in matched_files:
                                    matched_files.append(sub_candidate)
            except Exception:
                pass

        if len(matched_files) == 1:
            return matched_files[0], None
        elif len(matched_files) > 1:
            candidates_str = ", ".join(f"'{p.parent.name}/{p.name}'" for p in matched_files[:3])
            return None, (
                f"Multiple documents named '{target}' were found ({candidates_str}). "
                "Please specify which folder (e.g. 'Desktop' or 'Documents')."
            )

        return None, (
            f"Document '{target}' was not found in standard user directories (Desktop, Documents, Downloads, Project). "
            "Please specify the folder location."
        )

    def run(
        self,
        args: dict[str, Any],
        gatekeeper: SafetyGatekeeper | None = None,
        action_id: str | None = None,
        client: OllamaClient | None = None,
    ) -> ToolResult:
        """Execute safe document extraction and local intelligence."""
        target = args.get("target")
        location = args.get("location")
        task = str(args.get("task", "summarize")).lower()
        question = args.get("question")

        # Ambiguity handling
        if not target or not isinstance(target, str) or not target.strip():
            task_verb = {
                "summarize": "summarize",
                "explain": "explain",
                "key_points": "extract key points from",
                "read": "read",
                "qa": "answer questions about",
            }.get(task, "analyze")
            return ToolResult(
                success=False,
                error=f"Which document would you like to {task_verb}? Please specify the file name or path (e.g. 'Summarize report.pdf on Desktop').",
                safety_level=self.safety_level,
            )

        resolved_file, err = self._resolve_document_file(target.strip(), location, gatekeeper)
        if not resolved_file:
            return ToolResult(
                success=False,
                error=err or f"Could not find or access document: '{target}'.",
                safety_level=self.safety_level,
            )

        # Inspect local Ollama models if client available
        ollama_client = client or OllamaClient()
        installed_models: list[str] = []
        try:
            installed_models = ollama_client.list_local_models(timeout=2.0)
        except Exception:
            pass

        # Read and extract document content
        reader = DocumentReader()
        content: DocumentContent = reader.read(resolved_file, installed_models=installed_models)

        if content.error:
            return ToolResult(
                success=False,
                error=content.error,
                safety_level=self.safety_level,
                metadata={"document": resolved_file.name, "path": str(resolved_file)},
            )

        # Handle scanned PDF
        if content.is_scanned:
            return ToolResult(
                success=True,
                data={
                    "document": resolved_file.name,
                    "path": str(resolved_file),
                    "is_scanned": True,
                    "page_count": content.page_count,
                    "message": content.text,
                },
                safety_level=self.safety_level,
                metadata={"document": resolved_file.name, "path": str(resolved_file), "is_scanned": True},
            )

        # Handle raw text display
        if task == "read":
            return ToolResult(
                success=True,
                data={
                    "document": resolved_file.name,
                    "path": str(resolved_file),
                    "task": "read",
                    "document_text": content.text,
                    "page_count": content.page_count,
                    "size_bytes": content.size_bytes,
                },
                safety_level=self.safety_level,
                metadata={"document": resolved_file.name, "path": str(resolved_file), "task": "read"},
            )

        # Check local AI readiness for intelligence tasks (summarize, explain, key_points, qa)
        resolver = ModelResolver(client=ollama_client)
        resolution = resolver.resolve(allow_fallback=True)

        if not resolution.is_ready:
            # Return bounded extracted text along with the diagnostic model status
            preview = content.text[:500] + ("..." if len(content.text) > 500 else "")
            return ToolResult(
                success=False,
                error=(
                    f"Document '{resolved_file.name}' was extracted successfully ({content.page_count} page(s)), "
                    f"but local AI intelligence is currently unavailable: {resolution.message}\n\n"
                    f"Extracted content preview:\n{preview}"
                ),
                safety_level=self.safety_level,
                metadata={"document": resolved_file.name, "path": str(resolved_file), "ollama_online": resolution.ollama_online},
            )

        # Run local AI analysis
        analysis_text, analysis_err = run_document_analysis(
            content=content,
            task=task,
            question=question,
            client=ollama_client,
            model=resolution.effective_model,
        )

        if analysis_err:
            return ToolResult(
                success=False,
                error=analysis_err,
                safety_level=self.safety_level,
                metadata={"document": resolved_file.name, "path": str(resolved_file)},
            )

        return ToolResult(
            success=True,
            data={
                "document": resolved_file.name,
                "path": str(resolved_file),
                "task": task,
                "analysis": analysis_text,
                "page_count": content.page_count,
                "size_bytes": content.size_bytes,
                "model_used": resolution.effective_model,
            },
            safety_level=self.safety_level,
            metadata={
                "document": resolved_file.name,
                "path": str(resolved_file),
                "task": task,
                "model_used": resolution.effective_model,
            },
        )
