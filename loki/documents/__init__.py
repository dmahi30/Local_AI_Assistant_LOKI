"""LOKI local document reading and document intelligence package."""

from .intelligence import (
    build_system_prompt_for_task,
    build_user_prompt,
    run_document_analysis,
    sanitize_untrusted_document_content,
)
from .reader import (
    DEFAULT_MAX_EXTRACTED_CHARS,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_PAGES,
    DocumentContent,
    DocumentReader,
    DocumentType,
)

__all__ = [
    "DEFAULT_MAX_EXTRACTED_CHARS",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_PAGES",
    "DocumentContent",
    "DocumentReader",
    "DocumentType",
    "build_system_prompt_for_task",
    "build_user_prompt",
    "run_document_analysis",
    "sanitize_untrusted_document_content",
]
