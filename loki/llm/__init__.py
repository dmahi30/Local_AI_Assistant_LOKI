"""Local AI inference package for LOKI."""

from .client import (
    ALLOWED_LOCAL_HOSTS,
    OllamaClient,
    SafeLocalRedirectHandler,
    normalize_model_tag,
)
from .resolver import ModelResolutionResult, ModelResolver

__all__ = [
    "ALLOWED_LOCAL_HOSTS",
    "ModelResolutionResult",
    "ModelResolver",
    "OllamaClient",
    "SafeLocalRedirectHandler",
    "normalize_model_tag",
]
