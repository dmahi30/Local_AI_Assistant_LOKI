"""LOKI: Hardware-adaptive, privacy-focused, offline-first local AI computer assistant."""

from .agent import AgentResponse, LokiAgent, ResponseSource

__version__ = "0.1.0"
__app_name__ = "LOKI"

__all__ = [
    "AgentResponse",
    "LokiAgent",
    "ResponseSource",
    "__app_name__",
    "__version__",
]
