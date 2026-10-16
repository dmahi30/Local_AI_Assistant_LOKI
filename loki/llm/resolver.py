"""Hardware-aware and local-library-aware model resolver for LOKI.

Validates model readiness against the local Ollama instance, checks hardware
tier recommendations, and enforces strict privacy and safety rules:
- Verifies local availability before initiating inference.
- Never downloads models silently or without user authorization.
- Never silently switches to unapproved or unexpected models.
- Gracefully handles offline daemon or missing model libraries with actionable messages.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from ..core.config import LokiConfig, ModelRuntimeConfig
from ..core.exceptions import (
    ModelError,
    ModelInferenceError,
    ModelOutputParseError,
)
from ..core.logging import get_logger
from ..core.types import HardwareTier
from ..hardware.classifier import HardwareTierClassifier
from ..hardware.models import HardwareSpec
from .client import OllamaClient

logger = get_logger("llm.resolver")


class ModelResolutionResult(BaseModel):
    """Result of resolving a target model against local hardware and Ollama daemon."""
    is_ready: bool = Field(..., description="Whether a valid model is confirmed ready for inference")
    effective_model: str | None = Field(
        default=None, description="Actual model name selected for inference"
    )
    installed_models: list[str] = Field(
        default_factory=list, description="All models currently discovered in local Ollama"
    )
    ollama_online: bool = Field(
        default=False, description="Whether the local Ollama daemon responded to health checks"
    )
    message: str = Field(
        default="", description="Human-readable status or actionable diagnostic message"
    )
    using_fallback: bool = Field(
        default=False, description="Whether an approved fallback was selected instead of primary"
    )


class ModelResolver:
    """Resolves models against local hardware capabilities and local Ollama daemon."""

    def __init__(
        self,
        config: LokiConfig | ModelRuntimeConfig | None = None,
        client: OllamaClient | None = None,
    ) -> None:
        if isinstance(config, LokiConfig):
            self.runtime_config = config.models
        elif isinstance(config, ModelRuntimeConfig):
            self.runtime_config = config
        else:
            self.runtime_config = ModelRuntimeConfig()

        self.client = client or OllamaClient(self.runtime_config)

    @staticmethod
    def get_tier_recommended_model(tier: HardwareTier) -> str:
        """Return the pre-calibrated default model for a given hardware tier."""
        tier_map = {
            HardwareTier.ECO: "qwen2.5:1.5b",
            HardwareTier.BALANCED_CPU: "qwen2.5:3b",
            HardwareTier.BALANCED_GPU: "qwen2.5:7b",
            HardwareTier.POWER: "qwen2.5:7b",
        }
        return tier_map.get(tier, "qwen2.5:1.5b")

    def check_health(self) -> bool:
        """Probe local Ollama daemon availability."""
        return self.client.check_health()

    def list_installed_models(self) -> list[str]:
        """Fetch list of all models installed in local Ollama library."""
        try:
            return self.client.list_local_models()
        except Exception as e:
            logger.warning("Could not list local Ollama models: %s", e)
            return []

    def resolve(
        self,
        requested_model: str | None = None,
        allow_fallback: bool = False,
    ) -> ModelResolutionResult:
        """Resolve a model for inference against the local Ollama daemon.

        Strict invariants:
        - Never triggers unprompted network downloads.
        - If the requested model is missing and allow_fallback is False, fails closed
          with a diagnostic message.
        - If allow_fallback is True, strictly checks explicitly configured fallback_models.
        """
        if not self.client.check_health():
            return ModelResolutionResult(
                is_ready=False,
                effective_model=None,
                installed_models=[],
                ollama_online=False,
                message=(
                    f"Local Ollama service is offline or unreachable at {self.client.host}. "
                    "Please ensure the Ollama daemon is running."
                ),
                using_fallback=False,
            )

        try:
            installed = self.client.list_local_models()
        except (ModelInferenceError, ModelOutputParseError, ModelError) as e:
            logger.warning("Could not list local Ollama models: %s", e)
            return ModelResolutionResult(
                is_ready=False,
                effective_model=None,
                installed_models=[],
                ollama_online=False,
                message=f"Failed to communicate with local Ollama daemon: {e}",
                using_fallback=False,
            )

        target = requested_model or self.runtime_config.active_model

        # 1. Primary check: Is target model installed?
        try:
            target_available = self.client.is_model_available(target)
        except (ModelInferenceError, ModelOutputParseError, ModelError) as e:
            return ModelResolutionResult(
                is_ready=False,
                effective_model=None,
                installed_models=installed,
                ollama_online=False,
                message=f"Failed to verify model availability with local Ollama daemon: {e}",
                using_fallback=False,
            )

        if target_available:
            return ModelResolutionResult(
                is_ready=True,
                effective_model=target,
                installed_models=installed,
                ollama_online=True,
                message=f"Model '{target}' is available locally.",
                using_fallback=False,
            )

        # 2. Fallback check: Only if explicitly permitted
        if allow_fallback:
            for fallback in self.runtime_config.fallback_models:
                try:
                    fallback_available = self.client.is_model_available(fallback)
                except (ModelInferenceError, ModelOutputParseError, ModelError):
                    fallback_available = False

                if fallback_available:
                    logger.info("Using approved fallback model '%s' instead of '%s'", fallback, target)
                    return ModelResolutionResult(
                        is_ready=True,
                        effective_model=fallback,
                        installed_models=installed,
                        ollama_online=True,
                        message=(
                            f"Primary model '{target}' is not installed locally. "
                            f"Operating using approved fallback '{fallback}'."
                        ),
                        using_fallback=True,
                    )

        # 3. Model is missing and no approved fallback is ready
        installed_str = ", ".join(installed) if installed else "None"
        return ModelResolutionResult(
            is_ready=False,
            effective_model=None,
            installed_models=installed,
            ollama_online=True,
            message=(
                f"Required model '{target}' is not installed in local Ollama library. "
                f"Run 'ollama pull {target}' to install it. "
                f"(Currently installed: {installed_str})"
            ),
            using_fallback=False,
        )

    def resolve_for_tier(
        self,
        tier: HardwareTier,
        allow_fallback: bool = False,
    ) -> ModelResolutionResult:
        """Resolve the model calibrated for a specific hardware tier."""
        target = self.get_tier_recommended_model(tier)
        return self.resolve(requested_model=target, allow_fallback=allow_fallback)

    def resolve_for_spec(
        self,
        spec: HardwareSpec,
        allow_fallback: bool = False,
    ) -> ModelResolutionResult:
        """Classify hardware spec and resolve the calibrated model."""
        classifier = HardwareTierClassifier()
        classification = classifier.classify(spec)
        return self.resolve(
            requested_model=classification.suggested_model,
            allow_fallback=allow_fallback,
        )
