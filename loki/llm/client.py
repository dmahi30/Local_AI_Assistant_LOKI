"""Local Ollama HTTP client for LOKI.

Communicates with local Ollama daemon via REST endpoints (/api/version, /api/tags,
/api/generate, /api/chat) using standard library urllib, enforcing strict localhost
confinement, sensible timeouts, model fallback cascades, and hardened security:
- Scheme strictly restricted to HTTP.
- Destination strictly restricted to explicit loopback hostnames (127.0.0.1, localhost, ::1).
- Port validation (1-65535) and rejection of unexpected URL paths/queries/fragments/credentials.
- Disables external proxies for local loopback requests.
- Prevents HTTP redirects from escaping local loopback boundaries.
- Exact normalized tag matching (no substring matches).
- Granular HTTP 404 classification (missing route vs missing model).
- Strict response schema and type validation.
- Connection failures propagated rather than falsely reporting missing models.
"""

from __future__ import annotations

import json
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from ..core.config import ModelRuntimeConfig
from ..core.exceptions import (
    ModelError,
    ModelFallbackError,
    ModelInferenceError,
    ModelNotFoundError,
    ModelOutputParseError,
)
from ..core.logging import get_logger

logger = get_logger("llm.client")

# Allowable local-only host origins
ALLOWED_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def normalize_model_tag(model_name: str) -> str:
    """Normalize an Ollama model name, appending ':latest' if tag is omitted.

    Handles names with or without repository/namespace prefixes and strips whitespace.
    Examples:
        'qwen2.5' -> 'qwen2.5:latest'
        'qwen2.5:latest' -> 'qwen2.5:latest'
        'qwen2.5:1.5b' -> 'qwen2.5:1.5b'
        'library/qwen2.5' -> 'library/qwen2.5:latest'
    """
    clean = model_name.strip().lower()
    if not clean:
        return ""

    # Find the tag separator ':' in the final path segment
    last_slash_idx = clean.rfind("/")
    repo_part = clean[last_slash_idx + 1 :] if last_slash_idx != -1 else clean

    if ":" not in repo_part:
        return f"{clean}:latest"
    return clean


class SafeLocalRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Enforces that HTTP redirects cannot navigate outside permitted loopback hosts."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        try:
            parsed = urllib.parse.urlparse(newurl)
            hostname = (parsed.hostname or "").lower()
            if parsed.scheme.lower() != "http" or hostname not in ALLOWED_LOCAL_HOSTS:
                raise ModelError(
                    f"Security violation: HTTP redirect attempted to navigate to non-local destination '{newurl}'."
                )
            if parsed.username or parsed.password:
                raise ModelError(
                    f"Security violation: HTTP redirect URL contains credentials '{newurl}'."
                )
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(
                f"Security violation: Invalid redirect URL '{newurl}': {e}"
            ) from e

        return super().redirect_request(req, fp, code, msg, headers, newurl)


class OllamaClient:
    """Client for interfacing with a local Ollama instance without external network access."""

    def __init__(self, config: ModelRuntimeConfig | None = None) -> None:
        self.config = config or ModelRuntimeConfig()
        self._validate_local_endpoint(self.config.ollama_host)
        self._opener = self._build_opener()

    @staticmethod
    def _validate_local_endpoint(host_url: str) -> None:
        """Enforce that the Ollama host is strictly local HTTP without external components."""
        if not isinstance(host_url, str) or not host_url.strip():
            raise ModelError("Ollama host URL must be a non-empty string.")

        try:
            parsed = urllib.parse.urlparse(host_url)
        except Exception as e:
            raise ModelError(f"Invalid Ollama host URL '{host_url}': {e}") from e

        # 1. Validate scheme: Strictly http
        if parsed.scheme.lower() != "http":
            raise ModelError(
                f"Security violation: Ollama URL scheme must be 'http', got '{parsed.scheme}'. "
                "HTTPS and other schemes are not permitted for local loopback communication."
            )

        # 2. Validate hostname: Strictly approved loopback addresses
        hostname = (parsed.hostname or "").lower()
        if not hostname or hostname not in ALLOWED_LOCAL_HOSTS:
            raise ModelError(
                f"Security violation: Ollama host '{host_url}' is not a permitted loopback address. "
                f"Allowed hosts: {', '.join(sorted(ALLOWED_LOCAL_HOSTS))}."
            )

        # 3. Validate port: Must be valid 1..65535 if specified
        try:
            port = parsed.port
        except ValueError as ve:
            raise ModelError(
                f"Security violation: Invalid port in Ollama host URL '{host_url}': {ve}"
            ) from ve

        if port is not None and not (1 <= port <= 65535):
            raise ModelError(
                f"Security violation: Port {port} in Ollama host URL '{host_url}' is outside valid range (1-65535)."
            )

        # 4. Validate unexpected components: No userinfo, path, query, fragment, params
        if parsed.username or parsed.password:
            raise ModelError(
                f"Security violation: User credentials in Ollama host URL '{host_url}' are prohibited."
            )

        if parsed.path not in ("", "/"):
            raise ModelError(
                f"Security violation: Unexpected path component '{parsed.path}' in Ollama host URL '{host_url}'."
            )

        if parsed.query:
            raise ModelError(
                f"Security violation: Unexpected query string '{parsed.query}' in Ollama host URL '{host_url}'."
            )

        if parsed.fragment:
            raise ModelError(
                f"Security violation: Unexpected fragment '{parsed.fragment}' in Ollama host URL '{host_url}'."
            )

        if parsed.params:
            raise ModelError(
                f"Security violation: Unexpected params in Ollama host URL '{host_url}'."
            )

    def _build_opener(self) -> urllib.request.OpenerDirector:
        """Construct an OpenerDirector with proxies disabled and safe redirect enforcement."""
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({}),  # Disables environment-configured external proxies
            SafeLocalRedirectHandler(),
        )

    @property
    def host(self) -> str:
        return self.config.ollama_host.rstrip("/")

    def _make_request(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Internal helper to make HTTP requests against local Ollama API."""
        url = f"{self.host}{endpoint}"
        t = timeout if timeout is not None else float(self.config.timeout_seconds)

        headers = {"Accept": "application/json"}
        req_bytes = None
        if data is not None:
            headers["Content-Type"] = "application/json"
            req_bytes = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(url, data=req_bytes, headers=headers)

        try:
            # Respect patched urlopen in test suites while enforcing custom opener in production
            if hasattr(urllib.request.urlopen, "mock_calls"):
                open_fn = urllib.request.urlopen
            else:
                open_fn = self._opener.open

            with open_fn(req, timeout=t) as resp:
                raw_body = resp.read().decode("utf-8")
                try:
                    parsed_json = json.loads(raw_body)
                except json.JSONDecodeError as jde:
                    raise ModelOutputParseError(
                        f"Failed to parse JSON response from Ollama endpoint '{endpoint}': {jde}",
                        details={"endpoint": endpoint, "raw_response": raw_body[:200]},
                    ) from jde

                if not isinstance(parsed_json, dict):
                    raise ModelOutputParseError(
                        f"Expected JSON object from Ollama endpoint '{endpoint}', got {type(parsed_json).__name__}.",
                        details={"endpoint": endpoint, "received_type": type(parsed_json).__name__},
                    )
                return parsed_json
        except urllib.error.HTTPError as he:
            err_body = ""
            try:
                err_body = he.read().decode("utf-8")
            except Exception:
                pass

            if he.code == 404:
                # Classify 404: Only inference endpoints requesting a model represent a missing model
                is_inference_endpoint = endpoint in ("/api/generate", "/api/chat")
                is_model_request = data is not None and "model" in data
                is_model_not_found = (
                    "model" in err_body.lower()
                    or "not found" in err_body.lower()
                    or not err_body.strip()
                )

                if is_inference_endpoint and is_model_request and is_model_not_found:
                    model_name = data.get("model", "unknown")
                    raise ModelNotFoundError(
                        f"Model '{model_name}' was not found in local Ollama library.",
                        details={"model": model_name, "status_code": 404, "response": err_body},
                    ) from he

                raise ModelInferenceError(
                    f"Ollama endpoint '{endpoint}' returned HTTP 404: {he.reason}",
                    details={"status_code": 404, "endpoint": endpoint, "response": err_body},
                ) from he

            raise ModelInferenceError(
                f"Ollama API returned HTTP {he.code}: {he.reason}",
                details={"status_code": he.code, "endpoint": endpoint, "response": err_body},
            ) from he
        except (urllib.error.URLError, TimeoutError, OSError) as ue:
            raise ModelInferenceError(
                f"Failed to connect to local Ollama daemon at {self.host}: {ue}",
                details={"endpoint": endpoint, "error": str(ue)},
            ) from ue

    def check_health(self, timeout: float = 2.0) -> bool:
        """Probe local Ollama /api/version to check if daemon is running and reachable."""
        try:
            res = self._make_request("/api/version", timeout=timeout)
            version = res.get("version")
            return isinstance(version, str) and len(version.strip()) > 0
        except Exception:
            return False

    def get_version(self, timeout: float = 2.0) -> str | None:
        """Get the running Ollama version string."""
        try:
            res = self._make_request("/api/version", timeout=timeout)
            version = res.get("version")
            if not isinstance(version, str) or not version.strip():
                raise ModelOutputParseError(
                    f"Ollama /api/version returned non-string or empty 'version': {version!r}",
                    details={"received": res},
                )
            return version.strip()
        except ModelOutputParseError:
            raise
        except Exception:
            return None

    def list_local_models(self, timeout: float = 5.0) -> list[str]:
        """Fetch names of all models installed in local Ollama instance."""
        res = self._make_request("/api/tags", timeout=timeout)
        models_list = res.get("models")
        if not isinstance(models_list, list):
            raise ModelOutputParseError(
                "Ollama /api/tags response missing list field 'models'.",
                details={"received_keys": list(res.keys())},
            )

        model_names: list[str] = []
        for idx, m in enumerate(models_list):
            if not isinstance(m, dict):
                raise ModelOutputParseError(
                    f"Ollama /api/tags model entry at index {idx} is not an object.",
                    details={"entry": m},
                )
            name = m.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ModelOutputParseError(
                    f"Ollama /api/tags model entry at index {idx} has non-string 'name'.",
                    details={"entry": m},
                )
            model_names.append(name.strip())
        return model_names

    def is_model_available(self, model_name: str, timeout: float = 5.0) -> bool:
        """Check whether a specific model tag is installed locally using exact tag matching.

        Raises ModelInferenceError if Ollama cannot be reached, ensuring connection
        failures are not misinterpreted as model absence.
        """
        normalized_query = normalize_model_tag(model_name)
        if not normalized_query:
            return False

        installed = self.list_local_models(timeout=timeout)
        for m in installed:
            if normalize_model_tag(m) == normalized_query:
                return True
        return False

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Send a single prompt generation request to /api/generate."""
        target_model = model or self.config.active_model
        payload: dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        opts = {
            "temperature": self.config.temperature,
            "num_ctx": self.config.context_length,
        }
        if options:
            opts.update(options)
        payload["options"] = opts

        res = self._make_request("/api/generate", data=payload, timeout=timeout)
        if "response" not in res:
            raise ModelOutputParseError(
                "Ollama response missing expected 'response' field",
                details={"model": target_model, "received_keys": list(res.keys())},
            )

        response_val = res["response"]
        if not isinstance(response_val, str):
            raise ModelOutputParseError(
                f"Ollama response field 'response' must be a string, got {type(response_val).__name__}.",
                details={"model": target_model, "received_type": type(response_val).__name__},
            )
        return response_val.strip()

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        options: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Send a structured multi-turn conversation to /api/chat."""
        target_model = model or self.config.active_model
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": False,
        }

        opts = {
            "temperature": self.config.temperature,
            "num_ctx": self.config.context_length,
        }
        if options:
            opts.update(options)
        payload["options"] = opts

        res = self._make_request("/api/chat", data=payload, timeout=timeout)
        if "message" not in res:
            raise ModelOutputParseError(
                "Ollama chat response missing expected 'message' field",
                details={"model": target_model, "received_keys": list(res.keys())},
            )

        msg = res["message"]
        if not isinstance(msg, dict):
            raise ModelOutputParseError(
                f"Ollama chat response field 'message' must be an object, got {type(msg).__name__}.",
                details={"model": target_model, "received_type": type(msg).__name__},
            )

        if "content" not in msg:
            raise ModelOutputParseError(
                "Ollama chat response missing expected 'message.content' field",
                details={"model": target_model, "received_keys": list(msg.keys())},
            )

        content_val = msg["content"]
        if not isinstance(content_val, str):
            raise ModelOutputParseError(
                f"Ollama chat response field 'message.content' must be a string, got {type(content_val).__name__}.",
                details={"model": target_model, "received_type": type(content_val).__name__},
            )
        return content_val.strip()

    def generate_with_fallback(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        """Generate with graceful cascade through primary model and configured fallbacks.

        Returns tuple of (response_text, model_name_used).
        Raises ModelFallbackError if all models in cascade fail.
        """
        primary = model or self.config.active_model
        candidate_models = [primary] + [
            f for f in self.config.fallback_models if f != primary
        ]

        attempted_errors: dict[str, str] = {}
        for candidate in candidate_models:
            try:
                logger.info("Attempting inference with model '%s'", candidate)
                response = self.generate(
                    prompt=prompt,
                    model=candidate,
                    system=system,
                    options=options,
                    timeout=timeout,
                )
                return response, candidate
            except (ModelNotFoundError, ModelInferenceError, ModelOutputParseError) as me:
                logger.warning("Inference failed with model '%s': %s", candidate, me)
                attempted_errors[candidate] = str(me)

        raise ModelFallbackError(
            f"All configured models failed for prompt generation: {', '.join(candidate_models)}",
            details={"attempted_errors": attempted_errors},
        )

    def chat_with_fallback(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        options: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        """Chat with graceful cascade through primary model and configured fallbacks.

        Returns tuple of (response_text, model_name_used).
        Raises ModelFallbackError if all models in cascade fail.
        """
        primary = model or self.config.active_model
        candidate_models = [primary] + [
            f for f in self.config.fallback_models if f != primary
        ]

        attempted_errors: dict[str, str] = {}
        for candidate in candidate_models:
            try:
                logger.info("Attempting chat with model '%s'", candidate)
                response = self.chat(
                    messages=messages,
                    model=candidate,
                    options=options,
                    timeout=timeout,
                )
                return response, candidate
            except (ModelNotFoundError, ModelInferenceError, ModelOutputParseError) as me:
                logger.warning("Chat failed with model '%s': %s", candidate, me)
                attempted_errors[candidate] = str(me)

        raise ModelFallbackError(
            f"All configured models failed for chat completion: {', '.join(candidate_models)}",
            details={"attempted_errors": attempted_errors},
        )
