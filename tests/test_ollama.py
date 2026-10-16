"""Unit tests for OllamaClient with 100% mocked network calls."""

import json
from unittest.mock import MagicMock, patch
import urllib.error
import urllib.request
import pytest

from loki.core.config import ModelRuntimeConfig
from loki.core.exceptions import (
    ModelError,
    ModelFallbackError,
    ModelInferenceError,
    ModelNotFoundError,
    ModelOutputParseError,
)
from loki.llm.client import (
    ALLOWED_LOCAL_HOSTS,
    OllamaClient,
    SafeLocalRedirectHandler,
    normalize_model_tag,
)


def test_ollama_client_enforces_local_host():
    """Verify that OllamaClient refuses external network endpoints."""
    # Remote host should raise ModelError
    cfg_remote = ModelRuntimeConfig(ollama_host="http://api.external-ai.com:11434")
    with pytest.raises(ModelError) as exc_info:
        OllamaClient(cfg_remote)
    assert "not a permitted loopback address" in str(exc_info.value)

    # Local hosts should be accepted
    for host in ["http://127.0.0.1:11434", "http://localhost:11434"]:
        client = OllamaClient(ModelRuntimeConfig(ollama_host=host))
        assert client.host == host


def test_check_health_and_version():
    client = OllamaClient()

    # Success case
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"version": "0.40.5"}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert client.check_health() is True
        assert client.get_version() == "0.40.5"

    # Offline / Connection Refused case
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        assert client.check_health() is False
        assert client.get_version() is None


def test_list_local_models():
    client = OllamaClient()
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "models": [{"name": "qwen2.5:0.5b"}, {"name": "nomic-embed-text:latest"}]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        models = client.list_local_models()
        assert "qwen2.5:0.5b" in models
        assert client.is_model_available("qwen2.5:0.5b") is True
        assert client.is_model_available("llama3:8b") is False


def test_generate_success():
    client = OllamaClient()
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "model": "qwen2.5:1.5b",
        "response": "Hello, I am LOKI."
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = client.generate("Introduce yourself")
        assert res == "Hello, I am LOKI."


def test_chat_success():
    client = OllamaClient()
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "model": "qwen2.5:1.5b",
        "message": {"role": "assistant", "content": "Sure, here is how Python works."}
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = client.chat([{"role": "user", "content": "Explain Python"}])
        assert res == "Sure, here is how Python works."


def test_generate_model_not_found():
    client = OllamaClient()
    err = urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/generate",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=MagicMock(read=lambda: b'{"error":"model \'missing_model:latest\' not found"}'),
    )

    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(ModelNotFoundError) as exc_info:
            client.generate("hello", model="missing_model:latest")
        assert "not found in local Ollama library" in str(exc_info.value)


def test_generate_timeout():
    client = OllamaClient()
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")):
        with pytest.raises(ModelInferenceError) as exc_info:
            client.generate("hello")
        assert "Failed to connect to local Ollama daemon" in str(exc_info.value)


def test_generate_malformed_json():
    client = OllamaClient()
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"{broken_json_not_valid"
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(ModelOutputParseError):
            client.generate("hello")


def test_generate_with_fallback_success():
    cfg = ModelRuntimeConfig(
        active_model="primary-missing:latest",
        fallback_models=["fallback-available:latest"],
    )
    client = OllamaClient(cfg)

    # First call (primary) raises 404, second call (fallback) succeeds
    err_404 = urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/generate",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=MagicMock(read=lambda: b'{"error":"model not found"}'),
    )

    mock_resp_success = MagicMock()
    mock_resp_success.read.return_value = json.dumps({
        "response": "Fallback response received."
    }).encode("utf-8")
    mock_resp_success.__enter__.return_value = mock_resp_success

    with patch("urllib.request.urlopen", side_effect=[err_404, mock_resp_success]):
        response, used_model = client.generate_with_fallback("test prompt")
        assert response == "Fallback response received."
        assert used_model == "fallback-available:latest"


def test_generate_with_fallback_all_fail():
    cfg = ModelRuntimeConfig(
        active_model="model1:latest",
        fallback_models=["model2:latest"],
    )
    client = OllamaClient(cfg)

    err = urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/generate",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=MagicMock(read=lambda: b'{"error":"model not found"}'),
    )

    with patch("urllib.request.urlopen", side_effect=[err, err]):
        with pytest.raises(ModelFallbackError) as exc_info:
            client.generate_with_fallback("test prompt")
        assert "All configured models failed" in str(exc_info.value)


# =========================================================================
# Phase 4 Hardening Regression Tests
# =========================================================================


def test_url_validation_schemes_and_hosts():
    """Verify strict HTTP, loopback host, valid port, and clean component validation."""
    # 1. Reject non-HTTP schemes
    for invalid_scheme in [
        "https://127.0.0.1:11434",
        "ftp://127.0.0.1:11434",
        "file:///etc/passwd",
        "ws://localhost:11434",
    ]:
        with pytest.raises(ModelError) as exc:
            OllamaClient(ModelRuntimeConfig(ollama_host=invalid_scheme))
        assert "scheme must be 'http'" in str(exc.value)

    # 2. Reject non-loopback hosts
    for non_local_host in [
        "http://example.com:11434",
        "http://192.168.1.100:11434",
        "http://10.0.0.1:11434",
    ]:
        with pytest.raises(ModelError) as exc:
            OllamaClient(ModelRuntimeConfig(ollama_host=non_local_host))
        assert "not a permitted loopback address" in str(exc.value)

    # 3. Reject invalid ports
    for bad_port_url in [
        "http://127.0.0.1:0",
        "http://127.0.0.1:70000",
        "http://localhost:invalid",
    ]:
        with pytest.raises(ModelError) as exc:
            OllamaClient(ModelRuntimeConfig(ollama_host=bad_port_url))
        assert "Invalid port" in str(exc.value) or "outside valid range" in str(exc.value)

    # 4. Reject unexpected URL components
    with pytest.raises(ModelError) as exc:
        OllamaClient(ModelRuntimeConfig(ollama_host="http://127.0.0.1:11434/api/v1"))
    assert "Unexpected path component" in str(exc.value)

    with pytest.raises(ModelError) as exc:
        OllamaClient(ModelRuntimeConfig(ollama_host="http://127.0.0.1:11434?token=secret"))
    assert "Unexpected query string" in str(exc.value)

    with pytest.raises(ModelError) as exc:
        OllamaClient(ModelRuntimeConfig(ollama_host="http://127.0.0.1:11434#section"))
    assert "Unexpected fragment" in str(exc.value)

    with pytest.raises(ModelError) as exc:
        OllamaClient(ModelRuntimeConfig(ollama_host="http://admin:pass@127.0.0.1:11434"))
    assert "User credentials" in str(exc.value)


def test_redirect_handler_blocks_non_local_destinations():
    """Verify SafeLocalRedirectHandler prevents redirects escaping local loopback."""
    handler = SafeLocalRedirectHandler()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate")

    # Redirect to external domain
    with pytest.raises(ModelError) as exc:
        handler.redirect_request(req, None, 302, "Found", {}, "http://evil.com/leak")
    assert "non-local destination" in str(exc.value)

    # Redirect to HTTPS
    with pytest.raises(ModelError) as exc:
        handler.redirect_request(req, None, 302, "Found", {}, "https://127.0.0.1:11434/api/generate")
    assert "non-local destination" in str(exc.value)

    # Redirect containing credentials
    with pytest.raises(ModelError) as exc:
        handler.redirect_request(req, None, 302, "Found", {}, "http://user:pass@127.0.0.1:11434/api/generate")
    assert "contains credentials" in str(exc.value)

    # Valid loopback redirect succeeds
    redirected_req = handler.redirect_request(req, None, 302, "Found", {}, "http://127.0.0.1:11434/api/redirected")
    assert redirected_req is not None
    assert redirected_req.full_url == "http://127.0.0.1:11434/api/redirected"


def test_exact_model_matching_and_tag_normalization():
    """Verify exact normalized tag matching and rejection of partial substring matches."""
    # Test normalization function
    assert normalize_model_tag("qwen2.5") == "qwen2.5:latest"
    assert normalize_model_tag("qwen2.5:latest") == "qwen2.5:latest"
    assert normalize_model_tag("qwen2.5:1.5b") == "qwen2.5:1.5b"
    assert normalize_model_tag("nomic-embed-text") == "nomic-embed-text:latest"

    client = OllamaClient()
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "models": [
            {"name": "qwen2.5:1.5b"},
            {"name": "llama3:latest"},
            {"name": "mistral:7b-instruct-q4_0"},
        ]
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        # Exact match
        assert client.is_model_available("qwen2.5:1.5b") is True

        # Omitted tag matching :latest
        assert client.is_model_available("llama3") is True
        assert client.is_model_available("llama3:latest") is True

        # Tag mismatch: qwen2.5 normalizes to qwen2.5:latest which is NOT qwen2.5:1.5b
        assert client.is_model_available("qwen2.5") is False

        # Substring / partial matches must be strictly rejected
        assert client.is_model_available("qwen") is False
        assert client.is_model_available("qwen2.5:1.5") is False
        assert client.is_model_available("qwen2.5:1.5b-custom") is False
        assert client.is_model_available("mistral") is False
        assert client.is_model_available("mistral:7b") is False
        assert client.is_model_available("mistral:7b-instruct-q4_0") is True


def test_http_404_classification_by_endpoint():
    """Verify 404 classification distinguishes missing endpoints from missing models."""
    client = OllamaClient()

    # 1. 404 on /api/version must raise ModelInferenceError, NOT ModelNotFoundError
    err_version = urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/version",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=MagicMock(read=lambda: b"404 page not found"),
    )
    with patch("urllib.request.urlopen", side_effect=err_version):
        with pytest.raises(ModelInferenceError) as exc_info:
            client._make_request("/api/version")
        assert not isinstance(exc_info.value, ModelNotFoundError)
        assert "returned HTTP 404" in str(exc_info.value)

    # 2. 404 on /api/tags must raise ModelInferenceError, NOT ModelNotFoundError
    err_tags = urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/tags",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=MagicMock(read=lambda: b"404 page not found"),
    )
    with patch("urllib.request.urlopen", side_effect=err_tags):
        with pytest.raises(ModelInferenceError) as exc_info:
            client.list_local_models()
        assert not isinstance(exc_info.value, ModelNotFoundError)

    # 3. 404 on /api/generate with model request must raise ModelNotFoundError
    err_gen = urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/generate",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=MagicMock(read=lambda: b'{"error":"model \'foo:bar\' not found"}'),
    )
    with patch("urllib.request.urlopen", side_effect=err_gen):
        with pytest.raises(ModelNotFoundError) as exc_info:
            client.generate("hello", model="foo:bar")
        assert "not found in local Ollama library" in str(exc_info.value)


def test_malformed_json_response_structures():
    """Verify malformed response bodies and schemas raise ModelOutputParseError."""
    client = OllamaClient()

    # 1. Response is a JSON list instead of object
    mock_list_resp = MagicMock()
    mock_list_resp.read.return_value = json.dumps([1, 2, 3]).encode("utf-8")
    mock_list_resp.__enter__.return_value = mock_list_resp

    with patch("urllib.request.urlopen", return_value=mock_list_resp):
        with pytest.raises(ModelOutputParseError) as exc_info:
            client._make_request("/api/tags")
        assert "Expected JSON object" in str(exc_info.value)

    # 2. /api/tags has invalid models field (string instead of list)
    mock_bad_tags = MagicMock()
    mock_bad_tags.read.return_value = json.dumps({"models": "not_a_list"}).encode("utf-8")
    mock_bad_tags.__enter__.return_value = mock_bad_tags

    with patch("urllib.request.urlopen", return_value=mock_bad_tags):
        with pytest.raises(ModelOutputParseError) as exc_info:
            client.list_local_models()
        assert "missing list field 'models'" in str(exc_info.value)

    # 3. /api/tags model entry is not a dict
    mock_bad_entry = MagicMock()
    mock_bad_entry.read.return_value = json.dumps({"models": ["string_entry"]}).encode("utf-8")
    mock_bad_entry.__enter__.return_value = mock_bad_entry

    with patch("urllib.request.urlopen", return_value=mock_bad_entry):
        with pytest.raises(ModelOutputParseError) as exc_info:
            client.list_local_models()
        assert "is not an object" in str(exc_info.value)

    # 4. /api/version returns non-string version
    mock_bad_version = MagicMock()
    mock_bad_version.read.return_value = json.dumps({"version": 12345}).encode("utf-8")
    mock_bad_version.__enter__.return_value = mock_bad_version

    with patch("urllib.request.urlopen", return_value=mock_bad_version):
        with pytest.raises(ModelOutputParseError):
            client.get_version()


def test_non_string_model_responses():
    """Verify non-string values for 'response' or 'message.content' raise ModelOutputParseError."""
    client = OllamaClient()

    # 1. /api/generate response is an integer
    mock_int_resp = MagicMock()
    mock_int_resp.read.return_value = json.dumps({"response": 12345}).encode("utf-8")
    mock_int_resp.__enter__.return_value = mock_int_resp

    with patch("urllib.request.urlopen", return_value=mock_int_resp):
        with pytest.raises(ModelOutputParseError) as exc:
            client.generate("hello")
        assert "must be a string" in str(exc.value)

    # 2. /api/chat message is not a dict
    mock_str_msg = MagicMock()
    mock_str_msg.read.return_value = json.dumps({"message": "just a string"}).encode("utf-8")
    mock_str_msg.__enter__.return_value = mock_str_msg

    with patch("urllib.request.urlopen", return_value=mock_str_msg):
        with pytest.raises(ModelOutputParseError) as exc:
            client.chat([{"role": "user", "content": "hello"}])
        assert "must be an object" in str(exc.value)

    # 3. /api/chat message.content is not a string (None)
    mock_none_content = MagicMock()
    mock_none_content.read.return_value = json.dumps({"message": {"content": None}}).encode("utf-8")
    mock_none_content.__enter__.return_value = mock_none_content

    with patch("urllib.request.urlopen", return_value=mock_none_content):
        with pytest.raises(ModelOutputParseError) as exc:
            client.chat([{"role": "user", "content": "hello"}])
        assert "must be a string" in str(exc.value)


def test_connection_failure_propagates_in_is_model_available():
    """Connection failure during availability check raises ModelInferenceError, not False."""
    client = OllamaClient()
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        with pytest.raises(ModelInferenceError) as exc_info:
            client.is_model_available("qwen2.5:1.5b")
        assert "Failed to connect to local Ollama daemon" in str(exc_info.value)
