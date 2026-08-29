"""Verifier-backend routing: the right client and the right model id.

Run: pytest tests/test_verifier_client.py
No network — every client constructor here only builds an object.
"""

import os
import tempfile
from pathlib import Path

import pytest

from turbo_proxy.utils import (  # noqa: E402
    Config, ModelConfig, build_verifier_client, verifier_model_id,
)
from turbo_proxy.verifier.verifier import Verifier  # noqa: E402
from turbo_proxy.utils.verifier_client import (  # noqa: E402
    _append_base_query,
    _url_origin,
)


@pytest.fixture(autouse=True)
def _isolate_provider_environment(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("VERTEX_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)


def test_model_id_strips_the_backend_prefix():
    assert verifier_model_id(ModelConfig("deepseek/deepseek-v4-flash")) == \
        "deepseek-v4-flash"
    assert verifier_model_id(
        ModelConfig("gemini/gemini-2.5-flash", provider="vertex_ai")
    ) == "gemini-2.5-flash"
    assert verifier_model_id(ModelConfig(
        "openai/gpt-4o", api_key="test-key",
        base_url="http://localhost:7999/v1",
    )) == "gpt-4o"
    assert verifier_model_id(
        ModelConfig("vertex_ai/gemini-2.5-flash", provider="vertex_ai")
    ) == "gemini-2.5-flash"
    # An organization-style id needs an endpoint or explicit provider; it must
    # not fall through to llm-verifier's environment-based credential routing.
    with pytest.raises(ValueError, match="cannot determine.*provider"):
        verifier_model_id(ModelConfig("Qwen/Qwen3.5-9B"))


def test_deepseek_prefix_builds_a_tagged_deepseek_client():
    client = build_verifier_client(
        ModelConfig("deepseek/deepseek-v4-flash", api_key="test-key"))
    # The tag is what makes llm-verifier read DeepSeek's sampled score tags
    # instead of the vLLM-only prefill trick.
    assert getattr(client, "_llm_verifier_deepseek", False) is True
    # The prefix must be stripped before it reaches the API.
    assert client._llm_verifier_model == "deepseek-v4-flash"
    assert "api.deepseek.com" in str(client.base_url)


def test_default_deepseek_client_isolates_openai_environment_headers(
    monkeypatch,
):
    import httpx

    monkeypatch.setenv("DEEPSEEK_API_KEY", "model-key")
    monkeypatch.setenv("OPENAI_ORG_ID", "ambient-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-project")
    monkeypatch.setenv(
        "OPENAI_CUSTOM_HEADERS",
        "X-Ambient-Secret: ambient-header\n"
        "Authorization: Bearer ambient-key",
    )
    client = build_verifier_client(
        ModelConfig("deepseek/deepseek-v4-flash")
    )
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"data": []})

    client._client._transport = httpx.MockTransport(handle)
    try:
        client.models.list()
    finally:
        client.close()

    assert len(requests) == 1
    request = requests[0]
    assert request.headers["authorization"] == "Bearer model-key"
    assert "openai-organization" not in request.headers
    assert "openai-project" not in request.headers
    assert "x-ambient-secret" not in request.headers
    assert client._llm_verifier_model == "deepseek-v4-flash"
    assert client._llm_verifier_deepseek is True


def test_base_url_builds_an_openai_compatible_client():
    client = build_verifier_client(
        ModelConfig("Qwen/Qwen3.5-9B", api_key="test-key",
                    base_url="http://localhost:8000/v1"))
    assert getattr(client, "_llm_verifier_deepseek", False) is False
    assert "localhost:8000" in str(client.base_url)


def test_openai_compatible_client_preserves_base_url_query(monkeypatch):
    import httpx
    from openai import DEFAULT_TIMEOUT, DefaultHttpxClient

    client = build_verifier_client(ModelConfig(
        "Qwen/Qwen3.5-9B",
        api_key="test-key",
        base_url="https://gateway.example/root/v1?tenant=acme#ignored",
    ))
    requests = []

    assert isinstance(client._client, DefaultHttpxClient)
    assert client._client.timeout == DEFAULT_TIMEOUT

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"data": []})

    client._client._transport = httpx.MockTransport(handle)
    try:
        client.models.list()
    finally:
        client.close()

    assert len(requests) == 1
    assert str(requests[0].url) == (
        "https://gateway.example/root/v1/models?tenant=acme"
    )


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        (
            "https://gateway.example/v1",
            "https://gateway.example/v1/models",
        ),
        (
            "https://gateway.example/v1?tenant=acme",
            "https://gateway.example/v1/models?tenant=acme",
        ),
    ],
)
def test_custom_openai_client_isolates_environment_headers(
    monkeypatch, base_url, expected_url,
):
    import httpx

    monkeypatch.setenv("OPENAI_ORG_ID", "ambient-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-project")
    monkeypatch.setenv(
        "OPENAI_CUSTOM_HEADERS",
        "X-Ambient-Secret: ambient-header\n"
        "Authorization: Bearer ambient-key",
    )
    client = build_verifier_client(ModelConfig(
        "Qwen/Qwen3.5-9B",
        api_key="model-key",
        base_url=base_url,
    ))
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"data": []})

    client._client._transport = httpx.MockTransport(handle)
    try:
        client.models.list()
    finally:
        client.close()

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == expected_url
    assert request.headers["authorization"] == "Bearer model-key"
    assert "openai-organization" not in request.headers
    assert "openai-project" not in request.headers
    assert "x-ambient-secret" not in request.headers


def test_official_openai_progress_client_keeps_environment_headers(monkeypatch):
    import httpx

    monkeypatch.setenv("OPENAI_ORG_ID", "ambient-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-project")
    monkeypatch.setenv(
        "OPENAI_CUSTOM_HEADERS",
        "X-Ambient-Header: retained\nAuthorization: Bearer ambient-key",
    )
    client = build_verifier_client(
        ModelConfig("openai/gpt-4o", api_key="model-key"),
        purpose="progress",
    )
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"data": []})

    client._client._transport = httpx.MockTransport(handle)
    try:
        client.models.list()
    finally:
        client.close()

    assert len(requests) == 1
    request = requests[0]
    assert request.headers["authorization"] == "Bearer model-key"
    assert request.headers["openai-organization"] == "ambient-org"
    assert request.headers["openai-project"] == "ambient-project"
    assert request.headers["x-ambient-header"] == "retained"


def test_openai_compatible_client_encodes_unicode_base_query():
    import httpx

    client = build_verifier_client(ModelConfig(
        "Qwen/Qwen3.5-9B",
        api_key="test-key",
        base_url="https://gateway.example/v1?tenant=台灣",
    ))
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"data": []})

    client._client._transport = httpx.MockTransport(handle)
    try:
        client.models.list()
    finally:
        client.close()

    assert [str(request.url) for request in requests] == [
        "https://gateway.example/v1/models?tenant=%E5%8F%B0%E7%81%A3"
    ]


def test_openai_base_query_hook_is_idempotent_and_origin_scoped():
    import httpx

    raw_query = b"tenant=a&tenant=b&empty=&bare"
    origin = _url_origin("https://gateway.example/v1")
    same_origin = httpx.Request(
        "GET",
        "https://gateway.example/v1/chat/completions?request=value",
    )
    other_origin = httpx.Request(
        "GET",
        "https://redirect.example/v1/chat/completions",
    )

    _append_base_query(same_origin, raw_query, origin)
    _append_base_query(same_origin, raw_query, origin)
    _append_base_query(other_origin, raw_query, origin)

    assert str(same_origin.url) == (
        "https://gateway.example/v1/chat/completions"
        "?request=value&tenant=a&tenant=b&empty=&bare"
    )
    assert str(other_origin.url) == (
        "https://redirect.example/v1/chat/completions"
    )


@pytest.mark.parametrize(
    ("name", "base_url", "expected_url", "is_deepseek"),
    [
        (
            "Qwen/Qwen3.5-9B",
            "https://gateway.example/v1/chat/completions",
            "https://gateway.example/v1/chat/completions",
            False,
        ),
        (
            "Qwen/Qwen3.5-9B",
            "https://gateway.example/v1/chat/completions/?tenant=acme#ignored",
            "https://gateway.example/v1/chat/completions?tenant=acme",
            False,
        ),
        (
            "deepseek/custom-reasoner",
            "https://deepseek-proxy.example/v1/chat/completions?tenant=acme",
            "https://deepseek-proxy.example/v1/chat/completions?tenant=acme",
            True,
        ),
        (
            "Qwen/Qwen3.5-9B",
            "https://gateway.example/v1/chat/completions-api",
            "https://gateway.example/v1/chat/completions-api/chat/completions",
            False,
        ),
    ],
)
def test_openai_compatible_client_accepts_full_chat_completions_url(
    name, base_url, expected_url, is_deepseek,
):
    import httpx

    client = build_verifier_client(ModelConfig(
        name,
        api_key="test-key",
        base_url=base_url,
    ))
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [],
            },
        )

    client._client._transport = httpx.MockTransport(handle)
    try:
        client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "ping"}],
        )
    finally:
        client.close()

    assert [str(request.url) for request in requests] == [expected_url]
    assert client._llm_verifier_deepseek is is_deepseek


def test_deepseek_custom_base_url_preserves_deepseek_scoring_metadata():
    client = build_verifier_client(
        ModelConfig(
            "deepseek/custom-reasoner",
            api_key="test-key",
            base_url="http://localhost:8001/v1",
        )
    )
    assert "localhost:8001" in str(client.base_url)
    assert client._llm_verifier_deepseek is True
    assert client._llm_verifier_model == "custom-reasoner"


def test_deepseek_custom_client_dispatches_through_deepseek_path(monkeypatch):
    import llm_verifier.fine_grained_reward as reward

    client = build_verifier_client(
        ModelConfig(
            "deepseek/custom-reasoner",
            api_key="test-key",
            base_url="http://localhost:8002/v1",
        )
    )
    marker = ("deepseek", ["token"], [[("A", 0.0)]])
    monkeypatch.setattr(reward, "call_deepseek", lambda *args, **kwargs: marker)
    monkeypatch.setattr(
        reward,
        "call_openai",
        lambda *args, **kwargs: pytest.fail("OpenAI path must not be used"),
    )

    assert reward.call_verifier(client, "prompt") == marker


def test_explicit_openai_provider_overrides_deepseek_prefix():
    client = build_verifier_client(
        ModelConfig(
            "deepseek/served-model",
            provider="openai",
            api_key="test-key",
            base_url="http://localhost:8003/v1",
        )
    )

    assert getattr(client, "_llm_verifier_deepseek", False) is False
    assert "localhost:8003" in str(client.base_url)


def test_explicit_openai_provider_overrides_deepseek_url_heuristic():
    client = build_verifier_client(
        ModelConfig(
            "deepseek/served-model",
            provider="openai",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
        )
    )

    assert client._llm_verifier_deepseek is False
    assert client._llm_verifier_model == "served-model"


def test_explicit_deepseek_provider_replaces_the_prefix():
    client = build_verifier_client(
        ModelConfig(
            "openai/custom-reasoner",
            provider="deepseek",
            api_key="test-key",
        )
    )

    assert client._llm_verifier_deepseek is True
    assert client._llm_verifier_model == "custom-reasoner"
    assert "api.deepseek.com" in str(client.base_url)


def test_official_openai_is_rejected_as_tournament_verifier():
    with pytest.raises(ValueError, match="official OpenAI API"):
        build_verifier_client(
            ModelConfig("openai/gpt-4o", api_key="test-key")
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.openai.com/v1",
        "https://api.openai.com/v1/",
        "https://API.OPENAI.COM/v1?source=turbo-proxy",
        "https://api.openai.com./v1",
    ],
)
def test_official_openai_custom_origin_is_rejected_as_tournament_verifier(
    base_url,
):
    with pytest.raises(ValueError, match="official OpenAI API"):
        build_verifier_client(
            ModelConfig("openai/gpt-4o", api_key="test-key", base_url=base_url)
        )


def test_official_openai_remains_available_for_progress_monitor(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = build_verifier_client(
        ModelConfig("openai/gpt-4o", api_key="test-key"),
        purpose="progress",
    )
    assert str(client.base_url).rstrip("/") == "https://api.openai.com/v1"


def test_openai_progress_client_honors_openai_base_url(monkeypatch):
    import httpx

    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        (
            "https://progress.example/api/v1/chat/completions"
            "?tenant=progress#ignored"
        ),
    )
    client = build_verifier_client(
        ModelConfig("openai/gpt-4o", api_key="test-key"),
        purpose="progress",
    )
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [],
            },
        )

    client._client._transport = httpx.MockTransport(handle)
    try:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "ping"}],
        )
    finally:
        client.close()

    assert len(requests) == 1
    assert str(requests[0].url) == (
        "https://progress.example/api/v1/chat/completions?tenant=progress"
    )


def test_openai_progress_client_pins_the_configured_default_model(monkeypatch):
    import httpx
    import llm_verifier.fine_grained_reward as reward

    monkeypatch.setenv("OPENAI_BASE_URL", "https://progress.example/v1")
    client = build_verifier_client(
        ModelConfig("openai/gemini-2.5-flash", api_key="test-key"),
        purpose="progress",
    )
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": "wrong-model"}]})

    client._client._transport = httpx.MockTransport(handle)
    try:
        assert reward.resolve_model(client) == "gemini-2.5-flash"
    finally:
        client.close()

    assert requests == []


@pytest.mark.parametrize("purpose", ["verifier", "progress"])
def test_openai_verifier_routes_reject_invalid_environment_base_url(
    monkeypatch, purpose,
):
    monkeypatch.setenv("OPENAI_BASE_URL", "not-a-url")

    with pytest.raises(ValueError, match=r"absolute HTTP\(S\) URL"):
        verifier_model_id(
            ModelConfig("openai/gpt-4o", api_key="test-key"),
            purpose=purpose,
        )


def test_vertex_custom_verifier_endpoint_uses_adc_without_generic_key_leak(
    monkeypatch,
):
    from google.auth.credentials import AnonymousCredentials

    credentials = AnonymousCredentials()
    monkeypatch.delenv("VERTEX_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "must-not-leak")
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-leak-either")
    monkeypatch.setattr(
        "google.auth.default",
        lambda **kwargs: (credentials, "adc-project"),
    )
    cfg = ModelConfig(
        "vertex_ai/gemini-2.5-flash",
        provider="vertex_ai",
        base_url="http://localhost:8004/v1",
    )

    client = build_verifier_client(cfg)
    assert client.vertexai is True
    assert client._api_client.api_key is None
    assert client._api_client._credentials is credentials
    assert client._api_client.project == "adc-project"
    assert client._api_client.location == "global"
    assert "x-goog-api-key" not in client._api_client._http_options.headers
    assert client._api_client._http_options.base_url == "http://localhost:8004/v1"


def test_vertex_custom_verifier_endpoint_accepts_vertex_api_key():
    client = build_verifier_client(
        ModelConfig(
            "gemini/gemini-2.5-flash",
            provider="vertex_ai",
            api_key="test-key",
            base_url="http://localhost:8004/v1",
        )
    )

    assert client.vertexai is True
    assert client._api_client.api_key == "test-key"


def test_vertex_custom_endpoint_does_not_duplicate_default_api_version(
    monkeypatch,
):
    import httpx
    import google.genai.types as genai_types

    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "ok"}],
                    }
                }]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle))
    original_http_options = genai_types.HttpOptions

    def mock_http_options(**kwargs):
        return original_http_options(httpx_client=http_client, **kwargs)

    monkeypatch.setattr(genai_types, "HttpOptions", mock_http_options)
    try:
        client = build_verifier_client(ModelConfig(
            "vertex_ai/gemini-2.5-flash",
            provider="vertex_ai",
            api_key="test-key",
            base_url="https://vertex.example/gateway/v1?tenant=acme#ignored",
        ))
        client.models.generate_content(
            model="gemini-2.5-flash",
            contents="ping",
        )
    finally:
        http_client.close()

    assert len(requests) == 1
    request_url = str(requests[0].url)
    assert "/v1/v1beta1/" not in request_url
    assert request_url.split("#", 1)[0].endswith("?tenant=acme")
    assert "/gateway/v1/publishers/" in request_url


def test_vertex_custom_endpoint_stream_preserves_base_query(monkeypatch):
    import json
    import httpx
    import google.genai.types as genai_types

    requests = []
    event = json.dumps({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": "ok"}],
            }
        }]
    })

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {event}\n\n".encode(),
            request=request,
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle))
    original_http_options = genai_types.HttpOptions

    def mock_http_options(**kwargs):
        return original_http_options(httpx_client=http_client, **kwargs)

    monkeypatch.setattr(genai_types, "HttpOptions", mock_http_options)
    try:
        client = build_verifier_client(ModelConfig(
            "vertex_ai/gemini-2.5-flash",
            provider="vertex_ai",
            api_key="test-key",
            base_url="https://vertex.example/v1?tenant=acme#ignored",
        ))
        chunks = list(client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents="ping",
        ))
    finally:
        http_client.close()

    assert chunks[0].text == "ok"
    assert len(requests) == 1
    assert str(requests[0].url).endswith(
        ":streamGenerateContent?alt=sse&tenant=acme"
    )


def test_verifier_model_id_rejects_non_string_api_key():
    with pytest.raises(ValueError, match="api_key must be a string or null"):
        verifier_model_id(ModelConfig("vertex_ai/gemini-2.5-flash", api_key=123))


def test_plain_gemini_api_key_is_rejected_for_verification():
    cfg = ModelConfig("gemini/gemini-2.5-flash", api_key="gemini-key")

    with pytest.raises(ValueError, match="provider: vertex_ai"):
        build_verifier_client(cfg)


def test_claude_is_rejected_as_a_verifier():
    """The Messages API returns no logprobs, so a Claude verifier must fail
    loudly rather than silently reaching the Gemini branch."""
    cfg = ModelConfig("anthropic/claude-opus-4-5", api_key="test-key")
    for call in (lambda: verifier_model_id(cfg),
                 lambda: build_verifier_client(cfg)):
        try:
            call()
        except ValueError as e:
            assert "logprob" in str(e)
            assert "backend model" in str(e)
        else:
            raise AssertionError("a Claude verifier should raise")

    # It must fail when the Verifier is constructed — at proxy startup — not
    # on the first scoring call.
    try:
        Verifier(_verifier_config("anthropic/claude-opus-4-5"))
    except ValueError:
        pass
    else:
        raise AssertionError("constructing a Claude verifier should raise")


def test_gemini_prefix_requires_vertex_provider_even_without_a_key():
    with pytest.raises(ValueError, match="provider: vertex_ai"):
        build_verifier_client(ModelConfig("gemini/gemini-2.5-flash"))


def test_openai_progress_client_never_falls_back_to_deepseek_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-be-sent-to-openai")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_verifier_client(
            ModelConfig("openai/gpt-4o"), purpose="progress"
        )


def test_custom_verifier_client_requires_explicit_key():
    with pytest.raises(ValueError, match="explicit.*api_key"):
        build_verifier_client(
            ModelConfig(
                "local-model",
                base_url="http://localhost:8005/v1",
            )
        )


def test_unknown_verifier_provider_never_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://untrusted.example/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-be-forwarded")

    with pytest.raises(ValueError, match="cannot determine.*provider"):
        build_verifier_client(ModelConfig("unrecognized/verifier-model"))


@pytest.mark.parametrize(
    ("base_url", "expected_base_url"),
    [
        (None, "https://aiplatform.googleapis.com/"),
        ("https://vertex.example/v1", "https://vertex.example/v1"),
    ],
)
def test_vertex_provider_uses_only_vertex_environment_key(
    monkeypatch, base_url, expected_base_url,
):
    monkeypatch.setenv("VERTEX_API_KEY", "vertex-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("GOOGLE_VERTEX_BASE_URL", "https://untrusted.example/v1")
    monkeypatch.setattr(
        "google.auth.default",
        lambda **kwargs: pytest.fail("ADC must not be loaded when a Vertex key exists"),
    )

    client = build_verifier_client(
        ModelConfig(
            "gemini/gemini-2.5-flash",
            provider="vertex_ai",
            base_url=base_url,
        )
    )

    assert client.vertexai is True
    assert client._api_client.api_key == "vertex-key"
    assert (
        client._api_client.get_read_only_http_options()["base_url"]
        == expected_base_url
    )


@pytest.mark.parametrize(
    ("location", "expected_base_url"),
    [
        (None, "https://aiplatform.googleapis.com/"),
        (
            "us-central1",
            "https://us-central1-aiplatform.googleapis.com/",
        ),
        ("eu", "https://aiplatform.eu.rep.googleapis.com/"),
    ],
)
def test_vertex_provider_can_use_adc_without_an_api_key(
    monkeypatch, location, expected_base_url,
):
    from google.auth.credentials import AnonymousCredentials
    import google.genai as genai

    calls = {}
    credentials = AnonymousCredentials()

    class FakeClient:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.delenv("VERTEX_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "must-not-be-forwarded")
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-be-forwarded")
    monkeypatch.setenv("GOOGLE_VERTEX_BASE_URL", "https://untrusted.example/v1")
    if location is None:
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    else:
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", location)
    monkeypatch.setattr(
        "google.auth.default",
        lambda **kwargs: (credentials, "adc-project"),
    )
    monkeypatch.setattr(genai, "Client", FakeClient)

    client = build_verifier_client(
        ModelConfig("gemini/gemini-2.5-flash", provider="vertex_ai")
    )

    assert isinstance(client, FakeClient)
    http_options = calls.pop("http_options")
    assert http_options.base_url == expected_base_url
    assert calls == {
        "vertexai": True,
        "credentials": credentials,
        "project": "adc-project",
        "location": location or "global",
    }


def test_vertex_default_endpoint_rejects_unsafe_location(monkeypatch):
    from google.auth.credentials import AnonymousCredentials

    monkeypatch.delenv("VERTEX_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "untrusted.example/path")
    monkeypatch.setattr(
        "google.auth.default",
        lambda **kwargs: (AnonymousCredentials(), "adc-project"),
    )

    with pytest.raises(ValueError, match="GOOGLE_CLOUD_LOCATION"):
        build_verifier_client(ModelConfig(
            "gemini/gemini-2.5-flash",
            provider="vertex_ai",
        ))


def test_vertex_custom_endpoint_adc_request_uses_bearer_auth(monkeypatch):
    import httpx
    from google.auth.credentials import AnonymousCredentials
    import google.genai.types as genai_types

    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "ok"}],
                    }
                }]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle))
    original_http_options = genai_types.HttpOptions

    def mock_http_options(**kwargs):
        return original_http_options(httpx_client=http_client, **kwargs)

    credentials = AnonymousCredentials()
    credentials.token = "adc-token"
    monkeypatch.delenv("VERTEX_API_KEY", raising=False)
    monkeypatch.setattr(
        "google.auth.default",
        lambda **kwargs: (credentials, "adc-project"),
    )
    monkeypatch.setattr(genai_types, "HttpOptions", mock_http_options)

    try:
        client = build_verifier_client(ModelConfig(
            "vertex_ai/gemini-2.5-flash",
            provider="vertex_ai",
            base_url="https://vertex.example/v1?tenant=acme",
        ))
        client.models.generate_content(
            model="gemini-2.5-flash",
            contents="ping",
        )
    finally:
        http_client.close()

    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer adc-token"
    assert "x-goog-api-key" not in requests[0].headers
    assert str(requests[0].url).endswith("?tenant=acme")


def test_vertex_adc_requires_a_project(monkeypatch):
    from google.auth.credentials import AnonymousCredentials

    monkeypatch.delenv("VERTEX_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setattr(
        "google.auth.default",
        lambda **kwargs: (AnonymousCredentials(), None),
    )

    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        build_verifier_client(
            ModelConfig("gemini/gemini-2.5-flash", provider="vertex_ai")
        )


def test_verifier_model_id_is_correct_before_the_client_is_built():
    """Regression: `llm_verifier.select(model=..., client=...)` evaluates its
    keyword arguments left to right, so the model id must not depend on the
    client property having run first."""
    verifier = Verifier(_verifier_config("deepseek/deepseek-v4-flash"))
    assert verifier.model_id == "deepseek-v4-flash"  # no client access yet


def test_config_resolves_base_url_from_the_environment():
    os.environ["TEST_VERIFIER_BASE_URL"] = "http://localhost:9001/v1"
    yaml_text = """
backend:
  models:
    - name: gemini/gemini-2.5-flash
      api_key: test
verifier:
  model:
    name: openai/local-model
    base_url: $TEST_VERIFIER_BASE_URL
    api_key: test-key
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "turbo-proxy.yaml"
        path.write_text(yaml_text)
        cfg = Config(str(path))
        assert cfg.verifier_config.model.base_url == "http://localhost:9001/v1"


def _verifier_config(model_name):
    from turbo_proxy.utils import (CriterionConfig, PivotTournamentConfig,
                                   VerifierConfig)
    return VerifierConfig(
        model=ModelConfig(model_name, api_key="test-key"),
        method=PivotTournamentConfig(
            criteria=[CriterionConfig(name="Task Success", description="d")]),
    )
