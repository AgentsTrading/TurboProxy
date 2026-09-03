"""Regression coverage for backend model ``base_url`` propagation."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import anyio
import httpx
import pytest
import yaml

import turbo_proxy.proxy.backend as backend_module
import turbo_proxy.context.refiner as refiner_module
import turbo_proxy.utils.llm as llm_module
from turbo_proxy.context.refiner import ContextRefiner
from turbo_proxy.proxy.backend import Backend
from turbo_proxy.utils import Config


def _config(tmp_path, models):
    config_path = tmp_path / "turbo-proxy.yaml"
    config_path.write_text(yaml.safe_dump({"backend": {"models": models}}))
    return Config(str(config_path))


def _config_from_raw(tmp_path, raw_config):
    config_path = tmp_path / "turbo-proxy.yaml"
    config_path.write_text(yaml.safe_dump(raw_config))
    return Config(str(config_path))


@pytest.mark.parametrize(
    ("raw_config", "expected_message"),
    [
        ({"backend": None}, "backend must be a mapping"),
        ({"backend": []}, "backend must be a mapping"),
        (
            {"backend": {"models": None}},
            "backend.models must be a list",
        ),
        (
            {"backend": {"models": {}}},
            "backend.models must be a list",
        ),
        ({}, "No models configured under backend.models"),
        (
            {"backend": {"models": []}},
            "No models configured under backend.models",
        ),
        (
            {"backend": {"models": ["openai/gpt-4o"]}},
            "backend.models[0] must be a mapping",
        ),
        (
            {"backend": {"models": [{}]}},
            "backend.models[0].name must be a non-empty string",
        ),
        (
            {"backend": {"models": [{"name": "   "}]}},
            "backend.models[0].name must be a non-empty string",
        ),
        (
            {"backend": {"models": [{"name": 123}]}},
            "backend.models[0].name must be a non-empty string",
        ),
    ],
)
def test_config_rejects_invalid_backend_structure(
    tmp_path, raw_config, expected_message,
):
    with pytest.raises(ValueError) as exc_info:
        _config_from_raw(tmp_path, raw_config)

    assert str(exc_info.value) == expected_message


@pytest.mark.parametrize(
    ("section_config", "expected_message"),
    [
        ({"context": []}, "context must be a mapping"),
        ({"verifier": "invalid"}, "verifier must be a mapping"),
        (
            {"progress_monitor": False},
            "progress_monitor must be a mapping",
        ),
        (
            {"context": {"refinement_model": []}},
            "context.refinement_model must be a mapping",
        ),
        (
            {"verifier": {"model": "invalid"}},
            "verifier.model must be a mapping",
        ),
        (
            {"progress_monitor": {"model": False}},
            "progress_monitor.model must be a mapping",
        ),
    ],
)
def test_config_rejects_invalid_optional_section_structure(
    tmp_path, section_config, expected_message,
):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        **section_config,
    }

    with pytest.raises(ValueError) as exc_info:
        _config_from_raw(tmp_path, raw_config)

    assert str(exc_info.value) == expected_message


@pytest.mark.parametrize(
    ("method", "expected_message"),
    [
        (None, "verifier.method must be a mapping"),
        ([], "verifier.method must be a mapping"),
        (
            {"criteria": None},
            "verifier.method.criteria must be a list",
        ),
        (
            {"criteria": {}},
            "verifier.method.criteria must be a list",
        ),
        (
            {"criteria": ["invalid"]},
            "verifier.method.criteria[0] must be a mapping",
        ),
    ],
)
def test_config_rejects_invalid_verifier_method_structure(
    tmp_path, method, expected_message,
):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "verifier": {"method": method},
    }

    with pytest.raises(ValueError) as exc_info:
        _config_from_raw(tmp_path, raw_config)

    assert str(exc_info.value) == expected_message


def test_config_preserves_optional_null_sections_and_models(tmp_path):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "context": None,
        "verifier": None,
        "progress_monitor": None,
    }
    config = _config_from_raw(tmp_path, raw_config)

    assert config.context_config is None
    assert config.verifier_config is None
    assert config.progress_monitor_config is None

    raw_config.update({
        "context": {"refinement_model": None},
        "verifier": {"model": {}},
        "progress_monitor": {"model": None},
    })
    config = _config_from_raw(tmp_path, raw_config)

    assert config.context_config is None
    assert config.verifier_config is None
    assert config.progress_monitor_config is None


def test_sanitized_config_handles_null_optional_sections(tmp_path):
    config = _config_from_raw(
        tmp_path,
        {
            "backend": {"models": [{"name": "openai/gpt-4o"}]},
            "context": None,
            "verifier": None,
            "progress_monitor": None,
        },
    )

    sanitized = Backend(config)._sanitized_config()

    assert sanitized["context"] is None
    assert sanitized["verifier"] is None
    assert sanitized["progress_monitor"] is None


def test_sanitized_config_redacts_base_url_credentials_and_parameters(tmp_path):
    secret_url = "https://user:pass@gateway.example/v1?token=secret#fragment"
    model = {
        "name": "openai/local-model",
        "base_url": secret_url,
        "api_key": "api-secret",
    }
    config = _config_from_raw(
        tmp_path,
        {
            "backend": {"models": [model]},
            "context": {
                "refinement_model": model,
                "refinement_prompt": "Refine {context}",
            },
            "verifier": {"model": model},
            "progress_monitor": {"model": model},
        },
    )

    sanitized = Backend(config)._sanitized_config()
    expected_url = (
        "https://<redacted>@gateway.example/v1?<redacted>#<redacted>"
    )

    assert sanitized["backend"]["models"][0]["base_url"] == expected_url
    assert sanitized["context"]["refinement_model"]["base_url"] == expected_url
    assert sanitized["verifier"]["model"]["base_url"] == expected_url
    assert sanitized["progress_monitor"]["model"]["base_url"] == expected_url
    assert "pass" not in json.dumps(sanitized)
    assert "token=secret" not in json.dumps(sanitized)


@pytest.mark.parametrize(
    ("base_url", "expected_root", "expected_url"),
    [
        (
            "https://gateway.example/v1?tenant=acme",
            "https://gateway.example/v1",
            "https://gateway.example/v1/chat/completions?tenant=acme",
        ),
        (
            "https://gateway.example/v1/?tenant=acme#ignored",
            "https://gateway.example/v1",
            "https://gateway.example/v1/chat/completions?tenant=acme",
        ),
        (
            "https://gateway.example/v1/chat/completions?tenant=acme#ignored",
            "https://gateway.example/v1",
            "https://gateway.example/v1/chat/completions?tenant=acme",
        ),
    ],
)
@pytest.mark.parametrize(
    ("provider", "model"),
    [("openai", "local-model"), ("deepseek", "local-model")],
)
def test_litellm_base_url_with_query_is_endpoint_safe(
    provider, model, base_url, expected_root, expected_url,
):
    params = llm_module._build_kwargs(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        base_url=base_url,
        provider=provider,
    )

    if provider == "openai":
        assert params["base_url"] == expected_root
        assert params["base_url"].raw_query == b"tenant=acme"
        return

    from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig

    assert params["base_url"] == expected_url
    assert DeepSeekChatConfig().get_complete_url(
        api_base=params["base_url"],
        api_key="test-key",
        model=model,
        optional_params={},
        litellm_params={},
    ) == expected_url


@pytest.mark.parametrize(
    ("provider", "base_url", "expected_url"),
    [
        (
            "openai",
            "https://gateway.example/v1",
            "https://gateway.example/v1/chat/completions",
        ),
        (
            "openai",
            "https://gateway.example/v1?tenant=acme",
            "https://gateway.example/v1/chat/completions?tenant=acme",
        ),
        (
            "openai",
            "https://gateway.example/v1/chat/completions",
            "https://gateway.example/v1/chat/completions",
        ),
        (
            "openai",
            (
                "https://gateway.example/v1/chat/completions"
                "?tenant=acme#ignored"
            ),
            "https://gateway.example/v1/chat/completions?tenant=acme",
        ),
        (
            "openai",
            "https://gateway.example/v1?tenant=台灣",
            (
                "https://gateway.example/v1/chat/completions"
                "?tenant=%E5%8F%B0%E7%81%A3"
            ),
        ),
        (
            "deepseek",
            "https://gateway.example/v1",
            "https://gateway.example/v1/chat/completions",
        ),
        (
            "deepseek",
            "https://gateway.example/v1/",
            "https://gateway.example/v1/chat/completions",
        ),
        (
            "deepseek",
            "https://gateway.example/v1?tenant=acme",
            "https://gateway.example/v1/chat/completions?tenant=acme",
        ),
        (
            "deepseek",
            "https://gateway.example/v1/chat/completions",
            "https://gateway.example/v1/chat/completions",
        ),
        (
            "deepseek",
            "https://gateway.example/v1/chat/completions/",
            "https://gateway.example/v1/chat/completions",
        ),
        (
            "deepseek",
            (
                "https://gateway.example/v1/chat/completions"
                "?tenant=acme#ignored"
            ),
            "https://gateway.example/v1/chat/completions?tenant=acme",
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:coroutine 'Logging.async_success_handler' was never awaited:RuntimeWarning"
)
def test_litellm_runtime_sends_custom_endpoint_once(
    monkeypatch, provider, base_url, expected_url,
):
    requests = []

    async def fake_send(_client, request, **kwargs):
        for hook in _client._event_hooks.get("request", ()):
            await hook(request)
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "local-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    result = asyncio.run(llm_module.llm_completion(
        model="local-model",
        provider=provider,
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        base_url=base_url,
    ))

    assert result["choices"][0]["message"]["content"] == "ok"
    assert [str(request.url) for request in requests] == [expected_url]


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        (
            "https://gateway.example/v1",
            "https://gateway.example/v1/chat/completions",
        ),
        (
            "https://gateway.example/v1?tenant=acme",
            "https://gateway.example/v1/chat/completions?tenant=acme",
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:coroutine 'Logging.async_success_handler' was never awaited:RuntimeWarning"
)
def test_custom_openai_runtime_isolates_environment_headers(
    monkeypatch, base_url, expected_url,
):
    requests = []

    async def fake_send(_client, request, **kwargs):
        for hook in _client._event_hooks.get("request", ()):
            await hook(request)
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "local-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
            },
            request=request,
        )

    monkeypatch.setenv("OPENAI_ORG_ID", "ambient-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-project")
    monkeypatch.setenv(
        "OPENAI_CUSTOM_HEADERS",
        "X-Ambient-Secret: ambient-header\n"
        "Authorization: Bearer ambient-key",
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    result = asyncio.run(llm_module.llm_completion(
        model="local-model",
        provider="openai",
        messages=[{"role": "user", "content": "hello"}],
        api_key="model-key",
        base_url=base_url,
    ))

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == expected_url
    assert request.headers["authorization"] == "Bearer model-key"
    assert "openai-organization" not in request.headers
    assert "openai-project" not in request.headers
    assert "x-ambient-secret" not in request.headers


@pytest.mark.parametrize(
    "override",
    ["OPENAI_BASE_URL", "OPENAI_API_BASE", "litellm.api_base"],
)
@pytest.mark.filterwarnings(
    "ignore:coroutine 'Logging.async_success_handler' was never awaited:RuntimeWarning"
)
def test_openai_backend_ignores_process_wide_endpoint_overrides(
    monkeypatch, override,
):
    requests = []

    async def fake_send(_client, request, **kwargs):
        for hook in _client._event_hooks.get("request", ()):
            await hook(request)
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
            },
            request=request,
        )

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(llm_module.litellm, "api_base", None)
    if override == "litellm.api_base":
        monkeypatch.setattr(
            llm_module.litellm,
            "api_base",
            "https://untrusted.example/v1",
        )
    else:
        monkeypatch.setenv(override, "https://untrusted.example/v1")
    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    result = asyncio.run(llm_module.llm_completion(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
    ))

    assert result["choices"][0]["message"]["content"] == "ok"
    assert [str(request.url) for request in requests] == [
        "https://api.openai.com/v1/chat/completions"
    ]


@pytest.mark.filterwarnings(
    "ignore:coroutine 'Logging.async_success_handler' was never awaited:RuntimeWarning"
)
def test_openai_backend_keeps_environment_api_key_fallback(monkeypatch):
    requests = []

    async def fake_send(_client, request, **kwargs):
        for hook in _client._event_hooks.get("request", ()):
            await hook(request)
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
            },
            request=request,
        )

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setattr(llm_module.litellm, "api_base", None)
    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    result = asyncio.run(llm_module.llm_completion(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    ))

    assert result["choices"][0]["message"]["content"] == "ok"
    assert [str(request.url) for request in requests] == [
        "https://api.openai.com/v1/chat/completions"
    ]
    assert requests[0].headers["authorization"] == "Bearer environment-key"


@pytest.mark.parametrize(
    ("provider", "model", "expected_url", "response_json"),
    [
        (
            "deepseek",
            "deepseek/deepseek-chat",
            "https://api.deepseek.com/beta/chat/completions",
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
            },
        ),
        (
            "gemini",
            "gemini/gemini-2.5-flash",
            (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                "gemini-2.5-flash:generateContent"
            ),
            {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "ok"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
                "modelVersion": "gemini-2.5-flash",
                "responseId": "resp-test",
            },
        ),
        (
            "anthropic",
            "anthropic/claude-3-5-sonnet-latest",
            "https://api.anthropic.com/v1/messages",
            {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-latest",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:coroutine 'Logging.async_success_handler' was never awaited:RuntimeWarning"
)
def test_non_openai_backends_ignore_ambient_endpoint_overrides(
    monkeypatch, provider, model, expected_url, response_json,
):
    requests = []

    async def fake_send(_client, request, **kwargs):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json=response_json,
            request=request,
        )

    for env_var in (
        "DEEPSEEK_API_BASE",
        "GEMINI_API_BASE",
        "ANTHROPIC_API_BASE",
        "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.setenv(env_var, "https://untrusted.example/v1")
    monkeypatch.setattr(
        llm_module.litellm,
        "api_base",
        "https://also-untrusted.example/v1",
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    result = asyncio.run(llm_module.llm_completion(
        model=model,
        provider=provider,
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
    ))

    assert result["choices"][0]["message"]["content"] == "ok"
    assert [str(request.url) for request in requests] == [expected_url]


@pytest.mark.parametrize("override", ["litellm.api_base", "VERTEXAI_API_BASE"])
@pytest.mark.filterwarnings(
    "ignore:coroutine 'Logging.async_success_handler' was never awaited:RuntimeWarning"
)
def test_vertex_backend_ignores_ambient_endpoint_overrides(
    monkeypatch, override,
):
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

    requests = []

    async def fake_token(self, credentials, project_id, custom_llm_provider):
        return "fake-vertex-token", "test-project"

    async def fake_send(_client, request, **kwargs):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "candidates": [{
                    "content": {"parts": [{"text": "ok"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
                "modelVersion": "gemini-2.5-flash",
                "responseId": "resp-test",
            },
            request=request,
        )

    monkeypatch.delenv("VERTEXAI_API_BASE", raising=False)
    monkeypatch.setattr(llm_module.litellm, "api_base", None)
    if override == "litellm.api_base":
        monkeypatch.setattr(
            llm_module.litellm, "api_base", "https://untrusted.example/root"
        )
    else:
        monkeypatch.setenv(
            "VERTEXAI_API_BASE", "https://untrusted.example/root"
        )
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")
    monkeypatch.setattr(VertexBase, "_ensure_access_token_async", fake_token)
    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    result = asyncio.run(llm_module.llm_completion(
        model="vertex_ai/gemini-2.5-flash",
        messages=[{"role": "user", "content": "hello"}],
    ))

    assert result["choices"][0]["message"]["content"] == "ok"
    assert [str(request.url) for request in requests] == [
        "https://us-central1-aiplatform.googleapis.com/v1/projects/"
        "test-project/locations/us-central1/publishers/google/models/"
        "gemini-2.5-flash:generateContent"
    ]
    assert requests[0].headers["authorization"] == "Bearer fake-vertex-token"


@pytest.mark.parametrize(
    "completion_name", ["llm_completion", "llm_stream_completion"]
)
def test_other_litellm_providers_require_model_level_base_url(
    monkeypatch, completion_name,
):
    send = AsyncMock()
    monkeypatch.setenv("GROQ_API_BASE", "https://untrusted.example/v1")
    monkeypatch.setattr(
        llm_module.litellm,
        "api_base",
        "https://also-untrusted.example/v1",
    )
    monkeypatch.setattr(httpx.AsyncClient, "send", send)

    with pytest.raises(
        ValueError,
        match=r"provider 'groq' requires an explicit model-level base_url",
    ):
        asyncio.run(
            getattr(llm_module, completion_name)(
                model="groq/llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "hello"}],
                api_key="must-not-be-sent",
            )
        )

    send.assert_not_awaited()


@pytest.mark.parametrize(
    ("section", "config_property", "field_name"),
    [
        ("backend", None, "backend.models[0]"),
        ("context", "context_config", "context.refinement_model"),
    ],
)
@pytest.mark.parametrize(
    "model",
    [
        {
            "name": "groq/llama-3.1-8b-instant",
            "api_key": "must-not-be-sent",
        },
        {
            "name": "llama-3.1-8b-instant",
            "provider": "groq",
            "api_key": "must-not-be-sent",
        },
    ],
    ids=["litellm-prefix", "explicit-provider"],
)
def test_config_rejects_other_litellm_provider_without_base_url(
    tmp_path, section, config_property, field_name, model,
):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o-mini"}]},
    }
    if section == "backend":
        raw_config["backend"]["models"] = [dict(model)]
    else:
        raw_config["context"] = {
            "refinement_model": dict(model),
            "refinement_prompt": "Refine {context}",
        }

    config_path = tmp_path / "missing-provider-base-url.yaml"
    config_path.write_text(yaml.safe_dump(raw_config))
    with pytest.raises(
        ValueError,
        match=r"provider 'groq' requires an explicit model-level base_url",
    ) as exc_info:
        config = Config(str(config_path))
        if config_property is not None:
            getattr(config, config_property)

    assert field_name in str(exc_info.value)


@pytest.mark.parametrize(
    ("section", "config_property", "field_name"),
    [
        ("backend", None, "backend.models[0]"),
        ("context", "context_config", "context.refinement_model"),
    ],
)
def test_config_rejects_unknown_provider_without_explicit_routing(
    tmp_path, section, config_property, field_name,
):
    model = {"name": "unrecognized/turboproxy-unknown-model"}
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o-mini"}]},
    }
    if section == "backend":
        raw_config["backend"]["models"] = [model]
    else:
        raw_config["context"] = {
            "refinement_model": model,
            "refinement_prompt": "Refine {context}",
        }

    config_path = tmp_path / "unknown-provider.yaml"
    config_path.write_text(yaml.safe_dump(raw_config))
    with pytest.raises(ValueError) as exc_info:
        config = Config(str(config_path))
        if config_property is not None:
            getattr(config, config_property)

    error = str(exc_info.value)
    assert field_name in error
    assert "has no recognized provider" in error
    assert "configure both provider and base_url explicitly" in error


@pytest.mark.parametrize("entry_point", ["_build_kwargs", "llm_completion"])
def test_unknown_provider_fails_before_completion_or_http(
    monkeypatch, entry_point,
):
    completion = AsyncMock()
    send = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "acompletion", completion)
    monkeypatch.setattr(httpx.AsyncClient, "send", send)

    kwargs = {
        "model": "unrecognized/turboproxy-unknown-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    with pytest.raises(ValueError) as exc_info:
        if entry_point == "_build_kwargs":
            llm_module._build_kwargs(**kwargs)
        else:
            asyncio.run(llm_module.llm_completion(**kwargs))

    error = str(exc_info.value)
    assert "has no recognized provider" in error
    assert "configure both provider and base_url explicitly" in error
    completion.assert_not_awaited()
    send.assert_not_awaited()


@pytest.mark.parametrize(
    "completion_name", ["llm_completion", "llm_stream_completion"]
)
def test_other_litellm_providers_accept_explicit_routing(
    monkeypatch, completion_name,
):
    response = Mock()
    response.model_dump.return_value = {"choices": []}
    completion = AsyncMock(return_value=response)
    monkeypatch.setattr(llm_module.litellm, "acompletion", completion)

    asyncio.run(
        getattr(llm_module, completion_name)(
            model="llama-3.1-8b-instant",
            provider="groq",
            base_url="https://api.groq.com/openai/v1",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
        )
    )

    sent = completion.await_args.kwargs
    assert sent["model"] == "llama-3.1-8b-instant"
    assert sent["custom_llm_provider"] == "groq"
    assert sent["base_url"] == "https://api.groq.com/openai/v1"
    assert sent["api_key"] == "test-key"


@pytest.mark.parametrize(
    ("provider", "model", "base_url", "expected_url", "response_json"),
    [
        (
            "anthropic",
            "anthropic/claude-3-5-sonnet-latest",
            "https://gateway.example?tenant=acme#ignored",
            "https://gateway.example/v1/messages?tenant=acme",
            {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-latest",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ),
        (
            "anthropic",
            "anthropic/claude-3-5-sonnet-latest",
            "https://gateway.example/v1/messages?tenant=acme#ignored",
            "https://gateway.example/v1/messages?tenant=acme",
            {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-latest",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ),
        (
            "gemini",
            "gemini/gemini-2.5-flash",
            "https://gateway.example/v1beta?tenant=acme#ignored",
            (
                "https://gateway.example/v1beta/models/"
                "gemini-2.5-flash:generateContent?tenant=acme"
            ),
            {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "ok"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
                "modelVersion": "gemini-2.5-flash",
                "responseId": "resp-test",
            },
        ),
        (
            "gemini",
            "gemini/gemini-2.5-flash",
            (
                "https://gateway.example/v1beta/models/"
                "gemini-2.5-flash:generateContent?tenant=acme#ignored"
            ),
            (
                "https://gateway.example/v1beta/models/"
                "gemini-2.5-flash:generateContent?tenant=acme"
            ),
            {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "ok"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
                "modelVersion": "gemini-2.5-flash",
                "responseId": "resp-test",
            },
        ),
        (
            "vertex_ai",
            "vertex_ai/gemini-2.5-flash",
            "https://vertex.example?tenant=acme#ignored",
            (
                "https://vertex.example/v1/projects/test-project/locations/"
                "us-central1/publishers/google/models/"
                "gemini-2.5-flash:generateContent?tenant=acme"
            ),
            {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "ok"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
                "modelVersion": "gemini-2.5-flash",
                "responseId": "resp-test",
            },
        ),
        (
            "vertex_ai",
            "vertex_ai/gemini-2.5-flash",
            "https://vertex.example/v1?tenant=acme#ignored",
            (
                "https://vertex.example/v1/projects/test-project/locations/"
                "us-central1/publishers/google/models/"
                "gemini-2.5-flash:generateContent?tenant=acme"
            ),
            {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "ok"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
                "modelVersion": "gemini-2.5-flash",
                "responseId": "resp-test",
            },
        ),
        (
            "vertex_ai",
            "vertex_ai/gemini-2.5-flash",
            (
                "https://vertex.example/v1/projects/test-project/locations/"
                "us-central1/publishers/google/models/"
                "gemini-2.5-flash:generateContent?tenant=acme#ignored"
            ),
            (
                "https://vertex.example/v1/projects/test-project/locations/"
                "us-central1/publishers/google/models/"
                "gemini-2.5-flash:generateContent?tenant=acme"
            ),
            {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "ok"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
                "modelVersion": "gemini-2.5-flash",
                "responseId": "resp-test",
            },
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:coroutine 'Logging.async_success_handler' was never awaited:RuntimeWarning"
)
def test_litellm_runtime_preserves_provider_base_query(
    monkeypatch, provider, model, base_url, expected_url, response_json,
):
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

    requests = []

    async def fake_token(self, credentials, project_id, custom_llm_provider):
        return "fake-token", "test-project"

    async def fake_send(_client, request, **kwargs):
        for hook in _client._event_hooks.get("request", ()):
            await hook(request)
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json=response_json,
            request=request,
        )

    monkeypatch.setattr(VertexBase, "_ensure_access_token_async", fake_token)
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")
    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    result = asyncio.run(llm_module.llm_completion(
        model=model,
        provider=provider,
        messages=[{"role": "user", "content": "hello"}],
        api_key=None if provider == "vertex_ai" else "test-key",
        base_url=base_url,
    ))

    assert result["choices"][0]["message"]["content"] == "ok"
    assert [str(request.url) for request in requests] == [expected_url]


def test_litellm_stream_runtime_preserves_custom_openai_query(monkeypatch):
    requests = []
    event = json.dumps({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "local-model",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": "ok"},
            "finish_reason": None,
        }],
    })

    async def fake_send(_client, request, **kwargs):
        for hook in _client._event_hooks.get("request", ()):
            await hook(request)
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {event}\n\ndata: [DONE]\n\n".encode(),
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    async def consume():
        stream = await llm_module.llm_stream_completion(
            model="local-model",
            provider="openai",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            base_url=(
                "https://gateway.example/v1/chat/completions"
                "?tenant=acme#ignored"
            ),
        )
        return [chunk async for chunk in stream]

    chunks = asyncio.run(consume())

    assert chunks
    assert [str(request.url) for request in requests] == [
        "https://gateway.example/v1/chat/completions?tenant=acme"
    ]


@pytest.mark.filterwarnings(
    "ignore:coroutine 'Logging.async_success_handler' was never awaited:RuntimeWarning"
)
def test_litellm_stream_runtime_expands_vertex_v1_base_query(monkeypatch):
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

    requests = []
    event = json.dumps({
        "candidates": [{
            "content": {
                "parts": [{"text": "ok"}],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": 2,
        },
        "modelVersion": "gemini-2.5-flash",
        "responseId": "resp-test",
    })

    async def fake_token(self, credentials, project_id, custom_llm_provider):
        return "fake-token", "test-project"

    async def fake_send(_client, request, **kwargs):
        for hook in _client._event_hooks.get("request", ()):
            await hook(request)
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {event}\n\n".encode(),
            request=request,
        )

    monkeypatch.setattr(VertexBase, "_ensure_access_token_async", fake_token)
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")
    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    async def consume():
        stream = await llm_module.llm_stream_completion(
            model="vertex_ai/gemini-2.5-flash",
            provider="vertex_ai",
            messages=[{"role": "user", "content": "hello"}],
            base_url="https://vertex.example/v1?tenant=acme#ignored",
        )
        return [chunk async for chunk in stream]

    chunks = asyncio.run(consume())

    assert chunks
    assert chunks[0].choices[0].delta.content == "ok"
    assert [str(request.url) for request in requests] == [
        "https://vertex.example/v1/projects/test-project/locations/"
        "us-central1/publishers/google/models/"
        "gemini-2.5-flash:streamGenerateContent?alt=sse&tenant=acme"
    ]


def test_owned_query_stream_can_close_before_consumption(monkeypatch):
    class CloseTrackedStream:
        def __init__(self):
            self.closed = 0
            self.next_calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.next_calls += 1
            raise StopAsyncIteration

        async def aclose(self):
            self.closed += 1

    source = CloseTrackedStream()
    client = Mock()
    client.close = AsyncMock()
    completion = AsyncMock(return_value=source)
    monkeypatch.setattr(llm_module.litellm, "acompletion", completion)
    monkeypatch.setattr(
        llm_module,
        "_attach_openai_query_client",
        Mock(return_value=client),
    )

    async def close_without_consuming():
        stream = await llm_module.llm_stream_completion(
            model="local-model",
            provider="openai",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            base_url="https://gateway.example/v1?tenant=acme",
        )
        await stream.aclose()
        await stream.aclose()

    asyncio.run(close_without_consuming())

    assert source.next_calls == 0
    assert source.closed == 1
    client.close.assert_awaited_once()


def test_owned_stream_closes_litellm_fallback_before_consumption():
    class CloseTrackedWrapper:
        def __init__(self):
            self.closed = 0

        async def aclose(self):
            self.closed += 1

    class LiteLLMFallbackStream:
        def __init__(self, wrapper):
            self.litellm_custom_stream_wrapper = wrapper
            self.iterator_calls = 0

        def __aiter__(self):
            self.iterator_calls += 1
            return self

        async def __anext__(self):
            raise AssertionError("stream should not be consumed")

    wrapper = CloseTrackedWrapper()
    source = LiteLLMFallbackStream(wrapper)
    client = Mock()
    client.close = AsyncMock()
    stream = llm_module._OwnedClientStream(source, client)

    async def close_twice():
        await stream.aclose()
        await stream.aclose()

    asyncio.run(close_twice())

    assert source.iterator_calls == 0
    assert wrapper.closed == 1
    client.close.assert_awaited_once()


def test_owned_stream_closes_litellm_fallback_after_read_error():
    class CloseTrackedWrapper:
        def __init__(self):
            self.closed = 0

        async def aclose(self):
            self.closed += 1

    class LiteLLMFallbackStream:
        def __init__(self, wrapper):
            self.litellm_custom_stream_wrapper = wrapper

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise httpx.ReadError("upstream read failed")

    wrapper = CloseTrackedWrapper()
    client = Mock()
    client.close = AsyncMock()
    stream = llm_module._OwnedClientStream(
        LiteLLMFallbackStream(wrapper), client
    )

    with pytest.raises(httpx.ReadError, match="upstream read failed"):
        asyncio.run(stream.__anext__())

    assert wrapper.closed == 1
    client.close.assert_awaited_once()


def test_owned_stream_preserves_error_when_nested_close_fails():
    class RetryableCloseWrapper:
        def __init__(self):
            self.close_calls = 0

        async def aclose(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("nested close failed")

    class FailingLiteLLMStream:
        def __init__(self, wrapper):
            self.litellm_custom_stream_wrapper = wrapper

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ValueError("upstream failed")

    wrapper = RetryableCloseWrapper()
    client = Mock()
    client.close = AsyncMock()
    stream = llm_module._OwnedClientStream(
        FailingLiteLLMStream(wrapper), client
    )

    async def read_then_retry_close():
        with pytest.raises(ValueError, match="upstream failed") as exc_info:
            await stream.__anext__()
        await stream.aclose()
        await stream.aclose()
        return exc_info.value

    error = asyncio.run(read_then_retry_close())

    assert isinstance(error.__cause__, RuntimeError)
    assert str(error.__cause__) == "nested close failed"
    assert wrapper.close_calls == 2
    client.close.assert_awaited_once()


def test_owned_stream_cleans_up_when_first_iterator_acquisition_fails():
    class FailingIterable:
        def __init__(self):
            self.iterator_calls = 0
            self.closed = 0

        def __aiter__(self):
            self.iterator_calls += 1
            raise ValueError("iterator creation failed")

        async def aclose(self):
            self.closed += 1

    source = FailingIterable()
    client = Mock()
    client.close = AsyncMock()
    stream = llm_module._OwnedClientStream(source, client)

    assert source.iterator_calls == 0
    with pytest.raises(ValueError, match="iterator creation failed"):
        asyncio.run(stream.__anext__())

    assert source.iterator_calls == 1
    assert source.closed == 1
    client.close.assert_awaited_once()


def test_owned_query_stream_closes_inside_cancelled_anyio_scope():
    class CloseTrackedStream:
        def __init__(self):
            self.closed = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            await anyio.sleep(0)
            self.closed += 1

    class CloseTrackedClient:
        def __init__(self):
            self.closed = 0

        async def close(self):
            await anyio.sleep(0)
            self.closed += 1

    source = CloseTrackedStream()
    client = CloseTrackedClient()
    stream = llm_module._OwnedClientStream(source, client)

    async def close_while_cancelled():
        with anyio.CancelScope() as scope:
            scope.cancel()
            await stream.aclose()
        await stream.aclose()

    anyio.run(close_while_cancelled)

    assert source.closed == 1
    assert client.closed == 1


def test_query_completion_client_closes_inside_cancelled_anyio_scope(
    monkeypatch,
):
    class Response:
        def model_dump(self):
            return {"choices": []}

    class CloseTrackedClient:
        def __init__(self):
            self.closed = False

        async def close(self):
            await anyio.sleep(0)
            self.closed = True

    client = CloseTrackedClient()
    monkeypatch.setattr(
        llm_module,
        "_attach_openai_query_client",
        Mock(return_value=client),
    )
    monkeypatch.setattr(
        llm_module.litellm,
        "acompletion",
        AsyncMock(return_value=Response()),
    )

    async def complete_while_cancelled():
        with anyio.CancelScope() as scope:
            scope.cancel()
            await llm_module.llm_completion(
                model="local-model",
                provider="openai",
                messages=[{"role": "user", "content": "hello"}],
                api_key="test-key",
                base_url="https://gateway.example/v1?tenant=acme",
            )

    anyio.run(complete_while_cancelled)

    assert client.closed


def test_openai_query_client_keeps_sdk_network_defaults():
    from openai import DEFAULT_TIMEOUT, DefaultAsyncHttpxClient

    params = llm_module._build_kwargs(
        model="local-model",
        provider="openai",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        base_url="https://gateway.example/v1?tenant=acme",
    )
    client = llm_module._attach_openai_query_client(params)
    assert client is not None
    try:
        assert isinstance(client._client, DefaultAsyncHttpxClient)
        assert client._client.timeout == DEFAULT_TIMEOUT
    finally:
        asyncio.run(client.close())


def test_query_stream_startup_cancellation_closes_client(monkeypatch):
    class CloseTrackedClient:
        def __init__(self):
            self.closed = False

        async def close(self):
            await anyio.sleep(0)
            self.closed = True

    async def cancelled_completion(**kwargs):
        await anyio.sleep(0)

    client = CloseTrackedClient()
    monkeypatch.setattr(
        llm_module,
        "_attach_openai_query_client",
        Mock(return_value=client),
    )
    monkeypatch.setattr(
        llm_module.litellm,
        "acompletion",
        cancelled_completion,
    )

    async def start_stream_while_cancelled():
        with anyio.CancelScope() as scope:
            scope.cancel()
            await llm_module.llm_stream_completion(
                model="local-model",
                provider="openai",
                messages=[{"role": "user", "content": "hello"}],
                api_key="test-key",
                base_url="https://gateway.example/v1?tenant=acme",
            )

    anyio.run(start_stream_while_cancelled)

    assert client.closed


def test_owned_stream_preserves_iteration_error_when_cleanup_fails():
    class FailingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ValueError("upstream failed")

        async def aclose(self):
            raise RuntimeError("source close failed")

    client = Mock()
    client.close = AsyncMock()
    stream = llm_module._OwnedClientStream(FailingStream(), client)

    async def read_one():
        await stream.__anext__()

    with pytest.raises(ValueError, match="upstream failed") as exc_info:
        asyncio.run(read_one())

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "source close failed"
    client.close.assert_awaited_once()


def test_owned_stream_propagates_cleanup_error_after_natural_exhaustion():
    class EmptyStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            raise RuntimeError("source close failed")

    client = Mock()
    client.close = AsyncMock()
    stream = llm_module._OwnedClientStream(EmptyStream(), client)

    async def consume():
        return [item async for item in stream]

    with pytest.raises(RuntimeError, match="source close failed"):
        asyncio.run(consume())

    client.close.assert_awaited_once()


@pytest.mark.parametrize(
    "wrapper_name", ["llm_completion", "llm_stream_completion"]
)
@pytest.mark.parametrize(
    "primary_type", [ValueError, asyncio.CancelledError], ids=["error", "cancel"]
)
def test_query_request_preserves_primary_error_when_client_close_fails(
    monkeypatch, wrapper_name, primary_type,
):
    class FailingClient:
        async def close(self):
            raise RuntimeError("client close failed")

    async def fail_request(**kwargs):
        raise primary_type("request failed")

    monkeypatch.setattr(
        llm_module,
        "_attach_openai_query_client",
        Mock(return_value=FailingClient()),
    )
    monkeypatch.setattr(llm_module.litellm, "acompletion", fail_request)
    wrapper = getattr(llm_module, wrapper_name)

    async def call():
        await wrapper(
            model="local-model",
            provider="openai",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            base_url="https://gateway.example/v1?tenant=acme",
        )

    with pytest.raises(primary_type, match="request failed") as exc_info:
        asyncio.run(call())

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "client close failed"


def test_owned_stream_chains_both_cleanup_errors():
    class FailingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            raise RuntimeError("source close failed")

    client = Mock()
    client.close = AsyncMock(side_effect=RuntimeError("client close failed"))
    stream = llm_module._OwnedClientStream(FailingStream(), client)

    with pytest.raises(RuntimeError, match="source close failed") as exc_info:
        asyncio.run(stream.aclose())

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "client close failed"


def test_raw_cancellation_keeps_completed_cleanup_failure_as_cause():
    async def fail_cleanup():
        raise RuntimeError("cleanup failed")

    async def cancel_while_cleanup_finishes():
        cleanup = asyncio.create_task(fail_cleanup())
        current = asyncio.current_task()
        assert current is not None
        asyncio.get_running_loop().call_soon(current.cancel)
        await llm_module._await_cleanup_task(cleanup)

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(cancel_while_cleanup_finishes())

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "cleanup failed"


def test_config_resolves_backend_base_url_from_environment(tmp_path, monkeypatch):
    base_url = "http://localhost:8100/v1"
    monkeypatch.setenv("TEST_BACKEND_BASE_URL", base_url)

    config = _config(
        tmp_path,
        [
            {
                "name": "openai/local-model",
                "base_url": "$TEST_BACKEND_BASE_URL",
                "api_key": "test-key",
            }
        ],
    )

    assert config.default_model["base_url"] == base_url


@pytest.mark.parametrize(
    "base_url",
    [
        "not-a-url",
        "ftp://gateway.example/v1",
        "https:///missing-host",
        "https://bad host.example/v1",
    ],
)
def test_config_rejects_non_http_base_url(tmp_path, base_url):
    with pytest.raises(ValueError, match=r"absolute HTTP\(S\) URL"):
        _config(
            tmp_path,
            [{
                "name": "openai/local-model",
                "base_url": base_url,
                "api_key": "test-key",
            }],
        )


@pytest.mark.parametrize(
    ("section", "config_property", "field_name"),
    [
        ("backend", None, "backend.models[0].base_url"),
        ("verifier", "verifier_config", "verifier.model.base_url"),
        (
            "progress_monitor",
            "progress_monitor_config",
            "progress_monitor.model.base_url",
        ),
        ("context", "context_config", "context.refinement_model.base_url"),
    ],
)
@pytest.mark.parametrize(
    ("env_value", "env_state"),
    [(None, "missing"), ("", "empty")],
    ids=["missing-env-var", "empty-env-var"],
)
def test_environment_base_url_fails_closed(
    tmp_path,
    monkeypatch,
    section,
    config_property,
    field_name,
    env_value,
    env_state,
):
    env_var = f"TEST_{section.upper()}_BASE_URL_{env_state.upper()}"
    monkeypatch.delenv(env_var, raising=False)
    if env_value is not None:
        monkeypatch.setenv(env_var, env_value)

    raw_config = {
        "backend": {"models": [{"name": "openai/backend-model"}]},
    }
    model = {
        "name": f"openai/{section}-model",
        "base_url": f"${env_var}",
    }
    if section == "backend":
        raw_config["backend"]["models"] = [model]
    elif section == "context":
        raw_config["context"] = {
            "refinement_model": model,
            "refinement_prompt": "Refine {context}",
        }
    else:
        raw_config[section] = {"model": model}

    config_path = tmp_path / "fail-closed.yaml"
    config_path.write_text(yaml.safe_dump(raw_config))

    with pytest.raises(ValueError) as exc_info:
        config = Config(str(config_path))
        if config_property is not None:
            getattr(config, config_property)

    message = str(exc_info.value)
    assert field_name in message
    assert f"${env_var}" in message
    assert "not set or is empty" in message


@pytest.mark.parametrize("base_url", [None, "", "   "])
@pytest.mark.parametrize(
    ("section", "config_property", "field_name"),
    [
        ("backend", None, "backend.models[0].base_url"),
        ("verifier", "verifier_config", "verifier.model.base_url"),
        (
            "progress_monitor",
            "progress_monitor_config",
            "progress_monitor.model.base_url",
        ),
        ("context", "context_config", "context.refinement_model.base_url"),
    ],
)
def test_explicit_empty_base_url_fails_closed(
    tmp_path,
    section,
    config_property,
    field_name,
    base_url,
):
    model = {
        "name": f"openai/{section}-model",
        "base_url": base_url,
        "api_key": "must-not-be-sent",
    }
    raw_config = {
        "backend": {"models": [{"name": "openai/backend-model"}]},
    }
    if section == "backend":
        raw_config["backend"]["models"] = [model]
    elif section == "context":
        raw_config["context"] = {
            "refinement_model": model,
            "refinement_prompt": "Refine {context}",
        }
    else:
        raw_config[section] = {"model": model}

    config_path = tmp_path / "empty-base-url.yaml"
    config_path.write_text(yaml.safe_dump(raw_config))
    with pytest.raises(ValueError) as exc_info:
        config = Config(str(config_path))
        if config_property is not None:
            getattr(config, config_property)

    assert field_name in str(exc_info.value)


@pytest.mark.parametrize(
    ("section", "config_property", "field_name"),
    [
        ("backend", None, "backend.models[0].api_key"),
        ("verifier", "verifier_config", "verifier.model.api_key"),
        (
            "progress_monitor",
            "progress_monitor_config",
            "progress_monitor.model.api_key",
        ),
        ("context", "context_config", "context.refinement_model.api_key"),
    ],
)
@pytest.mark.parametrize(
    ("key_state", "env_value"),
    [("absent", None), ("missing-env", None), ("empty-env", "")],
)
def test_custom_endpoint_api_key_fails_closed(
    tmp_path,
    monkeypatch,
    section,
    config_property,
    field_name,
    key_state,
    env_value,
):
    env_var = f"TEST_{section.upper()}_CUSTOM_KEY_{key_state.upper()}"
    monkeypatch.delenv(env_var, raising=False)
    if env_value is not None:
        monkeypatch.setenv(env_var, env_value)

    model = {
        "name": f"openai/{section}-model",
        "base_url": "http://localhost:8150/v1",
    }
    if key_state != "absent":
        model["api_key"] = f"${env_var}"

    raw_config = {
        "backend": {"models": [{"name": "openai/backend-model"}]},
    }
    if section == "backend":
        raw_config["backend"]["models"] = [model]
    elif section == "context":
        raw_config["context"] = {
            "refinement_model": model,
            "refinement_prompt": "Refine {context}",
        }
    else:
        raw_config[section] = {"model": model}

    config_path = tmp_path / "missing-custom-key.yaml"
    config_path.write_text(yaml.safe_dump(raw_config))

    with pytest.raises(ValueError) as exc_info:
        config = Config(str(config_path))
        if config_property is not None:
            getattr(config, config_property)

    message = str(exc_info.value)
    assert field_name in message
    assert "not set or is empty" in message or "must not be empty" in message


def test_base_params_include_default_model_base_url(tmp_path):
    base_url = "http://localhost:8200/v1"
    backend = Backend(
        _config(
            tmp_path,
            [
                {
                    "name": "openai/default-model",
                    "base_url": base_url,
                    "api_key": "default-key",
                }
            ],
        )
    )

    assert backend._base_params()["base_url"] == base_url


def test_context_refiner_forwards_base_url_and_provider(tmp_path, monkeypatch):
    config_path = tmp_path / "context-custom-endpoint.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "backend": {"models": [{"name": "openai/backend-model"}]},
            "context": {
                "refinement_model": {
                    "name": "deepseek/context-model",
                    "provider": "openai",
                    "base_url": "https://context.example/v1",
                    "api_key": "context-key",
                },
                "refinement_prompt": "Refine {context}",
            },
        })
    )
    context_config = Config(str(config_path)).context_config
    completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "refined"}}]
    })
    monkeypatch.setattr(refiner_module, "llm_completion", completion)

    asyncio.run(
        ContextRefiner(context_config).refine(
            [{"role": "user", "content": "hello"}]
        )
    )

    assert completion.await_args.kwargs["model"] == "deepseek/context-model"
    assert completion.await_args.kwargs["provider"] == "openai"
    assert completion.await_args.kwargs["base_url"] == "https://context.example/v1"
    assert completion.await_args.kwargs["api_key"] == "context-key"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (None, None),
        ({"type": "text", "text": ""}, None),
        ({"content": []}, None),
        ([{"type": "text", "text": ""}], None),
        ([{"type": "text", "text": "refined"}], "refined"),
        (["first", {"text": "second"}], "first\nsecond"),
        ({"content": "refined"}, "refined"),
    ],
)
def test_context_refiner_normalizes_non_string_content(
    tmp_path, monkeypatch, content, expected,
):
    config_path = tmp_path / "context-content-shapes.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "backend": {"models": [{"name": "openai/backend-model"}]},
            "context": {
                "refinement_model": {
                    "name": "openai/context-model",
                    "api_key": "context-key",
                },
                "refinement_prompt": "Refine {context}",
            },
        })
    )
    context_config = Config(str(config_path)).context_config
    completion = AsyncMock(return_value={
        "choices": [{"message": {"content": content}}]
    })
    monkeypatch.setattr(refiner_module, "llm_completion", completion)
    messages = [{"role": "user", "content": "hello"}]

    refined = asyncio.run(ContextRefiner(context_config).refine(messages))

    if expected is None:
        assert refined == messages
    else:
        assert refined[0] == {
            "role": "system",
            "content": expected,
        }
        assert refined[1:] == messages


def test_context_refiner_normalizes_existing_system_content_parts(
    tmp_path, monkeypatch,
):
    config_path = tmp_path / "context-system-content-parts.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "backend": {"models": [{"name": "openai/backend-model"}]},
            "context": {
                "refinement_model": {
                    "name": "openai/context-model",
                    "api_key": "context-key",
                },
                "refinement_prompt": "Refine {context}",
            },
        })
    )
    context_config = Config(str(config_path)).context_config
    completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "refined"}}]
    })
    monkeypatch.setattr(refiner_module, "llm_completion", completion)
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "first policy"},
                {"type": "text", "text": "second policy"},
            ],
        },
        {"role": "user", "content": "hello"},
    ]

    refined = asyncio.run(ContextRefiner(context_config).refine(messages))

    assert refined[0] == {
        "role": "system",
        "content": "refined\n\nfirst policy\nsecond policy",
    }
    assert refined[1:] == messages[1:]


def test_context_refiner_replaces_audio_payload_with_placeholder():
    audio_data = "SECRET-AUDIO-DATA" * 100
    formatted = ContextRefiner._format_messages([{
        "role": "user",
        "content": [{
            "type": "input_audio",
            "input_audio": {"data": audio_data, "format": "wav"},
        }],
    }])

    assert formatted == "USER: [audio]"
    assert audio_data not in formatted


def test_custom_endpoint_key_does_not_replace_global_provider_key(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "official-openai-key")

    Backend(
        _config(
            tmp_path,
            [
                {
                    "name": "openai/local-model",
                    "base_url": "http://localhost:8250/v1",
                    "api_key": "custom-endpoint-key",
                },
                {"name": "openai/gpt-4o"},
            ],
        )
    )

    assert os.environ["OPENAI_API_KEY"] == "official-openai-key"


def test_model_keys_do_not_mutate_provider_environment(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    Backend(
        _config(
            tmp_path,
            [
                {
                    "name": "deepseek/deepseek-chat",
                    "provider": "openai",
                    "api_key": "openai-override-key",
                },
                {"name": "deepseek/deepseek-reasoner"},
            ],
        )
    )

    assert "OPENAI_API_KEY" not in os.environ
    assert "DEEPSEEK_API_KEY" not in os.environ


@pytest.mark.parametrize(
    ("section", "config_property"),
    [
        ("backend", None),
        ("verifier", "verifier_config"),
        ("progress_monitor", "progress_monitor_config"),
        ("context", "context_config"),
    ],
)
def test_official_endpoint_whitespace_key_is_treated_as_unset(
    tmp_path, section, config_property,
):
    model = {"name": "openai/gpt-4o", "api_key": "   "}
    raw_config = {"backend": {"models": [{"name": "openai/backend-model"}]}}
    if section == "backend":
        raw_config["backend"]["models"] = [model]
    elif section == "context":
        raw_config["context"] = {
            "refinement_model": model,
            "refinement_prompt": "Refine {context}",
        }
    else:
        raw_config[section] = {"model": model}

    config_path = tmp_path / "whitespace-key.yaml"
    config_path.write_text(yaml.safe_dump(raw_config))
    config = Config(str(config_path))

    if config_property is None:
        assert config.default_model["api_key"] == ""
    elif section == "context":
        assert getattr(config, config_property).api_key == ""
    else:
        assert getattr(config, config_property).model.api_key is None


@pytest.mark.parametrize(
    ("section", "config_property", "field_name"),
    [
        ("backend", None, "backend.models[0].api_key"),
        ("context", "context_config", "context.refinement_model.api_key"),
    ],
)
def test_litellm_vertex_api_key_is_rejected(
    tmp_path, section, config_property, field_name,
):
    vertex_model = {
        "name": "gemini/gemini-2.5-flash",
        "provider": "vertex_ai",
        "api_key": "$VERTEX_API_KEY",
    }
    raw_config = {
        "backend": {"models": [{"name": "openai/backend-model"}]},
    }
    if section == "backend":
        raw_config["backend"]["models"] = [vertex_model]
    else:
        raw_config["context"] = {
            "refinement_model": vertex_model,
            "refinement_prompt": "Refine {context}",
        }
    config_path = tmp_path / "vertex-api-key.yaml"
    config_path.write_text(yaml.safe_dump(raw_config))

    with pytest.raises(ValueError, match="Vertex ADC/project") as exc_info:
        config = Config(str(config_path))
        if config_property is not None:
            getattr(config, config_property)

    assert field_name in str(exc_info.value)


def test_reference_config_uses_gemini_key_for_litellm_backend():
    config_path = Path(__file__).parents[1] / "turbo-proxy.yaml"
    raw = yaml.safe_load(config_path.read_text())
    backend_model = raw["backend"]["models"][0]

    assert backend_model["api_key"] == "$GEMINI_API_KEY"
    assert backend_model.get("provider") is None


def test_model_entries_preserve_each_models_base_url(tmp_path):
    first_url = "http://localhost:8301/v1"
    second_url = "http://localhost:8302/v1"
    backend = Backend(
        _config(
            tmp_path,
            [
                {
                    "name": "openai/first-model",
                    "base_url": first_url,
                    "api_key": "first-key",
                    "num_candidates": 2,
                },
                {
                    "name": "openai/second-model",
                    "base_url": second_url,
                    "api_key": "second-key",
                    "num_candidates": 1,
                },
            ],
        )
    )

    assert [
        (entry["name"], entry["base_url"]) for entry in backend._model_entries()
    ] == [
        ("openai/first-model", first_url),
        ("openai/first-model", first_url),
        ("openai/second-model", second_url),
    ]


def test_concurrent_models_keep_route_and_credentials_isolated(
    tmp_path, monkeypatch,
):
    default_url = "http://localhost:8350/v1"
    backend = Backend(
        _config(
            tmp_path,
            [
                {
                    "name": "deepseek/default-model",
                    "provider": "openai",
                    "base_url": default_url,
                    "api_key": "default-key",
                },
                {
                    "name": "openai/public-model",
                    "provider": "openai",
                    "api_key": "public-key",
                },
            ],
        )
    )
    completion = AsyncMock(return_value={"choices": []})
    monkeypatch.setattr(backend_module, "llm_completion", completion)

    asyncio.run(
        backend._gather_completions(
            {"messages": [{"role": "user", "content": "hello"}],
             "base_url": default_url},
        )
    )

    calls = completion.await_args_list
    assert (
        calls[0].kwargs["model"],
        calls[0].kwargs["provider"],
        calls[0].kwargs["base_url"],
        calls[0].kwargs["api_key"],
    ) == ("deepseek/default-model", "openai", default_url, "default-key")
    assert (
        calls[1].kwargs["model"],
        calls[1].kwargs["provider"],
        calls[1].kwargs["base_url"],
        calls[1].kwargs["api_key"],
    ) == ("openai/public-model", "openai", None, "public-key")


def test_llm_completion_forwards_base_url(monkeypatch):
    base_url = "http://localhost:8400/v1"
    response = Mock()
    response.model_dump.return_value = {"choices": []}
    completion = AsyncMock(return_value=response)
    monkeypatch.setattr(llm_module.litellm, "acompletion", completion)

    result = asyncio.run(
        llm_module.llm_completion(
            model="openai/local-model",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            base_url=base_url,
        )
    )

    completion.assert_awaited_once()
    assert completion.await_args.kwargs["base_url"] == base_url
    assert result == {"choices": []}


def test_llm_stream_completion_forwards_base_url(monkeypatch):
    class EmptyStream:
        def __init__(self):
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            self.closed = True

    base_url = "http://localhost:8500/v1"
    stream = EmptyStream()
    completion = AsyncMock(return_value=stream)
    monkeypatch.setattr(llm_module.litellm, "acompletion", completion)

    async def consume():
        result = await llm_module.llm_stream_completion(
            model="openai/local-model",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            base_url=base_url,
            max_completion_tokens=7,
        )
        return result, [chunk async for chunk in result]

    result, chunks = asyncio.run(consume())

    completion.assert_awaited_once()
    assert completion.await_args.kwargs["base_url"] == base_url
    assert completion.await_args.kwargs["stream"] is True
    assert completion.await_args.kwargs["max_completion_tokens"] == 7
    assert isinstance(result, llm_module._OwnedClientStream)
    assert chunks == []
    assert stream.closed


def test_llm_stream_completion_forwards_openai_parameters(monkeypatch):
    stream = object()
    completion = AsyncMock(return_value=stream)
    monkeypatch.setattr(llm_module.litellm, "acompletion", completion)
    response_format = {"type": "json_object"}
    logit_bias = {"42": -2}

    result = asyncio.run(
        llm_module.llm_stream_completion(
            model="openai/local-model",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            response_format=response_format,
            seed=17,
            n=2,
            presence_penalty=0.25,
            frequency_penalty=0.5,
            logit_bias=logit_bias,
        )
    )

    completion.assert_awaited_once()
    sent = completion.await_args.kwargs
    assert sent["stream"] is True
    assert sent["response_format"] == response_format
    assert sent["seed"] == 17
    assert sent["n"] == 2
    assert sent["presence_penalty"] == 0.25
    assert sent["frequency_penalty"] == 0.5
    assert sent["logit_bias"] == logit_bias
    assert result is stream


def test_backend_stream_openai_forwards_openai_parameters(
    tmp_path, monkeypatch,
):
    backend = Backend(
        _config(
            tmp_path,
            [{"name": "openai/gpt-4o", "api_key": "test-key"}],
        )
    )

    async def empty_stream():
        if False:
            yield None

    completion = AsyncMock(return_value=empty_stream())
    monkeypatch.setattr(llm_module.litellm, "acompletion", completion)
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "response_format": {"type": "json_object"},
        "seed": 17,
        "n": 2,
        "presence_penalty": 0.25,
        "frequency_penalty": 0.5,
        "logit_bias": {"42": -2},
    }

    async def consume():
        return [
            event async for event in backend.stream_openai(json.dumps(body))
        ]

    events = asyncio.run(consume())

    completion.assert_awaited_once()
    sent = completion.await_args.kwargs
    assert sent["stream"] is True
    for key in (
        "response_format", "seed", "n", "presence_penalty",
        "frequency_penalty", "logit_bias",
    ):
        assert sent[key] == body[key]
    assert events == ["data: [DONE]\n\n"]


@pytest.mark.parametrize(
    ("method_name", "body"),
    [
        (
            "stream_anthropic",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "stream": True,
            },
        ),
        (
            "stream_openai",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ),
    ],
)
def test_backend_stream_paths_do_not_forward_stream_as_an_override(
    tmp_path, monkeypatch, method_name, body,
):
    backend = Backend(
        _config(
            tmp_path,
            [{"name": "openai/gpt-4o", "api_key": "test-key"}],
        )
    )

    async def empty_stream():
        if False:
            yield None

    completion = AsyncMock(return_value=empty_stream())
    monkeypatch.setattr(backend_module, "llm_stream_completion", completion)

    async def consume():
        return [
            event
            async for event in getattr(backend, method_name)(json.dumps(body))
        ]

    asyncio.run(consume())

    completion.assert_awaited_once()
    assert "stream" not in completion.await_args.kwargs


@pytest.mark.parametrize(
    ("method_name", "body", "events_before_disconnect"),
    [
        (
            "stream_anthropic",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "stream": True,
            },
            2,
        ),
        (
            "stream_openai",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            1,
        ),
    ],
)
def test_backend_disconnect_closes_upstream_stream(
    tmp_path, monkeypatch, method_name, body, events_before_disconnect,
):
    backend = Backend(
        _config(
            tmp_path,
            [{"name": "openai/gpt-4o", "api_key": "test-key"}],
        )
    )

    class CloseTrackedStream:
        def __init__(self):
            self.closed = 0
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return {
                "choices": [{
                    "delta": {"content": "partial"},
                    "finish_reason": None,
                }]
            }

        async def aclose(self):
            self.closed += 1

    source = CloseTrackedStream()
    completion = AsyncMock(return_value=source)
    monkeypatch.setattr(backend_module, "llm_stream_completion", completion)

    async def disconnect():
        response = getattr(backend, method_name)(json.dumps(body))
        events = []
        for _ in range(events_before_disconnect):
            events.append(await response.__anext__())
        await response.aclose()
        return events

    events = asyncio.run(disconnect())

    assert events
    completion.assert_awaited_once()
    assert source.closed == 1


@pytest.mark.parametrize(
    ("method_name", "body"),
    [
        (
            "stream_anthropic",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "stream": True,
            },
        ),
        (
            "stream_openai",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ),
    ],
)
def test_backend_stream_preserves_iteration_error_when_close_fails(
    tmp_path, monkeypatch, method_name, body,
):
    backend = Backend(
        _config(
            tmp_path,
            [{"name": "openai/gpt-4o", "api_key": "test-key"}],
        )
    )

    class BadChunk:
        def model_dump(self):
            raise ValueError("chunk conversion failed")

    class CloseFailingStream:
        def __init__(self):
            self.sent = False
            self.closed = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return BadChunk()

        async def aclose(self):
            self.closed += 1
            raise RuntimeError("stream close failed")

    source = CloseFailingStream()
    monkeypatch.setattr(
        backend_module,
        "llm_stream_completion",
        AsyncMock(return_value=source),
    )

    async def consume():
        return [
            event
            async for event in getattr(backend, method_name)(json.dumps(body))
        ]

    with pytest.raises(ValueError, match="chunk conversion failed") as exc_info:
        asyncio.run(consume())

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "stream close failed"
    assert source.closed == 1


@pytest.mark.parametrize(
    ("method_name", "body"),
    [
        (
            "stream_anthropic",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "stream": True,
            },
        ),
        (
            "stream_openai",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ),
    ],
)
def test_backend_stream_preserves_cancellation_when_close_fails(
    tmp_path, monkeypatch, method_name, body,
):
    backend = Backend(
        _config(
            tmp_path,
            [{"name": "openai/gpt-4o", "api_key": "test-key"}],
        )
    )
    source = None

    class BlockingCloseFailingStream:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.closed = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.started.set()
            await self.release.wait()
            raise StopAsyncIteration

        async def aclose(self):
            self.closed += 1
            await asyncio.sleep(0)
            raise RuntimeError("stream close failed")

    async def fake_completion(**kwargs):
        nonlocal source
        source = BlockingCloseFailingStream()
        return source

    monkeypatch.setattr(
        backend_module, "llm_stream_completion", fake_completion,
    )

    async def cancel_during_iteration():
        response = getattr(backend, method_name)(json.dumps(body))
        if method_name == "stream_anthropic":
            await response.__anext__()
        next_event = asyncio.create_task(response.__anext__())
        await asyncio.sleep(0)
        assert source is not None
        await source.started.wait()
        next_event.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await next_event
        return exc_info.value

    cancellation = asyncio.run(cancel_during_iteration())

    assert isinstance(cancellation.__cause__, RuntimeError)
    assert str(cancellation.__cause__) == "stream close failed"
    assert source is not None
    assert source.closed == 1


@pytest.mark.parametrize(
    ("model", "provider", "base_url", "expected_model", "expected_provider"),
    [
        (
            "local-model",
            None,
            "http://localhost:8510/v1",
            "local-model",
            "openai",
        ),
        (
            "Qwen/Qwen3.5-9B",
            None,
            "http://localhost:8520/v1",
            "Qwen/Qwen3.5-9B",
            "openai",
        ),
        (
            "openai/served-model",
            None,
            "http://localhost:8530/v1",
            "openai/served-model",
            None,
        ),
        (
            "deepseek/deepseek-chat",
            "openai",
            "http://localhost:8540/v1",
            "deepseek-chat",
            "openai",
        ),
        (
            "openai/served-model",
            "deepseek",
            None,
            "served-model",
            "deepseek",
        ),
    ],
)
def test_litellm_route_uses_provider_precedence_without_stripping_org_names(
    monkeypatch,
    model,
    provider,
    base_url,
    expected_model,
    expected_provider,
):
    response = Mock()
    response.model_dump.return_value = {"choices": []}
    completion = AsyncMock(return_value=response)
    monkeypatch.setattr(llm_module.litellm, "acompletion", completion)

    asyncio.run(
        llm_module.llm_completion(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            base_url=base_url,
            provider=provider,
        )
    )

    sent = completion.await_args.kwargs
    assert sent["model"] == expected_model
    if expected_provider is None:
        assert "custom_llm_provider" not in sent
    else:
        assert sent["custom_llm_provider"] == expected_provider


@pytest.mark.parametrize(
    ("section", "config_property", "field_name"),
    [
        ("backend", None, "backend.models[0].api_key"),
        ("verifier", "verifier_config", "verifier.model.api_key"),
        (
            "progress_monitor",
            "progress_monitor_config",
            "progress_monitor.model.api_key",
        ),
        (
            "context",
            "context_config",
            "context.refinement_model.api_key",
        ),
    ],
)
def test_explicit_key_reference_never_falls_back_to_provider_default(
    tmp_path, monkeypatch, section, config_property, field_name,
):
    env_var = f"MISSING_{section.upper()}_KEY"
    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    model = {"name": "openai/test-model", "api_key": f"${env_var}"}
    raw_config = {
        "backend": {"models": [{"name": "openai/backend-model"}]},
    }
    if section == "backend":
        raw_config["backend"]["models"] = [model]
    elif section == "context":
        raw_config["context"] = {
            "refinement_model": model,
            "refinement_prompt": "Refine {context}",
        }
    else:
        raw_config[section] = {"model": model}
    config_path = tmp_path / "missing-key.yaml"
    config_path.write_text(yaml.safe_dump(raw_config))

    with pytest.raises(ValueError) as exc_info:
        config = Config(str(config_path))
        if config_property is not None:
            getattr(config, config_property)

    message = str(exc_info.value)
    assert field_name in message
    assert f"${env_var}" in message


@pytest.mark.parametrize(
    "completion_name", ["llm_completion", "llm_stream_completion"]
)
@pytest.mark.parametrize("api_key", [None, ""])
def test_llm_rejects_custom_base_url_without_explicit_key(
    monkeypatch, completion_name, api_key,
):
    litellm_completion = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "acompletion", litellm_completion)

    with pytest.raises(ValueError, match="api_key must be explicitly set"):
        asyncio.run(
            getattr(llm_module, completion_name)(
                model="openai/local-model",
                messages=[{"role": "user", "content": "hello"}],
                api_key=api_key,
                base_url="http://localhost:8600/v1",
            )
        )

    litellm_completion.assert_not_awaited()


@pytest.mark.parametrize(
    "completion_name", ["llm_completion", "llm_stream_completion"]
)
@pytest.mark.parametrize("base_url", [123, "", "   "])
def test_llm_rejects_explicit_empty_base_url_before_http(
    monkeypatch, completion_name, base_url,
):
    litellm_completion = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "acompletion", litellm_completion)

    with pytest.raises(ValueError, match="base_url must be a non-empty string"):
        asyncio.run(
            getattr(llm_module, completion_name)(
                model="openai/local-model",
                messages=[{"role": "user", "content": "hello"}],
                api_key="must-not-be-sent",
                base_url=base_url,
            )
        )

    litellm_completion.assert_not_awaited()


@pytest.mark.parametrize(
    "base_url",
    ["not-a-url", "ftp://gateway.example/v1", "https:///missing-host"],
)
def test_llm_rejects_non_http_base_url_before_http(monkeypatch, base_url):
    completion = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "acompletion", completion)

    with pytest.raises(ValueError, match=r"absolute HTTP\(S\) URL"):
        asyncio.run(llm_module.llm_completion(
            model="openai/local-model",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            base_url=base_url,
        ))

    completion.assert_not_awaited()


@pytest.mark.parametrize(
    "completion_name", ["llm_completion", "llm_stream_completion"]
)
@pytest.mark.parametrize(
    ("protected_kwarg", "supported_name"),
    [
        ("api_base", "base_url"),
        ("custom_llm_provider", "provider"),
        ("azure", "provider"),
        ("deployment_id", "model/provider"),
        (
            "client",
            "top-level model/provider/base_url/api_key parameters",
        ),
        ("fallbacks", "top-level model/provider/base_url parameters"),
        ("model_list", "top-level model/provider/base_url parameters"),
        ("headers", "api_key"),
        ("extra_headers", "api_key"),
        ("extra_body", "provider-specific parameters"),
    ],
)
def test_llm_rejects_kwargs_that_bypass_routing_guards(
    monkeypatch, completion_name, protected_kwarg, supported_name,
):
    litellm_completion = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "acompletion", litellm_completion)

    with pytest.raises(ValueError, match=supported_name):
        asyncio.run(
            getattr(llm_module, completion_name)(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
                **{protected_kwarg: "untrusted-override"},
            )
        )

    litellm_completion.assert_not_awaited()


@pytest.mark.parametrize(
    "completion_name", ["llm_completion", "llm_stream_completion"]
)
def test_llm_rejects_unknown_kwargs_instead_of_forwarding_them(
    monkeypatch, completion_name,
):
    litellm_completion = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "acompletion", litellm_completion)

    with pytest.raises(ValueError, match="unsupported completion parameter.*future"):
        asyncio.run(
            getattr(llm_module, completion_name)(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
                future_litellm_override="untrusted",
            )
        )

    litellm_completion.assert_not_awaited()


@pytest.mark.parametrize(
    "completion_name", ["llm_completion", "llm_stream_completion"]
)
@pytest.mark.parametrize("api_key", [123, False, ["secret"]])
def test_llm_rejects_non_string_api_keys(
    monkeypatch, completion_name, api_key,
):
    litellm_completion = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "acompletion", litellm_completion)

    with pytest.raises(ValueError, match="api_key must be a string or null"):
        asyncio.run(
            getattr(llm_module, completion_name)(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
                api_key=api_key,
            )
        )

    litellm_completion.assert_not_awaited()


@pytest.mark.parametrize(
    "completion_name", ["llm_completion", "llm_stream_completion"]
)
def test_vertex_custom_base_url_uses_vertex_route_without_api_key(
    monkeypatch, completion_name,
):
    response = Mock()
    response.model_dump.return_value = {"choices": []}
    litellm_completion = AsyncMock(return_value=response)
    monkeypatch.setattr(llm_module.litellm, "acompletion", litellm_completion)

    asyncio.run(
        getattr(llm_module, completion_name)(
            model="vertex_ai/gemini-2.5-flash",
            messages=[{"role": "user", "content": "hello"}],
            base_url="https://vertex.example/v1",
        )
    )

    sent = litellm_completion.await_args.kwargs
    assert sent["model"] == "vertex_ai/gemini-2.5-flash"
    assert sent["base_url"] == "https://vertex.example"
    assert "custom_llm_provider" not in sent
    assert "api_key" not in sent


def test_llm_rejects_api_key_for_vertex_backend(monkeypatch):
    litellm_completion = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "acompletion", litellm_completion)

    with pytest.raises(ValueError, match="Vertex ADC/project"):
        asyncio.run(
            llm_module.llm_completion(
                model="vertex_ai/gemini-2.5-flash",
                messages=[{"role": "user", "content": "hello"}],
                api_key="vertex-express-key",
            )
        )

    litellm_completion.assert_not_awaited()


@pytest.mark.parametrize(
    "completion_name", ["llm_completion", "llm_stream_completion"]
)
def test_llm_omits_whitespace_api_key_for_provider_fallback(
    monkeypatch, completion_name,
):
    response = Mock()
    response.model_dump.return_value = {"choices": []}
    litellm_completion = AsyncMock(return_value=response)
    monkeypatch.setattr(llm_module.litellm, "acompletion", litellm_completion)

    asyncio.run(
        getattr(llm_module, completion_name)(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            api_key="   ",
        )
    )

    assert "api_key" not in litellm_completion.await_args.kwargs


@pytest.mark.parametrize(
    ("completion_name", "stream"),
    [("llm_completion", True), ("llm_stream_completion", False)],
)
def test_llm_rejects_stream_mode_override(
    monkeypatch, completion_name, stream,
):
    litellm_completion = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "acompletion", litellm_completion)

    with pytest.raises(ValueError, match="stream"):
        asyncio.run(
            getattr(llm_module, completion_name)(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
                stream=stream,
            )
        )

    litellm_completion.assert_not_awaited()
