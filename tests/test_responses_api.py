"""Focused coverage for the native OpenAI Responses proxy endpoint."""

import asyncio
import json
from enum import Enum
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import yaml
from pydantic import TypeAdapter
from openai.types.responses import ResponsesServerEvent
from openai.types.responses.response import Response as OpenAIResponse

import turbo_proxy.proxy.backend as backend_module
import turbo_proxy.utils.llm as llm_module
from turbo_proxy.proxy.backend import Backend
from turbo_proxy.proxy.proxy import ProxyServer
from turbo_proxy.utils import Config


def _config(tmp_path, models=None):
    config_path = tmp_path / "turbo-proxy.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "backend": {
                    "models": models
                    or [{"name": "openai/gpt-4o", "api_key": "test-key"}],
                },
                "log_dir": str(tmp_path / "logs"),
            }
        )
    )
    return Config(str(config_path))


def _response_with_tool_call():
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1_725_000_000,
        "status": "completed",
        "model": "gpt-4o",
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I will check.",
                        "annotations": [],
                    }
                ],
            },
            {
                "id": "fc_test",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_test",
                "name": "lookup_weather",
                "arguments": '{"city":"Taipei"}',
            },
        ],
        "usage": {
            "input_tokens": 8,
            "output_tokens": 12,
            "total_tokens": 20,
        },
    }


def _sse_payload(event):
    assert event.endswith("\n\n")
    data_lines = [
        line.removeprefix("data: ")
        for line in event.rstrip().splitlines()
        if line.startswith("data: ")
    ]
    assert len(data_lines) == 1
    assert data_lines[0] != "[DONE]"
    return json.loads(data_lines[0])


class _CleanupFailingResponsesStream:
    def __init__(self, events, *, error=None, hang=False):
        self._events = list(events)
        self._error = error
        self._hang = hang
        self.started = asyncio.Event()
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._events:
            return self._events.pop(0)
        self.started.set()
        if self._hang:
            await asyncio.Future()
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        raise StopAsyncIteration

    async def aclose(self):
        self.close_calls += 1
        raise RuntimeError(f"cleanup-{self.close_calls}")


def _upstream_bad_request():
    error = {
        "error": {
            "message": "temperature must be <= 2",
            "type": "invalid_request_error",
            "param": "temperature",
            "code": "invalid_value",
        }
    }
    response = httpx.Response(
        400,
        request=httpx.Request(
            "POST", "https://api.openai.com/v1/responses",
        ),
    )
    return llm_module.litellm.BadRequestError(
        # LiteLLM's Responses adapter currently retains the provider JSON in
        # its message while replacing the attached response body.
        message=f"OpenAIException - {json.dumps(error, separators=(',', ':'))}",
        model="openai/gpt-4o",
        llm_provider="openai",
        response=response,
    )


def test_format_action_omits_image_generation_binary_result():
    binary_result = "base64-secret-" + ("A" * 10_000)

    action = Backend.format_action({
        "output": [{
            "id": "ig_test",
            "type": "image_generation_call",
            "status": "completed",
            "result": binary_result,
        }],
    })

    assert action == "[image_generation_call status=completed]"
    assert binary_result not in action


def test_responses_base_url_accepts_full_resource_path_and_query():
    params = llm_module._build_responses_kwargs(
        model="local-model",
        provider="openai",
        input="hello",
        api_key="test-key",
        base_url=(
            "https://gateway.example/v1/responses/"
            "?tenant=acme#ignored"
        ),
    )

    normalized = params["api_base"]
    assert normalized == "https://gateway.example/v1"
    assert normalized.raw_query == b"tenant=acme"


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        (
            "https://gateway.example/v1?tenant=acme#ignored",
            "https://gateway.example/v1/responses?tenant=acme",
        ),
        (
            "https://gateway.example/v1/responses/?tenant=acme#ignored",
            "https://gateway.example/v1/responses?tenant=acme",
        ),
    ],
)
def test_llm_response_runtime_calls_native_responses_endpoint_once(
    monkeypatch, base_url, expected_url,
):
    requests = []
    response_json = {
        "id": "resp_runtime",
        "object": "response",
        "created_at": 1_725_000_000,
        "status": "completed",
            "model": "gpt-4o",
        "output": [],
    }

    async def fake_send(client, request, **kwargs):
        for hook in client._event_hooks.get("request", ()):
            await hook(request)
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json=response_json,
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    result = asyncio.run(
        llm_module.llm_response(
            model="gpt-4o",
            provider="openai",
            input="hello",
            api_key="test-key",
            base_url=(
                base_url
            ),
        )
    )

    # LiteLLM namespaces response ids with provider/model metadata.
    assert result["id"].startswith("resp_")
    assert result["object"] == "response"
    assert [str(request.url) for request in requests] == [expected_url]


@pytest.mark.parametrize(
    ("base_url", "extra_query", "expected_url"),
    [
        (
            "https://gateway.example/v1",
            {"tenant": "acme", "flag": None},
            "https://gateway.example/v1/responses?tenant=acme&flag=",
        ),
        (
            "https://gateway.example/v1?region=us",
            {"tenant": "acme"},
            "https://gateway.example/v1/responses?region=us&tenant=acme",
        ),
    ],
)
def test_llm_response_runtime_applies_extra_query_to_endpoint(
    monkeypatch, base_url, extra_query, expected_url,
):
    requests = []

    async def fake_send(client, request, **kwargs):
        for hook in client._event_hooks.get("request", ()):
            await hook(request)
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "resp_extra_query",
                "object": "response",
                "created_at": 1_725_000_000,
                "status": "completed",
                "model": "gpt-4o",
                "output": [],
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    asyncio.run(
        llm_module.llm_response(
            model="gpt-4o",
            provider="openai",
            input="hello",
            api_key="test-key",
            base_url=base_url,
            extra_query=extra_query,
        )
    )

    assert [str(request.url) for request in requests] == [expected_url]


@pytest.mark.parametrize(
    ("base_url", "extra_query", "expected_url"),
    [
        (
            "https://gateway.example/openai?api-version=2024-01-01",
            None,
            "https://gateway.example/openai/responses?api-version=2024-01-01",
        ),
        (
            "https://gateway.example/openai/v1?api-version=2024-01-01",
            None,
            "https://gateway.example/openai/responses?api-version=2024-01-01",
        ),
        (
            "https://gateway.example/openai/deployments/weather",
            None,
            "https://gateway.example/openai/v1/responses?api-version=preview",
        ),
        (
            "https://gateway.example/openai/deployments/weather"
            "?api-version=2024-01-01",
            None,
            "https://gateway.example/openai/responses?api-version=2024-01-01",
        ),
        (
            "https://gateway.example/openai/v1/responses?api-version=2024-01-01",
            {"tenant": "acme"},
            "https://gateway.example/openai/responses?api-version=2024-01-01&tenant=acme",
        ),
        (
            "https://gateway.example/openai/v1",
            None,
            "https://gateway.example/openai/v1/responses?api-version=preview",
        ),
    ],
)
def test_azure_responses_runtime_preserves_api_version_and_route(
    monkeypatch, base_url, extra_query, expected_url,
):
    requests = []

    async def fake_send(client, request, **kwargs):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "resp_azure",
                "object": "response",
                "created_at": 1_725_000_000,
                "status": "completed",
                "model": "gpt-4o",
                "output": [],
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    asyncio.run(
        llm_module.llm_response(
            model="gpt-4o",
            provider="azure",
            input="hello",
            api_key="test-key",
            base_url=base_url,
            extra_query=extra_query,
        )
    )

    assert [str(request.url) for request in requests] == [expected_url]


def test_azure_responses_extra_query_overrides_base_query(monkeypatch):
    requests = []

    async def fake_send(client, request, **kwargs):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "resp_azure_override",
                "object": "response",
                "created_at": 1_725_000_000,
                "status": "completed",
                "model": "gpt-4o",
                "output": [],
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    asyncio.run(
        llm_module.llm_response(
            model="gpt-4o",
            provider="azure",
            input="hello",
            api_key="test-key",
            base_url=(
                "https://gateway.example/openai"
                "?api-version=2024-01-01&tenant=base"
            ),
            extra_query={
                "api-version": "2025-01-01",
                "tenant": "request",
            },
        )
    )

    assert [str(request.url) for request in requests] == [
        "https://gateway.example/openai/responses"
        "?api-version=2025-01-01&tenant=request"
    ]


def test_llm_stream_response_runtime_uses_native_responses_endpoint(
    monkeypatch,
):
    requests = []
    event = json.dumps({
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            "id": "resp_stream_runtime",
            "object": "response",
            "status": "in_progress",
            "output": [],
        },
    })

    async def fake_send(client, request, **kwargs):
        for hook in client._event_hooks.get("request", ()):
            await hook(request)
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {event}\n\n".encode(),
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    async def consume():
        stream = await llm_module.llm_stream_response(
            model="gpt-4o",
            provider="openai",
            input="hello",
            api_key="test-key",
            base_url="https://gateway.example/v1?tenant=acme",
        )
        return [item async for item in stream]

    events = asyncio.run(consume())

    assert len(events) == 1
    payload = events[0].model_dump() if hasattr(events[0], "model_dump") else events[0]
    assert payload["type"] == "response.created"
    assert [str(request.url) for request in requests] == [
        "https://gateway.example/v1/responses?tenant=acme"
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_native_responses_omitted_input_is_absent_from_wire_body(
    monkeypatch, stream,
):
    requests = []
    response_json = {
        "id": "resp_without_input",
        "object": "response",
        "created_at": 1_725_000_000,
        "status": "completed",
        "model": "gpt-4o",
        "output": [],
    }
    completed_event = json.dumps({
        "type": "response.completed",
        "sequence_number": 0,
        "response": response_json,
    })

    async def fake_send(client, request, **kwargs):
        for hook in client._event_hooks.get("request", ()):
            await hook(request)
        requests.append(request)
        if stream:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {completed_event}\n\n".encode(),
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json=response_json,
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    async def request_without_input():
        kwargs = {
            "model": "gpt-4o",
            "provider": "openai",
            "api_key": "test-key",
            "base_url": "https://gateway.example/v1",
            "previous_response_id": "resp_previous",
        }
        if stream:
            source = await llm_module.llm_stream_response(**kwargs)
            return [event async for event in source]
        return await llm_module.llm_response(**kwargs)

    asyncio.run(request_without_input())

    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload["previous_response_id"] == "resp_previous"
    assert "input" not in payload


def test_llm_response_preserves_compatibility_fields_in_wire_body(monkeypatch):
    requests = []

    async def fake_send(client, request, **kwargs):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "resp_compat",
                "object": "response",
                "created_at": 1_725_000_000,
                "status": "completed",
                "output": [],
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    asyncio.run(
        llm_module.llm_response(
            model="gpt-4o",
            provider="openai",
            input="hello",
            api_key="test-key",
            base_url="https://gateway.example/v1",
            conversation="conv_123",
            moderation={"safety_identifier": "test"},
            prompt_cache_options={"mode": "explicit", "ttl": "24h"},
            thinking_budget=123,
        )
    )

    payload = json.loads(requests[0].content)
    assert payload["conversation"] == "conv_123"
    assert payload["moderation"] == {"safety_identifier": "test"}
    assert payload["prompt_cache_options"] == {
        "mode": "explicit",
        "ttl": "24h",
    }
    assert payload["thinking"] == {
        "type": "enabled",
        "budget_tokens": 123,
    }


@pytest.mark.parametrize(
    ("wrapper_name", "stream"),
    [("llm_response", True), ("llm_stream_response", False)],
)
def test_responses_wrappers_reject_stream_mode_override(
    monkeypatch, wrapper_name, stream,
):
    aresponses = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "aresponses", aresponses)

    with pytest.raises(ValueError, match="stream"):
        asyncio.run(
            getattr(llm_module, wrapper_name)(
                model="openai/gpt-4o",
                input="hello",
                stream=stream,
            )
        )

    aresponses.assert_not_awaited()


@pytest.mark.parametrize("wrapper_name", ["llm_response", "llm_stream_response"])
@pytest.mark.parametrize(
    ("background", "message"),
    [
        (
            True,
            "background=true is not supported because Responses retrieval "
            "endpoints are not implemented",
        ),
        (1, "background must be a boolean or null"),
        ("true", "background must be a boolean or null"),
    ],
)
def test_responses_wrappers_reject_unsupported_background(
    monkeypatch, wrapper_name, background, message,
):
    aresponses = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "aresponses", aresponses)

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            getattr(llm_module, wrapper_name)(
                model="openai/gpt-4o",
                input="hello",
                background=background,
            )
        )

    assert str(exc_info.value) == message
    aresponses.assert_not_awaited()


def test_responses_wrapper_allows_foreground_mode(monkeypatch):
    response = Mock()
    response.model_dump.return_value = {
        "id": "resp_foreground",
        "object": "response",
        "status": "completed",
        "output": [],
    }
    aresponses = AsyncMock(return_value=response)
    monkeypatch.setattr(llm_module.litellm, "aresponses", aresponses)

    asyncio.run(
        llm_module.llm_response(
            model="openai/gpt-4o",
            input="hello",
            background=False,
        )
    )

    assert aresponses.await_args.kwargs["background"] is False


def test_responses_wrapper_forwards_supported_extensions(monkeypatch):
    response = Mock()
    response.model_dump.return_value = {
        "id": "resp_extensions",
        "object": "response",
        "status": "completed",
        "output": [],
    }
    aresponses = AsyncMock(return_value=response)
    monkeypatch.setattr(llm_module.litellm, "aresponses", aresponses)

    asyncio.run(
        llm_module.llm_response(
            model="openai/gpt-4o",
            input="hello",
            text_format={"type": "json_object"},
            extra_headers={"X-Request-ID": "trace-123"},
            extra_query={"tenant": "acme"},
            timeout=30,
            allowed_openai_params=["vendor_param"],
            extra_body={"vendor_param": "enabled"},
        )
    )

    sent = aresponses.await_args.kwargs
    assert sent["text_format"] == {"type": "json_object"}
    assert sent["extra_headers"] == {"X-Request-ID": "trace-123"}
    assert sent["extra_query"] == {"tenant": "acme"}
    assert sent["timeout"] == 30
    assert sent["allowed_openai_params"] == ["vendor_param"]
    assert sent["extra_body"] == {"vendor_param": "enabled"}


@pytest.mark.parametrize(
    ("response_format", "expected_text"),
    [
        (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": {"type": "object"},
                },
            },
            {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "strict": True,
                    "schema": {"type": "object"},
                }
            },
        ),
        (
            {"type": "json_object"},
            {"format": {"type": "json_object"}},
        ),
        (
            {"format": {"type": "json_schema", "name": "native"}},
            {"format": {"type": "json_schema", "name": "native"}},
        ),
    ],
)
def test_responses_wrapper_translates_chat_response_format(
    monkeypatch, response_format, expected_text,
):
    response = Mock()
    response.model_dump.return_value = {
        "id": "resp_format",
        "object": "response",
        "status": "completed",
        "output": [],
    }
    aresponses = AsyncMock(return_value=response)
    monkeypatch.setattr(llm_module.litellm, "aresponses", aresponses)

    asyncio.run(
        llm_module.llm_response(
            model="openai/gpt-4o",
            input="hello",
            response_format=response_format,
        )
    )

    assert aresponses.await_args.kwargs["text"] == expected_text
    assert "response_format" not in aresponses.await_args.kwargs


def test_responses_wrapper_prefers_native_text_over_response_format(
    monkeypatch,
):
    response = Mock()
    response.model_dump.return_value = {
        "id": "resp_format_precedence",
        "object": "response",
        "status": "completed",
        "output": [],
    }
    aresponses = AsyncMock(return_value=response)
    monkeypatch.setattr(llm_module.litellm, "aresponses", aresponses)

    native_text = {"format": {"type": "text"}}
    asyncio.run(
        llm_module.llm_response(
            model="openai/gpt-4o",
            input="hello",
            response_format={"type": "json_object"},
            text=native_text,
        )
    )

    sent = aresponses.await_args.kwargs
    assert sent["text"] == native_text
    assert "response_format" not in sent


@pytest.mark.parametrize("header", [
    "Authorization", "api-key", "X-Api-Key", "X-Goog-Api-Key", "Host",
])
def test_responses_wrapper_rejects_protected_extra_headers(monkeypatch, header):
    aresponses = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "aresponses", aresponses)

    with pytest.raises(ValueError, match="protected header"):
        asyncio.run(
            llm_module.llm_response(
                model="openai/gpt-4o",
                input="hello",
                extra_headers={header: "override"},
            )
        )

    aresponses.assert_not_awaited()


@pytest.mark.parametrize("field", [
    "model", "input", "stream", "api_key", "api_base", "custom_llm_provider",
    "background", "context_management", "conversation", "previous_response_id",
    "prompt", "temperature", "tools",
])
def test_responses_wrapper_rejects_protected_extra_body_fields(
    monkeypatch, field,
):
    aresponses = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "aresponses", aresponses)

    with pytest.raises(ValueError, match="protected field"):
        asyncio.run(
            llm_module.llm_response(
                model="openai/gpt-4o",
                input="hello",
                extra_body={field: "override"},
            )
        )

    aresponses.assert_not_awaited()


def test_backend_complete_responses_forwards_native_fields_unchanged(
    tmp_path, monkeypatch,
):
    backend = Backend(_config(tmp_path))
    upstream_response = _response_with_tool_call()
    response_call = AsyncMock(return_value=upstream_response)
    monkeypatch.setattr(backend_module, "llm_response", response_call)
    monkeypatch.setattr(backend_module, "save_request_log", Mock())

    body = {
        "model": "client-selected-model",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Weather?"}],
            }
        ],
        "instructions": "Answer briefly.",
        "tools": [
            {
                "type": "function",
                "name": "lookup_weather",
                "description": "Look up current weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "tool_choice": "auto",
        "reasoning": {"effort": "high", "summary": "auto"},
        "text": {"format": {"type": "text"}},
        "max_output_tokens": 321,
        "parallel_tool_calls": True,
        "previous_response_id": "resp_previous",
        "prompt_cache_options": {"mode": "explicit", "ttl": "24h"},
        "store": False,
        "metadata": {"trace_id": "trace-test"},
        "temperature": 0.2,
        "top_p": 0.9,
        "truncation": "auto",
        "user": "user-test",
        "stream": False,
    }

    result, error = asyncio.run(
        backend.complete_responses(json.dumps(body))
    )

    assert error is None
    assert result == upstream_response
    response_call.assert_awaited_once()
    sent = response_call.await_args.kwargs
    assert sent["model"] == "openai/gpt-4o"
    assert sent["api_key"] == "test-key"
    for key, value in body.items():
        if key not in {"model", "stream"}:
            assert sent[key] == value
    assert "messages" not in sent
    assert "stream" not in sent


def test_backend_complete_responses_preserves_omitted_input(
    tmp_path, monkeypatch,
):
    backend = Backend(_config(tmp_path))
    upstream_response = _response_with_tool_call()
    response_call = AsyncMock(return_value=upstream_response)
    monkeypatch.setattr(backend_module, "llm_response", response_call)
    monkeypatch.setattr(backend_module, "save_request_log", Mock())

    result, error = asyncio.run(backend.complete_responses(json.dumps({
        "previous_response_id": "resp_previous",
    })))

    assert error is None
    assert result == upstream_response
    sent = response_call.await_args.kwargs
    assert sent["previous_response_id"] == "resp_previous"
    assert "input" not in sent


def test_backend_stream_responses_forwards_events_as_sse_and_closes_source(
    tmp_path, monkeypatch,
):
    backend = Backend(_config(tmp_path))
    payloads = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": "resp_stream",
                "object": "response",
                "status": "in_progress",
                "output": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_stream",
            "output_index": 0,
            "content_index": 0,
            "delta": "hello",
        },
        {
            "type": "response.completed",
            "sequence_number": 2,
            "response": {
                "id": "resp_stream",
                "object": "response",
                "status": "completed",
                "output": [],
            },
        },
    ]

    class ResponseEvent:
        def __init__(self, payload):
            self.payload = payload

        def model_dump(self):
            return self.payload

    class CloseTrackedStream:
        def __init__(self):
            self._events = iter(ResponseEvent(payload) for payload in payloads)
            self.closed = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

        async def aclose(self):
            self.closed += 1

    source = CloseTrackedStream()
    response_call = AsyncMock(return_value=source)
    monkeypatch.setattr(backend_module, "llm_stream_response", response_call)
    monkeypatch.setattr(backend_module, "save_request_log", Mock())

    body = {
        "input": "Say hello.",
        "instructions": "Be concise.",
        "stream": True,
    }

    async def consume():
        return [
            event
            async for event in backend.stream_responses(json.dumps(body))
        ]

    events = asyncio.run(consume())

    emitted = [_sse_payload(event) for event in events]
    adapter = TypeAdapter(ResponsesServerEvent)
    assert [adapter.validate_python(event).type for event in emitted] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert [event["sequence_number"] for event in emitted] == [0, 1, 2]
    assert emitted[1]["logprobs"] == []
    assert emitted[0]["response"]["id"] == "resp_stream"
    assert emitted[2]["response"]["id"] == "resp_stream"
    assert emitted[0]["response"]["created_at"] == (
        emitted[2]["response"]["created_at"]
    )
    assert source.closed == 1
    response_call.assert_awaited_once()
    sent = response_call.await_args.kwargs
    assert sent["input"] == body["input"]
    assert sent["instructions"] == body["instructions"]
    assert "stream" not in sent


def test_responses_sse_normalizes_event_enum_to_protocol_value():
    class EventType(Enum):
        CREATED = "response.created"

    event = {
        "type": EventType.CREATED,
        "sequence_number": 0,
        "response": {"id": "resp_enum", "output": []},
    }

    formatted = Backend._format_responses_sse(event)

    assert formatted.startswith("event: response.created\n")
    payload = _sse_payload(formatted)
    assert payload["type"] == "response.created"


def test_responses_sse_normalizes_cancelled_event_to_failed():
    event = {
        "type": "response.cancelled",
        "sequence_number": 0,
        "response": {"id": "resp_cancelled", "status": "cancelled"},
    }

    formatted = Backend._format_responses_sse(event)

    assert formatted.startswith("event: response.failed\n")
    payload = _sse_payload(formatted)
    assert payload["type"] == "response.failed"
    assert payload["response"]["status"] == "cancelled"


def test_backend_stream_responses_closes_nested_http_response(tmp_path, monkeypatch):
    backend = Backend(_config(tmp_path))
    payload = {
        "type": "response.completed",
        "sequence_number": 0,
        "response": {"id": "resp_nested", "output": []},
    }

    class NestedResponse:
        def __init__(self):
            self.closed = 0

        async def aclose(self):
            self.closed += 1

    class IteratorWithoutClose:
        def __init__(self, response):
            self.response = response
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            return payload

    nested_response = NestedResponse()
    monkeypatch.setattr(
        backend_module,
        "llm_stream_response",
        AsyncMock(return_value=IteratorWithoutClose(nested_response)),
    )

    async def consume():
        return [
            event
            async for event in backend.stream_responses(
                json.dumps({"input": "hello", "stream": True})
            )
        ]

    events = asyncio.run(consume())

    assert len(events) == 1
    assert nested_response.closed == 1


def test_backend_stream_responses_disconnect_closes_nested_litellm_wrapper(
    tmp_path, monkeypatch,
):
    backend = Backend(_config(tmp_path))
    payload = {
        "type": "response.created",
        "response": {"id": "resp_disconnect", "output": []},
    }

    class NestedWrapper:
        def __init__(self):
            self.closed = 0

        async def aclose(self):
            self.closed += 1

    class IteratorWithoutClose:
        def __init__(self, wrapper):
            self.litellm_custom_stream_wrapper = wrapper
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return payload

    nested_wrapper = NestedWrapper()
    monkeypatch.setattr(
        backend_module,
        "llm_stream_response",
        AsyncMock(return_value=IteratorWithoutClose(nested_wrapper)),
    )

    async def disconnect():
        response = backend.stream_responses(json.dumps({
            "input": "hello",
            "stream": True,
        }))
        event = await response.__anext__()
        await response.aclose()
        return event

    event = asyncio.run(disconnect())

    assert _sse_payload(event)["type"] == "response.created"
    assert nested_wrapper.closed == 1


def test_backend_stream_responses_cancellation_closes_nested_litellm_wrapper(
    tmp_path, monkeypatch,
):
    backend = Backend(_config(tmp_path))
    source = None

    class NestedWrapper:
        def __init__(self):
            self.closed = 0

        async def aclose(self):
            self.closed += 1
            await asyncio.sleep(0)

    class BlockingIteratorWithoutClose:
        def __init__(self):
            self.litellm_custom_stream_wrapper = NestedWrapper()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.started.set()
            await self.release.wait()
            raise StopAsyncIteration

    async def fake_stream_response(**kwargs):
        nonlocal source
        source = BlockingIteratorWithoutClose()
        return source

    monkeypatch.setattr(
        backend_module, "llm_stream_response", fake_stream_response,
    )

    async def cancel_during_iteration():
        response = backend.stream_responses(json.dumps({
            "input": "hello",
            "stream": True,
        }))
        next_event = asyncio.create_task(response.__anext__())
        await asyncio.sleep(0)
        assert source is not None
        await source.started.wait()
        next_event.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await next_event
        return exc_info.value

    cancellation = asyncio.run(cancel_during_iteration())

    assert isinstance(cancellation, asyncio.CancelledError)
    assert source is not None
    assert source.litellm_custom_stream_wrapper.closed == 1


def test_backend_verifier_stream_replay_keeps_tool_calls(
    tmp_path, monkeypatch,
):
    backend = Backend(_config(tmp_path))
    backend.verifier = object()
    selected_response = _response_with_tool_call()
    response_call = AsyncMock(return_value=selected_response)
    monkeypatch.setattr(backend_module, "llm_response", response_call)
    monkeypatch.setattr(backend_module, "save_request_log", Mock())

    async def consume():
        return [
            event
            async for event in backend.stream_responses(
                json.dumps({"input": "Use the weather tool.", "stream": True})
            )
        ]

    payloads = [_sse_payload(event) for event in asyncio.run(consume())]

    assert payloads[0]["type"] == "response.created"
    assert payloads[-1]["type"] == "response.completed"
    completed_output = payloads[-1]["response"]["output"]
    assert selected_response["output"][1] in completed_output
    response_call.assert_awaited_once()


def test_backend_verifier_fanout_drops_stream_options(
    tmp_path, monkeypatch,
):
    backend = Backend(_config(tmp_path))
    backend.verifier = object()
    response_call = AsyncMock(return_value=_response_with_tool_call())
    monkeypatch.setattr(backend_module, "llm_response", response_call)
    monkeypatch.setattr(backend_module, "save_request_log", Mock())

    result, error = asyncio.run(backend.complete_responses(json.dumps({
        "input": "Use the weather tool.",
        "stream_options": {"include_usage": True},
    })))

    assert error is None
    assert result == _response_with_tool_call()
    response_call.assert_awaited_once()
    sent = response_call.await_args.kwargs
    assert "stream" not in sent
    assert "stream_options" not in sent


@pytest.mark.parametrize("stream", [False, True])
def test_responses_route_rejects_item_reference_with_verifier(
    tmp_path, monkeypatch, stream,
):
    server = ProxyServer(_config(tmp_path))
    server.backend.verifier = object()
    complete_call = AsyncMock()
    stream_call = AsyncMock()
    monkeypatch.setattr(server.backend, "complete_responses", complete_call)
    monkeypatch.setattr(server.backend, "stream_responses", stream_call)
    body = {
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": "Continue",
                "context": {"type": "item_reference", "id": "item_1"},
            }],
        }],
        "stream": stream,
    }

    async def request_invalid():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post("/v1/responses", json=body)

    response = asyncio.run(request_invalid())

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "Responses parameter(s) input.item_reference cannot be used "
        "when verifier is enabled"
    )
    complete_call.assert_not_awaited()
    stream_call.assert_not_awaited()


def test_backend_gather_responses_propagates_candidate_cancellation(
    tmp_path, monkeypatch,
):
    backend = Backend(_config(tmp_path))
    response_call = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(backend_module, "llm_response", response_call)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(backend._gather_responses({"input": "hello"}))

    response_call.assert_awaited_once()


@pytest.mark.parametrize("best_index", [-1, 2, True, "0"])
def test_backend_verifier_invalid_index_falls_back_to_first_response(
    tmp_path, best_index,
):
    backend = Backend(_config(
        tmp_path,
        models=[
            {"name": "openai/first", "api_key": "first-key"},
            {"name": "openai/second", "api_key": "second-key"},
        ],
    ))
    backend.verifier = Mock()
    backend.verifier.select_best = AsyncMock(return_value=Mock(
        best_index=best_index,
        scores=[0.2, 0.8],
        comparisons=[],
    ))
    responses = [
        ({"id": "first", "output": [{"type": "message"}]}, "first"),
        ({"id": "second", "output": [{"type": "message"}]}, "second"),
    ]

    selected = asyncio.run(backend._pick_best(
        responses,
        [{"role": "user", "content": "hello"}],
    ))

    assert selected == responses[0]
    backend.verifier.select_best.assert_awaited_once()


def test_backend_responses_refinement_skips_empty_history(tmp_path):
    backend = Backend(_config(tmp_path))
    backend.refiner = Mock()
    backend.refiner.refine = AsyncMock()
    params = {"input": "hello"}
    history = []

    refined_params, refined_history = asyncio.run(
        backend._refine_responses(params, history)
    )

    assert refined_params is params
    assert refined_history is history
    assert params == {"input": "hello"}
    assert history == []
    backend.refiner.refine.assert_not_awaited()


@pytest.mark.parametrize(
    "referenced_context",
    [
        {"previous_response_id": "resp_previous"},
        {"conversation": "conv_shared"},
        {"prompt": {"id": "pmpt_shared"}},
    ],
)
def test_backend_responses_refinement_skips_referenced_context(
    tmp_path, referenced_context,
):
    backend = Backend(_config(tmp_path))
    backend.refiner = Mock()
    backend.refiner.refine = AsyncMock()
    params = {"input": "hello", **referenced_context}
    history = [{"role": "user", "content": "hello"}]
    expected_params = json.loads(json.dumps(params))
    expected_history = json.loads(json.dumps(history))

    refined_params, refined_history = asyncio.run(
        backend._refine_responses(params, history)
    )

    assert refined_params is params
    assert refined_history is history
    assert params == expected_params
    assert history == expected_history
    backend.refiner.refine.assert_not_awaited()


def test_backend_responses_refinement_invalid_messages_falls_back_to_history(
    tmp_path,
):
    backend = Backend(_config(tmp_path))
    backend.refiner = Mock()
    backend.refiner.refine = AsyncMock(return_value=[None])
    params = {"input": "hello"}
    history = [{"role": "user", "content": "hello"}]

    refined_params, refined_history = asyncio.run(
        backend._refine_responses(params, history)
    )

    assert refined_params == params
    assert refined_history == history


def test_responses_history_redacts_computer_screenshot_payload():
    image_url = "data:image/png;base64," + "A" * 1024

    history = Backend._responses_history({
        "input": [{
            "type": "computer_call_output",
            "call_id": "call_computer",
            "output": {
                "type": "computer_screenshot",
                "image_url": image_url,
            },
        }],
    })

    assert history == [{
        "role": "tool",
        "tool_call_id": "call_computer",
        "content": "[image]",
    }]
    assert image_url not in json.dumps(history)


def test_responses_replay_hides_terminal_fields_until_completion(tmp_path):
    backend = Backend(_config(tmp_path))
    response = {
        "id": "resp_terminal_fields",
        "object": "response",
        "created_at": 1_725_000_000,
        "status": "completed",
        "completed_at": 1_725_000_001,
        "error": {"code": "server_error", "message": "late error"},
        "incomplete_details": {"reason": "max_output_tokens"},
        "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        "output_text": "FULL ANSWER",
        "audio": {"data": "FULL-AUDIO", "transcript": "FULL TRANSCRIPT"},
        "audio_transcript": "FULL TRANSCRIPT",
        "output": [],
    }

    async def consume():
        return [
            _sse_payload(event)
            async for event in backend._replay_responses_sse(response)
        ]

    payloads = asyncio.run(consume())

    for payload in payloads[:2]:
        lifecycle = payload["response"]
        assert lifecycle["status"] == "in_progress"
        assert lifecycle["output"] == []
        assert lifecycle["completed_at"] is None
        assert lifecycle["error"] is None
        assert lifecycle["incomplete_details"] is None
        assert lifecycle["usage"] is None
        assert lifecycle["output_text"] is None
        assert lifecycle["audio"] is None
        assert lifecycle["audio_transcript"] is None

    terminal = payloads[-1]["response"]
    assert terminal["status"] == "completed"
    assert terminal["completed_at"] == response["completed_at"]
    assert terminal["error"] == response["error"]
    assert terminal["incomplete_details"] == response["incomplete_details"]
    assert {
        key: terminal["usage"][key] for key in response["usage"]
    } == response["usage"]
    assert terminal["usage"]["input_tokens_details"] == {"cached_tokens": 0}
    assert terminal["usage"]["output_tokens_details"] == {
        "reasoning_tokens": 0,
    }
    assert terminal["output_text"] == response["output_text"]
    assert terminal["audio"] == response["audio"]
    assert terminal["audio_transcript"] == response["audio_transcript"]


def test_live_responses_preserves_progress_message_until_done(tmp_path):
    backend = Backend(_config(tmp_path))
    state = {
        "sequence": 0,
        "response_id": None,
        "created_at": None,
        "compact_output_indexes": False,
        "items_by_id": {},
        "item_indexes": {},
        "output_indexes": {},
        "next_output_index": 0,
        "done_items": {},
    }
    message = {
        "id": "msg_progress",
        "type": "message",
        "role": "assistant",
        "status": "in_progress",
        "content": [{
            "type": "output_text",
            "text": "partial",
            "annotations": [],
        }],
    }

    progress = backend._normalise_live_responses_event({
        "type": "response.in_progress",
        "response": {
            "id": "resp_progress",
            "status": "in_progress",
            "model": "gpt-5",
            "output": [message],
        },
    }, state)
    done = backend._normalise_live_responses_event({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": message,
    }, state)

    assert progress["response"]["output"][0]["status"] == "in_progress"
    assert done["item"]["status"] == "completed"


def test_responses_replay_standard_events_validate_with_openai_sdk(tmp_path):
    backend = Backend(_config(tmp_path))

    async def consume():
        return [
            _sse_payload(event)
            async for event in backend._replay_responses_sse(
                _response_with_tool_call()
            )
        ]

    payloads = asyncio.run(consume())
    adapter = TypeAdapter(ResponsesServerEvent)

    assert [adapter.validate_python(payload).type for payload in payloads] == [
        payload["type"] for payload in payloads
    ]
    usage = payloads[-1]["response"]["usage"]
    assert usage["input_tokens_details"] == {"cached_tokens": 0}
    assert usage["output_tokens_details"] == {"reasoning_tokens": 0}


@pytest.mark.parametrize(
    ("annotations_present", "raw_annotations", "expected_annotations"),
    [
        (False, None, []),
        (True, None, []),
        (
            True,
            {
                "type": "url_citation",
                "start_index": 0,
                "end_index": 5,
                "title": "Example",
                "url": "https://example.com",
            },
            [{
                "type": "url_citation",
                "start_index": 0,
                "end_index": 5,
                "title": "Example",
                "url": "https://example.com",
            }],
        ),
    ],
)
def test_responses_replay_normalizes_output_text_annotations(
    tmp_path, annotations_present, raw_annotations, expected_annotations,
):
    backend = Backend(_config(tmp_path))
    part = {"type": "output_text", "text": "hello"}
    if annotations_present:
        part["annotations"] = raw_annotations
    response = {
        "id": "resp_annotations",
        "object": "response",
        "status": "completed",
        "model": "gpt-5",
        "output": [{
            "id": "msg_annotations",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [part],
        }],
    }

    async def consume():
        return [
            _sse_payload(event)
            async for event in backend._replay_responses_sse(response)
        ]

    payloads = asyncio.run(consume())
    adapter = TypeAdapter(ResponsesServerEvent)

    assert [adapter.validate_python(payload).type for payload in payloads] == [
        payload["type"] for payload in payloads
    ]
    content_done = next(
        payload for payload in payloads
        if payload["type"] == "response.content_part.done"
    )
    terminal_part = payloads[-1]["response"]["output"][0]["content"][0]
    assert content_done["part"]["annotations"] == expected_annotations
    assert terminal_part["annotations"] == expected_annotations


def test_responses_replay_keeps_generated_identity_stable(tmp_path):
    backend = Backend(_config(tmp_path))

    async def consume():
        return [
            _sse_payload(event)
            async for event in backend._replay_responses_sse({
                "status": "completed",
                "model": "gpt-5",
                "output": [],
            })
        ]

    payloads = asyncio.run(consume())
    snapshots = [payload["response"] for payload in payloads]

    assert len({snapshot["id"] for snapshot in snapshots}) == 1
    assert len({snapshot["created_at"] for snapshot in snapshots}) == 1


@pytest.mark.parametrize("status", ["queued", "in_progress"])
def test_responses_replay_does_not_finalize_background_snapshot(
    tmp_path, status,
):
    backend = Backend(_config(tmp_path))
    response = {
        "id": "resp_background",
        "status": status,
        "model": "gpt-5",
        "output": [{
            "id": "msg_partial",
            "type": "message",
            "role": "assistant",
            "content": [],
        }],
    }

    async def consume():
        return [
            _sse_payload(event)
            async for event in backend._replay_responses_sse(response)
        ]

    payloads = asyncio.run(consume())
    expected_state_event = f"response.{status}"

    assert [payload["type"] for payload in payloads] == [
        "response.created",
        expected_state_event,
    ]
    assert all(
        TypeAdapter(ResponsesServerEvent).validate_python(payload).type
        == payload["type"]
        for payload in payloads
    )


@pytest.mark.parametrize("status", ["failed", "incomplete"])
def test_responses_replay_normalizes_partial_message_status(
    tmp_path, status,
):
    backend = Backend(_config(tmp_path))
    response = {
        "id": f"resp_{status}",
        "status": status,
        "model": "gpt-5",
        "output": [{
            "id": "msg_partial",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "partial",
                "annotations": [],
            }],
        }],
    }

    async def consume():
        return [
            _sse_payload(event)
            async for event in backend._replay_responses_sse(response)
        ]

    payloads = asyncio.run(consume())
    adapter = TypeAdapter(ResponsesServerEvent)
    done = next(
        payload for payload in payloads
        if payload["type"] == "response.output_item.done"
    )

    assert [adapter.validate_python(payload).type for payload in payloads] == [
        payload["type"] for payload in payloads
    ]
    assert done["item"]["status"] == "incomplete"
    assert payloads[-1]["response"]["output"][0]["status"] == "incomplete"


def test_responses_replay_compacts_filtered_output_indexes(tmp_path):
    backend = Backend(_config(tmp_path))
    response = {
        "id": "resp_dense_indexes",
        "status": "completed",
        "model": "gpt-5",
        "output": [
            None,
            {
                "id": "msg_dense",
                "type": "message",
                "role": "assistant",
                "content": [
                    None,
                    {
                        "type": "output_text",
                        "text": "hello",
                        "annotations": [],
                    },
                ],
            },
        ],
    }

    async def consume():
        return [
            _sse_payload(event)
            async for event in backend._replay_responses_sse(response)
        ]

    payloads = asyncio.run(consume())
    indexed = [payload for payload in payloads if "output_index" in payload]
    content_indexed = [
        payload for payload in indexed if "content_index" in payload
    ]

    assert {payload["output_index"] for payload in indexed} == {0}
    assert {payload["content_index"] for payload in content_indexed} == {0}
    assert len(payloads[-1]["response"]["output"]) == 1
    assert len(payloads[-1]["response"]["output"][0]["content"]) == 1


@pytest.mark.parametrize(
    ("cached_tokens", "reasoning_tokens"),
    [(None, None), (True, 1.5), ("1", "2")],
)
def test_responses_replay_normalizes_invalid_usage_details(
    tmp_path, cached_tokens, reasoning_tokens,
):
    backend = Backend(_config(tmp_path))
    response = {
        "id": "resp_usage_details",
        "status": "completed",
        "model": "gpt-5",
        "output": [],
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": cached_tokens},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "total_tokens": 3,
        },
    }

    async def consume():
        return [
            _sse_payload(event)
            async for event in backend._replay_responses_sse(response)
        ]

    terminal = asyncio.run(consume())[-1]
    parsed = TypeAdapter(ResponsesServerEvent).validate_python(terminal)
    usage = terminal["response"]["usage"]

    assert parsed.type == "response.completed"
    assert usage["input_tokens_details"]["cached_tokens"] == 0
    assert usage["output_tokens_details"]["reasoning_tokens"] == 0


def test_responses_replay_rejects_unknown_status(tmp_path):
    backend = Backend(_config(tmp_path))

    async def consume():
        return [
            event
            async for event in backend._replay_responses_sse({
                "status": "future_state",
                "model": "gpt-5",
                "output": [],
            })
        ]

    with pytest.raises(ValueError, match="Unsupported Responses status"):
        asyncio.run(consume())


def test_responses_replay_preserves_statusless_output_items(tmp_path):
    backend = Backend(_config(tmp_path))
    output = [
        {
            "id": "ct_test",
            "type": "custom_tool_call",
            "call_id": "call_test",
            "name": "run_code",
            "input": "print('ok')",
        },
        {
            "id": "prog_test",
            "type": "program",
            "call_id": "program_call_test",
            "code": "await tools.run()",
            "fingerprint": "opaque",
        },
        {
            "id": "mcp_tools_test",
            "type": "mcp_list_tools",
            "server_label": "docs",
            "tools": [],
        },
        {
            "id": "mcp_request_test",
            "type": "mcp_approval_request",
            "arguments": "{}",
            "name": "publish",
            "server_label": "docs",
        },
        {
            "id": "mcp_response_test",
            "type": "mcp_approval_response",
            "approval_request_id": "mcp_request_test",
            "approve": True,
        },
        {
            "id": "cmp_test",
            "type": "compaction",
            "encrypted_content": "encrypted",
        },
        {
            "id": "tools_test",
            "type": "additional_tools",
            "role": "developer",
            "tools": [],
        },
    ]
    response = {
        "id": "resp_statusless",
        "object": "response",
        "status": "completed",
        "model": "gpt-5",
        "output": output,
    }

    async def consume():
        return [
            _sse_payload(event)
            async for event in backend._replay_responses_sse(response)
        ]

    payloads = asyncio.run(consume())
    added = [
        payload["item"]
        for payload in payloads
        if payload["type"] == "response.output_item.added"
    ]
    done = [
        payload["item"]
        for payload in payloads
        if payload["type"] == "response.output_item.done"
    ]
    completed = payloads[-1]["response"]["output"]

    assert [item["type"] for item in added] == [
        item["type"] for item in output
    ]
    assert all("status" not in item for item in added)
    assert all("status" not in item for item in done)
    assert all("status" not in item for item in completed)


@pytest.mark.parametrize(
    ("item_type", "item_status"),
    [
        ("program_output", "completed"),
        ("program_output", "incomplete"),
        ("apply_patch_call_output", "completed"),
        ("apply_patch_call_output", "failed"),
    ],
)
def test_responses_replay_preserves_terminal_only_output_status(
    tmp_path, item_type, item_status,
):
    backend = Backend(_config(tmp_path))
    item = {
        "id": "terminal_output_test",
        "type": item_type,
        "call_id": "call_terminal_output_test",
        "status": item_status,
    }
    if item_type == "program_output":
        item["result"] = "done"
    else:
        item["output"] = "done"
    response = {
        "id": "resp_terminal_output",
        "object": "response",
        "status": {
            "incomplete": "incomplete",
            "failed": "failed",
        }.get(item_status, "completed"),
        "model": "gpt-5",
        "output": [item],
    }

    async def consume():
        return [
            _sse_payload(event)
            async for event in backend._replay_responses_sse(response)
        ]

    payloads = asyncio.run(consume())
    added = next(
        payload["item"] for payload in payloads
        if payload["type"] == "response.output_item.added"
    )
    done = next(
        payload["item"] for payload in payloads
        if payload["type"] == "response.output_item.done"
    )

    assert added["status"] == item_status
    assert done["status"] == item_status
    assert payloads[-1]["response"]["output"][0]["status"] == item_status


def test_responses_replay_uses_failed_event_for_cancelled_status(tmp_path):
    backend = Backend(_config(tmp_path))
    response = {
        "id": "resp_cancelled",
        "object": "response",
        "status": "cancelled",
        "model": "gpt-5",
        "output": [],
    }

    async def consume():
        return [
            _sse_payload(event)
            async for event in backend._replay_responses_sse(response)
        ]

    payloads = asyncio.run(consume())

    assert payloads[-1]["type"] == "response.failed"
    assert payloads[-1]["response"]["status"] == "cancelled"
    assert all(
        payload["type"] != "response.cancelled" for payload in payloads
    )


def test_responses_replay_emits_shell_call_events(tmp_path):
    backend = Backend(_config(tmp_path))
    shell_call = {
        "id": "sh_test",
        "type": "shell_call",
        "call_id": "call_shell_test",
        "action": {
            "commands": ["pwd", "ls -la"],
            "max_output_length": 4096,
            "timeout_ms": 10_000,
        },
        "environment": {"type": "local"},
        "status": "completed",
    }
    shell_output = {
        "id": "sho_test",
        "type": "shell_call_output",
        "call_id": "call_shell_test",
        "max_output_length": 4096,
        "output": [
            {
                "stdout": "/tmp\n",
                "stderr": "",
                "outcome": {"type": "exit", "exit_code": 0},
            },
            {
                "stdout": "file.txt\n",
                "stderr": "warning\n",
                "outcome": {"type": "exit", "exit_code": 0},
            },
        ],
        "status": "completed",
    }
    response = {
        "id": "resp_shell",
        "object": "response",
        "status": "completed",
        "model": "gpt-5",
        "output": [shell_call, shell_output],
    }

    async def consume():
        return [
            _sse_payload(event)
            async for event in backend._replay_responses_sse(response)
        ]

    payloads = asyncio.run(consume())
    added = [
        payload["item"]
        for payload in payloads
        if payload["type"] == "response.output_item.added"
    ]
    command_events = [
        payload for payload in payloads
        if payload["type"].startswith("response.shell_call_command.")
    ]
    output_events = [
        payload for payload in payloads
        if payload["type"].startswith(
            "response.shell_call_output_content."
        )
    ]

    assert added[0]["action"]["commands"] == []
    assert added[1]["output"] == []
    assert [payload["type"] for payload in command_events] == [
        "response.shell_call_command.added",
        "response.shell_call_command.delta",
        "response.shell_call_command.done",
        "response.shell_call_command.added",
        "response.shell_call_command.delta",
        "response.shell_call_command.done",
    ]
    assert command_events[1]["delta"] == "pwd"
    assert command_events[5]["command"] == "ls -la"
    assert [payload["type"] for payload in output_events] == [
        "response.shell_call_output_content.delta",
        "response.shell_call_output_content.done",
        "response.shell_call_output_content.delta",
        "response.shell_call_output_content.done",
    ]
    assert output_events[0]["delta"] == {"stdout": "/tmp\n"}
    assert output_events[2]["delta"] == {
        "stdout": "file.txt\n",
        "stderr": "warning\n",
    }
    assert payloads[-1]["response"]["output"] == [
        shell_call, shell_output,
    ]
    assert [payload["sequence_number"] for payload in payloads] == list(
        range(len(payloads))
    )


def test_responses_route_dispatches_streaming_and_non_streaming(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))
    completed = _response_with_tool_call()
    complete_call = AsyncMock(return_value=(completed, None))
    stream_bodies = []

    async def stream_responses(body):
        stream_bodies.append(json.loads(body))
        yield 'data: {"type":"response.created"}\n\n'
        yield 'data: {"type":"response.completed"}\n\n'

    monkeypatch.setattr(server.backend, "complete_responses", complete_call)
    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def request_both_modes():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            non_streaming = await client.post(
                "/v1/responses", json={"input": "hello"}
            )
            streaming = await client.post(
                "/v1/responses", json={"input": "hello", "stream": True}
            )
        return non_streaming, streaming

    non_streaming, streaming = asyncio.run(request_both_modes())

    assert non_streaming.status_code == 200
    assert non_streaming.json() == completed
    assert complete_call.await_count == 1
    assert json.loads(complete_call.await_args.args[0]) == {"input": "hello"}
    assert streaming.status_code == 200
    assert streaming.headers["content-type"].startswith("text/event-stream")
    assert "response.created" in streaming.text
    assert "response.completed" in streaming.text
    assert stream_bodies == [{"input": "hello", "stream": True}]


@pytest.mark.parametrize(
    "authorization",
    ["Bearer secret", "Bearer this-is-a-much-longer-secret-token"],
)
def test_responses_route_redacts_authorization_debug_log(
    tmp_path, monkeypatch, authorization,
):
    server = ProxyServer(_config(tmp_path))
    complete_call = AsyncMock(return_value=(_response_with_tool_call(), None))
    debug_log = Mock()
    monkeypatch.setattr(server.backend, "complete_responses", complete_call)
    monkeypatch.setattr("turbo_proxy.proxy.proxy.logger.debug", debug_log)

    async def request_response():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello"},
                headers={"Authorization": authorization},
            )

    response = asyncio.run(request_response())
    messages = [call.args[0] for call in debug_log.call_args_list]

    assert response.status_code == 200
    assert "HDR authorization: ***" in messages
    assert all(authorization not in message for message in messages)


def test_non_responses_suffix_path_uses_upstream_passthrough(
    tmp_path, monkeypatch,
):
    from starlette.responses import Response

    server = ProxyServer(_config(tmp_path))
    complete_call = AsyncMock()
    upstream_call = AsyncMock(return_value=Response(status_code=204))
    monkeypatch.setattr(server.backend, "complete_responses", complete_call)
    monkeypatch.setattr(server, "_handle_upstream", upstream_call)

    async def request_response():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post("/admin/responses", json={"input": "hello"})

    response = asyncio.run(request_response())

    assert response.status_code == 204
    complete_call.assert_not_awaited()
    upstream_call.assert_awaited_once()


def test_responses_route_disconnect_closes_backend_stream(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))
    closed = 0

    async def stream_responses(body):
        nonlocal closed
        try:
            yield 'data: {"type":"response.in_progress"}\n\n'
            await asyncio.Future()
        finally:
            closed += 1

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def disconnect():
        response = await server._responses_streaming(b"{}", 0.0)
        iterator = response.body_iterator
        first = await iterator.__anext__()
        await iterator.aclose()
        await asyncio.sleep(0)
        return first

    first = asyncio.run(disconnect())

    assert 'event: response.in_progress' in first
    assert closed == 1


def test_responses_route_disconnect_before_body_closes_prefetched_stream(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))
    closed = 0

    async def stream_responses(body):
        nonlocal closed
        try:
            yield 'data: {"type":"response.in_progress"}\n\n'
            await asyncio.Future()
        finally:
            closed += 1

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def disconnect_before_body_iteration():
        incoming = [
            {
                "type": "http.request",
                "body": b'{"input":"hello","stream":true}',
                "more_body": False,
            },
            {"type": "http.disconnect"},
        ]

        async def receive():
            if incoming:
                return incoming.pop(0)
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                # The disconnect listener cancels this task before Starlette
                # starts iterating the response body.
                await asyncio.Future()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/responses",
            "raw_path": b"/v1/responses",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 80),
            "root_path": "",
        }
        await server.app(scope, receive, send)
        await asyncio.sleep(0)

    asyncio.run(disconnect_before_body_iteration())

    assert closed == 1


def test_responses_stream_cancellation_preserves_primary_when_cleanup_fails(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))
    stream = _CleanupFailingResponsesStream(
        ['data: {"type":"response.created"}\n\n'],
        hang=True,
    )
    monkeypatch.setattr(
        server.backend, "stream_responses", lambda body: stream,
    )

    async def cancel_during_iteration():
        response = await server._responses_streaming(b"{}", 0.0)
        iterator = response.body_iterator
        await anext(iterator)
        pending = asyncio.create_task(anext(iterator))
        await stream.started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await pending
        return caught.value

    cancellation = asyncio.run(cancel_during_iteration())

    assert str(cancellation.__cause__) == "cleanup-1"
    assert stream.close_calls == 1


def test_responses_route_disconnect_does_not_retry_failed_cleanup(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))
    stream = _CleanupFailingResponsesStream(
        ['data: {"type":"response.created"}\n\n'],
        hang=True,
    )
    monkeypatch.setattr(
        server.backend, "stream_responses", lambda body: stream,
    )

    async def disconnect():
        response = await server._responses_streaming(b"{}", 0.0)

        async def receive():
            await stream.started.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        await response(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
            },
            receive,
            send,
        )

    asyncio.run(disconnect())

    assert stream.close_calls == 1


def test_responses_terminal_stream_ignores_cleanup_failure(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))
    stream = _CleanupFailingResponsesStream(
        ['data: {"type":"response.completed"}\n\n'],
    )
    monkeypatch.setattr(
        server.backend, "stream_responses", lambda body: stream,
    )

    async def request_stream():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_stream())

    assert response.status_code == 200
    assert "event: response.completed" in response.text
    assert stream.close_calls == 1


def test_responses_iteration_error_survives_cleanup_failure(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))
    stream = _CleanupFailingResponsesStream(
        ['data: {"type":"response.created"}\n\n'],
        error=RuntimeError("upstream iteration failed"),
    )
    monkeypatch.setattr(
        server.backend, "stream_responses", lambda body: stream,
    )

    async def request_stream():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_stream())
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]

    assert response.status_code == 200
    assert payloads[-1]["type"] == "error"
    assert payloads[-1]["message"] == "upstream iteration failed"
    assert stream.close_calls == 1


def test_responses_route_returns_openai_error_when_backend_raises(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))
    complete_call = AsyncMock(side_effect=RuntimeError("upstream unavailable"))
    monkeypatch.setattr(server.backend, "complete_responses", complete_call)

    async def request_response():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post("/v1/responses", json={"input": "hello"})

    response = asyncio.run(request_response())

    assert response.status_code == 500
    assert response.json()["error"]["type"] == "server_error"
    assert "upstream unavailable" in response.json()["error"]["message"]
    complete_call.assert_awaited_once()


def test_responses_route_preserves_upstream_4xx_error(tmp_path, monkeypatch):
    server = ProxyServer(_config(tmp_path))
    upstream_error = _upstream_bad_request()
    wrapped_error = RuntimeError("all candidate Responses requests failed")
    wrapped_error.__cause__ = upstream_error
    complete_call = AsyncMock(side_effect=wrapped_error)
    monkeypatch.setattr(server.backend, "complete_responses", complete_call)

    async def request_response():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post("/v1/responses", json={"input": "hello"})

    response = asyncio.run(request_response())

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "temperature must be <= 2",
            "type": "invalid_request_error",
            "param": "temperature",
            "code": "invalid_value",
        }
    }
    complete_call.assert_awaited_once()


def test_responses_stream_preserves_4xx_before_sending_headers(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))

    async def stream_responses(body):
        raise _upstream_bad_request()
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def request_response():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_response())

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"] == {
        "message": "temperature must be <= 2",
        "type": "invalid_request_error",
        "param": "temperature",
        "code": "invalid_value",
    }


def test_responses_empty_stream_returns_protocol_error_before_headers(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))

    async def stream_responses(body):
        if False:
            yield ""  # pragma: no cover

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def request_response():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_response())

    assert response.status_code == 502
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"] == {
        "message": "Upstream Responses stream ended before emitting an event",
        "type": "server_error",
        "param": None,
        "code": None,
    }


def test_responses_stream_without_terminal_event_emits_protocol_error(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))

    async def stream_responses(body):
        yield 'data: {"type":"response.created"}\n\n'

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def request_stream():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_stream())
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]

    assert response.status_code == 200
    assert response.text.count("event: error") == 1
    assert payloads[-1] == {
        "type": "error",
        "code": "server_error",
        "message": "Upstream Responses stream ended without a terminal event",
        "param": None,
        "sequence_number": 1,
    }
    assert TypeAdapter(ResponsesServerEvent).validate_python(payloads[-1]).type == (
        "error"
    )


def test_responses_route_drops_done_sentinel(tmp_path, monkeypatch):
    server = ProxyServer(_config(tmp_path))

    async def stream_responses(body):
        yield 'data: {"type":"response.completed"}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def request_stream():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_stream())

    assert response.status_code == 200
    assert '"type":"response.completed"' in response.text
    assert "[DONE]" not in response.text


def test_responses_route_drops_done_sentinel_when_shared_with_payload(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))

    async def stream_responses(body):
        yield (
            "event: response.completed\n"
            'data: {"type":"response.completed"}\n'
            "data: [DONE]\n\n"
        )

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def request_stream():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_stream())

    assert response.status_code == 200
    assert response.text.count('"type":"response.completed"') == 1
    assert "[DONE]" not in response.text


def test_responses_route_does_not_emit_error_after_terminal_event(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))

    async def stream_responses(body):
        yield (
            "event: response.created\n"
            'data: {"type":"response.created"}\n\n'
        )
        yield (
            "event: response.completed\n"
            'data: {"type":"response.completed"}\n\n'
        )
        raise RuntimeError("late upstream failure")

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def request_stream():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_stream())

    assert response.status_code == 200
    assert "event: response.completed" in response.text
    assert "event: error" not in response.text


def test_responses_route_stops_after_first_terminal_event(tmp_path, monkeypatch):
    server = ProxyServer(_config(tmp_path))

    async def stream_responses(body):
        yield 'data: {"type":"response.completed"}\n\n'
        yield 'data: {"type":"response.failed"}\n\n'

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def request_stream():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_stream())
    terminal_lines = [
        line for line in response.text.splitlines()
        if line in {
            "event: response.completed",
            "event: response.incomplete",
            "event: response.failed",
            "event: error",
        }
    ]

    assert terminal_lines == ["event: response.completed"]


@pytest.mark.parametrize(
    "event",
    [
        "event: response.completed\ndata: not-json\n\n",
        (
            "event: response.completed\n"
            'data: {"type":"not-a-responses-event"}\n\n'
        ),
    ],
)
def test_responses_route_replaces_invalid_terminal_frame_with_error(
    tmp_path, monkeypatch, event,
):
    server = ProxyServer(_config(tmp_path))

    async def stream_responses(body):
        yield event

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def request_stream():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_stream())
    event_names = [
        line.removeprefix("event: ")
        for line in response.text.splitlines()
        if line.startswith("event: ")
    ]
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]

    assert response.status_code == 200
    assert event_names == ["error"]
    assert len(payloads) == 1
    assert TypeAdapter(ResponsesServerEvent).validate_python(payloads[0]).type == (
        "error"
    )


def test_responses_route_preserves_error_details_after_stream_start(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))
    upstream_error = RuntimeError("wrapped upstream failure")
    upstream_error.status_code = 429
    upstream_error.body = {
        "error": {
            "message": "slow down",
            "type": "rate_limit_error",
            "param": "model",
            "code": "rate_limit_exceeded",
        }
    }

    async def stream_responses(body):
        yield 'data: {"type":"response.created"}\n\n'
        raise upstream_error

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def request_stream():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_stream())
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]

    assert response.status_code == 200
    assert payloads[-1] == {
        "type": "error",
        "code": "rate_limit_exceeded",
        "message": "slow down",
        "param": "model",
        "sequence_number": 1,
    }


def test_responses_route_detects_data_only_terminal_event(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))

    async def stream_responses(body):
        # Some compatible upstreams omit the SSE event name and send only the
        # protocol JSON payload.
        yield 'data: {"type":"response.completed","sequence_number":7}\n\n'
        raise RuntimeError("late upstream failure")

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def request_stream():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_stream())

    assert response.status_code == 200
    assert '"type":"response.completed"' in response.text
    assert "event: error" not in response.text


def test_responses_route_normalizes_enum_payload_type(tmp_path, monkeypatch):
    server = ProxyServer(_config(tmp_path))

    async def stream_responses(body):
        yield 'data: {"type":"RESPONSE_COMPLETED"}\n\n'

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def request_stream():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_stream())

    assert response.status_code == 200
    assert "event: response.completed" in response.text
    data_line = next(
        line for line in response.text.splitlines() if line.startswith("data: ")
    )
    payload = json.loads(data_line.removeprefix("data: "))
    assert payload["type"] == "response.completed"


def test_responses_route_parses_multiline_sse_data_before_terminal_detection(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))

    async def stream_responses(body):
        yield (
            "event: response.completed\n"
            "data: {\n"
            'data: "type": "response.completed",\n'
            'data: "sequence_number": 7\n'
            "data: }\n\n"
        )
        raise RuntimeError("late upstream failure")

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def request_stream():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_stream())

    assert response.status_code == 200
    assert '"type":"response.completed"' in response.text
    assert '"sequence_number":0' in response.text
    assert "event: error" not in response.text


def test_responses_route_does_not_emit_error_after_cancelled_response(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))

    async def stream_responses(body):
        yield (
            "event: response.failed\n"
            'data: {"type":"response.failed",'
            '"response":{"status":"cancelled"}}\n\n'
        )
        raise RuntimeError("late upstream failure")

    monkeypatch.setattr(server.backend, "stream_responses", stream_responses)

    async def request_stream():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True},
            )

    response = asyncio.run(request_stream())

    assert response.status_code == 200
    assert "event: response.failed" in response.text
    assert '"status":"cancelled"' in response.text
    assert "event: error" not in response.text


def test_responses_route_returns_openai_error_for_invalid_json(tmp_path):
    server = ProxyServer(_config(tmp_path))

    async def request_invalid_json():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post(
                "/v1/responses",
                content=b'{"input":',
                headers={"content-type": "application/json"},
            )

    response = asyncio.run(request_invalid_json())

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["message"].startswith("Invalid JSON:")


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_responses_route_rejects_nonfinite_json_constants(
    tmp_path, monkeypatch, constant,
):
    server = ProxyServer(_config(tmp_path))
    complete_call = AsyncMock()
    monkeypatch.setattr(server.backend, "complete_responses", complete_call)

    async def request_invalid_json():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/responses",
                content=(
                    f'{{"input":"hello","temperature":{constant}}}'
                ).encode(),
                headers={"content-type": "application/json"},
            )

    response = asyncio.run(request_invalid_json())

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        f"Invalid JSON: non-finite number {constant} is not valid JSON"
    )
    complete_call.assert_not_awaited()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"input": None}, "input must be a string or array"),
        ({"input": ["hello"]}, "input array items must be objects"),
        ({"input": "hello", "stream": "true"}, "stream must be a boolean"),
        (
            {"input": "hello", "background": True},
            "background=true is not supported because Responses retrieval "
            "endpoints are not implemented",
        ),
        (
            {"input": "hello", "background": 1},
            "background must be a boolean or null",
        ),
        (
            {"input": "hello", "background": "true"},
            "background must be a boolean or null",
        ),
        (
            {"input": "hello", "extra_headers": []},
            "extra_headers must be an object or null",
        ),
        (
            {"input": "hello", "extra_headers": {"X-Trace": 123}},
            "extra_headers values must be strings",
        ),
        (
            {
                "input": "hello",
                "extra_body": {
                    "conversation": "conv_bypass",
                    "background": True,
                },
            },
            "extra_body cannot override protected field(s): "
            "background, conversation",
        ),
    ],
)
def test_responses_route_rejects_invalid_requests_before_backend(
    tmp_path, monkeypatch, body, message,
):
    server = ProxyServer(_config(tmp_path))
    complete_call = AsyncMock()
    monkeypatch.setattr(server.backend, "complete_responses", complete_call)

    async def request_invalid():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post("/v1/responses", json=body)

    response = asyncio.run(request_invalid())

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["message"] == message
    complete_call.assert_not_awaited()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("conversation", "conv_shared"),
        ("previous_response_id", "resp_previous"),
        ("prompt", {"id": "pmpt_shared"}),
    ],
)
def test_responses_route_rejects_stateful_fanout_with_verifier(
    tmp_path, monkeypatch, field, value,
):
    server = ProxyServer(_config(tmp_path))
    server.backend.verifier = object()
    complete_call = AsyncMock()
    monkeypatch.setattr(server.backend, "complete_responses", complete_call)
    body = {"input": "hello", field: value}

    async def request_invalid():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post("/v1/responses", json=body)

    response = asyncio.run(request_invalid())

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        f"Responses parameter(s) {field} cannot be used when verifier is enabled"
    )
    complete_call.assert_not_awaited()


def test_responses_route_accepts_omitted_input(tmp_path, monkeypatch):
    server = ProxyServer(_config(tmp_path))
    complete_call = AsyncMock(
        return_value=({"id": "resp_without_input", "object": "response"}, None)
    )
    monkeypatch.setattr(server.backend, "complete_responses", complete_call)

    async def request_without_input():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post("/v1/responses", json={})

    response = asyncio.run(request_without_input())

    assert response.status_code == 200
    assert response.json()["id"] == "resp_without_input"
    complete_call.assert_awaited_once()


def test_responses_route_rejects_invalid_stream_request_before_headers(
    tmp_path, monkeypatch,
):
    server = ProxyServer(_config(tmp_path))
    stream_call = AsyncMock()
    monkeypatch.setattr(server.backend, "stream_responses", stream_call)

    async def request_invalid():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.post(
                "/v1/responses",
                json={"input": "hello", "stream": True, "timeout": 0},
            )

    response = asyncio.run(request_invalid())

    assert response.status_code == 400
    assert not response.headers["content-type"].startswith("text/event-stream")
    assert response.json()["error"]["message"] == "timeout must be a positive number"
    stream_call.assert_not_awaited()


def _collect_replayed_payloads(backend, response):
    async def consume():
        return [
            _sse_payload(event)
            async for event in backend._replay_responses_sse(response)
        ]

    return asyncio.run(consume())


@pytest.mark.parametrize(
    ("item", "completed_event"),
    [
        (
            {
                "id": "fs_transition",
                "type": "file_search_call",
                "status": "searching",
                "queries": ["query"],
            },
            "response.file_search_call.completed",
        ),
        (
            {
                "id": "ig_transition",
                "type": "image_generation_call",
                "status": "generating",
                "result": "aW1hZ2U=",
            },
            "response.image_generation_call.completed",
        ),
        (
            {
                "id": "ci_transition",
                "type": "code_interpreter_call",
                "status": "interpreting",
                "container_id": "container_test",
                "code": "print('ok')",
                "outputs": [],
            },
            "response.code_interpreter_call.completed",
        ),
        (
            {
                "id": "mcp_transition",
                "type": "mcp_call",
                "status": "calling",
                "server_label": "docs",
                "name": "lookup",
                "arguments": "{}",
            },
            "response.mcp_call.completed",
        ),
    ],
)
def test_responses_replay_converts_transitional_items_to_terminal_status(
    tmp_path, item, completed_event,
):
    backend = Backend(_config(tmp_path))
    payloads = _collect_replayed_payloads(backend, {
        "id": "resp_transitional_item",
        "object": "response",
        "status": "completed",
        "model": "gpt-5",
        "output": [item],
    })

    done = next(
        payload["item"] for payload in payloads
        if payload["type"] == "response.output_item.done"
    )
    assert done["status"] == "completed"
    assert payloads[-1]["response"]["output"][0]["status"] == "completed"
    assert completed_event in {payload["type"] for payload in payloads}


@pytest.mark.parametrize(
    (
        "response_status", "item_status", "error", "expected_status",
        "expected_event",
    ),
    [
        ("failed", None, None, "failed", "response.mcp_call.failed"),
        (
            "completed", "completed", "tool failed", "failed",
            "response.mcp_call.failed",
        ),
        (
            "failed", "completed", None, "completed",
            "response.mcp_call.completed",
        ),
    ],
)
def test_responses_replay_uses_one_terminal_status_for_mcp_events_and_items(
    tmp_path, response_status, item_status, error, expected_status,
    expected_event,
):
    backend = Backend(_config(tmp_path))
    item = {
        "id": "mcp_terminal",
        "type": "mcp_call",
        "server_label": "docs",
        "name": "lookup",
        "arguments": "{}",
    }
    if item_status is not None:
        item["status"] = item_status
    if error is not None:
        item["error"] = error
    payloads = _collect_replayed_payloads(backend, {
        "id": "resp_mcp_terminal",
        "object": "response",
        "status": response_status,
        "model": "gpt-5",
        "output": [item],
    })

    lifecycle = {
        payload["type"] for payload in payloads
        if payload["type"] in {
            "response.mcp_call.completed", "response.mcp_call.failed",
        }
    }
    done = next(
        payload["item"] for payload in payloads
        if payload["type"] == "response.output_item.done"
    )
    assert lifecycle == {expected_event}
    assert done["status"] == expected_status
    assert payloads[-1]["response"]["output"][0]["status"] == expected_status


@pytest.mark.parametrize(
    ("item_status", "response_status", "expected_status"),
    [
        (None, "completed", "completed"),
        ("invalid", "incomplete", "incomplete"),
    ],
)
def test_responses_replay_normalizes_program_output_status(
    tmp_path, item_status, response_status, expected_status,
):
    backend = Backend(_config(tmp_path))
    item = {
        "id": "program_output_status",
        "type": "program_output",
        "call_id": "program_call_status",
        "result": "done",
    }
    if item_status is not None:
        item["status"] = item_status
    payloads = _collect_replayed_payloads(backend, {
        "id": "resp_program_output_status",
        "object": "response",
        "status": response_status,
        "model": "gpt-5",
        "output": [item],
    })

    item_payloads = [
        payload["item"] for payload in payloads
        if payload["type"] in {
            "response.output_item.added", "response.output_item.done",
        }
    ]
    assert [payload["status"] for payload in item_payloads] == [
        expected_status, expected_status,
    ]
    assert payloads[-1]["response"]["output"][0]["status"] == expected_status


@pytest.mark.parametrize(
    ("raw_index", "expected_index"),
    [
        (None, 0),
        (True, 0),
        (-1, 0),
        (1.5, 0),
        ("2", 0),
        ({}, 0),
        (3, 3),
    ],
)
def test_responses_replay_normalizes_partial_image_index(
    tmp_path, raw_index, expected_index,
):
    backend = Backend(_config(tmp_path))
    payloads = _collect_replayed_payloads(backend, {
        "id": "resp_partial_image_index",
        "object": "response",
        "status": "completed",
        "model": "gpt-5",
        "output": [{
            "id": "ig_partial_image_index",
            "type": "image_generation_call",
            "status": "completed",
            "result": "aW1hZ2U=",
            "partial_image_index": raw_index,
        }],
    })

    partial = next(
        payload for payload in payloads
        if payload["type"] == "response.image_generation_call.partial_image"
    )
    assert partial["partial_image_index"] == expected_index
    assert isinstance(partial["partial_image_index"], int)
    assert not isinstance(partial["partial_image_index"], bool)


def test_backend_responses_request_log_redacts_transport_extras(
    tmp_path, monkeypatch,
):
    backend = Backend(_config(tmp_path))
    response_call = AsyncMock(return_value=_response_with_tool_call())
    save_log = Mock()
    monkeypatch.setattr(backend_module, "llm_response", response_call)
    monkeypatch.setattr(backend_module, "save_request_log", save_log)
    body = {
        "input": "hello",
        "extra_body": {
            "vendor_access_token": "body-secret",
            "routing": {"tenant": "private-tenant"},
        },
        "extra_headers": {
            "X-Custom-Token": "header-secret",
            "X-Request-ID": "trace-123",
        },
        "extra_query": {
            "access_token": "query-secret",
            "tenant": "acme",
        },
        "tools": [
            {
                "type": "mcp",
                "server_label": "private-server",
                "server_url": (
                    "https://user:pass@mcp.example/sse?access_token=url-secret"
                    "#private"
                ),
                "authorization": "Bearer mcp-secret",
                "headers": {
                    "Authorization": "Bearer header-secret",
                    "X-Tenant": "private-tenant",
                },
            },
            {
                "type": "function",
                "name": "lookup",
                "parameters": {"type": "object"},
            },
        ],
    }

    result, error = asyncio.run(backend.complete_responses(json.dumps(body)))

    assert error is None
    assert result == _response_with_tool_call()
    sent = response_call.await_args.kwargs
    assert sent["extra_body"] == body["extra_body"]
    assert sent["extra_headers"] == body["extra_headers"]
    assert sent["extra_query"] == body["extra_query"]
    assert sent["tools"] == body["tools"]
    logged_request = save_log.call_args.args[0]["request"]
    assert logged_request["extra_body"] == {
        "vendor_access_token": "***",
        "routing": "***",
    }
    assert logged_request["extra_headers"] == {
        "X-Custom-Token": "***",
        "X-Request-ID": "***",
    }
    assert logged_request["extra_query"] == {
        "access_token": "***",
        "tenant": "***",
    }
    assert logged_request["tools"] == [
        {
            "type": "mcp",
            "server_label": "private-server",
            "server_url": (
                "https://<redacted>@mcp.example/sse?<redacted>"
                "#<redacted>"
            ),
            "authorization": "***",
            "headers": {
                "Authorization": "***",
                "X-Tenant": "***",
            },
        },
        body["tools"][1],
    ]


@pytest.mark.parametrize(
    ("summary_present", "raw_summary", "expected_summary"),
    [
        (False, None, []),
        (True, None, []),
        (
            True,
            {"type": "summary_text", "text": "why"},
            [{"type": "summary_text", "text": "why"}],
        ),
        (
            True,
            "why",
            [{"type": "summary_text", "text": "why"}],
        ),
        (
            True,
            ["first", {"text": "second"}],
            [
                {"type": "summary_text", "text": "first"},
                {"type": "summary_text", "text": "second"},
            ],
        ),
    ],
)
def test_responses_replay_normalizes_reasoning_summary_in_terminal_items(
    tmp_path, summary_present, raw_summary, expected_summary,
):
    backend = Backend(_config(tmp_path))
    item = {
        "id": "reasoning_summary",
        "type": "reasoning",
        "status": "completed",
    }
    if summary_present:
        item["summary"] = raw_summary
    payloads = _collect_replayed_payloads(backend, {
        "id": "resp_reasoning_summary",
        "object": "response",
        "status": "completed",
        "model": "gpt-5",
        "output": [item],
    })
    adapter = TypeAdapter(ResponsesServerEvent)

    assert [adapter.validate_python(payload).type for payload in payloads] == [
        payload["type"] for payload in payloads
    ]
    done = next(
        payload["item"] for payload in payloads
        if payload["type"] == "response.output_item.done"
    )
    assert done["summary"] == expected_summary
    assert payloads[-1]["response"]["output"][0]["summary"] == expected_summary
