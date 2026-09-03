"""Regression coverage for LiteLLM's Responses-to-Chat provider bridge."""

import asyncio
import json
from collections import Counter
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import yaml
from openai.types.responses import Response, ResponsesServerEvent
from pydantic import TypeAdapter

import turbo_proxy.proxy.backend as backend_module
import turbo_proxy.utils.llm as llm_module
from turbo_proxy.proxy.backend import Backend
from turbo_proxy.proxy.proxy import ProxyServer
from turbo_proxy.utils import Config


PROVIDER_MODELS = {
    "deepseek": "deepseek/deepseek-chat",
    "anthropic": "anthropic/claude-3-5-sonnet-latest",
    "gemini": "gemini/gemini-2.5-flash",
    "vertex_ai": "vertex_ai/gemini-2.5-flash",
}

UNSUPPORTED_FALLBACK_TOOL_TYPES = (
    "apply_patch",
    "code_interpreter",
    "computer",
    "computer_use_preview",
    "computer_use",
    "file_search",
    "future_tool",
    "image_generation",
    "local_shell",
    "mcp",
    "namespace",
    "programmatic_tool_calling",
    "shell",
    "tool_search",
    "web_search",
    "web_search_preview",
)

NON_STREAM_ENDPOINTS = {
    "deepseek": "/chat/completions",
    "anthropic": "/v1/messages",
    "gemini": ":generateContent",
    "vertex_ai": ":generateContent",
}

STREAM_ENDPOINTS = {
    "deepseek": "/chat/completions",
    "anthropic": "/v1/messages",
    "gemini": ":streamGenerateContent",
    "vertex_ai": ":streamGenerateContent",
}


def _config(tmp_path, provider):
    model = {"name": PROVIDER_MODELS[provider]}
    if provider != "vertex_ai":
        model["api_key"] = "test-key"

    config_path = tmp_path / f"turbo-proxy-{provider}.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "backend": {"models": [model]},
            "log_dir": str(tmp_path / "logs"),
        })
    )
    return Config(str(config_path))


def _forced_chat_completions_config(tmp_path):
    config_path = tmp_path / "turbo-proxy-forced-chat.yaml"
    config_path.write_text(yaml.safe_dump({
        "backend": {
            "models": [{
                "name": "openai/chat_completions/gpt-4o",
                "api_key": "test-key",
            }],
        },
        "log_dir": str(tmp_path / "logs"),
    }))
    return Config(str(config_path))


def _prepare_provider(monkeypatch, provider):
    if provider != "vertex_ai":
        return

    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

    async def fake_access_token(
        self, credentials, project_id, custom_llm_provider,
    ):
        return "test-vertex-token", "test-project"

    monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")
    monkeypatch.setattr(
        VertexBase,
        "_ensure_access_token_async",
        fake_access_token,
    )


def _function_tool():
    return {
        "type": "function",
        "name": "lookup_weather",
        "description": "Look up the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }


def _provider_response(provider, *, tool_call=False):
    if provider == "deepseek":
        message = {"role": "assistant", "content": "fallback text"}
        finish_reason = "stop"
        if tool_call:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city":"Taipei"}',
                    },
                }],
            }
            finish_reason = "tool_calls"
        return {
            "id": "chatcmpl-deepseek",
            "object": "chat.completion",
            "created": 1_725_000_000,
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None,
            }],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        }

    if provider == "anthropic":
        content = [{"type": "text", "text": "fallback text"}]
        stop_reason = "end_turn"
        if tool_call:
            content = [{
                "type": "tool_use",
                "id": "call_weather",
                "name": "lookup_weather",
                "input": {"city": "Taipei"},
            }]
            stop_reason = "tool_use"
        return {
            "id": "msg_anthropic",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-latest",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }

    part = {"text": "fallback text"}
    if tool_call:
        part = {
            "functionCall": {
                "name": "lookup_weather",
                "args": {"city": "Taipei"},
            }
        }
    return {
        "candidates": [{
            "content": {"parts": [part], "role": "model"},
            "finishReason": "STOP",
            "index": 0,
            "safetyRatings": [],
        }],
        "usageMetadata": {
            "promptTokenCount": 7,
            "candidatesTokenCount": 3,
            "totalTokenCount": 10,
        },
        "modelVersion": "gemini-2.5-flash",
        "responseId": f"{provider}-response",
    }


def _deepseek_tool_stream():
    base = {
        "id": "chatcmpl-deepseek-stream",
        "object": "chat.completion.chunk",
        "created": 1_725_000_000,
        "model": "deepseek-chat",
    }
    chunks = [
        {
            **base,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "arguments": '{"city":',
                        },
                    }],
                },
                "finish_reason": None,
            }],
        },
        {
            **base,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '"Taipei"}'},
                    }]
                },
                "finish_reason": None,
            }],
        },
        {
            **base,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls",
            }],
        },
        {
            **base,
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        },
    ]
    return "".join(
        f"data: {json.dumps(chunk)}\n\n" for chunk in chunks
    ) + "data: [DONE]\n\n"


def _anthropic_tool_stream():
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_anthropic_stream",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet-latest",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 7, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_weather",
                    "name": "lookup_weather",
                    "input": {},
                },
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"city":"Taipei"}',
                },
            },
        ),
        (
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use",
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(
        f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
        for event_type, payload in events
    )


def _gemini_tool_stream(provider):
    payload = {
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "lookup_weather",
                        "args": {"city": "Taipei"},
                    }
                }],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
            "safetyRatings": [],
        }],
        "usageMetadata": {
            "promptTokenCount": 7,
            "candidatesTokenCount": 3,
            "totalTokenCount": 10,
        },
        "modelVersion": "gemini-2.5-flash",
        "responseId": f"{provider}-stream-response",
    }
    return f"data: {json.dumps(payload)}\n\n"


def _provider_tool_stream(provider):
    if provider == "deepseek":
        return _deepseek_tool_stream()
    if provider == "anthropic":
        return _anthropic_tool_stream()
    return _gemini_tool_stream(provider)


def _install_http_stub(monkeypatch, provider, *, tool_call=False, stream=False):
    requests = []

    async def fake_send(client, request, **kwargs):
        for hook in client._event_hooks.get("request", ()):
            await hook(request)
        requests.append(request)
        if stream:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_provider_tool_stream(provider).encode(),
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json=_provider_response(provider, tool_call=tool_call),
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    return requests


def _sse_payloads(events):
    payloads = []
    for event in events:
        data_lines = []
        for line in event.splitlines():
            if not line.startswith("data:"):
                continue
            value = line.removeprefix("data:")
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        if data == "[DONE]":
            continue
        payloads.append(json.loads(data))
    return payloads


def _normalised_live_payloads(tmp_path, monkeypatch, events):
    async def fake_stream_response(**kwargs):
        async def stream():
            for event in events:
                yield event

        return stream()

    monkeypatch.setattr(
        backend_module, "llm_stream_response", fake_stream_response,
    )
    backend = Backend(_config(tmp_path, "deepseek"))

    async def consume():
        return [
            event
            async for event in backend.stream_responses(json.dumps({
                "input": "Use the tools.",
                "stream": True,
            }))
        ]

    return _sse_payloads(asyncio.run(consume()))


@pytest.mark.parametrize("provider", PROVIDER_MODELS)
def test_non_stream_chat_fallback_returns_sdk_valid_response(
    tmp_path, monkeypatch, provider,
):
    _prepare_provider(monkeypatch, provider)
    requests = _install_http_stub(monkeypatch, provider)
    monkeypatch.setattr(backend_module, "save_request_log", Mock())
    backend = Backend(_config(tmp_path, provider))

    result, error = asyncio.run(
        backend.complete_responses(json.dumps({"input": "Say hello."}))
    )

    assert error is None
    response = Response.model_validate(result)
    assert response.status == "completed"
    assert response.output_text == "fallback text"
    assert len(requests) == 1
    assert NON_STREAM_ENDPOINTS[provider] in str(requests[0].url)
    assert "/responses" not in requests[0].url.path


@pytest.mark.parametrize("provider", PROVIDER_MODELS)
def test_pure_function_call_fallback_has_no_empty_text_message(
    tmp_path, monkeypatch, provider,
):
    _prepare_provider(monkeypatch, provider)
    requests = _install_http_stub(monkeypatch, provider, tool_call=True)
    monkeypatch.setattr(backend_module, "save_request_log", Mock())
    backend = Backend(_config(tmp_path, provider))

    result, error = asyncio.run(backend.complete_responses(json.dumps({
        "input": "Use the weather tool.",
        "tools": [_function_tool()],
        "tool_choice": "required",
    })))

    assert error is None
    Response.model_validate(result)
    output = result["output"]
    function_calls = [
        item for item in output if item.get("type") == "function_call"
    ]
    assert function_calls
    messages = [item for item in output if item.get("type") == "message"]
    assert all(message.get("content") for message in messages)
    assert not any(
        part.get("type") == "output_text" and part.get("text") is None
        for message in messages
        for part in message.get("content", [])
    )
    assert len(requests) == 1


def test_anthropic_fallback_thinking_has_no_literal_extra_body(monkeypatch):
    requests = []

    async def fake_send(client, request, **kwargs):
        for hook in client._event_hooks.get("request", ()):
            await hook(request)
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json=_provider_response("anthropic"),
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    asyncio.run(llm_module.llm_response(
        model="anthropic/claude-3-7-sonnet-20250219",
        provider="anthropic",
        input="Think before answering.",
        api_key="test-key",
        max_output_tokens=2048,
        thinking_budget=1024,
    ))

    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload["thinking"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }
    assert "extra_body" not in payload


@pytest.mark.parametrize("provider", PROVIDER_MODELS)
def test_tool_call_stream_fallback_is_sdk_valid_and_internally_consistent(
    tmp_path, monkeypatch, provider,
):
    _prepare_provider(monkeypatch, provider)
    requests = _install_http_stub(monkeypatch, provider, stream=True)
    backend = Backend(_config(tmp_path, provider))

    async def consume():
        return [
            event
            async for event in backend.stream_responses(json.dumps({
                "input": "Use the weather tool.",
                "stream": True,
                "tools": [_function_tool()],
                "tool_choice": "required",
            }))
        ]

    payloads = _sse_payloads(asyncio.run(consume()))
    adapter = TypeAdapter(ResponsesServerEvent)

    assert payloads
    assert [
        adapter.validate_python(payload).type for payload in payloads
    ] == [payload["type"] for payload in payloads]
    assert [payload["sequence_number"] for payload in payloads] == list(
        range(len(payloads))
    )

    lifecycle_types = {
        "response.created", "response.in_progress", "response.completed",
    }
    lifecycle = [
        payload for payload in payloads if payload["type"] in lifecycle_types
    ]
    assert {payload["type"] for payload in lifecycle} == lifecycle_types
    assert len({payload["response"]["id"] for payload in lifecycle}) == 1

    added = [
        payload
        for payload in payloads
        if payload["type"] == "response.output_item.added"
    ]
    done = [
        payload for payload in payloads
        if payload["type"] == "response.output_item.done"
    ]
    arguments_done = [
        payload for payload in payloads
        if payload["type"] == "response.function_call_arguments.done"
    ]

    completed = next(
        payload for payload in reversed(payloads)
        if payload["type"] == "response.completed"
    )
    terminal_output = completed["response"]["output"]
    added_keys = Counter(
        (payload["output_index"], payload["item"]["id"], payload["item"]["type"])
        for payload in added
    )
    done_keys = Counter(
        (payload["output_index"], payload["item"]["id"], payload["item"]["type"])
        for payload in done
    )
    terminal_keys = Counter(
        (output_index, item["id"], item["type"])
        for output_index, item in enumerate(terminal_output)
    )

    assert added_keys
    assert added_keys == done_keys == terminal_keys
    assert sorted({payload["output_index"] for payload in added}) == list(
        range(len(added_keys))
    )
    assert Counter(
        (payload["output_index"], payload["item_id"])
        for payload in arguments_done
    ) == Counter(
        (payload["output_index"], payload["item"]["id"])
        for payload in added
        if payload["item"]["type"] == "function_call"
    )
    assert {payload["name"] for payload in arguments_done} == {
        "lookup_weather"
    }
    assert not any(
        payload["type"] in {
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.added",
            "response.content_part.done",
        }
        for payload in payloads
    )
    assert len(requests) == 1
    assert STREAM_ENDPOINTS[provider] in str(requests[0].url)
    assert "/responses" not in requests[0].url.path


def test_live_function_call_names_follow_interleaved_item_ids(
    tmp_path, monkeypatch,
):
    first_done = {
        "type": "function_call",
        "id": "call_first",
        "call_id": "call_first",
        "name": "first_tool",
        "arguments": '{"value":1}',
        "status": "completed",
    }
    second_done = {
        "type": "function_call",
        "id": "call_second",
        "call_id": "call_second",
        "name": "second_tool",
        "arguments": '{"value":2}',
        "status": "completed",
    }
    payloads = _normalised_live_payloads(tmp_path, monkeypatch, [
        {
            "type": "response.created",
            "response": {"id": "resp_interleaved", "output": []},
        },
        {
            "type": "response.output_item.added",
            "output_index": 3,
            "item": {**first_done, "arguments": "", "status": "in_progress"},
        },
        {
            "type": "response.output_item.added",
            "output_index": 7,
            "item": {**second_done, "arguments": "", "status": "in_progress"},
        },
        {
            "type": "response.function_call_arguments.done",
            "output_index": 7,
            "item_id": "call_second",
            "arguments": second_done["arguments"],
        },
        {
            "type": "response.function_call_arguments.done",
            "output_index": 3,
            "item_id": "call_first",
            "arguments": first_done["arguments"],
        },
        {
            "type": "response.output_item.done",
            "output_index": 7,
            "item": second_done,
        },
        {
            "type": "response.output_item.done",
            "output_index": 3,
            "item": first_done,
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_interleaved",
                "status": "completed",
                "output": [first_done, second_done],
            },
        },
    ])

    adapter = TypeAdapter(ResponsesServerEvent)
    assert [adapter.validate_python(payload).type for payload in payloads] == [
        payload["type"] for payload in payloads
    ]
    arguments_done = [
        payload for payload in payloads
        if payload["type"] == "response.function_call_arguments.done"
    ]
    assert [
        (payload["item_id"], payload["name"], payload["output_index"])
        for payload in arguments_done
    ] == [
        ("call_second", "second_tool", 1),
        ("call_first", "first_tool", 0),
    ]

    added = [
        payload for payload in payloads
        if payload["type"] == "response.output_item.added"
    ]
    done = [
        payload for payload in payloads
        if payload["type"] == "response.output_item.done"
    ]
    completed = payloads[-1]["response"]["output"]
    assert Counter(
        (payload["output_index"], payload["item"]["id"])
        for payload in added
    ) == Counter(
        (payload["output_index"], payload["item"]["id"])
        for payload in done
    ) == Counter(
        (output_index, item["id"])
        for output_index, item in enumerate(completed)
    )


@pytest.mark.parametrize("empty_text", [None, ""])
def test_live_empty_message_lifecycle_is_atomic(
    tmp_path, monkeypatch, empty_text,
):
    added_item = {
        "id": "msg_empty",
        "type": "message",
        "role": "assistant",
        "status": "in_progress",
        "content": [],
    }
    done_item = {
        **added_item,
        "status": "completed",
        "content": [{
            "type": "output_text",
            "text": empty_text,
            "annotations": [],
        }],
    }
    payloads = _normalised_live_payloads(tmp_path, monkeypatch, [
        {
            "type": "response.created",
            "response": {"id": "resp_empty", "output": []},
        },
        {
            "type": "response.output_item.added",
            "output_index": 4,
            "item": added_item,
        },
        {
            "type": "response.output_item.done",
            "output_index": 4,
            "item": done_item,
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_empty",
                "status": "completed",
                "output": [done_item],
            },
        },
    ])

    adapter = TypeAdapter(ResponsesServerEvent)
    assert [adapter.validate_python(payload).type for payload in payloads] == [
        payload["type"] for payload in payloads
    ]
    added = [
        payload for payload in payloads
        if payload["type"] == "response.output_item.added"
    ]
    done = [
        payload for payload in payloads
        if payload["type"] == "response.output_item.done"
    ]
    completed_output = payloads[-1]["response"]["output"]

    assert bool(added) == bool(done)
    if added:
        assert len(added) == len(done) == len(completed_output) == 1
        assert (
            added[0]["output_index"], added[0]["item"]["id"]
        ) == (
            done[0]["output_index"], done[0]["item"]["id"]
        ) == (0, completed_output[0]["id"])
        assert done[0]["item"]["content"][0]["text"] == ""
        assert completed_output[0]["content"][0]["text"] == ""
    else:
        assert completed_output == []


def test_deepseek_fallback_rejects_previous_response_id_before_http(
    tmp_path, monkeypatch,
):
    requests = _install_http_stub(monkeypatch, "deepseek")
    backend = Backend(_config(tmp_path, "deepseek"))

    async def consume():
        return [
            event
            async for event in backend.stream_responses(json.dumps({
                "input": "Continue.",
                "stream": True,
                "previous_response_id": "resp_previous",
            }))
        ]

    with pytest.raises(
        ValueError,
        match=r"previous_response_id.*Chat Completions fallback",
    ):
        asyncio.run(consume())

    assert requests == []


@pytest.mark.parametrize("tool_type", UNSUPPORTED_FALLBACK_TOOL_TYPES)
def test_deepseek_fallback_rejects_responses_only_tools_before_http(
    tmp_path, monkeypatch, tool_type,
):
    requests = _install_http_stub(monkeypatch, "deepseek")
    backend = Backend(_config(tmp_path, "deepseek"))

    async def consume():
        return [
            event
            async for event in backend.stream_responses(json.dumps({
                "input": "Use the tool.",
                "stream": True,
                "tools": [{"type": tool_type}],
            }))
        ]

    with pytest.raises(
        ValueError,
        match=rf"tool type\(s\): {tool_type}.*Chat Completions fallback",
    ):
        asyncio.run(consume())

    assert requests == []


@pytest.mark.parametrize("stream", [False, True])
def test_deepseek_route_rejects_omitted_input_before_dispatch(
    tmp_path, monkeypatch, stream,
):
    server = ProxyServer(_config(tmp_path, "deepseek"))
    complete_call = AsyncMock()
    stream_call = AsyncMock()
    monkeypatch.setattr(server.backend, "complete_responses", complete_call)
    monkeypatch.setattr(server.backend, "stream_responses", stream_call)
    body = {"stream": stream} if stream else {}

    async def request_invalid():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post("/v1/responses", json=body)

    response = asyncio.run(request_invalid())

    assert response.status_code == 400
    error = response.json()["error"]
    assert "input (omitted)" in error["message"]
    assert "Chat Completions fallback" in error["message"]
    complete_call.assert_not_awaited()
    stream_call.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("item_type", [
    "item_reference", "reasoning", "web_search_call",
])
def test_deepseek_route_rejects_lossy_input_items_before_dispatch(
    tmp_path, monkeypatch, stream, item_type,
):
    server = ProxyServer(_config(tmp_path, "deepseek"))
    complete_call = AsyncMock()
    stream_call = AsyncMock()
    monkeypatch.setattr(server.backend, "complete_responses", complete_call)
    monkeypatch.setattr(server.backend, "stream_responses", stream_call)
    body = {"input": [{"type": item_type, "id": "item_test"}]}
    if stream:
        body["stream"] = True

    async def request_invalid():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post("/v1/responses", json=body)

    response = asyncio.run(request_invalid())

    assert response.status_code == 400
    error = response.json()["error"]
    assert f"input item type(s): {item_type}" in error["message"]
    assert "Chat Completions fallback" in error["message"]
    complete_call.assert_not_awaited()
    stream_call.assert_not_awaited()


def test_deepseek_fallback_accepts_function_call_history():
    params = llm_module._build_responses_kwargs(
        model="deepseek/deepseek-chat",
        api_key="test-key",
        input=[
            {
                "type": "function_call",
                "call_id": "call_weather",
                "name": "lookup_weather",
                "arguments": '{"city":"Taipei"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_weather",
                "output": "sunny",
            },
        ],
    )

    assert params["input"][0]["type"] == "function_call"
    assert params["input"][1]["type"] == "function_call_output"


@pytest.mark.parametrize(
    "tool_choice",
    [
        "auto",
        "none",
        "required",
        {"type": "function", "name": "lookup_weather"},
        {"type": "custom", "name": "grammar"},
    ],
)
def test_deepseek_fallback_accepts_chat_compatible_tool_choices(tool_choice):
    params = llm_module._build_responses_kwargs(
        model="deepseek/deepseek-chat",
        api_key="test-key",
        input="Use a tool.",
        tools=[_function_tool()],
        tool_choice=tool_choice,
    )

    assert params["tool_choice"] == tool_choice


def test_deepseek_fallback_accepts_text_format_without_verbosity():
    text = {"format": {"type": "text"}}

    params = llm_module._build_responses_kwargs(
        model="deepseek/deepseek-chat",
        api_key="test-key",
        input="hello",
        text=text,
    )

    assert params["text"] == text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("extension", "message_fragment"),
    [
        ({"previous_response_id": "resp_previous"}, "previous_response_id"),
        ({"metadata": {"tenant": "acme"}}, "metadata"),
        ({"store": True}, "store"),
        ({"truncation": "auto"}, "truncation"),
        (
            {
                "context_management": [{
                    "type": "compaction",
                    "compact_threshold": 1,
                }],
            },
            "context_management",
        ),
        ({"prompt_cache_key": "cache-key"}, "prompt_cache_key"),
        (
            {"prompt_cache_retention": "24h"},
            "prompt_cache_retention",
        ),
        (
            {"prompt_cache_options": {"mode": "explicit", "ttl": "24h"}},
            "prompt_cache_options",
        ),
        (
            {
                "text": {
                    "format": {"type": "text"},
                    "verbosity": "low",
                },
            },
            "text.verbosity",
        ),
        (
            {"reasoning": {"effort": "high", "summary": "auto"}},
            "reasoning.summary",
        ),
        (
            {
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": "auto",
                    "tools": [{"type": "function", "name": "lookup_weather"}],
                },
            },
            "tool_choice.allowed_tools",
        ),
        *[
            (
                {"tool_choice": {"type": tool_type}},
                f"tool_choice.{tool_type}",
            )
            for tool_type in (
                "web_search_preview", "mcp", "apply_patch", "shell",
            )
        ],
        (
            {
                "input": [{
                    "role": "user",
                    "content": [{
                        "type": "input_image",
                        "file_id": "file_image",
                    }],
                }],
            },
            "input_image.file_id",
        ),
        (
            {
                "input": [{
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": "cache me",
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }],
                }],
            },
            "input_text.prompt_cache_breakpoint",
        ),
        (
            {
                "input": [{
                    "role": "user",
                    "content": [{
                        "type": "input_file",
                        "file_data": "data:text/plain;base64,aGk=",
                        "filename": "named.txt",
                    }],
                }],
            },
            "input_file.filename",
        ),
        (
            {
                "input": [{
                    "role": "user",
                    "content": [{
                        "type": "input_file",
                        "file_data": "data:text/plain;base64,aGk=",
                        "detail": "high",
                    }],
                }],
            },
            "input_file.detail",
        ),
        (
            {
                "tools": [{
                    **_function_tool(),
                    "output_schema": {"type": "object"},
                }],
            },
            "function.output_schema",
        ),
        (
            {
                "tools": [{
                    "type": "custom",
                    "name": "patch",
                    "defer_loading": True,
                }],
            },
            "custom.defer_loading",
        ),
        *[
            (
                {
                    "input": [{
                        "type": output_type,
                        "call_id": "call_1",
                        "output": [
                            {"type": "input_text", "text": "answer"},
                            {
                                "type": media_type,
                                **(
                                    {"image_url": "https://example.test/image.png"}
                                    if media_type == "input_image"
                                    else {
                                        "file_data": "data:text/plain;base64,aGk=",
                                    }
                                ),
                            },
                        ],
                    }],
                },
                f"{output_type}.output.{media_type}",
            )
            for output_type in (
                "function_call_output", "custom_tool_call_output",
            )
            for media_type in ("input_image", "input_file")
        ],
        (
            {
                "input": [{
                    "role": "assistant",
                    "content": "working",
                    "phase": "commentary",
                }],
            },
            "message.phase",
        ),
        *[
            ({"tools": [{"type": tool_type}]}, tool_type)
            for tool_type in UNSUPPORTED_FALLBACK_TOOL_TYPES
        ],
        (
            {"tools": {"type": "function"}},
            "tools (must be an array)",
        ),
        (
            {"tools": [{"name": "missing_type"}]},
            "tool type(s): invalid",
        ),
        (
            {"tools": ["function"]},
            "tool type(s): invalid",
        ),
    ],
)
def test_deepseek_route_rejects_fallback_only_features_before_dispatch(
    tmp_path, monkeypatch, stream, extension, message_fragment,
):
    server = ProxyServer(_config(tmp_path, "deepseek"))
    complete_call = AsyncMock()
    stream_call = AsyncMock()
    monkeypatch.setattr(server.backend, "complete_responses", complete_call)
    monkeypatch.setattr(server.backend, "stream_responses", stream_call)

    body = {"input": "hello", **extension}
    if stream:
        body["stream"] = True

    async def request_invalid():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post("/v1/responses", json=body)

    response = asyncio.run(request_invalid())

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert not response.headers["content-type"].startswith("text/event-stream")
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert message_fragment in error["message"]
    assert "Chat Completions fallback" in error["message"]
    complete_call.assert_not_awaited()
    stream_call.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
def test_responses_route_rejects_unknown_top_level_field_before_dispatch(
    tmp_path, monkeypatch, stream,
):
    server = ProxyServer(_config(tmp_path, "deepseek"))
    complete_call = AsyncMock()
    stream_call = AsyncMock()
    monkeypatch.setattr(server.backend, "complete_responses", complete_call)
    monkeypatch.setattr(server.backend, "stream_responses", stream_call)
    body = {"input": "hello", "max_output_token": 1}
    if stream:
        body["stream"] = True

    async def request_invalid():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post("/v1/responses", json=body)

    response = asyncio.run(request_invalid())

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["message"] == (
        "unsupported Responses parameter(s): max_output_token"
    )
    complete_call.assert_not_awaited()
    stream_call.assert_not_awaited()


def test_responses_request_validation_allows_schema_compat_and_transport_fields():
    allowed_fields = {
        **{key: None for key in llm_module._RESPONSES_OPTIONAL_PARAMS},
        "model": "gpt-4o",
        "input": "hello",
        "max_output_tokens": 10,
        "stream": False,
        "max_tokens": 10,
        "response_format": {"type": "json_object"},
        "extra_body": {},
        "extra_headers": {},
        "extra_query": {},
        "timeout": 1,
        "allowed_openai_params": [],
    }

    for key, value in allowed_fields.items():
        llm_module._validate_responses_request({key: value})


def test_openai_chat_completions_prefix_is_classified_as_fallback():
    for provider in (None, "openai"):
        assert llm_module._responses_provider_uses_chat_fallback(
            "openai/chat_completions/gpt-4o", provider=provider,
        ) == (True, "openai")


def test_explicit_openai_provider_preserves_forced_chat_prefix_on_wire(
    monkeypatch,
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
                "id": "chatcmpl-forced",
                "object": "chat.completion",
                "created": 1_725_000_000,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    asyncio.run(llm_module.llm_response(
        model="openai/chat_completions/gpt-4o",
        provider="openai",
        api_key="test-key",
        input="hello",
    ))

    assert len(requests) == 1
    assert requests[0].url.path == "/v1/chat/completions"
    assert json.loads(requests[0].content)["model"] == "gpt-4o"


@pytest.mark.parametrize("provider", [None, "openai"])
def test_forced_chat_completions_extra_query_uses_openai_client_on_wire(
    monkeypatch, provider,
):
    from openai import AsyncOpenAI

    requests = []
    owned_clients = []
    attach_client = llm_module._attach_responses_query_client

    def record_client(*args, **kwargs):
        client = attach_client(*args, **kwargs)
        owned_clients.append(client)
        return client

    async def fake_send(client, request, **kwargs):
        for hook in client._event_hooks.get("request", ()):
            await hook(request)
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "chatcmpl-extra-query",
                "object": "chat.completion",
                "created": 1_725_000_000,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    monkeypatch.setattr(
        llm_module, "_attach_responses_query_client", record_client,
    )

    asyncio.run(llm_module.llm_response(
        model="openai/chat_completions/gpt-4o",
        provider=provider,
        api_key="test-key",
        input="hello",
        extra_query={"trace": "forced-chat"},
    ))

    assert len(owned_clients) == 1
    assert isinstance(owned_clients[0], AsyncOpenAI)
    assert [str(request.url) for request in requests] == [
        "https://api.openai.com/v1/chat/completions?trace=forced-chat"
    ]


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
def test_forced_chat_completions_uses_custom_base_url_on_wire(
    monkeypatch, base_url, expected_url,
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
                "id": "chatcmpl-custom-base",
                "object": "chat.completion",
                "created": 1_725_000_000,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    asyncio.run(llm_module.llm_response(
        model="openai/chat_completions/gpt-4o",
        provider="openai",
        api_key="test-key",
        base_url=base_url,
        input="hello",
    ))

    assert [str(request.url) for request in requests] == [expected_url]


@pytest.mark.parametrize("stream", [False, True])
def test_forced_chat_completions_route_rejects_responses_state_before_dispatch(
    tmp_path, monkeypatch, stream,
):
    server = ProxyServer(_forced_chat_completions_config(tmp_path))
    complete_call = AsyncMock()
    stream_call = AsyncMock()
    monkeypatch.setattr(server.backend, "complete_responses", complete_call)
    monkeypatch.setattr(server.backend, "stream_responses", stream_call)
    body = {
        "input": "Continue.",
        "previous_response_id": "resp_previous",
    }
    if stream:
        body["stream"] = True

    async def request_invalid():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.post("/v1/responses", json=body)

    response = asyncio.run(request_invalid())

    assert response.status_code == 400
    assert "previous_response_id" in response.json()["error"]["message"]
    assert "Chat Completions fallback" in response.json()["error"]["message"]
    complete_call.assert_not_awaited()
    stream_call.assert_not_awaited()


def test_forced_chat_completions_backend_normalizes_bridge_output(
    tmp_path, monkeypatch,
):
    backend = Backend(_forced_chat_completions_config(tmp_path))
    bridge_response = {
        "id": "resp_bridge",
        "object": "response",
        "status": "completed",
        "model": "gpt-4o",
        "output": [{
            "id": "msg_empty",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": None,
                "annotations": [],
            }],
        }],
    }
    response_call = AsyncMock(return_value=bridge_response)
    monkeypatch.setattr(backend_module, "llm_response", response_call)
    monkeypatch.setattr(backend_module, "save_request_log", Mock())

    result, error = asyncio.run(
        backend.complete_responses(json.dumps({"input": "hello"}))
    )

    assert error is None
    assert result["output"] == []
    response_call.assert_awaited_once()
