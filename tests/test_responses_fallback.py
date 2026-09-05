"""Regression coverage for the native-only Responses API contract."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

import turbo_proxy.proxy.backend as backend_module
import turbo_proxy.utils.llm as llm_module
from turbo_proxy.proxy.backend import Backend
from turbo_proxy.utils import Config


NON_NATIVE_MODELS = (
    ("deepseek/deepseek-chat", "deepseek"),
    ("anthropic/claude-3-5-sonnet-latest", "anthropic"),
    ("gemini/gemini-2.5-flash", "gemini"),
    ("openai/chat_completions/gpt-4o", "openai"),
    ("openai/openai/chat_completions/gpt-4o", "openai"),
)


def _config(tmp_path, models):
    config_path = tmp_path / "turbo-proxy.yaml"
    config_path.write_text(yaml.safe_dump({
        "backend": {"models": models},
        "log_dir": str(tmp_path / "logs"),
    }))
    return Config(str(config_path))


def _backend(tmp_path, model, provider=None):
    model_config = {"name": model, "api_key": "test-key"}
    if provider is not None:
        model_config["provider"] = provider
    return Backend(_config(tmp_path, [model_config]))


@pytest.mark.parametrize("model, provider", NON_NATIVE_MODELS)
def test_build_responses_kwargs_rejects_non_native_provider(model, provider):
    with pytest.raises(ValueError, match="native Responses provider") as exc_info:
        llm_module._build_responses_kwargs(
            model=model,
            input="hello",
            api_key="test-key",
            provider=provider,
        )

    assert model in str(exc_info.value)
    assert provider in str(exc_info.value)


@pytest.mark.parametrize("prefix", ["", "bedrock_mantle/"])
@pytest.mark.parametrize("model_id, native", [
    ("openai.gpt-oss-120b", True),
    ("openai.gpt-oss-safeguard-120b", False),
])
def test_native_responses_capability_uses_normalized_model(
    monkeypatch, prefix, model_id, native,
):
    route_resolver = Mock(side_effect=AssertionError("Dynamic provider resolution is forbidden"))
    monkeypatch.setattr(llm_module.litellm, "get_llm_provider", route_resolver)
    model = prefix + model_id
    params = {
        "model": model,
        "provider": "bedrock_mantle",
        "base_url": "https://gateway.example",
        "api_key": "test-key",
        "input": "hello",
    }

    assert llm_module._responses_provider_uses_chat_fallback(
        model, params["provider"], params["base_url"],
    ) == (not native, "bedrock_mantle")
    if native:
        built = llm_module._build_responses_kwargs(**params)
        assert built["model"] == model
        assert built["custom_llm_provider"] == "bedrock_mantle"
    else:
        with pytest.raises(ValueError, match="native Responses provider"):
            llm_module._build_responses_kwargs(**params)

    route_resolver.assert_not_called()


@pytest.mark.parametrize("prefix", ["", "github_copilot/"])
@pytest.mark.parametrize("operation", ["build", "complete", "stream"])
def test_responses_precheck_rejects_copilot_without_authentication(
    monkeypatch, prefix, operation,
):
    route_resolver = Mock(side_effect=AssertionError("Dynamic provider resolution is forbidden"))
    response_call = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "get_llm_provider", route_resolver)
    monkeypatch.setattr(llm_module.litellm, "aresponses", response_call)
    model = prefix + "claude-sonnet-4"
    params = {
        "model": model,
        "provider": "github_copilot",
        "base_url": "https://gateway.example",
        "api_key": "test-key",
        "input": "hello",
    }

    with pytest.raises(ValueError, match="native Responses provider") as exc_info:
        if operation == "build":
            llm_module._build_responses_kwargs(**params)
        elif operation == "stream":
            asyncio.run(llm_module.llm_stream_response(**params))
        else:
            asyncio.run(llm_module.llm_response(**params))

    assert model in str(exc_info.value)
    assert "github_copilot" in str(exc_info.value)
    route_resolver.assert_not_called()
    response_call.assert_not_awaited()


@pytest.mark.parametrize("prefix", ["", "github_copilot/"])
@pytest.mark.parametrize("operation", ["validate", "complete", "stream"])
def test_backend_responses_precheck_rejects_copilot_without_authentication(
    tmp_path, monkeypatch, prefix, operation,
):
    route_resolver = Mock(side_effect=AssertionError("Dynamic provider resolution is forbidden"))
    response_call = AsyncMock()
    stream_call = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "get_llm_provider", route_resolver)
    monkeypatch.setattr(backend_module, "llm_response", response_call)
    monkeypatch.setattr(backend_module, "llm_stream_response", stream_call)
    model = prefix + "claude-sonnet-4"
    backend = Backend(_config(tmp_path, [{
        "name": model,
        "provider": "github_copilot",
        "base_url": "https://gateway.example",
        "api_key": "test-key",
    }]))

    if operation == "complete":
        result, error = asyncio.run(backend.complete_responses('{"input":"hello"}'))
        assert result is None
        assert error is not None
    else:
        with pytest.raises(ValueError, match="native Responses provider") as exc_info:
            if operation == "stream":
                asyncio.run(_consume_backend_stream(backend))
            else:
                backend.validate_responses_body({"input": "hello"})
        error = str(exc_info.value)

    assert "native Responses provider" in error
    assert model in error
    assert "github_copilot" in error
    route_resolver.assert_not_called()
    response_call.assert_not_awaited()
    stream_call.assert_not_awaited()


@pytest.mark.parametrize("prefix", ["", "bedrock_mantle/"])
def test_backend_accepts_prefixed_native_responses_model(tmp_path, prefix):
    backend = Backend(_config(tmp_path, [{
        "name": prefix + "openai.gpt-oss-120b",
        "provider": "bedrock_mantle",
        "base_url": "https://gateway.example",
        "api_key": "test-key",
    }]))

    backend.validate_responses_body({"input": "hello"})


def test_build_responses_kwargs_rejects_nested_chat_prefix_without_provider():
    model = "openai/openai/chat_completions/gpt-4o"

    with pytest.raises(ValueError, match="native Responses provider") as exc_info:
        llm_module._build_responses_kwargs(
            model=model,
            input="hello",
            api_key="test-key",
        )

    assert model in str(exc_info.value)
    assert "openai" in str(exc_info.value)


def test_llm_response_rejects_nested_chat_prefix_without_provider(
    monkeypatch,
):
    model = "openai/openai/chat_completions/gpt-4o"
    responses_call = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "aresponses", responses_call)

    with pytest.raises(ValueError, match="native Responses provider") as exc_info:
        asyncio.run(llm_module.llm_response(
            model=model,
            input="hello",
            api_key="test-key",
        ))

    assert model in str(exc_info.value)
    responses_call.assert_not_awaited()


@pytest.mark.parametrize("model, provider", NON_NATIVE_MODELS)
def test_llm_response_rejects_non_native_provider_before_litellm(
    monkeypatch, model, provider,
):
    responses_call = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "aresponses", responses_call)

    with pytest.raises(ValueError, match="native Responses provider") as exc_info:
        asyncio.run(llm_module.llm_response(
            model=model,
            input="hello",
            api_key="test-key",
            provider=provider,
        ))

    assert model in str(exc_info.value)
    assert provider in str(exc_info.value)
    responses_call.assert_not_awaited()


@pytest.mark.parametrize("model, provider", NON_NATIVE_MODELS)
def test_llm_stream_response_rejects_non_native_provider_before_litellm(
    monkeypatch, model, provider,
):
    responses_call = AsyncMock()
    monkeypatch.setattr(llm_module.litellm, "aresponses", responses_call)

    with pytest.raises(ValueError, match="native Responses provider") as exc_info:
        asyncio.run(_consume_stream_wrapper(
            llm_module.llm_stream_response(
                model=model,
                input="hello",
                api_key="test-key",
                provider=provider,
            )
        ))

    assert model in str(exc_info.value)
    assert provider in str(exc_info.value)
    responses_call.assert_not_awaited()


async def _consume_stream_wrapper(stream_or_awaitable):
    stream = await stream_or_awaitable
    return [event async for event in stream]


@pytest.mark.parametrize("provider", ["azure", "openrouter"])
@pytest.mark.parametrize("stream", [False, True])
def test_responses_wrapper_rejects_chat_prefix_with_native_provider(
    monkeypatch, provider, stream,
):
    model = "openai/openai/chat_completions/gpt-4o"
    responses_call = AsyncMock(return_value={"output": []})
    monkeypatch.setattr(llm_module.litellm, "aresponses", responses_call)
    params = {
        "model": model,
        "provider": provider,
        "base_url": "https://gateway.example",
        "api_key": "test-key",
        "input": "hello",
    }

    with pytest.raises(ValueError, match="native Responses provider") as exc_info:
        if stream:
            asyncio.run(llm_module.llm_stream_response(**params))
        else:
            asyncio.run(llm_module.llm_response(**params))

    assert model in str(exc_info.value)
    assert provider in str(exc_info.value)
    responses_call.assert_not_awaited()


@pytest.mark.parametrize("model, provider", NON_NATIVE_MODELS)
def test_backend_complete_responses_rejects_non_native_before_dispatch(
    tmp_path, monkeypatch, model, provider,
):
    response_call = AsyncMock()
    monkeypatch.setattr(backend_module, "llm_response", response_call)
    backend = _backend(tmp_path, model, provider)

    result, error = asyncio.run(
        backend.complete_responses('{"input":"hello"}')
    )

    assert result is None
    assert error is not None
    assert "native Responses provider" in error
    assert model in error
    assert provider in error
    response_call.assert_not_awaited()


@pytest.mark.parametrize("model, provider", NON_NATIVE_MODELS)
def test_backend_stream_responses_rejects_non_native_before_dispatch(
    tmp_path, monkeypatch, model, provider,
):
    response_call = AsyncMock()
    monkeypatch.setattr(backend_module, "llm_stream_response", response_call)
    backend = _backend(tmp_path, model, provider)

    with pytest.raises(ValueError, match="native Responses provider") as exc_info:
        asyncio.run(_consume_backend_stream(backend))

    assert model in str(exc_info.value)
    assert provider in str(exc_info.value)
    response_call.assert_not_awaited()


async def _consume_backend_stream(backend):
    return [
        event
        async for event in backend.stream_responses(
            '{"input":"hello","stream":true}'
        )
    ]


@pytest.mark.parametrize("provider", ["azure", "openrouter"])
@pytest.mark.parametrize("stream", [False, True])
def test_backend_rejects_chat_prefix_with_native_provider_before_dispatch(
    tmp_path, monkeypatch, provider, stream,
):
    model = "openai/openai/chat_completions/gpt-4o"
    response_call = AsyncMock(return_value={"output": []})
    stream_call = AsyncMock()
    monkeypatch.setattr(backend_module, "llm_response", response_call)
    monkeypatch.setattr(backend_module, "llm_stream_response", stream_call)
    backend = Backend(_config(tmp_path, [{
        "name": model,
        "provider": provider,
        "base_url": "https://gateway.example",
        "api_key": "test-key",
    }]))

    if stream:
        with pytest.raises(ValueError, match="native Responses provider") as exc_info:
            asyncio.run(_consume_backend_stream(backend))
        error = str(exc_info.value)
    else:
        result, error = asyncio.run(
            backend.complete_responses('{"input":"hello"}')
        )
        assert result is None
        assert error is not None

    assert "native Responses provider" in error
    assert model in error
    assert provider in error
    response_call.assert_not_awaited()
    stream_call.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
def test_backend_verifier_rejects_mixed_native_candidates_before_dispatch(
    tmp_path, monkeypatch, stream,
):
    response_call = AsyncMock()
    stream_call = AsyncMock()
    monkeypatch.setattr(backend_module, "llm_response", response_call)
    monkeypatch.setattr(backend_module, "llm_stream_response", stream_call)
    backend = Backend(_config(tmp_path, [
        {"name": "openai/gpt-4o", "api_key": "openai-key"},
        {"name": "deepseek/deepseek-chat", "api_key": "deepseek-key"},
    ]))
    backend.verifier = object()

    if stream:
        with pytest.raises(ValueError, match="native Responses provider") as exc_info:
            asyncio.run(_consume_backend_stream(backend))
        message = str(exc_info.value)
    else:
        result, error = asyncio.run(
            backend.complete_responses('{"input":"hello"}')
        )
        assert result is None
        assert error is not None
        message = error

    assert "native Responses provider" in message
    assert "deepseek/deepseek-chat" in message
    assert "provider='deepseek'" in message
    response_call.assert_not_awaited()
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
