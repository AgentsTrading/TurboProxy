"""
LLM completion wrapper using litellm for multi-provider routing and format
conversion. litellm handles model prefix routing (gemini/, openai/, anthropic/),
provider-specific API formatting, and cross-provider tool call compatibility.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import anyio
import httpx
import litellm

from .config import (
    append_url_path,
    resolve_base_url,
    resolve_litellm_route,
    resolve_model_provider,
    split_model_name,
    validate_litellm_api_key,
    validate_litellm_endpoint,
)
from .verifier_client import (
    _append_base_query,
    _model_scoped_openai_headers,
    _openai_endpoint_parts,
    _url_origin,
)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True

# Disable litellm's Anthropic/Gemini context-caching transform. Otherwise litellm
# creates explicit Google CachedContent resources and then references them in the
# same generateContent call that still carries system_instruction/tools, which the
# Gemini API rejects: "CachedContent can not be used with GenerateContent request
# setting system_instruction, tools or tool_config."
litellm.disable_anthropic_gemini_context_caching_transform = True

_PROTECTED_LITELLM_KWARGS = {
    "api_base": "base_url",
    "custom_llm_provider": "provider",
    "azure": "provider",
    "deployment_id": "model/provider",
    "headers": "api_key",
    "extra_headers": "api_key",
    "extra_body": "provider-specific parameters",
    "client": "top-level model/provider/base_url/api_key parameters",
    "fallbacks": "top-level model/provider/base_url parameters",
    "model_list": "top-level model/provider/base_url parameters",
    "litellm_params": "top-level model/provider/base_url/api_key parameters",
    "use_litellm_proxy": "top-level model/provider/base_url parameters",
    "model_alias_map": "top-level model/provider parameters",
    "api_version": "provider configuration",
    "azure_endpoint": "provider configuration",
    "azure_deployment": "model/provider",
    "azure_ad_token": "provider credentials",
    "azure_ad_token_provider": "provider credentials",
    "tenant_id": "provider credentials",
    "client_id": "provider credentials",
    "client_secret": "provider credentials",
    "azure_username": "provider credentials",
    "azure_password": "provider credentials",
    "azure_scope": "provider credentials",
    "vertex_credentials": "provider credentials",
    "vertex_project": "provider configuration",
    "vertex_location": "provider configuration",
    "vertex_ai_credentials": "provider credentials",
    "vertex_ai_project": "provider configuration",
    "vertex_ai_location": "provider configuration",
    "aws_access_key_id": "provider credentials",
    "aws_secret_access_key": "provider credentials",
    "aws_session_token": "provider credentials",
    "aws_session_name": "provider credentials",
    "aws_profile_name": "provider credentials",
    "aws_role_name": "provider credentials",
    "aws_web_identity_token": "provider credentials",
    "aws_sts_endpoint": "provider configuration",
    "aws_external_id": "provider credentials",
    "aws_region_name": "provider configuration",
    "aws_bedrock_runtime_endpoint": "provider configuration",
    "aws_bedrock_project_id": "provider configuration",
}

_CHAT_COMPLETIONS_PATH = "chat/completions"
_URL_JOINING_PROVIDERS = {"deepseek"}
_QUERY_CLIENT_PROVIDERS = {"anthropic", "gemini", "vertex_ai"}
_PROVIDER_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/beta",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com",
}


class _OpenAIQueryBaseURL(str):
    """A model-level OpenAI API root and its post-path query, if any."""

    raw_query: bytes

    def __new__(cls, value: str, raw_query: bytes) -> "_OpenAIQueryBaseURL":
        instance = super().__new__(cls, value)
        instance.raw_query = raw_query
        return instance


class _LiteLLMQueryBaseURL(str):
    """A provider base whose query is applied after LiteLLM builds its path."""

    raw_query: bytes

    def __new__(cls, value: str, raw_query: bytes) -> "_LiteLLMQueryBaseURL":
        instance = super().__new__(cls, value)
        instance.raw_query = raw_query
        return instance


class _EndpointAwareURL(str):
    """URL string that lets DeepSeek's 1.98.0 suffix check see its path.

    DeepSeek inherits the OpenAI transformation but overrides URL detection
    with ``api_base.endswith('/chat/completions')``.  A query string makes a
    normal endpoint URL fail that check.  Keeping the URL as a ``str``
    subclass preserves all HTTPX behavior while making the suffix check use
    the parsed path as well.
    """

    def endswith(self, suffix: Any, start: int = 0, end: Optional[int] = None) -> bool:
        if super().endswith(suffix, start, len(self) if end is None else end):
            return True
        path = urlsplit(self).path
        return path.endswith(suffix, start, len(path) if end is None else end)


class _OwnedClientStream:
    """An async stream that deterministically closes its owned SDK client."""

    def __init__(self, stream: Any, client: Any):
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._client = client
        self._source_closed = False
        self._client_closed = False
        self._closed = False
        self._close_task: Optional[asyncio.Task[None]] = None

    def __aiter__(self) -> "_OwnedClientStream":
        return self

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._iterator.__anext__()
        except BaseException as exc:
            try:
                await self.aclose()
            except BaseException as close_exc:
                if close_exc is exc:
                    raise
                if isinstance(exc, StopAsyncIteration):
                    raise close_exc
                raise exc from close_exc
            raise

    async def _close_resources(self) -> None:
        source_error: Optional[BaseException] = None
        close_stream = getattr(self._iterator, "aclose", None)
        if close_stream is None and self._iterator is not self._stream:
            close_stream = getattr(self._stream, "aclose", None)
        if not self._source_closed:
            try:
                if close_stream is not None:
                    await close_stream()
            except BaseException as exc:
                source_error = exc
            else:
                self._source_closed = True

        client_error: Optional[BaseException] = None
        if not self._client_closed:
            try:
                await self._client.close()
            except BaseException as exc:
                client_error = exc
            else:
                self._client_closed = True

        self._closed = self._source_closed and self._client_closed
        if source_error is not None:
            if client_error is not None:
                raise source_error from client_error
            raise source_error
        if client_error is not None:
            raise client_error

    async def aclose(self) -> None:
        if self._closed:
            return
        # Starlette/AnyIO can keep a disconnected request's cancel scope
        # active across every cleanup await. A shared task prevents concurrent
        # closes, the shield lets cleanup finish, and a failed task is replaced
        # on the next call so partially closed resources remain retryable.
        with anyio.CancelScope(shield=True):
            if self._close_task is None or (
                self._close_task.done() and not self._closed
            ):
                self._close_task = asyncio.create_task(
                    self._close_resources()
                )
            await _await_cleanup_task(self._close_task)


async def _await_cleanup_task(task: "asyncio.Task[None]") -> None:
    """Wait for cleanup while preserving raw asyncio cancellation as primary."""
    cancellation: Optional[asyncio.CancelledError] = None
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:
            if task.done():
                primary = cancellation or exc
                try:
                    task.result()
                except BaseException as cleanup_exc:
                    if cleanup_exc is primary:
                        raise
                    raise primary from cleanup_exc
                raise primary
            if cancellation is None:
                cancellation = exc
        except BaseException as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation


async def _close_client(client: Any) -> None:
    """Close an owned SDK client despite AnyIO or asyncio cancellation."""
    with anyio.CancelScope(shield=True):
        await _await_cleanup_task(asyncio.create_task(client.close()))


async def _close_client_after_error(
    client: Any, primary: BaseException,
) -> None:
    """Close a client without replacing the request's primary exception."""
    try:
        await _close_client(client)
    except BaseException as cleanup_exc:
        if cleanup_exc is primary:
            raise
        raise primary from cleanup_exc


def _query_safe_provider_base_url(
    base_url: str,
    provider: str,
    model: str,
) -> str:
    """Return the base shape LiteLLM expects, retaining its raw query aside."""
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    model_id = split_model_name(model)[0]

    if provider == "vertex_ai" and path == "/v1":
        # LiteLLM 1.98 only grafts the canonical Vertex resource route onto a
        # bare host; retaining /v1 produces /v1:generateContent (or its
        # streaming equivalent) without the project/location/model path.
        path = ""

    if provider == "anthropic":
        suffixes = ("/v1/messages",)
    elif provider == "gemini":
        suffixes = (
            f"/models/{model_id}:generateContent",
            f"/models/{model_id}:streamGenerateContent",
        )
    else:
        suffixes = (":generateContent", ":streamGenerateContent")

    for suffix in suffixes:
        if path.endswith(suffix):
            path = path[:-len(suffix)].rstrip("/")
            break

    clean_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    raw_query = httpx.URL(base_url).query
    if raw_query:
        return _LiteLLMQueryBaseURL(clean_url, raw_query)
    return clean_url


def _prepare_litellm_base_url(
    base_url: str,
    provider: Optional[str],
    model: str,
) -> str:
    """Prepare provider-specific URL joining without moving query parameters.

    LiteLLM 1.98.0 sends OpenAI chat through ``AsyncOpenAI``, whose ``base_url``
    must be the API root. DeepSeek instead appends its endpoint before using
    LiteLLM's HTTP handler. Fragments are dropped because HTTP never sends them.
    """
    if provider == "openai":
        api_root, raw_query = _openai_endpoint_parts(base_url)
        return _OpenAIQueryBaseURL(api_root, raw_query)

    if provider in _QUERY_CLIENT_PROVIDERS:
        return _query_safe_provider_base_url(base_url, provider, model)

    if provider not in _URL_JOINING_PROVIDERS:
        return base_url

    parsed = urlsplit(base_url)
    if not parsed.query and not parsed.fragment:
        # DeepSeek checks the literal suffix before deciding whether to append
        # /chat/completions. A trailing slash on a complete resource URL makes
        # that check fail and duplicates the endpoint.
        return urlunsplit((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip("/"),
            "",
            "",
        ))

    path = parsed.path.rstrip("/")
    if path == _CHAT_COMPLETIONS_PATH or path.endswith(
        f"/{_CHAT_COMPLETIONS_PATH}"
    ):
        normalized = urlunsplit(
            (parsed.scheme, parsed.netloc, path, parsed.query, "")
        )
    else:
        normalized = append_url_path(base_url, _CHAT_COMPLETIONS_PATH)

    if provider == "deepseek":
        return _EndpointAwareURL(normalized)
    return normalized


def _default_vertex_base_url(model: str) -> str:
    """Return Vertex's official host, bypassing ambient endpoint overrides."""
    from litellm.llms.vertex_ai.common_utils import get_vertex_base_url
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

    configured_location = (
        getattr(litellm, "vertex_location", None)
        or os.environ.get("VERTEXAI_LOCATION")
        or os.environ.get("VERTEX_LOCATION")
    )
    location = VertexBase().get_vertex_region(configured_location, model)
    return get_vertex_base_url(location)


def _attach_openai_query_client(params: Dict[str, Any]) -> Optional[Any]:
    """Attach an isolated, owned client for a model-level OpenAI endpoint."""
    base_url = params.get("base_url")
    if not isinstance(base_url, _OpenAIQueryBaseURL):
        return None

    from openai import AsyncOpenAI, DefaultAsyncHttpxClient

    raw_query = base_url.raw_query
    origin = _url_origin(str(base_url))
    api_key = params.get("api_key") or "EMPTY"
    client_kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "base_url": str(base_url),
        "default_headers": _model_scoped_openai_headers(
            str(base_url), api_key
        ),
    }
    if raw_query:
        async def append_base_query(request: httpx.Request) -> None:
            _append_base_query(request, raw_query, origin)

        client_kwargs["http_client"] = DefaultAsyncHttpxClient(
            event_hooks={"request": [append_base_query]},
            follow_redirects=True,
        )
    client = AsyncOpenAI(**client_kwargs)
    params["base_url"] = str(base_url)
    params["client"] = client
    return client


def _attach_litellm_query_client(params: Dict[str, Any]) -> Optional[Any]:
    """Attach LiteLLM's HTTP handler when a provider base carries a query."""
    base_url = params.get("base_url")
    if not isinstance(base_url, _LiteLLMQueryBaseURL):
        return None

    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    raw_query = base_url.raw_query
    origin = _url_origin(str(base_url))

    async def append_base_query(request: httpx.Request) -> None:
        _append_base_query(request, raw_query, origin)

    client = AsyncHTTPHandler(event_hooks={"request": [append_base_query]})
    params["base_url"] = str(base_url)
    params["client"] = client
    return client


def _validate_extra_kwargs(kwargs: Dict[str, Any]) -> None:
    if "stream" in kwargs:
        raise ValueError(
            "stream cannot be passed through kwargs; choose "
            "llm_completion or llm_stream_completion explicitly"
        )
    for key, supported_name in _PROTECTED_LITELLM_KWARGS.items():
        if key in kwargs:
            raise ValueError(
                f"{key} cannot be passed through kwargs; use {supported_name} "
                "so routing and credential isolation can be enforced"
            )
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise ValueError(
            "unsupported completion parameter(s): "
            f"{unknown}; add supported parameters to the wrapper explicitly"
        )


def _build_kwargs(
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[Any] = None,
    stream: Optional[bool] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    tools: Optional[List[Any]] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[Any] = None,
    seed: Optional[int] = None,
    n: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[Dict[str, int]] = None,
    stream_options: Optional[Any] = None,
    reasoning_effort: Optional[str] = None,
    thinking_budget: Optional[int] = None,
) -> Dict[str, Any]:
    if api_key is not None and not isinstance(api_key, str):
        raise ValueError("api_key must be a string or null")
    if isinstance(api_key, str):
        api_key = api_key.strip()
    has_configured_base_url = base_url is not None
    if base_url is not None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        base_url = resolve_base_url(base_url, "base_url")
    provider_for_auth = resolve_model_provider(model, provider, base_url)
    validate_litellm_endpoint(
        provider_for_auth, has_configured_base_url, "model"
    )
    validate_litellm_api_key(provider_for_auth, api_key, "api_key")
    routed_model, custom_llm_provider = resolve_litellm_route(
        model, provider, base_url
    )
    if base_url is not None:
        base_url = _prepare_litellm_base_url(
            base_url, provider_for_auth, routed_model
        )
    elif provider_for_auth == "vertex_ai":
        base_url = _default_vertex_base_url(routed_model)
    elif provider_for_auth in _PROVIDER_DEFAULT_BASE_URLS:
        # LiteLLM otherwise reads process-wide endpoint overrides such as
        # OPENAI_BASE_URL, DEEPSEEK_API_BASE, GEMINI_API_BASE, and
        # ANTHROPIC_BASE_URL. They must not silently reroute a model's
        # credentials; custom backend endpoints use the validated per-model
        # base_url field.
        base_url = _PROVIDER_DEFAULT_BASE_URLS[provider_for_auth]
    kwargs: Dict[str, Any] = {
        "model": routed_model,
        "messages": messages,
    }
    if custom_llm_provider is not None:
        kwargs["custom_llm_provider"] = custom_llm_provider
    if has_configured_base_url:
        if provider_for_auth != "vertex_ai" and not (
            isinstance(api_key, str) and api_key.strip()
        ):
            raise ValueError(
                "api_key must be explicitly set when base_url is configured"
            )
    if api_key:
        kwargs["api_key"] = api_key
    if base_url is not None:
        kwargs["base_url"] = base_url
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if stop is not None:
        kwargs["stop"] = stop
    if stream is not None:
        kwargs["stream"] = stream
    if logprobs is not None:
        kwargs["logprobs"] = logprobs
    if top_logprobs is not None:
        kwargs["top_logprobs"] = top_logprobs
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format
    if seed is not None:
        kwargs["seed"] = seed
    if n is not None:
        kwargs["n"] = n
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty
    if logit_bias is not None:
        kwargs["logit_bias"] = logit_bias
    if stream_options is not None:
        kwargs["stream_options"] = stream_options
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if thinking_budget is not None:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    return kwargs


async def llm_completion(
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[Any] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    tools: Optional[List[Any]] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[Any] = None,
    seed: Optional[int] = None,
    n: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[Dict[str, int]] = None,
    stream_options: Optional[Any] = None,
    reasoning_effort: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Non-streaming LLM completion. Returns response as a dict."""
    _validate_extra_kwargs(kwargs)
    params = _build_kwargs(
        model, messages,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        seed=seed,
        n=n,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        stream_options=stream_options,
        reasoning_effort=reasoning_effort,
        thinking_budget=thinking_budget,
    )
    client = _attach_openai_query_client(params)
    if client is None:
        client = _attach_litellm_query_client(params)
    try:
        response = await litellm.acompletion(**params)
        result = response.model_dump()
    except BaseException as exc:
        if client is not None:
            await _close_client_after_error(client, exc)
        raise
    if client is not None:
        await _close_client(client)
    return result


async def llm_stream_completion(
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[Any] = None,
    tools: Optional[List[Any]] = None,
    tool_choice: Optional[Any] = None,
    stream_options: Optional[Any] = None,
    reasoning_effort: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
    *,
    max_completion_tokens: Optional[int] = None,
    response_format: Optional[Any] = None,
    seed: Optional[int] = None,
    n: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> Any:
    """Streaming LLM completion. Returns an async iterable of chunk objects."""
    _validate_extra_kwargs(kwargs)
    params = _build_kwargs(
        model, messages,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        stream=True,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        seed=seed,
        n=n,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        stream_options=stream_options,
        reasoning_effort=reasoning_effort,
        thinking_budget=thinking_budget,
    )
    client = _attach_openai_query_client(params)
    if client is None:
        client = _attach_litellm_query_client(params)
    try:
        stream = await litellm.acompletion(**params)
    except BaseException as exc:
        if client is not None:
            await _close_client_after_error(client, exc)
        raise
    if client is None:
        return stream
    return _OwnedClientStream(stream, client)
