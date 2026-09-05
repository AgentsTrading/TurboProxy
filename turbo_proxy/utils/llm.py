"""
LLM completion wrapper using litellm for multi-provider routing and format
conversion. litellm handles model prefix routing (gemini/, openai/, anthropic/),
provider-specific API formatting, and cross-provider tool call compatibility.
"""

import asyncio
import json
import math
import os
from numbers import Real
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
_RESPONSES_PATH = "responses"
_OPENAI_CHAT_COMPLETIONS_RESPONSES_PREFIX = "openai/chat_completions/"
_URL_JOINING_PROVIDERS = {"deepseek"}
_QUERY_CLIENT_PROVIDERS = {"anthropic", "gemini", "vertex_ai"}
_AZURE_V1_API_VERSIONS = frozenset({"latest", "preview", "v1"})
_PROVIDER_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/beta",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com",
}

_MISSING_RESPONSES_INPUT = object()


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


def _encode_query_mapping(query: Optional[Dict[str, Any]]) -> bytes:
    """Encode extra query parameters using HTTPX's URL query rules."""
    if not query:
        return b""
    try:
        encoded = str(httpx.QueryParams(query))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "extra_query values must be URL-query-compatible"
        ) from exc
    return encoded.encode("ascii")


def _join_raw_queries(*queries: bytes) -> bytes:
    """Combine raw query fragments without decoding or reordering them."""
    return b"&".join(query for query in queries if query)


def _merge_query_params(base_query: bytes, extra_query: bytes) -> bytes:
    """Merge URL queries with request-level values taking precedence.

    ``extra_query`` is supplied for one call, so a key there should replace a
    value baked into the model's endpoint. Preserve the original raw spelling
    when keys do not overlap; this keeps repeated and valueless base
    parameters intact while avoiding needless re-encoding.
    """
    if not extra_query:
        return base_query
    if not base_query:
        return extra_query

    base_items = list(httpx.QueryParams(base_query).multi_items())
    extra_items = list(httpx.QueryParams(extra_query).multi_items())
    extra_keys = {key for key, _ in extra_items}
    if not any(key in extra_keys for key, _ in base_items):
        return _join_raw_queries(base_query, extra_query)

    merged_items = [
        (key, value) for key, value in base_items if key not in extra_keys
    ]
    merged_items.extend(extra_items)
    return str(httpx.QueryParams(merged_items)).encode("ascii")


class _OwnedClientStream:
    """An async stream that deterministically closes its owned SDK client."""

    def __init__(self, stream: Any, client: Any):
        self._stream = stream
        self._iterator: Optional[Any] = None
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
            if self._iterator is None:
                self._iterator = self._stream.__aiter__()
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

    def _source_aclose(self) -> Optional[Any]:
        sources = []
        if self._iterator is not None:
            sources.append(self._iterator)
        if self._stream is not self._iterator:
            sources.append(self._stream)

        for source in sources:
            close_stream = getattr(source, "aclose", None)
            if close_stream is not None:
                return close_stream

        # Some LiteLLM stream iterators own the real HTTP stream in this
        # wrapper but do not expose an aclose() method of their own.
        for source in sources:
            wrapper = getattr(
                source, "litellm_custom_stream_wrapper", None
            )
            close_stream = getattr(wrapper, "aclose", None)
            if close_stream is not None:
                return close_stream

        for source in sources:
            response = getattr(source, "response", None)
            close_stream = getattr(response, "aclose", None)
            if close_stream is not None:
                return close_stream
        return None

    async def _close_resources(self) -> None:
        source_error: Optional[BaseException] = None
        if not self._source_closed:
            try:
                close_stream = self._source_aclose()
                if close_stream is not None:
                    await close_stream()
                self._source_closed = True
            except BaseException as exc:
                source_error = exc

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
    *,
    responses: bool = False,
) -> str:
    """Prepare provider-specific URL joining without moving query parameters.

    LiteLLM 1.98.0 sends OpenAI chat through ``AsyncOpenAI``, whose ``base_url``
    must be the API root. DeepSeek instead appends its endpoint before using
    LiteLLM's HTTP handler. Fragments are dropped because HTTP never sends them.
    """
    # Azure's Responses transformer owns the ``/openai(/v1)/responses`` route.
    # Give it an API root, but keep a configured query in the URL itself so its
    # ``api-version`` detection can prevent the default version being added a
    # second time.  This branch is deliberately limited to Responses calls;
    # chat-completions uses Azure's separate deployment URL rules.
    if responses and provider == "azure":
        parsed = urlsplit(base_url)
        path = parsed.path.rstrip("/")
        chat_suffix = f"/{_CHAT_COMPLETIONS_PATH}"
        if path == chat_suffix or path.endswith(chat_suffix):
            # A Chat Completions resource URL can also be supplied for a
            # Responses request. Strip it before looking for deployments;
            # otherwise the endpoint becomes part of the deployment name.
            path = path[:-len(chat_suffix)].rstrip("/")
        for suffix in (
            "/openai/v1/responses",
            "/openai/responses",
            "/responses",
        ):
            if path == suffix or path.endswith(suffix):
                path = path[:-len(suffix)].rstrip("/")
                break
        # Azure chat configurations commonly point at a deployment resource
        # (``/openai/deployments/<name>``). Responses puts the model/deployment
        # in the request body and always uses the account-level route, so strip
        # that resource segment before LiteLLM appends ``/openai/responses``.
        lower_path = path.lower()
        for marker in ("/openai/v1/deployments/", "/openai/deployments/"):
            marker_index = lower_path.find(marker)
            if marker_index < 0:
                continue
            deployment = path[marker_index + len(marker):]
            if deployment and "/" not in deployment:
                path = path[:marker_index].rstrip("/")
                break
        for suffix in ("/openai/v1", "/openai"):
            if path == suffix or path.endswith(suffix):
                path = path[:-len(suffix)].rstrip("/")
                break
        return urlunsplit((
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.query,
            "",
        ))

    if provider == "azure":
        parsed = urlsplit(base_url)
        path = parsed.path.rstrip("/")
        chat_suffix = f"/{_CHAT_COMPLETIONS_PATH}"
        if path == chat_suffix or path.endswith(chat_suffix):
            path = path[:-len(chat_suffix)].rstrip("/")
        clean_url = urlunsplit((
            parsed.scheme,
            parsed.netloc,
            path,
            "",
            "",
        ))
        raw_query = httpx.URL(base_url).query
        if raw_query:
            return _LiteLLMQueryBaseURL(clean_url, raw_query)
        return clean_url

    if responses:
        # A configured full Responses resource URL is also accepted. The
        # common API root is needed because LiteLLM appends the resource path.
        parsed = urlsplit(base_url)
        path = parsed.path.rstrip("/")
        if path == f"/{_RESPONSES_PATH}" or path.endswith(
            f"/{_RESPONSES_PATH}"
        ):
            base_url = urlunsplit((
                parsed.scheme,
                parsed.netloc,
                path[:-len(_RESPONSES_PATH)].rstrip("/"),
                parsed.query,
                parsed.fragment,
            ))

    if provider == "openai":
        api_root, raw_query = _openai_endpoint_parts(base_url)
        return _OpenAIQueryBaseURL(api_root, raw_query)

    if provider in _QUERY_CLIENT_PROVIDERS:
        return _query_safe_provider_base_url(base_url, provider, model)

    if provider not in _URL_JOINING_PROVIDERS:
        # Native Responses provider configs generally append ``/responses``
        # with a string operation. Keep their configured query out of the
        # path and re-attach it in the request hook after that suffix exists.
        # This also covers OpenAI-compatible/custom providers (for example
        # OpenRouter) whose config is not known to this wrapper.
        parsed = urlsplit(base_url)
        clean_url = urlunsplit((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip("/"),
            "",
            "",
        ))
        raw_query = httpx.URL(base_url).query
        if raw_query:
            return _LiteLLMQueryBaseURL(clean_url, raw_query)
        return clean_url

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


def _attach_openai_query_client(
    params: Dict[str, Any], *, endpoint_key: str = "base_url",
) -> Optional[Any]:
    """Attach an isolated, owned client for a model-level OpenAI endpoint."""
    base_url = params.get(endpoint_key)
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
    params[endpoint_key] = str(base_url)
    params["client"] = client
    return client


def _model_scoped_azure_headers(api_key: str) -> Dict[str, Any]:
    """Keep process-wide OpenAI headers off a model-level Azure client."""
    from openai import omit

    headers: Dict[str, Any] = {
        "Authorization": omit,
        "OpenAI-Organization": omit,
        "OpenAI-Project": omit,
    }
    custom_headers = os.environ.get("OPENAI_CUSTOM_HEADERS")
    if custom_headers is not None:
        for line in custom_headers.split("\n"):
            colon = line.find(":")
            if colon >= 0 and (name := line[:colon].strip()):
                headers[name] = omit
    headers["api-key"] = api_key
    return headers


def _attach_azure_query_client(
    params: Dict[str, Any],
    request_hooks: Optional[List[Any]] = None,
) -> Optional[Any]:
    """Attach an OpenAI SDK client for a model-level Azure endpoint query."""
    endpoint_key = "api_base" if "api_base" in params else "base_url"
    base_url = params.get(endpoint_key)
    if not isinstance(base_url, _LiteLLMQueryBaseURL):
        return None

    from openai import AsyncAzureOpenAI, AsyncOpenAI, DefaultAsyncHttpxClient

    raw_query = base_url.raw_query
    parsed = urlsplit(str(base_url))
    path = parsed.path.rstrip("/")
    clean_url = urlunsplit((
        parsed.scheme,
        parsed.netloc,
        path,
        "",
        "",
    ))
    query_params = httpx.QueryParams(raw_query)
    api_version = params.get("api_version")
    if not isinstance(api_version, str) or not api_version:
        for key, value in query_params.multi_items():
            if key == "api-version":
                api_version = value
                break
    if not isinstance(api_version, str) or not api_version:
        api_version = os.environ.get("AZURE_API_VERSION")
    if not isinstance(api_version, str) or not api_version:
        api_version = getattr(
            litellm, "AZURE_DEFAULT_API_VERSION", "2025-02-01-preview"
        )

    origin = _url_origin(clean_url)
    hooks = list(request_hooks or [])

    async def merge_base_query(request: httpx.Request) -> None:
        if _url_origin(request.url) != origin:
            return
        current_url = httpx.URL(request.url)
        request.url = current_url.copy_with(
            query=_merge_query_params(current_url.query, raw_query)
        )

    hooks.append(merge_base_query)
    http_client = DefaultAsyncHttpxClient(
        event_hooks={"request": hooks},
        follow_redirects=True,
    )

    lower_path = path.lower()
    has_deployment_path = (
        "/openai/deployments/" in lower_path
        or "/openai/v1/deployments/" in lower_path
    )
    uses_v1 = (
        not has_deployment_path
        and (
            api_version in _AZURE_V1_API_VERSIONS
            or lower_path.endswith("/openai/v1")
        )
    )
    if uses_v1:
        if lower_path.endswith("/openai"):
            client_base_url = append_url_path(clean_url, "v1")
        elif lower_path.endswith("/openai/v1"):
            client_base_url = clean_url
        else:
            client_base_url = append_url_path(clean_url, "openai/v1")
        client = AsyncOpenAI(
            api_key=params.get("api_key") or "EMPTY",
            base_url=client_base_url,
            default_headers=_model_scoped_openai_headers(
                client_base_url, params.get("api_key") or "EMPTY"
            ),
            http_client=http_client,
        )
    elif has_deployment_path:
        client = AsyncAzureOpenAI(
            api_key=params.get("api_key") or "EMPTY",
            api_version=api_version,
            base_url=clean_url,
            default_headers=_model_scoped_azure_headers(
                params.get("api_key") or "EMPTY"
            ),
            http_client=http_client,
        )
    else:
        endpoint_path = path
        if lower_path.endswith("/openai"):
            endpoint_path = path[:-len("/openai")].rstrip("/")
        azure_endpoint = urlunsplit((
            parsed.scheme,
            parsed.netloc,
            endpoint_path,
            "",
            "",
        ))
        # Let the SDK derive the deployment from LiteLLM's final request model.
        # The model in params may still carry an Azure provider prefix.
        client = AsyncAzureOpenAI(
            api_key=params.get("api_key") or "EMPTY",
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            default_headers=_model_scoped_azure_headers(
                params.get("api_key") or "EMPTY"
            ),
            http_client=http_client,
        )

    if isinstance(client, AsyncAzureOpenAI):
        # The SDK reads AZURE_OPENAI_AD_TOKEN even with an explicit API key.
        # This model-scoped client must use only its configured key.
        client._azure_ad_token = None

    params[endpoint_key] = clean_url
    params["client"] = client
    return client


def _attach_litellm_query_client(
    params: Dict[str, Any],
    request_hooks: Optional[List[Any]] = None,
) -> Optional[Any]:
    """Attach a provider-compatible client when a provider base carries a query."""
    endpoint_key = "api_base" if "api_base" in params else "base_url"
    base_url = params.get(endpoint_key)
    if not isinstance(base_url, _LiteLLMQueryBaseURL):
        return None

    provider = params.get("custom_llm_provider")
    if isinstance(provider, str) and provider.lower() == "azure":
        return _attach_azure_query_client(params, request_hooks)

    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    raw_query = base_url.raw_query
    origin = _url_origin(str(base_url))

    async def append_base_query(request: httpx.Request) -> None:
        _append_base_query(request, raw_query, origin)

    hooks = list(request_hooks or [])
    hooks.append(append_base_query)
    client = AsyncHTTPHandler(event_hooks={"request": hooks})
    params[endpoint_key] = str(base_url)
    params["client"] = client
    return client


def _attach_responses_query_client(
    params: Dict[str, Any], *, omit_input: bool = False,
) -> Optional[Any]:
    """Attach an isolated Responses client for query or body adaptation."""
    base_url = params.get("api_base", params.get("base_url"))
    extra_query = _encode_query_mapping(params.get("extra_query"))
    request_hooks: List[Any] = []

    if omit_input:
        async def remove_placeholder_input(request: httpx.Request) -> None:
            """Remove the LiteLLM-only input placeholder from the wire body."""
            try:
                payload = json.loads(request.content)
            except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "could not preserve omitted Responses input"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError("could not preserve omitted Responses input")
            payload.pop("input", None)
            content = json.dumps(
                payload, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")
            # HTTPX event hooks run after request construction, so replace both
            # the replayable stream and cached body before the transport sends.
            request.stream = httpx.ByteStream(content)
            request._content = content
            request.headers["content-length"] = str(len(content))
            request.headers.pop("transfer-encoding", None)

        request_hooks.append(remove_placeholder_input)

    # Azure parses ``api_base`` itself and uses its query to decide whether to
    # add the default ``api-version``.  Applying a configured query later in an
    # HTTP hook makes Azure append its default first, yielding duplicate
    # ``api-version`` values.  Merge the query into ``api_base`` before
    # LiteLLM builds the route so its normal de-duplication can run.
    provider = params.get("custom_llm_provider")
    if isinstance(provider, str) and provider.lower() == "azure":
        if isinstance(base_url, _LiteLLMQueryBaseURL):
            parsed = httpx.URL(str(base_url))
            base_query = base_url.raw_query
        else:
            parsed = httpx.URL(str(base_url))
            base_query = parsed.query
        merged_query = _merge_query_params(base_query, extra_query)
        if merged_query:
            params["api_base"] = str(parsed.copy_with(params=merged_query))
            # The query has been folded into api_base. Leaving it in the
            # LiteLLM kwargs would allow a future transformer to append it a
            # second time.
            if extra_query:
                params.pop("extra_query", None)
            # LiteLLM's Azure transformer uses the explicit ``api_version``
            # argument to choose between ``/openai/responses`` and
            # ``/openai/v1/responses``. Derive it from the URL query so a
            # dated configured version is not silently treated as ``preview``.
            if "api_version" not in params:
                query_params = httpx.QueryParams(merged_query)
                api_version = None
                for key, value in query_params.multi_items():
                    if key == "api-version":
                        api_version = value
                        break
                if api_version:
                    params["api_version"] = api_version
        elif isinstance(base_url, _LiteLLMQueryBaseURL):
            params["api_base"] = str(parsed)
        if request_hooks:
            from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

            client = AsyncHTTPHandler(
                event_hooks={"request": request_hooks},
            )
            params["client"] = client
            return client
        return None

    if not isinstance(base_url, _OpenAIQueryBaseURL):
        if isinstance(base_url, _LiteLLMQueryBaseURL):
            if extra_query:
                params["api_base"] = _LiteLLMQueryBaseURL(
                    str(base_url),
                    _merge_query_params(base_url.raw_query, extra_query),
                )
        elif extra_query:
            parsed = urlsplit(str(base_url))
            clean_url = urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
            )
            params["api_base"] = _LiteLLMQueryBaseURL(
                clean_url,
                _merge_query_params(
                    httpx.URL(str(base_url)).query, extra_query,
                ),
            )
        client = _attach_litellm_query_client(params, request_hooks)
        if client is not None or not request_hooks:
            return client
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        client = AsyncHTTPHandler(event_hooks={"request": request_hooks})
        params["client"] = client
        return client

    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    raw_query = _merge_query_params(base_url.raw_query, extra_query)
    origin = _url_origin(str(base_url))

    async def append_base_query(request: httpx.Request) -> None:
        _append_base_query(request, raw_query, origin)

    request_hooks.append(append_base_query)
    client = AsyncHTTPHandler(event_hooks={"request": request_hooks})
    params["api_base"] = str(base_url)
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
    responses: bool = False,
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
            base_url,
            provider_for_auth,
            routed_model,
            responses=responses,
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


_RESPONSES_OPTIONAL_PARAMS = (
    "include",
    "instructions",
    "prompt",
    "metadata",
    "conversation",
    "parallel_tool_calls",
    "previous_response_id",
    "reasoning",
    "store",
    "background",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_p",
    "truncation",
    "user",
    "service_tier",
    "safety_identifier",
    "stream_options",
    "top_logprobs",
    "max_tool_calls",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "context_management",
    "moderation",
    "partial_images",
    "thinking",
    "text_format",
)

_RESPONSES_REQUEST_FIELDS = frozenset(_RESPONSES_OPTIONAL_PARAMS) | frozenset({
    "allowed_openai_params",
    "extra_body",
    "extra_headers",
    "extra_query",
    "input",
    "max_output_tokens",
    "max_tokens",
    "model",
    "response_format",
    "stream",
    "timeout",
})

# LiteLLM 1.98 accepts these names at the Python boundary but does not yet
# include all of them in ``ResponsesAPIOptionalRequestParams``. Keep the
# direct kwargs for newer LiteLLM versions and also place them in
# ``extra_body`` so the current native HTTP handler does not silently drop
# them.
_RESPONSES_COMPAT_EXTRA_BODY_PARAMS = frozenset({
    "conversation",
    "moderation",
    "partial_images",
    "prompt_cache_options",
    "thinking",
})

_RESPONSES_PROTECTED_HEADERS = frozenset({
    "authorization",
    "api-key",
    "x-api-key",
    "x-goog-api-key",
    "proxy-authorization",
    "host",
    "content-length",
    "transfer-encoding",
    "cookie",
})

# ``extra_body`` is merged into the provider payload after LiteLLM has built the
# request. Standard fields must stay owned by this wrapper so extensions cannot
# override routing or bypass native-provider validation.
_RESPONSES_PROTECTED_EXTRA_BODY_KEYS = _RESPONSES_REQUEST_FIELDS | frozenset({
    "messages",
    "api_key",
    "api_base",
    "base_url",
    "custom_llm_provider",
    "client",
})


def _responses_provider_uses_chat_fallback(
    model: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> tuple[bool, str]:
    """Return whether LiteLLM will bridge Responses through Chat Completions."""
    from litellm.utils import ProviderConfigManager

    resolved_provider = resolve_model_provider(model, provider, base_url)
    routed_model, custom_llm_provider = resolve_litellm_route(
        model, provider, base_url,
    )
    provider_name = custom_llm_provider or resolved_provider
    # LiteLLM treats this model prefix as an explicit request to route the
    # Responses API through its Chat Completions bridge, even though OpenAI
    # otherwise has a native Responses provider config.
    # Check both the caller's model and the model after explicit-provider
    # normalization.  ``resolve_litellm_route`` strips the first recognized
    # provider prefix when ``provider`` is supplied, so a value such as
    # ``openai/openai/chat_completions/gpt-4o`` would otherwise become the
    # bridge-selecting ``openai/chat_completions/gpt-4o`` after this check.
    # When no provider is supplied, LiteLLM performs the same one-prefix
    # normalization internally.  ``resolve_litellm_route`` intentionally
    # preserves the prefix for that case, so include ``split_model_name``'s
    # provider-neutral model id as an additional candidate.
    normalized_model = split_model_name(model)[0] if provider is None else None
    candidate_models = (model, routed_model, normalized_model)
    # LiteLLM honors this prefix even with a different explicit provider.
    forced_openai_chat = any(
        isinstance(candidate, str)
        and candidate.strip().startswith(
            _OPENAI_CHAT_COMPLETIONS_RESPONSES_PREFIX
        )
        for candidate in candidate_models
    )
    if forced_openai_chat:
        return True, provider_name
    # Capability can depend on the bare model id (for example Bedrock Mantle).
    # Do not call LiteLLM's runtime resolver here: it can acquire credentials
    # and start synchronous device authentication before validation finishes.
    routed_model = routed_model.removeprefix(f"{provider_name}/")
    native_config = ProviderConfigManager.get_provider_responses_api_config(
        provider=provider_name,
        model=routed_model,
    )
    return native_config is None, provider_name


def _require_native_responses_provider(
    model: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """Reject providers that LiteLLM would bridge through Chat Completions."""
    uses_chat_fallback, resolved_provider = (
        _responses_provider_uses_chat_fallback(model, provider, base_url)
    )
    if uses_chat_fallback:
        raise ValueError(
            "/v1/responses requires a native Responses provider; "
            f"provider={resolved_provider!r}, model={model!r}"
        )
    return resolved_provider


def _validate_responses_extras(
    extra_headers: Optional[Dict[str, Any]],
    extra_query: Optional[Dict[str, Any]],
    timeout: Any,
    allowed_openai_params: Optional[List[str]],
) -> None:
    if extra_headers is not None:
        if not isinstance(extra_headers, dict):
            raise ValueError("extra_headers must be an object or null")
        if any(not isinstance(key, str) or not key.strip() for key in extra_headers):
            raise ValueError("extra_headers keys must be non-empty strings")
        if any(not isinstance(value, str) for value in extra_headers.values()):
            raise ValueError("extra_headers values must be strings")
        protected = sorted(
            key for key in extra_headers
            if key.lower() in _RESPONSES_PROTECTED_HEADERS
        )
        if protected:
            raise ValueError(
                "extra_headers cannot override protected header(s): "
                + ", ".join(protected)
            )

    if extra_query is not None:
        if not isinstance(extra_query, dict):
            raise ValueError("extra_query must be an object or null")
        if any(not isinstance(key, str) or not key.strip() for key in extra_query):
            raise ValueError("extra_query keys must be non-empty strings")

    if timeout is not None:
        if isinstance(timeout, bool):
            raise ValueError("timeout must be a positive number or httpx.Timeout")
        if isinstance(timeout, Real):
            if not math.isfinite(float(timeout)) or timeout <= 0:
                raise ValueError("timeout must be a positive number")
        elif not isinstance(timeout, httpx.Timeout):
            raise ValueError("timeout must be a positive number or httpx.Timeout")

    if allowed_openai_params is not None:
        if not isinstance(allowed_openai_params, (list, tuple)):
            raise ValueError("allowed_openai_params must be an array or null")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in allowed_openai_params
        ):
            raise ValueError("allowed_openai_params must contain strings")


def _validate_responses_background(background: Any) -> None:
    if background is not None and not isinstance(background, bool):
        raise ValueError("background must be a boolean or null")
    if background is True:
        raise ValueError(
            "background=true is not supported because Responses retrieval "
            "endpoints are not implemented"
        )


def _validate_responses_request(body: Any) -> None:
    """Validate request-shape fields before a Responses request starts.

    LiteLLM validates these values only when it is called. The proxy needs the
    same checks before constructing a streaming response so malformed client
    requests can still receive an HTTP 400 status.
    """
    if not isinstance(body, dict):
        raise ValueError("Invalid JSON: request body must be an object")
    unknown = sorted(
        str(key) for key in body if key not in _RESPONSES_REQUEST_FIELDS
    )
    if unknown:
        raise ValueError(
            "unsupported Responses parameter(s): " + ", ".join(unknown)
        )
    # OpenAI's Responses API allows callers to omit ``input`` (for example
    # when continuing a conversation by ``previous_response_id``).  Validate
    # its shape only when the field is present.
    if "input" in body and not isinstance(body["input"], (str, list)):
        raise ValueError("input must be a string or array")
    if isinstance(body.get("input"), list) and any(
        not isinstance(item, dict) for item in body["input"]
    ):
        raise ValueError("input array items must be objects")
    if "stream" in body and not isinstance(body["stream"], bool):
        raise ValueError("stream must be a boolean")
    _validate_responses_background(body.get("background"))

    extra_body = body.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("extra_body must be an object or null")
    if extra_body:
        protected = sorted(
            str(key) for key in extra_body
            if isinstance(key, str)
            and key.lower() in _RESPONSES_PROTECTED_EXTRA_BODY_KEYS
        )
        if protected:
            raise ValueError(
                "extra_body cannot override protected field(s): "
                + ", ".join(protected)
            )

    _validate_responses_extras(
        body.get("extra_headers"),
        body.get("extra_query"),
        body.get("timeout"),
        body.get("allowed_openai_params"),
    )


def _responses_text_from_response_format(response_format: Any) -> Any:
    """Normalize compatibility structured-output formats for Responses.

    Chat Completions nests JSON schema metadata under ``json_schema`` while
    Responses expects ``name``, ``schema`` and ``strict`` directly under
    ``text.format``. Native Responses ``{"format": ...}`` values are kept
    unchanged so callers can use either request shape.
    """
    if isinstance(response_format, type):
        from litellm.llms.base_llm.base_utils import type_to_response_format_param

        response_format = type_to_response_format_param(response_format)

    if not isinstance(response_format, dict):
        return {"format": response_format}

    if "format" in response_format:
        format_value = response_format["format"]
        if not isinstance(format_value, dict):
            raise ValueError("Responses text.format must be an object")
        if format_value.get("type") == "json_schema":
            nested_schema = format_value.get("json_schema")
            if isinstance(nested_schema, dict):
                format_value = {
                    "type": "json_schema",
                    "name": nested_schema.get("name", "response_schema"),
                    "schema": nested_schema.get("schema", {}),
                    "strict": nested_schema.get("strict", False),
                }
        return {
            **{
                key: value for key, value in response_format.items()
                if key != "format"
            },
            "format": format_value,
        }

    format_type = response_format.get("type")
    if isinstance(format_type, str) and format_type in {"json_object", "text"}:
        return {"format": response_format}

    if format_type != "json_schema":
        raise ValueError(
            "structured output format must be json_schema, json_object, or text"
        )

    json_schema = response_format.get("json_schema")
    if isinstance(json_schema, dict):
        format_value = {
            "type": "json_schema",
            "name": json_schema.get("name", "response_schema"),
            "schema": json_schema.get("schema", {}),
            "strict": json_schema.get("strict", False),
        }
        return {"format": format_value}

    # Also accept the flattened Responses format object when it is supplied
    # directly as a text_format/response_format dictionary.
    if "schema" in response_format or "name" in response_format:
        return {"format": dict(response_format)}
    raise ValueError(
        "json_schema format must contain a json_schema object or a schema"
    )


def _build_responses_kwargs(
    model: str,
    input: Any = _MISSING_RESPONSES_INPUT,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    stream: Optional[bool] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    text_format: Any = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    timeout: Any = None,
    allowed_openai_params: Optional[List[str]] = None,
    response_format: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("extra_body must be an object or null")
    if extra_body:
        protected = sorted(
            str(key) for key in extra_body
            if isinstance(key, str)
            and key.lower() in _RESPONSES_PROTECTED_EXTRA_BODY_KEYS
        )
        if protected:
            raise ValueError(
                "extra_body cannot override protected field(s): "
                + ", ".join(protected)
            )

    _validate_responses_extras(
        extra_headers,
        extra_query,
        timeout,
        allowed_openai_params,
    )

    unknown = sorted(set(kwargs) - set(_RESPONSES_OPTIONAL_PARAMS))
    if unknown:
        raise ValueError(
            "unsupported Responses parameter(s): " + ", ".join(unknown)
        )
    _validate_responses_background(kwargs.get("background"))

    _require_native_responses_provider(model, provider, base_url)

    routing = _build_kwargs(
        model=model,
        messages=[],
        api_key=api_key,
        base_url=base_url,
        provider=provider,
        responses=True,
    )
    routing.pop("messages")
    # LiteLLM's Responses API accepts the endpoint as ``api_base``. Its
    # native wrapper accepts ``api_base``; normalize the configuration helper's
    # alias before handing it to LiteLLM.
    if "base_url" in routing:
        routing["api_base"] = routing.pop("base_url")
    input_omitted = input is _MISSING_RESPONSES_INPUT
    # LiteLLM 1.98 requires ``input`` in its Python signature even though the
    # Responses API makes it optional. Native calls use a temporary placeholder
    # that the owned request hook removes before the HTTP request is sent.
    params: Dict[str, Any] = {
        **routing,
        "input": "" if input_omitted else input,
    }

    output_limit = max_output_tokens if max_output_tokens is not None else max_tokens
    if output_limit is not None:
        params["max_output_tokens"] = output_limit
    if reasoning_effort is not None and kwargs.get("reasoning") is None:
        params["reasoning"] = {"effort": reasoning_effort}
    compat_extra_body: Dict[str, Any] = {}
    if thinking_budget is not None:
        thinking = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        params["thinking"] = thinking
        compat_extra_body["thinking"] = thinking
    if stream is not None:
        params["stream"] = stream

    if text_format is not None and kwargs.get("text") is None:
        text = _responses_text_from_response_format(text_format)
        params["text"] = text
    if (
        response_format is not None
        and text_format is None
        and kwargs.get("text") is None
    ):
        # Compatibility with clients that reuse Chat Completions' structured
        # output field when switching to the Responses endpoint.
        text = _responses_text_from_response_format(response_format)
        params["text"] = text
    if extra_headers is not None:
        params["extra_headers"] = extra_headers
    if extra_query is not None:
        params["extra_query"] = extra_query
    if timeout is not None:
        params["timeout"] = timeout
    if allowed_openai_params is not None:
        params["allowed_openai_params"] = list(allowed_openai_params)

    for key in _RESPONSES_OPTIONAL_PARAMS:
        value = kwargs.get(key)
        if value is not None:
            params[key] = value
            if key in _RESPONSES_COMPAT_EXTRA_BODY_PARAMS:
                compat_extra_body[key] = value

    if extra_body:
        compat_extra_body = {**compat_extra_body, **extra_body}
    if compat_extra_body:
        params["extra_body"] = compat_extra_body
    return params


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


async def llm_response(
    model: str,
    input: Any = _MISSING_RESPONSES_INPUT,
    api_key: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
    *,
    stream: Optional[bool] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    text_format: Any = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    timeout: Any = None,
    allowed_openai_params: Optional[List[str]] = None,
    response_format: Any = None,
    **kwargs: Any,
) -> dict:
    """Non-streaming Responses API call through LiteLLM."""
    if stream is not None:
        raise ValueError(
            "stream cannot be passed through llm_response; use "
            "llm_stream_response explicitly"
        )
    params = _build_responses_kwargs(
        model=model,
        input=input,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
        max_output_tokens=max_output_tokens,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        thinking_budget=thinking_budget,
        extra_body=extra_body,
        text_format=text_format,
        extra_headers=extra_headers,
        extra_query=extra_query,
        timeout=timeout,
        allowed_openai_params=allowed_openai_params,
        response_format=response_format,
        **kwargs,
    )
    client = _attach_responses_query_client(
        params, omit_input=input is _MISSING_RESPONSES_INPUT,
    )
    try:
        response = await litellm.aresponses(**params)
        result = response.model_dump() if hasattr(response, "model_dump") else response
    except BaseException as exc:
        if client is not None:
            await _close_client_after_error(client, exc)
        raise
    if client is not None:
        await _close_client(client)
    return result


async def llm_stream_response(
    model: str,
    input: Any = _MISSING_RESPONSES_INPUT,
    api_key: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
    *,
    stream: Optional[bool] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    text_format: Any = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    timeout: Any = None,
    allowed_openai_params: Optional[List[str]] = None,
    response_format: Any = None,
    **kwargs: Any,
) -> Any:
    """Streaming Responses API call through LiteLLM."""
    if stream is not None:
        raise ValueError(
            "stream cannot be passed through llm_stream_response; use "
            "llm_response explicitly for non-streaming calls"
        )
    params = _build_responses_kwargs(
        model=model,
        input=input,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
        max_output_tokens=max_output_tokens,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        thinking_budget=thinking_budget,
        stream=True,
        extra_body=extra_body,
        text_format=text_format,
        extra_headers=extra_headers,
        extra_query=extra_query,
        timeout=timeout,
        allowed_openai_params=allowed_openai_params,
        response_format=response_format,
        **kwargs,
    )
    client = _attach_responses_query_client(
        params, omit_input=input is _MISSING_RESPONSES_INPUT,
    )
    try:
        stream = await litellm.aresponses(**params)
    except BaseException as exc:
        if client is not None:
            await _close_client_after_error(client, exc)
        raise
    if client is None:
        return stream
    return _OwnedClientStream(stream, client)
