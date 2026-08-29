"""Build the llm-verifier client for a configured verifier model.

llm-verifier scores with token logprobs, and every backend exposes them
differently: Gemini through Vertex AI, DeepSeek through its own sampled score
tags, and vLLM / SGLang through the OpenAI-compatible ``logprobs`` field.
Routing on the model-name prefix lets the verifier and the progress monitor
point at any of them instead of assuming Gemini.

The model id and the client are resolved separately so the id is available
without paying for a client the caller may never use.
"""

import os
import re
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import (
    ModelConfig,
    is_official_openai_base_url,
    resolve_base_url,
    resolve_model_provider,
    split_model_name,
)


def _usable_key(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _model_scoped_openai_headers(
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    """Keep process-wide OpenAI headers off model-level custom endpoints."""
    from openai import omit

    custom_header_names = []
    custom_headers = os.environ.get("OPENAI_CUSTOM_HEADERS")
    if custom_headers is not None:
        for line in custom_headers.split("\n"):
            colon = line.find(":")
            if colon >= 0 and (name := line[:colon].strip()):
                custom_header_names.append(name)

    authorization = f"Bearer {api_key}"
    headers: dict[str, Any] = {}
    if not is_official_openai_base_url(base_url):
        for name in custom_header_names:
            headers[name] = omit
        headers["OpenAI-Organization"] = omit
        headers["OpenAI-Project"] = omit

    # OPENAI_CUSTOM_HEADERS is merged after the SDK's auth header. Override
    # every spelling it supplied so an ambient Authorization value cannot
    # replace the key belonging to this model.
    for name in custom_header_names:
        if name.lower() == "authorization":
            headers[name] = authorization
    headers["Authorization"] = authorization
    return headers


def _configured_openai_base_url(cfg: ModelConfig, provider: str) -> Optional[str]:
    """Return the endpoint that an OpenAI-compatible config will use.

    ``llm-verifier`` supports ``OPENAI_BASE_URL`` as the process-level
    fallback.  Resolve it during validation too, so the tournament verifier
    does not reject a custom endpoint before the client can be built.
    """
    if provider != "openai":
        return cfg.base_url
    return cfg.base_url or _usable_key(os.environ.get("OPENAI_BASE_URL"))


def _append_url_path(base_url: str, path: str) -> str:
    """Append an endpoint path without moving query or fragment components.

    ``urllib.parse.urljoin`` treats a leading slash as an absolute path and
    string concatenation places a base URL's query before the endpoint path.
    URL components are split explicitly so a URL such as
    ``https://gateway.example/v1?tenant=acme#route`` becomes
    ``https://gateway.example/v1/chat/completions?tenant=acme#route``.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    parsed = urlsplit(base_url.strip())
    base_path = parsed.path.rstrip("/")
    suffix = path.lstrip("/")
    if suffix:
        joined_path = f"{base_path}/{suffix}" if base_path else f"/{suffix}"
    else:
        joined_path = base_path or "/"
    return urlunsplit((
        parsed.scheme,
        parsed.netloc,
        joined_path,
        parsed.query,
        parsed.fragment,
    ))


def _normalise_base_url(base_url: str) -> str:
    """Trim only the base path slash, leaving query/fragment untouched."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    parsed = urlsplit(base_url.strip())
    return urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path.rstrip("/"),
        parsed.query,
        parsed.fragment,
    ))


def _openai_api_base_url(base_url: str) -> str:
    """Return the API root when a full Chat Completions URL is configured."""
    normalised = _normalise_base_url(base_url)
    parsed = urlsplit(normalised)
    suffix = "/chat/completions"
    path = parsed.path
    if path == suffix or path.endswith(suffix):
        path = path[:-len(suffix)]
    return urlunsplit((
        parsed.scheme,
        parsed.netloc,
        path,
        parsed.query,
        parsed.fragment,
    ))


def _openai_endpoint_parts(base_url: str) -> tuple[str, bytes]:
    """Return an OpenAI-safe base URL and the raw query string.

    OpenAI's client enforces a trailing slash by mutating ``raw_path``.  In
    httpx that raw path includes the query bytes, so handing it a URL with a
    query turns ``?token=x`` into ``?token=x/``.  Supplying the raw query from
    an HTTPX request hook keeps it after the endpoint path instead.
    """
    parsed = urlsplit(_openai_api_base_url(base_url))
    clean_url = urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path.rstrip("/"),
        "",
        "",
    ))
    return clean_url, httpx.URL(_openai_api_base_url(base_url)).query


def _url_origin(url: Any) -> tuple[str, str, Optional[int]]:
    parsed = httpx.URL(url)
    return parsed.scheme.lower(), parsed.host.lower(), parsed.port


def _append_base_query(
    request: httpx.Request,
    raw_query: bytes,
    origin: tuple[str, str, Optional[int]],
) -> None:
    """Append a configured query once, and never forward it cross-origin."""
    request.url = httpx.URL(_url_with_base_query(request.url, raw_query, origin))


def _url_with_base_query(
    url: Any,
    raw_query: bytes,
    origin: tuple[str, str, Optional[int]],
) -> str:
    """Place a base query after an SDK-generated operation query."""
    parsed = httpx.URL(url)
    if _url_origin(parsed) != origin:
        return str(parsed)
    existing = parsed.query
    if existing == raw_query or existing.endswith(b"&" + raw_query):
        return str(parsed)
    query = raw_query if not existing else existing + b"&" + raw_query
    return str(parsed.copy_with(query=query))


def _attach_vertex_base_query(
    client: Any,
    raw_query: bytes,
    origin: tuple[str, str, Optional[int]],
) -> None:
    """Apply a custom Vertex base query after google-genai builds each path."""
    api_client = client._api_client
    build_request = api_client._build_request

    def build_request_with_base_query(
        http_method: str,
        path: str,
        request_dict: dict[str, object],
        http_options: Optional[Any] = None,
    ) -> Any:
        request = build_request(
            http_method,
            path,
            request_dict,
            http_options,
        )
        request.url = _url_with_base_query(request.url, raw_query, origin)
        return request

    api_client._build_request = build_request_with_base_query


def _create_openai_compatible_client(
    base_url: str,
    api_key: Optional[str],
) -> Any:
    """Build an OpenAI client while preserving endpoint query parameters."""
    from openai import DefaultHttpxClient, OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    clean_url, raw_query = _openai_endpoint_parts(base_url)
    resolved_api_key = api_key or "EMPTY"
    kwargs: dict[str, Any] = {
        "base_url": clean_url,
        "api_key": resolved_api_key,
        "default_headers": _model_scoped_openai_headers(
            clean_url, resolved_api_key
        ),
    }
    if raw_query:
        # OpenAI 2.x appends resource paths to the base URL before applying
        # the URL path. An event hook retains the raw query exactly,
        # including repeated or valueless parameters, after that path.
        origin = _url_origin(clean_url)

        def append_base_query(request: httpx.Request) -> None:
            _append_base_query(request, raw_query, origin)

        kwargs["http_client"] = DefaultHttpxClient(
            event_hooks={"request": [append_base_query]},
            follow_redirects=True,
        )
    return OpenAI(**kwargs)


def _official_vertex_base_url(location: Optional[str]) -> str:
    """Return an official Vertex origin without consulting SDK overrides."""
    resolved_location = _usable_key(location) or "global"
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", resolved_location):
        raise ValueError(
            "GOOGLE_CLOUD_LOCATION must contain only lowercase letters, "
            "digits, and hyphens"
        )
    if resolved_location == "global":
        return "https://aiplatform.googleapis.com/"
    if resolved_location in ("us", "eu"):
        return f"https://aiplatform.{resolved_location}.rep.googleapis.com/"
    return f"https://{resolved_location}-aiplatform.googleapis.com/"


def build_vertex_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    *,
    force_adc: bool = False,
) -> Any:
    """Build a Vertex client without google-genai's generic-key fallback."""
    from google import auth, genai

    key = None if force_adc else (
        _usable_key(api_key) or _usable_key(os.environ.get("VERTEX_API_KEY"))
    )
    client_kwargs: dict[str, Any] = {"vertexai": True}
    if key:
        client_kwargs["api_key"] = key
    else:
        credentials, default_project = auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        project = (
            _usable_key(os.environ.get("GOOGLE_CLOUD_PROJECT"))
            or _usable_key(default_project)
        )
        if not project:
            raise ValueError(
                "Vertex ADC requires a Google Cloud project; set "
                "GOOGLE_CLOUD_PROJECT or configure a project in ADC"
            )
        client_kwargs.update({
            "credentials": credentials,
            "project": project,
            "location": (
                _usable_key(os.environ.get("GOOGLE_CLOUD_LOCATION"))
                or "global"
            ),
        })

    from google.genai.types import HttpOptions

    vertex_query = b""
    vertex_origin: Optional[tuple[str, str, Optional[int]]] = None
    if base_url is not None:
        # google-genai 2.20.0 defaults Vertex requests to /v1beta1 after the
        # supplied base URL.  A custom endpoint already owns its complete
        # path (for example /v1), so disable that implicit version component.
        # Keep its query aside until the operation path is complete. Streaming
        # includes ``?alt=sse`` in that path, so leaving a query on the base URL
        # would produce two question marks.
        normalised_base_url = _normalise_base_url(base_url)
        parsed_base_url = urlsplit(normalised_base_url)
        clean_base_url = urlunsplit((
            parsed_base_url.scheme,
            parsed_base_url.netloc,
            parsed_base_url.path,
            "",
            "",
        ))
        vertex_query = httpx.URL(normalised_base_url).query
        vertex_origin = _url_origin(clean_base_url)
        client_kwargs["http_options"] = HttpOptions(
            base_url=clean_base_url,
            api_version="",
        )
    else:
        # google-genai also accepts process-wide and global base URL overrides.
        # Model credentials must never follow those implicit endpoints.
        client_kwargs["http_options"] = HttpOptions(
            base_url=_official_vertex_base_url(client_kwargs.get("location")),
        )
    client = genai.Client(**client_kwargs)
    if vertex_query and vertex_origin is not None:
        _attach_vertex_base_query(client, vertex_query, vertex_origin)
    return client


def _ensure_verifiable(cfg: ModelConfig, purpose: str = "verifier") -> None:
    """Reject a backend that cannot score.

    The Messages API exposes no logprobs, and the fine-grained reward is an
    expectation over the score-token distribution — there is nothing to take
    an expectation over. Raising here rather than in the lazy client property
    means a bad config fails at startup, not on the first request.
    """
    if purpose not in ("verifier", "progress"):
        raise ValueError(f"unsupported verifier-client purpose: {purpose}")
    if cfg.api_key is not None and not isinstance(cfg.api_key, str):
        raise ValueError("verifier api_key must be a string or null")
    provider = resolve_model_provider(cfg.name, cfg.provider, cfg.base_url)
    if provider is None:
        raise ValueError(
            f"cannot determine the verifier provider for '{cfg.name}'; add a "
            "recognized model prefix or an explicit provider"
        )
    if provider == "anthropic":
        raise ValueError(
            f"'{cfg.name}' cannot be a verifier: the Anthropic Messages API "
            "does not return token logprobs, which the fine-grained reward "
            "needs. Claude works as a backend model — put it under "
            "`backend.models` and verify with a logprob backend "
            "(DeepSeek, Vertex AI, or a vLLM/SGLang base_url).")
    if provider == "gemini":
        raise ValueError(
            f"'{cfg.name}' cannot use the plain Gemini API as a verifier: "
            "it does not expose the token logprobs the reward needs. Set "
            "`provider: vertex_ai` and use a Vertex API key."
        )
    if cfg.base_url is not None:
        if not isinstance(cfg.base_url, str) or not cfg.base_url.strip():
            raise ValueError("verifier base_url must be a non-empty string")
        resolve_base_url(cfg.base_url, "verifier base_url")
        if provider != "vertex_ai" and _usable_key(cfg.api_key) is None:
            raise ValueError(
                "a verifier base_url requires an explicit, non-empty api_key"
            )
    effective_base_url = _configured_openai_base_url(cfg, provider)
    if provider == "openai" and effective_base_url is not None:
        resolve_base_url(
            effective_base_url,
            "verifier base_url" if cfg.base_url else "OPENAI_BASE_URL",
        )
    if purpose == "verifier" and is_official_openai_base_url(
        effective_base_url
    ):
        raise ValueError(
            "the official OpenAI API cannot be used as the tournament verifier: "
            "llm-verifier 0.2.0 requires the vLLM/SGLang score-prefill "
            "extensions. Use Vertex AI, DeepSeek, or an OpenAI-compatible "
            "vLLM/SGLang base_url."
        )
    if cfg.base_url and provider not in ("openai", "deepseek", "vertex_ai"):
        raise ValueError(
            "a verifier base_url must use the OpenAI-compatible or DeepSeek "
            f"protocol, not provider '{provider}'"
        )
    if purpose == "verifier" and provider == "openai" and not effective_base_url:
        raise ValueError(
            "the official OpenAI API cannot be used as the tournament verifier: "
            "llm-verifier 0.2.0 requires the vLLM/SGLang score-prefill "
            "extensions. Use Vertex AI, DeepSeek, or an OpenAI-compatible "
            "vLLM/SGLang base_url."
        )
    if provider not in ("deepseek", "openai", "gemini", "vertex_ai"):
        raise ValueError(
            f"'{provider}' is not a supported verifier provider; use "
            "deepseek, vertex_ai, or an OpenAI-compatible vLLM/SGLang base_url"
        )


def verifier_model_id(cfg: ModelConfig, purpose: str = "verifier") -> str:
    """The bare model name to send to the verifier backend."""
    _ensure_verifiable(cfg, purpose)
    return split_model_name(cfg.name)[0]


def build_verifier_client(
    cfg: ModelConfig, purpose: str = "verifier"
) -> Any:
    """Build the llm-verifier client for the validated provider in ``cfg``."""
    model_id = verifier_model_id(cfg, purpose)
    provider = resolve_model_provider(cfg.name, cfg.provider, cfg.base_url)

    if cfg.base_url:
        if provider == "vertex_ai":
            return build_vertex_client(cfg.api_key, cfg.base_url)
        client = _create_openai_compatible_client(
            cfg.base_url,
            _usable_key(cfg.api_key),
        )
        # llm-verifier guesses DeepSeek from the URL. The explicit provider is
        # authoritative here, including for proxies whose hostname happens to
        # contain api.deepseek.com.
        client._llm_verifier_model = model_id
        client._llm_verifier_deepseek = provider == "deepseek"
        return client

    if provider == "deepseek":
        from dotenv import load_dotenv
        from llm_verifier import MissingAPIKeyError

        load_dotenv()
        api_key = _usable_key(cfg.api_key) or _usable_key(
            os.environ.get("DEEPSEEK_API_KEY")
        )
        if not api_key:
            raise MissingAPIKeyError(
                "set DEEPSEEK_API_KEY in .env or environment to use DeepSeek "
                "as the verifier"
            )
        client = _create_openai_compatible_client(
            "https://api.deepseek.com",
            api_key,
        )
        client._llm_verifier_model = model_id
        client._llm_verifier_deepseek = True
        return client

    if provider == "openai":
        api_key = _usable_key(cfg.api_key) or _usable_key(
            os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "an OpenAI verifier requires api_key or OPENAI_API_KEY"
            )
        # Progress monitoring may intentionally use OPENAI_BASE_URL.  The
        # tournament verifier still rejects the official OpenAI route above.
        client = _create_openai_compatible_client(
            _usable_key(os.environ.get("OPENAI_BASE_URL"))
            or "https://api.openai.com/v1",
            api_key,
        )
        client._llm_verifier_model = model_id
        client._llm_verifier_deepseek = False
        return client

    if provider == "vertex_ai":
        return build_vertex_client(cfg.api_key)

    raise ValueError(f"cannot build a verifier client for provider '{provider}'")
