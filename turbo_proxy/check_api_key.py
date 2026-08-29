#!/usr/bin/env python3
"""
Check every API provider TurboProxy supports and report whether its key works.

For each provider it loads the key from the environment (or the ``.env`` next to
``turbo-proxy.yaml``), makes one minimal live request, and prints a status:

    ✅ working      key authenticated and the call succeeded
    ❌ failing      key is missing/expired/invalid (auth error)
    ⚠️  unverified   key looks set but the test call failed for another reason
    ⚪️ not set      no key in the environment

Vertex AI and DeepSeek checks for verifier/progress models also confirm token
logprobs are returned, since those scoring paths depend on them.

Usage:
    turbo-proxy check
"""

import asyncio
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, quote, quote_plus, unquote, urlsplit, urlunsplit

import httpx

from turbo_proxy.utils.config import (
    ModelConfig,
    load_yaml_mapping,
    resolve_api_key,
    resolve_base_url,
    is_official_openai_base_url,
    resolve_model_provider,
    resolve_optional_api_key,
    redact_base_url,
    split_model_name,
    validate_config_structure,
    validate_litellm_api_key,
    validate_litellm_endpoint,
)
from turbo_proxy.utils.verifier_client import (
    _append_url_path,
    _normalise_base_url,
    _openai_api_base_url,
    build_vertex_client,
    verifier_model_id,
)

# .env and turbo-proxy.yaml are resolved from the directory the user runs in.
ROOT = Path.cwd()

# Keywords that mark an error as an authentication / key problem (vs. anything
# else, like a wrong model name or a transient network error).
_AUTH_HINTS = (
    "api key", "api_key", "apikey", "unauthorized", "authentication",
    "auth failed", "auth failure", "invalid credential", "invalid credentials",
    "invalid token", "expired credential", "expired credentials",
    "credential expired", "credentials expired", "credential has expired",
    "credentials have expired", "credential is expired",
    "credentials are expired", "expired token", "token expired",
    "token has expired", "token is expired", "provided token is expired",
    "expired or revoked", "revoked token", "token revoked", "invalid_grant",
    "api_key_invalid", "permission denied", "permission_denied", "forbidden",
    "default credentials", "defaultcredentialserror", "adc requires",
    "google cloud project", "could not resolve project_id",
    "cannot resolve project_id", "credentials cannot be refreshed",
    "credential cannot be refreshed", "anonymous credentials cannot be",
    "could not refresh credentials",
)

_SCORE_LETTERS = tuple(chr(65 + i) for i in range(20))
_SCORE_CHOICES = _SCORE_LETTERS + tuple(f" {letter}" for letter in _SCORE_LETTERS)

# Status -> emoji
_EMOJI = {"ok": "✅", "fail": "❌", "warn": "⚠️ ", "skip": "⚪️"}
_LABEL = {"ok": "working", "fail": "failing", "warn": "unverified",
          "skip": "not set"}


def _looks_like_auth_error(message: str) -> bool:
    low = message.lower()
    if any(hint in low for hint in _AUTH_HINTS):
        return True
    # A status code is meaningful only when the surrounding text identifies it
    # as an HTTP/response/status code. Bare numbers such as token counts are
    # ordinary model/request errors and must remain unverified warnings.
    return bool(
        re.search(
            r"\b(?:http(?:\s+status|\s+error)?|status(?:\s+code)?|"
            r"response(?:\s+status)?|error\s+code)\s*[:=]?\s*(?:401|403)\b",
            low,
        )
    )


def _redact(text: str, secrets: Tuple[str, ...] = ()) -> str:
    """Hide anything that looks like a secret so the report is safe to share."""
    for secret in secrets:
        if secret:
            variants = {
                secret,
                quote(secret, safe=""),
                quote_plus(secret, safe=""),
            }
            for variant in sorted(variants, key=len, reverse=True):
                text = text.replace(variant, "<redacted>")
    return re.sub(r"(sk|key|AIza)[-_A-Za-z0-9]{6,}", "<redacted>", text)


def _redact_base_url_text(text: str, base_url: Any) -> str:
    """Remove configured URL credentials from endpoint error details."""
    if not isinstance(base_url, str) or not base_url.strip():
        return text
    endpoint = base_url.strip()
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        return text.replace(endpoint, "<redacted endpoint>")

    fragmentless = urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.query,
        "",
    ))
    for candidate in (endpoint, fragmentless):
        text = text.replace(candidate, str(redact_base_url(candidate)))

    secrets = [
        parsed.username,
        parsed.password,
        parsed.query,
        parsed.fragment,
    ]
    secrets.extend(
        value
        for _, value in parse_qsl(parsed.query, keep_blank_values=True)
        if value
    )
    decoded = [
        unquote(secret)
        for secret in secrets
        if isinstance(secret, str) and secret
    ]
    return _redact(
        text,
        tuple(secret for secret in (*secrets, *decoded) if secret),
    )


def _short_error(
    resp: httpx.Response,
    secrets: Tuple[str, ...] = (),
    base_url: Any = None,
) -> str:
    """One-line, redacted summary of an error response body."""
    detail = resp.text
    try:
        err = resp.json().get("error", {})
        detail = err.get("message", detail) if isinstance(err, dict) else detail
    except Exception:
        pass
    detail = _redact_base_url_text(
        " ".join(str(detail).split()), base_url
    )
    detail = _redact(detail, secrets)[:160]
    return f"HTTP {resp.status_code}: {detail}"


@dataclass
class CheckResult:
    name: str
    env_var: str
    status: str           # "ok" | "fail" | "warn" | "skip"
    detail: str
    roles: Tuple[str, ...] = ()

    def line(self) -> str:
        emoji = _EMOJI[self.status]
        label = _LABEL[self.status]
        role = f"  ({', '.join(self.roles)})" if self.roles else ""
        return (f"{emoji} {self.name:<14} [{self.env_var}] — "
                f"{label}: {self.detail}{role}")


class ProviderChecker:
    """Base class: resolve a key from the environment and validate it."""

    name: str = ""
    env_var: str = ""

    def __init__(
        self,
        roles: Tuple[str, ...] = (),
        *,
        api_key: Optional[str] = None,
        env_var: Optional[str] = None,
    ):
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError("api_key must be a string or null")
        self.roles = roles
        self._api_key = api_key
        if env_var is not None:
            self.env_var = env_var

    def get_key(self) -> Optional[str]:
        if self._api_key is not None:
            return self._api_key.strip() or None
        key = os.environ.get(self.env_var, "").strip()
        return key or None

    def validate(self, key: str) -> Tuple[str, str]:
        """Return (status, detail). Override per provider."""
        raise NotImplementedError

    def _redact_detail(
        self, detail: str, secrets: Tuple[str, ...] = ()
    ) -> str:
        detail = _redact_base_url_text(
            detail, getattr(self, "base_url", None)
        )
        return _redact(detail, secrets)

    def run(self) -> CheckResult:
        key = self.get_key()
        if not key:
            return CheckResult(self.name, self.env_var, "skip",
                               "no key in environment", self.roles)
        try:
            status, detail = self.validate(key)
            detail = _redact(detail, (key,))
        except Exception as e:  # never let one provider abort the whole script
            msg = self._redact_detail(
                " ".join(f"{type(e).__name__}: {e}".split()), (key,)
            )[:200]
            status = "fail" if _looks_like_auth_error(msg) else "warn"
            detail = msg
        return CheckResult(self.name, self.env_var, status, detail, self.roles)

    # -- shared helper for REST-based checks --------------------------------

    def _classify_http(
        self,
        resp: httpx.Response,
        ok_detail: str,
        secrets: Tuple[str, ...] = (),
    ) -> Tuple[str, str]:
        if resp.status_code == 200:
            return "ok", ok_detail
        summary = _short_error(
            resp, secrets, getattr(self, "base_url", None)
        )
        if resp.status_code in (401, 403) or _looks_like_auth_error(resp.text):
            return "fail", summary
        return "warn", summary


class UnsupportedCustomEndpointChecker(ProviderChecker):
    """Report an endpoint we cannot probe without guessing its protocol."""

    def __init__(
        self,
        roles: Tuple[str, ...],
        *,
        provider: str,
        api_key: str,
        env_var: str,
    ):
        super().__init__(roles, api_key=api_key, env_var=env_var)
        self.provider = provider
        self.name = f"{provider} custom"

    def validate(self, key: str) -> Tuple[str, str]:
        return (
            "warn",
            f"no protocol-specific endpoint check is available for {self.provider}",
        )


class GeminiChecker(ProviderChecker):
    """Google Gemini API (the backend's ``gemini/`` models use this)."""

    name = "Gemini"
    env_var = "GEMINI_API_KEY"

    def __init__(
        self,
        roles: Tuple[str, ...] = (),
        *,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        api_key: Optional[str] = None,
        env_var: str = "GEMINI_API_KEY",
        name: str = "Gemini",
    ):
        super().__init__(roles, api_key=api_key, env_var=env_var)
        self.base_url = base_url.strip()
        self.name = name

    def validate(self, key: str) -> Tuple[str, str]:
        resp = httpx.get(
            _append_url_path(self.base_url, "models"),
            headers={"x-goog-api-key": key},
            timeout=30.0,
        )
        return self._classify_http(resp, "models list reachable", (key,))


class OpenAIChecker(ProviderChecker):
    name = "OpenAI"
    env_var = "OPENAI_API_KEY"

    def __init__(
        self,
        roles: Tuple[str, ...] = (),
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        env_var: str = "OPENAI_API_KEY",
        name: str = "OpenAI",
    ):
        super().__init__(roles, api_key=api_key, env_var=env_var)
        self.base_url = _openai_api_base_url(base_url)
        self.name = name

    def validate(self, key: str) -> Tuple[str, str]:
        resp = httpx.get(
            _append_url_path(self.base_url, "models"),
            headers={"Authorization": f"Bearer {key}"},
            timeout=30.0,
        )
        return self._classify_http(resp, "models list reachable", (key,))


def _logprob_content(resp: httpx.Response) -> Tuple[dict, ...]:
    """Return the OpenAI ``logprobs.content`` positions, if present."""
    try:
        content = resp.json()["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError, ValueError):
        return ()
    return tuple(position for position in content if isinstance(position, dict))


def _openai_choice(resp: httpx.Response) -> Optional[dict]:
    """Return the first complete OpenAI choice, or ``None`` if malformed."""
    try:
        choices = resp.json()["choices"]
        choice = choices[0]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        return None
    return choice


def _score_letter_from_token(
    token: Any, tag: Optional[str] = None
) -> Optional[str]:
    """Extract a supported score letter from a tokenized output fragment."""
    token = str(token).strip()
    if token.startswith(">"):
        token = token[1:].strip()
    if len(token) == 1 and token.upper() in _SCORE_LETTERS:
        return token.upper()
    if tag:
        for marker in (tag, tag[:-1]):
            if not marker:
                continue
            marker_index = token.find(marker)
            if marker_index < 0:
                continue
            suffix = token[marker_index + len(marker):].lstrip(">").strip()
            if suffix and suffix[0].upper() in _SCORE_LETTERS:
                return suffix[0].upper()
    return None


def _has_score_letter_alternative(
    position: dict, tag: Optional[str] = None
) -> bool:
    alternatives = list(position.get("top_logprobs") or [])
    if position.get("token"):
        alternatives.insert(0, position)
    for alternative in alternatives:
        if not isinstance(alternative, dict):
            continue
        logprob = alternative.get("logprob")
        if (
            _score_letter_from_token(alternative.get("token", ""), tag)
            and isinstance(logprob, (int, float))
            and not isinstance(logprob, bool)
            and math.isfinite(logprob)
        ):
            return True
    return False


def _has_score_distribution(resp: httpx.Response) -> bool:
    positions = _logprob_content(resp)
    return bool(positions and _has_score_letter_alternative(positions[0]))


def _has_tagged_score_distribution(resp: httpx.Response, tag: str) -> bool:
    """Check the logprob position immediately after the last emitted tag."""
    positions = _logprob_content(resp)
    if not positions:
        return False
    candidate = None
    text = ""
    suffixes = tuple(suffix for suffix in (tag, tag[:-1]) if suffix)
    for index, position in enumerate(positions):
        token = str(position.get("token", ""))
        # Some tokenizers fuse the closing angle bracket and score letter into
        # the same token as the tag (for example ``<c1>A``). In that case the
        # distribution is on this position rather than the following one.
        if any(marker in token for marker in suffixes) and _has_score_letter_alternative(
            position, tag
        ):
            return True
        text += token
        if (
            index + 1 < len(positions)
            and any(text.rstrip().endswith(suffix) for suffix in suffixes)
        ):
            candidate = positions[index + 1]
    return candidate is not None and _has_score_letter_alternative(candidate, tag)


class OpenAILogprobChecker(OpenAIChecker):
    """Validate the score/progress token contract used by llm-verifier."""

    def __init__(
        self,
        *args: Any,
        model: str,
        profile: str,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if profile not in ("verifier", "progress"):
            raise ValueError(f"unknown OpenAI logprob check profile: {profile}")
        self.model = model
        self.profile = profile

    def _payload(self) -> dict:
        if self.profile == "verifier":
            # This is the prefill request used by llm-verifier's OpenAI path.
            return {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Score this answer."},
                    {"role": "assistant", "content": "Analysis.\n<score_A>"},
                ],
                "max_tokens": 1,
                "temperature": 1,
                "logprobs": True,
                "top_logprobs": 20,
                "add_generation_prompt": False,
                "continue_final_message": True,
                "structured_outputs": {"choice": list(_SCORE_CHOICES)},
            }
        return {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": "Reply with exactly <c1>A</c1> and nothing else.",
            }],
            "max_tokens": 8,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
        }

    def validate(self, key: str) -> Tuple[str, str]:
        resp = httpx.post(
            _append_url_path(self.base_url, "chat/completions"),
            headers={"Authorization": f"Bearer {key}"},
            json=self._payload(),
            timeout=30.0,
        )
        status, detail = self._classify_http(resp, "generation OK", (key,))
        if status != "ok":
            return status, detail
        if _openai_choice(resp) is None:
            return (
                "warn",
                "generation OK but response has no usable choices[0].message "
                f"({self.profile} verifier path may fail)",
            )
        has_distribution = (
            _has_score_distribution(resp)
            if self.profile == "verifier"
            else _has_tagged_score_distribution(resp, "<c1>")
        )
        if has_distribution:
            return "ok", f"{self.profile} score logprobs OK"
        return (
            "warn",
            "generation OK but no usable score-letter logprobs returned "
            f"({self.profile} verifier path may fall back to 0.5)",
        )


class DeepSeekLogprobChecker(OpenAILogprobChecker):
    """Check the DeepSeek-tagged path with the production reasoning params."""

    def _payload(self) -> dict:
        from llm_verifier.fine_grained_reward import deepseek_reasoning_params

        if self.profile == "verifier":
            content = (
                "Return exactly these two lines:\n"
                "<score_A> A </score_A>\n<score_B> B </score_B>"
            )
        else:
            content = "Return exactly <c1>A</c1> and nothing else."
        extra_body, max_tokens = deepseek_reasoning_params()
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 1,
            "logprobs": True,
            "top_logprobs": 20,
            **extra_body,
        }

    def validate(self, key: str) -> Tuple[str, str]:
        resp = httpx.post(
            _append_url_path(self.base_url, "chat/completions"),
            headers={"Authorization": f"Bearer {key}"},
            json=self._payload(),
            timeout=30.0,
        )
        status, detail = self._classify_http(resp, "generation OK", (key,))
        if status != "ok":
            return status, detail
        choice = _openai_choice(resp)
        if choice is None:
            return (
                "warn",
                "generation OK but response has no usable choices[0].message "
                f"(DeepSeek {self.profile} verifier path may fail)",
            )
        try:
            text = choice.get("message", {}).get("content", "")
        except (KeyError, IndexError, TypeError, ValueError):
            text = ""
        tags = (
            ("<score_A>", "<score_B>")
            if self.profile == "verifier" else ("<c1>",)
        )
        has_tags = all(tag in text for tag in tags)
        has_distributions = all(
            _has_tagged_score_distribution(resp, tag) for tag in tags
        )
        if has_tags and has_distributions:
            return "ok", f"DeepSeek {self.profile} score logprobs OK"
        return (
            "warn",
            "generation OK but DeepSeek score tags/logprobs were not "
            f"returned ({self.profile} verifier path may fail)",
        )


class AnthropicChecker(ProviderChecker):
    name = "Anthropic"
    env_var = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        roles: Tuple[str, ...] = (),
        *,
        base_url: str = "https://api.anthropic.com/v1",
        api_key: Optional[str] = None,
        env_var: str = "ANTHROPIC_API_KEY",
        name: str = "Anthropic",
    ):
        super().__init__(roles, api_key=api_key, env_var=env_var)
        self.base_url = base_url.strip()
        self.name = name

    def validate(self, key: str) -> Tuple[str, str]:
        resp = httpx.get(
            _append_url_path(self.base_url, "models"),
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            timeout=30.0,
        )
        return self._classify_http(resp, "models list reachable", (key,))


class DeepSeekChecker(ProviderChecker):
    """DeepSeek's hosted API — usable both as a backend model (``deepseek/``)
    and as the verifier, which needs token logprobs. A reachable models list
    is not enough, so this also confirms logprobs come back."""

    name = "DeepSeek"
    env_var = "DEEPSEEK_API_KEY"

    def validate(self, key: str) -> Tuple[str, str]:
        resp = httpx.post(
            _append_url_path(
                "https://api.deepseek.com/v1", "chat/completions"
            ),
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": 5,
            },
            timeout=30.0,
        )
        status, detail = self._classify_http(resp, "generation OK", (key,))
        if status != "ok":
            return status, detail
        try:
            logprobs = resp.json()["choices"][0].get("logprobs")
            has_logprobs = bool(logprobs and logprobs.get("content"))
        except Exception:
            has_logprobs = False
        if has_logprobs:
            return "ok", "generation + logprobs OK"
        return "warn", "generation OK but no logprobs returned (verifier needs them)"


class VertexChecker(ProviderChecker):
    """Vertex AI via google-genai — the verifier's logprob path.

    A successful generation alone is not enough: the verifier needs token
    logprobs, so this also confirms they come back."""

    name = "Vertex AI"
    env_var = "VERTEX_API_KEY"

    def __init__(
        self,
        roles: Tuple[str, ...] = (),
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        env_var: str = "VERTEX_API_KEY",
        name: str = "Vertex AI",
        model: str = "gemini-2.5-flash",
        allow_adc: bool = False,
        force_adc: bool = False,
    ):
        super().__init__(roles, api_key=api_key, env_var=env_var)
        self.base_url = base_url.strip() if base_url else None
        self.name = name
        self.model = model
        self.allow_adc = allow_adc or force_adc
        self.force_adc = force_adc

    def get_key(self) -> Optional[str]:
        if self.force_adc:
            return None
        return super().get_key()

    def run(self) -> CheckResult:
        key = self.get_key()
        if not self.allow_adc and not key:
            return super().run()
        try:
            # Passing None lets google-genai use ADC/project authentication.
            status, detail = self.validate(key)
            detail = _redact(detail, (key,) if key else ())
        except Exception as e:  # never let one provider abort the whole script
            msg = self._redact_detail(
                " ".join(f"{type(e).__name__}: {e}".split()),
                (key,) if key else (),
            )[:200]
            status = "fail" if _looks_like_auth_error(msg) else "warn"
            detail = msg
        return CheckResult(self.name, self.env_var, status, detail, self.roles)

    def validate(self, key: Optional[str]) -> Tuple[str, str]:
        from google.genai.types import (
            GenerateContentConfig,
            ThinkingConfig,
        )

        client = build_vertex_client(
            key,
            self.base_url,
            force_adc=self.force_adc,
        )
        resp = client.models.generate_content(
            model=self.model,
            contents="ping",
            config=GenerateContentConfig(
                max_output_tokens=1,
                temperature=0.0,
                response_logprobs=True,
                logprobs=5,
                thinking_config=ThinkingConfig(thinking_budget=0),
            ),
        )
        candidate = resp.candidates[0] # type: ignore
        has_logprobs = bool(
            getattr(candidate, "logprobs_result", None)
            and candidate.logprobs_result.top_candidates # type: ignore
        )
        if has_logprobs:
            return "ok", "generation + logprobs OK"
        return "warn", "generation OK but no logprobs returned (verifier needs them)"


class LiteLLMVertexChecker(ProviderChecker):
    """Probe Vertex backend/context models through the production wrapper."""

    name = "Vertex AI (LiteLLM)"
    env_var = "Vertex ADC"

    def __init__(
        self,
        roles: Tuple[str, ...] = (),
        *,
        model: str,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(roles, env_var=self.env_var)
        self.model = model
        self.provider = provider
        self.base_url = base_url

    def get_key(self) -> Optional[str]:
        return None

    def run(self) -> CheckResult:
        try:
            status, detail = self.validate("")
        except Exception as exc:  # keep one model from aborting the report
            msg = self._redact_detail(
                " ".join(f"{type(exc).__name__}: {exc}".split())
            )[:200]
            status = "fail" if _looks_like_auth_error(msg) else "warn"
            detail = msg
        return CheckResult(self.name, self.env_var, status, detail, self.roles)

    def validate(self, key: str) -> Tuple[str, str]:
        # Import lazily so checks that do not use LiteLLM Vertex avoid loading
        # its provider registry. This is the exact wrapper used in production.
        from turbo_proxy.utils.llm import llm_completion

        response = asyncio.run(asyncio.wait_for(
            llm_completion(
                model=self.model,
                provider=self.provider,
                base_url=self.base_url,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            ),
            timeout=30.0,
        ))
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            message = None
        if isinstance(message, dict):
            return "ok", "generation OK through LiteLLM"
        return "warn", "generation returned no usable choices[0].message"


class LiteLLMEndpointChecker(ProviderChecker):
    """Probe a keyed custom endpoint through the production LiteLLM wrapper."""

    def __init__(
        self,
        roles: Tuple[str, ...] = (),
        *,
        model: str,
        provider: Optional[str],
        base_url: str,
        api_key: str,
        env_var: str,
        name: str,
    ):
        super().__init__(roles, api_key=api_key, env_var=env_var)
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.name = name

    def validate(self, key: str) -> Tuple[str, str]:
        from turbo_proxy.utils.llm import llm_completion

        response = asyncio.run(asyncio.wait_for(
            llm_completion(
                model=self.model,
                provider=self.provider,
                base_url=self.base_url,
                api_key=key,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            ),
            timeout=30.0,
        ))
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            message = None
        if isinstance(message, dict):
            return "ok", "generation OK through LiteLLM"
        return "warn", "generation returned no usable choices[0].message"


def _config_usage() -> Tuple[dict, List[ProviderChecker]]:
    """Return provider roles and checks for explicitly configured credentials."""
    roles: Dict[str, List[str]] = {}
    endpoint_specs: Dict[
        Tuple[
            str,
            Optional[str],
            str,
            Optional[str],
            Optional[str],
            Optional[str],
            Optional[str],
        ],
        List[str],
    ] = {}
    endpoint_labels: Dict[
        Tuple[
            str,
            Optional[str],
            str,
            Optional[str],
            Optional[str],
            Optional[str],
            Optional[str],
        ],
        str,
    ] = {}
    provider_specs: Dict[Tuple[Any, str], List[str]] = {}
    provider_labels: Dict[Tuple[Any, str], str] = {}
    missing_provider_specs: Dict[Tuple[Any, str], List[str]] = {}
    protocol_specs: Dict[
        Tuple[str, Optional[str], str, str, str, bool], List[str]
    ] = {}
    protocol_labels: Dict[
        Tuple[str, Optional[str], str, str, str, bool], str
    ] = {}
    litellm_vertex_specs: Dict[
        Tuple[str, Optional[str], Optional[str]], List[str]
    ] = {}
    config_path = ROOT / "turbo-proxy.yaml"
    if not config_path.exists():
        return roles, []

    raw = load_yaml_mapping(config_path)
    validate_config_structure(raw)

    def append_role(mapping: dict, key: Any, role: str) -> None:
        values = mapping.setdefault(key, [])
        if role not in values:
            values.append(role)

    def checker_type(model: dict) -> Optional[Any]:
        name = model.get("name", "")
        provider = resolve_model_provider(name, model.get("provider"))
        if provider == "vertex_ai":
            return VertexChecker
        if provider == "gemini":
            return GeminiChecker
        if provider == "deepseek":
            return DeepSeekChecker
        if provider == "anthropic":
            return AnthropicChecker
        if provider == "openai":
            return OpenAIChecker
        return None

    def validate_api_key_type(value: Any, field_name: str) -> None:
        """Keep checker parsing consistent with Config's YAML scalar rules."""
        if value is not None and not isinstance(value, str):
            resolve_optional_api_key(value, field_name)

    def note_standard_model(model: dict, role: str) -> None:
        if not model.get("name"):
            return
        provider_checker = checker_type(model)
        if provider_checker is None:
            return

        api_key_ref = model.get("api_key", "")
        if isinstance(api_key_ref, str):
            api_key_ref = api_key_ref.strip()
        default_env_var = provider_checker.env_var
        requires_logprobs = role in ("verifier", "progress monitor")

        if isinstance(api_key_ref, str) and api_key_ref.startswith("$"):
            env_var = api_key_ref[1:]
            resolved_key = os.environ.get(env_var, "").strip() or None
        elif isinstance(api_key_ref, str) and api_key_ref.strip():
            env_var = "config api_key"
            resolved_key = api_key_ref.strip()
        else:
            env_var = default_env_var
            resolved_key = None

        if requires_logprobs:
            provider = resolve_model_provider(
                model.get("name", ""), model.get("provider")
            )
            model_id = split_model_name(model["name"])[0]
            profile = "progress" if role == "progress monitor" else "verifier"
            # Keep missing environment references distinct, while deduplicating
            # aliases that resolve to the same actual key.
            credential_id = resolved_key or f"${env_var}"
            allow_adc = not (
                isinstance(api_key_ref, str)
                and api_key_ref.startswith("$")
            )
            spec = (
                provider,
                resolved_key,
                credential_id,
                model_id,
                profile,
                allow_adc,
            )
            protocol_labels.setdefault(spec, env_var)
            append_role(protocol_specs, spec, role)
            return

        if not api_key_ref:
            append_role(roles, default_env_var, role)
            return

        if isinstance(api_key_ref, str) and api_key_ref.startswith("$"):
            env_var = api_key_ref[1:]
            if env_var == default_env_var:
                append_role(roles, env_var, role)
                return
            api_key = os.environ.get(env_var, "").strip()
            if not api_key:
                append_role(roles, env_var, role)
                append_role(
                    missing_provider_specs, (provider_checker, env_var), role
                )
                return
            label = env_var
        elif isinstance(api_key_ref, str):
            api_key = api_key_ref.strip()
            if not api_key:
                append_role(roles, default_env_var, role)
                return
            label = "config api_key"
        else:
            return

        provider_key = (provider_checker, api_key)
        provider_labels.setdefault(provider_key, label)
        append_role(provider_specs, provider_key, role)

    def note_model(model: Any, role: str, field_name: str) -> None:
        if model is None:
            return
        if not isinstance(model, dict):
            raise ValueError(f"{field_name} must be a mapping")
        if not model.get("name"):
            return
        api_key_ref = model.get("api_key", "")
        validate_api_key_type(api_key_ref, f"{field_name}.api_key")
        if isinstance(api_key_ref, str):
            api_key_ref = api_key_ref.strip()
        has_base_url = "base_url" in model
        raw_base_url = model.get("base_url")
        provider = resolve_model_provider(
            model.get("name", ""), model.get("provider"), raw_base_url
        )
        requires_logprobs = role in ("verifier", "progress monitor")
        if not requires_logprobs:
            validate_litellm_api_key(provider, api_key_ref, f"{field_name}.api_key")
            validate_litellm_endpoint(provider, has_base_url, field_name)
        if provider == "vertex_ai" and not requires_logprobs:
            # Backend/context traffic goes through LiteLLM and ADC, not the
            # google-genai client used by verifier/progress. Probe that exact
            # route so the URL shape and auth behavior cannot diverge.
            resolve_optional_api_key(api_key_ref, f"{field_name}.api_key")
            base_url = (
                resolve_base_url(raw_base_url, f"{field_name}.base_url")
                if has_base_url else None
            )
            append_role(
                litellm_vertex_specs,
                (model["name"], model.get("provider"), base_url),
                role,
            )
            return
        if has_base_url:
            base_url = _normalise_base_url(
                resolve_base_url(raw_base_url, f"{field_name}.base_url")
            )
            api_key = (
                resolve_optional_api_key(
                    api_key_ref, f"{field_name}.api_key"
                ) or None
                if provider == "vertex_ai"
                else resolve_api_key(api_key_ref, f"{field_name}.api_key")
            )
            if requires_logprobs:
                verifier_model_id(
                    ModelConfig(
                        name=model["name"],
                        provider=model.get("provider"),
                        api_key=api_key,
                        base_url=base_url,
                    ),
                    purpose=(
                        "progress" if role == "progress monitor" else "verifier"
                    ),
                )
            check_kind = (
                "logprobs"
                if requires_logprobs
                else "completion"
                if provider in ("gemini", "anthropic")
                else "models"
            )
            model_id = (
                split_model_name(model.get("name", ""))[0]
                if requires_logprobs or provider == "vertex_ai"
                else model["name"]
                if check_kind == "completion"
                else None
            )
            profile = (
                "progress" if role == "progress monitor" else "verifier"
            ) if requires_logprobs else None
            endpoint_key = (
                base_url,
                api_key,
                check_kind,
                model_id,
                profile,
                provider,
                model.get("provider") if check_kind == "completion" else None,
            )
            if isinstance(api_key_ref, str) and api_key_ref.startswith("$"):
                endpoint_labels.setdefault(endpoint_key, api_key_ref[1:])
            elif provider == "vertex_ai" and not api_key:
                # google-genai logprob checks may use VERTEX_API_KEY when no
                # per-model key is configured.  LiteLLM backend/context checks
                # are forced to ADC below, so keep their label separate.
                endpoint_labels.setdefault(
                    endpoint_key,
                    "VERTEX_API_KEY" if check_kind == "logprobs" else "Vertex ADC",
                )
            else:
                endpoint_labels.setdefault(endpoint_key, "config api_key")
            append_role(endpoint_specs, endpoint_key, role)
        else:
            if requires_logprobs:
                verifier_model_id(
                    ModelConfig(
                        name=model["name"],
                        provider=model.get("provider"),
                        api_key=api_key_ref or None,
                    ),
                    purpose=(
                        "progress" if role == "progress monitor" else "verifier"
                    ),
                )
            note_standard_model(model, role)

    backend = raw.get("backend", {})
    if not isinstance(backend, dict):
        raise ValueError("backend must be a mapping")
    models = backend.get("models", [])
    if not isinstance(models, list):
        raise ValueError("backend.models must be a list")
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"backend.models[{index}] must be a mapping")
        if not model.get("name"):
            raise ValueError(
                f"backend.models[{index}].name must be a non-empty string"
            )
        note_model(model, "backend", f"backend.models[{index}]")

    verifier = raw.get("verifier", {})
    if verifier is None:
        verifier = {}
    if not isinstance(verifier, dict):
        raise ValueError("verifier must be a mapping")
    verifier_model = verifier.get("model", {})
    note_model(verifier_model, "verifier", "verifier.model")

    progress = raw.get("progress_monitor", {})
    if progress is None:
        progress = {}
    if not isinstance(progress, dict):
        raise ValueError("progress_monitor must be a mapping")
    progress_model = progress.get("model", {})
    note_model(progress_model, "progress monitor", "progress_monitor.model")

    context = raw.get("context", {})
    if context is None:
        context = {}
    if not isinstance(context, dict):
        raise ValueError("context must be a mapping")
    # Config.context_config uses the refinement prompt as the feature gate.
    # Do not report credentials for a model that runtime will never invoke.
    context_model = (
        context.get("refinement_model", {})
        if context.get("refinement_prompt")
        else {}
    )
    note_model(context_model, "context", "context.refinement_model")

    custom_checkers = []
    for endpoint_key, endpoint_roles in endpoint_specs.items():
        (
            base_url,
            api_key,
            check_kind,
            model_id,
            profile,
            provider,
            configured_provider,
        ) = endpoint_key
        if check_kind == "logprobs" and profile is None:
            raise ValueError("logprob endpoint is missing its check profile")
        common_kwargs = {
            "base_url": base_url,
            "api_key": api_key,
            "env_var": endpoint_labels[endpoint_key],
        }
        if check_kind == "logprobs":
            if provider == "vertex_ai":
                custom_checkers.append(VertexChecker(
                    tuple(endpoint_roles),
                    model=model_id or "gemini-2.5-flash",
                    name="Vertex-compatible",
                    allow_adc=True,
                    **common_kwargs,
                ))
            else:
                checker_cls = (
                    DeepSeekLogprobChecker
                    if provider == "deepseek" else OpenAILogprobChecker
                )
                endpoint_name = (
                    "DeepSeek-compatible"
                    if provider == "deepseek" else "OpenAI-compatible"
                )
                custom_checkers.append(checker_cls(
                    tuple(endpoint_roles),
                    model=model_id,
                    profile=profile,
                    name=endpoint_name,
                    **common_kwargs,
                ))
        elif check_kind == "completion":
            if not model_id or provider not in ("gemini", "anthropic"):
                raise ValueError("LiteLLM endpoint check is missing its route")
            endpoint_name = (
                "Gemini-compatible"
                if provider == "gemini" else "Anthropic-compatible"
            )
            custom_checkers.append(LiteLLMEndpointChecker(
                tuple(endpoint_roles),
                model=model_id,
                provider=configured_provider,
                name=endpoint_name,
                **common_kwargs,
            ))
        elif provider in ("openai", "deepseek"):
            endpoint_name = (
                "DeepSeek-compatible"
                if provider == "deepseek" else "OpenAI-compatible"
            )
            custom_checkers.append(OpenAIChecker(
                tuple(endpoint_roles),
                name=endpoint_name,
                **common_kwargs,
            ))
        else:
            custom_checkers.append(UnsupportedCustomEndpointChecker(
                tuple(endpoint_roles),
                provider=provider or "unknown",
                api_key=api_key,
                env_var=endpoint_labels[endpoint_key],
            ))
    for protocol_spec, protocol_roles in protocol_specs.items():
        (
            provider,
            api_key,
            credential_id,
            model_id,
            profile,
            allow_adc,
        ) = protocol_spec
        env_var = protocol_labels[protocol_spec]
        common_kwargs = {
            "api_key": api_key,
            "env_var": env_var,
            "model": model_id,
            "profile": profile,
        }
        if provider == "deepseek":
            custom_checkers.append(DeepSeekLogprobChecker(
                tuple(protocol_roles),
                base_url="https://api.deepseek.com",
                name="DeepSeek",
                **common_kwargs,
            ))
        elif provider == "openai":
            configured_base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
            custom_checkers.append(OpenAILogprobChecker(
                tuple(protocol_roles),
                base_url=_normalise_base_url(
                    configured_base_url or "https://api.openai.com/v1"
                ),
                name="OpenAI",
                **common_kwargs,
            ))
        elif provider == "vertex_ai":
            custom_checkers.append(VertexChecker(
                tuple(protocol_roles),
                api_key=api_key,
                env_var=env_var,
                model=model_id,
                allow_adc=allow_adc,
            ))
        else:
            raise ValueError(
                f"no protocol checker is available for provider '{provider}'"
            )
    for (provider_checker, api_key), provider_roles in provider_specs.items():
        checker_kwargs = {
            "api_key": api_key,
            "env_var": provider_labels[(provider_checker, api_key)],
        }
        if provider_checker is VertexChecker:
            checker_kwargs["allow_adc"] = True
        custom_checkers.append(
            provider_checker(
                tuple(provider_roles),
                **checker_kwargs,
            )
        )
    for (provider_checker, env_var), provider_roles in missing_provider_specs.items():
        checker_kwargs = {"env_var": env_var}
        if provider_checker is VertexChecker:
            checker_kwargs["allow_adc"] = True
        custom_checkers.append(
            provider_checker(tuple(provider_roles), **checker_kwargs)
        )
    for vertex_spec, vertex_roles in litellm_vertex_specs.items():
        model, configured_provider, base_url = vertex_spec
        custom_checkers.append(LiteLLMVertexChecker(
            tuple(vertex_roles),
            model=model,
            provider=configured_provider,
            base_url=base_url,
        ))
    return roles, custom_checkers


def _roles_from_config() -> dict:
    """Map env-var name to the non-custom config roles that reference it."""
    roles, _ = _config_usage()
    return roles


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(str(env_path), override=False)
    except Exception:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(
                    k.strip(), v.strip().strip('"').strip("'")
                )


def main() -> int:
    _load_dotenv()
    try:
        config_roles, custom_checkers = _config_usage()
    except ValueError as exc:
        print(f"Configuration error: {_redact(str(exc))}", file=sys.stderr)
        return 1

    gemini = GeminiChecker()
    vertex = VertexChecker()
    deepseek = DeepSeekChecker()
    openai = OpenAIChecker()
    anthropic = AnthropicChecker()
    standard_checkers = [gemini, vertex, deepseek, openai, anthropic]
    for c in standard_checkers:
        c.roles = tuple(config_roles.get(c.env_var, ()))

    def include_official(checker: ProviderChecker) -> bool:
        official_key = checker.get_key()
        key_is_configured_elsewhere = any(
            custom.env_var == checker.env_var
            or (official_key and custom.get_key() == official_key)
            for custom in custom_checkers
        )
        return bool(checker.roles) or not key_is_configured_elsewhere

    checkers: List[ProviderChecker] = [
        checker for checker in (gemini, vertex, deepseek)
        if include_official(checker)
    ]
    checkers.extend(custom_checkers)
    if include_official(openai):
        checkers.append(openai)
    if include_official(anthropic):
        checkers.append(anthropic)

    print("Checking TurboProxy provider API keys\n" + "─" * 60)
    results = [c.run() for c in checkers]
    for r in results:
        print(r.line())

    print("─" * 60)
    n_ok = sum(r.status == "ok" for r in results)
    n_fail = sum(r.status == "fail" for r in results)
    n_warn = sum(r.status == "warn" for r in results)
    n_skip = sum(r.status == "skip" for r in results)
    print(f"Summary: {n_ok} ✅  {n_fail} ❌  {n_warn} ⚠️   {n_skip} ⚪️")

    # Exit non-zero if a configured key is missing/rejected, or if a scoring
    # role's required protocol could not be verified.
    configured_fail = any(
        r.roles and (
            r.status in ("fail", "skip")
            or (
                r.status == "warn"
                and any(
                    role in ("verifier", "progress monitor")
                    for role in r.roles
                )
            )
        )
        for r in results
    )
    return 1 if configured_fail else 0


if __name__ == "__main__":
    sys.exit(main())
