"""Regression tests for custom OpenAI-compatible API checks."""

from collections import Counter
from types import SimpleNamespace

import httpx
import pytest
import yaml

import turbo_agent.check_api_key as check_api_key


_PROVIDER_ENV_VARS = (
    "GEMINI_API_KEY",
    "VERTEX_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolate_provider_environment(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("VERTEX_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("DEEPSEEK_EFFORT", raising=False)
    monkeypatch.delenv("DEEPSEEK_MAX_TOKENS", raising=False)


def _configure(tmp_path, monkeypatch, models):
    (tmp_path / "turbo-agent.yaml").write_text(
        yaml.safe_dump({"backend": {"models": models}})
    )
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    for env_var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


def _record_gets(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((str(url), kwargs))
        return httpx.Response(
            200,
            json={"data": []},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(check_api_key.httpx, "get", fake_get)
    return calls


def _chat_response(url, text, positions):
    return httpx.Response(
        200,
        json={
            "choices": [{
                "message": {"content": text},
                "logprobs": {"content": positions},
            }]
        },
        request=httpx.Request("POST", url),
    )


def _letter_position(token="A"):
    return {
        "token": token,
        "logprob": -0.1,
        "top_logprobs": [
            {"token": token, "logprob": -0.1},
            {"token": "B", "logprob": -2.0},
        ],
    }


def _isolate_deepseek_reasoning(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_EFFORT", raising=False)
    monkeypatch.delenv("DEEPSEEK_MAX_TOKENS", raising=False)
    monkeypatch.setattr(
        "llm_verifier.fine_grained_reward.load_dotenv",
        lambda: None,
    )


def test_main_checks_custom_openai_endpoint_with_its_model_key(
    tmp_path, monkeypatch,
):
    custom_key = "custom-only-secret"
    _configure(
        tmp_path,
        monkeypatch,
        [
            {
                "name": "openai/local-model",
                "base_url": "https://gateway.example/openai/v1/",
                "api_key": "$OPENAI_API_KEY",
            }
        ],
    )
    monkeypatch.setenv("OPENAI_API_KEY", custom_key)
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 0

    assert calls == [
        (
            "https://gateway.example/openai/v1/models",
            {
                "headers": {"Authorization": f"Bearer {custom_key}"},
                "timeout": 30.0,
            },
        )
    ]
    assert all(url != "https://api.openai.com/v1/models" for url, _ in calls)


def test_custom_checker_appends_models_after_path_before_query_and_fragment(
    monkeypatch,
):
    calls = _record_gets(monkeypatch)
    checker = check_api_key.OpenAIChecker(
        base_url="https://gateway.example/root/v1/?tenant=acme#route",
        api_key="custom-secret",
    )

    result = checker.run()

    assert result.status == "ok"
    assert calls == [(
        "https://gateway.example/root/v1/models?tenant=acme#route",
        {
            "headers": {"Authorization": "Bearer custom-secret"},
            "timeout": 30.0,
        },
    )]


def test_custom_gemini_checker_preserves_base_query(monkeypatch):
    requests = []

    def fake_get(url, **kwargs):
        kwargs.pop("timeout", None)
        request = httpx.Request("GET", url, **kwargs)
        requests.append(request)
        return httpx.Response(200, json={"models": []}, request=request)

    monkeypatch.setattr(check_api_key.httpx, "get", fake_get)
    checker = check_api_key.GeminiChecker(
        base_url="https://gateway.example/root/v1beta?tenant=acme#route",
        api_key="custom-secret",
    )

    result = checker.run()

    assert result.status == "ok"
    assert [request.url.raw_path for request in requests] == [
        b"/root/v1beta/models?tenant=acme"
    ]


def test_custom_checker_uses_api_root_from_full_chat_completions_url(monkeypatch):
    calls = _record_gets(monkeypatch)
    checker = check_api_key.OpenAIChecker(
        base_url=(
            "https://gateway.example/root/v1/chat/completions/"
            "?tenant=acme#route"
        ),
        api_key="custom-secret",
    )

    result = checker.run()

    assert result.status == "ok"
    assert calls == [(
        "https://gateway.example/root/v1/models?tenant=acme#route",
        {
            "headers": {"Authorization": "Bearer custom-secret"},
            "timeout": 30.0,
        },
    )]


@pytest.mark.parametrize(
    ("role", "model", "provider", "base_url", "checker_name"),
    [
        (
            "backend",
            "gemini/gemini-2.5-flash",
            None,
            (
                "https://gemini-proxy.example/v1beta/models/"
                "gemini-2.5-flash:generateContent?tenant=acme#route"
            ),
            "Gemini-compatible",
        ),
        (
            "context",
            "gemini-2.5-flash",
            "gemini",
            "https://gemini-proxy.example/v1beta",
            "Gemini-compatible",
        ),
        (
            "backend",
            "anthropic/claude-sonnet-4-5",
            None,
            (
                "https://anthropic-proxy.example/v1/messages"
                "?tenant=acme#route"
            ),
            "Anthropic-compatible",
        ),
        (
            "context",
            "claude-sonnet-4-5",
            "anthropic",
            "https://anthropic-proxy.example/v1",
            "Anthropic-compatible",
        ),
    ],
)
def test_custom_litellm_endpoint_uses_the_production_completion_route(
    tmp_path,
    monkeypatch,
    role,
    model,
    provider,
    base_url,
    checker_name,
):
    model_config = {
        "name": model,
        "base_url": base_url,
        "api_key": "custom-secret",
    }
    if provider is not None:
        model_config["provider"] = provider
    raw_config = {
        "backend": {
            "models": [
                model_config
                if role == "backend"
                else {"name": "openai/gpt-4o"}
            ]
        }
    }
    if role == "context":
        raw_config["context"] = {
            "refinement_prompt": "Refine this context.",
            "refinement_model": model_config,
        }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    calls = []

    async def fake_completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(
        "turbo_agent.utils.llm.llm_completion",
        fake_completion,
    )
    monkeypatch.setattr(
        check_api_key.httpx,
        "get",
        lambda *args, **kwargs: pytest.fail(
            "custom completion probes must not require a models endpoint"
        ),
    )

    _, checkers = check_api_key._config_usage()

    assert len(checkers) == 1
    checker = checkers[0]
    assert isinstance(checker, check_api_key.LiteLLMEndpointChecker)
    assert checker.name == checker_name
    assert checker.roles == (role,)
    assert checker.run().status == "ok"
    assert calls == [{
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "api_key": "custom-secret",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }]


def test_unknown_custom_provider_is_not_probed_with_a_guessed_protocol(
    tmp_path, monkeypatch, capsys,
):
    _configure(
        tmp_path,
        monkeypatch,
        [{
            "name": "served-model",
            "provider": "custom_vendor",
            "base_url": "https://custom.example/v1",
            "api_key": "custom-secret",
        }],
    )

    def unexpected_http(*args, **kwargs):
        pytest.fail("unknown providers must not be probed over a guessed protocol")

    monkeypatch.setattr(check_api_key.httpx, "get", unexpected_http)
    monkeypatch.setattr(check_api_key.httpx, "post", unexpected_http)

    assert check_api_key.main() == 0

    output = capsys.readouterr().out
    assert "custom_vendor custom" in output
    assert "unverified" in output


def test_main_deduplicates_custom_endpoints_and_checks_each_distinct_endpoint(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("CUSTOM_OPENAI_KEY_A", "secret-a")
    monkeypatch.setenv("CUSTOM_OPENAI_KEY_B", "secret-b")
    _configure(
        tmp_path,
        monkeypatch,
        [
            {
                "name": "openai/model-a",
                "base_url": "https://first.example/v1/",
                "api_key": "$CUSTOM_OPENAI_KEY_A",
            },
            {
                "name": "openai/model-a-alternate",
                "base_url": "https://first.example/v1",
                "api_key": "$CUSTOM_OPENAI_KEY_A",
            },
            {
                "name": "openai/model-b",
                "base_url": "https://second.example/api/v1",
                "api_key": "$CUSTOM_OPENAI_KEY_B",
            },
        ],
    )
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 0

    assert Counter(url for url, _ in calls) == Counter(
        {
            "https://first.example/v1/models": 1,
            "https://second.example/api/v1/models": 1,
        }
    )
    assert {
        url: kwargs["headers"]["Authorization"] for url, kwargs in calls
    } == {
        "https://first.example/v1/models": "Bearer secret-a",
        "https://second.example/api/v1/models": "Bearer secret-b",
    }
    assert all(url != "https://api.openai.com/v1/models" for url, _ in calls)


def test_main_deduplicates_custom_endpoint_by_resolved_key(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("CUSTOM_KEY_ALIAS_A", "shared-secret")
    monkeypatch.setenv("CUSTOM_KEY_ALIAS_B", "shared-secret")
    _configure(
        tmp_path,
        monkeypatch,
        [
            {
                "name": "openai/model-a",
                "base_url": "https://gateway.example/v1/",
                "api_key": "$CUSTOM_KEY_ALIAS_A",
            },
            {
                "name": "openai/model-b",
                "base_url": "https://gateway.example/v1",
                "api_key": "$CUSTOM_KEY_ALIAS_B",
            },
        ],
    )
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 0
    assert [url for url, _ in calls] == [
        "https://gateway.example/v1/models"
    ]


@pytest.mark.parametrize(
    ("api_key_config", "env_var"),
    [
        ("literal-official-secret", None),
        ("$MY_OPENAI_KEY", "MY_OPENAI_KEY"),
    ],
)
def test_main_checks_standard_openai_endpoint_with_configured_key(
    tmp_path, monkeypatch, api_key_config, env_var,
):
    if env_var:
        monkeypatch.setenv(env_var, "alternate-env-secret")
        expected_key = "alternate-env-secret"
    else:
        expected_key = api_key_config
    _configure(
        tmp_path,
        monkeypatch,
        [{"name": "openai/gpt-4o", "api_key": api_key_config}],
    )
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 0
    assert calls == [
        (
            "https://api.openai.com/v1/models",
            {
                "headers": {"Authorization": f"Bearer {expected_key}"},
                "timeout": 30.0,
            },
        )
    ]


def test_prefixless_known_model_is_associated_with_its_provider_role(
    tmp_path, monkeypatch,
):
    _configure(
        tmp_path,
        monkeypatch,
        [{"name": "gpt-4o", "api_key": "$OPENAI_API_KEY"}],
    )

    roles, checkers = check_api_key._config_usage()

    assert roles == {"OPENAI_API_KEY": ["backend"]}
    assert checkers == []


def test_main_checks_verifier_and_progress_monitor_custom_endpoints(
    tmp_path, monkeypatch,
):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "verifier": {
            "model": {
                "name": "openai/verifier-model",
                "base_url": "https://verifier.example/v1",
                "api_key": "$VERIFIER_ENDPOINT_KEY",
            }
        },
        "progress_monitor": {
            "model": {
                "name": "openai/progress-model",
                "base_url": "https://progress.example/v1/",
                "api_key": "$PROGRESS_ENDPOINT_KEY",
            }
        },
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    for env_var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("VERIFIER_ENDPOINT_KEY", "verifier-secret")
    monkeypatch.setenv("PROGRESS_ENDPOINT_KEY", "progress-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "backend-secret")
    gets = _record_gets(monkeypatch)
    calls = []

    def fake_post(url, **kwargs):
        calls.append((str(url), kwargs))
        if "verifier.example" in str(url):
            return _chat_response(url, "A", [_letter_position()])
        return _chat_response(
            url,
            "<c1>A</c1>",
            [
                {"token": "<c1>", "logprob": 0, "top_logprobs": []},
                _letter_position(),
                {"token": "</c1>", "logprob": 0, "top_logprobs": []},
            ],
        )

    monkeypatch.setattr(check_api_key.httpx, "post", fake_post)

    assert check_api_key.main() == 0
    assert [url for url, _ in gets] == [
        "https://api.openai.com/v1/models"
    ]

    assert {
        url: (kwargs["headers"]["Authorization"], kwargs["json"]["model"])
        for url, kwargs in calls
    } == {
        "https://verifier.example/v1/chat/completions": (
            "Bearer verifier-secret",
            "verifier-model",
        ),
        "https://progress.example/v1/chat/completions": (
            "Bearer progress-secret",
            "progress-model",
        ),
    }
    payloads = {url: kwargs["json"] for url, kwargs in calls}
    verifier_payload = payloads[
        "https://verifier.example/v1/chat/completions"
    ]
    assert verifier_payload["continue_final_message"] is True
    assert verifier_payload["add_generation_prompt"] is False
    assert verifier_payload["structured_outputs"]["choice"][:2] == ["A", "B"]
    assert verifier_payload["logprobs"] is True
    assert verifier_payload["top_logprobs"] == 20
    progress_payload = payloads[
        "https://progress.example/v1/chat/completions"
    ]
    assert progress_payload["logprobs"] is True
    assert progress_payload["top_logprobs"] == 20
    assert "<c1>" in progress_payload["messages"][0]["content"]


def test_standard_deepseek_verifier_checks_configured_score_protocol(
    tmp_path, monkeypatch,
):
    _isolate_deepseek_reasoning(monkeypatch)
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "verifier": {
            "model": {
                "name": "deepseek/custom-reasoner",
                "api_key": "$DEEPSEEK_API_KEY",
            }
        },
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    for env_var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "backend-secret")
    gets = _record_gets(monkeypatch)
    calls = []

    def fake_post(url, **kwargs):
        calls.append((str(url), kwargs))
        return _chat_response(
            url,
            "<score_A>A</score_A><score_B>B</score_B>",
            [
                {"token": "<score_A>", "logprob": 0, "top_logprobs": []},
                _letter_position(),
                {"token": "</score_A>", "logprob": 0, "top_logprobs": []},
                {"token": "<score_B>", "logprob": 0, "top_logprobs": []},
                _letter_position("B"),
                {"token": "</score_B>", "logprob": 0, "top_logprobs": []},
            ],
        )

    monkeypatch.setattr(check_api_key.httpx, "post", fake_post)

    assert check_api_key.main() == 0
    assert [url for url, _ in gets] == [
        "https://api.openai.com/v1/models"
    ]
    assert [url for url, _ in calls] == [
        "https://api.deepseek.com/chat/completions"
    ]
    payload = calls[0][1]["json"]
    assert payload["model"] == "custom-reasoner"
    assert "<score_A>" in payload["messages"][0]["content"]
    assert payload["logprobs"] is True
    assert payload["max_tokens"] == 32768
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    ("effort", "max_tokens", "thinking", "reasoning_effort"),
    [
        ("off", "321", {"type": "disabled"}, None),
        ("max", "654", {"type": "enabled"}, "max"),
    ],
)
def test_deepseek_logprob_checker_honors_production_reasoning_environment(
    monkeypatch,
    effort,
    max_tokens,
    thinking,
    reasoning_effort,
):
    _isolate_deepseek_reasoning(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_EFFORT", effort)
    monkeypatch.setenv("DEEPSEEK_MAX_TOKENS", max_tokens)
    checker = check_api_key.DeepSeekLogprobChecker(
        model="custom-reasoner",
        profile="verifier",
        base_url="https://api.deepseek.com",
        api_key="deepseek-secret",
    )

    payload = checker._payload()

    assert payload["max_tokens"] == int(max_tokens)
    assert payload["thinking"] == thinking
    if reasoning_effort is None:
        assert "reasoning_effort" not in payload
    else:
        assert payload["reasoning_effort"] == reasoning_effort


def test_standard_openai_progress_checks_configured_score_protocol(
    tmp_path, monkeypatch,
):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "progress_monitor": {
            "model": {
                "name": "openai/gpt-4o",
                "api_key": "$OPENAI_API_KEY",
            }
        },
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    for env_var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    posts = []
    gets = _record_gets(monkeypatch)

    def fake_post(url, **kwargs):
        posts.append((str(url), kwargs))
        return _chat_response(
            url,
            "<c1>A</c1>",
            [
                {"token": "<c1>", "logprob": 0, "top_logprobs": []},
                _letter_position(),
                {"token": "</c1>", "logprob": 0, "top_logprobs": []},
            ],
        )

    monkeypatch.setattr(check_api_key.httpx, "post", fake_post)

    assert check_api_key.main() == 0
    assert [url for url, _ in gets] == [
        "https://api.openai.com/v1/models"
    ]
    assert [url for url, _ in posts] == [
        "https://api.openai.com/v1/chat/completions"
    ]
    payload = posts[0][1]["json"]
    assert payload["model"] == "gpt-4o"
    assert "<c1>" in payload["messages"][0]["content"]


def test_standard_vertex_checker_uses_the_configured_model(
    tmp_path, monkeypatch,
):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "verifier": {
            "model": {
                "name": "gemini/gemini-2.5-pro",
                "provider": "vertex_ai",
                "api_key": "$VERTEX_API_KEY",
            }
        },
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    monkeypatch.setenv("VERTEX_API_KEY", "vertex-secret")

    roles, checkers = check_api_key._config_usage()

    assert roles == {"OPENAI_API_KEY": ["backend"]}
    assert len(checkers) == 1
    assert isinstance(checkers[0], check_api_key.VertexChecker)
    assert checkers[0].model == "gemini-2.5-pro"
    assert checkers[0].roles == ("verifier",)


def test_custom_vertex_logprob_checker_allows_adc_without_api_key(
    tmp_path, monkeypatch,
):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "verifier": {
            "model": {
                "name": "vertex_ai/gemini-2.5-pro",
                "provider": "vertex_ai",
                "base_url": "https://vertex.example/v1",
            }
        },
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)

    roles, checkers = check_api_key._config_usage()

    assert roles == {"OPENAI_API_KEY": ["backend"]}
    assert len(checkers) == 1
    checker = checkers[0]
    assert isinstance(checker, check_api_key.VertexChecker)
    assert checker.allow_adc is True
    assert checker.base_url == "https://vertex.example/v1"
    assert checker.get_key() is None


def test_vertex_checker_uses_adc_and_custom_base_url_without_api_key(monkeypatch):
    from google.auth.credentials import AnonymousCredentials
    import google.genai as genai

    calls = {}

    class FakeClient:
        def __init__(self, **kwargs):
            calls.update(kwargs)
            self.models = SimpleNamespace(
                generate_content=lambda **kwargs: SimpleNamespace(
                    candidates=[SimpleNamespace(
                        logprobs_result=SimpleNamespace(top_candidates=["A"])
                    )]
                )
            )

    monkeypatch.delenv("MISSING_VERTEX_ADC_KEY", raising=False)
    credentials = AnonymousCredentials()
    monkeypatch.setattr(
        "google.auth.default",
        lambda **kwargs: (credentials, "adc-project"),
    )
    monkeypatch.setattr(genai, "Client", FakeClient)
    checker = check_api_key.VertexChecker(
        ("verifier",),
        base_url="https://vertex.example/v1",
        env_var="MISSING_VERTEX_ADC_KEY",
        model="gemini-2.5-pro",
        allow_adc=True,
    )

    result = checker.run()

    assert result.status == "ok"
    assert calls["vertexai"] is True
    assert calls["credentials"] is credentials
    assert calls["project"] == "adc-project"
    assert calls["location"] == "global"
    assert "api_key" not in calls
    assert calls["http_options"].base_url == "https://vertex.example/v1"


def test_custom_vertex_logprob_checker_uses_vertex_key_and_deduplicates(
    tmp_path, monkeypatch,
):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "verifier": {
            "model": {
                "name": "vertex_ai/gemini-2.5-pro",
                "provider": "vertex_ai",
                "base_url": "https://vertex.example/v1",
            }
        },
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    for env_var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("VERTEX_API_KEY", "vertex-secret")

    _, checkers = check_api_key._config_usage()

    assert len(checkers) == 1
    checker = checkers[0]
    assert isinstance(checker, check_api_key.VertexChecker)
    assert checker.env_var == "VERTEX_API_KEY"
    assert checker.get_key() == "vertex-secret"
    assert checker.force_adc is False

    # The custom checker owns the configured Vertex credential, so main must
    # not probe the official endpoint a second time.
    monkeypatch.setattr(check_api_key, "_config_usage", lambda: ({}, checkers))
    monkeypatch.setattr(check_api_key, "_load_dotenv", lambda: None)
    monkeypatch.setattr(check_api_key.VertexChecker, "validate", lambda self, key: ("ok", "ok"))
    assert check_api_key.main() == 0


def test_custom_vertex_backend_checker_forces_adc(
    tmp_path, monkeypatch,
):
    raw_config = {
        "backend": {
            "models": [{
                "name": "vertex_ai/gemini-2.5-flash",
                "provider": "vertex_ai",
                "base_url": "https://vertex.example/v1",
            }]
        }
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    monkeypatch.setenv("VERTEX_API_KEY", "must-not-be-used-by-backend")

    _, checkers = check_api_key._config_usage()

    assert len(checkers) == 1
    checker = checkers[0]
    assert isinstance(checker, check_api_key.LiteLLMVertexChecker)
    assert checker.model == "vertex_ai/gemini-2.5-flash"
    assert checker.provider == "vertex_ai"
    assert checker.base_url == "https://vertex.example/v1"
    assert checker.get_key() is None


def test_standard_vertex_backend_whitespace_key_uses_litellm_adc(
    tmp_path, monkeypatch,
):
    raw_config = {
        "backend": {
            "models": [{
                "name": "gemini/gemini-2.5-flash",
                "provider": "vertex_ai",
                "api_key": "   ",
            }]
        }
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)

    roles, checkers = check_api_key._config_usage()

    assert roles == {}
    assert len(checkers) == 1
    checker = checkers[0]
    assert isinstance(checker, check_api_key.LiteLLMVertexChecker)
    assert checker.roles == ("backend",)
    assert checker.get_key() is None


def test_standard_backend_strips_environment_reference_whitespace(
    tmp_path, monkeypatch,
):
    _configure(
        tmp_path,
        monkeypatch,
        [{
            "name": "deepseek/deepseek-chat",
            "api_key": " $PADDED_KEY ",
        }],
    )
    monkeypatch.setenv("PADDED_KEY", "actual-secret")

    roles, checkers = check_api_key._config_usage()

    assert roles == {}
    assert len(checkers) == 1
    assert isinstance(checkers[0], check_api_key.DeepSeekChecker)
    assert checkers[0].env_var == "PADDED_KEY"
    assert checkers[0].get_key() == "actual-secret"


@pytest.mark.parametrize(
    ("model", "checker_cls"),
    [
        (
            {"name": "deepseek/custom-reasoner"},
            check_api_key.DeepSeekLogprobChecker,
        ),
        (
            {
                "name": "gemini/gemini-2.5-flash",
                "provider": "vertex_ai",
            },
            check_api_key.VertexChecker,
        ),
    ],
)
def test_standard_verifier_strips_environment_reference_whitespace(
    tmp_path, monkeypatch, model, checker_cls,
):
    model = {**model, "api_key": " $PADDED_KEY "}
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump({
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "verifier": {"model": model},
    }))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    for env_var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("PADDED_KEY", "actual-secret")

    roles, checkers = check_api_key._config_usage()

    assert roles == {"OPENAI_API_KEY": ["backend"]}
    assert len(checkers) == 1
    assert isinstance(checkers[0], checker_cls)
    assert checkers[0].env_var == "PADDED_KEY"
    assert checkers[0].get_key() == "actual-secret"
    if isinstance(checkers[0], check_api_key.VertexChecker):
        assert checkers[0].allow_adc is False


def test_litellm_vertex_checker_uses_the_production_completion_wrapper(
    monkeypatch,
):
    calls = []

    async def fake_completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(
        "turbo_agent.utils.llm.llm_completion", fake_completion
    )
    checker = check_api_key.LiteLLMVertexChecker(
        ("backend", "context"),
        model="gemini/gemini-2.5-flash",
        provider="vertex_ai",
        base_url="https://vertex.example/v1",
    )

    result = checker.run()

    assert result.status == "ok"
    assert calls == [{
        "model": "gemini/gemini-2.5-flash",
        "provider": "vertex_ai",
        "base_url": "https://vertex.example/v1",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }]


@pytest.mark.parametrize(
    "error",
    [
        "Could not resolve project_id",
        "Anonymous credentials cannot be refreshed",
    ],
)
def test_litellm_vertex_adc_setup_errors_are_failures(monkeypatch, error):
    checker = check_api_key.LiteLLMVertexChecker(
        ("backend",),
        model="vertex_ai/gemini-2.5-flash",
        provider="vertex_ai",
    )

    def fail_validation(_key):
        raise RuntimeError(error)

    monkeypatch.setattr(checker, "validate", fail_validation)

    result = checker.run()

    assert result.status == "fail"


def test_vertex_checker_without_allow_adc_keeps_missing_key_skip(monkeypatch):
    monkeypatch.delenv("MISSING_VERTEX_KEY", raising=False)
    checker = check_api_key.VertexChecker(
        env_var="MISSING_VERTEX_KEY",
        allow_adc=False,
    )

    result = checker.run()

    assert result.status == "skip"
    assert result.detail == "no key in environment"


def test_configured_vertex_adc_without_project_is_a_failure(monkeypatch):
    from google.auth.credentials import AnonymousCredentials

    monkeypatch.delenv("VERTEX_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setattr(
        "google.auth.default",
        lambda **kwargs: (AnonymousCredentials(), None),
    )
    checker = check_api_key.VertexChecker(
        ("backend",),
        env_var="Vertex ADC",
        allow_adc=True,
        force_adc=True,
    )

    result = checker.run()

    assert result.status == "fail"
    assert "Google Cloud project" in result.detail


def test_main_rejects_missing_base_url_environment_before_http(
    tmp_path, monkeypatch, capsys,
):
    missing_var = "MISSING_CUSTOM_BASE_URL"
    monkeypatch.delenv(missing_var, raising=False)
    _configure(
        tmp_path,
        monkeypatch,
        [
            {
                "name": "openai/local-model",
                "base_url": f"${missing_var}",
                "api_key": "$OPENAI_API_KEY",
            }
        ],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-sent")
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 1

    captured = capsys.readouterr()
    assert "backend.models[0].base_url" in captured.err
    assert f"${missing_var}" in captured.err
    assert calls == []


@pytest.mark.parametrize("base_url", [None, "", "   "])
def test_main_rejects_explicit_empty_base_url_before_http(
    tmp_path, monkeypatch, capsys, base_url,
):
    _configure(
        tmp_path,
        monkeypatch,
        [
            {
                "name": "openai/local-model",
                "base_url": base_url,
                "api_key": "must-not-be-sent",
            }
        ],
    )
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 1
    assert "backend.models[0].base_url" in capsys.readouterr().err
    assert calls == []


def test_main_requires_base_url_for_unpinned_litellm_provider(
    tmp_path, monkeypatch, capsys,
):
    _configure(
        tmp_path,
        monkeypatch,
        [{
            "name": "llama-3.3-70b-versatile",
            "provider": "groq",
            "api_key": "must-not-be-sent",
        }],
    )

    def unexpected_call(*args, **kwargs):
        pytest.fail("invalid endpoint configuration must fail before I/O")

    monkeypatch.setattr(check_api_key.httpx, "get", unexpected_call)
    monkeypatch.setattr(check_api_key.httpx, "post", unexpected_call)
    monkeypatch.setattr(
        "turbo_agent.utils.llm.llm_completion",
        unexpected_call,
    )

    assert check_api_key.main() == 1

    error = capsys.readouterr().err
    assert "provider 'groq'" in error
    assert "base_url" in error
    assert "backend.models[0]" in error


def test_main_rejects_missing_custom_endpoint_key_before_http(
    tmp_path, monkeypatch, capsys,
):
    missing_var = "MISSING_CUSTOM_ENDPOINT_KEY"
    monkeypatch.delenv(missing_var, raising=False)
    _configure(
        tmp_path,
        monkeypatch,
        [
            {
                "name": "openai/local-model",
                "base_url": "https://gateway.example/v1",
                "api_key": f"${missing_var}",
            }
        ],
    )
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 1

    captured = capsys.readouterr()
    assert "backend.models[0].api_key" in captured.err
    assert f"${missing_var}" in captured.err
    assert calls == []


def test_main_uses_project_dotenv_for_custom_endpoint_and_key_when_shell_unset(
    tmp_path, monkeypatch,
):
    endpoint_var = "CUSTOM_DOTENV_BASE_URL"
    key_var = "CUSTOM_DOTENV_API_KEY"
    _configure(
        tmp_path,
        monkeypatch,
        [
            {
                "name": "openai/local-model",
                "base_url": f"${endpoint_var}",
                "api_key": f"${key_var}",
            }
        ],
    )
    (tmp_path / ".env").write_text(
        f"{endpoint_var}=https://project.example/v1\n"
        f"{key_var}=project-secret\n"
    )
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 0

    assert calls == [
        (
            "https://project.example/v1/models",
            {
                "headers": {"Authorization": "Bearer project-secret"},
                "timeout": 30.0,
            },
        )
    ]


def test_main_preserves_shell_environment_over_project_dotenv(
    tmp_path, monkeypatch,
):
    endpoint_var = "CUSTOM_DOTENV_PRIORITY_BASE_URL"
    key_var = "CUSTOM_DOTENV_PRIORITY_API_KEY"
    _configure(
        tmp_path,
        monkeypatch,
        [{
            "name": "openai/local-model",
            "base_url": f"${endpoint_var}",
            "api_key": f"${key_var}",
        }],
    )
    monkeypatch.setenv(endpoint_var, "https://shell.example/v1")
    monkeypatch.setenv(key_var, "shell-secret")
    (tmp_path / ".env").write_text(
        f"{endpoint_var}=https://project.example/v1\n"
        f"{key_var}=project-secret\n"
    )
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 0
    assert calls == [
        (
            "https://shell.example/v1/models",
            {
                "headers": {"Authorization": "Bearer shell-secret"},
                "timeout": 30.0,
            },
        )
    ]


def test_main_redacts_custom_key_echoed_by_endpoint(
    tmp_path, monkeypatch, capsys,
):
    custom_key = "arbitrary-custom-secret-" + "x" * 240
    _configure(
        tmp_path,
        monkeypatch,
        [
            {
                "name": "openai/local-model",
                "base_url": "https://gateway.example/v1",
                "api_key": custom_key,
            }
        ],
    )

    def fake_get(url, **kwargs):
        return httpx.Response(
            401,
            json={"error": {"message": f"invalid credential {custom_key}"}},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(check_api_key.httpx, "get", fake_get)

    assert check_api_key.main() == 1

    captured = capsys.readouterr()
    assert custom_key not in captured.out
    assert "arbitrary-custom-secret-" not in captured.out
    assert custom_key not in captured.err
    assert "<redacted>" in captured.out


@pytest.mark.parametrize(
    ("status_code", "message", "expected"),
    [
        (400, "invalid model: supports at most 4032 tokens", "warn"),
        (400, "request 4010 exceeded the token limit", "warn"),
        (400, "invalid model model-403", "warn"),
        (400, "requested exactly 401 tokens", "warn"),
        (400, "invalid api key", "fail"),
        (400, "Credentials have expired", "fail"),
        (400, "The provided token is expired", "fail"),
        (400, "invalid_grant", "fail"),
        (400, "Could not resolve project_id", "fail"),
        (400, "Anonymous credentials cannot be refreshed", "fail"),
        (400, "HTTP status 401", "fail"),
        (401, "request rejected", "fail"),
        (403, "request rejected", "fail"),
    ],
)
def test_http_error_classification_uses_auth_semantics(
    status_code, message, expected,
):
    response = httpx.Response(
        status_code,
        json={"error": {"message": message}},
        request=httpx.Request("GET", "https://example.test/v1/models"),
    )

    status, _ = check_api_key.OpenAIChecker()._classify_http(response, "ok")

    assert status == expected


def test_invalid_url_is_unverified_not_an_auth_failure(monkeypatch):
    def fake_get(*args, **kwargs):
        raise httpx.InvalidURL("Invalid URL")

    monkeypatch.setattr(check_api_key.httpx, "get", fake_get)
    result = check_api_key.OpenAIChecker(api_key="not-secret").run()

    assert result.status == "warn"


@pytest.mark.parametrize("failure_kind", ["exception", "response"])
def test_checker_redacts_base_url_secrets(monkeypatch, failure_kind):
    base_url = (
        "https://user-secret:pass-secret@gateway.example/v1"
        "?token=query-secret#fragment-secret"
    )

    def fake_get(url, **kwargs):
        detail = f"connection failed for {url}"
        if failure_kind == "exception":
            raise RuntimeError(detail)
        return httpx.Response(
            500,
            text=detail,
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(check_api_key.httpx, "get", fake_get)
    result = check_api_key.OpenAIChecker(
        base_url=base_url,
        api_key="header-secret",
    ).run()

    assert result.status == "warn"
    for secret in (
        "user-secret",
        "pass-secret",
        "query-secret",
        "fragment-secret",
        "header-secret",
    ):
        assert secret not in result.detail
    assert "gateway.example" in result.detail


def test_main_rejects_non_http_base_url_before_request(
    tmp_path, monkeypatch, capsys,
):
    _configure(
        tmp_path,
        monkeypatch,
        [{
            "name": "openai/local-model",
            "base_url": "not-a-url",
            "api_key": "test-key",
        }],
    )

    def unexpected_http(*args, **kwargs):
        pytest.fail("an invalid base_url must fail before HTTP")

    monkeypatch.setattr(check_api_key.httpx, "get", unexpected_http)
    monkeypatch.setattr(check_api_key.httpx, "post", unexpected_http)

    assert check_api_key.main() == 1
    assert "absolute HTTP(S) URL" in capsys.readouterr().err


def test_main_rejects_invalid_openai_base_url_environment_before_request(
    tmp_path, monkeypatch, capsys,
):
    raw_config = {
        "backend": {"models": [{"name": "gemini/gemini-2.5-flash"}]},
        "progress_monitor": {
            "model": {
                "name": "openai/gpt-4o",
                "api_key": "test-key",
            }
        },
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "not-a-url")

    def unexpected_http(*args, **kwargs):
        pytest.fail("an invalid OPENAI_BASE_URL must fail before HTTP")

    monkeypatch.setattr(check_api_key.httpx, "get", unexpected_http)
    monkeypatch.setattr(check_api_key.httpx, "post", unexpected_http)

    assert check_api_key.main() == 1
    assert "OPENAI_BASE_URL must be an absolute HTTP(S) URL" in (
        capsys.readouterr().err
    )


def test_malformed_yaml_does_not_echo_literal_secrets(
    tmp_path, monkeypatch, capsys,
):
    secret = "literal-credential-that-must-never-appear"
    (tmp_path / "turbo-agent.yaml").write_text(
        "backend:\n  models:\n    - name: openai/test\n"
        f"      api_key: [{secret}\n"
    )
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)

    assert check_api_key.main() == 1

    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "invalid YAML" in captured.err


@pytest.mark.parametrize(
    ("raw_config", "expected_error"),
    [
        (
            {"backend": {"models": []}},
            "No models configured under backend.models",
        ),
        (
            {
                "backend": {"models": [{"name": "openai/gpt-4o"}]},
                "verifier": {"method": None},
            },
            "verifier.method must be a mapping",
        ),
    ],
)
def test_main_reuses_runtime_structure_validation(
    tmp_path, monkeypatch, capsys, raw_config, expected_error,
):
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_api_key.httpx,
        "get",
        lambda *args, **kwargs: pytest.fail("invalid config must not make HTTP calls"),
    )

    assert check_api_key.main() == 1

    assert expected_error in capsys.readouterr().err


def test_missing_explicit_key_is_reported_under_its_own_variable(
    tmp_path, monkeypatch, capsys,
):
    _configure(
        tmp_path,
        monkeypatch,
        [{"name": "openai/gpt-4o", "api_key": "$CUSTOM_OPENAI_KEY"}],
    )
    monkeypatch.delenv("CUSTOM_OPENAI_KEY", raising=False)
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 1

    output = capsys.readouterr().out
    assert "[CUSTOM_OPENAI_KEY]" in output
    assert "not set" in output
    assert "(backend)" in output
    assert calls == []


def test_custom_deepseek_verifier_never_checks_the_official_origin(
    tmp_path, monkeypatch,
):
    _isolate_deepseek_reasoning(monkeypatch)
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "verifier": {
            "model": {
                "name": "deepseek/custom-reasoner",
                "base_url": "https://deepseek-proxy.example/v1",
                "api_key": "$DEEPSEEK_API_KEY",
            }
        },
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    for env_var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-proxy-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "backend-secret")
    gets = _record_gets(monkeypatch)
    calls = []

    def fake_post(url, **kwargs):
        calls.append((str(url), kwargs))
        return _chat_response(
            url,
            "<score_A>A</score_A><score_B>B</score_B>",
            [
                {"token": "<score_A>", "logprob": 0, "top_logprobs": []},
                _letter_position(),
                {"token": "</score_A>", "logprob": 0, "top_logprobs": []},
                {"token": "<score_B>", "logprob": 0, "top_logprobs": []},
                _letter_position("B"),
                {"token": "</score_B>", "logprob": 0, "top_logprobs": []},
            ],
        )

    monkeypatch.setattr(check_api_key.httpx, "post", fake_post)

    assert check_api_key.main() == 0
    assert [url for url, _ in gets] == [
        "https://api.openai.com/v1/models"
    ]
    assert [url for url, _ in calls] == [
        "https://deepseek-proxy.example/v1/chat/completions"
    ]
    payload = calls[0][1]["json"]
    assert payload["model"] == "custom-reasoner"
    assert payload["max_tokens"] == 32768
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert all("api.deepseek.com" not in url for url, _ in calls)


def test_logprob_checker_warns_without_usable_score_distribution(monkeypatch):
    def fake_post(url, **kwargs):
        return _chat_response(
            url,
            "not a score",
            [{"token": "x", "logprob": 0, "top_logprobs": []}],
        )

    monkeypatch.setattr(check_api_key.httpx, "post", fake_post)
    result = check_api_key.OpenAILogprobChecker(
        ("verifier",),
        base_url="https://verifier.example/v1",
        api_key="test-key",
        model="test-model",
        profile="verifier",
    ).run()

    assert result.status == "warn"
    assert "fall back to 0.5" in result.detail


@pytest.mark.parametrize(
    "checker_cls", [
        check_api_key.OpenAILogprobChecker,
        check_api_key.DeepSeekLogprobChecker,
    ],
)
@pytest.mark.parametrize(
    "base_url",
    [
        "https://gateway.example/root/v1/?tenant=acme#route",
        (
            "https://gateway.example/root/v1/chat/completions/"
            "?tenant=acme#route"
        ),
    ],
)
def test_logprob_checkers_append_endpoint_after_path_before_query(
    monkeypatch, checker_cls, base_url,
):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((str(url), kwargs))
        return _chat_response(
            url,
            "A" if checker_cls is check_api_key.OpenAILogprobChecker
            else "<score_A>A</score_A><score_B>B</score_B>",
            [_letter_position()],
        )

    monkeypatch.setattr(check_api_key.httpx, "post", fake_post)
    result = checker_cls(
        ("verifier",),
        base_url=base_url,
        api_key="custom-secret",
        model="test-model",
        profile="verifier",
    ).run()

    assert result.status in ("ok", "warn")
    assert calls[0][0] == (
        "https://gateway.example/root/v1/chat/completions"
        "?tenant=acme#route"
    )


def test_logprob_checker_warns_when_message_is_missing(monkeypatch):
    def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "logprobs": {"content": [_letter_position()]},
                }]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(check_api_key.httpx, "post", fake_post)
    result = check_api_key.OpenAILogprobChecker(
        ("verifier",),
        base_url="https://verifier.example/v1",
        api_key="test-key",
        model="test-model",
        profile="verifier",
    ).run()

    assert result.status == "warn"
    assert "choices[0].message" in result.detail


def test_main_fails_when_configured_verifier_protocol_is_unverified(
    tmp_path, monkeypatch,
):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "verifier": {
            "model": {
                "name": "openai/local-verifier",
                "base_url": "https://verifier.example/v1",
                "api_key": "verifier-secret",
            }
        },
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    for env_var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    def fake_post(url, **kwargs):
        return _chat_response(
            url,
            "not a score",
            [{"token": "x", "logprob": 0, "top_logprobs": []}],
        )

    monkeypatch.setattr(check_api_key.httpx, "post", fake_post)

    assert check_api_key.main() == 1


def test_plain_gemini_verifier_is_rejected_before_http(
    tmp_path, monkeypatch, capsys,
):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "verifier": {
            "model": {
                "name": "gemini/gemini-2.5-flash",
                "api_key": "$GEMINI_API_KEY",
            }
        },
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 1

    assert "provider: vertex_ai" in capsys.readouterr().err
    assert calls == []


def test_official_openai_tournament_verifier_is_rejected_before_http(
    tmp_path, monkeypatch, capsys,
):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "verifier": {
            "model": {
                "name": "openai/gpt-4o",
                "api_key": "$OPENAI_API_KEY",
            }
        },
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 1

    assert "official OpenAI API" in capsys.readouterr().err
    assert calls == []


def test_explicit_official_openai_base_url_tournament_verifier_is_rejected(
    tmp_path, monkeypatch, capsys,
):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "verifier": {
            "model": {
                "name": "openai/gpt-4o",
                "base_url": "https://API.OPENAI.COM/v1/",
                "api_key": "openai-secret",
            }
        },
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(
        check_api_key.httpx,
        "post",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert check_api_key.main() == 1
    assert "official OpenAI API" in capsys.readouterr().err
    assert calls == []


def test_vertex_api_key_is_rejected_for_litellm_backend(
    tmp_path, monkeypatch, capsys,
):
    _configure(
        tmp_path,
        monkeypatch,
        [{
            "name": "gemini/gemini-2.5-flash",
            "provider": "vertex_ai",
            "api_key": "$VERTEX_API_KEY",
        }],
    )
    monkeypatch.setenv("VERTEX_API_KEY", "vertex-secret")
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 1

    assert "Vertex ADC/project" in capsys.readouterr().err
    assert calls == []


def test_unsupported_standard_verifier_provider_is_rejected_before_http(
    tmp_path, monkeypatch, capsys,
):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "verifier": {
            "model": {
                "name": "deployment-name",
                "provider": "azure",
                "api_key": "$AZURE_API_KEY",
            }
        },
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 1

    assert "not a supported verifier provider" in capsys.readouterr().err
    assert calls == []


def test_unknown_verifier_provider_is_rejected_before_environment_fallback(
    tmp_path, monkeypatch, capsys,
):
    raw_config = {
        "backend": {"models": [{"name": "openai/gpt-4o"}]},
        "verifier": {"model": {"name": "unrecognized/verifier-model"}},
    }
    (tmp_path / "turbo-agent.yaml").write_text(yaml.safe_dump(raw_config))
    monkeypatch.setattr(check_api_key, "ROOT", tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://untrusted.example/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-be-forwarded")
    calls = _record_gets(monkeypatch)

    assert check_api_key.main() == 1

    assert "cannot determine the verifier provider" in capsys.readouterr().err
    assert calls == []


def test_unknown_backend_provider_is_rejected_before_http(
    tmp_path, monkeypatch, capsys,
):
    _configure(
        tmp_path,
        monkeypatch,
        [{"name": "unrecognized/backend-model"}],
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://untrusted.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-forwarded")
    gets = _record_gets(monkeypatch)

    def fail_post(*args, **kwargs):
        pytest.fail("an unknown provider must fail before HTTP")

    monkeypatch.setattr(check_api_key.httpx, "post", fail_post)

    assert check_api_key.main() == 1

    error = capsys.readouterr().err
    assert "has no recognized provider" in error
    assert "configure both provider and base_url explicitly" in error
    assert gets == []
