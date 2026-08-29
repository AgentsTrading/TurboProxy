import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import yaml
from dotenv import load_dotenv


_MODEL_PREFIX_PROVIDERS = {
    "deepseek/": "deepseek",
    "openai/": "openai",
    "gemini/": "gemini",
    "vertex_ai/": "vertex_ai",
    "anthropic/": "anthropic",
}
_IMPLICIT_BASE_URL_PROVIDERS = frozenset(_MODEL_PREFIX_PROVIDERS.values())


@lru_cache(maxsize=None)
def _infer_litellm_provider(model: str) -> Optional[str]:
    """Use the installed LiteLLM registry for prefixless known model ids."""
    try:
        import litellm

        _, provider, _, _ = litellm.get_llm_provider(model=model)
    except Exception:
        return None
    return provider or None


def split_model_name(name: Any) -> tuple[str, Optional[str]]:
    """Return the provider-neutral model id and any recognized prefix."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model name must be a non-empty string")
    name = name.strip()
    for prefix, provider in _MODEL_PREFIX_PROVIDERS.items():
        if name.startswith(prefix):
            model_id = name.removeprefix(prefix)
            if not model_id:
                raise ValueError(f"model name must follow the '{prefix}' prefix")
            return model_id, provider
    return name, None


def resolve_model_provider(
    name: Any,
    provider: Any = None,
    base_url: Any = None,
) -> Optional[str]:
    """Resolve the provider using the same precedence as runtime routing."""
    model_id, prefix_provider = split_model_name(name)
    if provider is not None:
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be a non-empty string")
        return provider.strip().lower()
    if prefix_provider:
        return prefix_provider
    if base_url is not None:
        return "openai"
    return _infer_litellm_provider(model_id)


def validate_litellm_endpoint(
    provider: Optional[str], has_base_url: bool, field_name: str,
) -> None:
    """Require an isolated endpoint for providers without a pinned default."""
    if has_base_url:
        return
    if provider is None:
        raise ValueError(
            f"{field_name} has no recognized provider; use a supported model "
            "prefix or configure both provider and base_url explicitly"
        )
    if provider not in _IMPLICIT_BASE_URL_PROVIDERS:
        raise ValueError(
            f"provider '{provider}' requires an explicit model-level base_url "
            "(and provider when no supported prefix exists); TurboProxy cannot "
            "isolate its credentials from process-wide endpoint overrides "
            f"otherwise ({field_name})"
        )


def resolve_litellm_route(
    name: Any,
    provider: Any = None,
    base_url: Any = None,
) -> tuple[str, Optional[str]]:
    """Return ``(model, custom_llm_provider)`` for a LiteLLM call.

    LiteLLM handles a recognized prefix itself. An explicit ``provider`` must
    instead be passed as ``custom_llm_provider`` and receives the bare model
    id. A prefixless custom endpoint is OpenAI-compatible by default.
    """
    model_id, prefix_provider = split_model_name(name)
    resolved_provider = resolve_model_provider(name, provider, base_url)
    if provider is not None:
        return model_id, resolved_provider
    if prefix_provider:
        return str(name).strip(), None
    if base_url is not None:
        return model_id, "openai"
    return model_id, None


def _resolve_required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    value = value.strip()
    if value.startswith("$"):
        env_var = value[1:]
        resolved = os.environ.get(env_var, "").strip()
        if not resolved:
            raise ValueError(
                f"{field_name} references environment variable {value}, "
                "but it is not set or is empty"
            )
        return resolved
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def resolve_base_url(value: Any, field_name: str) -> str:
    """Resolve a base URL reference without allowing provider fallbacks."""
    resolved = _resolve_required_string(value, field_name)
    try:
        parsed = urlsplit(resolved)
        _ = parsed.port
    except ValueError:
        parsed = None
    if (
        parsed is None
        or parsed.scheme.lower() not in ("http", "https")
        or not parsed.hostname
        or any(char.isspace() for char in resolved)
    ):
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    return resolved


def append_url_path(base_url: Any, path: str) -> str:
    """Append an endpoint path while keeping URL query parameters in place.

    String concatenation places a query before the endpoint (``?key=x/models``)
    and produces an invalid request.  URL parsing also makes trailing slashes
    and query-only base URLs deterministic for all provider checkers.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    if not isinstance(path, str) or not path.strip("/"):
        raise ValueError("path must be a non-empty string")
    try:
        parsed = urlsplit(base_url.strip())
    except ValueError as exc:
        raise ValueError("base_url is not a valid URL") from exc
    base_path = parsed.path.rstrip("/")
    suffix = path.strip("/")
    joined_path = f"{base_path}/{suffix}" if base_path else f"/{suffix}"
    return urlunsplit(
        (parsed.scheme, parsed.netloc, joined_path, parsed.query, "")
    )


def redact_base_url(value: Any) -> Any:
    """Redact URL userinfo, query values, and fragments before request logs."""
    if not isinstance(value, str) or not value.strip():
        return value
    try:
        parsed = urlsplit(value.strip())
        hostname = parsed.hostname
        if not hostname:
            return "***"
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        try:
            port = f":{parsed.port}" if parsed.port is not None else ""
        except ValueError:
            return "***"
        userinfo = "<redacted>@" if (
            parsed.username is not None or parsed.password is not None
        ) else ""
        netloc = f"{userinfo}{hostname}{port}"
        return urlunsplit(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                "<redacted>" if parsed.query else "",
                "<redacted>" if parsed.fragment else "",
            )
        )
    except ValueError:
        return "***"


def is_official_openai_base_url(value: Any) -> bool:
    """Return whether a URL targets OpenAI's public API origin."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return False
    return bool(
        parsed.hostname
        and parsed.hostname.rstrip(".").lower() == "api.openai.com"
    )


def resolve_api_key(value: Any, field_name: str) -> str:
    """Resolve a key required by an explicitly configured custom endpoint."""
    return _resolve_required_string(value, field_name)


def validate_litellm_api_key(
    provider: Optional[str], value: Any, field_name: str
) -> None:
    """Reject API-key auth that LiteLLM does not use for this provider."""
    if (
        provider == "vertex_ai"
        and isinstance(value, str)
        and value.strip()
    ):
        raise ValueError(
            f"{field_name} cannot authenticate LiteLLM provider 'vertex_ai'; "
            "remove api_key and configure Vertex ADC/project authentication, "
            "or use provider 'gemini' with GEMINI_API_KEY"
        )


def resolve_optional_api_key(value: Any, field_name: str) -> Optional[str]:
    """Resolve an optional key while rejecting ambiguous YAML scalar types."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    value = value.strip()
    if value.startswith("$"):
        return resolve_api_key(value, field_name)
    return value


def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping without echoing source lines in parse errors."""
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = (
            f" at line {mark.line + 1}, column {mark.column + 1}"
            if mark is not None else ""
        )
        raise ValueError(
            f"could not parse {path.name}{location}: invalid YAML"
        ) from None
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"could not read {path.name}: {type(exc).__name__}"
        ) from None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must contain a mapping")
    return raw


def validate_config_structure(raw: Dict[str, Any]) -> None:
    """Validate YAML containers shared by runtime and diagnostic commands."""
    backend = raw.get("backend", {})
    if not isinstance(backend, dict):
        raise ValueError("backend must be a mapping")

    models = backend.get("models", [])
    if not isinstance(models, list):
        raise ValueError("backend.models must be a list")
    if not models:
        raise ValueError("No models configured under backend.models")
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"backend.models[{index}] must be a mapping")
        name = model.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"backend.models[{index}].name must be a non-empty string"
            )

    optional_models = (
        ("context", "refinement_model"),
        ("verifier", "model"),
        ("progress_monitor", "model"),
    )
    sections: Dict[str, Dict[str, Any]] = {}
    for section_name, model_field in optional_models:
        section = raw.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} must be a mapping")
        sections[section_name] = section

        model = section.get(model_field)
        if model is not None and not isinstance(model, dict):
            raise ValueError(f"{section_name}.{model_field} must be a mapping")

    verifier = sections.get("verifier")
    if verifier is None:
        return

    method = verifier.get("method", {})
    if not isinstance(method, dict):
        raise ValueError("verifier.method must be a mapping")
    criteria = method.get("criteria", [])
    if not isinstance(criteria, list):
        raise ValueError("verifier.method.criteria must be a list")
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, dict):
            raise ValueError(
                f"verifier.method.criteria[{index}] must be a mapping"
            )


@dataclass
class ModelConfig:
    name: str
    provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class CriterionConfig:
    name: str
    description: str = ""


@dataclass
class PivotTournamentConfig:
    """Parameters for the Probabilistic Pivot Tournament selection method."""
    pivots: int = 2            # k: number of pivot (empirical-leader) candidates
    n_verifications: int = 4   # K: repeated verifications per directed pair
    seed: int = 0             # seed for the random ring pass (reproducible)
    note: str = ""             # ground-truth note injected into the prompt
    criteria: List[CriterionConfig] = field(default_factory=list)


@dataclass
class VerifierConfig:
    model: ModelConfig
    method: PivotTournamentConfig
    majority_voting: bool = False


@dataclass
class ContextConfig:
    model_name: str
    api_key: str
    refinement_prompt: str
    provider: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class ProgressMonitorConfig:
    """A post-hoc progress score for the selected trajectory, computed with
    `llm_verifier.track` (K repeated verifications, averaged). Observability
    only — it never changes the response."""
    model: ModelConfig
    n_verifications: int = 4   # K: repeated verifications


# Default holistic criterion used when the config declares none.
_DEFAULT_CRITERIA = [
    CriterionConfig(
        name="Task Success",
        description=(
            "How likely the agent correctly and completely solved the task. "
            "The strongest signal is the agent verifying its solution against "
            "the task's specific requirements. Trajectory length, number of "
            "steps, and apparent confidence do not predict correctness."
        ),
    ),
]


class Config:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(Path.cwd() / "turbo-proxy.yaml")
            if not Path(config_path).exists():
                raise FileNotFoundError(
                    f"No turbo-proxy.yaml found in {Path.cwd()}. "
                    "Run turbo-proxy from a directory containing a "
                    "turbo-proxy.yaml config file."
                )

        # Load .env from the same directory as the config file.
        env_path = Path(config_path).parent / ".env"
        if env_path.exists():
            load_dotenv(str(env_path), override=False)

        self._raw = load_yaml_mapping(Path(config_path))

        self._validate_structure()
        self._expand_env_vars()

    def _validate_structure(self) -> None:
        validate_config_structure(self._raw)

    def _expand_env_vars(self) -> None:
        for index, model in enumerate(self.models):
            api_key = model.get("api_key", "")
            has_base_url = "base_url" in model
            provider = resolve_model_provider(
                model.get("name"), model.get("provider"), model.get("base_url")
            )
            validate_litellm_endpoint(
                provider, has_base_url, f"backend.models[{index}]"
            )
            validate_litellm_api_key(
                provider, api_key, f"backend.models[{index}].api_key"
            )
            if has_base_url:
                model["base_url"] = resolve_base_url(
                    model.get("base_url"),
                    f"backend.models[{index}].base_url",
                )
                if provider == "vertex_ai":
                    # LiteLLM's Vertex route authenticates with ADC; a custom
                    # API base does not change that credential contract.
                    model["api_key"] = resolve_optional_api_key(
                        api_key, f"backend.models[{index}].api_key"
                    ) or ""
                else:
                    model["api_key"] = resolve_api_key(
                        api_key, f"backend.models[{index}].api_key"
                    )
            elif "api_key" in model:
                model["api_key"] = resolve_optional_api_key(
                    api_key, f"backend.models[{index}].api_key"
                )

    # ------------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------------

    @property
    def models(self) -> List[Dict[str, Any]]:
        return self._raw.get("backend", {}).get("models", [])

    @property
    def default_model(self) -> Dict[str, Any]:
        if not self.models:
            raise ValueError("No models configured under backend.models")
        return self.models[0]

    @property
    def total_candidates(self) -> int:
        return sum(m.get("num_candidates", 1) for m in self.models)

    # ------------------------------------------------------------------
    # Context refinement (optional)
    # ------------------------------------------------------------------

    @property
    def context_config(self) -> Optional[ContextConfig]:
        raw_ctx = self._raw.get("context")
        if not raw_ctx:
            return None
        raw_model = raw_ctx.get("refinement_model")
        prompt = raw_ctx.get("refinement_prompt")
        if not raw_model or not raw_model.get("name") or not prompt:
            return None
        raw_api_key = raw_model.get("api_key", "")
        provider = resolve_model_provider(
            raw_model["name"], raw_model.get("provider"), raw_model.get("base_url")
        )
        has_base_url = "base_url" in raw_model
        validate_litellm_endpoint(
            provider, has_base_url, "context.refinement_model"
        )
        validate_litellm_api_key(
            provider, raw_api_key, "context.refinement_model.api_key"
        )
        base_url = (
            resolve_base_url(
                raw_model.get("base_url"), "context.refinement_model.base_url"
            )
            if has_base_url else None
        )
        api_key = (
            resolve_optional_api_key(
                raw_api_key, "context.refinement_model.api_key"
            )
            if has_base_url and provider == "vertex_ai"
            else resolve_api_key(
                raw_api_key, "context.refinement_model.api_key"
            )
            if has_base_url
            else resolve_optional_api_key(
                raw_api_key, "context.refinement_model.api_key"
            )
        ) or ""
        return ContextConfig(
            model_name=raw_model["name"],
            api_key=api_key,
            refinement_prompt=prompt,
            provider=raw_model.get("provider"),
            base_url=base_url,
        )

    # ------------------------------------------------------------------
    # Verifier
    # ------------------------------------------------------------------

    @property
    def verifier_config(self) -> Optional[VerifierConfig]:
        raw_v = self._raw.get("verifier")
        if not raw_v:
            return None

        raw_model = raw_v.get("model")
        if not raw_model or not raw_model.get("name"):
            return None
        raw_api_key = raw_model.get("api_key", "")
        has_base_url = "base_url" in raw_model
        raw_base_url = raw_model.get("base_url")
        base_url = (
            resolve_base_url(raw_base_url, "verifier.model.base_url")
            if has_base_url else None
        )
        provider = resolve_model_provider(
            raw_model["name"], raw_model.get("provider"), raw_base_url
        )
        if has_base_url and provider == "vertex_ai":
            api_key = resolve_optional_api_key(
                raw_api_key, "verifier.model.api_key"
            )
        elif has_base_url:
            api_key = resolve_api_key(raw_api_key, "verifier.model.api_key")
        else:
            api_key = resolve_optional_api_key(
                raw_api_key, "verifier.model.api_key"
            ) or None
        model_cfg = ModelConfig(
            name=raw_model["name"],
            provider=raw_model.get("provider"),
            api_key=api_key,
            base_url=base_url,
        )

        raw_method = raw_v.get("method", {})
        method_name = raw_method.get("name", "pivot_tournament")
        if method_name != "pivot_tournament":
            raise ValueError(
                f"Unknown verifier method '{method_name}'. "
                f"Only 'pivot_tournament' is supported."
            )

        criteria = [
            CriterionConfig(
                name=c.get("name", ""),
                description=c.get("description", ""),
            )
            for c in raw_method.get("criteria", [])
        ] or list(_DEFAULT_CRITERIA)

        method_cfg = PivotTournamentConfig(
            pivots=raw_method.get("pivots", 2),
            n_verifications=raw_method.get("n_verifications", 4),
            seed=raw_method.get("seed", 0),
            note=raw_method.get("note", ""),
            criteria=criteria,
        )

        return VerifierConfig(
            model=model_cfg,
            method=method_cfg,
            majority_voting=raw_v.get("majority_voting", False),
        )

    # ------------------------------------------------------------------
    # Progress monitor (optional, post-hoc observability)
    # ------------------------------------------------------------------

    @property
    def progress_monitor_config(self) -> Optional[ProgressMonitorConfig]:
        raw_pm = self._raw.get("progress_monitor")
        if not raw_pm:
            return None
        raw_model = raw_pm.get("model")
        if not raw_model or not raw_model.get("name"):
            return None
        raw_api_key = raw_model.get("api_key", "")
        has_base_url = "base_url" in raw_model
        raw_base_url = raw_model.get("base_url")
        base_url = (
            resolve_base_url(raw_base_url, "progress_monitor.model.base_url")
            if has_base_url else None
        )
        provider = resolve_model_provider(
            raw_model["name"], raw_model.get("provider"), raw_base_url
        )
        if has_base_url and provider == "vertex_ai":
            api_key = resolve_optional_api_key(
                raw_api_key, "progress_monitor.model.api_key"
            )
        elif has_base_url:
            api_key = resolve_api_key(
                raw_api_key, "progress_monitor.model.api_key"
            )
        else:
            api_key = resolve_optional_api_key(
                raw_api_key, "progress_monitor.model.api_key"
            ) or None
        model_cfg = ModelConfig(
            name=raw_model["name"],
            provider=raw_model.get("provider"),
            api_key=api_key,
            base_url=base_url,
        )
        return ProgressMonitorConfig(
            model=model_cfg,
            n_verifications=raw_pm.get("n_verifications", 4),
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @property
    def log_dir(self) -> str:
        dir_name = self._raw.get("log_dir", "default")
        return str(Path(".turbo-proxy") / dir_name)

    @property
    def raw_config(self) -> Dict[str, Any]:
        return self._raw
