import asyncio
import json
import time
import uuid
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import anyio

from ..utils import (
    Config,
    AnthropicToOpenAI,
    OpenAIToAnthropic,
    STOP_REASON_MAP,
    SSEFormatter,
    llm_completion,
    llm_stream_completion,
    llm_response,
    llm_stream_response,
    create_logger,
    create_request_log,
    save_request_log,
)
from ..utils.config import redact_base_url
from ..utils.llm import (
    _await_cleanup_task,
    _build_responses_kwargs,
    _responses_provider_uses_chat_fallback,
    _validate_responses_request,
)
from ..context import ContextRefiner
from ..progress_monitor import ProgressMonitor
from ..verifier import Verifier

logger = create_logger("backend")


async def _close_upstream_stream(
    stream: Any, primary: Optional[BaseException] = None,
) -> None:
    close_stream = getattr(stream, "aclose", None)
    if close_stream is None:
        wrapper = getattr(stream, "litellm_custom_stream_wrapper", None)
        close_stream = getattr(wrapper, "aclose", None)
    if close_stream is None:
        response = getattr(stream, "response", None)
        close_stream = getattr(response, "aclose", None)
    if close_stream is None:
        return

    async def close() -> None:
        await close_stream()

    try:
        with anyio.CancelScope(shield=True):
            await _await_cleanup_task(asyncio.create_task(close()))
    except BaseException as cleanup_exc:
        if primary is None or cleanup_exc is primary:
            raise
        raise primary from cleanup_exc


class Backend:
    """Request pipeline: (optional) context refinement -> concurrent inference
    -> pivot-tournament verification -> best response."""

    def __init__(self, config: Config):
        self.config = config

        self.refiner: Optional[ContextRefiner] = None
        self.verifier: Optional[Verifier] = None
        self.progress_monitor: Optional[ProgressMonitor] = None
        self._bg_tasks: set = set()  # strong refs so background tasks aren't GC'd

        ctx_cfg = config.context_config
        if ctx_cfg:
            self.refiner = ContextRefiner(ctx_cfg)
            logger.info(f"Context refinement enabled (model={ctx_cfg.model_name})")

        ver_cfg = config.verifier_config
        if ver_cfg and config.total_candidates > 1:
            self.verifier = Verifier(ver_cfg)
            logger.info(
                f"Verifier enabled (total_candidates={config.total_candidates})"
            )

        pm_cfg = config.progress_monitor_config
        if pm_cfg:
            self.progress_monitor = ProgressMonitor(pm_cfg)
            logger.info(f"Progress monitor enabled (model={pm_cfg.model.name})")

    @property
    def model_name(self) -> str:
        return self.config.default_model["name"]

    @property
    def api_key(self) -> str:
        return self.config.default_model.get("api_key", "")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_model_params(model: dict) -> dict:
        params: dict = {}
        if model.get("temperature") is not None:
            params["temperature"] = model["temperature"]
        if model.get("max_tokens") is not None:
            params["max_tokens"] = model["max_tokens"]
        thinking = model.get("thinking")
        if thinking is not None:
            if isinstance(thinking, (int, float)):
                params["thinking_budget"] = int(thinking)
            elif isinstance(thinking, str):
                params["reasoning_effort"] = thinking
        return params

    def _base_params(self) -> dict:
        return {
            "model": self.model_name,
            "api_key": self.api_key,
            "base_url": self.config.default_model.get("base_url") or None,
            "provider": self.config.default_model.get("provider"),
            **self._parse_model_params(self.config.default_model),
        }

    def _model_entries(self) -> List[dict]:
        """One entry per candidate to generate (num_candidates per model)."""
        entries: List[dict] = []
        for model in self.config.models:
            num = model.get("num_candidates", 1)
            entry = {
                "name": model["name"],
                "api_key": model.get("api_key", ""),
                **self._parse_model_params(model),
                "base_url": model.get("base_url") or None,
                "provider": model.get("provider"),
            }
            for _ in range(num):
                entries.append(entry)
        return entries

    def _sanitized_config(self) -> dict:
        raw = dict(self.config.raw_config)

        def sanitize_model(model: dict) -> dict:
            sanitized = {**model, "api_key": "***"}
            if "base_url" in model:
                sanitized["base_url"] = redact_base_url(model["base_url"])
            return sanitized

        backend = raw.get("backend")
        if isinstance(backend, dict) and backend.get("models"):
            raw["backend"] = {
                **backend,
                "models": [
                    sanitize_model(model) for model in backend["models"]
                ],
            }

        for section, model_key in (
            ("context", "refinement_model"),
            ("verifier", "model"),
            ("progress_monitor", "model"),
        ):
            section_config = raw.get(section)
            if not isinstance(section_config, dict):
                continue
            model = section_config.get(model_key)
            if not isinstance(model, dict):
                continue
            raw[section] = {
                **section_config,
                model_key: sanitize_model(model),
            }
        return raw

    @staticmethod
    def _responses_request_for_log(body: dict) -> dict:
        """Copy a Responses request while redacting provider credentials."""
        logged = dict(body)
        for field in ("extra_body", "extra_headers", "extra_query"):
            values = logged.get(field)
            if isinstance(values, dict):
                logged[field] = {key: "***" for key in values}

        tools = logged.get("tools")
        if isinstance(tools, list):
            sanitized_tools = []
            for tool in tools:
                if not isinstance(tool, dict):
                    sanitized_tools.append(tool)
                    continue
                sanitized_tool = dict(tool)
                if "authorization" in sanitized_tool:
                    sanitized_tool["authorization"] = "***"
                headers = sanitized_tool.get("headers")
                if isinstance(headers, dict):
                    sanitized_tool["headers"] = {
                        key: "***" for key in headers
                    }
                if "server_url" in sanitized_tool:
                    sanitized_tool["server_url"] = redact_base_url(
                        sanitized_tool["server_url"]
                    )
                sanitized_tools.append(sanitized_tool)
            logged["tools"] = sanitized_tools
        return logged

    async def _refine_messages(
        self, params: dict, req_log: Optional[dict] = None,
    ) -> dict:
        if self.refiner:
            original_messages = params["messages"]
            refined = await self.refiner.refine(params["messages"])
            if req_log:
                req_log["contextRefinement"] = {
                    "enabled": True,
                    "originalMessages": original_messages,
                    "refinedMessages": refined,
                }
            return {**params, "messages": refined}
        return params

    async def _gather_completions(
        self, params_base: dict,
    ) -> List[Tuple[dict, str]]:
        entries = self._model_entries()

        async def call_model(entry: dict) -> Tuple[dict, str]:
            name = entry["name"]
            api_key = entry.get("api_key", "")
            model_params = {
                k: v for k, v in entry.items() if k not in ("name", "api_key")
            }
            p = {**params_base, "model": name, "api_key": api_key, **model_params}
            p.pop("stream", None)
            resp = await llm_completion(**p)
            return resp, name

        tasks = [call_model(entry) for entry in entries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes: List[Tuple[dict, str]] = []
        errors: List[Exception] = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"CONCURRENT REQUEST FAILED: {type(r).__name__}: {r}")
                errors.append(r)
            else:
                successes.append(r)

        if not successes:
            raise RuntimeError(
                f"All {len(errors)} concurrent requests failed. "
                f"First error: {type(errors[0]).__name__}: {errors[0]}"
            ) from errors[0]
        return successes

    async def _gather_responses(
        self, params_base: dict,
    ) -> List[Tuple[dict, str]]:
        entries = self._model_entries()

        async def call_model(entry: dict) -> Tuple[dict, str]:
            name = entry["name"]
            api_key = entry.get("api_key", "")
            model_params = {
                k: v for k, v in entry.items() if k not in ("name", "api_key")
            }
            params = {
                **params_base,
                "model": name,
                "api_key": api_key,
                **model_params,
            }
            params.pop("stream", None)
            # Verifier fan-out collects complete responses and replays the
            # selected one, so stream-only options must not reach the native
            # Responses call for each candidate.
            params.pop("stream_options", None)
            response = await llm_response(**params)
            return response, name

        results = await asyncio.gather(
            *(call_model(entry) for entry in entries),
            return_exceptions=True,
        )

        successes: List[Tuple[dict, str]] = []
        errors: List[Exception] = []
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                result, Exception
            ):
                raise result
            if isinstance(result, Exception):
                logger.error(
                    "CONCURRENT RESPONSES REQUEST FAILED: "
                    f"{type(result).__name__}: {result}"
                )
                errors.append(result)
            else:
                successes.append(result)

        if not successes:
            raise RuntimeError(
                f"All {len(errors)} concurrent Responses requests failed. "
                f"First error: {type(errors[0]).__name__}: {errors[0]}"
            ) from errors[0]
        return successes

    async def _pick_best(
        self,
        responses: List[Tuple[dict, str]],
        messages: list,
        req_log: Optional[dict] = None,
    ) -> Tuple[dict, str]:
        if len(responses) == 1 or not self.verifier:
            return responses[0]

        history_str = Backend.format_history(messages)

        # Drop responses with no usable Chat Completions or Responses output.
        valid_responses: List[Tuple[dict, str]] = []
        for resp, model_name in responses:
            if resp.get("choices") or resp.get("output"):
                valid_responses.append((resp, model_name))
            else:
                logger.warn(
                    f"RESPONSE model={model_name} returned empty output, skipping"
                )

        if not valid_responses:
            logger.error("All responses had empty output, falling back to first")
            return responses[0]
        if len(valid_responses) == 1:
            return valid_responses[0]

        actions = [Backend.format_action(resp) for resp, _ in valid_responses]
        for (_, model_name), action in zip(valid_responses, actions):
            logger.info(f"RESPONSE model={model_name} text='{action[:50]}'")

        try:
            result = await self.verifier.select_best(history_str, actions)
        except Exception as e:
            logger.error(
                f"Verifier failed ({type(e).__name__}: {e}); "
                f"falling back to first response"
            )
            return valid_responses[0]

        best_idx = result.best_index
        if (
            isinstance(best_idx, bool)
            or not isinstance(best_idx, int)
            or not 0 <= best_idx < len(valid_responses)
        ):
            logger.error(
                f"Verifier returned invalid best index {best_idx!r}; "
                "falling back to first response"
            )
            return valid_responses[0]
        verifier_scores = [
            {
                "index": i,
                "model": valid_responses[i][1],
                "score": result.scores[i] if i < len(result.scores) else 0.0,
                "details": {
                    "score": result.scores[i] if i < len(result.scores) else 0.0,
                    "criterionScores": [],
                },
            }
            for i in range(len(valid_responses))
        ]
        for s in verifier_scores:
            logger.info(f"VERIFY model={s['model']} score={s['score']:.3f}")

        best_resp, best_model = valid_responses[best_idx]
        best_score = result.scores[best_idx] if best_idx < len(result.scores) else 0.0
        logger.info(f"BEST model={best_model} score={best_score:.3f}")

        if req_log:
            req_log["verifier"] = {
                "enabled": True,
                "scores": verifier_scores,
                "comparisons": [c.to_dict() for c in result.comparisons],
                "bestIndex": best_idx,
                "bestModel": best_model,
                "bestScore": best_score,
            }

        return best_resp, best_model

    def _spawn_progress(
        self, messages: list, final_response: Optional[dict], req_log: dict,
    ) -> None:
        """Kick off progress evaluation in the background so it never delays the
        client's response. When it finishes it updates req_log and re-saves the
        log file (already written once without progress)."""
        if not self.progress_monitor:
            return
        log_dir = self.config.log_dir

        async def _run() -> None:
            await self._evaluate_progress(messages, final_response, req_log)
            save_request_log(req_log, log_dir)

        task = asyncio.create_task(_run())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _evaluate_progress(
        self, messages: list, final_response: Optional[dict],
        req_log: Optional[dict] = None,
    ) -> None:
        """Post-hoc progress estimate. Runs after the response is selected and
        never changes it — observability only. The score lands in the request
        log and the visualizer's progress node."""
        if not self.progress_monitor:
            return
        problem = Backend.format_history(messages)
        response_text = (
            Backend.format_action(final_response)
            if final_response and (
                final_response.get("choices") or final_response.get("output")
            )
            else "(empty response)"
        )
        try:
            result = await self.progress_monitor.evaluate(problem, response_text)
            if req_log is not None:
                req_log["progressMonitor"] = {
                    "enabled": True,
                    "score": result.score,
                    "details": result.to_dict(),
                }
        except Exception as e:
            logger.error(f"Progress monitor failed: {type(e).__name__}: {e}")
            if req_log is not None:
                req_log["progressMonitor"] = {"enabled": True, "error": str(e)}

    @staticmethod
    def format_history(messages: list) -> str:
        parts: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                rendered = Backend._responses_value_text(msg)
                if rendered:
                    parts.append(f"UNKNOWN: {rendered}")
                continue
            role_value = msg.get("role") or "unknown"
            role = str(role_value).upper()
            phase = msg.get("phase")
            if phase:
                role = f"{role}({phase})"
            content = Backend._responses_value_text(msg.get("content", ""))
            if msg.get("tool_call_id"):
                parts.append(
                    f"{role}: [tool_result {msg['tool_call_id']}: "
                    f"{content}]"
                )
            elif content:
                parts.append(f"{role}: {content}")
            tool_calls = msg.get("tool_calls") or []
            if not isinstance(tool_calls, (list, tuple)):
                tool_calls = [tool_calls]
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    rendered = Backend._responses_value_text(tool_call)
                    if rendered:
                        parts.append(f"{role}: [tool_call: {rendered}]")
                    continue
                function = tool_call.get("function") or {}
                if not isinstance(function, dict):
                    function = {"arguments": function}
                parts.append(
                    f"{role}: [tool_call: {function.get('name', '')}"
                    f"({Backend._responses_value_text(function.get('arguments', ''))})]"
                )
        return "\n\n".join(parts)

    @staticmethod
    def format_action(response: dict) -> str:
        if not isinstance(response, dict):
            return str(response) if response else "(empty response)"
        parts: List[str] = []
        if response.get("choices"):
            message = response["choices"][0]["message"]
            if message.get("content"):
                parts.append(message["content"])
            for tool_call in message.get("tool_calls", []):
                function = tool_call["function"]
                parts.append(
                    f"[tool_call: {function['name']}"
                    f"({function['arguments']})]"
                )
            return "\n".join(parts) if parts else "(empty response)"

        output = response.get("output", [])
        if not isinstance(output, list):
            return "(empty response)"
        for item in output:
            if not isinstance(item, dict):
                if item:
                    parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type == "message":
                content_items = item.get("content", [])
                if not isinstance(content_items, list):
                    continue
                item_parts_before = len(parts)
                for content in content_items:
                    if not isinstance(content, dict):
                        if content:
                            parts.append(str(content))
                        continue
                    if content.get("type") in ("output_text", "input_text"):
                        if content.get("text"):
                            parts.append(content["text"])
                    elif content.get("type") == "refusal" and content.get("refusal"):
                        parts.append(f"[refusal: {content['refusal']}]")
                if len(parts) == item_parts_before:
                    rendered = Backend._responses_value_text(item)
                    if rendered:
                        parts.append(rendered)
            elif item_type in ("function_call", "custom_tool_call"):
                parts.append(
                    f"[tool_call: {item.get('name', '')}"
                    f"({item.get('arguments', item.get('input', ''))})]"
                )
            else:
                rendered = Backend._responses_value_text(item)
                if rendered:
                    parts.append(rendered)
        return "\n".join(parts) if parts else "(empty response)"

    # ------------------------------------------------------------------
    # Anthropic-format API
    # ------------------------------------------------------------------

    def _build_anthropic_params(self, anthropic_body: dict) -> dict:
        params: dict = {
            **self._base_params(),
            "messages": AnthropicToOpenAI.messages(anthropic_body),
        }

        for key in ("max_tokens", "temperature", "top_p"):
            if key in anthropic_body:
                params[key] = anthropic_body[key]

        if anthropic_body.get("stop_sequences"):
            params["stop"] = anthropic_body["stop_sequences"]
        if "stream" in anthropic_body:
            params["stream"] = anthropic_body["stream"]
        if anthropic_body.get("tools"):
            params["tools"] = AnthropicToOpenAI.tools(anthropic_body["tools"])
        if anthropic_body.get("tool_choice"):
            params["tool_choice"] = AnthropicToOpenAI.tool_choice(
                anthropic_body["tool_choice"]
            )

        return params

    async def complete_anthropic(
        self, body: bytes | str,
    ) -> Tuple[Optional[dict], Optional[str]]:
        start = time.monotonic()
        try:
            anthropic_body = json.loads(
                body if isinstance(body, str) else body.decode()
            )
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON: {e}"

        req_log = create_request_log("anthropic", self._sanitized_config())
        req_log["request"] = anthropic_body

        params = self._build_anthropic_params(anthropic_body)
        params.pop("stream", None)
        params = await self._refine_messages(params, req_log)

        if self.verifier:
            logger.info(
                f"BACKEND sending {self.config.total_candidates} "
                f"concurrent requests (anthropic)"
            )
            responses = await self._gather_completions(params)
            req_log["responses"] = [
                {"model": m, "response": r} for r, m in responses
            ]
            response, model_name = await self._pick_best(
                responses, params["messages"], req_log,
            )
            final_result = OpenAIToAnthropic.response(response, model_name)
        else:
            logger.info(f"BACKEND calling {self.model_name} (anthropic)")
            response = await llm_completion(**params)
            req_log["responses"] = [
                {"model": self.model_name, "response": response}
            ]
            final_result = OpenAIToAnthropic.response(response, self.model_name)

        req_log["finalResponse"] = final_result
        req_log["elapsedMs"] = (time.monotonic() - start) * 1000
        save_request_log(req_log, self.config.log_dir)
        self._spawn_progress(params["messages"], response, req_log)
        return final_result, None

    async def stream_anthropic(
        self, body: bytes | str,
    ) -> AsyncIterator[str]:
        start = time.monotonic()
        anthropic_body = json.loads(
            body if isinstance(body, str) else body.decode()
        )
        req_log = create_request_log(
            "anthropic_stream", self._sanitized_config(),
        )
        req_log["request"] = anthropic_body

        params = self._build_anthropic_params(anthropic_body)
        params.pop("stream", None)
        params = await self._refine_messages(params, req_log)

        # When the verifier is active, collect all responses, verify, replay.
        if self.verifier:
            logger.info(
                f"BACKEND sending {self.config.total_candidates} concurrent "
                f"requests for verification (anthropic stream)"
            )
            responses = await self._gather_completions(params)
            req_log["responses"] = [
                {"model": m, "response": r} for r, m in responses
            ]
            best_resp, best_model = await self._pick_best(
                responses, params["messages"], req_log,
            )
            req_log["finalResponse"] = best_resp
            req_log["elapsedMs"] = (time.monotonic() - start) * 1000
            save_request_log(req_log, self.config.log_dir)
            self._spawn_progress(params["messages"], best_resp, req_log)
            async for event in self._replay_anthropic_sse(best_resp, best_model):
                yield event
            return

        logger.info(f"BACKEND streaming {self.model_name} (anthropic)")

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield SSEFormatter.message_start(self.model_name, msg_id)

        stream = await llm_stream_completion(**params)

        block_index = 0
        text_block_open = False
        tool_blocks: Dict[str, dict] = {}
        current_tool_id: Optional[str] = None
        output_tokens = 0

        try:
            async for chunk in stream:
                chunk_dict = (
                    chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                )
                choices = chunk_dict.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})

                usage = chunk_dict.get("usage")
                if usage and usage.get("completion_tokens"):
                    output_tokens = usage["completion_tokens"]

                if delta.get("content"):
                    if not text_block_open:
                        yield SSEFormatter.content_block_start(block_index, "text")
                        text_block_open = True
                    yield SSEFormatter.text_delta(block_index, delta["content"])

                if delta.get("tool_calls"):
                    for tc_delta in delta["tool_calls"]:
                        tc_id = tc_delta.get("id")

                        if tc_id and tc_id not in tool_blocks:
                            if text_block_open:
                                yield SSEFormatter.content_block_stop(block_index)
                                block_index += 1
                                text_block_open = False

                            tool_blocks[tc_id] = {
                                "index": block_index,
                                "name": tc_delta.get("function", {}).get("name", ""),
                            }
                            current_tool_id = tc_id

                            yield SSEFormatter.content_block_start(
                                block_index,
                                "tool_use",
                                tool_id=tc_id,
                                tool_name=tool_blocks[tc_id]["name"],
                            )

                        target_id = tc_id or current_tool_id
                        if target_id and target_id in tool_blocks:
                            func = tc_delta.get("function", {})
                            if func.get("arguments"):
                                yield SSEFormatter.input_json_delta(
                                    tool_blocks[target_id]["index"],
                                    func["arguments"],
                                )

                if choice.get("finish_reason"):
                    if text_block_open:
                        yield SSEFormatter.content_block_stop(block_index)
                        block_index += 1
                        text_block_open = False

                    for tinfo in tool_blocks.values():
                        yield SSEFormatter.content_block_stop(tinfo["index"])
                        block_index += 1

                    stop_reason = STOP_REASON_MAP.get(
                        choice["finish_reason"], "end_turn"
                    )
                    yield SSEFormatter.message_delta(stop_reason, output_tokens)
                    yield SSEFormatter.message_stop()
        except BaseException as exc:
            await _close_upstream_stream(
                stream, None if isinstance(exc, GeneratorExit) else exc,
            )
            raise
        else:
            await _close_upstream_stream(stream)

    async def _replay_anthropic_sse(
        self, response: dict, model_name: str,
    ) -> AsyncIterator[str]:
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield SSEFormatter.message_start(model_name, msg_id)

        choice = response["choices"][0]
        message = choice["message"]
        block_index = 0

        if message.get("content"):
            yield SSEFormatter.content_block_start(block_index, "text")
            yield SSEFormatter.text_delta(block_index, message["content"])
            yield SSEFormatter.content_block_stop(block_index)
            block_index += 1

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                yield SSEFormatter.content_block_start(
                    block_index,
                    "tool_use",
                    tool_id=tc.get("id", ""),
                    tool_name=tc["function"]["name"],
                )
                if tc["function"].get("arguments"):
                    yield SSEFormatter.input_json_delta(
                        block_index, tc["function"]["arguments"],
                    )
                yield SSEFormatter.content_block_stop(block_index)
                block_index += 1

        stop_reason = STOP_REASON_MAP.get(
            choice.get("finish_reason", "stop"), "end_turn",
        )
        output_tokens = response.get("usage", {}).get("completion_tokens", 0)
        yield SSEFormatter.message_delta(stop_reason, output_tokens)
        yield SSEFormatter.message_stop()

    # ------------------------------------------------------------------
    # OpenAI-format API
    # ------------------------------------------------------------------

    def _build_openai_params(self, openai_body: dict) -> dict:
        params: dict = {
            **self._base_params(),
            "messages": openai_body.get("messages", []),
        }

        direct_keys = [
            "temperature", "top_p", "stop", "tools", "tool_choice",
            "response_format", "seed", "n", "presence_penalty",
            "frequency_penalty", "logit_bias",
        ]
        for key in direct_keys:
            if key in openai_body:
                params[key] = openai_body[key]

        if openai_body.get("max_tokens"):
            params["max_tokens"] = openai_body["max_tokens"]
        if openai_body.get("max_completion_tokens"):
            params["max_completion_tokens"] = openai_body["max_completion_tokens"]
        if "stream" in openai_body:
            params["stream"] = openai_body["stream"]
        if openai_body.get("stream_options"):
            params["stream_options"] = openai_body["stream_options"]

        return params

    async def complete_openai(
        self, body: bytes | str,
    ) -> Tuple[Optional[dict], Optional[str]]:
        start = time.monotonic()
        try:
            openai_body = json.loads(
                body if isinstance(body, str) else body.decode()
            )
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON: {e}"

        req_log = create_request_log("openai", self._sanitized_config())
        req_log["request"] = openai_body

        params = self._build_openai_params(openai_body)
        params.pop("stream", None)
        params = await self._refine_messages(params, req_log)

        if self.verifier:
            logger.info(
                f"BACKEND sending {self.config.total_candidates} "
                f"concurrent requests (openai)"
            )
            responses = await self._gather_completions(params)
            req_log["responses"] = [
                {"model": m, "response": r} for r, m in responses
            ]
            response, _ = await self._pick_best(
                responses, params["messages"], req_log,
            )
            final_result = response
        else:
            logger.info(f"BACKEND calling {self.model_name} (openai)")
            final_result = await llm_completion(**params)
            req_log["responses"] = [
                {"model": self.model_name, "response": final_result}
            ]

        req_log["finalResponse"] = final_result
        req_log["elapsedMs"] = (time.monotonic() - start) * 1000
        save_request_log(req_log, self.config.log_dir)
        self._spawn_progress(params["messages"], final_result, req_log)
        return final_result, None

    async def stream_openai(
        self, body: bytes | str,
    ) -> AsyncIterator[str]:
        start = time.monotonic()
        openai_body = json.loads(
            body if isinstance(body, str) else body.decode()
        )
        req_log = create_request_log("openai_stream", self._sanitized_config())
        req_log["request"] = openai_body

        params = self._build_openai_params(openai_body)
        params.pop("stream", None)
        params = await self._refine_messages(params, req_log)

        if self.verifier:
            logger.info(
                f"BACKEND sending {self.config.total_candidates} concurrent "
                f"requests for verification (openai stream)"
            )
            responses = await self._gather_completions(params)
            req_log["responses"] = [
                {"model": m, "response": r} for r, m in responses
            ]
            best_resp, _ = await self._pick_best(
                responses, params["messages"], req_log,
            )
            req_log["finalResponse"] = best_resp
            req_log["elapsedMs"] = (time.monotonic() - start) * 1000
            save_request_log(req_log, self.config.log_dir)
            self._spawn_progress(params["messages"], best_resp, req_log)
            async for event in self._replay_openai_sse(best_resp):
                yield event
            return

        logger.info(f"BACKEND streaming {self.model_name} (openai)")

        stream = await llm_stream_completion(**params)
        try:
            async for chunk in stream:
                chunk_dict = (
                    chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                )
                yield f"data: {json.dumps(chunk_dict, default=str)}\n\n"
        except BaseException as exc:
            await _close_upstream_stream(
                stream, None if isinstance(exc, GeneratorExit) else exc,
            )
            raise
        else:
            await _close_upstream_stream(stream)

        yield "data: [DONE]\n\n"

    async def _replay_openai_sse(
        self, response: dict,
    ) -> AsyncIterator[str]:
        choices = response.get("choices", [])
        choice = choices[0] if choices else {}

        chunk = {
            "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            "object": "chat.completion.chunk",
            "created": response.get("created", 0),
            "model": response.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "delta": choice.get("message", {}),
                    "finish_reason": choice.get("finish_reason"),
                },
            ],
        }
        yield f"data: {json.dumps(chunk, default=str)}\n\n"
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # OpenAI Responses API
    # ------------------------------------------------------------------

    @staticmethod
    def parse_responses_body(body: bytes | str) -> dict:
        """Parse an RFC-compliant Responses JSON body."""
        def reject_nonfinite_number(value: str) -> None:
            raise ValueError(f"non-finite number {value} is not valid JSON")

        try:
            payload = json.loads(
                body if isinstance(body, str) else body.decode(),
                parse_constant=reject_nonfinite_number,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        return payload

    def _build_responses_params(self, responses_body: dict) -> dict:
        params: dict = self._base_params()
        if "input" in responses_body:
            params["input"] = responses_body["input"]
        # Keep this list aligned with the public Responses request schema.
        # Unknown client metadata must not leak into LiteLLM's routing kwargs,
        # while provider-specific values can still be carried by extra_body.
        direct_keys = (
            "include", "instructions", "prompt", "metadata", "conversation",
            "parallel_tool_calls", "previous_response_id", "reasoning",
            "store", "background", "temperature", "text", "tool_choice",
            "tools", "top_p", "truncation", "user", "service_tier",
            "safety_identifier", "stream_options", "top_logprobs",
            "max_tool_calls", "prompt_cache_key", "prompt_cache_retention",
            "prompt_cache_options", "context_management", "moderation",
            "partial_images", "thinking", "text_format", "response_format",
            "extra_body", "extra_headers", "extra_query", "timeout",
            "allowed_openai_params",
        )
        for key in direct_keys:
            if key in responses_body:
                params[key] = responses_body[key]
        if "max_output_tokens" in responses_body:
            params["max_output_tokens"] = responses_body["max_output_tokens"]
        elif "max_tokens" in responses_body:
            # ``max_tokens`` was used by older OpenAI-compatible clients.
            # The Responses wrapper normalizes it to max_output_tokens.
            params["max_tokens"] = responses_body["max_tokens"]
        if "stream" in responses_body:
            params["stream"] = responses_body["stream"]
        return params

    def validate_responses_body(self, responses_body: dict) -> None:
        """Validate request shape and providers before any Responses I/O."""
        _validate_responses_request(responses_body)
        if self.verifier:
            incompatible = []
            for field in ("conversation", "previous_response_id", "prompt"):
                if responses_body.get(field) is not None:
                    incompatible.append(field)
            if self._responses_input_contains_item_reference(
                responses_body.get("input")
            ):
                incompatible.append("input.item_reference")
            if incompatible:
                raise ValueError(
                    "Responses parameter(s) "
                    + ", ".join(incompatible)
                    + " cannot be used when verifier is enabled"
                )
        params_base = self._build_responses_params(responses_body)
        params_base.pop("stream", None)
        entries = self._model_entries() if self.verifier else [
            {
                "name": self.model_name,
                "api_key": self.api_key,
                **self._parse_model_params(self.config.default_model),
                "base_url": self.config.default_model.get("base_url") or None,
                "provider": self.config.default_model.get("provider"),
            }
        ]
        non_native = []
        for entry in entries:
            name = entry["name"]
            api_key = entry.get("api_key", "")
            params = {
                **params_base,
                "model": name,
                "api_key": api_key,
                **{
                    key: value
                    for key, value in entry.items()
                    if key not in ("name", "api_key", "num_candidates")
                },
            }
            uses_chat_fallback, resolved_provider = _responses_provider_uses_chat_fallback(
                name,
                params.get("provider"),
                params.get("base_url"),
            )
            if uses_chat_fallback:
                non_native.append(
                    f"provider={resolved_provider!r}, model={name!r}"
                )

        if non_native:
            raise ValueError(
                "/v1/responses requires a native Responses provider; "
                + "; ".join(non_native)
            )

        for entry in entries:
            name = entry["name"]
            api_key = entry.get("api_key", "")
            params = {
                **params_base,
                "model": name,
                "api_key": api_key,
                **{
                    key: value
                    for key, value in entry.items()
                    if key not in ("name", "api_key", "num_candidates")
                },
            }
            _build_responses_kwargs(**params)

    @staticmethod
    def _responses_input_contains_item_reference(value: Any) -> bool:
        """Detect Responses references that cannot be shared across verifier candidates."""
        if isinstance(value, dict):
            if value.get("type") == "item_reference":
                return True
            return any(
                Backend._responses_input_contains_item_reference(child)
                for child in value.values()
            )
        if isinstance(value, (list, tuple)):
            return any(
                Backend._responses_input_contains_item_reference(child)
                for child in value
            )
        return False

    @staticmethod
    def _responses_value_text(value: Any) -> str:
        """Render arbitrary Responses content into stable verifier text."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return json.dumps(value)
        if isinstance(value, (list, tuple)):
            rendered = [Backend._responses_value_text(item) for item in value]
            return "\n".join(part for part in rendered if part)
        if isinstance(value, dict):
            item_type = value.get("type")
            if item_type in (
                "text", "input_text", "output_text", "summary_text",
                "reasoning_text",
            ):
                return Backend._responses_value_text(value.get("text", ""))
            if item_type == "refusal":
                refusal = Backend._responses_value_text(value.get("refusal", ""))
                return f"[refusal: {refusal}]" if refusal else "[refusal]"
            if item_type in (
                "image_url", "input_image", "output_image",
                "computer_screenshot",
            ):
                return "[image]"
            if item_type == "image_generation_call":
                status = value.get("status")
                suffix = f" status={status}" if status else ""
                return f"[image_generation_call{suffix}]"
            if item_type in ("input_file", "file"):
                return "[file]"
            if item_type == "reasoning":
                summary = value.get("summary")
                rendered = Backend._responses_value_text(summary)
                return f"[reasoning: {rendered}]" if rendered else "[reasoning]"
            # Tool outputs frequently use a nested ``content``/``output``
            # value. Prefer that readable payload while retaining the type.
            for key in (
                "content", "output", "input", "arguments", "summary",
                "action", "query", "results", "code", "transcript", "text",
            ):
                if key in value and value[key] not in (None, "", [], {}):
                    rendered = Backend._responses_value_text(value[key])
                    if rendered:
                        return f"[{item_type or 'item'}: {rendered}]"
            try:
                return json.dumps(
                    value, sort_keys=True, separators=(",", ":"),
                    default=str,
                )
            except (TypeError, ValueError):
                return str(value)
        return str(value)

    @staticmethod
    def _responses_history_content(content: Any) -> Any:
        """Keep rich input blocks while making malformed values readable."""
        if isinstance(content, (str, list)):
            return content
        if content is None:
            return ""
        if isinstance(content, dict):
            return [content]
        return Backend._responses_value_text(content)

    @staticmethod
    def _responses_history(responses_body: dict) -> list:
        history: list = []
        instructions = responses_body.get("instructions")
        if instructions:
            history.append({"role": "system", "content": instructions})

        response_input = responses_body.get("input", "")
        if isinstance(response_input, str):
            if response_input:
                history.append({"role": "user", "content": response_input})
            return history
        if not isinstance(response_input, list):
            return history

        for item in response_input:
            if not isinstance(item, dict):
                history.append({"role": "user", "content": str(item)})
                continue
            raw_item_type = item.get("type")
            if raw_item_type is None:
                item_type = "message"
            elif isinstance(raw_item_type, str):
                item_type = raw_item_type
            else:
                item_type = str(raw_item_type)
            if item_type == "message":
                history.append({
                    "role": item.get("role", "user"),
                    "content": Backend._responses_history_content(
                        item.get("content", "")
                    ),
                })
            elif item_type in (
                "input_text", "input_image", "input_file",
                "output_text", "refusal",
            ):
                # Responses also permits content items directly in the input
                # array. Preserve them as user content for refinement and
                # verification instead of silently dropping them.
                history.append({"role": "user", "content": [item]})
            elif item_type == "function_call":
                history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": item.get("call_id") or item.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": Backend._responses_value_text(
                                item.get("arguments", "")
                            ),
                        },
                    }],
                })
            elif item_type == "function_call_output":
                history.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": Backend._responses_value_text(
                        item.get("output", "")
                    ) or "[empty tool output]",
                })
            elif item_type == "reasoning":
                summary = Backend._responses_value_text(item.get("summary"))
                history.append({
                    "role": "assistant",
                    "content": f"[reasoning summary: {summary}]"
                    if summary else "[reasoning]",
                })
            elif item_type in (
                "web_search_call", "file_search_call", "computer_call",
                "code_interpreter_call", "image_generation_call",
                "mcp_call", "custom_tool_call",
            ):
                arguments = item.get(
                    "arguments",
                    item.get(
                        "input",
                        item.get(
                            "action",
                            item.get("query", item.get("results", "")),
                        ),
                    ),
                )
                label = Backend._responses_value_text(arguments)
                history.append({
                    "role": "assistant",
                    "content": (
                        f"[{item_type}: {item.get('name', '')}"
                        f"({label}) status={item.get('status', 'unknown')}]"
                    ),
                })
            elif item_type.endswith("_output"):
                # Responses tool outputs (computer/local-shell/MCP/custom)
                # are semantically tool messages even when their payload is a
                # list or nested object.
                output = item.get("output", item.get("content", item))
                history.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id", ""),
                    "content": Backend._responses_value_text(output)
                    or f"[{item_type}: empty]",
                })
            else:
                # Keep unknown future Responses items visible to verifier and
                # context refinement instead of dropping them silently.
                role = item.get("role") or (
                    "tool" if "output" in item else "user"
                )
                history.append({
                    "role": role,
                    "content": Backend._responses_value_text(item)
                    or f"[{item_type or 'item'}]",
                })
        return history

    async def _refine_responses(
        self, params: dict, history: list, req_log: Optional[dict] = None,
    ) -> Tuple[dict, list]:
        if not self.refiner or not history:
            return params, history
        if any(
            params.get(field) is not None
            for field in ("previous_response_id", "conversation", "prompt")
        ):
            logger.info(
                "Skipping context refinement for a Responses request with "
                "referenced context"
            )
            return params, history

        refined = await self.refiner.refine(history)
        if not isinstance(refined, list) or any(
            not isinstance(message, dict) for message in refined
        ):
            logger.error(
                "Context refinement returned an invalid message list; "
                "skipping refinement"
            )
            refined = history
        if req_log is not None:
            req_log["contextRefinement"] = {
                "enabled": True,
                "originalMessages": history,
                "refinedMessages": refined,
            }

        if refined and refined[0].get("role") == "system":
            refined_instructions = refined[0].get("content") or ""
            if not params.get("instructions") and history and (
                history[0].get("role") == "system"
            ):
                original = history[0].get("content") or ""
                suffix = f"\n\n{original}"
                if original and refined_instructions.endswith(suffix):
                    refined_instructions = refined_instructions[:-len(suffix)]
            params = {**params, "instructions": refined_instructions}
        return params, refined

    async def complete_responses(
        self, body: bytes | str,
    ) -> Tuple[Optional[dict], Optional[str]]:
        start = time.monotonic()
        try:
            responses_body = self.parse_responses_body(body)
        except ValueError as exc:
            return None, str(exc)
        try:
            self.validate_responses_body(responses_body)
        except ValueError as exc:
            return None, str(exc)

        req_log = create_request_log("responses", self._sanitized_config())
        req_log["request"] = self._responses_request_for_log(responses_body)
        params = self._build_responses_params(responses_body)
        params.pop("stream", None)
        history = self._responses_history(responses_body)
        params, history = await self._refine_responses(params, history, req_log)

        if self.verifier:
            logger.info(
                f"BACKEND sending {self.config.total_candidates} "
                "concurrent requests (responses)"
            )
            responses = await self._gather_responses(params)
            req_log["responses"] = [
                {"model": model, "response": response}
                for response, model in responses
            ]
            final_result, _ = await self._pick_best(
                responses, history, req_log,
            )
        else:
            logger.info(f"BACKEND calling {self.model_name} (responses)")
            final_result = await llm_response(**params)
            req_log["responses"] = [
                {"model": self.model_name, "response": final_result}
            ]

        req_log["finalResponse"] = final_result
        req_log["elapsedMs"] = (time.monotonic() - start) * 1000
        save_request_log(req_log, self.config.log_dir)
        self._spawn_progress(history, final_result, req_log)
        return final_result, None

    async def stream_responses(
        self, body: bytes | str,
    ) -> AsyncIterator[str]:
        start = time.monotonic()
        responses_body = self.parse_responses_body(body)
        self.validate_responses_body(responses_body)

        req_log = create_request_log(
            "responses_stream", self._sanitized_config(),
        )
        req_log["request"] = self._responses_request_for_log(responses_body)
        params = self._build_responses_params(responses_body)
        params.pop("stream", None)
        history = self._responses_history(responses_body)
        params, history = await self._refine_responses(params, history, req_log)

        if self.verifier:
            logger.info(
                f"BACKEND sending {self.config.total_candidates} concurrent "
                "requests for verification (responses stream)"
            )
            responses = await self._gather_responses(params)
            req_log["responses"] = [
                {"model": model, "response": response}
                for response, model in responses
            ]
            best_response, _ = await self._pick_best(
                responses, history, req_log,
            )
            req_log["finalResponse"] = best_response
            req_log["elapsedMs"] = (time.monotonic() - start) * 1000
            save_request_log(req_log, self.config.log_dir)
            self._spawn_progress(history, best_response, req_log)
            async for event in self._replay_responses_sse(best_response):
                yield event
            return

        logger.info(f"BACKEND streaming {self.model_name} (responses)")
        stream = await llm_stream_response(**params)
        stream_state = {
            "sequence": 0,
            "response_id": None,
            "created_at": None,
            "items_by_id": {},
            "item_indexes": {},
            "output_indexes": {},
            "next_output_index": 0,
            "done_items": {},
        }
        try:
            async for event in stream:
                event_dict = (
                    self._dump_responses_event(event)
                )
                if event_dict == "[DONE]":
                    continue
                if not isinstance(event_dict, dict):
                    raise TypeError(
                        "Responses stream yielded a non-object event"
                    )
                event_dict = self._normalise_live_responses_event(
                    event_dict, stream_state,
                )
                if event_dict is None:
                    continue
                yield self._format_responses_sse(event_dict)
        except BaseException as exc:
            await _close_upstream_stream(
                stream, None if isinstance(exc, GeneratorExit) else exc,
            )
            raise
        else:
            await _close_upstream_stream(stream)

    @staticmethod
    def _format_responses_sse(event: dict) -> str:
        event_type = event.get("type", "response.event")
        if isinstance(event_type, Enum):
            event_type = event_type.value
        if not isinstance(event_type, str):
            event_type = str(event_type)
        if event_type == "response.cancelled":
            # Response.status supports cancelled, but the streaming event union
            # has no response.cancelled discriminator. Use the standard failed
            # envelope while preserving the response's cancelled status.
            event_type = "response.failed"
            event = {**event, "type": event_type}
        return (
            f"event: {event_type}\n"
            f"data: {json.dumps(event, default=Backend._responses_json_default)}\n\n"
        )

    @staticmethod
    def _dump_responses_event(event: Any) -> Any:
        if not hasattr(event, "model_dump"):
            return event
        try:
            return event.model_dump(mode="json")
        except TypeError:
            return event.model_dump()

    def _normalise_live_responses_event(
        self, event: dict, state: dict,
    ) -> Optional[dict]:
        """Normalize one live upstream event while retaining stream state."""
        event = dict(event)
        event_type = event.get("type")
        if isinstance(event_type, Enum):
            event_type = event_type.value
        if not isinstance(event_type, str):
            event_type = str(event_type or "response.event")
        event_aliases = {
            "RESPONSE_CREATED": "response.created",
            "RESPONSE_QUEUED": "response.queued",
            "RESPONSE_IN_PROGRESS": "response.in_progress",
            "RESPONSE_COMPLETED": "response.completed",
            "RESPONSE_INCOMPLETE": "response.incomplete",
            "RESPONSE_FAILED": "response.failed",
            "RESPONSE_CANCELLED": "response.cancelled",
        }
        event_type = event_aliases.get(
            event_type.rsplit(".", 1)[-1], event_type,
        )
        event["type"] = event_type

        if event_type == "response.output_item.added":
            item = self._dump_responses_event(event.get("item"))
            if not isinstance(item, dict):
                return None
            event["item"] = item
            item_id = item.get("id")
            source_index = event.get("output_index")
            valid_source_index = (
                isinstance(source_index, int)
                and not isinstance(source_index, bool)
                and source_index >= 0
            )
            mapped_index = (
                state["output_indexes"].get(source_index)
                if valid_source_index
                else None
            )
            if mapped_index is None and isinstance(item_id, str) and item_id:
                mapped_index = state["item_indexes"].get(item_id)
            if mapped_index is None and valid_source_index:
                mapped_index = source_index
            if mapped_index is None:
                mapped_index = state["next_output_index"]
                state["next_output_index"] += 1
            else:
                state["next_output_index"] = max(
                    state["next_output_index"], mapped_index + 1,
                )
            if valid_source_index:
                state["output_indexes"][source_index] = mapped_index
            event["output_index"] = mapped_index
            if isinstance(item_id, str) and item_id:
                state["items_by_id"][item_id] = item
                state["item_indexes"][item_id] = mapped_index
        elif "output_index" in event:
            source_index = event.get("output_index")
            mapped_index = (
                state["output_indexes"].get(source_index)
                if isinstance(source_index, int)
                and not isinstance(source_index, bool)
                and source_index >= 0
                else None
            )
            item_id = event.get("item_id")
            if not isinstance(item_id, str) or not item_id:
                item = self._dump_responses_event(event.get("item"))
                item_id = item.get("id") if isinstance(item, dict) else None
            if mapped_index is None and isinstance(item_id, str) and item_id:
                mapped_index = state["item_indexes"].get(item_id)
            if mapped_index is None:
                if (
                    not isinstance(source_index, int)
                    or isinstance(source_index, bool)
                    or source_index < 0
                ):
                    return None
                mapped_index = source_index
            event["output_index"] = mapped_index

        if event_type in {
            "response.output_text.delta", "response.output_text.done",
        }:
            logprobs = event.get("logprobs")
            event["logprobs"] = logprobs if isinstance(logprobs, list) else []
            text_key = "delta" if event_type.endswith(".delta") else "text"
            event[text_key] = self._responses_value_text(event.get(text_key, ""))

        if event_type in {
            "response.content_part.added", "response.content_part.done",
        }:
            part = self._dump_responses_event(event.get("part"))
            if isinstance(part, dict) and part.get("type") == "output_text":
                part = dict(part)
                part["text"] = self._responses_value_text(part.get("text", ""))
                annotations = part.get("annotations")
                part["annotations"] = (
                    list(annotations)
                    if isinstance(annotations, (list, tuple))
                    else []
                )
                logprobs = part.get("logprobs")
                if logprobs is not None and not isinstance(logprobs, list):
                    part["logprobs"] = []
                event["part"] = part

        if event_type == "response.function_call_arguments.done":
            name = event.get("name")
            if not isinstance(name, str) or not name:
                item_id = event.get("item_id")
                item = state["items_by_id"].get(item_id)
                if isinstance(item, dict):
                    item_name = item.get("name")
                    if isinstance(item_name, str) and item_name:
                        event["name"] = item_name

        if event_type == "response.output_item.done":
            item = self._dump_responses_event(event.get("item"))
            items = self._normalise_responses_output(
                [item], response_status="completed",
                preserve_empty_messages=True,
            )
            if not items:
                return None
            item = items[0]
            event["item"] = item
            output_index = event.get("output_index")
            if not isinstance(output_index, int) or isinstance(
                output_index, bool
            ) or output_index < 0:
                output_index = len(state["done_items"])
                event["output_index"] = output_index
            state["done_items"][output_index] = item

        response = self._dump_responses_event(event.get("response"))
        if isinstance(response, dict):
            terminal_statuses = {
                "response.completed": "completed",
                "response.incomplete": "incomplete",
                "response.failed": "failed",
                "response.cancelled": "cancelled",
            }
            progress_statuses = {
                "response.created": response.get("status") or "in_progress",
                "response.queued": "queued",
                "response.in_progress": "in_progress",
            }
            status = terminal_statuses.get(
                event_type, progress_statuses.get(event_type)
            )
            if event_type == "response.failed" and response.get("status") == (
                "cancelled"
            ):
                status = "cancelled"
            if status is None:
                status = response.get("status") or "in_progress"
            if isinstance(status, Enum):
                status = status.value

            raw_output = response.get("output")
            if not isinstance(raw_output, list):
                raw_output = []
            if event_type in terminal_statuses:
                indexed_output = {
                    index: item for index, item in enumerate(raw_output)
                }
                for index, item in state["done_items"].items():
                    indexed_output.setdefault(index, item)
                raw_output = [
                    indexed_output[index]
                    for index in sorted(indexed_output)
                ]
            output = self._normalise_responses_output(
                raw_output,
                response_status=status,
                preserve_empty_messages=bool(state["item_indexes"]),
            )
            response = self._normalise_responses_envelope(
                response,
                status=status,
                output=output,
                in_progress=event_type not in terminal_statuses,
            )
            if state["response_id"] is None:
                state["response_id"] = response["id"]
            else:
                response["id"] = state["response_id"]
            if state["created_at"] is None:
                state["created_at"] = response["created_at"]
            else:
                response["created_at"] = state["created_at"]
            event["response"] = response

        event["sequence_number"] = state["sequence"]
        state["sequence"] += 1
        return event

    @staticmethod
    def _responses_json_default(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump(mode="json")
            except TypeError:
                return value.model_dump()
        return str(value)

    def _normalise_responses_envelope(
        self, response: Any, *, status: Optional[str] = None,
        output: Optional[list] = None, in_progress: bool = False,
    ) -> dict:
        """Return an OpenAI SDK-compatible response envelope."""
        if hasattr(response, "model_dump"):
            try:
                response = response.model_dump(mode="json")
            except TypeError:
                response = response.model_dump()
        source = dict(response) if isinstance(response, dict) else {}
        resolved_status = status or source.get("status") or "completed"
        created_at = source.get("created_at")
        if not isinstance(created_at, (int, float)) or isinstance(created_at, bool):
            created_at = time.time()
        parallel_tool_calls = source.get("parallel_tool_calls")
        if not isinstance(parallel_tool_calls, bool):
            parallel_tool_calls = True
        tool_choice = source.get("tool_choice") or "auto"
        usage = source.get("usage")
        if isinstance(usage, dict):
            usage = dict(usage)
            input_tokens = usage.get("input_tokens")
            if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
                input_tokens = 0
            output_tokens = usage.get("output_tokens")
            if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
                output_tokens = 0
            total_tokens = usage.get("total_tokens")
            if not isinstance(total_tokens, int) or isinstance(total_tokens, bool):
                total_tokens = input_tokens + output_tokens
            input_details = usage.get("input_tokens_details")
            if not isinstance(input_details, dict):
                input_details = {}
            cached_tokens = input_details.get("cached_tokens")
            if not isinstance(cached_tokens, int) or isinstance(
                cached_tokens, bool
            ):
                cached_tokens = 0
            output_details = usage.get("output_tokens_details")
            if not isinstance(output_details, dict):
                output_details = {}
            reasoning_tokens = output_details.get("reasoning_tokens")
            if not isinstance(reasoning_tokens, int) or isinstance(
                reasoning_tokens, bool
            ):
                reasoning_tokens = 0
            usage.update({
                "input_tokens": input_tokens,
                "input_tokens_details": {
                    **input_details,
                    "cached_tokens": cached_tokens,
                },
                "output_tokens": output_tokens,
                "output_tokens_details": {
                    **output_details,
                    "reasoning_tokens": reasoning_tokens,
                },
                "total_tokens": total_tokens,
            })
        envelope = {
            **source,
            "id": source.get("id") or f"resp_{uuid.uuid4().hex[:24]}",
            "object": "response",
            "created_at": created_at,
            "model": source.get("model") or self.model_name,
            "status": resolved_status,
            "output": [] if output is None else output,
            "parallel_tool_calls": parallel_tool_calls,
            "tool_choice": tool_choice,
            "tools": source.get("tools") or [],
        }
        if isinstance(usage, dict):
            envelope["usage"] = usage
        if in_progress:
            # Terminal fields describe the selected non-streaming result and
            # must not appear in the created/in-progress snapshots.
            for key in (
                "completed_at", "error", "incomplete_details", "usage",
                "output_text", "audio", "audio_transcript",
            ):
                if key in envelope:
                    envelope[key] = None
        return envelope

    def _normalise_responses_output(
        self, output: Any, *, response_status: str,
        preserve_empty_messages: bool = False,
    ) -> list:
        """Normalize output items for the OpenAI Responses event schema."""
        if not isinstance(output, list):
            return []

        normalised = []
        for raw_item in output:
            if hasattr(raw_item, "model_dump"):
                try:
                    raw_item = raw_item.model_dump(mode="json")
                except TypeError:
                    raw_item = raw_item.model_dump()
            if not isinstance(raw_item, dict):
                continue

            item = dict(raw_item)
            item_type = item.get("type")
            if isinstance(item_type, Enum):
                item_type = item_type.value
                item["type"] = item_type

            if item_type == "message":
                content = []
                raw_content = item.get("content")
                if isinstance(raw_content, list):
                    for raw_part in raw_content:
                        if hasattr(raw_part, "model_dump"):
                            try:
                                raw_part = raw_part.model_dump(mode="json")
                            except TypeError:
                                raw_part = raw_part.model_dump()
                        if not isinstance(raw_part, dict):
                            continue
                        part = dict(raw_part)
                        part_type = part.get("type")
                        if isinstance(part_type, Enum):
                            part_type = part_type.value
                            part["type"] = part_type
                        if part_type == "output_text":
                            if part.get("text") is None:
                                if not preserve_empty_messages:
                                    continue
                                part["text"] = ""
                            part["text"] = self._responses_value_text(
                                part.get("text", "")
                            )
                            annotations = part.get("annotations")
                            part["annotations"] = (
                                list(annotations)
                                if isinstance(annotations, (list, tuple))
                                else []
                            )
                            logprobs = part.get("logprobs")
                            if logprobs is not None and not isinstance(
                                logprobs, list
                            ):
                                part["logprobs"] = []
                        elif part_type == "refusal":
                            if part.get("refusal") is None:
                                if not preserve_empty_messages:
                                    continue
                                part["refusal"] = ""
                            part["refusal"] = self._responses_value_text(
                                part.get("refusal", "")
                            )
                        content.append(part)

                # LiteLLM emits a message with output_text.text=None beside a
                # pure function call. It carries no output and is not valid in
                # the OpenAI SDK union.
                if not content and not preserve_empty_messages:
                    continue
                item["content"] = content
                item["id"] = item.get("id") or (
                    f"msg_{uuid.uuid4().hex[:24]}"
                )
                item["role"] = item.get("role") or "assistant"
                item_status = item.get("status")
                progress_status = response_status in {"queued", "in_progress"}
                if item_status not in {
                    "in_progress", "completed", "incomplete",
                } or (item_status == "in_progress" and not progress_status):
                    item["status"] = (
                        "completed"
                        if response_status == "completed"
                        else "in_progress" if progress_status else "incomplete"
                    )
            elif item_type == "function_call":
                item_id = item.get("id") or f"fc_{uuid.uuid4().hex[:24]}"
                item["id"] = item_id
                item["call_id"] = item.get("call_id") or item_id
                item["name"] = self._responses_value_text(item.get("name", ""))
                item["arguments"] = self._responses_value_text(
                    item.get("arguments", "")
                )

            normalised.append(item)
        return normalised

    async def _replay_responses_sse(
        self, response: dict,
    ) -> AsyncIterator[str]:
        sequence = 0

        def make_event(event_type: str, **payload: Any) -> str:
            nonlocal sequence
            event = {
                "type": event_type,
                "sequence_number": sequence,
                **payload,
            }
            sequence += 1
            return self._format_responses_sse(event)

        raw_status = response.get("status")
        if isinstance(raw_status, Enum):
            raw_status = raw_status.value
        valid_statuses = {
            "completed", "failed", "in_progress", "cancelled",
            "queued", "incomplete",
        }
        if raw_status is None:
            status = "completed"
        elif raw_status in valid_statuses:
            status = raw_status
        else:
            raise ValueError(f"Unsupported Responses status: {raw_status!r}")
        response_base = self._normalise_responses_envelope(
            response,
            status=status,
            output=[],
        )
        in_progress = self._normalise_responses_envelope(
            response_base,
            status="queued" if status == "queued" else "in_progress",
            output=[],
            in_progress=True,
        )
        yield make_event("response.created", response=in_progress)
        if status in {"queued", "in_progress"}:
            # Background snapshots are not terminal. They may contain partial
            # output, but replaying a non-streaming snapshot as output_item.done
            # would falsely finalize it. Callers can poll the response id.
            event_type = (
                "response.queued"
                if status == "queued"
                else "response.in_progress"
            )
            yield make_event(event_type, response=in_progress)
            return
        yield make_event("response.in_progress", response=in_progress)

        final_output: list = []
        response_output = response.get("output", [])
        if not isinstance(response_output, list):
            response_output = []
        statusless_item_types = {
            "custom_tool_call", "program", "mcp_list_tools",
            "mcp_approval_request", "mcp_approval_response", "compaction",
            "additional_tools",
        }
        terminal_status_item_types = {
            "apply_patch_call_output", "program_output",
        }
        item_status_values = {
            "message": {"in_progress", "completed", "incomplete"},
            "file_search_call": {
                "in_progress", "searching", "completed", "incomplete",
                "failed",
            },
            "function_call": {"in_progress", "completed", "incomplete"},
            "function_call_output": {
                "in_progress", "completed", "incomplete",
            },
            "web_search_call": {
                "in_progress", "searching", "completed", "failed",
            },
            "computer_call": {"in_progress", "completed", "incomplete"},
            "computer_call_output": {
                "in_progress", "completed", "incomplete", "failed",
            },
            "reasoning": {"in_progress", "completed", "incomplete"},
            "tool_search_call": {
                "in_progress", "completed", "incomplete",
            },
            "tool_search_output": {
                "in_progress", "completed", "incomplete",
            },
            "image_generation_call": {
                "in_progress", "generating", "completed", "failed",
            },
            "code_interpreter_call": {
                "in_progress", "interpreting", "completed", "incomplete",
                "failed",
            },
            "local_shell_call": {
                "in_progress", "completed", "incomplete",
            },
            "local_shell_call_output": {
                "in_progress", "completed", "incomplete",
            },
            "shell_call": {"in_progress", "completed", "incomplete"},
            "shell_call_output": {
                "in_progress", "completed", "incomplete",
            },
            "apply_patch_call": {"in_progress", "completed"},
            "apply_patch_call_output": {"completed", "failed"},
            "program_output": {"completed", "incomplete"},
            "mcp_call": {
                "in_progress", "calling", "completed", "incomplete",
                "failed",
            },
            "custom_tool_call_output": {
                "in_progress", "completed", "incomplete",
            },
        }

        def terminal_item_status(item_type: Any, current: Any) -> Any:
            allowed = item_status_values.get(item_type)
            if not allowed:
                return current
            if current in allowed and current in {
                "completed", "incomplete", "failed",
            }:
                return current
            if status == "completed":
                candidates = ("completed", "incomplete", "failed")
            elif status in {"failed", "cancelled"}:
                candidates = ("failed", "incomplete", "completed")
            else:
                candidates = ("incomplete", "failed", "completed")
            return next(
                (candidate for candidate in candidates if candidate in allowed),
                current,
            )

        for raw_item in response_output:
            if not isinstance(raw_item, dict):
                continue
            output_index = len(final_output)
            item = dict(raw_item)
            raw_item_type = item.get("type")
            item_type = (
                raw_item_type.value
                if isinstance(raw_item_type, Enum)
                else raw_item_type
            )
            if not item.get("id"):
                prefix = {
                    "message": "msg",
                    "function_call": "fc",
                    "custom_tool_call": "ct",
                    "reasoning": "rs",
                    "web_search_call": "ws",
                    "file_search_call": "fs",
                    "mcp_call": "mcp",
                    "mcp_list_tools": "mcp",
                    "mcp_approval_request": "mcp",
                    "mcp_approval_response": "mcp",
                    "code_interpreter_call": "ci",
                    "image_generation_call": "ig",
                    "computer_call": "cmp",
                    "computer_call_output": "cmpo",
                    "local_shell_call": "ls",
                    "local_shell_call_output": "lso",
                    "shell_call": "sh",
                    "shell_call_output": "sho",
                    "apply_patch_call": "ap",
                    "apply_patch_call_output": "apo",
                    "tool_search_call": "ts",
                    "tool_search_output": "tso",
                    "program": "prog",
                    "program_output": "progo",
                    "compaction": "cmp",
                    "additional_tools": "tools",
                }.get(item_type, "item")
                item["id"] = f"{prefix}_{uuid.uuid4().hex[:24]}"
            if item_type in (
                "function_call", "custom_tool_call", "computer_call",
                "local_shell_call", "shell_call", "apply_patch_call",
            ) and not item.get("call_id"):
                item["call_id"] = item["id"]
            if item_type not in statusless_item_types:
                normalised_status = terminal_item_status(
                    item_type, item.get("status"),
                )
                if item_type == "mcp_call" and item.get("error"):
                    normalised_status = "failed"
                if normalised_status is not None:
                    item["status"] = normalised_status

            # Most output item unions accept in_progress while they stream.
            # Some records have no status, while program/apply-patch outputs
            # only accept terminal statuses, so preserve those records.
            added_item = (
                dict(item)
                if (
                    item_type in statusless_item_types
                    or item_type in terminal_status_item_types
                )
                else {**item, "status": "in_progress"}
            )
            if item_type == "message":
                added_item["content"] = []
            elif item_type == "function_call":
                added_item["arguments"] = ""
            elif item_type == "custom_tool_call":
                # Custom tool input is streamed separately from the output item.
                added_item["input"] = ""
            elif item_type == "reasoning":
                # Summary text is emitted through reasoning summary events below.
                added_item["summary"] = []
                if "content" in item:
                    added_item["content"] = []
            elif item_type == "mcp_call":
                # MCP arguments and result arrive in their own lifecycle events.
                added_item["arguments"] = ""
                if "output" in item:
                    added_item["output"] = None
                if "error" in item:
                    added_item["error"] = None
            elif item_type == "code_interpreter_call":
                # Code and execution output are populated by the dedicated events.
                added_item["code"] = None
                if "outputs" in item:
                    added_item["outputs"] = []
            elif item_type == "image_generation_call":
                added_item["result"] = None
            elif item_type == "shell_call":
                action = item.get("action")
                if isinstance(action, dict):
                    added_item["action"] = {**action, "commands": []}
            elif item_type == "shell_call_output":
                added_item["output"] = []
            yield make_event(
                "response.output_item.added",
                output_index=output_index,
                item=added_item,
            )

            if item_type == "message":
                final_content: list = []
                for raw_part in item.get("content", []):
                    if not isinstance(raw_part, dict):
                        continue
                    content_index = len(final_content)
                    part = dict(raw_part)
                    part_type = part.get("type")
                    if part_type == "output_text":
                        text = part.get("text", "")
                        annotations = part.get("annotations") or []
                        if not isinstance(annotations, (list, tuple)):
                            annotations = [annotations]
                        annotations = list(annotations)
                        logprobs = part.get("logprobs") or []
                        if not isinstance(logprobs, (list, tuple)):
                            logprobs = []
                        logprobs = list(logprobs)
                        # The terminal output item reuses ``part`` below. Keep
                        # it aligned with the streamed events even when a
                        # compatible provider omits these arrays or returns a
                        # single annotation object.
                        part["annotations"] = annotations
                        part["logprobs"] = logprobs
                        yield make_event(
                            "response.content_part.added",
                            item_id=item["id"],
                            output_index=output_index,
                            content_index=content_index,
                            part={
                                **part,
                                "text": "",
                                "annotations": [],
                                "logprobs": logprobs,
                            },
                        )
                        if text:
                            yield make_event(
                                "response.output_text.delta",
                                item_id=item["id"],
                                output_index=output_index,
                                content_index=content_index,
                                delta=text,
                                logprobs=logprobs,
                            )
                        for annotation_index, annotation in enumerate(annotations):
                            yield make_event(
                                "response.output_text.annotation.added",
                                item_id=item["id"],
                                output_index=output_index,
                                content_index=content_index,
                                annotation_index=annotation_index,
                                annotation=annotation,
                            )
                        yield make_event(
                            "response.output_text.done",
                            item_id=item["id"],
                            output_index=output_index,
                            content_index=content_index,
                            text=text,
                            logprobs=logprobs,
                        )
                    elif part_type == "refusal":
                        refusal = part.get("refusal", "")
                        yield make_event(
                            "response.content_part.added",
                            item_id=item["id"],
                            output_index=output_index,
                            content_index=content_index,
                            part={**part, "refusal": ""},
                        )
                        if refusal:
                            yield make_event(
                                "response.refusal.delta",
                                item_id=item["id"],
                                output_index=output_index,
                                content_index=content_index,
                                delta=refusal,
                            )
                        yield make_event(
                            "response.refusal.done",
                            item_id=item["id"],
                            output_index=output_index,
                            content_index=content_index,
                            refusal=refusal,
                        )
                    yield make_event(
                        "response.content_part.done",
                        item_id=item["id"],
                        output_index=output_index,
                        content_index=content_index,
                        part=part,
                    )
                    final_content.append(part)
                item["content"] = final_content
            elif item_type == "function_call":
                arguments = item.get("arguments", "")
                if arguments:
                    yield make_event(
                        "response.function_call_arguments.delta",
                        item_id=item["id"],
                        output_index=output_index,
                        delta=arguments,
                    )
                yield make_event(
                    "response.function_call_arguments.done",
                    item_id=item["id"],
                    output_index=output_index,
                    name=item.get("name", ""),
                    arguments=arguments,
                )
            elif item_type == "reasoning":
                reasoning_content = item.get("content") or []
                if isinstance(reasoning_content, dict):
                    reasoning_content = [reasoning_content]
                elif not isinstance(reasoning_content, (list, tuple)):
                    reasoning_content = []
                final_reasoning_content: list = []
                for content_index, raw_content in enumerate(reasoning_content):
                    if isinstance(raw_content, dict):
                        content = dict(raw_content)
                    else:
                        content = {
                            "type": "reasoning_text",
                            "text": Backend._responses_value_text(raw_content),
                        }
                    content_type = content.get("type") or "reasoning_text"
                    content_text = Backend._responses_value_text(
                        content.get("text", "")
                    )
                    if content_type == "reasoning_text":
                        yield make_event(
                            "response.content_part.added",
                            item_id=item["id"],
                            output_index=output_index,
                            content_index=content_index,
                            part={"type": "reasoning_text", "text": ""},
                        )
                        if content_text:
                            yield make_event(
                                "response.reasoning_text.delta",
                                item_id=item["id"],
                                output_index=output_index,
                                content_index=content_index,
                                delta=content_text,
                            )
                        yield make_event(
                            "response.reasoning_text.done",
                            item_id=item["id"],
                            output_index=output_index,
                            content_index=content_index,
                            text=content_text,
                        )
                        yield make_event(
                            "response.content_part.done",
                            item_id=item["id"],
                            output_index=output_index,
                            content_index=content_index,
                            part={"type": "reasoning_text", "text": content_text},
                        )
                    final_reasoning_content.append(
                        {**content, "type": content_type, "text": content_text}
                    )
                if final_reasoning_content:
                    item["content"] = final_reasoning_content
                summaries = item.get("summary", [])
                if isinstance(summaries, dict):
                    summaries = [summaries]
                elif isinstance(summaries, str):
                    summaries = [{"type": "summary_text", "text": summaries}]
                elif not isinstance(summaries, (list, tuple)):
                    summaries = []

                final_summaries: list = []
                for summary_index, raw_summary in enumerate(summaries):
                    if isinstance(raw_summary, dict):
                        summary = dict(raw_summary)
                    else:
                        summary = {
                            "type": "summary_text",
                            "text": Backend._responses_value_text(raw_summary),
                        }
                    summary["type"] = summary.get("type") or "summary_text"
                    summary_text = Backend._responses_value_text(
                        summary.get("text", "")
                    )
                    terminal_summary = {
                        **summary,
                        "type": "summary_text",
                        "text": summary_text,
                    }
                    final_summaries.append(terminal_summary)
                    yield make_event(
                        "response.reasoning_summary_part.added",
                        item_id=item["id"],
                        output_index=output_index,
                        summary_index=summary_index,
                        part={**summary, "type": "summary_text", "text": ""},
                    )
                    if summary_text:
                        yield make_event(
                            "response.reasoning_summary_text.delta",
                            item_id=item["id"],
                            output_index=output_index,
                            summary_index=summary_index,
                            delta=summary_text,
                        )
                    yield make_event(
                        "response.reasoning_summary_text.done",
                        item_id=item["id"],
                        output_index=output_index,
                        summary_index=summary_index,
                        text=summary_text,
                    )
                    yield make_event(
                        "response.reasoning_summary_part.done",
                        item_id=item["id"],
                        output_index=output_index,
                        summary_index=summary_index,
                        part=terminal_summary,
                    )
                item["summary"] = final_summaries
            elif item_type == "custom_tool_call":
                tool_input = Backend._responses_value_text(item.get("input", ""))
                if tool_input:
                    yield make_event(
                        "response.custom_tool_call_input.delta",
                        item_id=item["id"],
                        output_index=output_index,
                        delta=tool_input,
                    )
                yield make_event(
                    "response.custom_tool_call_input.done",
                    item_id=item["id"],
                    output_index=output_index,
                    input=tool_input,
                )
            elif item_type == "web_search_call":
                # Search calls expose progress even when the selected response is
                # already complete. There is no separate failed event in the
                # Responses stream union; output_item.done carries that status.
                yield make_event(
                    "response.web_search_call.in_progress",
                    item_id=item["id"],
                    output_index=output_index,
                )
                search_status = item.get("status")
                if search_status != "in_progress":
                    yield make_event(
                        "response.web_search_call.searching",
                        item_id=item["id"],
                        output_index=output_index,
                    )
                if search_status in (None, "completed"):
                    yield make_event(
                        "response.web_search_call.completed",
                        item_id=item["id"],
                        output_index=output_index,
                    )
            elif item_type == "file_search_call":
                yield make_event(
                    "response.file_search_call.in_progress",
                    item_id=item["id"],
                    output_index=output_index,
                )
                search_status = item.get("status")
                if search_status != "in_progress":
                    yield make_event(
                        "response.file_search_call.searching",
                        item_id=item["id"],
                        output_index=output_index,
                    )
                if search_status in (None, "completed"):
                    yield make_event(
                        "response.file_search_call.completed",
                        item_id=item["id"],
                        output_index=output_index,
                    )
            elif item_type == "mcp_call":
                yield make_event(
                    "response.mcp_call.in_progress",
                    item_id=item["id"],
                    output_index=output_index,
                )
                arguments = Backend._responses_value_text(
                    item.get("arguments", "")
                )
                if arguments:
                    yield make_event(
                        "response.mcp_call_arguments.delta",
                        item_id=item["id"],
                        output_index=output_index,
                        delta=arguments,
                    )
                yield make_event(
                    "response.mcp_call_arguments.done",
                    item_id=item["id"],
                    output_index=output_index,
                    arguments=arguments,
                )
                mcp_failed = item.get("status") in ("failed", "incomplete")
                yield make_event(
                    "response.mcp_call.failed" if mcp_failed else "response.mcp_call.completed",
                    item_id=item["id"],
                    output_index=output_index,
                )
            elif item_type == "mcp_list_tools":
                # Tool discovery has lifecycle events but no argument/output
                # deltas. The output item itself carries the discovered tool
                # schemas, so preserve it in the terminal item event.
                yield make_event(
                    "response.mcp_list_tools.in_progress",
                    item_id=item["id"],
                    output_index=output_index,
                )
                yield make_event(
                    "response.mcp_list_tools.failed"
                    if item.get("error")
                    else "response.mcp_list_tools.completed",
                    item_id=item["id"],
                    output_index=output_index,
                )
            elif item_type == "code_interpreter_call":
                yield make_event(
                    "response.code_interpreter_call.in_progress",
                    item_id=item["id"],
                    output_index=output_index,
                )
                code_status = item.get("status")
                if code_status != "in_progress":
                    yield make_event(
                        "response.code_interpreter_call.interpreting",
                        item_id=item["id"],
                        output_index=output_index,
                    )
                code = Backend._responses_value_text(item.get("code", ""))
                if code:
                    yield make_event(
                        "response.code_interpreter_call_code.delta",
                        item_id=item["id"],
                        output_index=output_index,
                        delta=code,
                    )
                    yield make_event(
                        "response.code_interpreter_call_code.done",
                        item_id=item["id"],
                        output_index=output_index,
                        code=code,
                    )
                if code_status in (None, "completed"):
                    yield make_event(
                        "response.code_interpreter_call.completed",
                        item_id=item["id"],
                        output_index=output_index,
                    )
            elif item_type == "image_generation_call":
                yield make_event(
                    "response.image_generation_call.in_progress",
                    item_id=item["id"],
                    output_index=output_index,
                )
                image_status = item.get("status")
                if image_status != "in_progress":
                    yield make_event(
                        "response.image_generation_call.generating",
                        item_id=item["id"],
                        output_index=output_index,
                    )
                partial_image = item.get(
                    "partial_image_b64",
                    item.get("result", item.get("b64_json", "")),
                )
                partial_image = Backend._responses_value_text(partial_image)
                if partial_image:
                    partial_image_index = item.get("partial_image_index", 0)
                    if (
                        isinstance(partial_image_index, bool)
                        or not isinstance(partial_image_index, int)
                        or partial_image_index < 0
                    ):
                        partial_image_index = 0
                    yield make_event(
                        "response.image_generation_call.partial_image",
                        item_id=item["id"],
                        output_index=output_index,
                        partial_image_index=partial_image_index,
                        partial_image_b64=partial_image,
                    )
                if image_status in (None, "completed"):
                    yield make_event(
                        "response.image_generation_call.completed",
                        item_id=item["id"],
                        output_index=output_index,
                    )
            elif item_type == "shell_call":
                action = item.get("action")
                raw_commands = (
                    action.get("commands", [])
                    if isinstance(action, dict)
                    else []
                )
                if isinstance(raw_commands, str):
                    raw_commands = [raw_commands]
                elif not isinstance(raw_commands, (list, tuple)):
                    raw_commands = []
                for command_index, raw_command in enumerate(raw_commands):
                    command = Backend._responses_value_text(raw_command)
                    yield make_event(
                        "response.shell_call_command.added",
                        output_index=output_index,
                        command_index=command_index,
                        command="",
                    )
                    if command:
                        yield make_event(
                            "response.shell_call_command.delta",
                            output_index=output_index,
                            command_index=command_index,
                            delta=command,
                        )
                    yield make_event(
                        "response.shell_call_command.done",
                        output_index=output_index,
                        command_index=command_index,
                        command=command,
                    )
            elif item_type == "shell_call_output":
                shell_output = item.get("output") or []
                if isinstance(shell_output, dict):
                    shell_output = [shell_output]
                elif not isinstance(shell_output, (list, tuple)):
                    shell_output = []
                for command_index, raw_entry in enumerate(shell_output):
                    if not isinstance(raw_entry, dict):
                        continue
                    entry = dict(raw_entry)
                    delta = {
                        key: Backend._responses_value_text(entry[key])
                        for key in ("stdout", "stderr")
                        if entry.get(key)
                    }
                    if delta:
                        yield make_event(
                            "response.shell_call_output_content.delta",
                            item_id=item["id"],
                            output_index=output_index,
                            command_index=command_index,
                            delta=delta,
                        )
                    yield make_event(
                        "response.shell_call_output_content.done",
                        item_id=item["id"],
                        output_index=output_index,
                        command_index=command_index,
                        output=[entry],
                    )

            yield make_event(
                "response.output_item.done",
                output_index=output_index,
                item=item,
            )
            final_output.append(item)

        # Audio deltas are response-level stream events rather than output
        # items. A non-streaming Responses result may still contain an audio
        # payload from an OpenAI-compatible provider; replay it when present.
        def stream_values(
            value: Any, keys: tuple[str, ...],
        ) -> tuple[list[str], bool]:
            if value is None:
                return [], False
            if isinstance(value, str):
                return [value], True
            if isinstance(value, (list, tuple)):
                values: list[str] = []
                for entry in value:
                    nested, _ = stream_values(entry, keys)
                    values.extend(nested)
                return values, True
            if isinstance(value, dict):
                for key in keys:
                    if key in value:
                        nested, _ = stream_values(value[key], keys)
                        return nested, True
                return [], True
            return [str(value)], True

        audio_payload = response.get("audio")
        audio_values, audio_present = stream_values(
            audio_payload,
            (
                "delta", "audio", "data", "b64_json", "audio_b64",
                "audio_base64",
            ),
        )
        if audio_present:
            for delta in audio_values:
                if delta:
                    yield make_event("response.audio.delta", delta=delta)
            yield make_event("response.audio.done")

        transcript_value = None
        transcript_present = False
        if isinstance(audio_payload, dict) and "transcript" in audio_payload:
            transcript_value = audio_payload["transcript"]
            transcript_present = True
        elif "audio_transcript" in response:
            transcript_value = response.get("audio_transcript")
            transcript_present = True
        if transcript_present:
            transcript_values, _ = stream_values(
                transcript_value, ("delta", "text", "transcript"),
            )
            for delta in transcript_values:
                if delta:
                    yield make_event(
                        "response.audio.transcript.delta", delta=delta,
                    )
            yield make_event("response.audio.transcript.done")

        final_response = self._normalise_responses_envelope(
            response_base,
            status=status,
            output=final_output,
        )
        terminal_type = {
            "completed": "response.completed",
            "failed": "response.failed",
            "incomplete": "response.incomplete",
            "cancelled": "response.failed",
            "in_progress": "response.in_progress",
        }[status]
        yield make_event(terminal_type, response=final_response)

    def get_models_response(self) -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": model["name"],
                    "object": "model",
                    "created": 0,
                    "owned_by": "turbo-proxy",
                }
                for model in self.config.models
            ],
        }
