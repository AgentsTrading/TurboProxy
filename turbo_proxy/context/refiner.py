from typing import Any, Dict, List

from ..utils import ContextConfig, llm_completion, create_logger

_logger = create_logger("context")


class ContextRefiner:
    def __init__(self, config: ContextConfig):
        self.config = config

    @staticmethod
    def _format_messages(messages: List[Dict[str, Any]]) -> str:
        def render(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float, bool)):
                return str(value)
            if isinstance(value, (list, tuple)):
                return "\n".join(
                    rendered for item in value
                    if (rendered := render(item))
                )
            if isinstance(value, dict):
                block_type = value.get("type")
                if block_type in {
                    "text", "input_text", "output_text", "summary_text",
                    "reasoning_text",
                }:
                    return render(value.get("text", ""))
                if block_type == "refusal":
                    refusal = render(value.get("refusal", ""))
                    return f"[refusal: {refusal}]" if refusal else "[refusal]"
                if block_type in {"image_url", "input_image", "output_image"}:
                    return "[image]"
                if block_type in {"input_file", "file"}:
                    return "[file]"
                if block_type in {"input_audio", "audio", "output_audio"}:
                    return "[audio]"
                if block_type == "reasoning":
                    summary = render(value.get("summary", []))
                    return (
                        f"[reasoning: {summary}]"
                        if summary else "[reasoning]"
                    )
                # Tool results and future Responses blocks often carry their
                # useful text under one of these nested fields.
                for key in (
                    "content", "output", "input", "arguments", "summary",
                    "action", "query", "results", "code", "transcript",
                ):
                    if key in value and value[key] not in (None, "", [], {}):
                        nested = render(value[key])
                        if nested:
                            return (
                                f"[{block_type or 'item'}: {nested}]"
                            )
                return str(value)
            return str(value)

        parts: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                rendered = render(msg)
                if rendered:
                    parts.append(f"UNKNOWN: {rendered}")
                continue
            role = (msg.get("role") or "unknown").upper()
            content = render(msg.get("content", ""))
            if msg.get("tool_call_id"):
                parts.append(
                    f"{role}: [tool_result {msg['tool_call_id']}: {content}]"
                )
            elif content:
                parts.append(f"{role}: {content}")
            tool_calls = msg.get("tool_calls") or []
            if not isinstance(tool_calls, (list, tuple)):
                tool_calls = [tool_calls]
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    rendered = render(tool_call)
                    if rendered:
                        parts.append(f"{role}: [tool_call: {rendered}]")
                    continue
                function = tool_call.get("function") or {}
                if not isinstance(function, dict):
                    function = {"arguments": function}
                parts.append(
                    f"{role}: [tool_call: {function.get('name', '')}"
                    f"({render(function.get('arguments', ''))})]"
                )
        return "\n\n".join(parts)

    @staticmethod
    def _coerce_refined_text(value: Any) -> str:
        """Normalize provider-specific message content to refinement text."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            parts = [
                ContextRefiner._coerce_refined_text(item)
                for item in value
            ]
            return "\n".join(part for part in parts if part)
        if isinstance(value, dict):
            recognized = False
            for key in ("text", "content", "output_text", "value"):
                if key not in value:
                    continue
                recognized = True
                rendered = ContextRefiner._coerce_refined_text(value[key])
                if rendered:
                    return rendered
            if recognized:
                return ""
        return str(value)

    async def refine(
        self, messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        context_str = self._format_messages(messages)
        prompt = self.config.refinement_prompt.replace("{context}", context_str)

        _logger.info(f"Refining context with {self.config.model_name}")

        try:
            response = await llm_completion(
                model=self.config.model_name,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                provider=self.config.provider,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_refined = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            refined = self._coerce_refined_text(raw_refined)
        except Exception as e:
            _logger.error(f"Context refinement error: {e} — skipping refinement")
            return messages

        if not refined.strip():
            _logger.warn("Context refinement returned empty content — skipping")
            return messages

        _logger.info(f"Refined context ({len(refined)} chars)")

        new_messages = list(messages)
        if new_messages and new_messages[0].get("role") == "system":
            original_system = self._coerce_refined_text(
                new_messages[0].get("content")
            )
            new_messages[0] = {
                **new_messages[0],
                "content": (
                    f"{refined}\n\n{original_system}"
                    if original_system else refined
                ),
            }
        else:
            new_messages.insert(0, {"role": "system", "content": refined})

        return new_messages
