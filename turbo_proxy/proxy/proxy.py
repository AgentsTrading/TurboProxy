import json
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.responses import StreamingResponse

from .backend import Backend, _close_upstream_stream
from ..utils import (
    Config,
    SSEFormatter,
    create_logger,
    log_response_summary,
    summarize_request_body,
)
from ..visualizer import register_visualizer_routes

logger = create_logger("proxy")

UPSTREAM = "https://api.anthropic.com"


class _ResponsesStreamProtocolError(RuntimeError):
    status_code = 502


class _ResponsesStreamingResponse(StreamingResponse):
    """Streaming response that also owns a prefetched backend stream."""

    def __init__(self, *args, close_stream, **kwargs):
        super().__init__(*args, **kwargs)
        self._close_stream = close_stream

    async def __call__(self, scope, receive, send) -> None:
        primary = None
        try:
            await super().__call__(scope, receive, send)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            await self._close_stream(primary)


class ProxyServer:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._backend = Backend(self.config)
        self.app = FastAPI()
        register_visualizer_routes(self.app, self.config.log_dir)
        self._register_routes()

    @property
    def backend(self) -> Backend:
        return self._backend

    def _register_routes(self) -> None:
        app = self.app

        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            if not request.url.path.startswith("/visualizer"):
                logger.info(f"REQ incoming {request.method} {request.url.path}")
            response = await call_next(request)
            return response

        # Catch-all proxy route (must be after visualizer routes)
        @app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
        )
        async def proxy_all(request: Request, path: str):
            return await self._proxy(request)

    async def _proxy(self, request: Request) -> Response:
        path = request.url.path
        method = request.method
        body = await request.body()

        clean_path = path.split("?")[0].rstrip("/")

        # --- OpenAI: GET /v1/models ---
        if method == "GET" and (
            clean_path.endswith("/models") or clean_path == "models"
        ):
            logger.info(f"REQ {method} {path} (openai models)")
            return JSONResponse(content=self._backend.get_models_response())

        # --- OpenAI: POST /v1/chat/completions ---
        if method == "POST" and clean_path.endswith("/chat/completions"):
            return await self._handle_openai(request, path, body)

        # --- OpenAI: POST /v1/responses ---
        if method == "POST" and clean_path.endswith("/v1/responses"):
            return await self._handle_responses(request, path, body)

        # --- Anthropic: POST /v1/messages ---
        if method == "POST" and clean_path.endswith("/messages"):
            return await self._handle_anthropic(request, path, body)

        # --- Upstream passthrough ---
        return await self._handle_upstream(request, path, body)

    # ------------------------------------------------------------------
    # Anthropic path
    # ------------------------------------------------------------------

    async def _handle_anthropic(
        self, request: Request, path: str, body: bytes,
    ) -> Response:
        logger.info(f"REQ {request.method} {path} (anthropic)")
        for h in ("x-api-key", "anthropic-version", "content-type", "anthropic-beta"):
            val = request.headers.get(h)
            if val:
                display = val[:12] + "..." if h == "x-api-key" and len(val) > 12 else val
                logger.debug(f"HDR {h}: {display}")
        if body:
            logger.info(f"BODY {summarize_request_body(body)}")

        is_streaming = self._body_is_streaming(body)
        start = time.monotonic()

        if is_streaming:
            return await self._anthropic_streaming(body, start)
        else:
            return await self._anthropic_non_streaming(body, start)

    async def _anthropic_non_streaming(
        self, body: bytes, start: float,
    ) -> Response:
        result, error = await self._backend.complete_anthropic(body)
        elapsed = time.monotonic() - start

        if error:
            logger.error(f"BACKEND ERROR {error}")
            return JSONResponse(
                status_code=500,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": error},
                },
            )

        resp_body = json.dumps(result, default=str)
        log_response_summary(resp_body, 200)
        logger.info(f"TIME {elapsed:.2f}s")
        return Response(
            content=resp_body,
            media_type="application/json",
        )

    async def _anthropic_streaming(
        self, body: bytes, start: float,
    ) -> Response:
        async def generate():
            try:
                async for event in self._backend.stream_anthropic(body):
                    yield event
            except Exception as e:
                logger.error(f"BACKEND STREAM ERROR {e}")
                yield SSEFormatter.error(str(e))
            finally:
                elapsed = time.monotonic() - start
                logger.info(f"TIME {elapsed:.2f}s")

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ------------------------------------------------------------------
    # OpenAI path
    # ------------------------------------------------------------------

    async def _handle_openai(
        self, request: Request, path: str, body: bytes,
    ) -> Response:
        logger.info(f"REQ {request.method} {path} (openai)")
        for h in ("authorization", "content-type"):
            val = request.headers.get(h)
            if val:
                display = val[:20] + "..." if h == "authorization" and len(val) > 20 else val
                logger.debug(f"HDR {h}: {display}")
        if body:
            logger.info(f"BODY {summarize_request_body(body)}")

        is_streaming = self._body_is_streaming(body)
        start = time.monotonic()

        if is_streaming:
            return await self._openai_streaming(body, start)
        else:
            return await self._openai_non_streaming(body, start)

    async def _openai_non_streaming(
        self, body: bytes, start: float,
    ) -> Response:
        result, error = await self._backend.complete_openai(body)
        elapsed = time.monotonic() - start

        if error:
            logger.error(f"BACKEND ERROR {error}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": error,
                        "type": "invalid_request_error",
                        "code": None,
                    },
                },
            )

        resp_body = json.dumps(result, default=str)
        logger.info(f"RESP status=200 | model={result.get('model', '?')}")
        logger.info(f"TIME {elapsed:.2f}s")
        return Response(
            content=resp_body,
            media_type="application/json",
        )

    async def _openai_streaming(
        self, body: bytes, start: float,
    ) -> Response:
        async def generate():
            try:
                async for event in self._backend.stream_openai(body):
                    yield event
            except Exception as e:
                logger.error(f"BACKEND STREAM ERROR {e}")
                yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                elapsed = time.monotonic() - start
                logger.info(f"TIME {elapsed:.2f}s")

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ------------------------------------------------------------------
    # OpenAI Responses path
    # ------------------------------------------------------------------

    async def _handle_responses(
        self, request: Request, path: str, body: bytes,
    ) -> Response:
        logger.info(f"REQ {request.method} {path} (openai responses)")
        for h in ("authorization", "content-type"):
            val = request.headers.get(h)
            if val:
                display = "***" if h == "authorization" else val
                logger.debug(f"HDR {h}: {display}")
        if body:
            logger.info(f"BODY {summarize_request_body(body)}")

        start = time.monotonic()
        try:
            responses_body = self._backend.parse_responses_body(body)
            self._backend.validate_responses_body(responses_body)
        except ValueError as exc:
            message = str(exc)
            logger.error(f"INVALID RESPONSES REQUEST {exc}")
            return self._responses_invalid_request(message)

        if responses_body.get("stream") is True:
            return await self._responses_streaming(body, start)
        return await self._responses_non_streaming(body, start)

    @staticmethod
    def _responses_invalid_request(message: str) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                },
            },
        )

    @staticmethod
    def _responses_exception_details(exc: Exception) -> tuple[int, dict]:
        """Recover an OpenAI-compatible error from a wrapped upstream failure."""
        status_code = None
        upstream_error = None
        current: BaseException | None = exc
        seen = set()

        while current is not None and id(current) not in seen:
            seen.add(id(current))
            candidate_status = getattr(current, "status_code", None)
            response = getattr(current, "response", None)
            if candidate_status is None and response is not None:
                candidate_status = getattr(response, "status_code", None)
            if (
                status_code is None
                and isinstance(candidate_status, int)
                and not isinstance(candidate_status, bool)
                and 400 <= candidate_status <= 599
            ):
                status_code = candidate_status

            candidate_body = getattr(current, "body", None)
            if not isinstance(candidate_body, dict) and response is not None:
                try:
                    candidate_body = response.json()
                except (TypeError, ValueError):
                    candidate_body = None
            if upstream_error is None and isinstance(candidate_body, dict):
                candidate_error = candidate_body.get("error", candidate_body)
                if isinstance(candidate_error, dict) and candidate_error.get(
                    "message"
                ):
                    upstream_error = candidate_error

            if upstream_error is None:
                error_text = getattr(current, "message", None)
                if isinstance(error_text, str):
                    decoder = json.JSONDecoder()
                    for offset, character in enumerate(error_text):
                        if character != "{":
                            continue
                        try:
                            candidate_body, _ = decoder.raw_decode(
                                error_text[offset:]
                            )
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(candidate_body, dict):
                            continue
                        candidate_error = candidate_body.get(
                            "error", candidate_body,
                        )
                        if isinstance(candidate_error, dict) and (
                            candidate_error.get("message")
                        ):
                            upstream_error = candidate_error
                            break

            current = current.__cause__ or current.__context__

        status_code = status_code or 500
        default_type = {
            400: "invalid_request_error",
            401: "authentication_error",
            403: "permission_error",
            404: "invalid_request_error",
            409: "conflict_error",
            422: "invalid_request_error",
            429: "rate_limit_error",
        }.get(status_code, "server_error" if status_code >= 500 else "api_error")
        upstream_error = upstream_error or {}
        return status_code, {
            "message": str(upstream_error.get("message") or exc),
            "type": str(upstream_error.get("type") or default_type),
            "param": upstream_error.get("param"),
            "code": upstream_error.get("code"),
        }

    @classmethod
    def _responses_exception_response(cls, exc: Exception) -> JSONResponse:
        status_code, error = cls._responses_exception_details(exc)
        return JSONResponse(status_code=status_code, content={"error": error})

    @classmethod
    def _responses_stream_error(cls, exc: Exception, sequence_number: int) -> dict:
        _, details = cls._responses_exception_details(exc)
        return {
            "type": "error",
            "code": details.get("code") or details.get("type") or "server_error",
            "message": details["message"],
            "param": details.get("param"),
            "sequence_number": sequence_number,
        }

    async def _responses_non_streaming(
        self, body: bytes, start: float,
    ) -> Response:
        try:
            result, error = await self._backend.complete_responses(body)
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(f"BACKEND RESPONSES ERROR {exc}")
            logger.info(f"TIME {elapsed:.2f}s")
            return self._responses_exception_response(exc)
        elapsed = time.monotonic() - start

        if error:
            logger.error(f"BACKEND ERROR {error}")
            return self._responses_invalid_request(error)

        resp_body = json.dumps(result, default=str)
        logger.info(f"RESP status=200 | model={result.get('model', '?')}")
        logger.info(f"TIME {elapsed:.2f}s")
        return Response(content=resp_body, media_type="application/json")

    async def _responses_streaming(
        self, body: bytes, start: float,
    ) -> Response:
        backend_stream = aiter(self._backend.stream_responses(body))
        try:
            first_event = await anext(backend_stream)
        except StopAsyncIteration:
            exc = _ResponsesStreamProtocolError(
                "Upstream Responses stream ended before emitting an event"
            )
            try:
                await _close_upstream_stream(backend_stream, exc)
            except Exception:
                pass
            elapsed = time.monotonic() - start
            logger.error(f"BACKEND RESPONSES STREAM ERROR {exc}")
            logger.info(f"TIME {elapsed:.2f}s")
            return self._responses_exception_response(exc)
        except BaseException as exc:
            try:
                await _close_upstream_stream(backend_stream, exc)
            except BaseException:
                # _close_upstream_stream preserves the primary exception when
                # cleanup also fails; the original upstream error is returned.
                pass
            if not isinstance(exc, Exception):
                raise
            elapsed = time.monotonic() - start
            logger.error(f"BACKEND RESPONSES STREAM ERROR {exc}")
            logger.info(f"TIME {elapsed:.2f}s")
            return self._responses_exception_response(exc)

        stream_close_started = False

        async def close_backend_stream(
            primary: BaseException | None = None,
        ) -> None:
            nonlocal stream_close_started
            if stream_close_started:
                return
            stream_close_started = True
            try:
                await _close_upstream_stream(backend_stream, primary)
            except BaseException as cleanup_exc:
                if primary is not None or not isinstance(cleanup_exc, Exception):
                    raise
                logger.error(
                    "BACKEND RESPONSES STREAM CLEANUP ERROR "
                    f"{cleanup_exc}"
                )

        async def generate():
            next_sequence_number = 0
            terminal_sent = False
            primary = None
            terminal_types = {
                "response.completed",
                "response.incomplete",
                "response.failed",
                "error",
            }
            enum_event_types = {
                "RESPONSE_COMPLETED": "response.completed",
                "RESPONSE_INCOMPLETE": "response.incomplete",
                "RESPONSE_FAILED": "response.failed",
                "ERROR": "error",
            }

            async def events():
                yield first_event
                async for event in backend_stream:
                    yield event

            try:
                async for event in events():
                    event_type = ""
                    has_payload = False
                    done_sentinel = False
                    data_lines = []
                    for line in event.splitlines():
                        if line.startswith("event:"):
                            event_type = line.removeprefix("event:").strip()
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if data == "[DONE]":
                            done_sentinel = True
                            continue
                        data_lines.append(data)
                    if data_lines:
                        data = "\n".join(data_lines)
                        try:
                            payload = json.loads(data)
                        except (TypeError, json.JSONDecodeError) as exc:
                            raise _ResponsesStreamProtocolError(
                                "Upstream Responses stream emitted invalid JSON "
                                "event data"
                            ) from exc
                        if not isinstance(payload, dict):
                            raise _ResponsesStreamProtocolError(
                                "Upstream Responses stream emitted a non-object "
                                "event payload"
                            )
                        payload_type = payload.get("type")
                        if isinstance(payload_type, str):
                            payload_type = enum_event_types.get(
                                payload_type.rsplit(".", 1)[-1], payload_type,
                            )
                        if not isinstance(payload_type, str) or not (
                            payload_type.startswith("response.")
                            or payload_type == "error"
                        ):
                            raise _ResponsesStreamProtocolError(
                                "Upstream Responses stream emitted an invalid "
                                "event type"
                            )
                        has_payload = True
                        # The JSON discriminator is authoritative when compatible
                        # upstreams omit or mislabel the optional SSE event name.
                        event_type = payload_type
                        payload["type"] = payload_type
                    if done_sentinel and not has_payload:
                        continue
                    if not has_payload and (
                        event_type.startswith("response.")
                        or event_type == "error"
                    ):
                        raise _ResponsesStreamProtocolError(
                            "Upstream Responses stream emitted a protocol event "
                            "without an object payload"
                        )
                    is_terminal = has_payload and event_type in terminal_types
                    if is_terminal:
                        terminal_sent = True
                    if has_payload:
                        payload["sequence_number"] = next_sequence_number
                        next_sequence_number += 1
                        resolved_event_type = event_type or str(
                            payload.get("type") or "response.event"
                        )
                        event = (
                            f"event: {resolved_event_type}\n"
                            "data: "
                            f"{json.dumps(payload, default=str, separators=(',', ':'))}"
                            "\n\n"
                        )
                    elif done_sentinel:
                        # ``[DONE]`` is a Chat Completions compatibility
                        # sentinel, not a Responses event. A few gateways append
                        # it to the same SSE block as the terminal payload.
                        event = self._strip_responses_done_sentinel(event)
                    yield event
                    if is_terminal:
                        break
                if not terminal_sent:
                    exc = _ResponsesStreamProtocolError(
                        "Upstream Responses stream ended without a terminal event"
                    )
                    error = self._responses_stream_error(
                        exc, next_sequence_number,
                    )
                    yield (
                        "event: error\n"
                        f"data: {json.dumps(error, default=str)}\n\n"
                    )
            except Exception as exc:
                logger.error(f"BACKEND RESPONSES STREAM ERROR {exc}")
                if not terminal_sent:
                    error = self._responses_stream_error(
                        exc, next_sequence_number,
                    )
                    try:
                        yield (
                            "event: error\n"
                            f"data: {json.dumps(error, default=str)}\n\n"
                        )
                    except BaseException as stream_exc:
                        primary = stream_exc
                        raise
            except BaseException as exc:
                primary = exc
                raise
            finally:
                await close_backend_stream(primary)
                elapsed = time.monotonic() - start
                logger.info(f"TIME {elapsed:.2f}s")

        return _ResponsesStreamingResponse(
            generate(),
            close_stream=close_backend_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @staticmethod
    def _strip_responses_done_sentinel(event: str) -> str:
        """Remove an exact ``data: [DONE]`` line from a Responses SSE block."""
        return "".join(
            line
            for line in event.splitlines(keepends=True)
            if line.rstrip("\r\n").strip() not in {"data: [DONE]", "data:[DONE]"}
        )

    # ------------------------------------------------------------------
    # Upstream passthrough
    # ------------------------------------------------------------------

    async def _handle_upstream(
        self, request: Request, path: str, body: bytes,
    ) -> Response:
        upstream_url = f"{UPSTREAM}{path}"
        headers = {}
        for k, v in request.headers.items():
            if k.lower() not in (
                "host", "content-length", "transfer-encoding",
            ):
                headers[k] = v

        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method=request.method,
                url=upstream_url,
                headers=headers,
                content=body if body else None,
            )

        resp_headers = {}
        for k, v in resp.headers.items():
            if k.lower() not in (
                "transfer-encoding",
                "content-encoding",
                "content-length",
                "connection",
            ):
                resp_headers[k] = v

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp_headers,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _body_is_streaming(body: bytes) -> bool:
        if not body:
            return False
        try:
            return json.loads(body).get("stream") is True
        except Exception:
            return False
