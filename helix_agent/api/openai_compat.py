"""helix_agent.api.openai_compat — OpenAI-compatible client.

Works with any provider that speaks the OpenAI Chat Completions API:
- OpenAI       (base_url = https://api.openai.com/v1)
- Gemini       (base_url = https://generativelanguage.googleapis.com/v1beta/openai)
- Ollama       (base_url = http://localhost:11434/v1, no key required)

Internal :class:`MessageRequest` / :class:`MessageResponse` types are
translated to/from the OpenAI wire format transparently.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx

from .base import BaseClient
from .error import ApiResponseError, AuthError, HttpError, JsonError, is_retryable_status
from .types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InputJsonDelta,
    InputMessage,
    MessageDelta,
    MessageDeltaEvent,
    MessageRequest,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    OutputContentBlock,
    StreamEvent,
    TextDelta,
    TextInputBlock,
    TextOutputBlock,
    ToolResultInputBlock,
    ToolUseInputBlock,
    ToolUseOutputBlock,
    Usage,
)


# ---------------------------------------------------------------------------
# Request translation — internal → OpenAI wire format
# ---------------------------------------------------------------------------


def _role_content(msg: InputMessage) -> dict[str, Any]:
    """Convert an InputMessage to an OpenAI messages entry."""
    # Simple case: single text block
    if len(msg.content) == 1 and isinstance(msg.content[0], TextInputBlock):
        return {"role": msg.role, "content": msg.content[0].text}

    # Tool result → OpenAI tool message
    if len(msg.content) == 1 and isinstance(msg.content[0], ToolResultInputBlock):
        block = msg.content[0]
        # Flatten content to a single string for OpenAI
        parts = []
        for c in block.content:
            if hasattr(c, "text"):
                parts.append(c.text)  # type: ignore[union-attr]
            elif hasattr(c, "value"):
                parts.append(json.dumps(c.value))  # type: ignore[union-attr]
        return {
            "role": "tool",
            "tool_call_id": block.tool_use_id,
            "content": "\n".join(parts),
        }

    # Mixed / tool-use blocks
    parts_list: list[dict] = []
    tool_calls: list[dict] = []
    for b in msg.content:
        if isinstance(b, TextInputBlock):
            parts_list.append({"type": "text", "text": b.text})
        elif isinstance(b, ToolUseInputBlock):
            tool_calls.append({
                "id": b.id,
                "type": "function",
                "function": {"name": b.name, "arguments": json.dumps(b.input)},
            })

    result: dict[str, Any] = {"role": msg.role}
    if parts_list:
        result["content"] = parts_list
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _to_openai_payload(request: MessageRequest, stream: bool = False) -> dict[str, Any]:
    """Translate a :class:`MessageRequest` to an OpenAI Chat Completions payload."""
    messages: list[dict] = []

    # System prompt → first "system" message in OpenAI format
    if request.system:
        messages.append({"role": "system", "content": request.system})

    for msg in request.messages:
        messages.append(_role_content(msg))

    payload: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": messages,
    }

    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema,
                },
            }
            for t in request.tools
        ]

    if request.tool_choice is not None:
        from .types import ToolChoiceAny, ToolChoiceAuto, ToolChoiceTool
        if isinstance(request.tool_choice, ToolChoiceAuto):
            payload["tool_choice"] = "auto"
        elif isinstance(request.tool_choice, ToolChoiceAny):
            payload["tool_choice"] = "required"
        elif isinstance(request.tool_choice, ToolChoiceTool):
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": request.tool_choice.name},
            }

    return payload


# ---------------------------------------------------------------------------
# Response translation — OpenAI wire format → internal types
# ---------------------------------------------------------------------------


def _parse_openai_response(data: dict, request_id: Optional[str] = None) -> MessageResponse:
    """Convert an OpenAI Chat Completions response to a :class:`MessageResponse`."""
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason")

    content: list[OutputContentBlock] = []

    # Text content
    text = message.get("content") or ""
    if text:
        content.append(TextOutputBlock(text=text))

    # Tool calls
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            input_data = json.loads(fn.get("arguments", "{}"))
        except Exception:
            input_data = {}
        content.append(ToolUseOutputBlock(
            id=tc.get("id", ""),
            name=fn.get("name", ""),
            input=input_data,
        ))

    usage_data = data.get("usage", {})
    usage = Usage(
        input_tokens=usage_data.get("prompt_tokens", 0),
        output_tokens=usage_data.get("completion_tokens", 0),
    )

    # Map OpenAI finish_reason → Anthropic-style stop_reason
    stop_reason_map = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "stop_sequence",
    }
    stop_reason = stop_reason_map.get(finish_reason or "", finish_reason)

    return MessageResponse(
        id=data.get("id", ""),
        kind="message",
        role=message.get("role", "assistant"),
        content=content,
        model=data.get("model", ""),
        usage=usage,
        stop_reason=stop_reason,
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# SSE chunk translation — OpenAI streaming format → StreamEvent
# ---------------------------------------------------------------------------


def _parse_openai_chunk(data: dict, index_map: dict[int, str]) -> list[StreamEvent]:
    """Convert one OpenAI SSE chunk into zero or more :class:`StreamEvent` objects.

    *index_map* tracks which content-block index maps to which type ("text" | "tool").
    """
    events: list[StreamEvent] = []
    choices = data.get("choices", [])

    for choice in choices:
        idx: int = choice.get("index", 0)
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Determine / track block type
        if delta.get("content") is not None and idx not in index_map:
            index_map[idx] = "text"
            events.append(ContentBlockStartEvent(
                index=idx,
                content_block=TextOutputBlock(text=""),
            ))

        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                tc_idx: int = tc.get("index", idx)
                if tc_idx not in index_map:
                    index_map[tc_idx] = "tool"
                    fn = tc.get("function", {})
                    events.append(ContentBlockStartEvent(
                        index=tc_idx,
                        content_block=ToolUseOutputBlock(
                            id=tc.get("id", ""),
                            name=fn.get("name", ""),
                            input={},
                        ),
                    ))
                args_fragment = tc.get("function", {}).get("arguments", "")
                if args_fragment:
                    events.append(ContentBlockDeltaEvent(
                        index=tc_idx,
                        delta=InputJsonDelta(partial_json=args_fragment),
                    ))

        text_fragment = delta.get("content")
        if text_fragment:
            events.append(ContentBlockDeltaEvent(
                index=idx,
                delta=TextDelta(text=text_fragment),
            ))

        if finish_reason is not None:
            if idx in index_map:
                events.append(ContentBlockStopEvent(index=idx))

    # Usage chunk (stream_options.include_usage)
    if "usage" in data and data["usage"]:
        usage_data = data["usage"]
        events.append(MessageDeltaEvent(
            delta=MessageDelta(stop_reason=None, stop_sequence=None),
            usage=Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
            ),
        ))

    return events


# ---------------------------------------------------------------------------
# OpenAICompatClient
# ---------------------------------------------------------------------------


class OpenAICompatClient(BaseClient):
    """Async client for any OpenAI-compatible Chat Completions endpoint.

    Usage::

        # OpenAI
        client = OpenAICompatClient(api_key="sk-...", base_url="https://api.openai.com/v1")

        # Gemini
        client = OpenAICompatClient(
            api_key="AIza...",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )

        # Ollama (no key required)
        client = OpenAICompatClient(base_url="http://localhost:11434/v1")
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout)

    def _headers(self, stream: bool = False) -> dict[str, str]:
        h: dict[str, str] = {"content-type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        if stream:
            h["accept"] = "text/event-stream"
        return h

    def _extract_request_id(self, headers: httpx.Headers) -> Optional[str]:
        return headers.get("x-request-id") or headers.get("request-id")

    async def send_message(self, request: MessageRequest) -> MessageResponse:
        url = f"{self._base_url}/chat/completions"
        payload = _to_openai_payload(request, stream=False)
        try:
            resp = await self._http.post(url, headers=self._headers(), json=payload)
        except httpx.HTTPError as exc:
            raise HttpError(exc) from exc

        request_id = self._extract_request_id(resp.headers)

        if resp.status_code in {401, 403}:
            raise AuthError(f"HTTP {resp.status_code}: {resp.text}")
        if resp.status_code >= 400:
            raise ApiResponseError(
                status=resp.status_code,
                error_type=None,
                message=None,
                body=resp.text,
                retryable=is_retryable_status(resp.status_code),
            )
        try:
            data = json.loads(resp.text)
        except Exception as exc:
            raise JsonError(exc) from exc

        return _parse_openai_response(data, request_id=request_id)

    async def stream_message(self, request: MessageRequest) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        url = f"{self._base_url}/chat/completions"
        payload = _to_openai_payload(request, stream=True)

        try:
            req = self._http.build_request(
                "POST", url, headers=self._headers(stream=True), json=payload
            )
            resp = await self._http.send(req, stream=True)
        except httpx.HTTPError as exc:
            raise HttpError(exc) from exc

        request_id = self._extract_request_id(resp.headers)

        if resp.status_code in {401, 403}:
            body = await resp.aread()
            raise AuthError(f"HTTP {resp.status_code}: {body.decode()}")
        if resp.status_code >= 400:
            body = await resp.aread()
            raise ApiResponseError(
                status=resp.status_code,
                error_type=None,
                message=None,
                body=body.decode(),
                retryable=is_retryable_status(resp.status_code),
            )

        # Emit a synthetic MessageStartEvent so callers don't need special-casing
        yield MessageStartEvent(
            message=MessageResponse(
                id="", kind="message", role="assistant", content=[],
                model=request.model, usage=Usage(), request_id=request_id,
            )
        )

        index_map: dict[int, str] = {}
        buffer = bytearray()

        async for raw_chunk in resp.aiter_bytes():
            buffer.extend(raw_chunk)
            while True:
                # SSE frames are separated by \n\n or \r\n\r\n
                for sep in (b"\r\n\r\n", b"\n\n"):
                    pos = buffer.find(sep)
                    if pos != -1:
                        frame = buffer[:pos].decode("utf-8", errors="replace")
                        del buffer[: pos + len(sep)]

                        for line in frame.splitlines():
                            if not line.startswith("data:"):
                                continue
                            raw = line[len("data:"):].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                chunk_data = json.loads(raw)
                            except Exception:
                                continue
                            for ev in _parse_openai_chunk(chunk_data, index_map):
                                yield ev
                        break
                else:
                    break  # no complete frame yet

        yield MessageStopEvent()

    async def aclose(self) -> None:
        await self._http.aclose()
