"""Integration tests — hit real provider endpoints.

Run with your keys set:
    GEMINI_API_KEY=AIza...  pytest tests/test_integration.py -v -s

Each test is skipped automatically when its key is absent.
"""

from __future__ import annotations

import os
import pytest

from helix_agent.api.factory import create_client
from helix_agent.api.types import (
    InputMessage,
    MessageRequest,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    TextOutputBlock,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PING = "Reply with exactly one word: hello"


def _simple_request(model: str) -> MessageRequest:
    return MessageRequest(
        model=model,
        max_tokens=16,
        messages=[InputMessage.user_text(PING)],
    )


def _require_key(env: str) -> str:
    key = os.environ.get(env, "").strip()
    if not key:
        pytest.skip(f"{env} not set")
    return key


# ---------------------------------------------------------------------------
# Gemini (OpenAI-compat endpoint)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_gemini_send_message() -> None:
    key = _require_key("GEMINI_API_KEY")
    client = create_client("gemini", api_key=key)
    async with client:
        resp = await client.send_message(_simple_request("gemini-2.0-flash"))

    assert isinstance(resp, MessageResponse)
    assert resp.content, "Expected at least one content block"
    assert isinstance(resp.content[0], TextOutputBlock)
    text = resp.content[0].text.lower()
    assert "hello" in text, f"Unexpected reply: {text!r}"
    assert resp.usage.input_tokens > 0


@pytest.mark.anyio
async def test_gemini_stream_message() -> None:
    key = _require_key("GEMINI_API_KEY")
    client = create_client("gemini", api_key=key)
    events = []
    async with client:
        async for ev in client.stream_message(_simple_request("gemini-2.0-flash")):
            events.append(ev)

    event_types = {type(e).__name__ for e in events}
    assert "MessageStartEvent" in event_types
    assert "MessageStopEvent" in event_types
    assert "ContentBlockDeltaEvent" in event_types, \
        f"Got event types: {event_types}"


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_openai_send_message() -> None:
    key = _require_key("OPENAI_API_KEY")
    client = create_client("openai", api_key=key)
    async with client:
        resp = await client.send_message(_simple_request("gpt-4o-mini"))

    assert isinstance(resp, MessageResponse)
    assert resp.content
    assert isinstance(resp.content[0], TextOutputBlock)


@pytest.mark.anyio
async def test_openai_stream_message() -> None:
    key = _require_key("OPENAI_API_KEY")
    client = create_client("openai", api_key=key)
    events = []
    async with client:
        async for ev in client.stream_message(_simple_request("gpt-4o-mini")):
            events.append(ev)

    assert any(isinstance(e, MessageStartEvent) for e in events)
    assert any(isinstance(e, MessageStopEvent) for e in events)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_anthropic_send_message() -> None:
    key = _require_key("ANTHROPIC_API_KEY")
    client = create_client("anthropic", api_key=key)
    async with client:
        resp = await client.send_message(
            _simple_request("claude-haiku-4-5-20251001")
        )

    assert isinstance(resp, MessageResponse)
    assert resp.content
    assert isinstance(resp.content[0], TextOutputBlock)


@pytest.mark.anyio
async def test_anthropic_stream_message() -> None:
    key = _require_key("ANTHROPIC_API_KEY")
    client = create_client("anthropic", api_key=key)
    events = []
    async with client:
        async for ev in client.stream_message(
            _simple_request("claude-haiku-4-5-20251001")
        ):
            events.append(ev)

    assert any(isinstance(e, MessageStartEvent) for e in events)
    assert any(isinstance(e, MessageStopEvent) for e in events)


# ---------------------------------------------------------------------------
# Ollama (local — skip if not running)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_ollama_send_message() -> None:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as h:
            r = await h.get("http://localhost:11434/api/tags")
            if r.status_code != 200:
                pytest.skip("Ollama not running")
            models = [m["name"] for m in r.json().get("models", [])]
            if not models:
                pytest.skip("No Ollama models pulled")
            model = models[0]
    except Exception:
        pytest.skip("Ollama not running")

    client = create_client("ollama")
    async with client:
        resp = await client.send_message(_simple_request(model))

    assert isinstance(resp, MessageResponse)
    assert resp.content
