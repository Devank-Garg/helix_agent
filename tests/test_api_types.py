"""Tests for helix_agent.api.types — serialization and deserialization."""

import json
import pytest

from helix_agent.api.types import (
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
    TextDelta,
    TextInputBlock,
    TextOutputBlock,
    TextToolResult,
    ToolChoiceAny,
    ToolChoiceAuto,
    ToolChoiceTool,
    ToolDefinition,
    ToolResultInputBlock,
    ToolUseInputBlock,
    ToolUseOutputBlock,
    Usage,
    parse_stream_event,
)


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


def test_usage_total_tokens() -> None:
    u = Usage(
        input_tokens=10,
        cache_creation_input_tokens=5,
        cache_read_input_tokens=3,
        output_tokens=20,
    )
    assert u.total_tokens() == 38


def test_usage_defaults_zero() -> None:
    u = Usage()
    assert u.total_tokens() == 0


# ---------------------------------------------------------------------------
# InputMessage helpers
# ---------------------------------------------------------------------------


def test_input_message_user_text() -> None:
    msg = InputMessage.user_text("hello")
    assert msg.role == "user"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextInputBlock)
    assert msg.content[0].text == "hello"  # type: ignore[union-attr]


def test_input_message_user_tool_result() -> None:
    msg = InputMessage.user_tool_result(
        tool_use_id="tu_1",
        content=[TextToolResult(text="result")],
        is_error=False,
    )
    assert msg.role == "user"
    block = msg.content[0]
    assert isinstance(block, ToolResultInputBlock)
    assert block.tool_use_id == "tu_1"


# ---------------------------------------------------------------------------
# MessageRequest.to_dict — basic
# ---------------------------------------------------------------------------


def test_message_request_to_dict_minimal() -> None:
    req = MessageRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[InputMessage.user_text("hi")],
    )
    d = req.to_dict()
    assert d["model"] == "claude-3-5-sonnet-20241022"
    assert d["max_tokens"] == 1024
    assert d["messages"][0]["role"] == "user"
    assert d["messages"][0]["content"][0] == {"type": "text", "text": "hi"}
    assert "stream" not in d
    assert "system" not in d
    assert "tools" not in d


def test_message_request_with_streaming() -> None:
    req = MessageRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        messages=[InputMessage.user_text("hi")],
    )
    req.with_streaming()
    d = req.to_dict()
    assert d["stream"] is True


def test_message_request_with_system() -> None:
    req = MessageRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        messages=[InputMessage.user_text("hi")],
        system="You are helpful.",
    )
    d = req.to_dict()
    assert d["system"] == "You are helpful."


def test_message_request_with_tools() -> None:
    req = MessageRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        messages=[InputMessage.user_text("hi")],
        tools=[
            ToolDefinition(
                name="get_weather",
                description="Get current weather",
                input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ],
    )
    d = req.to_dict()
    assert len(d["tools"]) == 1
    assert d["tools"][0]["name"] == "get_weather"


def test_message_request_tool_choice_auto() -> None:
    req = MessageRequest(
        model="m", max_tokens=10,
        messages=[InputMessage.user_text("x")],
        tool_choice=ToolChoiceAuto(),
    )
    assert req.to_dict()["tool_choice"] == {"type": "auto"}


def test_message_request_tool_choice_any() -> None:
    req = MessageRequest(
        model="m", max_tokens=10,
        messages=[InputMessage.user_text("x")],
        tool_choice=ToolChoiceAny(),
    )
    assert req.to_dict()["tool_choice"] == {"type": "any"}


def test_message_request_tool_choice_specific() -> None:
    req = MessageRequest(
        model="m", max_tokens=10,
        messages=[InputMessage.user_text("x")],
        tool_choice=ToolChoiceTool(name="my_tool"),
    )
    assert req.to_dict()["tool_choice"] == {"type": "tool", "name": "my_tool"}


# ---------------------------------------------------------------------------
# Content block serialization
# ---------------------------------------------------------------------------


def test_tool_use_input_block_serialized() -> None:
    req = MessageRequest(
        model="m", max_tokens=10,
        messages=[
            InputMessage(
                role="assistant",
                content=[ToolUseInputBlock(id="tu_1", name="search", input={"q": "foo"})],
            )
        ],
    )
    block = req.to_dict()["messages"][0]["content"][0]
    assert block == {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "foo"}}


def test_tool_result_input_block_serialized() -> None:
    req = MessageRequest(
        model="m", max_tokens=10,
        messages=[
            InputMessage.user_tool_result(
                tool_use_id="tu_1",
                content=[TextToolResult(text="42 degrees")],
                is_error=False,
            )
        ],
    )
    block = req.to_dict()["messages"][0]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "tu_1"
    assert block["is_error"] is False
    assert block["content"][0] == {"type": "text", "text": "42 degrees"}


# ---------------------------------------------------------------------------
# MessageResponse.from_dict
# ---------------------------------------------------------------------------

SAMPLE_RESPONSE: dict = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello!"}],
    "model": "claude-3-5-sonnet-20241022",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 10,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 5,
    },
}


def test_message_response_from_dict() -> None:
    resp = MessageResponse.from_dict(SAMPLE_RESPONSE, request_id="req_01")
    assert resp.id == "msg_01"
    assert resp.role == "assistant"
    assert resp.stop_reason == "end_turn"
    assert resp.request_id == "req_01"
    assert isinstance(resp.content[0], TextOutputBlock)
    assert resp.content[0].text == "Hello!"  # type: ignore[union-attr]
    assert resp.usage.input_tokens == 10
    assert resp.total_tokens() == 15


def test_message_response_from_dict_tool_use_block() -> None:
    data = {
        "id": "msg_02",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "weather"}}
        ],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 20, "output_tokens": 8,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }
    resp = MessageResponse.from_dict(data)
    assert isinstance(resp.content[0], ToolUseOutputBlock)
    block = resp.content[0]  # type: ignore[assignment]
    assert block.name == "search"
    assert block.input == {"q": "weather"}


def test_message_response_unknown_block_falls_back_to_text() -> None:
    data = {**SAMPLE_RESPONSE, "content": [{"type": "unknown_block", "x": 1}]}
    resp = MessageResponse.from_dict(data)
    assert isinstance(resp.content[0], TextOutputBlock)


# ---------------------------------------------------------------------------
# parse_stream_event
# ---------------------------------------------------------------------------


def test_parse_stream_event_message_start() -> None:
    payload = json.dumps({
        "type": "message_start",
        "message": {**SAMPLE_RESPONSE},
    })
    ev = parse_stream_event("message_start", payload)
    assert isinstance(ev, MessageStartEvent)
    assert ev.message.id == "msg_01"


def test_parse_stream_event_message_delta() -> None:
    payload = json.dumps({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 12, "input_tokens": 0,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    })
    ev = parse_stream_event("message_delta", payload)
    assert isinstance(ev, MessageDeltaEvent)
    assert ev.delta.stop_reason == "end_turn"
    assert ev.usage.output_tokens == 12


def test_parse_stream_event_message_stop() -> None:
    payload = json.dumps({"type": "message_stop"})
    ev = parse_stream_event("message_stop", payload)
    assert isinstance(ev, MessageStopEvent)


def test_parse_stream_event_content_block_start_text() -> None:
    payload = json.dumps({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })
    ev = parse_stream_event("content_block_start", payload)
    assert isinstance(ev, ContentBlockStartEvent)
    assert ev.index == 0


def test_parse_stream_event_content_block_delta_text() -> None:
    payload = json.dumps({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello"},
    })
    ev = parse_stream_event("content_block_delta", payload)
    assert isinstance(ev, ContentBlockDeltaEvent)
    assert isinstance(ev.delta, TextDelta)
    assert ev.delta.text == "Hello"


def test_parse_stream_event_content_block_delta_input_json() -> None:
    payload = json.dumps({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": '{"q":'},
    })
    ev = parse_stream_event("content_block_delta", payload)
    assert isinstance(ev, ContentBlockDeltaEvent)
    assert isinstance(ev.delta, InputJsonDelta)
    assert ev.delta.partial_json == '{"q":'


def test_parse_stream_event_content_block_stop() -> None:
    payload = json.dumps({"type": "content_block_stop", "index": 1})
    ev = parse_stream_event("content_block_stop", payload)
    assert isinstance(ev, ContentBlockStopEvent)
    assert ev.index == 1


def test_parse_stream_event_ping_returns_none() -> None:
    ev = parse_stream_event("ping", "{}")
    assert ev is None


def test_parse_stream_event_bad_json_returns_none() -> None:
    ev = parse_stream_event("message_stop", "not-json")
    assert ev is None
