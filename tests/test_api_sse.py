"""Tests for helix_agent.api.sse — SSE frame parsing."""

import json
import pytest

from helix_agent.api.sse import SseParser, parse_frame
from helix_agent.api.types import (
    ContentBlockDeltaEvent,
    MessageStopEvent,
    TextDelta,
)


# ---------------------------------------------------------------------------
# parse_frame — single frame parsing
# ---------------------------------------------------------------------------


def _make_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}"


def test_parse_frame_message_stop_lf() -> None:
    frame = _make_frame("message_stop", {"type": "message_stop"})
    ev = parse_frame(frame)
    assert isinstance(ev, MessageStopEvent)


def test_parse_frame_skips_ping() -> None:
    frame = "event: ping\ndata: {}"
    assert parse_frame(frame) is None


def test_parse_frame_skips_done_sentinel() -> None:
    frame = "data: [DONE]"
    assert parse_frame(frame) is None


def test_parse_frame_skips_comment_lines() -> None:
    # Comment lines start with ':'
    frame = ": this is a comment\nevent: message_stop\ndata: {\"type\": \"message_stop\"}"
    ev = parse_frame(frame)
    assert isinstance(ev, MessageStopEvent)


def test_parse_frame_empty_returns_none() -> None:
    assert parse_frame("") is None
    assert parse_frame("   ") is None


def test_parse_frame_content_block_delta() -> None:
    data = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hi"},
    }
    frame = _make_frame("content_block_delta", data)
    ev = parse_frame(frame)
    assert isinstance(ev, ContentBlockDeltaEvent)
    assert isinstance(ev.delta, TextDelta)
    assert ev.delta.text == "Hi"


# ---------------------------------------------------------------------------
# SseParser — incremental byte-stream parsing
# ---------------------------------------------------------------------------


def _sse_bytes(event: str, data: dict, crlf: bool = False) -> bytes:
    sep = "\r\n" if crlf else "\n"
    terminator = "\r\n\r\n" if crlf else "\n\n"
    frame = f"event: {event}{sep}data: {json.dumps(data)}{terminator}"
    return frame.encode()


def test_sse_parser_single_event_lf() -> None:
    parser = SseParser()
    chunk = _sse_bytes("message_stop", {"type": "message_stop"}, crlf=False)
    events = parser.push(chunk)
    assert len(events) == 1
    assert isinstance(events[0], MessageStopEvent)


def test_sse_parser_single_event_crlf() -> None:
    parser = SseParser()
    chunk = _sse_bytes("message_stop", {"type": "message_stop"}, crlf=True)
    events = parser.push(chunk)
    assert len(events) == 1
    assert isinstance(events[0], MessageStopEvent)


def test_sse_parser_multiple_events_in_one_chunk() -> None:
    parser = SseParser()
    chunk = (
        _sse_bytes("message_stop", {"type": "message_stop"})
        + _sse_bytes("message_stop", {"type": "message_stop"})
    )
    events = parser.push(chunk)
    assert len(events) == 2


def test_sse_parser_split_across_chunks() -> None:
    parser = SseParser()
    full = _sse_bytes("message_stop", {"type": "message_stop"})
    # Split arbitrarily in the middle
    mid = len(full) // 2
    events1 = parser.push(full[:mid])
    events2 = parser.push(full[mid:])
    assert events1 == []
    assert len(events2) == 1
    assert isinstance(events2[0], MessageStopEvent)


def test_sse_parser_skips_ping_frames() -> None:
    parser = SseParser()
    chunk = _sse_bytes("ping", {})
    events = parser.push(chunk)
    assert events == []


def test_sse_parser_finish_flushes_remaining() -> None:
    parser = SseParser()
    # Push a frame WITHOUT the trailing double-newline
    frame = 'event: message_stop\ndata: {"type": "message_stop"}'
    parser.push(frame.encode())
    events = parser.finish()
    assert len(events) == 1
    assert isinstance(events[0], MessageStopEvent)


def test_sse_parser_finish_clears_buffer() -> None:
    parser = SseParser()
    parser.push(b"partial")
    parser.finish()
    # After finish, buffer should be empty — another finish returns nothing
    assert parser.finish() == []


def test_sse_parser_multiple_lf_frames() -> None:
    # LF-only stream: each frame terminated with \n\n
    parser = SseParser()
    chunk = (
        _sse_bytes("message_stop", {"type": "message_stop"}, crlf=False)
        + _sse_bytes("message_stop", {"type": "message_stop"}, crlf=False)
    )
    events = parser.push(chunk)
    assert len(events) == 2


def test_sse_parser_multiple_crlf_frames() -> None:
    # CRLF-only stream: each frame terminated with \r\n\r\n
    parser = SseParser()
    chunk = (
        _sse_bytes("message_stop", {"type": "message_stop"}, crlf=True)
        + _sse_bytes("message_stop", {"type": "message_stop"}, crlf=True)
    )
    events = parser.push(chunk)
    assert len(events) == 2


def test_sse_parser_done_sentinel_skipped() -> None:
    parser = SseParser()
    chunk = b"data: [DONE]\n\n"
    events = parser.push(chunk)
    assert events == []


def test_sse_parser_byte_by_byte() -> None:
    """Parser must handle one-byte-at-a-time streaming without crashing."""
    parser = SseParser()
    full = _sse_bytes("message_stop", {"type": "message_stop"})
    all_events = []
    for i in range(len(full)):
        all_events.extend(parser.push(full[i : i + 1]))
    assert len(all_events) == 1
    assert isinstance(all_events[0], MessageStopEvent)
