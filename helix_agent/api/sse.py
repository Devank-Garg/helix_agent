"""helix_agent.api.sse — SSE frame parsing and incremental byte-stream consumer."""

from __future__ import annotations

from typing import Optional

from .error import InvalidSseFrame
from .types import StreamEvent, parse_stream_event


def parse_frame(frame: str) -> Optional[StreamEvent]:
    """Parse one SSE frame (terminated by a blank line).

    Frame format::

        event: <event_name>
        data: <json_payload>

    Rules:
    - Lines starting with ``':'`` are comments and are skipped.
    - ``'event:'`` lines set the event name.
    - ``'data:'`` lines are accumulated and joined with ``'\\n'``.
    - Returns ``None`` if event is ``'ping'`` or data is ``'[DONE]'``.
    - Deserializes JSON data into a :class:`StreamEvent` via the ``'type'`` field.
    - Raises :class:`InvalidSseFrame` on deserialization failure.
    """
    event_name: Optional[str] = None
    data_lines: list[str] = []

    for line in frame.splitlines():
        if line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())

    if event_name == "ping":
        return None

    data = "\n".join(data_lines)
    if data == "[DONE]":
        return None

    if not data:
        return None

    try:
        event = parse_stream_event(event_name or "", data)
    except Exception as exc:
        raise InvalidSseFrame(f"Failed to parse SSE frame data: {exc}") from exc

    return event


class SseParser:
    """Incremental SSE parser.

    Feed raw chunks of bytes via :meth:`push`; receive fully-parsed
    :class:`StreamEvent` instances back.  Any unparseable trailing bytes are
    kept in an internal buffer until the next :meth:`push` or :meth:`finish`.
    """

    def __init__(self) -> None:
        self._buffer: bytearray = bytearray()

    def push(self, chunk: bytes) -> list[StreamEvent]:
        """Append *chunk* to the internal buffer and return all complete events.

        Frames are separated by ``\\r\\n\\r\\n`` (CRLF) or ``\\n\\n`` (LF).
        The ``\\r\\n\\r\\n`` terminator is checked first so that CRLF streams are
        handled correctly.
        """
        self._buffer.extend(chunk)
        events: list[StreamEvent] = []
        while True:
            found = False
            for sep in (b"\r\n\r\n", b"\n\n"):
                idx = self._buffer.find(sep)
                if idx != -1:
                    frame_bytes = bytes(self._buffer[:idx])
                    del self._buffer[: idx + len(sep)]
                    frame = frame_bytes.decode("utf-8", errors="replace")
                    event = parse_frame(frame)
                    if event is not None:
                        events.append(event)
                    found = True
                    break
            if not found:
                break
        return events

    def finish(self) -> list[StreamEvent]:
        """Flush any remaining buffered bytes as a final frame.

        Clears the internal buffer after processing.
        """
        remaining = self._buffer.decode("utf-8", errors="replace").strip()
        self._buffer.clear()
        if remaining:
            event = parse_frame(remaining)
            if event is not None:
                return [event]
        return []
