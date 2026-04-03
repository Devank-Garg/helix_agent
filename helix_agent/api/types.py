"""helix_agent.api.types — Request/response dataclasses for the Anthropic Messages API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Input content block hierarchy
# ---------------------------------------------------------------------------


@dataclass
class ToolResultContentBlock:
    """Abstract base for tool result content."""
    pass


@dataclass
class TextToolResult(ToolResultContentBlock):
    """Plain-text content for a tool result."""
    text: str


@dataclass
class JsonToolResult(ToolResultContentBlock):
    """JSON-valued content for a tool result."""
    value: Any


# ---------------------------------------------------------------------------


@dataclass
class InputContentBlock:
    """Abstract base for content blocks in an InputMessage."""
    pass


@dataclass
class TextInputBlock(InputContentBlock):
    """A plain-text content block sent by the user or assistant."""
    text: str


@dataclass
class ToolUseInputBlock(InputContentBlock):
    """Represents the assistant's request to invoke a tool."""
    id: str
    name: str
    input: Any  # arbitrary JSON value


@dataclass
class ToolResultInputBlock(InputContentBlock):
    """The user's response to a previous tool-use request."""
    tool_use_id: str
    content: list[ToolResultContentBlock]
    is_error: bool = False


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass
class InputMessage:
    """A single message in the conversation (role + content blocks)."""
    role: str  # "user" or "assistant"
    content: list[InputContentBlock]

    @classmethod
    def user_text(cls, text: str) -> "InputMessage":
        """Convenience constructor: create a plain user text message."""
        return cls(role="user", content=[TextInputBlock(text=text)])

    @classmethod
    def user_tool_result(
        cls,
        tool_use_id: str,
        content: list[ToolResultContentBlock],
        is_error: bool = False,
    ) -> "InputMessage":
        """Convenience constructor: create a tool-result user message."""
        return cls(
            role="user",
            content=[ToolResultInputBlock(tool_use_id, content, is_error)],
        )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@dataclass
class ToolDefinition:
    """Describes a tool available for the model to call."""
    name: str
    description: Optional[str]
    input_schema: dict  # JSON Schema object


# ---------------------------------------------------------------------------
# Tool choice
# ---------------------------------------------------------------------------


class ToolChoice:
    """Tagged union: Auto | Any | Tool(name)."""
    pass


@dataclass
class ToolChoiceAuto(ToolChoice):
    """Let the model decide whether to call a tool."""
    pass


@dataclass
class ToolChoiceAny(ToolChoice):
    """Force the model to call *some* tool."""
    pass


@dataclass
class ToolChoiceTool(ToolChoice):
    """Force the model to call a specific tool by name."""
    name: str


# ---------------------------------------------------------------------------
# MessageRequest
# ---------------------------------------------------------------------------


@dataclass
class MessageRequest:
    """Payload for the POST /v1/messages endpoint."""
    model: str
    max_tokens: int
    messages: list[InputMessage]
    system: Optional[str] = None
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[ToolChoice] = None
    stream: bool = False

    def with_streaming(self) -> "MessageRequest":
        """Enable streaming and return self for chaining."""
        self.stream = True
        return self

    def to_dict(self) -> dict:
        """Serialize to an API-compatible dict (snake_case keys)."""
        result: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [_serialize_input_message(m) for m in self.messages],
        }
        if self.system is not None:
            result["system"] = self.system
        if self.tools:
            result["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in self.tools
            ]
        if self.tool_choice is not None:
            result["tool_choice"] = _serialize_tool_choice(self.tool_choice)
        if self.stream:
            result["stream"] = True
        return result


def _serialize_tool_choice(tc: ToolChoice) -> dict:
    if isinstance(tc, ToolChoiceAuto):
        return {"type": "auto"}
    if isinstance(tc, ToolChoiceAny):
        return {"type": "any"}
    if isinstance(tc, ToolChoiceTool):
        return {"type": "tool", "name": tc.name}
    return {"type": "auto"}


def _serialize_input_message(msg: InputMessage) -> dict:
    return {
        "role": msg.role,
        "content": [_serialize_input_block(b) for b in msg.content],
    }


def _serialize_input_block(block: InputContentBlock) -> dict:
    if isinstance(block, TextInputBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseInputBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultInputBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": [_serialize_tool_result_block(c) for c in block.content],
            "is_error": block.is_error,
        }
    raise TypeError(f"Unknown InputContentBlock type: {type(block)}")


def _serialize_tool_result_block(block: ToolResultContentBlock) -> dict:
    if isinstance(block, TextToolResult):
        return {"type": "text", "text": block.text}
    if isinstance(block, JsonToolResult):
        return {"type": "json", "value": block.value}
    raise TypeError(f"Unknown ToolResultContentBlock type: {type(block)}")


# ---------------------------------------------------------------------------
# Output content blocks
# ---------------------------------------------------------------------------


@dataclass
class OutputContentBlock:
    """Abstract base for content blocks returned by the model."""
    pass


@dataclass
class TextOutputBlock(OutputContentBlock):
    """Plain-text content returned by the model."""
    text: str


@dataclass
class ToolUseOutputBlock(OutputContentBlock):
    """A tool-use request returned by the model."""
    id: str
    name: str
    input: Any  # JSON value (dict after full stream or non-stream)


# ---------------------------------------------------------------------------
# Usage / response
# ---------------------------------------------------------------------------


@dataclass
class Usage:
    """Token consumption counters for a Messages API call."""
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0

    def total_tokens(self) -> int:
        """Sum of all token counters."""
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
            + self.output_tokens
        )


@dataclass
class MessageResponse:
    """Full (non-streaming) response from the Messages API."""
    id: str
    kind: str  # "message"
    role: str
    content: list[OutputContentBlock]
    model: str
    usage: Usage
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    request_id: Optional[str] = None

    def total_tokens(self) -> int:
        """Convenience: total tokens consumed by this response."""
        return self.usage.total_tokens()

    @classmethod
    def from_dict(cls, data: dict, request_id: Optional[str] = None) -> "MessageResponse":
        """Deserialize a Messages API response dict."""
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
        )
        content: list[OutputContentBlock] = []
        for block in data.get("content", []):
            content.append(_parse_output_block(block))
        return cls(
            id=data.get("id", ""),
            kind=data.get("type", "message"),
            role=data.get("role", "assistant"),
            content=content,
            model=data.get("model", ""),
            usage=usage,
            stop_reason=data.get("stop_reason"),
            stop_sequence=data.get("stop_sequence"),
            request_id=request_id,
        )


def _parse_output_block(data: dict) -> OutputContentBlock:
    block_type = data.get("type", "")
    if block_type == "text":
        return TextOutputBlock(text=data.get("text", ""))
    if block_type == "tool_use":
        return ToolUseOutputBlock(
            id=data.get("id", ""),
            name=data.get("name", ""),
            input=data.get("input", {}),
        )
    # Fallback: represent unknown blocks as text
    return TextOutputBlock(text=json.dumps(data))


# ---------------------------------------------------------------------------
# Stream event types
# ---------------------------------------------------------------------------


@dataclass
class MessageDelta:
    """Partial update to a message (stop_reason, etc.) during streaming."""
    stop_reason: Optional[str]
    stop_sequence: Optional[str]


@dataclass
class ContentBlockDelta:
    """Abstract base for incremental content block deltas."""
    pass


@dataclass
class TextDelta(ContentBlockDelta):
    """Incremental text content."""
    text: str


@dataclass
class InputJsonDelta(ContentBlockDelta):
    """Incremental JSON fragment for tool-use input."""
    partial_json: str


class StreamEvent:
    """Abstract base for all SSE stream events."""
    pass


@dataclass
class MessageStartEvent(StreamEvent):
    """Signals the start of a new message; carries the initial MessageResponse shell."""
    message: MessageResponse


@dataclass
class MessageDeltaEvent(StreamEvent):
    """Carries stop_reason / usage updates during streaming."""
    delta: MessageDelta
    usage: Usage


@dataclass
class ContentBlockStartEvent(StreamEvent):
    """Signals that a new content block is beginning."""
    index: int
    content_block: OutputContentBlock


@dataclass
class ContentBlockDeltaEvent(StreamEvent):
    """Carries an incremental delta for a content block."""
    index: int
    delta: ContentBlockDelta


@dataclass
class ContentBlockStopEvent(StreamEvent):
    """Signals that a content block is complete."""
    index: int


@dataclass
class MessageStopEvent(StreamEvent):
    """Signals that the full message is complete."""
    pass


# ---------------------------------------------------------------------------
# Stream event parser
# ---------------------------------------------------------------------------


def parse_stream_event(event_name: str, data_json: str) -> Optional[StreamEvent]:
    """Deserialize one SSE data payload into the appropriate StreamEvent subclass.

    Returns None for events that should be silently skipped (ping, [DONE]).
    """
    import json as _json

    try:
        data = _json.loads(data_json)
    except Exception:
        return None

    event_type = data.get("type", event_name or "")

    if event_type == "message_start":
        msg_data = data.get("message", {})
        return MessageStartEvent(message=MessageResponse.from_dict(msg_data))

    if event_type == "message_delta":
        delta_data = data.get("delta", {})
        usage_data = data.get("usage", {})
        return MessageDeltaEvent(
            delta=MessageDelta(
                stop_reason=delta_data.get("stop_reason"),
                stop_sequence=delta_data.get("stop_sequence"),
            ),
            usage=Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
            ),
        )

    if event_type == "message_stop":
        return MessageStopEvent()

    if event_type == "content_block_start":
        block_data = data.get("content_block", {})
        return ContentBlockStartEvent(
            index=data.get("index", 0),
            content_block=_parse_output_block(block_data),
        )

    if event_type == "content_block_delta":
        delta_data = data.get("delta", {})
        delta_type = delta_data.get("type", "")
        if delta_type == "text_delta":
            delta: ContentBlockDelta = TextDelta(text=delta_data.get("text", ""))
        elif delta_type == "input_json_delta":
            delta = InputJsonDelta(partial_json=delta_data.get("partial_json", ""))
        else:
            delta = TextDelta(text="")
        return ContentBlockDeltaEvent(
            index=data.get("index", 0),
            delta=delta,
        )

    if event_type == "content_block_stop":
        return ContentBlockStopEvent(index=data.get("index", 0))

    # ping or unknown — skip
    return None
