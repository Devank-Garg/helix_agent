"""helix_agent.runtime.session — Session model with JSON round-trip support."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ContentBlock:
    """Abstract base for content blocks within a message."""
    pass


@dataclass
class TextBlock(ContentBlock):
    """A plain-text content block."""

    text: str

    def to_dict(self) -> dict:
        return {"type": "text", "text": self.text}

    @classmethod
    def from_dict(cls, data: dict) -> "TextBlock":
        return cls(text=data["text"])


@dataclass
class ToolUseBlock(ContentBlock):
    """Represents a tool invocation in an assistant message."""

    id: str
    name: str
    input: str  # JSON string

    def to_dict(self) -> dict:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}

    @classmethod
    def from_dict(cls, data: dict) -> "ToolUseBlock":
        return cls(id=data["id"], name=data["name"], input=data["input"])


@dataclass
class ToolResultBlock(ContentBlock):
    """Represents the result of a tool execution, attached to a user message."""

    tool_use_id: str
    tool_name: str
    output: str
    is_error: bool = False

    def to_dict(self) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "tool_name": self.tool_name,
            "output": self.output,
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolResultBlock":
        return cls(
            tool_use_id=data["tool_use_id"],
            tool_name=data["tool_name"],
            output=data["output"],
            is_error=data.get("is_error", False),
        )


def _block_to_dict(block: ContentBlock) -> dict:
    """Dispatch serialization for any ContentBlock subtype."""
    if isinstance(block, TextBlock):
        return block.to_dict()
    elif isinstance(block, ToolUseBlock):
        return block.to_dict()
    elif isinstance(block, ToolResultBlock):
        return block.to_dict()
    raise TypeError(f"Unknown ContentBlock type: {type(block)}")


def _block_from_dict(data: dict) -> ContentBlock:
    """Dispatch deserialization based on 'type' field."""
    t = data.get("type")
    if t == "text":
        return TextBlock.from_dict(data)
    elif t == "tool_use":
        return ToolUseBlock.from_dict(data)
    elif t == "tool_result":
        return ToolResultBlock.from_dict(data)
    raise ValueError(f"Unknown block type: {t!r}")


def _token_usage_to_dict(usage: Any) -> Optional[dict]:
    """Serialize a TokenUsage object to dict, or None."""
    if usage is None:
        return None
    # Import here to avoid circular deps; TokenUsage is a dataclass
    return {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
    }


def _token_usage_from_dict(data: Optional[dict]) -> Any:
    """Deserialize a dict into a TokenUsage object, or None."""
    if data is None:
        return None
    try:
        from helix_agent.runtime.usage import TokenUsage  # type: ignore[import]
        return TokenUsage(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_creation_input_tokens=data.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=data.get("cache_read_input_tokens", 0),
        )
    except ImportError:
        return data  # fall back to raw dict if usage module not yet available


@dataclass
class ConversationMessage:
    """A single turn in the conversation (user, assistant, or tool result)."""

    role: MessageRole
    blocks: list[ContentBlock]
    usage: Optional[Any] = None  # TokenUsage

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def user_text(cls, text: str) -> "ConversationMessage":
        """Create a user message with a single text block."""
        return cls(role=MessageRole.USER, blocks=[TextBlock(text)])

    @classmethod
    def assistant(cls, blocks: list[ContentBlock]) -> "ConversationMessage":
        """Create an assistant message from a list of blocks."""
        return cls(role=MessageRole.ASSISTANT, blocks=blocks)

    @classmethod
    def assistant_with_usage(cls, blocks: list[ContentBlock], usage: Any) -> "ConversationMessage":
        """Create an assistant message with token-usage metadata attached."""
        return cls(role=MessageRole.ASSISTANT, blocks=blocks, usage=usage)

    @classmethod
    def tool_result(
        cls,
        tool_use_id: str,
        tool_name: str,
        output: str,
        is_error: bool,
    ) -> "ConversationMessage":
        """Create a user message carrying a tool result."""
        return cls(
            role=MessageRole.USER,
            blocks=[ToolResultBlock(tool_use_id, tool_name, output, is_error)],
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "blocks": [_block_to_dict(b) for b in self.blocks],
            "usage": _token_usage_to_dict(self.usage),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMessage":
        role = MessageRole(data["role"])
        blocks = [_block_from_dict(b) for b in data.get("blocks", [])]
        usage = _token_usage_from_dict(data.get("usage"))
        return cls(role=role, blocks=blocks, usage=usage)


@dataclass
class Session:
    """
    Represents a full conversation session.

    Supports JSON round-trip via ``save_to_path`` / ``load_from_path``.
    """

    version: int = 1
    messages: list[ConversationMessage] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def new(cls) -> "Session":
        """Return an empty session."""
        return cls()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_path(self, path: pathlib.Path) -> None:
        """Serialise the session to *path* as pretty-printed JSON."""
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_from_path(cls, path: pathlib.Path) -> "Session":
        """Deserialise a session from *path*."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        version = data.get("version", 1)
        messages = [ConversationMessage.from_dict(m) for m in data.get("messages", [])]
        return cls(version=version, messages=messages)
