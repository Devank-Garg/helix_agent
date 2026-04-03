"""helix_agent.runtime.permissions — Permission policy for tool authorisation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum


# ---------------------------------------------------------------------------
# Permission mode
# ---------------------------------------------------------------------------


class PermissionMode(IntEnum):
    """
    Ordered permission levels.  Higher values grant more capabilities.

    Special values:
    - ``PROMPT`` (98): always prompt the user before executing.
    - ``ALLOW``  (99): always allow without any check.
    """

    READ_ONLY = 0
    WORKSPACE_WRITE = 1
    DANGER_FULL_ACCESS = 2
    PROMPT = 98   # Always prompt the user
    ALLOW = 99    # Always allow (testing / CI mode)


# ---------------------------------------------------------------------------
# Permission request / outcome
# ---------------------------------------------------------------------------


@dataclass
class PermissionRequest:
    """Describes a pending authorisation decision."""

    tool_name: str
    input: str
    current_mode: PermissionMode
    required_mode: PermissionMode


@dataclass
class PermissionOutcome:
    """The result of an authorisation check."""

    allowed: bool
    reason: str = ""
    denied: bool = False  # Convenience alias for ``not allowed``


# ---------------------------------------------------------------------------
# Prompter interface
# ---------------------------------------------------------------------------


class PermissionPrompter(ABC):
    """
    Abstract interface for interactively asking the user whether a tool
    execution should be allowed.
    """

    @abstractmethod
    def decide(self, request: PermissionRequest) -> PermissionOutcome:
        """Present *request* to the user and return their decision."""
        ...


# ---------------------------------------------------------------------------
# Permission policy
# ---------------------------------------------------------------------------


class PermissionPolicy:
    """
    Determines whether a tool execution should be allowed based on the
    active permission mode and per-tool requirements.

    Authorization logic:

    - If ``active_mode >= required_mode``: **Allow**.
    - Elif ``active_mode == PROMPT`` or
      ``(active_mode == WORKSPACE_WRITE and required == DANGER_FULL_ACCESS)``:
        - Call ``prompter.decide()`` if a prompter is provided.
        - Otherwise **Deny** with an explanatory reason.
    - Else: **Deny** with an explanatory reason.

    The default required mode for any tool that has not been explicitly
    registered is ``DANGER_FULL_ACCESS``.
    """

    def __init__(self, active_mode: PermissionMode) -> None:
        self._active = active_mode
        self._requirements: dict[str, PermissionMode] = {}

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def with_tool_requirement(
        self, tool_name: str, required: PermissionMode
    ) -> "PermissionPolicy":
        """Register a minimum permission level for *tool_name*."""
        self._requirements[tool_name] = required
        return self

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def active_mode(self) -> PermissionMode:
        """Return the currently active permission mode."""
        return self._active

    def required_mode_for(self, tool_name: str) -> PermissionMode:
        """Return the minimum permission required to execute *tool_name*."""
        return self._requirements.get(tool_name, PermissionMode.DANGER_FULL_ACCESS)

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def authorize(
        self,
        tool_name: str,
        input: str,
        prompter: PermissionPrompter | None = None,
    ) -> PermissionOutcome:
        """
        Decide whether the tool may run.

        Returns a :class:`PermissionOutcome`.  The ``denied`` field is
        ``True`` whenever ``allowed`` is ``False`` (both are always set
        consistently).
        """
        required = self.required_mode_for(tool_name)

        # Shortcut: ALLOW mode bypasses all checks.
        if self._active == PermissionMode.ALLOW:
            return PermissionOutcome(allowed=True)

        # Standard comparison: sufficient mode → allow.
        if self._active >= required:
            return PermissionOutcome(allowed=True)

        # Prompt modes: ask the user if we can.
        needs_prompt = (
            self._active == PermissionMode.PROMPT
            or (
                self._active == PermissionMode.WORKSPACE_WRITE
                and required == PermissionMode.DANGER_FULL_ACCESS
            )
        )
        if needs_prompt:
            if prompter is not None:
                req = PermissionRequest(
                    tool_name=tool_name,
                    input=input,
                    current_mode=self._active,
                    required_mode=required,
                )
                return prompter.decide(req)
            return PermissionOutcome(
                allowed=False,
                denied=True,
                reason="No prompter available to request user confirmation",
            )

        # Default deny.
        return PermissionOutcome(
            allowed=False,
            denied=True,
            reason=(
                f"Tool '{tool_name}' requires {required.name} "
                f"but the current mode is {self._active.name}"
            ),
        )
