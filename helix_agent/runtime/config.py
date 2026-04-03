"""helix_agent.runtime.config — Configuration loading and merging from multiple sources."""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ConfigSource(Enum):
    USER = 0
    PROJECT = 1
    LOCAL = 2  # Highest precedence


class ResolvedPermissionMode(Enum):
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL_ACCESS = "danger-full-access"


# ---------------------------------------------------------------------------
# MCP server config variants
# ---------------------------------------------------------------------------


@dataclass
class McpStdioServerConfig:
    """Config for an MCP server launched as a subprocess (stdio transport)."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class McpRemoteServerConfig:
    """Config for a remote MCP server reached via HTTP."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    headers_helper: Optional[str] = None
    oauth: Optional[Any] = None  # McpOAuthConfig


@dataclass
class McpWebSocketServerConfig:
    """Config for a remote MCP server reached via WebSocket."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    headers_helper: Optional[str] = None


@dataclass
class McpSdkServerConfig:
    """Config for a built-in SDK MCP server identified by name."""

    name: str


@dataclass
class McpClaudeAiProxyServerConfig:
    """Config for an MCP server proxied through Claude.ai."""

    url: str
    id: str


# Tagged union type alias
McpServerConfig = (
    McpStdioServerConfig
    | McpRemoteServerConfig
    | McpWebSocketServerConfig
    | McpSdkServerConfig
    | McpClaudeAiProxyServerConfig
)


@dataclass
class ScopedMcpServerConfig:
    """An MCP server config paired with the config scope it was loaded from."""

    scope: ConfigSource
    config: McpServerConfig


# ---------------------------------------------------------------------------
# Sandbox config
# ---------------------------------------------------------------------------


@dataclass
class SandboxConfig:
    """Configuration for the Linux sandbox (bwrap/firejail)."""

    enabled: Optional[bool] = None
    namespace_restrictions: Optional[bool] = None
    network_isolation: Optional[bool] = None
    filesystem_mode: Optional[str] = None  # "off" | "workspace_only" | "allow_list"
    allowed_mounts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hook config
# ---------------------------------------------------------------------------


@dataclass
class RuntimeHookConfig:
    """
    Hook command lists parsed from settings.json under the "hooks" key.

    Expected JSON shape::

        {
            "PreToolUse": ["cmd1", "cmd2"],
            "PostToolUse": ["cmd3"]
        }
    """

    pre_tool_use: list[str] = field(default_factory=list)
    post_tool_use: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OAuth config
# ---------------------------------------------------------------------------


@dataclass
class OAuthConfig:
    """OAuth 2.0 client configuration."""

    client_id: str
    authorize_url: str
    token_url: str
    callback_port: int = 4545
    scopes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------


@dataclass
class ConfigEntry:
    """A single resolved config file with its scope."""

    source: ConfigSource
    path: pathlib.Path


@dataclass
class RuntimeConfig:
    """
    Merged configuration from all discovered config files.

    Access specific sub-sections via the accessor methods rather than
    reading ``_merged`` directly.
    """

    _merged: dict[str, Any] = field(default_factory=dict)
    _loaded_entries: list[ConfigEntry] = field(default_factory=list)
    _mcp: dict[str, ScopedMcpServerConfig] = field(default_factory=dict)
    _oauth: Optional[OAuthConfig] = None
    _model: Optional[str] = None
    _permission_mode: Optional[ResolvedPermissionMode] = None
    _sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    _hooks: RuntimeHookConfig = field(default_factory=RuntimeHookConfig)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Return the raw merged value for *key*, or None."""
        return self._merged.get(key)

    def model(self) -> Optional[str]:
        """Return the configured model name, or None."""
        return self._model

    def permission_mode(self) -> Optional[ResolvedPermissionMode]:
        """Return the resolved permission mode, or None."""
        return self._permission_mode

    def oauth(self) -> Optional[OAuthConfig]:
        """Return the OAuth client config, or None."""
        return self._oauth

    def sandbox(self) -> SandboxConfig:
        """Return the sandbox config."""
        return self._sandbox

    def mcp(self) -> dict[str, ScopedMcpServerConfig]:
        """Return the map of MCP server name → scoped config."""
        return self._mcp

    def hooks(self) -> RuntimeHookConfig:
        """Return the hook runner config."""
        return self._hooks

    def loaded_entries(self) -> list[ConfigEntry]:
        """Return the ordered list of config files that were loaded."""
        return self._loaded_entries


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_mcp_server(raw: dict, scope: ConfigSource) -> Optional[ScopedMcpServerConfig]:
    """
    Parse a single MCP server entry from config JSON.

    Determines the variant by inspecting the keys present.
    """
    if "command" in raw:
        cfg: McpServerConfig = McpStdioServerConfig(
            command=raw["command"],
            args=raw.get("args", []),
            env=raw.get("env", {}),
        )
    elif "url" in raw and raw.get("type") == "websocket":
        cfg = McpWebSocketServerConfig(
            url=raw["url"],
            headers=raw.get("headers", {}),
            headers_helper=raw.get("headersHelper"),
        )
    elif "url" in raw and "id" in raw:
        cfg = McpClaudeAiProxyServerConfig(url=raw["url"], id=raw["id"])
    elif "url" in raw:
        cfg = McpRemoteServerConfig(
            url=raw["url"],
            headers=raw.get("headers", {}),
            headers_helper=raw.get("headersHelper"),
        )
    elif "name" in raw:
        cfg = McpSdkServerConfig(name=raw["name"])
    else:
        return None
    return ScopedMcpServerConfig(scope=scope, config=cfg)


def _parse_permission_mode(value: str) -> Optional[ResolvedPermissionMode]:
    """Parse a permission mode string, returning None on unknown values."""
    mapping = {
        "read-only": ResolvedPermissionMode.READ_ONLY,
        "workspace-write": ResolvedPermissionMode.WORKSPACE_WRITE,
        "danger-full-access": ResolvedPermissionMode.DANGER_FULL_ACCESS,
    }
    return mapping.get(value.lower())


def _parse_hooks(raw: dict) -> RuntimeHookConfig:
    """Parse the ``hooks`` sub-object from a settings file."""
    return RuntimeHookConfig(
        pre_tool_use=list(raw.get("PreToolUse", [])),
        post_tool_use=list(raw.get("PostToolUse", [])),
    )


def _parse_oauth(raw: dict) -> OAuthConfig:
    """Parse an OAuth config block."""
    return OAuthConfig(
        client_id=raw["clientId"],
        authorize_url=raw["authorizeUrl"],
        token_url=raw["tokenUrl"],
        callback_port=raw.get("callbackPort", 4545),
        scopes=raw.get("scopes", []),
    )


def _parse_sandbox(raw: dict) -> SandboxConfig:
    """Parse a sandbox config block."""
    return SandboxConfig(
        enabled=raw.get("enabled"),
        namespace_restrictions=raw.get("namespaceRestrictions"),
        network_isolation=raw.get("networkIsolation"),
        filesystem_mode=raw.get("filesystemMode"),
        allowed_mounts=raw.get("allowedMounts", []),
    )


def _build_runtime_config(
    merged: dict[str, Any],
    entries: list[ConfigEntry],
) -> RuntimeConfig:
    """
    Parse feature-specific sections from the merged dict and build
    a ``RuntimeConfig``.
    """
    mcp: dict[str, ScopedMcpServerConfig] = {}
    oauth: Optional[OAuthConfig] = None
    model: Optional[str] = merged.get("model")
    permission_mode: Optional[ResolvedPermissionMode] = None
    sandbox = SandboxConfig()
    hooks = RuntimeHookConfig()

    # Determine which scope contributed each key by replaying entries
    # (last writer wins — same order as merge).
    scope_for_mcp = ConfigSource.USER
    for entry in entries:
        try:
            data = json.loads(entry.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        raw_mcp = data.get("mcpServers", {})
        if raw_mcp and isinstance(raw_mcp, dict):
            scope_for_mcp = entry.source
            for name, server_raw in raw_mcp.items():
                parsed = _parse_mcp_server(server_raw, entry.source)
                if parsed is not None:
                    mcp[name] = parsed

        if "permissionMode" in data:
            pm = _parse_permission_mode(str(data["permissionMode"]))
            if pm is not None:
                permission_mode = pm

        if "hooks" in data and isinstance(data["hooks"], dict):
            hooks = _parse_hooks(data["hooks"])

        if "oauth" in data and isinstance(data["oauth"], dict):
            oauth = _parse_oauth(data["oauth"])

        if "sandbox" in data and isinstance(data["sandbox"], dict):
            sandbox = _parse_sandbox(data["sandbox"])

    return RuntimeConfig(
        _merged=merged,
        _loaded_entries=entries,
        _mcp=mcp,
        _oauth=oauth,
        _model=model,
        _permission_mode=permission_mode,
        _sandbox=sandbox,
        _hooks=hooks,
    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


class ConfigLoader:
    """
    Discovers and merges config files from user, project, and local scopes.

    Config file search order (each merged in order, later overrides earlier):

    - User:    ``~/.claude.json``, ``~/.claude/settings.json``
    - Project: ``.claude.json``, ``.claude/settings.json``  (relative to cwd)
    - Local:   ``.claude/settings.local.json``

    Usage::

        loader = ConfigLoader.default_for(cwd)
        config = loader.load()
    """

    def __init__(self, cwd: pathlib.Path, config_home: pathlib.Path) -> None:
        self.cwd = cwd
        self.config_home = config_home

    @classmethod
    def default_for(cls, cwd: pathlib.Path) -> "ConfigLoader":
        """Create a loader using the default config home directory."""
        default_home = pathlib.Path.home() / ".claude"
        config_home = pathlib.Path(os.environ.get("CLAUDE_CONFIG_HOME", str(default_home)))
        return cls(cwd, config_home)

    def discover(self) -> list[ConfigEntry]:
        """Return an ordered list of existing config files."""
        entries: list[ConfigEntry] = []

        # User scope
        user_home = self.config_home.parent
        for name, base in [
            (".claude.json", user_home),
            ("settings.json", self.config_home),
        ]:
            p = base / name
            if p.exists():
                entries.append(ConfigEntry(ConfigSource.USER, p))

        # Project scope
        for name in [".claude.json", ".claude/settings.json"]:
            p = self.cwd / name
            if p.exists():
                entries.append(ConfigEntry(ConfigSource.PROJECT, p))

        # Local scope (highest precedence)
        local_p = self.cwd / ".claude" / "settings.local.json"
        if local_p.exists():
            entries.append(ConfigEntry(ConfigSource.LOCAL, local_p))

        return entries

    def load(self) -> RuntimeConfig:
        """
        Discover config files, merge them left-to-right (later files override),
        and return a :class:`RuntimeConfig`.
        """
        merged: dict[str, Any] = {}
        entries = self.discover()

        for entry in entries:
            try:
                data = json.loads(entry.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    merged.update(data)
            except (OSError, json.JSONDecodeError):
                pass  # Skip unreadable / malformed files

        return _build_runtime_config(merged, entries)
