# Architecture

## Overview

Helix Agent is structured as a layered Python package:

```
CLI  →  Runtime/Session  →  API Client  →  Anthropic API
              ↕
         Tools / Commands
```

## Packages

| Package | Responsibility |
|---------|----------------|
| `api` | Raw HTTP + SSE transport, request/response types, error handling |
| `runtime` | Session lifecycle, configuration, permission model, usage accounting |
| `cli` | User-facing entry point (Click) |
| `commands` | Built-in slash commands (e.g. `/help`, `/clear`) |
| `tools` | Tool definitions sent to the model and their local executors |

## Key Design Decisions

- **Async-first**: all I/O uses `anyio` so the runtime works under both `asyncio` and `trio`.
- **Streaming**: responses are consumed via SSE (`text/event-stream`) for low-latency output.
- **Strict typing**: `mypy --strict` is enforced across the entire package.
