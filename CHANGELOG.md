# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Phase 2: Multi-provider API Client
- `api/base.py`: abstract `BaseClient` interface (`send_message`, `stream_message`, `aclose`)
- `api/anthropic.py`: native Anthropic client, inherits `BaseClient`, retries with exponential backoff
- `api/openai_compat.py`: OpenAI-compatible client covering OpenAI, Gemini (via compat endpoint), and Ollama
- `api/factory.py`: `create_client(provider)` factory — resolves keys from environment automatically
- `api/client.py`: backward-compat shim re-exporting `AnthropicClient`
- `tests/conftest.py`: auto-loads `.env` before test runs via `python-dotenv`
- 97 unit tests: error hierarchy, type serialization, SSE parsing (CRLF/LF), retry logic
- Integration tests with real provider keys (auto-skip when key absent)

### Integration Test Status
| Provider | Model | send_message | stream_message |
|---|---|---|---|
| OpenAI | gpt-4o-mini | ✅ | ✅ |
| Gemini | gemini-2.5-flash | ✅ | ✅ |
| Ollama | local (auto-detected) | ✅ | — |
| Anthropic | claude-haiku-4-5 | ⏳ on hold (no key) | ⏳ on hold |

> **Note — Gemini free-tier rate limit:** Gemini 2.5 Flash free tier allows 5 requests/minute.
> Running all 4 test variants (asyncio + trio) back-to-back trips this cap.
> Use `-k "gemini and asyncio"` for regular CI, or add a 20s delay between backends.

---

## [0.1.0] — Phase 1: Initial Scaffold
### Added
- Initial project scaffold: `helix_agent` package with `api`, `cli`, `commands`, `runtime`, `tools` sub-packages
- API client (`helix_agent/api/client.py`) with SSE streaming support
- Runtime session, config, permissions, and usage tracking modules
- CLI entry point via `helix-agent` command
- `pyproject.toml` with full dev toolchain (black, ruff, mypy, pytest)
- `README.md`, `docs/architecture.md`, `tests/` directory structure
