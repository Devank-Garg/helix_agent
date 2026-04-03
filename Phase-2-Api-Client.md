# Phase 2: API Client (Foundation for All)

## Project Summary
- Project: Python replication of Rust "HelixAgent".
- Aim: API correctness with Anthropic/Claude and robust connection handling.

## Quick Reference: Phase Overview
| Phase | Focus | Estimated Time | Blocker |
|-------|-------|----------------|---------|
| 1 | Setup | 1-2h | None |
| 2 | API Client | 8-12h | Phase 1 |
| 3 | Runtime Core | 6-8h | Phase 1 |
| 4 | File Ops & Utils | 6-8h | Phase 1 (parallel with 3) |
| 5 | Advanced Runtime | 12-16h | Phases 2,3,4 |
| 6 | Tools & Commands | 6-8h | Phase 5 |
| 7 | CLI | 8-10h | Phases 2,3,6 |
| 8 | Testing & Docs | 8-10h | All prior |

## Additional Overall Info
- Phase 2 enables all later runtime, tools, MCP, and CLI behavior.
- Must include `api/error.py`, `api/types.py`, `api/sse.py`, `api/client.py`, and `api/__init__.py`.

## Phase 2 Goals
1. Implement error hierarchy and retry classification.
2. Implement API data models (message/request/response; streaming events).
3. Create SSE parsing (frame-level + incremental).
4. Build client for HTTP + streaming + OAuth + retries.

## Modules & Responsibilities
- `api/error.py`: `ApiError` base and specializations, `is_retryable_status`.
- `api/types.py`: typed dataclasses for requests, responses, events.
- `api/sse.py`: SSE frame parsing and `SseParser`.
- `api/client.py`: `AnthropicClient` with methods `send_message`, `stream_message`, `exchange_oauth_code`, `refresh_oauth_token`.
- `api/__init__.py`: public exports.

## Verification Checklist
- [ ] errors classified and chained
- [ ] dataclasses to_dict/from parser checks
- [ ] streaming parser handles CRLF/LF and [DONE]/ping
- [ ] retry logic behavior with mocked transient error
