# Helix Agent

A Python replication of the HelixAgent AI coding assistant.

## Requirements

- Python 3.11+

## Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Usage

```bash
helix-agent --help
```

## Development

```bash
# Lint
ruff check helix_agent tests

# Format
black helix_agent tests

# Type-check
mypy helix_agent

# Run tests
pytest
```

## Project Structure

```
helix_agent/
├── api/          # HTTP client, SSE streaming, error types, API type models
├── cli/          # Click-based entry point
├── commands/     # Slash-command handlers
├── runtime/      # Session management, config, permissions, usage tracking
└── tools/        # Tool definitions and execution
tests/            # pytest test suite
docs/             # Additional documentation
```

## License

MIT
