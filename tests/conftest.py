"""Global pytest configuration — loads .env before any test runs."""

from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (one level above tests/)
load_dotenv(Path(__file__).parent.parent / ".env")
