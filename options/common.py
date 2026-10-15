"""Shared command-line option helpers."""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_bool(value):
    """Parse common textual boolean values accepted by the old CLI."""
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def repo_path(path):
    """Resolve repository-owned defaults while preserving absolute paths."""
    if not path:
        return path
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate.resolve())

