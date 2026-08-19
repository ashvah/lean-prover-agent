"""Shared presentation helpers for experiment command-line entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from leanproof.config import DEFAULT_CONFIG_PATH
from leanproof.models import ModelRegistry


def positive_integer(value: str) -> int:
    """Parse one strictly positive command-line integer."""

    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed_value


def positive_number(value: str) -> float:
    """Parse one strictly positive command-line number."""

    parsed_value = float(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed_value


def non_negative_integer(value: str) -> int:
    """Parse one command-line integer that may be zero."""

    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed_value


def add_runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    project_root: Path,
    include_retry: bool,
) -> None:
    """Add stable TOML/CLI override arguments shared by experiment runners."""

    parser.add_argument(
        "--config",
        default=str(project_root / DEFAULT_CONFIG_PATH),
        help="Runtime TOML path",
    )
    parser.add_argument("--dataset", help="Override the configured JSONL theorem dataset")
    parser.add_argument("--model", help="Override the configured model alias")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List configured model aliases with available API-key secrets and exit",
    )
    parser.add_argument("--limit", type=positive_integer, help="Override the configured limit")
    parser.add_argument(
        "--max-transport-retries",
        type=non_negative_integer,
        help="Override additional provider requests allowed after transport failures",
    )
    if include_retry:
        parser.add_argument(
            "--max-attempts",
            type=positive_integer,
            help="Override the configured independent-generation budget",
        )
    parser.add_argument(
        "--generation-timeout-seconds",
        type=positive_number,
        help="Override the selected model API generation timeout",
    )
    parser.add_argument(
        "--verification-timeout-seconds",
        type=positive_number,
        help="Override the Lean execution timeout",
    )
    parser.add_argument("--output", help="Output JSONL path; must not already exist")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=None,
        help="Override configured progress reporting",
    )


def resolve_project_path(project_root: Path, configured_path: str) -> Path:
    """Resolve one repository-relative runtime path without changing absolute paths."""

    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path


def safe_result_path(project_root: Path, path: Path) -> str:
    """Return reproducible repository-relative metadata without an absolute machine path."""

    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def print_progress(message: str) -> None:
    """Print one progress event immediately."""

    print(message, flush=True)


def print_registered_models(registry: ModelRegistry) -> None:
    """Print configured model aliases without provider credentials."""

    print("Registered models:")
    names = registry.names()
    if not names:
        print("- none")
        return
    for name in names:
        print(f"- {name}")
