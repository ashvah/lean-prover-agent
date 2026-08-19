"""Shared presentation helpers for experiment command-line entrypoints."""

from __future__ import annotations

import argparse

from leanproof.models import ModelRegistry


def positive_integer(value: str) -> int:
    """Parse one strictly positive command-line integer."""

    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed_value


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
