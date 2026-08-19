"""Run the Phase 2 independent retry baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "results"
sys.path.insert(0, str(PROJECT_ROOT))

from leanproof.model import ConfigurationError, OpenAICompatibleProofModel
from leanproof.model_registry import ModelRegistry
from leanproof.one_shot import DatasetError, load_dataset
from leanproof.retry import DEFAULT_MAX_ATTEMPTS, default_retry_output_path, run_retry
from leanproof.verifier import LeanVerifier


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run independent proof generations per theorem")
    parser.add_argument("--dataset", help="Path to a JSONL theorem dataset")
    parser.add_argument("--model", help="Registered model alias for this experiment")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List configured model aliases and exit",
    )
    parser.add_argument("--max-attempts", type=_positive_integer, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--limit", type=_positive_integer, help="Run only the first N theorems")
    parser.add_argument("--output", help="Output JSONL path; must not already exist")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-attempt progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not args.list_models:
        if not args.dataset:
            parser.error("--dataset is required unless --list-models is used")
        if not args.model:
            parser.error("--model is required unless --list-models is used")

    try:
        registry = ModelRegistry.from_env(PROJECT_ROOT / ".env")
        if args.list_models:
            _print_registered_models(registry)
            return 0
        config = registry.get(args.model)
    except ConfigurationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    dataset_path = Path(args.dataset)
    output_path = (
        Path(args.output)
        if args.output
        else default_retry_output_path(
            dataset_path,
            args.model,
            args.max_attempts,
            DEFAULT_RESULTS_DIRECTORY,
        )
    )
    try:
        tasks = load_dataset(dataset_path, limit=args.limit)
        model = OpenAICompatibleProofModel(config)
        verifier = LeanVerifier(project_root=PROJECT_ROOT)
        summary = run_retry(
            tasks,
            model,
            verifier,
            output_path,
            model_alias=args.model,
            max_attempts=args.max_attempts,
            progress_callback=_print_progress if args.verbose else None,
        )
    except (ConfigurationError, DatasetError, FileExistsError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print("Strategy: retry")
    print(f"Model alias: {args.model}")
    print(f"Provider model: {model.model_name}")
    print(f"Dataset: {dataset_path.as_posix()}")
    print(f"Generation budget: {summary.generation_budget}")
    print(f"Solved: {summary.solved} / {summary.total}")
    print(f"Success rate: {summary.success_rate:.1f}%")
    print(f"Average attempts / theorem: {summary.average_attempts_per_theorem:.2f}")
    print(
        "Average attempts / solved theorem: "
        + _format_optional(summary.average_attempts_per_solved_theorem)
    )
    print(f"Total prompt tokens (available): {_format_optional(summary.total_prompt_tokens)}")
    print(
        "Total completion tokens (available): " + _format_optional(summary.total_completion_tokens)
    )
    print(
        f"Average prompt tokens / available attempt: {_format_optional(summary.average_prompt_tokens)}"
    )
    print(
        "Average completion tokens / available attempt: "
        + _format_optional(summary.average_completion_tokens)
    )
    print(f"Average generation latency / attempt: {summary.average_generation_latency_ms:.1f} ms")
    print(
        f"Average verification latency / verifier call: "
        f"{summary.average_verification_latency_ms:.1f} ms"
    )
    print(f"Results: {summary.output_path.as_posix()}")
    return 0


def _positive_integer(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed_value


def _format_optional(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _print_registered_models(registry: ModelRegistry) -> None:
    print("Registered models:")
    names = registry.names()
    if not names:
        print("- none")
        return
    for name in names:
        print(f"- {name}")


if __name__ == "__main__":
    raise SystemExit(main())
