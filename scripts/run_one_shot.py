"""Run the Phase 1 one-shot full-proof baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "results"
sys.path.insert(0, str(PROJECT_ROOT))

from leanproof.model import ConfigurationError, LLMConfig, OpenAICompatibleProofModel
from leanproof.one_shot import DatasetError, default_output_path, load_dataset, run_one_shot
from leanproof.verifier import LeanVerifier


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one proof generation per Lean theorem")
    parser.add_argument("--dataset", required=True, help="Path to a JSONL theorem dataset")
    parser.add_argument("--limit", type=_positive_integer, help="Run only the first N theorems")
    parser.add_argument("--output", help="Output JSONL path; must not already exist")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-theorem progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    dataset_path = Path(args.dataset)
    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(dataset_path, DEFAULT_RESULTS_DIRECTORY)
    )

    try:
        tasks = load_dataset(dataset_path, limit=args.limit)
        config = LLMConfig.from_env(PROJECT_ROOT / ".env")
        model = OpenAICompatibleProofModel(config)
        verifier = LeanVerifier(project_root=PROJECT_ROOT)
        summary = run_one_shot(
            tasks,
            model,
            verifier,
            output_path,
            progress_callback=_print_progress if args.verbose else None,
        )
    except (ConfigurationError, DatasetError, FileExistsError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Model: {model.model_name}")
    print(f"Dataset: {dataset_path.as_posix()}")
    print(f"Solved: {summary.solved} / {summary.total}")
    print(f"Success rate: {summary.success_rate:.1f}%")
    print(f"Average generation latency: {summary.average_generation_latency_ms:.1f} ms")
    print(f"Average verification latency: {summary.average_verification_latency_ms:.1f} ms")
    print(f"Results: {summary.output_path.as_posix()}")
    return 0


def _positive_integer(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed_value


def _print_progress(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
