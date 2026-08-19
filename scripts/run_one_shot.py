"""Run the Phase 1 one-shot full-proof baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "artifacts" / "one_shot" / "results"
sys.path.insert(0, str(PROJECT_ROOT))

from leanproof.config import build_model_registry, load_config, resolve_experiment_config
from leanproof.lean import LeanVerifier
from leanproof.models import ConfigurationError, OpenAICompatibleProofModel
from leanproof.strategies import DatasetError, default_output_path, load_dataset, run_one_shot
from scripts._common import (
    add_runtime_arguments,
    print_progress,
    print_registered_models,
    resolve_project_path,
    safe_result_path,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one proof generation per Lean theorem")
    add_runtime_arguments(parser, project_root=PROJECT_ROOT, include_retry=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        runtime_config = load_config(args.config)
        registry = build_model_registry(runtime_config, dotenv_path=PROJECT_ROOT / ".env")
        if args.list_models:
            print_registered_models(registry)
            return 0
        resolved = resolve_experiment_config(
            runtime_config,
            workflow="run_one_shot",
            dataset_path=args.dataset,
            model_alias=args.model,
            limit=args.limit,
            verbose=args.verbose,
            generation_timeout_seconds=args.generation_timeout_seconds,
            verification_timeout_seconds=args.verification_timeout_seconds,
        )
        registry = build_model_registry(
            runtime_config,
            dotenv_path=PROJECT_ROOT / ".env",
            required_alias=resolved.model_alias,
            generation_timeout_seconds=resolved.generation_timeout_seconds,
        )
        config = registry.get(resolved.model_alias)
    except ConfigurationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    dataset_path = resolve_project_path(PROJECT_ROOT, resolved.dataset_path)
    results_directory = (
        resolve_project_path(PROJECT_ROOT, resolved.artifact_root) / "one_shot" / "results"
    )
    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(dataset_path, resolved.model_alias, results_directory)
    )

    try:
        tasks = load_dataset(dataset_path, limit=resolved.limit)
        model = OpenAICompatibleProofModel(config)
        verifier = LeanVerifier(
            project_root=PROJECT_ROOT,
            timeout_seconds=resolved.verification_timeout_seconds,
        )
        summary = run_one_shot(
            tasks,
            model,
            verifier,
            output_path,
            model_alias=resolved.model_alias,
            dataset=safe_result_path(PROJECT_ROOT, dataset_path),
            generation_timeout_seconds=resolved.generation_timeout_seconds,
            verification_timeout_seconds=resolved.verification_timeout_seconds,
            progress_callback=print_progress if resolved.verbose else None,
        )
    except (ConfigurationError, DatasetError, FileExistsError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Model alias: {resolved.model_alias}")
    print(f"Provider model: {model.model_name}")
    print(f"Dataset: {dataset_path.as_posix()}")
    print(f"Generation timeout: {resolved.generation_timeout_seconds:g} s")
    print(f"Verification timeout: {resolved.verification_timeout_seconds:g} s")
    print(f"Solved: {summary.solved} / {summary.total}")
    print(f"Success rate: {summary.success_rate:.1f}%")
    print(f"Average generation latency: {summary.average_generation_latency_ms:.1f} ms")
    print(f"Average verification latency: {summary.average_verification_latency_ms:.1f} ms")
    print(f"Results: {summary.output_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
