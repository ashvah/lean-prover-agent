"""Print lightweight statistics for a canonical theorem dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from leanproof.config import DEFAULT_CONFIG_PATH, load_config
from leanproof.datasets import (
    DataPaths,
    DatasetPathError,
    DatasetPipelineError,
    load_canonical_records,
    summarize_canonical_records,
)
from leanproof.models import ConfigurationError
from scripts._common import resolve_project_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a canonical theorem dataset")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / DEFAULT_CONFIG_PATH),
        help="Runtime TOML path",
    )
    parser.add_argument("--source", choices=("lean_workbook",), help="Override source adapter")
    parser.add_argument("--source-file", help="Override the configured processed filename")
    parser.add_argument("--input", help="Explicit processed canonical JSONL path override")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        runtime = load_config(args.config)
        workflow = runtime.inspect_dataset
        source = args.source or workflow.source
        source_file = args.source_file or workflow.source_file
        data_paths = DataPaths.from_configured_roots(
            PROJECT_ROOT,
            raw_data=runtime.paths.raw_data,
            processed_data=runtime.paths.processed_data,
            splits=runtime.paths.splits,
        )
        input_path = (
            resolve_project_path(PROJECT_ROOT, args.input)
            if args.input
            else data_paths.processed_input_path(source, source_file)
        )
        summary = summarize_canonical_records(load_canonical_records(input_path))
    except (
        ConfigurationError,
        DatasetPathError,
        DatasetPipelineError,
        OSError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Source: {summary['source']}")
    print(f"Records: {summary['records']}")
    print("Difficulty buckets:")
    for bucket, count in summary["difficulty_buckets"].items():
        print(f"- {bucket}: {count}")
    print("Source statuses:")
    for status, count in summary["source_statuses"].items():
        print(f"- {status}: {count}")
    for feature in ("statement_tokens", "num_binders", "num_hypotheses"):
        values = summary[feature]
        print(f"{feature}:")
        print(
            f"- min {values['min']} | mean {values['mean']} | median {values['median']} | "
            f"p90 {values['p90']} | p95 {values['p95']} | max {values['max']}"
        )
    print(
        f"Reference trajectories: {summary['reference_trajectories']} "
        f"({summary['reference_trajectory_percentage']:.1f}%)"
    )
    tactic_counts = summary["reference_tactic_count"]
    print(
        "reference_tactic_count:\n"
        f"- min {tactic_counts['min']} | mean {tactic_counts['mean']} | "
        f"median {tactic_counts['median']} | p90 {tactic_counts['p90']} | "
        f"p95 {tactic_counts['p95']} | max {tactic_counts['max']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
