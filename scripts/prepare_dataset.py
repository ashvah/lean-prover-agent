"""Prepare one manually downloaded raw dataset into canonical JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from leanproof.config import DEFAULT_CONFIG_PATH, load_config
from leanproof.datasets import DataPaths, DatasetPathError, DatasetPipelineError, prepare_dataset
from leanproof.datasets.adapters import LeanWorkbookSchemaError
from leanproof.models import ConfigurationError
from scripts._common import positive_integer, resolve_project_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a local raw theorem dataset")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / DEFAULT_CONFIG_PATH),
        help="Runtime TOML path",
    )
    parser.add_argument("--source", choices=("lean_workbook",), help="Override source adapter")
    parser.add_argument("--source-file", help="Override the configured raw filename")
    parser.add_argument("--input", help="Explicit raw input path override")
    parser.add_argument("--output", help="Explicit canonical JSONL path override")
    parser.add_argument("--manifest", help="Explicit manifest path override")
    parser.add_argument(
        "--limit",
        type=positive_integer,
        help="Process only the first N theorem groups after complete source traversal",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        runtime = load_config(args.config)
        workflow = runtime.prepare_dataset
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
            else data_paths.raw_dataset_path(source, source_file)
        )
        output_path = (
            resolve_project_path(PROJECT_ROOT, args.output)
            if args.output
            else data_paths.processed_dataset_path(source, source_file)
        )
        manifest_path = (
            resolve_project_path(PROJECT_ROOT, args.manifest)
            if args.manifest
            else data_paths.dataset_manifest_path(source, source_file)
        )
        summary = prepare_dataset(
            source=source,
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            limit=args.limit if args.limit is not None else workflow.limit,
        )
    except (
        ConfigurationError,
        DatasetPathError,
        DatasetPipelineError,
        LeanWorkbookSchemaError,
        OSError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Source: {summary.source}")
    print(f"Source tactic rows scanned: {summary.source_tactic_rows_scanned}")
    print(f"Raw tactic rows selected: {summary.raw_tactic_rows}")
    print(f"Theorem groups: {summary.theorem_groups}")
    print(f"Reference trajectory steps: {summary.total_trajectory_steps}")
    print(f"Proved theorems: {summary.proved_theorems}")
    print(f"Disproved theorems: {summary.disproved_theorems}")
    print(f"Invalid groups: {summary.invalid_groups}")
    print(f"Duplicate theorems removed: {summary.duplicate_theorems}")
    print(f"Final proving theorems: {summary.final_proving_theorems}")
    print("Difficulty:")
    for bucket in ("easy", "medium", "hard"):
        print(f"- {bucket}: {summary.bucket_counts[bucket]}")
    print(f"Output: {summary.output_path.as_posix()}")
    print(f"Manifest: {summary.manifest_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
