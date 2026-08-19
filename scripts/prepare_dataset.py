"""Prepare one manually downloaded raw dataset into canonical JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from leanproof.datasets import DatasetPipelineError, prepare_dataset
from leanproof.datasets.adapters import LeanWorkbookSchemaError
from scripts._common import positive_integer


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a local raw theorem dataset")
    parser.add_argument("--source", required=True, choices=("lean_workbook",))
    parser.add_argument("--input", required=True, help="Local Lean-Workbook Parquet path")
    parser.add_argument("--output", required=True, help="Canonical output JSONL path")
    parser.add_argument("--manifest", help="Manifest path; defaults beside the output JSONL")
    parser.add_argument("--limit", type=positive_integer, help="Process only the first N raw rows")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_path.parent / "manifest.json"
    try:
        summary = prepare_dataset(
            source=args.source,
            input_path=args.input,
            output_path=output_path,
            manifest_path=manifest_path,
            limit=args.limit,
        )
    except (DatasetPipelineError, LeanWorkbookSchemaError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Source: {summary.source}")
    print(f"Raw records: {summary.raw_records}")
    print(f"Valid mapped records: {summary.mapped_records}")
    print(f"Invalid records dropped: {summary.invalid_records}")
    print(f"Duplicates removed: {summary.duplicates_removed}")
    print(f"Final records: {summary.final_records}")
    print("Difficulty:")
    for bucket in ("easy", "medium", "hard"):
        print(f"- {bucket}: {summary.bucket_counts[bucket]}")
    print(f"Output: {summary.output_path.as_posix()}")
    print(f"Manifest: {summary.manifest_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
