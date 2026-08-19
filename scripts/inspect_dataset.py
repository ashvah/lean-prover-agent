"""Print lightweight statistics for a canonical theorem dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from leanproof.datasets import (
    DatasetPipelineError,
    load_canonical_records,
    summarize_canonical_records,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a canonical theorem dataset")
    parser.add_argument("dataset", help="Processed canonical JSONL path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        summary = summarize_canonical_records(load_canonical_records(args.dataset))
    except (DatasetPipelineError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Source: {summary['source']}")
    print(f"Records: {summary['records']}")
    print("Difficulty buckets:")
    for bucket, count in summary["difficulty_buckets"].items():
        print(f"- {bucket}: {count}")
    for feature in ("statement_tokens", "num_binders", "num_hypotheses"):
        values = summary[feature]
        print(f"{feature}:")
        print(
            f"- min {values['min']} | mean {values['mean']} | median {values['median']} | "
            f"p90 {values['p90']} | p95 {values['p95']} | max {values['max']}"
        )
    print(
        f"Reference proofs: {summary['reference_proofs']} "
        f"({summary['reference_proof_percentage']:.1f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
