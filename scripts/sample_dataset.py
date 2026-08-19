"""Create a deterministic canonical development subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from leanproof.datasets import (
    DatasetPipelineError,
    load_canonical_records,
    sample_canonical_records,
    write_canonical_records,
)
from scripts._common import positive_integer


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a canonical theorem dataset")
    parser.add_argument("--input", required=True, help="Processed canonical JSONL path")
    parser.add_argument("--bucket", required=True, choices=("easy", "medium", "hard", "all"))
    parser.add_argument("--size", required=True, type=positive_integer)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, help="Sample output JSONL path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        records = load_canonical_records(args.input)
        sampled = sample_canonical_records(
            records, bucket=args.bucket, size=args.size, seed=args.seed
        )
        write_canonical_records(args.output, sampled)
    except (DatasetPipelineError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Input: {Path(args.input).as_posix()}")
    print(f"Bucket: {args.bucket}")
    print(f"Size: {len(sampled)}")
    print(f"Seed: {args.seed}")
    print(f"Output: {Path(args.output).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
