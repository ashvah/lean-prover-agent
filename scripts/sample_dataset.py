"""Create a deterministic canonical development subset."""

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
    sample_canonical_records,
    write_canonical_records,
)
from leanproof.models import ConfigurationError
from scripts._common import positive_integer, resolve_project_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a canonical theorem dataset")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / DEFAULT_CONFIG_PATH),
        help="Runtime TOML path",
    )
    parser.add_argument("--source", choices=("lean_workbook",), help="Override source adapter")
    parser.add_argument("--source-file", help="Override the configured processed filename")
    parser.add_argument("--split", help="Override the configured split name")
    parser.add_argument("--input", help="Explicit processed canonical JSONL path override")
    parser.add_argument("--bucket", choices=("easy", "medium", "hard", "all"))
    parser.add_argument("--size", type=positive_integer)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", help="Explicit sample output JSONL path override")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        runtime = load_config(args.config)
        workflow = runtime.sample_dataset
        source = args.source or workflow.source
        source_file = args.source_file or workflow.source_file
        split = args.split or workflow.split
        bucket = args.bucket or workflow.bucket
        size = args.size if args.size is not None else workflow.size
        seed = args.seed if args.seed is not None else workflow.seed
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
        output_path = (
            resolve_project_path(PROJECT_ROOT, args.output)
            if args.output
            else data_paths.development_split_path(
                source_file=source_file,
                split=split,
                bucket=bucket,
                size=size,
                seed=seed,
            )
        )
        records = load_canonical_records(input_path)
        sampled = sample_canonical_records(records, bucket=bucket, size=size, seed=seed)
        write_canonical_records(output_path, sampled)
    except (
        ConfigurationError,
        DatasetPathError,
        DatasetPipelineError,
        OSError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Input: {input_path.as_posix()}")
    print(f"Bucket: {bucket}")
    print(f"Size: {len(sampled)}")
    print(f"Seed: {seed}")
    print(f"Output: {output_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
