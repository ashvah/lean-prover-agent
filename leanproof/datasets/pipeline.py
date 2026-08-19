"""Canonical dataset preparation, serialization, sampling, and inspection."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from leanproof.datasets.adapters import LeanWorkbookAdapter, RowMappingError
from leanproof.datasets.difficulty import DIFFICULTY_METHOD, assign_static_difficulty
from leanproof.datasets.features import (
    DIFFICULTY_FEATURE_NAMES,
    FEATURE_NAMES,
    extract_features,
)
from leanproof.datasets.schema import SCHEMA_VERSION, CanonicalTheorem

PIPELINE_VERSION = "dataset_v1"
SUPPORTED_SOURCE = "lean_workbook"
BUCKETS = ("easy", "medium", "hard")


class DatasetPipelineError(ValueError):
    """Raised for invalid global input or canonical dataset operations."""


@dataclass(frozen=True)
class PreparationSummary:
    """Actual record counts and output locations from one preparation run."""

    source: str
    raw_records: int
    mapped_records: int
    invalid_records: int
    duplicates_removed: int
    final_records: int
    bucket_counts: dict[str, int]
    invalid_reasons: dict[str, int]
    output_path: Path
    manifest_path: Path


def prepare_dataset(
    *,
    source: str,
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    limit: int | None = None,
) -> PreparationSummary:
    """Run Dataset Pipeline v1 from one manually downloaded local source file."""

    if source != SUPPORTED_SOURCE:
        raise DatasetPipelineError(
            f"Unsupported dataset source '{source}'. Available sources: {SUPPORTED_SOURCE}"
        )
    if limit is not None and limit <= 0:
        raise DatasetPipelineError("limit must be greater than zero")
    input_file = Path(input_path)
    output_file = Path(output_path)
    manifest_file = Path(manifest_path)
    if output_file.resolve() == manifest_file.resolve():
        raise DatasetPipelineError("Dataset output and manifest paths must be distinct")

    adapter = LeanWorkbookAdapter(input_file)
    raw_records = 0
    mapped_records = 0
    invalid_reasons: Counter[str] = Counter()
    duplicates_removed = 0
    seen_statement_keys: set[str] = set()
    seen_ids: dict[str, str] = {}
    canonical_records: list[CanonicalTheorem] = []
    for raw_row in adapter.iter_rows(limit=limit):
        raw_records += 1
        try:
            theorem = adapter.map_row(raw_row)
        except RowMappingError as error:
            invalid_reasons[str(error)] += 1
            continue
        mapped_records += 1
        previous_statement = seen_ids.get(theorem.id)
        if previous_statement is not None and previous_statement != theorem.statement:
            raise DatasetPipelineError(
                f"Canonical ID collision for source identity '{theorem.source_id}'"
            )
        statement_key = _statement_dedup_key(theorem.statement)
        if statement_key in seen_statement_keys:
            duplicates_removed += 1
            continue
        if not _is_json_serializable(theorem.to_dict()):
            invalid_reasons["record_not_json_serializable"] += 1
            mapped_records -= 1
            continue
        seen_ids[theorem.id] = theorem.statement
        seen_statement_keys.add(statement_key)
        canonical_records.append(replace(theorem, features=extract_features(theorem)))

    if not canonical_records:
        raise DatasetPipelineError("Dataset preparation produced no valid canonical records")
    scored_records = assign_static_difficulty(canonical_records)
    bucket_counts = _bucket_counts(record.to_dict() for record in scored_records)
    _write_jsonl(output_file, [record.to_dict() for record in scored_records])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "source": source,
        "raw_input": input_file.name,
        "raw_records": raw_records,
        "mapped_records": mapped_records,
        "invalid_records": sum(invalid_reasons.values()),
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "duplicates_removed": duplicates_removed,
        "final_records": len(scored_records),
        "difficulty_method": DIFFICULTY_METHOD,
        "difficulty_buckets": bucket_counts,
        "feature_names": list(FEATURE_NAMES),
        "lean_validation": {"performed": False},
        "deterministic_configuration": {
            "limit": limit,
            "source_order": "parquet_row_order",
            "deduplication": "sha256_of_line_normalized_stripped_collapsed_whitespace",
            "duplicate_survivor": "first_source_occurrence",
            "difficulty_features": list(DIFFICULTY_FEATURE_NAMES),
            "difficulty_formula": "equal_weight_mean_of_midrank_percentiles",
            "bucket_thresholds": {"easy_max_exclusive": 1 / 3, "hard_min_inclusive": 2 / 3},
        },
    }
    _write_json(manifest_file, manifest)
    return PreparationSummary(
        source=source,
        raw_records=raw_records,
        mapped_records=mapped_records,
        invalid_records=sum(invalid_reasons.values()),
        duplicates_removed=duplicates_removed,
        final_records=len(scored_records),
        bucket_counts=bucket_counts,
        invalid_reasons=dict(sorted(invalid_reasons.items())),
        output_path=output_file,
        manifest_path=manifest_file,
    )


def load_canonical_records(dataset_path: str | Path) -> list[dict[str, object]]:
    """Load and minimally validate canonical one-record-per-line JSONL."""

    path = Path(dataset_path)
    if not path.is_file():
        raise DatasetPipelineError(f"Canonical dataset does not exist: {path}")
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise DatasetPipelineError(f"Blank JSONL record at {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetPipelineError(
                    f"Invalid JSON at {path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise DatasetPipelineError(
                    f"Canonical record must be an object at {path}:{line_number}"
                )
            _validate_canonical_record(record, path, line_number)
            records.append(record)
    if not records:
        raise DatasetPipelineError(f"Canonical dataset is empty: {path}")
    return records


def sample_canonical_records(
    records: Sequence[dict[str, object]], *, bucket: str, size: int, seed: int
) -> list[dict[str, object]]:
    """Select without replacement and preserve source order in the output."""

    if bucket not in {*BUCKETS, "all"}:
        raise DatasetPipelineError("bucket must be one of: easy, medium, hard, all")
    if size <= 0:
        raise DatasetPipelineError("size must be greater than zero")
    eligible = [record for record in records if bucket == "all" or _record_bucket(record) == bucket]
    if size > len(eligible):
        raise DatasetPipelineError(
            f"Requested sample size {size} exceeds {len(eligible)} available '{bucket}' records"
        )
    selected_indices = sorted(random.Random(seed).sample(range(len(eligible)), size))
    return [eligible[index] for index in selected_indices]


def write_canonical_records(output_path: str | Path, records: Sequence[dict[str, object]]) -> None:
    """Write complete canonical records without stripping source metadata."""

    _write_jsonl(Path(output_path), records)


def summarize_canonical_records(records: Sequence[dict[str, object]]) -> dict[str, object]:
    """Compute lightweight console inspection statistics from stored features."""

    if not records:
        raise DatasetPipelineError("Cannot inspect an empty canonical dataset")
    sources = sorted({str(record.get("source", "unknown")) for record in records})
    reference_proofs = sum(
        isinstance(record.get("reference_proof"), str) and bool(record["reference_proof"])
        for record in records
    )
    return {
        "source": ", ".join(sources),
        "records": len(records),
        "difficulty_buckets": _bucket_counts(records),
        "statement_tokens": _numeric_summary(_feature_values(records, "statement_tokens")),
        "num_binders": _numeric_summary(_feature_values(records, "num_binders")),
        "num_hypotheses": _numeric_summary(_feature_values(records, "num_hypotheses")),
        "reference_proofs": reference_proofs,
        "reference_proof_percentage": 100.0 * reference_proofs / len(records),
    }


def _statement_dedup_key(statement: str) -> str:
    normalized = re.sub(r"\s+", " ", statement.replace("\r\n", "\n").replace("\r", "\n").strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _is_json_serializable(record: dict[str, object]) -> bool:
    try:
        json.dumps(record, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _write_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            )
            output.write("\n")
    temporary_path.replace(path)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, allow_nan=False, indent=2)
        output.write("\n")
    temporary_path.replace(path)


def _validate_canonical_record(record: dict[str, object], path: Path, line_number: int) -> None:
    for field_name in ("id", "source", "statement"):
        value = record.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise DatasetPipelineError(f"Missing {field_name} at {path}:{line_number}")
    if not isinstance(record.get("features"), dict):
        raise DatasetPipelineError(f"Missing features at {path}:{line_number}")
    difficulty = record.get("difficulty")
    if not isinstance(difficulty, dict) or difficulty.get("bucket") not in BUCKETS:
        raise DatasetPipelineError(f"Missing difficulty bucket at {path}:{line_number}")


def _record_bucket(record: dict[str, object]) -> str:
    difficulty = record.get("difficulty")
    return str(difficulty.get("bucket")) if isinstance(difficulty, dict) else ""


def _bucket_counts(
    records: Iterable[dict[str, object] | CanonicalTheorem],
) -> dict[str, int]:
    counts = {bucket: 0 for bucket in BUCKETS}
    for record in records:
        if isinstance(record, CanonicalTheorem):
            bucket = record.difficulty.bucket if record.difficulty is not None else ""
        elif isinstance(record, dict):
            bucket = _record_bucket(record)
        else:
            continue
        if bucket in counts:
            counts[bucket] += 1
    return counts


def _feature_values(records: Sequence[dict[str, object]], name: str) -> list[int]:
    values: list[int] = []
    for record in records:
        features = record.get("features")
        value = features.get(name) if isinstance(features, dict) else None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise DatasetPipelineError(f"Invalid feature '{name}' in record {record.get('id')}")
        values.append(value)
    return values


def _numeric_summary(values: Sequence[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "mean": round(statistics.fmean(ordered), 3),
        "median": statistics.median(ordered),
        "p90": ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)],
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
    }
