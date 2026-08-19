"""Shared dataset and experiment-accounting primitives for proof strategies."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from leanproof.lean import LeanResult
from leanproof.models import (
    ErrorDetails,
    GenerationRequestError,
    GenerationResult,
    ProofModel,
)
from leanproof.prompts import ReasoningMode

DEFAULT_MAX_TRANSPORT_RETRIES = 2


class DatasetError(ValueError):
    """Raised when an experiment JSONL dataset is missing or malformed."""


class ProofVerifier(Protocol):
    """Minimal verifier interface required by full-proof strategies."""

    def verify(self, statement: str, proof: str) -> LeanResult:
        """Verify one complete proof and return raw Lean diagnostics."""


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class TaskDifficulty:
    """Dataset-relative difficulty snapshot carried without entering model input."""

    score: float
    bucket: str
    method: str

    def to_dict(self) -> dict[str, object]:
        return {"score": self.score, "bucket": self.bucket, "method": self.method}


@dataclass(frozen=True)
class TaskMetadata:
    """Small answer-free analysis snapshot associated with one theorem task."""

    source: str | None = None
    source_id: str | None = None
    difficulty: TaskDifficulty | None = None
    reference_tactic_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        metadata: dict[str, object] = {}
        if self.source is not None:
            metadata["source"] = self.source
        if self.source_id is not None:
            metadata["source_id"] = self.source_id
        if self.difficulty is not None:
            metadata["difficulty"] = self.difficulty.to_dict()
        if self.reference_tactic_count is not None:
            metadata["reference_tactic_count"] = self.reference_tactic_count
        return metadata


@dataclass(frozen=True)
class TheoremTask:
    """One theorem and its result-only analysis metadata loaded from JSONL."""

    theorem_id: str
    statement: str
    metadata: TaskMetadata = field(default_factory=TaskMetadata)


@dataclass(frozen=True)
class RequestAttempt:
    """One provider request targeting a generation slot."""

    request_index: int
    target_generation_index: int
    status: str
    elapsed_ms: int
    error: ErrorDetails | None


@dataclass(frozen=True)
class GenerationAcquisition:
    """Result of requesting one completed generation with bounded request retries."""

    generation: GenerationResult | None
    request_attempts: tuple[RequestAttempt, ...]
    terminal_status: str | None
    error: ErrorDetails | None


def acquire_generation(
    model: ProofModel,
    statement: str,
    *,
    target_generation_index: int,
    first_request_index: int,
    max_transport_retries: int,
    reasoning_mode: ReasoningMode,
    progress_prefix: str,
    progress_callback: ProgressCallback | None,
) -> GenerationAcquisition:
    """Request one generation; retry only classified retryable request failures."""

    if max_transport_retries < 0:
        raise ValueError("max_transport_retries must be non-negative")
    request_attempts: list[RequestAttempt] = []
    for retry_index in range(max_transport_retries + 1):
        request_index = first_request_index + len(request_attempts)
        report_progress(
            progress_callback,
            f"{progress_prefix} | request {request_index} | generating...",
        )
        started = time.perf_counter()
        try:
            generation = model.generate_proof(statement, reasoning_mode=reasoning_mode)
        except GenerationRequestError as error:
            elapsed_ms = (
                error.elapsed_ms if error.elapsed_ms is not None else elapsed_since(started)
            )
            if error.transport:
                status = "transport_failure"
            elif error.transient_api:
                status = "transient_api_failure"
            else:
                status = "generation_error"
            request_attempts.append(
                RequestAttempt(
                    request_index=request_index,
                    target_generation_index=target_generation_index,
                    status=status,
                    elapsed_ms=elapsed_ms,
                    error=error.details,
                )
            )
            report_progress(
                progress_callback,
                f"{progress_prefix} | request {request_index} | {status.upper()} | {elapsed_ms} ms",
            )
            if error.retryable and retry_index < max_transport_retries:
                continue
            if error.transport and error.retryable:
                terminal_status = "transport_retry_exhausted"
            elif error.transient_api and error.retryable:
                terminal_status = "request_retry_exhausted"
            else:
                terminal_status = "generation_error"
            return GenerationAcquisition(
                generation=None,
                request_attempts=tuple(request_attempts),
                terminal_status=terminal_status,
                error=error.details,
            )
        except Exception as error:  # noqa: BLE001
            details = ErrorDetails.from_exception("generation", error)
            request_attempts.append(
                RequestAttempt(
                    request_index=request_index,
                    target_generation_index=target_generation_index,
                    status="generation_error",
                    elapsed_ms=elapsed_since(started),
                    error=details,
                )
            )
            return GenerationAcquisition(
                generation=None,
                request_attempts=tuple(request_attempts),
                terminal_status="generation_error",
                error=details,
            )
        request_attempts.append(
            RequestAttempt(
                request_index=request_index,
                target_generation_index=target_generation_index,
                status="completed",
                elapsed_ms=generation.latency_ms,
                error=None,
            )
        )
        return GenerationAcquisition(
            generation=generation,
            request_attempts=tuple(request_attempts),
            terminal_status=None,
            error=None,
        )
    raise AssertionError("bounded request loop did not return")


def load_dataset(dataset_path: str | Path, *, limit: int | None = None) -> list[TheoremTask]:
    """Load and validate theorem IDs and statements from a JSONL dataset."""

    path = Path(dataset_path)
    if not path.is_file():
        raise DatasetError(f"Dataset file does not exist: {path}")
    if limit is not None and limit <= 0:
        raise DatasetError("limit must be greater than zero")

    tasks: list[TheoremTask] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise DatasetError(f"Blank JSONL record at {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetError(f"Invalid JSON at {path}:{line_number}: {error.msg}") from error
            if not isinstance(record, dict):
                raise DatasetError(f"JSONL record must be an object at {path}:{line_number}")
            theorem_id = record.get("theorem_id", record.get("id"))
            statement = record.get("statement")
            if not isinstance(theorem_id, str) or not theorem_id.strip():
                raise DatasetError(f"Missing theorem_id at {path}:{line_number}")
            if not isinstance(statement, str) or not statement.strip():
                raise DatasetError(f"Missing statement at {path}:{line_number}")
            if theorem_id in seen_ids:
                raise DatasetError(f"Duplicate theorem_id at {path}:{line_number}: {theorem_id}")
            seen_ids.add(theorem_id)
            tasks.append(
                TheoremTask(
                    theorem_id=theorem_id,
                    statement=statement.strip(),
                    metadata=_load_task_metadata(record, path, line_number),
                )
            )
    if not tasks:
        raise DatasetError(f"Dataset is empty: {path}")
    return tasks[:limit]


def _load_task_metadata(record: dict[str, object], path: Path, line_number: int) -> TaskMetadata:
    source = _optional_metadata_string(record.get("source"), "source", path, line_number)
    source_id = _optional_metadata_string(record.get("source_id"), "source_id", path, line_number)
    difficulty_value = record.get("difficulty")
    difficulty: TaskDifficulty | None = None
    if difficulty_value is not None:
        if not isinstance(difficulty_value, dict):
            raise DatasetError(f"Invalid difficulty at {path}:{line_number}")
        score = difficulty_value.get("score")
        bucket = difficulty_value.get("bucket")
        method = difficulty_value.get("method")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not isinstance(bucket, str)
            or not bucket.strip()
            or not isinstance(method, str)
            or not method.strip()
        ):
            raise DatasetError(f"Invalid difficulty at {path}:{line_number}")
        difficulty = TaskDifficulty(float(score), bucket.strip(), method.strip())

    reference_tactic_count: int | None = None
    features = record.get("features")
    if isinstance(features, dict) and "reference_tactic_count" in features:
        count = features.get("reference_tactic_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise DatasetError(f"Invalid reference_tactic_count at {path}:{line_number}")
        reference_tactic_count = count
    return TaskMetadata(
        source=source,
        source_id=source_id,
        difficulty=difficulty,
        reference_tactic_count=reference_tactic_count,
    )


def _optional_metadata_string(
    value: object, field_name: str, path: Path, line_number: int
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"Invalid {field_name} at {path}:{line_number}")
    return value.strip()


def error_details(stage: str, error: Exception) -> ErrorDetails:
    """Create structured diagnostics for non-provider strategy failures."""

    return ErrorDetails.from_exception(stage, error)


def remove_empty_status_codes(value: object) -> None:
    """Remove optional null status codes from nested serialized error records."""

    if isinstance(value, dict):
        if value.get("status_code") is None:
            value.pop("status_code", None)
        for nested in value.values():
            remove_empty_status_codes(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            remove_empty_status_codes(nested)


def elapsed_since(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def report_progress(progress_callback: ProgressCallback | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)
