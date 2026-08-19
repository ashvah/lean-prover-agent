"""Dataset loading and batch execution for the one-shot baseline."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from leanproof.lean import LeanResult, VerificationStatus
from leanproof.models import GenerationResult, ProofGenerationError, ProofModel, normalize_proof


class DatasetError(ValueError):
    """Raised when a one-shot JSONL dataset is missing or malformed."""


class ProofVerifier(Protocol):
    """Minimal verifier interface required by the one-shot runner."""

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
class OneShotResult:
    """Serializable result for one theorem and one generation attempt."""

    theorem_id: str
    statement: str
    task_metadata: dict[str, object]
    strategy: str
    generation_budget: int
    dataset: str | None
    model_alias: str
    model: str
    generation_timeout_seconds: float | None
    verification_timeout_seconds: float | None
    raw_model_output: str
    reasoning_output: str | None
    proof_output: str
    normalized_proof: str
    verification_status: str | None
    verified: bool
    has_sorry: bool
    lean_stdout: str
    lean_stderr: str
    generation_latency_ms: int
    verification_latency_ms: int
    total_latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable record without modifying raw model output."""

        return asdict(self)


@dataclass(frozen=True)
class OneShotSummary:
    """Aggregate metrics printed after a one-shot batch."""

    total: int
    solved: int
    average_generation_latency_ms: float
    average_verification_latency_ms: float
    output_path: Path

    @property
    def success_rate(self) -> float:
        """Return the percentage of tasks verified successfully."""

        return 100.0 * self.solved / self.total


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


def default_output_path(
    dataset_path: str | Path, model_alias: str, output_directory: str | Path
) -> Path:
    """Build a timestamped output path without provider or credential data."""

    if re.fullmatch(r"[a-z][a-z0-9_]*", model_alias) is None:
        raise ValueError("model_alias is not safe for an output filename")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
    dataset_name = Path(dataset_path).stem
    return Path(output_directory) / f"one_shot_{dataset_name}_{model_alias}_{timestamp}.jsonl"


def run_one_shot(
    tasks: Sequence[TheoremTask],
    model: ProofModel,
    verifier: ProofVerifier,
    output_path: str | Path,
    *,
    model_alias: str,
    dataset: str | None = None,
    generation_timeout_seconds: float | None = None,
    verification_timeout_seconds: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> OneShotSummary:
    """Generate and verify each theorem once, recording failures without retrying."""

    if not tasks:
        raise ValueError("tasks must not be empty")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results: list[OneShotResult] = []
    verification_latencies: list[int] = []

    with destination.open("x", encoding="utf-8", newline="\n") as sink:
        for index, task in enumerate(tasks, start=1):
            result = _run_task_once(
                task,
                model,
                verifier,
                model_alias=model_alias,
                dataset=dataset,
                generation_timeout_seconds=generation_timeout_seconds,
                verification_timeout_seconds=verification_timeout_seconds,
                index=index,
                total=len(tasks),
                progress_callback=progress_callback,
            )
            results.append(result)
            if not result.error or not result.error.startswith("generation_error:"):
                verification_latencies.append(result.verification_latency_ms)
            sink.write(
                json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            sink.flush()

    solved = sum(result.verified for result in results)
    average_generation_latency = sum(result.generation_latency_ms for result in results) / len(
        results
    )
    average_verification_latency = (
        sum(verification_latencies) / len(verification_latencies) if verification_latencies else 0.0
    )
    return OneShotSummary(
        total=len(results),
        solved=solved,
        average_generation_latency_ms=average_generation_latency,
        average_verification_latency_ms=average_verification_latency,
        output_path=destination,
    )


def _run_task_once(
    task: TheoremTask,
    model: ProofModel,
    verifier: ProofVerifier,
    *,
    model_alias: str,
    dataset: str | None,
    generation_timeout_seconds: float | None,
    verification_timeout_seconds: float | None,
    index: int,
    total: int,
    progress_callback: ProgressCallback | None,
) -> OneShotResult:
    progress_prefix = f"[{index}/{total}] {task.theorem_id}"
    _report_progress(progress_callback, f"{progress_prefix} | generating...")
    started = time.perf_counter()
    generation_started = time.perf_counter()
    try:
        generation = model.generate_proof(task.statement)
    except Exception as error:  # noqa: BLE001
        generation_latency_ms = _elapsed_ms(generation_started)
        total_latency_ms = _elapsed_ms(started)
        _report_progress(
            progress_callback,
            f"{progress_prefix} | ERROR       | generation {generation_latency_ms} ms | "
            f"total {total_latency_ms} ms",
        )
        return OneShotResult(
            theorem_id=task.theorem_id,
            statement=task.statement,
            task_metadata=task.metadata.to_dict(),
            strategy="one_shot",
            generation_budget=1,
            dataset=dataset,
            model_alias=model_alias,
            model=model.model_name,
            generation_timeout_seconds=generation_timeout_seconds,
            verification_timeout_seconds=verification_timeout_seconds,
            raw_model_output="",
            reasoning_output=None,
            proof_output="",
            normalized_proof="",
            verification_status=None,
            verified=False,
            has_sorry=False,
            lean_stdout="",
            lean_stderr="",
            generation_latency_ms=generation_latency_ms,
            verification_latency_ms=0,
            total_latency_ms=total_latency_ms,
            prompt_tokens=None,
            completion_tokens=None,
            error=f"generation_error: {type(error).__name__}: {error}",
        )

    _report_progress(
        progress_callback,
        f"{progress_prefix} | generated   | {generation.latency_ms} ms",
    )
    try:
        normalized_proof = normalize_proof(generation.proof_output)
    except ProofGenerationError as error:
        total_latency_ms = _elapsed_ms(started)
        _report_progress(
            progress_callback,
            f"{progress_prefix} | ERROR       | output format | total {total_latency_ms} ms",
        )
        return _proof_format_failure_result(
            task,
            model,
            generation,
            model_alias,
            dataset,
            generation_timeout_seconds,
            verification_timeout_seconds,
            total_latency_ms,
            error,
        )
    _report_progress(progress_callback, f"{progress_prefix} | verifying...")
    verification_started = time.perf_counter()
    try:
        verification = verifier.verify(task.statement, normalized_proof)
    except Exception as error:  # noqa: BLE001
        verification_latency_ms = _elapsed_ms(verification_started)
        total_latency_ms = _elapsed_ms(started)
        _report_progress(
            progress_callback,
            f"{progress_prefix} | FAIL        | verify {verification_latency_ms} ms | "
            f"total {total_latency_ms} ms",
        )
        return _verification_exception_result(
            task,
            model,
            generation,
            normalized_proof,
            model_alias,
            dataset,
            generation_timeout_seconds,
            verification_timeout_seconds,
            total_latency_ms,
            verification_latency_ms,
            error,
        )

    total_latency_ms = _elapsed_ms(started)
    benchmark_status = "PASS" if verification.verified else "FAIL"
    _report_progress(
        progress_callback,
        f"{progress_prefix} | {benchmark_status:<11} | verifier "
        f"{verification.status.value.upper()} | verify {verification.elapsed_ms} ms | "
        f"total {total_latency_ms} ms",
    )
    return OneShotResult(
        theorem_id=task.theorem_id,
        statement=task.statement,
        task_metadata=task.metadata.to_dict(),
        strategy="one_shot",
        generation_budget=1,
        dataset=dataset,
        model_alias=model_alias,
        model=model.model_name,
        generation_timeout_seconds=generation_timeout_seconds,
        verification_timeout_seconds=verification_timeout_seconds,
        raw_model_output=generation.raw_output,
        reasoning_output=generation.reasoning_output,
        proof_output=generation.proof_output,
        normalized_proof=normalized_proof,
        verification_status=verification.status.value,
        verified=verification.verified,
        has_sorry=verification.has_sorry,
        lean_stdout=verification.stdout,
        lean_stderr=verification.stderr,
        generation_latency_ms=generation.latency_ms,
        verification_latency_ms=verification.elapsed_ms,
        total_latency_ms=total_latency_ms,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        error=None,
    )


def _verification_exception_result(
    task: TheoremTask,
    model: ProofModel,
    generation: GenerationResult,
    normalized_proof: str,
    model_alias: str,
    dataset: str | None,
    generation_timeout_seconds: float | None,
    verification_timeout_seconds: float | None,
    total_latency_ms: int,
    verification_latency_ms: int,
    error: Exception,
) -> OneShotResult:
    return OneShotResult(
        theorem_id=task.theorem_id,
        statement=task.statement,
        task_metadata=task.metadata.to_dict(),
        strategy="one_shot",
        generation_budget=1,
        dataset=dataset,
        model_alias=model_alias,
        model=model.model_name,
        generation_timeout_seconds=generation_timeout_seconds,
        verification_timeout_seconds=verification_timeout_seconds,
        raw_model_output=generation.raw_output,
        reasoning_output=generation.reasoning_output,
        proof_output=generation.proof_output,
        normalized_proof=normalized_proof,
        verification_status=VerificationStatus.EXECUTION_ERROR.value,
        verified=False,
        has_sorry=False,
        lean_stdout="",
        lean_stderr="",
        generation_latency_ms=generation.latency_ms,
        verification_latency_ms=verification_latency_ms,
        total_latency_ms=total_latency_ms,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        error=f"verification_error: {type(error).__name__}: {error}",
    )


def _proof_format_failure_result(
    task: TheoremTask,
    model: ProofModel,
    generation: GenerationResult,
    model_alias: str,
    dataset: str | None,
    generation_timeout_seconds: float | None,
    verification_timeout_seconds: float | None,
    total_latency_ms: int,
    error: ProofGenerationError,
) -> OneShotResult:
    return OneShotResult(
        theorem_id=task.theorem_id,
        statement=task.statement,
        task_metadata=task.metadata.to_dict(),
        strategy="one_shot",
        generation_budget=1,
        dataset=dataset,
        model_alias=model_alias,
        model=model.model_name,
        generation_timeout_seconds=generation_timeout_seconds,
        verification_timeout_seconds=verification_timeout_seconds,
        raw_model_output=generation.raw_output,
        reasoning_output=generation.reasoning_output,
        proof_output=generation.proof_output,
        normalized_proof="",
        verification_status=None,
        verified=False,
        has_sorry=False,
        lean_stdout="",
        lean_stderr="",
        generation_latency_ms=generation.latency_ms,
        verification_latency_ms=0,
        total_latency_ms=total_latency_ms,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        error=f"generation_error: {type(error).__name__}: {error}",
    )


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _report_progress(progress_callback: ProgressCallback | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)
