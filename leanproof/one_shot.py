"""Dataset loading and batch execution for the one-shot baseline."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from leanproof.model import GenerationResult, ProofModel, normalize_proof
from leanproof.verifier import LeanResult


class DatasetError(ValueError):
    """Raised when a one-shot JSONL dataset is missing or malformed."""


class ProofVerifier(Protocol):
    """Minimal verifier interface required by the one-shot runner."""

    def verify(self, statement: str, proof: str) -> LeanResult:
        """Verify one complete proof and return raw Lean diagnostics."""


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class TheoremTask:
    """One theorem loaded from a JSONL dataset."""

    theorem_id: str
    statement: str


@dataclass(frozen=True)
class OneShotResult:
    """Serializable result for one theorem and one generation attempt."""

    theorem_id: str
    statement: str
    model: str
    raw_model_output: str
    normalized_proof: str
    verified: bool
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
            theorem_id = record.get("theorem_id")
            statement = record.get("statement")
            if not isinstance(theorem_id, str) or not theorem_id.strip():
                raise DatasetError(f"Missing theorem_id at {path}:{line_number}")
            if not isinstance(statement, str) or not statement.strip():
                raise DatasetError(f"Missing statement at {path}:{line_number}")
            if theorem_id in seen_ids:
                raise DatasetError(f"Duplicate theorem_id at {path}:{line_number}: {theorem_id}")

            seen_ids.add(theorem_id)
            tasks.append(TheoremTask(theorem_id=theorem_id, statement=statement.strip()))

    if not tasks:
        raise DatasetError(f"Dataset is empty: {path}")
    return tasks[:limit]


def default_output_path(dataset_path: str | Path, output_directory: str | Path) -> Path:
    """Build a timestamped output path without provider or credential data."""

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
    dataset_name = Path(dataset_path).stem
    return Path(output_directory) / f"one_shot_{dataset_name}_{timestamp}.jsonl"


def run_one_shot(
    tasks: Sequence[TheoremTask],
    model: ProofModel,
    verifier: ProofVerifier,
    output_path: str | Path,
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
            model=model.model_name,
            raw_model_output="",
            normalized_proof="",
            verified=False,
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
    normalized_proof = normalize_proof(generation.raw_output)
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
            total_latency_ms,
            verification_latency_ms,
            error,
        )

    total_latency_ms = _elapsed_ms(started)
    status = "PASS" if verification.success else "FAIL"
    _report_progress(
        progress_callback,
        f"{progress_prefix} | {status:<11} | verify {verification.elapsed_ms} ms | "
        f"total {total_latency_ms} ms",
    )
    return OneShotResult(
        theorem_id=task.theorem_id,
        statement=task.statement,
        model=model.model_name,
        raw_model_output=generation.raw_output,
        normalized_proof=normalized_proof,
        verified=verification.success,
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
    total_latency_ms: int,
    verification_latency_ms: int,
    error: Exception,
) -> OneShotResult:
    return OneShotResult(
        theorem_id=task.theorem_id,
        statement=task.statement,
        model=model.model_name,
        raw_model_output=generation.raw_output,
        normalized_proof=normalized_proof,
        verified=False,
        lean_stdout="",
        lean_stderr="",
        generation_latency_ms=generation.latency_ms,
        verification_latency_ms=verification_latency_ms,
        total_latency_ms=total_latency_ms,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        error=f"verification_error: {type(error).__name__}: {error}",
    )


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _report_progress(progress_callback: ProgressCallback | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)
