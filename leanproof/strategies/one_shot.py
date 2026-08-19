"""Dataset execution for the one-shot full-proof baseline."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from leanproof.lean import VerificationStatus
from leanproof.models import (
    ErrorDetails,
    GenerationResult,
    ProofGenerationError,
    ProofModel,
    normalize_proof,
)
from leanproof.strategies.common import (
    DEFAULT_MAX_TRANSPORT_RETRIES,
    DatasetError,
    ProgressCallback,
    ProofVerifier,
    RequestAttempt,
    TaskDifficulty,
    TaskMetadata,
    TheoremTask,
    acquire_generation,
    elapsed_since,
    error_details,
    load_dataset,
    remove_empty_status_codes,
    report_progress,
)

__all__ = [
    "DatasetError",
    "OneShotResult",
    "OneShotSummary",
    "ProgressCallback",
    "ProofVerifier",
    "TaskDifficulty",
    "TaskMetadata",
    "TheoremTask",
    "default_output_path",
    "load_dataset",
    "run_one_shot",
]


@dataclass(frozen=True)
class OneShotResult:
    """Serializable result for one theorem with a one-generation budget."""

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
    max_transport_retries: int
    request_attempts: tuple[RequestAttempt, ...]
    raw_model_output: str
    reasoning_output: str | None
    proof_output: str
    normalized_proof: str
    verification_status: str | None
    verified: bool
    has_sorry: bool
    lean_stdout: str
    lean_stderr: str
    terminal_status: str
    api_requests: int
    request_failures: int
    transport_failures: int
    generations_used: int
    verifier_calls: int
    generation_latency_ms: int
    verification_latency_ms: int
    total_latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    error: ErrorDetails | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable record without modifying raw model output."""

        record = asdict(self)
        remove_empty_status_codes(record)
        return record


@dataclass(frozen=True)
class OneShotSummary:
    """Aggregate one-shot metrics with requests and generations kept separate."""

    total: int
    solved: int
    total_api_requests: int
    total_request_failures: int
    total_transport_failures: int
    total_generations: int
    total_verifier_calls: int
    average_generation_latency_ms: float
    average_verification_latency_ms: float
    output_path: Path

    @property
    def success_rate(self) -> float:
        """Return the percentage of tasks verified successfully."""

        return 100.0 * self.solved / self.total


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
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES,
    generation_timeout_seconds: float | None = None,
    verification_timeout_seconds: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> OneShotSummary:
    """Run one completed generation per theorem with bounded request retries."""

    if not tasks:
        raise ValueError("tasks must not be empty")
    if max_transport_retries < 0:
        raise ValueError("max_transport_retries must be non-negative")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results: list[OneShotResult] = []
    with destination.open("x", encoding="utf-8", newline="\n") as sink:
        for index, task in enumerate(tasks, start=1):
            result = _run_task_once(
                task,
                model,
                verifier,
                model_alias=model_alias,
                dataset=dataset,
                max_transport_retries=max_transport_retries,
                generation_timeout_seconds=generation_timeout_seconds,
                verification_timeout_seconds=verification_timeout_seconds,
                index=index,
                total=len(tasks),
                progress_callback=progress_callback,
            )
            results.append(result)
            sink.write(
                json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            sink.flush()
    return _summarize_results(results, destination)


def _run_task_once(
    task: TheoremTask,
    model: ProofModel,
    verifier: ProofVerifier,
    *,
    model_alias: str,
    dataset: str | None,
    max_transport_retries: int,
    generation_timeout_seconds: float | None,
    verification_timeout_seconds: float | None,
    index: int,
    total: int,
    progress_callback: ProgressCallback | None,
) -> OneShotResult:
    started = time.perf_counter()
    prefix = f"[{index}/{total}] {task.theorem_id} | generation 1/1"
    acquisition = acquire_generation(
        model,
        task.statement,
        target_generation_index=1,
        first_request_index=1,
        max_transport_retries=max_transport_retries,
        progress_prefix=prefix,
        progress_callback=progress_callback,
    )
    generation = acquisition.generation
    if generation is None:
        total_latency_ms = elapsed_since(started)
        report_progress(
            progress_callback,
            f"{prefix} | {(acquisition.terminal_status or 'generation_error').upper()} | "
            f"total {total_latency_ms} ms",
        )
        return _base_result(
            task,
            model,
            model_alias=model_alias,
            dataset=dataset,
            max_transport_retries=max_transport_retries,
            generation_timeout_seconds=generation_timeout_seconds,
            verification_timeout_seconds=verification_timeout_seconds,
            request_attempts=acquisition.request_attempts,
            terminal_status=acquisition.terminal_status or "generation_error",
            total_latency_ms=total_latency_ms,
            error=acquisition.error,
        )

    report_progress(progress_callback, f"{prefix} | generated | {generation.latency_ms} ms")
    try:
        normalized_proof = normalize_proof(generation.proof_output)
    except ProofGenerationError as error:
        return _generation_result(
            task,
            model,
            generation,
            model_alias=model_alias,
            dataset=dataset,
            max_transport_retries=max_transport_retries,
            generation_timeout_seconds=generation_timeout_seconds,
            verification_timeout_seconds=verification_timeout_seconds,
            request_attempts=acquisition.request_attempts,
            normalized_proof="",
            verification_status=None,
            verified=False,
            has_sorry=False,
            lean_stdout="",
            lean_stderr="",
            terminal_status="generation_budget_exhausted",
            verifier_calls=0,
            verification_latency_ms=0,
            total_latency_ms=elapsed_since(started),
            error=error_details("normalization", error),
        )

    report_progress(progress_callback, f"{prefix} | verifying...")
    verification_started = time.perf_counter()
    try:
        verification = verifier.verify(task.statement, normalized_proof)
    except Exception as error:  # noqa: BLE001
        result = _generation_result(
            task,
            model,
            generation,
            model_alias=model_alias,
            dataset=dataset,
            max_transport_retries=max_transport_retries,
            generation_timeout_seconds=generation_timeout_seconds,
            verification_timeout_seconds=verification_timeout_seconds,
            request_attempts=acquisition.request_attempts,
            normalized_proof=normalized_proof,
            verification_status=VerificationStatus.EXECUTION_ERROR.value,
            verified=False,
            has_sorry=False,
            lean_stdout="",
            lean_stderr="",
            terminal_status="generation_budget_exhausted",
            verifier_calls=1,
            verification_latency_ms=elapsed_since(verification_started),
            total_latency_ms=elapsed_since(started),
            error=error_details("verification", error),
        )
    else:
        result = _generation_result(
            task,
            model,
            generation,
            model_alias=model_alias,
            dataset=dataset,
            max_transport_retries=max_transport_retries,
            generation_timeout_seconds=generation_timeout_seconds,
            verification_timeout_seconds=verification_timeout_seconds,
            request_attempts=acquisition.request_attempts,
            normalized_proof=normalized_proof,
            verification_status=verification.status.value,
            verified=verification.verified,
            has_sorry=verification.has_sorry,
            lean_stdout=verification.stdout,
            lean_stderr=verification.stderr,
            terminal_status=(
                "verified" if verification.verified else "generation_budget_exhausted"
            ),
            verifier_calls=1,
            verification_latency_ms=verification.elapsed_ms,
            total_latency_ms=elapsed_since(started),
            error=None,
        )
    status = "PASS" if result.verified else "FAIL"
    report_progress(
        progress_callback,
        f"{prefix} | {status} | verify {result.verification_latency_ms} ms | "
        f"total {result.total_latency_ms} ms",
    )
    return result


def _base_result(
    task: TheoremTask,
    model: ProofModel,
    *,
    model_alias: str,
    dataset: str | None,
    max_transport_retries: int,
    generation_timeout_seconds: float | None,
    verification_timeout_seconds: float | None,
    request_attempts: tuple[RequestAttempt, ...],
    terminal_status: str,
    total_latency_ms: int,
    error: ErrorDetails | None,
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
        max_transport_retries=max_transport_retries,
        request_attempts=request_attempts,
        raw_model_output="",
        reasoning_output=None,
        proof_output="",
        normalized_proof="",
        verification_status=None,
        verified=False,
        has_sorry=False,
        lean_stdout="",
        lean_stderr="",
        terminal_status=terminal_status,
        api_requests=len(request_attempts),
        request_failures=sum(request.status != "completed" for request in request_attempts),
        transport_failures=sum(
            request.status == "transport_failure" for request in request_attempts
        ),
        generations_used=0,
        verifier_calls=0,
        generation_latency_ms=0,
        verification_latency_ms=0,
        total_latency_ms=total_latency_ms,
        prompt_tokens=None,
        completion_tokens=None,
        error=error,
    )


def _generation_result(
    task: TheoremTask,
    model: ProofModel,
    generation: GenerationResult,
    *,
    model_alias: str,
    dataset: str | None,
    max_transport_retries: int,
    generation_timeout_seconds: float | None,
    verification_timeout_seconds: float | None,
    request_attempts: tuple[RequestAttempt, ...],
    normalized_proof: str,
    verification_status: str | None,
    verified: bool,
    has_sorry: bool,
    lean_stdout: str,
    lean_stderr: str,
    terminal_status: str,
    verifier_calls: int,
    verification_latency_ms: int,
    total_latency_ms: int,
    error: ErrorDetails | None,
) -> OneShotResult:
    result = _base_result(
        task,
        model,
        model_alias=model_alias,
        dataset=dataset,
        max_transport_retries=max_transport_retries,
        generation_timeout_seconds=generation_timeout_seconds,
        verification_timeout_seconds=verification_timeout_seconds,
        request_attempts=request_attempts,
        terminal_status=terminal_status,
        total_latency_ms=total_latency_ms,
        error=error,
    )
    return replace(
        result,
        raw_model_output=generation.raw_output,
        reasoning_output=generation.reasoning_output,
        proof_output=generation.proof_output,
        normalized_proof=normalized_proof,
        verification_status=verification_status,
        verified=verified,
        has_sorry=has_sorry,
        lean_stdout=lean_stdout,
        lean_stderr=lean_stderr,
        generations_used=1,
        verifier_calls=verifier_calls,
        generation_latency_ms=generation.latency_ms,
        verification_latency_ms=verification_latency_ms,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
    )


def _summarize_results(results: Sequence[OneShotResult], output_path: Path) -> OneShotSummary:
    total_generations = sum(result.generations_used for result in results)
    total_verifier_calls = sum(result.verifier_calls for result in results)
    return OneShotSummary(
        total=len(results),
        solved=sum(result.verified for result in results),
        total_api_requests=sum(result.api_requests for result in results),
        total_request_failures=sum(result.request_failures for result in results),
        total_transport_failures=sum(result.transport_failures for result in results),
        total_generations=total_generations,
        total_verifier_calls=total_verifier_calls,
        average_generation_latency_ms=(
            sum(result.generation_latency_ms for result in results) / total_generations
            if total_generations
            else 0.0
        ),
        average_verification_latency_ms=(
            sum(result.verification_latency_ms for result in results) / total_verifier_calls
            if total_verifier_calls
            else 0.0
        ),
        output_path=output_path,
    )
