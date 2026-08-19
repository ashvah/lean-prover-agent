"""Independent repeated-sampling baseline for complete Lean proofs."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
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
from leanproof.prompts import ReasoningMode
from leanproof.strategies.common import (
    DEFAULT_MAX_TRANSPORT_RETRIES,
    ProgressCallback,
    ProofVerifier,
    RequestAttempt,
    TheoremTask,
    acquire_generation,
    elapsed_since,
    error_details,
    remove_empty_status_codes,
    report_progress,
)

DEFAULT_MAX_ATTEMPTS = 4
RETRY_STRATEGY = "retry"


@dataclass(frozen=True)
class RetryAttempt:
    """One completed independent generation and its optional Lean verification."""

    attempt_index: int
    request_index: int
    raw_model_output: str
    native_reasoning_output: str | None
    plan_output: str | None
    proof_output: str
    normalized_proof: str
    verification_status: str | None
    verified: bool
    has_sorry: bool
    lean_stdout: str
    lean_stderr: str
    prompt_tokens: int | None
    completion_tokens: int | None
    generation_latency_ms: int
    verification_latency_ms: int
    total_attempt_latency_ms: int
    error: ErrorDetails | None


@dataclass(frozen=True)
class RetryResult:
    """Serializable theorem-level request and generation trajectory."""

    theorem_id: str
    statement: str
    task_metadata: dict[str, object]
    strategy: str
    dataset: str | None
    model_alias: str
    model: str
    generation_budget: int
    reasoning_mode: ReasoningMode
    connect_timeout_seconds: float | None
    read_timeout_seconds: float | None
    write_timeout_seconds: float | None
    pool_timeout_seconds: float | None
    verification_timeout_seconds: float | None
    max_transport_retries: int
    request_attempts: tuple[RequestAttempt, ...]
    attempts: tuple[RetryAttempt, ...]
    terminal_status: str
    final_verification_status: str | None
    solved: bool
    api_requests: int
    request_failures: int
    transport_failures: int
    transient_api_failures: int
    generations_used: int
    verifier_calls: int
    prompt_tokens: int | None
    completion_tokens: int | None
    generation_latency_ms: int
    verification_latency_ms: int
    total_latency_ms: int
    error: ErrorDetails | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible theorem trajectory without changing raw outputs."""

        record = asdict(self)
        remove_empty_status_codes(record)
        return record


@dataclass(frozen=True)
class RetrySummary:
    """Aggregate metrics for requests, completed generations, and verifier calls."""

    total: int
    solved: int
    generation_budget: int
    total_api_requests: int
    total_request_failures: int
    total_transport_failures: int
    total_transient_api_failures: int
    total_generations: int
    total_verifier_calls: int
    average_generations_per_theorem: float
    average_generations_per_solved_theorem: float | None
    total_prompt_tokens: int | None
    total_completion_tokens: int | None
    average_prompt_tokens: float | None
    average_completion_tokens: float | None
    prompt_token_generations: int
    completion_token_generations: int
    average_generation_latency_ms: float
    average_verification_latency_ms: float
    output_path: Path

    @property
    def success_rate(self) -> float:
        """Return the percentage of theorem trajectories that reached VERIFIED."""

        return 100.0 * self.solved / self.total


def default_retry_output_path(
    dataset_path: str | Path,
    model_alias: str,
    max_attempts: int,
    output_directory: str | Path,
) -> Path:
    """Build a timestamped retry result path containing alias and generation budget."""

    _validate_max_attempts(max_attempts)
    if re.fullmatch(r"[a-z][a-z0-9_]*", model_alias) is None:
        raise ValueError("model_alias is not safe for an output filename")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
    dataset_name = Path(dataset_path).stem
    return (
        Path(output_directory)
        / f"retry_{dataset_name}_{model_alias}_k{max_attempts}_{timestamp}.jsonl"
    )


def run_retry(
    tasks: Sequence[TheoremTask],
    model: ProofModel,
    verifier: ProofVerifier,
    output_path: str | Path,
    *,
    model_alias: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES,
    dataset: str | None = None,
    reasoning_mode: ReasoningMode = "none",
    connect_timeout_seconds: float | None = None,
    read_timeout_seconds: float | None = None,
    write_timeout_seconds: float | None = None,
    pool_timeout_seconds: float | None = None,
    verification_timeout_seconds: float | None = None,
    progress_callback: ProgressCallback | None = None,
) -> RetrySummary:
    """Run independent completed generations until VERIFIED or budget exhaustion."""

    if not tasks:
        raise ValueError("tasks must not be empty")
    _validate_max_attempts(max_attempts)
    if max_transport_retries < 0:
        raise ValueError("max_transport_retries must be non-negative")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results: list[RetryResult] = []
    with destination.open("x", encoding="utf-8", newline="\n") as sink:
        for theorem_index, task in enumerate(tasks, start=1):
            result = _run_theorem_retry(
                task,
                model,
                verifier,
                model_alias=model_alias,
                max_attempts=max_attempts,
                max_transport_retries=max_transport_retries,
                dataset=dataset,
                reasoning_mode=reasoning_mode,
                connect_timeout_seconds=connect_timeout_seconds,
                read_timeout_seconds=read_timeout_seconds,
                write_timeout_seconds=write_timeout_seconds,
                pool_timeout_seconds=pool_timeout_seconds,
                verification_timeout_seconds=verification_timeout_seconds,
                theorem_index=theorem_index,
                total_theorems=len(tasks),
                progress_callback=progress_callback,
            )
            results.append(result)
            sink.write(json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ":")))
            sink.write("\n")
            sink.flush()
    return _summarize_results(results, max_attempts, destination)


def _run_theorem_retry(
    task: TheoremTask,
    model: ProofModel,
    verifier: ProofVerifier,
    *,
    model_alias: str,
    max_attempts: int,
    max_transport_retries: int,
    dataset: str | None,
    reasoning_mode: ReasoningMode,
    connect_timeout_seconds: float | None,
    read_timeout_seconds: float | None,
    write_timeout_seconds: float | None,
    pool_timeout_seconds: float | None,
    verification_timeout_seconds: float | None,
    theorem_index: int,
    total_theorems: int,
    progress_callback: ProgressCallback | None,
) -> RetryResult:
    theorem_started = time.perf_counter()
    attempts: list[RetryAttempt] = []
    request_attempts: list[RequestAttempt] = []
    verifier_calls = 0
    terminal_status: str | None = None
    terminal_error: ErrorDetails | None = None

    while len(attempts) < max_attempts:
        attempt_started = time.perf_counter()
        attempt_index = len(attempts) + 1
        prefix = (
            f"[{theorem_index}/{total_theorems}] {task.theorem_id} | "
            f"generation {attempt_index}/{max_attempts}"
        )
        acquisition = acquire_generation(
            model,
            task.statement,
            target_generation_index=attempt_index,
            first_request_index=len(request_attempts) + 1,
            max_transport_retries=max_transport_retries,
            reasoning_mode=reasoning_mode,
            progress_prefix=prefix,
            progress_callback=progress_callback,
        )
        request_attempts.extend(acquisition.request_attempts)
        generation = acquisition.generation
        if generation is None:
            terminal_status = acquisition.terminal_status or "generation_error"
            terminal_error = acquisition.error
            break

        request_index = acquisition.request_attempts[-1].request_index
        report_progress(progress_callback, f"{prefix} | generated | {generation.latency_ms} ms")
        try:
            normalized_proof = normalize_proof(generation.proof_output)
        except ProofGenerationError as error:
            attempt = _generation_attempt(
                attempt_index,
                request_index,
                generation,
                normalized_proof="",
                verification_status=None,
                verified=False,
                has_sorry=False,
                lean_stdout="",
                lean_stderr="",
                verification_latency_ms=0,
                total_attempt_latency_ms=elapsed_since(attempt_started),
                error=error_details("normalization", error),
            )
            attempts.append(attempt)
            report_progress(
                progress_callback,
                f"{prefix} | OUTPUT_FORMAT_ERROR | total {attempt.total_attempt_latency_ms} ms",
            )
            continue

        report_progress(progress_callback, f"{prefix} | verifying...")
        verification_started = time.perf_counter()
        verifier_calls += 1
        try:
            verification = verifier.verify(task.statement, normalized_proof)
        except Exception as error:  # noqa: BLE001
            attempt = _generation_attempt(
                attempt_index,
                request_index,
                generation,
                normalized_proof=normalized_proof,
                verification_status=VerificationStatus.EXECUTION_ERROR.value,
                verified=False,
                has_sorry=False,
                lean_stdout="",
                lean_stderr="",
                verification_latency_ms=elapsed_since(verification_started),
                total_attempt_latency_ms=elapsed_since(attempt_started),
                error=error_details("verification", error),
            )
            terminal_status = "verifier_execution_error"
            terminal_error = attempt.error
        else:
            attempt = _generation_attempt(
                attempt_index,
                request_index,
                generation,
                normalized_proof=normalized_proof,
                verification_status=verification.status.value,
                verified=verification.verified,
                has_sorry=verification.has_sorry,
                lean_stdout=verification.stdout,
                lean_stderr=verification.stderr,
                verification_latency_ms=verification.elapsed_ms,
                total_attempt_latency_ms=elapsed_since(attempt_started),
                error=None,
            )
            if verification.status is VerificationStatus.EXECUTION_ERROR:
                terminal_status = "verifier_execution_error"
        attempts.append(attempt)
        status = attempt.verification_status.upper() if attempt.verification_status else "NOT RUN"
        solved_suffix = " | solved" if attempt.verified else ""
        report_progress(
            progress_callback,
            f"{prefix} | {status} | verify {attempt.verification_latency_ms} ms | "
            f"total {attempt.total_attempt_latency_ms} ms{solved_suffix}",
        )
        if attempt.verified:
            terminal_status = "verified"
            break
        if terminal_status == "verifier_execution_error":
            break

    if terminal_status is None:
        terminal_status = (
            "verified"
            if any(attempt.verified for attempt in attempts)
            else ("generation_budget_exhausted")
        )
    attempt_tuple = tuple(attempts)
    request_tuple = tuple(request_attempts)
    return RetryResult(
        theorem_id=task.theorem_id,
        statement=task.statement,
        task_metadata=task.metadata.to_dict(),
        strategy=RETRY_STRATEGY,
        dataset=dataset,
        model_alias=model_alias,
        model=model.model_name,
        generation_budget=max_attempts,
        reasoning_mode=reasoning_mode,
        connect_timeout_seconds=connect_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
        write_timeout_seconds=write_timeout_seconds,
        pool_timeout_seconds=pool_timeout_seconds,
        verification_timeout_seconds=verification_timeout_seconds,
        max_transport_retries=max_transport_retries,
        request_attempts=request_tuple,
        attempts=attempt_tuple,
        terminal_status=terminal_status,
        final_verification_status=_last_verification_status(attempt_tuple),
        solved=any(attempt.verified for attempt in attempt_tuple),
        api_requests=len(request_tuple),
        request_failures=sum(request.status != "completed" for request in request_tuple),
        transport_failures=sum(request.status == "transport_failure" for request in request_tuple),
        transient_api_failures=sum(
            request.status == "transient_api_failure" for request in request_tuple
        ),
        generations_used=len(attempt_tuple),
        verifier_calls=verifier_calls,
        prompt_tokens=_complete_token_sum(attempt.prompt_tokens for attempt in attempt_tuple),
        completion_tokens=_complete_token_sum(
            attempt.completion_tokens for attempt in attempt_tuple
        ),
        generation_latency_ms=sum(attempt.generation_latency_ms for attempt in attempt_tuple),
        verification_latency_ms=sum(attempt.verification_latency_ms for attempt in attempt_tuple),
        total_latency_ms=elapsed_since(theorem_started),
        error=terminal_error,
    )


def _generation_attempt(
    attempt_index: int,
    request_index: int,
    generation: GenerationResult,
    *,
    normalized_proof: str,
    verification_status: str | None,
    verified: bool,
    has_sorry: bool,
    lean_stdout: str,
    lean_stderr: str,
    verification_latency_ms: int,
    total_attempt_latency_ms: int,
    error: ErrorDetails | None,
) -> RetryAttempt:
    return RetryAttempt(
        attempt_index=attempt_index,
        request_index=request_index,
        raw_model_output=generation.raw_output,
        native_reasoning_output=generation.native_reasoning_output,
        plan_output=generation.plan_output,
        proof_output=generation.proof_output,
        normalized_proof=normalized_proof,
        verification_status=verification_status,
        verified=verified,
        has_sorry=has_sorry,
        lean_stdout=lean_stdout,
        lean_stderr=lean_stderr,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        generation_latency_ms=generation.latency_ms,
        verification_latency_ms=verification_latency_ms,
        total_attempt_latency_ms=total_attempt_latency_ms,
        error=error,
    )


def _last_verification_status(attempts: Sequence[RetryAttempt]) -> str | None:
    return next(
        (
            attempt.verification_status
            for attempt in reversed(attempts)
            if attempt.verification_status is not None
        ),
        None,
    )


def _summarize_results(
    results: Sequence[RetryResult], max_attempts: int, output_path: Path
) -> RetrySummary:
    attempts = [attempt for result in results for attempt in result.attempts]
    solved_generation_counts = [result.generations_used for result in results if result.solved]
    prompt_tokens = [
        attempt.prompt_tokens for attempt in attempts if attempt.prompt_tokens is not None
    ]
    completion_tokens = [
        attempt.completion_tokens for attempt in attempts if attempt.completion_tokens is not None
    ]
    verification_latencies = [
        attempt.verification_latency_ms
        for attempt in attempts
        if attempt.verification_status is not None
    ]
    return RetrySummary(
        total=len(results),
        solved=len(solved_generation_counts),
        generation_budget=max_attempts,
        total_api_requests=sum(result.api_requests for result in results),
        total_request_failures=sum(result.request_failures for result in results),
        total_transport_failures=sum(result.transport_failures for result in results),
        total_transient_api_failures=sum(result.transient_api_failures for result in results),
        total_generations=len(attempts),
        total_verifier_calls=sum(result.verifier_calls for result in results),
        average_generations_per_theorem=len(attempts) / len(results),
        average_generations_per_solved_theorem=(
            sum(solved_generation_counts) / len(solved_generation_counts)
            if solved_generation_counts
            else None
        ),
        total_prompt_tokens=sum(prompt_tokens) if prompt_tokens else None,
        total_completion_tokens=sum(completion_tokens) if completion_tokens else None,
        average_prompt_tokens=(sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None),
        average_completion_tokens=(
            sum(completion_tokens) / len(completion_tokens) if completion_tokens else None
        ),
        prompt_token_generations=len(prompt_tokens),
        completion_token_generations=len(completion_tokens),
        average_generation_latency_ms=(
            sum(attempt.generation_latency_ms for attempt in attempts) / len(attempts)
            if attempts
            else 0.0
        ),
        average_verification_latency_ms=(
            sum(verification_latencies) / len(verification_latencies)
            if verification_latencies
            else 0.0
        ),
        output_path=output_path,
    )


def _complete_token_sum(values: Iterable[int | None]) -> int | None:
    token_values = tuple(values)
    if not token_values or any(value is None for value in token_values):
        return None
    return sum(value for value in token_values if value is not None)


def _validate_max_attempts(max_attempts: int) -> None:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")
