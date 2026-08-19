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
from leanproof.models import GenerationResult, ProofModel, normalize_proof
from leanproof.strategies.one_shot import ProgressCallback, ProofVerifier, TheoremTask

DEFAULT_MAX_ATTEMPTS = 4
RETRY_STRATEGY = "retry"


@dataclass(frozen=True)
class RetryAttempt:
    """One independent generation and its optional Lean verification."""

    attempt_index: int
    raw_model_output: str
    reasoning_output: str | None
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
    error: str | None


@dataclass(frozen=True)
class RetryResult:
    """Serializable theorem-level retry trajectory."""

    theorem_id: str
    statement: str
    strategy: str
    model_alias: str
    model: str
    generation_budget: int
    attempts: tuple[RetryAttempt, ...]
    final_verification_status: str | None
    solved: bool
    generations_used: int
    verifier_calls: int
    prompt_tokens: int | None
    completion_tokens: int | None
    generation_latency_ms: int
    verification_latency_ms: int
    total_latency_ms: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible theorem trajectory without changing raw outputs."""

        record = asdict(self)
        record["attempts"] = [asdict(attempt) for attempt in self.attempts]
        return record


@dataclass(frozen=True)
class RetrySummary:
    """Aggregate metrics printed after an independent retry batch."""

    total: int
    solved: int
    generation_budget: int
    total_attempts: int
    average_attempts_per_theorem: float
    average_attempts_per_solved_theorem: float | None
    total_prompt_tokens: int | None
    total_completion_tokens: int | None
    average_prompt_tokens: float | None
    average_completion_tokens: float | None
    prompt_token_attempts: int
    completion_token_attempts: int
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
    progress_callback: ProgressCallback | None = None,
) -> RetrySummary:
    """Run independent proof generations until VERIFIED or the budget is exhausted."""

    if not tasks:
        raise ValueError("tasks must not be empty")
    _validate_max_attempts(max_attempts)

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
    theorem_index: int,
    total_theorems: int,
    progress_callback: ProgressCallback | None,
) -> RetryResult:
    theorem_started = time.perf_counter()
    attempts: list[RetryAttempt] = []
    verifier_calls = 0
    for attempt_index in range(1, max_attempts + 1):
        prefix = (
            f"[{theorem_index}/{total_theorems}] {task.theorem_id} | "
            f"attempt {attempt_index}/{max_attempts}"
        )
        _report_progress(progress_callback, f"{prefix} | generating...")
        attempt_started = time.perf_counter()
        generation_started = time.perf_counter()
        try:
            generation = model.generate_proof(task.statement)
        except Exception as error:  # noqa: BLE001
            generation_latency_ms = _elapsed_ms(generation_started)
            attempt = RetryAttempt(
                attempt_index=attempt_index,
                raw_model_output="",
                reasoning_output=None,
                proof_output="",
                normalized_proof="",
                verification_status=None,
                verified=False,
                has_sorry=False,
                lean_stdout="",
                lean_stderr="",
                prompt_tokens=None,
                completion_tokens=None,
                generation_latency_ms=generation_latency_ms,
                verification_latency_ms=0,
                total_attempt_latency_ms=_elapsed_ms(attempt_started),
                error=f"generation_error: {type(error).__name__}: {error}",
            )
            attempts.append(attempt)
            _report_progress(
                progress_callback,
                f"{prefix} | ERROR | generation {generation_latency_ms} ms | "
                f"total {attempt.total_attempt_latency_ms} ms",
            )
            continue

        _report_progress(
            progress_callback,
            f"{prefix} | generated | {generation.latency_ms} ms",
        )
        normalized_proof = normalize_proof(generation.proof_output)
        _report_progress(progress_callback, f"{prefix} | verifying...")
        verification_started = time.perf_counter()
        verifier_calls += 1
        try:
            verification = verifier.verify(task.statement, normalized_proof)
        except Exception as error:  # noqa: BLE001
            verification_latency_ms = _elapsed_ms(verification_started)
            attempt = _verification_exception_attempt(
                attempt_index,
                generation,
                normalized_proof,
                verification_latency_ms,
                _elapsed_ms(attempt_started),
                error,
            )
        else:
            attempt = RetryAttempt(
                attempt_index=attempt_index,
                raw_model_output=generation.raw_output,
                reasoning_output=generation.reasoning_output,
                proof_output=generation.proof_output,
                normalized_proof=normalized_proof,
                verification_status=verification.status.value,
                verified=verification.verified,
                has_sorry=verification.has_sorry,
                lean_stdout=verification.stdout,
                lean_stderr=verification.stderr,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                generation_latency_ms=generation.latency_ms,
                verification_latency_ms=verification.elapsed_ms,
                total_attempt_latency_ms=_elapsed_ms(attempt_started),
                error=None,
            )
        attempts.append(attempt)
        status = (
            attempt.verification_status.upper()
            if attempt.verification_status is not None
            else "NOT RUN"
        )
        solved_suffix = " | solved" if attempt.verified else ""
        _report_progress(
            progress_callback,
            f"{prefix} | {status} | verify {attempt.verification_latency_ms} ms | "
            f"total {attempt.total_attempt_latency_ms} ms{solved_suffix}",
        )
        if attempt.verified:
            break

    attempt_tuple = tuple(attempts)
    final_attempt = attempt_tuple[-1]
    return RetryResult(
        theorem_id=task.theorem_id,
        statement=task.statement,
        strategy=RETRY_STRATEGY,
        model_alias=model_alias,
        model=model.model_name,
        generation_budget=max_attempts,
        attempts=attempt_tuple,
        final_verification_status=final_attempt.verification_status,
        solved=final_attempt.verified,
        generations_used=len(attempt_tuple),
        verifier_calls=verifier_calls,
        prompt_tokens=_complete_token_sum(attempt.prompt_tokens for attempt in attempt_tuple),
        completion_tokens=_complete_token_sum(
            attempt.completion_tokens for attempt in attempt_tuple
        ),
        generation_latency_ms=sum(attempt.generation_latency_ms for attempt in attempt_tuple),
        verification_latency_ms=sum(attempt.verification_latency_ms for attempt in attempt_tuple),
        total_latency_ms=_elapsed_ms(theorem_started),
    )


def _verification_exception_attempt(
    attempt_index: int,
    generation: GenerationResult,
    normalized_proof: str,
    verification_latency_ms: int,
    total_attempt_latency_ms: int,
    error: Exception,
) -> RetryAttempt:
    return RetryAttempt(
        attempt_index=attempt_index,
        raw_model_output=generation.raw_output,
        reasoning_output=generation.reasoning_output,
        proof_output=generation.proof_output,
        normalized_proof=normalized_proof,
        verification_status=VerificationStatus.EXECUTION_ERROR.value,
        verified=False,
        has_sorry=False,
        lean_stdout="",
        lean_stderr="",
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        generation_latency_ms=generation.latency_ms,
        verification_latency_ms=verification_latency_ms,
        total_attempt_latency_ms=total_attempt_latency_ms,
        error=f"verification_error: {type(error).__name__}: {error}",
    )


def _summarize_results(
    results: Sequence[RetryResult], max_attempts: int, output_path: Path
) -> RetrySummary:
    attempts = [attempt for result in results for attempt in result.attempts]
    solved_attempt_counts = [result.generations_used for result in results if result.solved]
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
        solved=len(solved_attempt_counts),
        generation_budget=max_attempts,
        total_attempts=len(attempts),
        average_attempts_per_theorem=len(attempts) / len(results),
        average_attempts_per_solved_theorem=(
            sum(solved_attempt_counts) / len(solved_attempt_counts)
            if solved_attempt_counts
            else None
        ),
        total_prompt_tokens=sum(prompt_tokens) if prompt_tokens else None,
        total_completion_tokens=sum(completion_tokens) if completion_tokens else None,
        average_prompt_tokens=(sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None),
        average_completion_tokens=(
            sum(completion_tokens) / len(completion_tokens) if completion_tokens else None
        ),
        prompt_token_attempts=len(prompt_tokens),
        completion_token_attempts=len(completion_tokens),
        average_generation_latency_ms=(
            sum(attempt.generation_latency_ms for attempt in attempts) / len(attempts)
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


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _report_progress(progress_callback: ProgressCallback | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)
