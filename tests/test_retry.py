from __future__ import annotations

import json
from pathlib import Path

from leanproof.lean import LeanResult, VerificationStatus
from leanproof.models import GenerationResult
from leanproof.strategies import (
    DEFAULT_MAX_ATTEMPTS,
    TheoremTask,
    default_retry_output_path,
    run_retry,
)
from scripts.run_retry import DEFAULT_RESULTS_DIRECTORY, PROJECT_ROOT, build_argument_parser


def test_first_attempt_verified_stops_after_one_generation(tmp_path: Path) -> None:
    statement = "example : True"
    model = FakeModel([generation("by\n  trivial", prompt_tokens=10, completion_tokens=2)])
    verifier = FakeVerifier([VerificationStatus.VERIFIED])
    output_path = tmp_path / "first.jsonl"

    summary = run_retry(
        [TheoremTask("first", statement)],
        model,
        verifier,
        output_path,
        model_alias="mock",
        max_attempts=4,
    )
    record = load_record(output_path)

    assert model.calls == [statement]
    assert len(verifier.calls) == 1
    assert summary.solved == 1
    assert summary.total_generations == 1
    assert record["strategy"] == "retry"
    assert record["generation_budget"] == 4
    assert record["generations_used"] == 1
    assert record["verifier_calls"] == 1
    assert record["final_verification_status"] == "verified"
    assert record["solved"] is True
    assert len(record["attempts"]) == 1


def test_rejected_then_verified_uses_same_clean_statement_only(tmp_path: Path) -> None:
    statement = "example : True"
    failed_proof = "by\n  exact nonexistent_theorem"
    model = FakeModel(
        [
            generation(
                failed_proof,
                native_reasoning="provider scratch",
                plan="Use the available theorem.",
            ),
            generation("by\n  trivial"),
        ]
    )
    verifier = FakeVerifier([VerificationStatus.REJECTED, VerificationStatus.VERIFIED])
    output_path = tmp_path / "second.jsonl"
    progress: list[str] = []

    summary = run_retry(
        [TheoremTask("second", statement)],
        model,
        verifier,
        output_path,
        model_alias="mock",
        max_attempts=4,
        progress_callback=progress.append,
    )
    record = load_record(output_path)

    assert model.calls == [statement, statement]
    assert verifier.calls == [(statement, failed_proof), (statement, "by\n  trivial")]
    assert summary.solved == 1
    assert record["generations_used"] == 2
    assert [attempt["verification_status"] for attempt in record["attempts"]] == [
        "rejected",
        "verified",
    ]
    assert record["attempts"][0]["raw_model_output"] == failed_proof
    assert record["attempts"][0]["native_reasoning_output"] == "provider scratch"
    assert record["attempts"][0]["plan_output"] == "Use the available theorem."
    assert record["attempts"][1]["raw_model_output"] == "by\n  trivial"
    assert any("generation 1/4" in message and "generating..." in message for message in progress)
    assert any("generation 1/4 | REJECTED" in message for message in progress)
    assert any(
        "generation 2/4 | VERIFIED" in message and "solved" in message for message in progress
    )


def test_all_unsolved_statuses_use_full_generation_budget(tmp_path: Path) -> None:
    statement = "example : False"
    model = FakeModel(
        [
            generation("by\n  sorry"),
            generation("by\n  exact nonexistent"),
            generation("by\n  exact False.elim (by contradiction)"),
        ]
    )
    verifier = FakeVerifier(
        [
            VerificationStatus.INCOMPLETE,
            VerificationStatus.REJECTED,
            VerificationStatus.TIMEOUT,
        ]
    )
    output_path = tmp_path / "exhausted.jsonl"

    summary = run_retry(
        [TheoremTask("exhausted", statement)],
        model,
        verifier,
        output_path,
        model_alias="mock",
        max_attempts=3,
    )
    record = load_record(output_path)

    assert len(model.calls) == 3
    assert len(verifier.calls) == 3
    assert summary.solved == 0
    assert record["solved"] is False
    assert record["generations_used"] == 3
    assert record["final_verification_status"] == "timeout"
    assert record["attempts"][0]["verification_status"] == "incomplete"
    assert record["attempts"][0]["verified"] is False
    assert record["attempts"][0]["has_sorry"] is True


def test_generation_error_is_recorded_without_consuming_generation(tmp_path: Path) -> None:
    statement = "example : True"
    model = FakeModel([RuntimeError("provider unavailable"), generation("by\n  trivial")])
    verifier = FakeVerifier([VerificationStatus.VERIFIED])
    output_path = tmp_path / "generation_error.jsonl"

    run_retry(
        [TheoremTask("generation-error", statement)],
        model,
        verifier,
        output_path,
        model_alias="mock",
        max_attempts=2,
    )
    record = load_record(output_path)

    assert model.calls == [statement]
    assert record["attempts"] == []
    assert record["request_attempts"][0]["error"]["type"] == "RuntimeError"
    assert record["terminal_status"] == "generation_error"
    assert record["generations_used"] == 0
    assert record["verifier_calls"] == 0


def test_output_format_failure_skips_lean_and_next_attempt_remains_independent(
    tmp_path: Path,
) -> None:
    statement = "example : True"
    model = FakeModel([generation("not a proof"), generation("by\n  trivial")])
    verifier = FakeVerifier([VerificationStatus.VERIFIED])
    output_path = tmp_path / "format_error.jsonl"

    run_retry(
        [TheoremTask("format-error", statement)],
        model,
        verifier,
        output_path,
        model_alias="mock",
        max_attempts=2,
    )
    record = load_record(output_path)

    assert model.calls == [statement, statement]
    assert verifier.calls == [(statement, "by\n  trivial")]
    assert record["generations_used"] == 2
    assert len(record["attempts"]) == 2
    assert record["attempts"][0]["normalized_proof"] == ""
    assert record["attempts"][0]["error"]["stage"] == "normalization"
    assert record["attempts"][1]["verification_status"] == "verified"


def test_result_and_summary_preserve_token_availability(tmp_path: Path) -> None:
    statement = "example : True"
    model = FakeModel(
        [
            generation("by\n  exact nonexistent_theorem", prompt_tokens=10, completion_tokens=2),
            generation("by\n  trivial", prompt_tokens=None, completion_tokens=3),
        ]
    )
    verifier = FakeVerifier([VerificationStatus.REJECTED, VerificationStatus.VERIFIED])
    output_path = tmp_path / "tokens.jsonl"

    summary = run_retry(
        [TheoremTask("tokens", statement)],
        model,
        verifier,
        output_path,
        model_alias="configured_alias",
        max_attempts=2,
        dataset="data/retry.jsonl",
        reasoning_mode="prompted",
        connect_timeout_seconds=11,
        read_timeout_seconds=450,
        write_timeout_seconds=31,
        pool_timeout_seconds=12,
        verification_timeout_seconds=75,
    )
    record = load_record(output_path)

    assert record["model_alias"] == "configured_alias"
    assert record["model"] == "provider-model"
    assert record["dataset"] == "data/retry.jsonl"
    assert record["reasoning_mode"] == "prompted"
    assert record["connect_timeout_seconds"] == 11
    assert record["read_timeout_seconds"] == 450
    assert record["write_timeout_seconds"] == 31
    assert record["pool_timeout_seconds"] == 12
    assert record["verification_timeout_seconds"] == 75
    assert record["prompt_tokens"] is None
    assert record["completion_tokens"] == 5
    assert summary.total_prompt_tokens == 10
    assert summary.total_completion_tokens == 5
    assert summary.average_prompt_tokens == 10.0
    assert summary.average_completion_tokens == 2.5
    assert summary.prompt_token_generations == 1
    assert summary.completion_token_generations == 2
    assert summary.average_generations_per_theorem == 2.0
    assert summary.average_generations_per_solved_theorem == 2.0
    assert model.reasoning_modes == ["prompted", "prompted"]


def test_verifier_execution_error_terminates_theorem_immediately(tmp_path: Path) -> None:
    statement = "example : True"
    model = FakeModel([generation("by\n  trivial"), generation("by\n  trivial")])
    verifier = FakeVerifier([VerificationStatus.EXECUTION_ERROR, VerificationStatus.VERIFIED])
    output_path = tmp_path / "execution_error.jsonl"

    run_retry(
        [TheoremTask("execution-error", statement)],
        model,
        verifier,
        output_path,
        model_alias="mock",
        max_attempts=2,
    )
    record = load_record(output_path)

    assert len(model.calls) == 1
    assert len(verifier.calls) == 1
    assert record["terminal_status"] == "verifier_execution_error"
    assert record["final_verification_status"] == "execution_error"
    assert record["generations_used"] == 1
    assert record["verifier_calls"] == 1


def test_raised_verifier_exception_terminates_theorem_immediately(tmp_path: Path) -> None:
    model = FakeModel([generation("by\n  trivial"), generation("by\n  trivial")])
    output_path = tmp_path / "raised_execution_error.jsonl"

    run_retry(
        [TheoremTask("raised", "example : True")],
        model,
        RaisingVerifier(),
        output_path,
        model_alias="mock",
        max_attempts=2,
    )
    record = load_record(output_path)

    assert len(model.calls) == 1
    assert record["terminal_status"] == "verifier_execution_error"
    assert record["attempts"][0]["error"]["stage"] == "verification"


def test_default_path_and_cli_default_identify_retry_budget(tmp_path: Path) -> None:
    path = default_retry_output_path("data/smoke.jsonl", "minimax", 4, tmp_path)
    args = build_argument_parser().parse_args(
        ["--dataset", "data/smoke.jsonl", "--model", "minimax"]
    )

    assert path.parent == tmp_path
    assert path.name.startswith("retry_smoke_minimax_k4_")
    assert path.suffix == ".jsonl"
    assert args.max_attempts is None
    assert DEFAULT_MAX_ATTEMPTS == 4
    assert DEFAULT_RESULTS_DIRECTORY == PROJECT_ROOT / "artifacts" / "retry" / "results"


def test_cli_preserves_explicit_output_path() -> None:
    args = build_argument_parser().parse_args(
        [
            "--dataset",
            "data/smoke.jsonl",
            "--model",
            "minimax",
            "--output",
            "custom/retry.jsonl",
        ]
    )

    assert args.output == "custom/retry.jsonl"


class FakeModel:
    model_name = "provider-model"

    def __init__(self, responses: list[GenerationResult | Exception]) -> None:
        self._responses = iter(responses)
        self.calls: list[str] = []
        self.reasoning_modes: list[str] = []

    def generate_proof(self, statement: str, *, reasoning_mode: str = "none") -> GenerationResult:
        self.calls.append(statement)
        self.reasoning_modes.append(reasoning_mode)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeVerifier:
    def __init__(self, statuses: list[VerificationStatus]) -> None:
        self._statuses = iter(statuses)
        self.calls: list[tuple[str, str]] = []

    def verify(self, statement: str, proof: str) -> LeanResult:
        self.calls.append((statement, proof))
        status = next(self._statuses)
        return LeanResult(
            status=status,
            stdout=(
                f"diagnostic: {status.value}" if status is not VerificationStatus.VERIFIED else ""
            ),
            stderr="",
            elapsed_ms=7,
        )


class RaisingVerifier:
    def verify(self, statement: str, proof: str) -> LeanResult:
        raise OSError("could not launch Lean")


def generation(
    proof: str,
    *,
    native_reasoning: str | None = None,
    plan: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> GenerationResult:
    return GenerationResult(
        raw_output=proof,
        proof_output=proof,
        native_reasoning_output=native_reasoning,
        latency_ms=5,
        plan_output=plan,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def load_record(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
