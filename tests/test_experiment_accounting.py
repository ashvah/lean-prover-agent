from __future__ import annotations

import json
from pathlib import Path

from leanproof.lean import LeanResult, VerificationStatus
from leanproof.models import ErrorDetails, GenerationRequestError, GenerationResult
from leanproof.strategies import TheoremTask, run_one_shot, run_retry
from leanproof.strategies.common import TheoremTask as CommonTheoremTask
from leanproof.strategies.one_shot import TheoremTask as OneShotFacadeTheoremTask


def test_shared_task_type_preserves_existing_strategy_imports() -> None:
    assert TheoremTask is CommonTheoremTask
    assert OneShotFacadeTheoremTask is CommonTheoremTask


def test_retry_separates_transport_requests_from_completed_generations(tmp_path: Path) -> None:
    statement = "example : True"
    model = FakeModel(
        [
            transport_error("APITimeoutError", "ReadTimeout", 101),
            transport_error("APITimeoutError", "ReadTimeout", 102),
            generation("by\n  exact nonexistent", latency_ms=11),
            transport_error("APIConnectionError", "ConnectTimeout", 103),
            generation("by\n  trivial", latency_ms=12),
        ]
    )
    verifier = FakeVerifier([VerificationStatus.REJECTED, VerificationStatus.VERIFIED])
    output_path = tmp_path / "accounting.jsonl"

    summary = run_retry(
        [TheoremTask("accounting", statement)],
        model,
        verifier,
        output_path,
        model_alias="mock",
        max_attempts=2,
        max_transport_retries=2,
    )
    record = load_record(output_path)

    assert record["api_requests"] == 5
    assert record["request_failures"] == 3
    assert record["transport_failures"] == 3
    assert record["generations_used"] == 2
    assert record["verifier_calls"] == 2
    assert record["solved"] is True
    assert record["terminal_status"] == "verified"
    assert len(record["request_attempts"]) == 5
    assert len(record["attempts"]) == record["generations_used"]
    assert [attempt["request_index"] for attempt in record["attempts"]] == [3, 5]
    assert [request["elapsed_ms"] for request in record["request_attempts"]] == [
        101,
        102,
        11,
        103,
        12,
    ]
    assert record["generation_latency_ms"] == 23
    assert summary.total_api_requests == 5
    assert summary.total_generations == 2
    assert summary.total_transport_failures == 3


def test_manual_transport_accounting_sequence(tmp_path: Path) -> None:
    model = FakeModel(
        [
            transport_error("APITimeoutError", "ReadTimeout", 31),
            transport_error("APITimeoutError", "ReadTimeout", 32),
            generation("by\n  exact nonexistent"),
            generation("by\n  trivial"),
        ]
    )
    output_path = tmp_path / "manual-sequence.jsonl"

    run_retry(
        [TheoremTask("manual-sequence", "example : True")],
        model,
        FakeVerifier([VerificationStatus.REJECTED, VerificationStatus.VERIFIED]),
        output_path,
        model_alias="mock",
        max_attempts=2,
        max_transport_retries=2,
    )
    record = load_record(output_path)

    assert record["api_requests"] == 4
    assert record["transport_failures"] == 2
    assert record["generations_used"] == 2
    assert record["verifier_calls"] == 2
    assert record["terminal_status"] == "verified"
    assert record["final_verification_status"] == "verified"


def test_transport_exhaustion_preserves_last_real_verification_status(tmp_path: Path) -> None:
    model = FakeModel(
        [
            generation("by\n  exact nonexistent"),
            transport_error("APIConnectionError", "ConnectTimeout", 20),
            transport_error("APIConnectionError", "ConnectTimeout", 21),
            transport_error("APIConnectionError", "ConnectTimeout", 22),
        ]
    )
    output_path = tmp_path / "transport_exhausted.jsonl"

    run_retry(
        [TheoremTask("transport", "example : True")],
        model,
        FakeVerifier([VerificationStatus.REJECTED]),
        output_path,
        model_alias="mock",
        max_attempts=2,
        max_transport_retries=2,
    )
    record = load_record(output_path)

    assert record["api_requests"] == 4
    assert record["generations_used"] == 1
    assert record["verifier_calls"] == 1
    assert record["terminal_status"] == "transport_retry_exhausted"
    assert record["final_verification_status"] == "rejected"
    assert record["attempts"][0]["verification_status"] == "rejected"
    assert record["error"]["cause_type"] == "ConnectTimeout"


def test_one_shot_transport_retry_does_not_consume_generation_budget(tmp_path: Path) -> None:
    model = FakeModel(
        [
            transport_error("APITimeoutError", "ReadTimeout", 40),
            generation("by\n  trivial", latency_ms=9),
        ]
    )
    output_path = tmp_path / "one_shot.jsonl"

    summary = run_one_shot(
        [TheoremTask("one", "example : True")],
        model,
        FakeVerifier([VerificationStatus.VERIFIED]),
        output_path,
        model_alias="mock",
        max_transport_retries=2,
    )
    record = load_record(output_path)

    assert record["generation_budget"] == 1
    assert record["api_requests"] == 2
    assert record["request_failures"] == 1
    assert record["generations_used"] == 1
    assert record["verified"] is True
    assert summary.total_api_requests == 2
    assert summary.total_generations == 1


def test_transient_api_failure_uses_request_retry_without_consuming_generation(
    tmp_path: Path,
) -> None:
    model = FakeModel(
        [
            transient_api_error(429, 21),
            generation("by\n  trivial", latency_ms=8),
        ]
    )
    output_path = tmp_path / "transient.jsonl"

    summary = run_one_shot(
        [TheoremTask("transient", "example : True")],
        model,
        FakeVerifier([VerificationStatus.VERIFIED]),
        output_path,
        model_alias="mock",
        max_transport_retries=1,
    )
    record = load_record(output_path)

    assert record["api_requests"] == 2
    assert record["request_failures"] == 1
    assert record["transport_failures"] == 0
    assert record["transient_api_failures"] == 1
    assert record["generations_used"] == 1
    assert record["request_attempts"][0]["error"]["status_code"] == 429
    assert summary.total_transient_api_failures == 1


def test_transient_api_retry_exhaustion_has_zero_generations(tmp_path: Path) -> None:
    output_path = tmp_path / "transient_exhausted.jsonl"

    run_retry(
        [TheoremTask("transient-exhausted", "example : True")],
        FakeModel([transient_api_error(503, 11), transient_api_error(503, 12)]),
        FakeVerifier([]),
        output_path,
        model_alias="mock",
        max_attempts=2,
        max_transport_retries=1,
    )
    record = load_record(output_path)

    assert record["api_requests"] == 2
    assert record["transient_api_failures"] == 2
    assert record["generations_used"] == 0
    assert record["terminal_status"] == "request_retry_exhausted"
    assert record["error"]["status_code"] == 503


def test_terminal_api_status_is_not_retried(tmp_path: Path) -> None:
    output_path = tmp_path / "terminal_status.jsonl"

    run_one_shot(
        [TheoremTask("terminal", "example : True")],
        FakeModel([terminal_api_error(401), generation("by\n  trivial")]),
        FakeVerifier([]),
        output_path,
        model_alias="mock",
        max_transport_retries=2,
    )
    record = load_record(output_path)

    assert record["api_requests"] == 1
    assert record["generations_used"] == 0
    assert record["terminal_status"] == "generation_error"
    assert record["error"]["status_code"] == 401


def test_one_shot_transport_exhaustion_has_zero_generations(tmp_path: Path) -> None:
    output_path = tmp_path / "one_shot_transport_exhausted.jsonl"
    model = FakeModel(
        [
            transport_error("APITimeoutError", "ReadTimeout", 10),
            transport_error("APITimeoutError", "ReadTimeout", 11),
        ]
    )

    run_one_shot(
        [TheoremTask("one-timeout", "example : True")],
        model,
        FakeVerifier([]),
        output_path,
        model_alias="mock",
        max_transport_retries=1,
    )
    record = load_record(output_path)

    assert record["api_requests"] == 2
    assert record["request_failures"] == 2
    assert record["transport_failures"] == 2
    assert record["generations_used"] == 0
    assert record["verifier_calls"] == 0
    assert record["terminal_status"] == "transport_retry_exhausted"


def test_verification_exception_counts_generation_and_verifier_call(tmp_path: Path) -> None:
    output_path = tmp_path / "verification_error.jsonl"

    run_retry(
        [TheoremTask("verify-error", "example : True")],
        FakeModel([generation("by\n  trivial")]),
        RaisingVerifier(),
        output_path,
        model_alias="mock",
        max_attempts=1,
    )
    record = load_record(output_path)

    assert record["generations_used"] == 1
    assert record["verifier_calls"] == 1
    assert record["attempts"][0]["verification_status"] == "execution_error"
    assert record["attempts"][0]["error"]["stage"] == "verification"
    assert record["terminal_status"] == "verifier_execution_error"


def test_unexpected_generation_error_is_terminal_and_does_not_consume_budget(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "unexpected.jsonl"

    run_retry(
        [TheoremTask("unexpected", "example : True")],
        FakeModel([RuntimeError("bad response")]),
        FakeVerifier([]),
        output_path,
        model_alias="mock",
        max_attempts=4,
    )
    record = load_record(output_path)

    assert record["api_requests"] == 1
    assert record["generations_used"] == 0
    assert record["attempts"] == []
    assert record["terminal_status"] == "generation_error"
    assert record["error"] == {
        "stage": "generation",
        "type": "RuntimeError",
        "cause_type": None,
        "message": "bad response",
    }


class FakeModel:
    model_name = "provider-model"

    def __init__(self, responses: list[GenerationResult | Exception]) -> None:
        self._responses = iter(responses)

    def generate_proof(self, statement: str, *, reasoning_mode: str = "none") -> GenerationResult:
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeVerifier:
    def __init__(self, statuses: list[VerificationStatus]) -> None:
        self._statuses = iter(statuses)

    def verify(self, statement: str, proof: str) -> LeanResult:
        return LeanResult(status=next(self._statuses), stdout="", stderr="", elapsed_ms=7)


class RaisingVerifier:
    def verify(self, statement: str, proof: str) -> LeanResult:
        raise OSError("Lean unavailable")


def generation(proof: str, *, latency_ms: int = 5) -> GenerationResult:
    return GenerationResult(
        raw_output=proof,
        proof_output=proof,
        native_reasoning_output=None,
        latency_ms=latency_ms,
        prompt_tokens=10,
        completion_tokens=2,
    )


def transport_error(error_type: str, cause_type: str, elapsed_ms: int) -> GenerationRequestError:
    return GenerationRequestError(
        ErrorDetails(
            stage="generation_request",
            type=error_type,
            cause_type=cause_type,
            message="provider transport failed",
        ),
        retryable=True,
        transport=True,
        elapsed_ms=elapsed_ms,
    )


def transient_api_error(status_code: int, elapsed_ms: int) -> GenerationRequestError:
    return GenerationRequestError(
        ErrorDetails(
            stage="generation_request",
            type="APIStatusError",
            cause_type=None,
            message="transient provider response",
            status_code=status_code,
        ),
        retryable=True,
        transport=False,
        transient_api=True,
        elapsed_ms=elapsed_ms,
    )


def terminal_api_error(status_code: int) -> GenerationRequestError:
    return GenerationRequestError(
        ErrorDetails(
            stage="generation_request",
            type="APIStatusError",
            cause_type=None,
            message="terminal provider response",
            status_code=status_code,
        ),
        retryable=False,
        transport=False,
        transient_api=False,
        elapsed_ms=7,
    )


def load_record(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
