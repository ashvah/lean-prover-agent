from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from leanproof import LeanVerifier, VerificationStatus

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def verifier() -> LeanVerifier:
    return LeanVerifier(project_root=PROJECT_ROOT)


def test_correct_proof(verifier: LeanVerifier) -> None:
    result = verifier.verify(
        statement="example (p : Prop) (h : p) : p",
        proof="by\n  exact h",
    )

    assert result.status is VerificationStatus.VERIFIED
    assert result.verified
    assert result.accepted
    assert not result.has_sorry
    assert result.success


def test_arithmetic_proof(verifier: LeanVerifier) -> None:
    result = verifier.verify(
        statement="example (x : ℝ) : (x + 1)^2 = x^2 + 2*x + 1",
        proof="by\n  ring",
    )

    assert result.status is VerificationStatus.VERIFIED
    assert result.verified


@pytest.mark.parametrize(
    "proof",
    ["by\n  sorry", "by\n  exact sorry", "by\n  admit"],
    ids=["sorry", "exact-sorry", "admit"],
)
def test_incomplete_proofs_are_accepted_but_not_verified(
    verifier: LeanVerifier, proof: str
) -> None:
    result = verifier.verify(statement="example : False", proof=proof)

    assert result.status is VerificationStatus.INCOMPLETE
    assert not result.verified
    assert result.accepted
    assert result.has_sorry
    assert not result.success


@pytest.mark.parametrize(
    ("statement", "proof"),
    [
        (
            "example (p q : Prop) (hp : p) : p ∧ q",
            "by\n  constructor\n  · exact hp",
        ),
        (
            "example (p : Prop) (h : p) : p",
            "by\n  exact nonexistent_theorem",
        ),
        (
            "example (p : Prop) (h : p) : p",
            "by\n  exact (",
        ),
        (
            "example : False",
            "by\n  definitely_not_a_tactic",
        ),
    ],
    ids=["unfinished-proof", "unknown-identifier", "parse-error", "invalid-tactic"],
)
def test_invalid_proofs_fail(verifier: LeanVerifier, statement: str, proof: str) -> None:
    result = verifier.verify(statement=statement, proof=proof)

    assert result.status is VerificationStatus.REJECTED
    assert not result.verified
    assert not result.accepted
    assert not result.has_sorry
    assert result.stdout or result.stderr


def test_multiple_calls_are_isolated(verifier: LeanVerifier) -> None:
    valid_before = verifier.verify("example (p : Prop) (h : p) : p", "by exact h")
    invalid = verifier.verify("example (p : Prop) (h : p) : p", "by exact nonexistent_theorem")
    valid_after = verifier.verify("example (p : Prop) (h : p) : p", "by exact h")

    assert valid_before.status is VerificationStatus.VERIFIED
    assert invalid.status is VerificationStatus.REJECTED
    assert valid_after.status is VerificationStatus.VERIFIED


def test_timeout_returns_failure(verifier: LeanVerifier) -> None:
    with patch(
        "leanproof.verifier.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="lake", timeout=0.01),
    ):
        result = verifier.verify("example : True", "by trivial")

    assert result.status is VerificationStatus.TIMEOUT
    assert not result.success
    assert "timed out" in result.stderr


def test_process_start_failure_returns_execution_error(verifier: LeanVerifier) -> None:
    with patch("leanproof.verifier.subprocess.run", side_effect=OSError("access denied")):
        result = verifier.verify("example : True", "by trivial")

    assert result.status is VerificationStatus.EXECUTION_ERROR
    assert not result.verified
    assert not result.accepted
    assert "Failed to start Lean verifier" in result.stderr


def test_nonzero_process_without_lean_diagnostic_is_execution_error(
    verifier: LeanVerifier,
) -> None:
    completed = subprocess.CompletedProcess(
        args=["lake"],
        returncode=1,
        stdout="",
        stderr="Lake environment failed before Lean diagnostics were available.",
    )
    with patch("leanproof.verifier.subprocess.run", return_value=completed):
        result = verifier.verify("example : True", "by trivial")

    assert result.status is VerificationStatus.EXECUTION_ERROR
    assert not result.verified
