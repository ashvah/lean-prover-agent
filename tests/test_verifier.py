from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from leanproof import LeanVerifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def verifier() -> LeanVerifier:
    return LeanVerifier(project_root=PROJECT_ROOT)


def test_correct_proof(verifier: LeanVerifier) -> None:
    result = verifier.verify(
        statement="example (p : Prop) (h : p) : p",
        proof="by\n  exact h",
    )

    assert result.success


def test_arithmetic_proof(verifier: LeanVerifier) -> None:
    result = verifier.verify(
        statement="example (x : ℝ) : (x + 1)^2 = x^2 + 2*x + 1",
        proof="by\n  ring",
    )

    assert result.success


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
def test_invalid_proofs_fail(
    verifier: LeanVerifier, statement: str, proof: str
) -> None:
    result = verifier.verify(statement=statement, proof=proof)

    assert not result.success
    assert result.stdout or result.stderr


def test_multiple_calls_are_isolated(verifier: LeanVerifier) -> None:
    valid_before = verifier.verify(
        "example (p : Prop) (h : p) : p", "by exact h"
    )
    invalid = verifier.verify(
        "example (p : Prop) (h : p) : p", "by exact nonexistent_theorem"
    )
    valid_after = verifier.verify(
        "example (p : Prop) (h : p) : p", "by exact h"
    )

    assert valid_before.success
    assert not invalid.success
    assert valid_after.success


def test_timeout_returns_failure(verifier: LeanVerifier) -> None:
    with patch(
        "leanproof.verifier.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="lake", timeout=0.01),
    ):
        result = verifier.verify("example : True", "by trivial")

    assert not result.success
    assert "timed out" in result.stderr
