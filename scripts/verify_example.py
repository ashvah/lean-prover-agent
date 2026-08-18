"""Run a small manual acceptance check for the Phase 0 verifier."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from leanproof import LeanVerifier  # noqa: E402


@dataclass(frozen=True)
class Example:
    label: str
    statement: str
    proof: str
    expected_success: bool


def main() -> int:
    verifier = LeanVerifier(project_root=PROJECT_ROOT)
    examples = (
        Example(
            "valid proof",
            "example (p : Prop) (h : p) : p",
            "by\n  exact h",
            True,
        ),
        Example(
            "arithmetic proof",
            "example (x : ℝ) : (x + 1)^2 = x^2 + 2*x + 1",
            "by\n  ring",
            True,
        ),
        Example(
            "unfinished proof",
            "example (p q : Prop) (hp : p) : p ∧ q",
            "by\n  constructor\n  · exact hp",
            False,
        ),
        Example(
            "unknown identifier",
            "example (p : Prop) (h : p) : p",
            "by\n  exact nonexistent_theorem",
            False,
        ),
        Example(
            "parse error",
            "example (p : Prop) (h : p) : p",
            "by\n  exact (",
            False,
        ),
    )

    unexpected = False
    for example in examples:
        result = verifier.verify(example.statement, example.proof)
        if result.success == example.expected_success:
            status = "PASS" if result.success else "FAIL"
            print(f"[{status}] {example.label}")
        else:
            unexpected = True
            outcome = "PASS" if result.success else "FAIL"
            print(f"[UNEXPECTED {outcome}] {example.label}")
            feedback = (result.stdout + result.stderr).strip()
            if feedback:
                print(feedback)

    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
