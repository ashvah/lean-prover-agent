"""Run a small manual acceptance check for the Phase 0 verifier."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from leanproof import LeanVerifier, VerificationStatus


@dataclass(frozen=True)
class Example:
    label: str
    statement: str
    proof: str
    expected_status: VerificationStatus


def main() -> int:
    verifier = LeanVerifier(project_root=PROJECT_ROOT)
    examples = (
        Example(
            "verified",
            "example (p : Prop) (h : p) : p",
            "by\n  exact h",
            VerificationStatus.VERIFIED,
        ),
        Example(
            "incomplete",
            "example : False",
            "by\n  sorry",
            VerificationStatus.INCOMPLETE,
        ),
        Example(
            "rejected",
            "example : False",
            "by\n  exact nonexistent_theorem",
            VerificationStatus.REJECTED,
        ),
    )

    unexpected = False
    for example in examples:
        result = verifier.verify(example.statement, example.proof)
        print(f"[{example.label}]")
        print(f"status: {result.status.value}")
        print(f"verified: {result.verified}")
        print(f"accepted: {result.accepted}")
        print(f"has_sorry: {result.has_sorry}")
        if result.status is not example.expected_status:
            unexpected = True
            print(f"expected_status: {example.expected_status.value}")
        print()

    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
