"""Deterministic lexical and structural theorem feature approximations."""

from __future__ import annotations

import re

from leanproof.datasets.schema import CanonicalTheorem

FEATURE_NAMES = (
    "statement_chars",
    "statement_lines",
    "statement_tokens",
    "num_binders",
    "num_hypotheses",
    "logical_complexity",
    "reference_proof_chars",
    "reference_proof_lines",
    "reference_proof_tokens",
)
DIFFICULTY_FEATURE_NAMES = (
    "statement_tokens",
    "num_binders",
    "num_hypotheses",
    "logical_complexity",
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_']*|\d+(?:\.\d+)?|[^\s]")
_BINDER_PATTERN = re.compile(r"[({\[][^(){}\[\]]*?:[^(){}\[\]]*?[)}\]]")
_HYPOTHESIS_NAME_PATTERN = re.compile(r"[({\[]\s*([^:]+?)\s*:")
_LOGICAL_PATTERNS = (
    re.compile(r"∀|∃|→|↔|∧|∨|¬"),
    re.compile(r"(?<![:<>=!])=(?!=)|≠|≤|≥|<|>"),
    re.compile(r"\b(?:Set|Finset|Membership|Subset)\b"),
)


def extract_features(theorem: CanonicalTheorem) -> dict[str, int | None]:
    """Extract reproducible heuristics without claiming Lean AST precision."""

    statement = theorem.statement
    proof = theorem.reference_proof
    binders = _BINDER_PATTERN.findall(statement)
    hypothesis_names = _HYPOTHESIS_NAME_PATTERN.findall(statement)
    return {
        "statement_chars": len(statement),
        "statement_lines": len(statement.splitlines()) or 1,
        "statement_tokens": _token_count(statement),
        "num_binders": len(binders),
        "num_hypotheses": sum(_looks_like_hypothesis(names) for names in hypothesis_names),
        "logical_complexity": sum(len(pattern.findall(statement)) for pattern in _LOGICAL_PATTERNS),
        "reference_proof_chars": len(proof) if proof is not None else None,
        "reference_proof_lines": len(proof.splitlines()) if proof is not None else None,
        "reference_proof_tokens": _token_count(proof) if proof is not None else None,
    }


def _token_count(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))


def _looks_like_hypothesis(names: str) -> bool:
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_']*", names)
    return any(identifier.startswith("h") for identifier in identifiers)
