"""Canonical dataset records shared by source adapters and experiment tooling."""

from __future__ import annotations

from dataclasses import dataclass, field

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DifficultyEstimate:
    """One deterministic dataset-relative structural difficulty estimate."""

    score: float
    bucket: str
    method: str = "static_v1"

    def to_dict(self) -> dict[str, object]:
        return {"score": self.score, "bucket": self.bucket, "method": self.method}


@dataclass(frozen=True)
class ReferenceTrajectoryStep:
    """One source tactic transition in deterministic theorem-local order."""

    step: int
    state_before: str | None
    tactic: str | None
    state_after: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "state_before": self.state_before,
            "tactic": self.tactic,
            "state_after": self.state_after,
        }


@dataclass(frozen=True)
class CanonicalTheorem:
    """Experiment-independent theorem data with optional source metadata."""

    id: str
    source: str
    source_id: str
    statement: str
    informal_statement: str | None = None
    answer: str | None = None
    source_status: str | None = None
    reference_trajectory: tuple[ReferenceTrajectoryStep, ...] = ()
    reference_proof: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    features: dict[str, int | None] = field(default_factory=dict)
    difficulty: DifficultyEstimate | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Serialize fields in a stable, human-inspectable order."""

        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "source": self.source,
            "source_id": self.source_id,
            "statement": self.statement,
            "informal_statement": self.informal_statement,
            "answer": self.answer,
            "source_status": self.source_status,
            "reference_trajectory": [step.to_dict() for step in self.reference_trajectory],
            "reference_proof": self.reference_proof,
            "metadata": self.metadata,
            "features": self.features,
            "difficulty": self.difficulty.to_dict() if self.difficulty is not None else None,
        }
