"""Model-independent static theorem difficulty scoring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from leanproof.datasets.features import DIFFICULTY_FEATURE_NAMES
from leanproof.datasets.schema import CanonicalTheorem, DifficultyEstimate

DIFFICULTY_METHOD = "static_v1"
_EASY_THRESHOLD = 1.0 / 3.0
_HARD_THRESHOLD = 2.0 / 3.0


def assign_static_difficulty(records: Sequence[CanonicalTheorem]) -> list[CanonicalTheorem]:
    """Assign equal-weight feature percentile scores and quantile-style buckets."""

    if not records:
        return []
    feature_values = {
        name: [_required_feature(record, name) for record in records]
        for name in DIFFICULTY_FEATURE_NAMES
    }
    ranked_records: list[CanonicalTheorem] = []
    for index, record in enumerate(records):
        ranks = [
            _midrank_percentile(feature_values[name], feature_values[name][index])
            for name in DIFFICULTY_FEATURE_NAMES
        ]
        score = round(sum(ranks) / len(ranks), 6)
        ranked_records.append(
            replace(
                record,
                difficulty=DifficultyEstimate(
                    score=score,
                    bucket=_bucket_for_score(score),
                    method=DIFFICULTY_METHOD,
                ),
            )
        )
    return ranked_records


def _required_feature(record: CanonicalTheorem, name: str) -> int:
    value = record.features.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Missing non-negative difficulty feature '{name}' for {record.id}")
    return value


def _midrank_percentile(values: Sequence[int], target: int) -> float:
    if len(values) == 1:
        return 0.5
    lower_count = sum(value < target for value in values)
    tied_count = sum(value == target for value in values)
    return (lower_count + (tied_count - 1) / 2.0) / (len(values) - 1)


def _bucket_for_score(score: float) -> str:
    if score < _EASY_THRESHOLD:
        return "easy"
    if score < _HARD_THRESHOLD:
        return "medium"
    return "hard"
