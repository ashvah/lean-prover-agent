from __future__ import annotations

import json
from pathlib import Path

import pyarrow as arrow
import pytest
from pyarrow import parquet

from leanproof.datasets import (
    CanonicalTheorem,
    DatasetPipelineError,
    ReferenceTrajectoryStep,
    assign_static_difficulty,
    extract_features,
    load_canonical_records,
    prepare_dataset,
    sample_canonical_records,
)
from leanproof.datasets.adapters import LeanWorkbookAdapter, LeanWorkbookSchemaError
from leanproof.lean import LeanResult, VerificationStatus
from leanproof.models import GenerationResult
from leanproof.strategies import load_dataset, run_one_shot, run_retry


def test_lean_workbook_pipeline_maps_sanitizes_deduplicates_and_is_stable(
    tmp_path: Path,
) -> None:
    input_path = write_workbook_fixture(tmp_path / "workbook.parquet")
    first_output = tmp_path / "first" / "theorems.jsonl"
    first_manifest = tmp_path / "first" / "manifest.json"
    second_output = tmp_path / "second" / "theorems.jsonl"
    second_manifest = tmp_path / "second" / "manifest.json"

    first = prepare_dataset(
        source="lean_workbook",
        input_path=input_path,
        output_path=first_output,
        manifest_path=first_manifest,
    )
    second = prepare_dataset(
        source="lean_workbook",
        input_path=input_path,
        output_path=second_output,
        manifest_path=second_manifest,
    )
    records = load_canonical_records(first_output)
    manifest = json.loads(first_manifest.read_text(encoding="utf-8"))

    assert first.source_tactic_rows_scanned == 9
    assert first.raw_tactic_rows == 9
    assert first.theorem_groups == 6
    assert first.total_trajectory_steps == 9
    assert first.proved_theorems == 4
    assert first.disproved_theorems == 1
    assert first.invalid_groups == 1
    assert first.duplicate_theorems == 1
    assert first.final_proving_theorems == 3
    assert first.invalid_reasons == {"natural_language_statement_conflict": 1}
    assert sum(first.bucket_counts.values()) == 3
    assert (
        first.source_tactic_rows_scanned,
        first.raw_tactic_rows,
        first.theorem_groups,
        first.total_trajectory_steps,
        first.proved_theorems,
        first.disproved_theorems,
        first.invalid_groups,
        first.duplicate_theorems,
        first.final_proving_theorems,
        first.bucket_counts,
        first.invalid_reasons,
    ) == (
        second.source_tactic_rows_scanned,
        second.raw_tactic_rows,
        second.theorem_groups,
        second.total_trajectory_steps,
        second.proved_theorems,
        second.disproved_theorems,
        second.invalid_groups,
        second.duplicate_theorems,
        second.final_proving_theorems,
        second.bucket_counts,
        second.invalid_reasons,
    )
    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert records[0]["statement"] == "theorem workbook_a : True"
    assert records[0]["source_status"] == "proved"
    assert records[0]["reference_proof"] is None
    assert records[0]["reference_trajectory"] == [
        {
            "step": 0,
            "state_before": "⊢ True",
            "tactic": "apply True.intro",
            "state_after": "⊢ True",
        },
        {
            "step": 1,
            "state_before": "⊢ True",
            "tactic": "exact True.intro",
            "state_after": "no goals",
        },
        {
            "step": 2,
            "state_before": "no goals",
            "tactic": "done",
            "state_after": "no goals",
        },
    ]
    assert records[0]["metadata"]["source_formal_statement"].endswith(":= by sorry  ")
    assert records[0]["metadata"]["category"] == "logic"
    assert records[0]["features"]["reference_trajectory_steps"] == 3
    assert records[0]["features"]["reference_tactic_count"] == 3
    assert all(record["source_status"] == "proved" for record in records)
    assert manifest["lean_validation"] == {"performed": False}
    assert manifest["raw_input"] == "workbook.parquet"
    assert manifest["difficulty_method"] == "static_v1"
    assert manifest["theorem_groups"] == 6
    assert manifest["disproved_theorems"] == 1
    assert manifest["duplicate_theorems"] == 1


def test_lean_workbook_adapter_requires_formal_statement_column(tmp_path: Path) -> None:
    input_path = tmp_path / "missing.parquet"
    parquet.write_table(
        arrow.Table.from_pylist(
            [
                {
                    "id": "missing-formal",
                    "status": "proved",
                    "tactic": "trivial",
                    "state_before": "⊢ True",
                    "state_after": "no goals",
                }
            ]
        ),
        input_path,
    )

    with pytest.raises(LeanWorkbookSchemaError, match="formal statement column"):
        LeanWorkbookAdapter(input_path)


def test_adapter_groups_ids_stably_and_preserves_trajectory_order(tmp_path: Path) -> None:
    input_path = write_workbook_fixture(tmp_path / "workbook.parquet")

    first_adapter = LeanWorkbookAdapter(input_path)
    second_adapter = LeanWorkbookAdapter(input_path)
    first_groups = first_adapter.load_groups().theorem_groups
    second_groups = second_adapter.load_groups().theorem_groups
    first = [first_adapter.map_group(group) for group in first_groups[:2]]
    second = [second_adapter.map_group(group) for group in second_groups[:2]]

    assert [record.id for record in first] == [record.id for record in second]
    assert first[0].source_id == "source-a"
    assert len(first[0].reference_trajectory) == 3
    assert [step.tactic for step in first[0].reference_trajectory] == [
        "apply True.intro",
        "exact True.intro",
        "done",
    ]
    assert first[0].reference_proof is None


def test_adapter_accounts_for_missing_source_id(tmp_path: Path) -> None:
    input_path = tmp_path / "no-id.parquet"
    parquet.write_table(
        arrow.Table.from_pylist(
            [
                {
                    "id": None,
                    "status": "proved",
                    "natural_language_statement": "Show True.",
                    "formal_statement": "theorem no_id : True := by sorry",
                    "tactic": "trivial",
                    "state_before": "⊢ True",
                    "state_after": "no goals",
                }
            ]
        ),
        input_path,
    )

    grouped = LeanWorkbookAdapter(input_path).load_groups()

    assert grouped.source_tactic_rows_scanned == 1
    assert grouped.theorem_groups == ()
    assert grouped.invalid_reasons == {"source_id_missing": 1}


def test_feature_extraction_is_deterministic_non_negative_and_proof_optional() -> None:
    theorem = CanonicalTheorem(
        id="feature",
        source="test",
        source_id="feature",
        statement="theorem feature (n : Nat) (h : n > 0) : ∃ m, m = n := by",
    )

    first = extract_features(theorem)
    second = extract_features(theorem)

    assert first == second
    assert all(value is None or value >= 0 for value in first.values())
    assert first["statement_tokens"] > 0
    assert first["num_binders"] == 2
    assert first["reference_proof_tokens"] is None


def test_static_v1_is_deterministic_and_ignores_reference_trajectory_features() -> None:
    short_trajectory = CanonicalTheorem(
        id="a",
        source="test",
        source_id="a",
        statement="theorem same : True",
        reference_trajectory=(ReferenceTrajectoryStep(0, "⊢ True", "trivial", "no goals"),),
    )
    long_trajectory = CanonicalTheorem(
        id="b",
        source="test",
        source_id="b",
        statement="theorem same : True",
        reference_trajectory=(
            ReferenceTrajectoryStep(0, "⊢ True", "apply True.intro", "⊢ True"),
            ReferenceTrajectoryStep(1, "⊢ True", "trivial", "no goals"),
        ),
    )
    records = [
        CanonicalTheorem(**{**record.__dict__, "features": extract_features(record)})
        for record in (short_trajectory, long_trajectory)
    ]

    first = assign_static_difficulty(records)
    second = assign_static_difficulty(records)

    assert [record.difficulty for record in first] == [record.difficulty for record in second]
    assert first[0].difficulty == first[1].difficulty
    assert first[0].difficulty is not None
    assert 0.0 <= first[0].difficulty.score <= 1.0
    assert first[0].difficulty.method == "static_v1"
    assert first[0].difficulty.bucket in {"easy", "medium", "hard"}


def test_sampling_is_seeded_without_replacement_and_preserves_canonical_records(
    tmp_path: Path,
) -> None:
    input_path = write_workbook_fixture(tmp_path / "workbook.parquet")
    output_path = tmp_path / "theorems.jsonl"
    prepare_dataset(
        source="lean_workbook",
        input_path=input_path,
        output_path=output_path,
        manifest_path=tmp_path / "manifest.json",
    )
    records = load_canonical_records(output_path)

    first = sample_canonical_records(records, bucket="all", size=2, seed=42)
    repeated = sample_canonical_records(records, bucket="all", size=2, seed=42)
    different = sample_canonical_records(records, bucket="all", size=2, seed=7)

    assert [record["id"] for record in first] == [record["id"] for record in repeated]
    assert [record["id"] for record in first] != [record["id"] for record in different]
    assert len({record["id"] for record in first}) == 2
    assert all("reference_trajectory" in record and "difficulty" in record for record in first)
    easy_count = sum(record["difficulty"]["bucket"] == "easy" for record in records)
    if easy_count:
        easy_sample = sample_canonical_records(records, bucket="easy", size=1, seed=1)
        assert easy_sample[0]["difficulty"]["bucket"] == "easy"
    with pytest.raises(DatasetPipelineError, match="exceeds"):
        sample_canonical_records(records, bucket="all", size=len(records) + 1, seed=42)


def test_canonical_extra_fields_do_not_leak_into_one_shot_or_retry_model_input(
    tmp_path: Path,
) -> None:
    statement = "theorem canonical_input : True"
    canonical_path = tmp_path / "canonical.jsonl"
    canonical_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "canonical-id",
                "source": "lean_workbook",
                "source_id": "source-id",
                "statement": statement,
                "informal_statement": "SECRET INFORMAL",
                "answer": "SECRET ANSWER",
                "source_status": "proved",
                "reference_trajectory": [
                    {
                        "step": 0,
                        "state_before": "SECRET STATE BEFORE",
                        "tactic": "SECRET TACTIC",
                        "state_after": "SECRET STATE AFTER",
                    }
                ],
                "reference_proof": "SECRET REFERENCE PROOF",
                "metadata": {},
                "features": {"statement_tokens": 3, "reference_tactic_count": 1},
                "difficulty": {"score": 0.5, "bucket": "medium", "method": "static_v1"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    tasks = load_dataset(canonical_path)
    one_shot_model = FakeModel([GenerationResult("by trivial", "by trivial", None, 1)])
    retry_model = FakeModel(
        [
            GenerationResult(
                "by exact nonexistent_theorem",
                "by exact nonexistent_theorem",
                None,
                1,
            ),
            GenerationResult("by trivial", "by trivial", None, 1),
        ]
    )

    run_one_shot(
        tasks,
        one_shot_model,
        StatusVerifier([VerificationStatus.VERIFIED]),
        tmp_path / "one-shot.jsonl",
        model_alias="mock",
    )
    run_retry(
        tasks,
        retry_model,
        StatusVerifier([VerificationStatus.REJECTED, VerificationStatus.VERIFIED]),
        tmp_path / "retry.jsonl",
        model_alias="mock",
        max_attempts=2,
    )
    one_shot_record = json.loads((tmp_path / "one-shot.jsonl").read_text(encoding="utf-8"))
    retry_record = json.loads((tmp_path / "retry.jsonl").read_text(encoding="utf-8"))

    assert tasks[0].theorem_id == "canonical-id"
    assert tasks[0].metadata.to_dict() == {
        "source": "lean_workbook",
        "source_id": "source-id",
        "difficulty": {"score": 0.5, "bucket": "medium", "method": "static_v1"},
        "reference_tactic_count": 1,
    }
    assert one_shot_model.calls == [statement]
    assert retry_model.calls == [statement, statement]
    assert one_shot_record["task_metadata"] == tasks[0].metadata.to_dict()
    assert retry_record["task_metadata"] == tasks[0].metadata.to_dict()
    assert all("task_metadata" not in attempt for attempt in retry_record["attempts"])
    serialized_results = json.dumps([one_shot_record, retry_record], ensure_ascii=False)
    assert "SECRET INFORMAL" not in serialized_results
    assert "SECRET ANSWER" not in serialized_results
    assert "SECRET TACTIC" not in serialized_results


def write_workbook_fixture(path: Path) -> Path:
    rows = [
        {
            "id": "source-a",
            "status": "proved",
            "natural_language_statement": "Show True.",
            "answer": "true",
            "formal_statement": "theorem workbook_a : True := by sorry  ",
            "tactic": "apply True.intro",
            "state_before": "⊢ True",
            "state_after": "⊢ True",
            "category": "logic",
        },
        {
            "id": "source-a",
            "status": "proved",
            "natural_language_statement": "Show True.",
            "answer": "true",
            "formal_statement": "theorem workbook_a : True := by sorry  ",
            "tactic": "exact True.intro",
            "state_before": "⊢ True",
            "state_after": "no goals",
            "category": "logic",
        },
        {
            "id": "source-a",
            "status": "proved",
            "natural_language_statement": "Show True.",
            "answer": "true",
            "formal_statement": "theorem workbook_a : True := by sorry  ",
            "tactic": "done",
            "state_before": "no goals",
            "state_after": "no goals",
            "category": "logic",
        },
        {
            "id": "source-b",
            "status": "proved",
            "natural_language_statement": None,
            "answer": None,
            "formal_statement": "theorem workbook_b : 1 = 1 := by sorry",
            "tactic": "rfl",
            "state_before": "⊢ 1 = 1",
            "state_after": "no goals",
            "category": None,
        },
        {
            "id": "source-c",
            "status": "disproved",
            "natural_language_statement": "A disproved conjecture.",
            "answer": "false",
            "formal_statement": "theorem workbook_c : False := by sorry",
            "tactic": "contradiction",
            "state_before": "⊢ False",
            "state_after": "⊢ False",
            "category": "logic",
        },
        {
            "id": "source-d",
            "status": "proved",
            "natural_language_statement": "Duplicate theorem D.",
            "answer": None,
            "formal_statement": "theorem duplicated : 2 = 2 := by sorry",
            "tactic": "rfl",
            "state_before": "⊢ 2 = 2",
            "state_after": "no goals",
            "category": "arithmetic",
        },
        {
            "id": "source-e",
            "status": "proved",
            "natural_language_statement": "Duplicate theorem E.",
            "answer": None,
            "formal_statement": "theorem   duplicated :  2 = 2 := by sorry",
            "tactic": "norm_num",
            "state_before": "⊢ 2 = 2",
            "state_after": "no goals",
            "category": "arithmetic",
        },
        {
            "id": "source-conflict",
            "status": "proved",
            "natural_language_statement": "First wording.",
            "answer": None,
            "formal_statement": "theorem conflict : True := by sorry",
            "tactic": "apply True.intro",
            "state_before": "⊢ True",
            "state_after": "⊢ True",
            "category": "invalid",
        },
        {
            "id": "source-conflict",
            "status": "proved",
            "natural_language_statement": "Conflicting wording.",
            "answer": None,
            "formal_statement": "theorem conflict : True := by sorry",
            "tactic": "trivial",
            "state_before": "⊢ True",
            "state_after": "no goals",
            "category": "invalid",
        },
    ]
    parquet.write_table(arrow.Table.from_pylist(rows), path)
    return path


class FakeModel:
    model_name = "mock-model"

    def __init__(self, responses: list[GenerationResult]) -> None:
        self._responses = iter(responses)
        self.calls: list[str] = []

    def generate_proof(self, statement: str) -> GenerationResult:
        self.calls.append(statement)
        return next(self._responses)


class StatusVerifier:
    def __init__(self, statuses: list[VerificationStatus]) -> None:
        self._statuses = iter(statuses)

    def verify(self, statement: str, proof: str) -> LeanResult:
        return LeanResult(next(self._statuses), "", "", 1)
