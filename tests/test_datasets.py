from __future__ import annotations

import json
from pathlib import Path

import pyarrow as arrow
import pytest
from pyarrow import parquet

from leanproof.datasets import (
    CanonicalTheorem,
    DatasetPipelineError,
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

    assert first.raw_records == 6
    assert first.mapped_records == 5
    assert first.invalid_records == 1
    assert first.duplicates_removed == 1
    assert first.final_records == 4
    assert first.invalid_reasons == {"formal_statement_blank": 1}
    assert sum(first.bucket_counts.values()) == 4
    assert (
        first.raw_records,
        first.mapped_records,
        first.invalid_records,
        first.duplicates_removed,
        first.final_records,
        first.bucket_counts,
        first.invalid_reasons,
    ) == (
        second.raw_records,
        second.mapped_records,
        second.invalid_records,
        second.duplicates_removed,
        second.final_records,
        second.bucket_counts,
        second.invalid_reasons,
    )
    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert records[0]["statement"] == "  theorem workbook_a : True"
    assert records[0]["reference_proof"] == "exact True.intro"
    assert records[0]["metadata"]["source_formal_statement"].endswith(":= by sorry  ")
    assert records[1]["reference_proof"] is None
    assert records[0]["metadata"]["category"] == "logic"
    assert manifest["lean_validation"] == {"performed": False}
    assert manifest["raw_input"] == "workbook.parquet"
    assert manifest["difficulty_method"] == "static_v1"


def test_lean_workbook_adapter_requires_formal_statement_column(tmp_path: Path) -> None:
    input_path = tmp_path / "missing.parquet"
    parquet.write_table(arrow.Table.from_pylist([{"problem": "No formal column"}]), input_path)

    with pytest.raises(LeanWorkbookSchemaError, match="formal statement column"):
        LeanWorkbookAdapter(input_path)


def test_adapter_ids_are_stable_and_missing_optional_proof_is_none(tmp_path: Path) -> None:
    input_path = write_workbook_fixture(tmp_path / "workbook.parquet")

    first_adapter = LeanWorkbookAdapter(input_path)
    second_adapter = LeanWorkbookAdapter(input_path)
    first = [first_adapter.map_row(row) for row in first_adapter.iter_rows(limit=2)]
    second = [second_adapter.map_row(row) for row in second_adapter.iter_rows(limit=2)]

    assert [record.id for record in first] == [record.id for record in second]
    assert first[0].source_id == "source-a"
    assert first[1].reference_proof is None


def test_adapter_derives_stable_id_when_source_id_is_unavailable(tmp_path: Path) -> None:
    input_path = tmp_path / "no-id.parquet"
    parquet.write_table(
        arrow.Table.from_pylist(
            [
                {
                    "natural_language_statement": "Show True.",
                    "formal_statement": "theorem no_id : True := by sorry",
                }
            ]
        ),
        input_path,
    )

    first_adapter = LeanWorkbookAdapter(input_path)
    second_adapter = LeanWorkbookAdapter(input_path)
    first = first_adapter.map_row(next(first_adapter.iter_rows()))
    second = second_adapter.map_row(next(second_adapter.iter_rows()))

    assert first.id == second.id
    assert first.source_id == second.source_id
    assert first.source_id.startswith("sha256:")


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


def test_static_v1_is_deterministic_and_ignores_reference_proof_features() -> None:
    without_proof = CanonicalTheorem(
        id="a",
        source="test",
        source_id="a",
        statement="theorem a : True",
    )
    with_proof = CanonicalTheorem(
        id="b",
        source="test",
        source_id="b",
        statement="theorem b : True",
        reference_proof="by\n  trivial\n  trivial",
    )
    records = [
        CanonicalTheorem(**{**record.__dict__, "features": extract_features(record)})
        for record in (without_proof, with_proof)
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

    first = sample_canonical_records(records, bucket="all", size=3, seed=42)
    repeated = sample_canonical_records(records, bucket="all", size=3, seed=42)
    different = sample_canonical_records(records, bucket="all", size=3, seed=7)

    assert [record["id"] for record in first] == [record["id"] for record in repeated]
    assert [record["id"] for record in first] != [record["id"] for record in different]
    assert len({record["id"] for record in first}) == 3
    assert all("reference_proof" in record and "difficulty" in record for record in first)
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
                "reference_proof": "SECRET REFERENCE PROOF",
                "metadata": {},
                "features": {"statement_tokens": 3},
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
            GenerationResult("bad", "bad", None, 1),
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

    assert tasks[0].theorem_id == "canonical-id"
    assert one_shot_model.calls == [statement]
    assert retry_model.calls == [statement, statement]


def write_workbook_fixture(path: Path) -> Path:
    rows = [
        {
            "id": "source-a",
            "status": "proved",
            "natural_language_statement": "Show True.",
            "answer": "true",
            "formal_statement": "  theorem workbook_a : True := by sorry  ",
            "tactic": "exact True.intro",
            "state_before": "⊢ True",
            "state_after": "no goals",
            "category": "logic",
        },
        {
            "id": "source-b",
            "status": "proved",
            "natural_language_statement": "Duplicate with whitespace.",
            "answer": "true",
            "formal_statement": "theorem   workbook_a :   True := by sorry",
            "tactic": None,
            "state_before": "⊢ True",
            "state_after": "no goals",
            "category": None,
        },
        {
            "id": "source-blank",
            "status": "failed",
            "natural_language_statement": "Invalid.",
            "answer": None,
            "formal_statement": "   ",
            "tactic": None,
            "state_before": None,
            "state_after": None,
            "category": "invalid",
        },
        {
            "id": "source-c",
            "status": "proved",
            "natural_language_statement": "Identity.",
            "answer": None,
            "formal_statement": "theorem workbook_c (n : Nat) : n = n := by sorry",
            "tactic": None,
            "state_before": "n : Nat\n⊢ n = n",
            "state_after": None,
            "category": "algebra",
        },
        {
            "id": "source-d",
            "status": "proved",
            "natural_language_statement": "Use a hypothesis.",
            "answer": None,
            "formal_statement": "theorem workbook_d (p : Prop) (h : p) : p := by sorry",
            "tactic": "exact h",
            "state_before": "p : Prop\nh : p\n⊢ p",
            "state_after": "no goals",
            "category": "logic",
        },
        {
            "id": "source-e",
            "status": "proved",
            "natural_language_statement": "Quantifiers.",
            "answer": None,
            "formal_statement": "theorem workbook_e : ∀ n : Nat, ∃ m : Nat, m ≥ n := by sorry",
            "tactic": "intro n; exact ⟨n, le_rfl⟩",
            "state_before": "⊢ ∀ n : Nat, ∃ m : Nat, m ≥ n",
            "state_after": "no goals",
            "category": "logic",
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
