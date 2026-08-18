from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from leanproof.model import GenerationResult
from leanproof.one_shot import DatasetError, TheoremTask, load_dataset, run_one_shot
from leanproof.verifier import LeanResult
from scripts.run_one_shot import DEFAULT_RESULTS_DIRECTORY, _print_progress, build_argument_parser

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_dataset_has_25_unique_single_line_records() -> None:
    dataset_path = PROJECT_ROOT / "data" / "smoke.jsonl"
    sanitation_source = (PROJECT_ROOT / "LeanproofAgent" / "SmokeDataset.lean").read_text(
        encoding="utf-8"
    )
    physical_lines = dataset_path.read_text(encoding="utf-8").splitlines()

    tasks = load_dataset(dataset_path)

    assert len(tasks) == 25
    assert len(physical_lines) == 25
    assert len({task.theorem_id for task in tasks}) == 25
    assert all(json.loads(line)["statement"] for line in physical_lines)
    assert all(task.statement in sanitation_source for task in tasks)


def test_load_dataset_fails_fast_on_duplicate_id(tmp_path) -> None:
    dataset_path = tmp_path / "duplicate.jsonl"
    dataset_path.write_text(
        '{"theorem_id":"same","statement":"example : True"}\n'
        '{"theorem_id":"same","statement":"example : False"}\n',
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="Duplicate theorem_id"):
        load_dataset(dataset_path)


def test_runner_records_raw_and_normalized_output_and_continues(tmp_path) -> None:
    tasks = [
        TheoremTask("valid", "example (p : Prop) (h : p) : p"),
        TheoremTask("malformed", "example : True"),
        TheoremTask("api-error", "example : False"),
    ]
    model = FakeModel(
        [
            GenerationResult(
                raw_output="  ```lean\nby\n  exact h\n```  ",
                latency_ms=11,
                prompt_tokens=20,
                completion_tokens=5,
            ),
            GenerationResult(raw_output="not a proof", latency_ms=12),
            RuntimeError("provider unavailable"),
        ]
    )
    verifier = FakeVerifier()
    output_path = tmp_path / "one_shot.jsonl"
    progress_messages: list[str] = []

    summary = run_one_shot(
        tasks,
        model,
        verifier,
        output_path,
        progress_callback=progress_messages.append,
    )
    physical_lines = output_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in physical_lines]

    assert model.calls == [task.statement for task in tasks]
    assert len(physical_lines) == 3
    assert len(verifier.calls) == 2
    assert summary.solved == 1
    assert summary.total == 3
    assert records[0]["raw_model_output"] == "  ```lean\nby\n  exact h\n```  "
    assert records[0]["normalized_proof"] == "by\n  exact h"
    assert records[0]["verified"] is True
    assert records[0]["prompt_tokens"] == 20
    assert records[0]["completion_tokens"] == 5
    assert records[1]["raw_model_output"] == "not a proof"
    assert records[1]["normalized_proof"] == "not a proof"
    assert records[1]["verified"] is False
    assert records[2]["raw_model_output"] == ""
    assert records[2]["error"].startswith("generation_error: RuntimeError")
    assert progress_messages[0] == "[1/3] valid | generating..."
    assert progress_messages[1] == "[1/3] valid | generated   | 11 ms"
    assert progress_messages[2] == "[1/3] valid | verifying..."
    assert progress_messages[3].startswith("[1/3] valid | PASS")
    assert progress_messages[7].startswith("[2/3] malformed | FAIL")
    assert progress_messages[8] == "[3/3] api-error | generating..."
    assert progress_messages[9].startswith("[3/3] api-error | ERROR")
    assert all("exact h" not in message for message in progress_messages)


@pytest.mark.parametrize("verbose_flag", ["-v", "--verbose"])
def test_cli_accepts_verbose_aliases(verbose_flag: str) -> None:
    args = build_argument_parser().parse_args(["--dataset", "data/smoke.jsonl", verbose_flag])

    assert args.verbose is True


def test_verbose_printer_flushes_immediately() -> None:
    with patch("builtins.print") as print_mock:
        _print_progress("progress")

    print_mock.assert_called_once_with("progress", flush=True)


def test_cli_default_artifact_directory_is_results() -> None:
    assert DEFAULT_RESULTS_DIRECTORY == PROJECT_ROOT / "results"


def test_runner_continues_after_verifier_exception(tmp_path) -> None:
    tasks = [
        TheoremTask("verifier-error", "example : True"),
        TheoremTask("next-task", "example : True"),
    ]
    model = FakeModel(
        [
            GenerationResult("by trivial", 1),
            GenerationResult("by trivial", 1),
        ]
    )
    verifier = RaisingOnceVerifier()
    output_path = tmp_path / "verifier_error.jsonl"

    summary = run_one_shot(tasks, model, verifier, output_path)
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert len(model.calls) == 2
    assert len(verifier.calls) == 2
    assert records[0]["verified"] is False
    assert records[0]["error"].startswith("verification_error: RuntimeError")
    assert records[1]["verified"] is True
    assert summary.solved == 1


class FakeModel:
    model_name = "mock-model"

    def __init__(self, responses: list[GenerationResult | Exception]) -> None:
        self.responses = iter(responses)
        self.calls: list[str] = []

    def generate_proof(self, statement: str) -> GenerationResult:
        self.calls.append(statement)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeVerifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def verify(self, statement: str, proof: str) -> LeanResult:
        self.calls.append((statement, proof))
        success = proof == "by\n  exact h"
        return LeanResult(
            success=success,
            stdout="" if success else "Lean rejected proof",
            stderr="",
            elapsed_ms=7,
        )


class RaisingOnceVerifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def verify(self, statement: str, proof: str) -> LeanResult:
        self.calls.append((statement, proof))
        if len(self.calls) == 1:
            raise RuntimeError("Lean process failed to start")
        return LeanResult(success=True, stdout="", stderr="", elapsed_ms=3)
