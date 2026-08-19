from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from leanproof.lean import LeanResult, VerificationStatus
from leanproof.models import GenerationResult, LLMConfig, ModelRegistry
from leanproof.strategies import (
    DatasetError,
    TheoremTask,
    default_output_path,
    load_dataset,
    run_one_shot,
)
from scripts._common import print_progress
from scripts.run_one_shot import (
    DEFAULT_RESULTS_DIRECTORY,
    build_argument_parser,
)
from scripts.run_one_shot import (
    main as run_cli,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_dataset_has_25_unique_single_line_records() -> None:
    dataset_path = PROJECT_ROOT / "data" / "smoke.jsonl"
    sanitation_source = (PROJECT_ROOT / "LeanProverAgent" / "SmokeDataset.lean").read_text(
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
                raw_output="<think>reasoning</think>\n\n  ```lean\nby\n  exact h\n```  ",
                proof_output="  ```lean\nby\n  exact h\n```  ",
                reasoning_output="reasoning",
                latency_ms=11,
                prompt_tokens=20,
                completion_tokens=5,
            ),
            GenerationResult(
                raw_output="not a proof",
                proof_output="not a proof",
                reasoning_output=None,
                latency_ms=12,
            ),
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
        model_alias="mock",
        progress_callback=progress_messages.append,
    )
    physical_lines = output_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in physical_lines]

    assert model.calls == [task.statement for task in tasks]
    assert len(physical_lines) == 3
    assert len(verifier.calls) == 2
    assert summary.solved == 1
    assert summary.total == 3
    assert all(record["model_alias"] == "mock" for record in records)
    assert all(record["model"] == "mock-model" for record in records)
    assert records[0]["raw_model_output"].startswith("<think>reasoning</think>")
    assert records[0]["reasoning_output"] == "reasoning"
    assert records[0]["proof_output"] == "  ```lean\nby\n  exact h\n```  "
    assert records[0]["normalized_proof"] == "by\n  exact h"
    assert records[0]["verification_status"] == "verified"
    assert records[0]["verified"] is True
    assert records[0]["has_sorry"] is False
    assert records[0]["prompt_tokens"] == 20
    assert records[0]["completion_tokens"] == 5
    assert records[1]["raw_model_output"] == "not a proof"
    assert records[1]["normalized_proof"] == "not a proof"
    assert records[1]["verification_status"] == "rejected"
    assert records[1]["verified"] is False
    assert records[2]["raw_model_output"] == ""
    assert records[2]["proof_output"] == ""
    assert records[2]["reasoning_output"] is None
    assert records[2]["verification_status"] is None
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
        print_progress("progress")

    print_mock.assert_called_once_with("progress", flush=True)


def test_cli_default_artifact_directory_is_strategy_results() -> None:
    assert DEFAULT_RESULTS_DIRECTORY == PROJECT_ROOT / "artifacts" / "one_shot" / "results"


def test_default_output_filename_includes_model_alias(tmp_path) -> None:
    output_path = default_output_path("data/smoke.jsonl", "minimax", tmp_path)

    assert output_path.parent == tmp_path
    assert output_path.name.startswith("one_shot_smoke_minimax_")
    assert output_path.suffix == ".jsonl"


def test_cli_preserves_explicit_output_path() -> None:
    args = build_argument_parser().parse_args(
        [
            "--dataset",
            "data/smoke.jsonl",
            "--model",
            "minimax",
            "--output",
            "custom/result.jsonl",
        ]
    )

    assert args.output == "custom/result.jsonl"


def test_list_models_does_not_require_dataset_or_model(capsys) -> None:
    registry = ModelRegistry(
        {
            "qwen": LLMConfig("qwen-key", "https://api.example.com/v1", "qwen-model"),
            "minimax": LLMConfig("minimax-key", "https://api.example.com/v1", "minimax-model"),
        }
    )

    with patch("scripts.run_one_shot.ModelRegistry.from_env", return_value=registry):
        exit_code = run_cli(["--list-models"])

    assert exit_code == 0
    assert capsys.readouterr().out == "Registered models:\n- minimax\n- qwen\n"


@pytest.mark.parametrize(
    ("arguments", "missing_option"),
    [
        (["--dataset", "data/smoke.jsonl"], "--model"),
        (["--model", "minimax"], "--dataset"),
    ],
)
def test_normal_execution_requires_dataset_and_model(arguments, missing_option, capsys) -> None:
    with pytest.raises(SystemExit) as captured:
        run_cli(arguments)

    assert captured.value.code == 2
    assert missing_option in capsys.readouterr().err


def test_incomplete_proof_is_serialized_but_not_counted_as_solved(tmp_path) -> None:
    model = FakeModel(
        [
            GenerationResult(
                raw_output="by\n  sorry",
                proof_output="by\n  sorry",
                reasoning_output=None,
                latency_ms=4,
            )
        ]
    )
    output_path = tmp_path / "incomplete.jsonl"

    summary = run_one_shot(
        [TheoremTask("incomplete", "example : False")],
        model,
        IncompleteVerifier(),
        output_path,
        model_alias="mock",
    )
    record = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary.solved == 0
    assert record["verification_status"] == "incomplete"
    assert record["verified"] is False
    assert record["has_sorry"] is True


def test_runner_continues_after_verifier_exception(tmp_path) -> None:
    tasks = [
        TheoremTask("verifier-error", "example : True"),
        TheoremTask("next-task", "example : True"),
    ]
    model = FakeModel(
        [
            GenerationResult("by trivial", "by trivial", None, 1),
            GenerationResult("by trivial", "by trivial", None, 1),
        ]
    )
    verifier = RaisingOnceVerifier()
    output_path = tmp_path / "verifier_error.jsonl"

    summary = run_one_shot(tasks, model, verifier, output_path, model_alias="mock")
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
            status=(VerificationStatus.VERIFIED if success else VerificationStatus.REJECTED),
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
        return LeanResult(
            status=VerificationStatus.VERIFIED,
            stdout="",
            stderr="",
            elapsed_ms=3,
        )


class IncompleteVerifier:
    def verify(self, statement: str, proof: str) -> LeanResult:
        assert statement == "example : False"
        assert proof == "by\n  sorry"
        return LeanResult(
            status=VerificationStatus.INCOMPLETE,
            stdout='{"kind":"hasSorry","severity":"warning"}\n',
            stderr="",
            elapsed_ms=2,
        )
