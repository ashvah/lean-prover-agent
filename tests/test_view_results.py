from __future__ import annotations

import json
from pathlib import Path

from scripts.view_results import (
    build_argument_parser,
    build_html,
    build_lean_export,
    calculate_summary,
    load_results,
    write_result_artifacts,
)


def test_legacy_results_path_uses_strategy_tree_without_changing_source(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "results" / "example.jsonl"
    records = sample_records()
    write_jsonl(result_path, records)
    original_bytes = result_path.read_bytes()

    loaded = load_results(result_path)
    artifacts = write_result_artifacts(result_path)

    assert loaded == records
    assert result_path.read_bytes() == original_bytes
    assert artifacts.html_path == tmp_path / "artifacts" / "one_shot" / "reports" / "example.html"
    assert artifacts.lean_path == tmp_path / "artifacts" / "one_shot" / "exports" / "example.lean"
    assert artifacts.html_path.is_file()
    assert artifacts.lean_path.is_file()


def test_explicit_output_paths_are_supported(tmp_path: Path) -> None:
    result_path = write_jsonl(tmp_path / "results" / "example.jsonl", sample_records())
    html_path = tmp_path / "custom" / "report.html"
    lean_path = tmp_path / "inspection" / "proofs.lean"

    args = build_argument_parser().parse_args(
        [
            str(result_path),
            "--html-output",
            str(html_path),
            "--lean-output",
            str(lean_path),
        ]
    )
    artifacts = write_result_artifacts(
        args.result,
        html_output_path=args.html_output,
        lean_output_path=args.lean_output,
    )

    assert artifacts.html_path == html_path
    assert artifacts.lean_path == lean_path
    assert html_path.is_file()
    assert lean_path.is_file()


def test_retry_results_path_uses_matching_strategy_artifact_tree(tmp_path: Path) -> None:
    result_path = write_jsonl(
        tmp_path / "artifacts" / "retry" / "results" / "example.jsonl",
        [sample_retry_record()],
    )

    artifacts = write_result_artifacts(result_path)

    assert artifacts.html_path == tmp_path / "artifacts" / "retry" / "reports" / "example.html"
    assert artifacts.lean_path == tmp_path / "artifacts" / "retry" / "exports" / "example.lean"


def test_result_metadata_selects_strategy_tree_instead_of_source_folder_name(
    tmp_path: Path,
) -> None:
    result_path = write_jsonl(
        tmp_path / "artifacts" / "one_shot" / "results" / "retry-record.jsonl",
        [sample_retry_record()],
    )

    artifacts = write_result_artifacts(result_path)

    assert artifacts.html_path.parent == tmp_path / "artifacts" / "retry" / "reports"
    assert artifacts.lean_path.parent == tmp_path / "artifacts" / "retry" / "exports"


def test_calculate_summary_uses_stored_millisecond_latencies() -> None:
    summary = calculate_summary(sample_records()[:2])

    assert summary.model_alias == "mock"
    assert summary.model == "mock-model"
    assert summary.total == 2
    assert summary.solved == 1
    assert summary.failed == 1
    assert summary.solve_rate == 50.0
    assert summary.average_generation_latency_ms == 200.0
    assert summary.average_verification_latency_ms == 100.0
    assert summary.average_total_latency_ms == 300.0


def test_html_escapes_all_experiment_text() -> None:
    records = sample_records()
    records[0]["theorem_id"] = 'unsafe"><script>alert(1)</script>'
    records[0]["statement"] = '<script>alert("statement")</script>'
    records[0]["reasoning_output"] = "<aside>reasoning</aside>"
    records[0]["proof_output"] = "<mark>proof</mark>"
    records[0]["normalized_proof"] = "</code><img src=x onerror=alert(1)>"
    records[0]["raw_model_output"] = "<b>raw</b>"
    records[0]["lean_stdout"] = "<em>stdout</em>"
    records[0]["lean_stderr"] = "<strong>stderr</strong>"
    records[0]["error"] = "<script>runner error</script>"
    records[0]["task_metadata"]["source"] = "<script>source</script>"
    summary = calculate_summary(records)

    document = build_html(records, summary, "unsafe.jsonl")

    assert '<script>alert("statement")</script>' not in document
    assert "</code><img src=x onerror=alert(1)>" not in document
    assert "&lt;script&gt;alert(&quot;statement&quot;)&lt;/script&gt;" in document
    assert "&lt;aside&gt;reasoning&lt;/aside&gt;" in document
    assert "&lt;mark&gt;proof&lt;/mark&gt;" in document
    assert "&lt;/code&gt;&lt;img src=x onerror=alert(1)&gt;" in document
    assert "&lt;b&gt;raw&lt;/b&gt;" in document
    assert "&lt;em&gt;stdout&lt;/em&gt;" in document
    assert "&lt;strong&gt;stderr&lt;/strong&gt;" in document
    assert "&lt;script&gt;runner error&lt;/script&gt;" in document
    assert "&lt;script&gt;source&lt;/script&gt;" in document


def test_html_renders_passed_failed_details_and_filters() -> None:
    records = sample_records()
    document = build_html(records, calculate_summary(records), "results.jsonl")

    assert "Lean Prover Agent results" in document
    assert 'data-status="passed"' in document
    assert 'data-status="failed"' in document
    assert ">PASS<" in document
    assert ">FAIL<" in document
    assert "Model alias</span><strong>mock" in document
    assert "Provider model</span><strong>mock-model" in document
    assert "Verifier status:" in document
    assert "VERIFIED" in document
    assert "INCOMPLETE" in document
    assert "REJECTED" in document
    assert document.index("INCOMPLETE") != document.index("REJECTED")
    assert "Benchmark result:" in document
    assert "Reasoning" in document
    assert "Proof output" in document
    assert "Raw model output" in document
    assert "Lean diagnostics" in document
    assert 'data-filter="all"' in document
    assert 'data-filter="passed"' in document
    assert 'data-filter="failed"' in document
    assert 'id="result-search"' in document


def test_all_explicit_verification_statuses_render_distinctly() -> None:
    records = []
    for status in ("verified", "incomplete", "rejected", "timeout", "execution_error"):
        records.append(
            {
                "theorem_id": status,
                "statement": "example : True",
                "verification_status": status,
                "verified": status == "verified",
                "normalized_proof": "by\n  trivial",
            }
        )

    document = build_html(records, calculate_summary(records), "statuses.jsonl")

    for label in ("VERIFIED", "INCOMPLETE", "REJECTED", "TIMEOUT", "EXECUTION_ERROR"):
        assert f">{label}</strong>" in document


def test_token_summary_handles_partially_missing_usage_without_zero_filling() -> None:
    records = [sample_records()[0], sample_records()[1], sample_records()[3]]

    summary = calculate_summary(records)
    document = build_html(records, summary, "tokens.jsonl")

    assert summary.total_prompt_tokens == 40
    assert summary.total_completion_tokens == 8
    assert summary.total_tokens == 38
    assert summary.average_prompt_tokens == 40 / 3
    assert summary.average_completion_tokens == 4.0
    assert summary.average_total_tokens == 19.0
    assert summary.token_usage_available == 2
    assert "Token usage available</span><strong>2 / 3 tasks" in document
    assert "Prompt tokens</dt><dd>10" in document
    assert "Completion tokens</dt><dd>unavailable" in document
    assert 'class="token-chart"' in document
    assert 'class="token-prompt"' in document
    assert 'class="token-completion"' in document


def test_all_missing_token_usage_remains_unavailable() -> None:
    records = [sample_records()[2]]

    summary = calculate_summary(records)
    document = build_html(records, summary, "missing-tokens.jsonl")

    assert summary.total_prompt_tokens is None
    assert summary.total_completion_tokens is None
    assert summary.total_tokens is None
    assert summary.average_total_tokens is None
    assert summary.token_usage_available == 0
    assert "Total tokens</span><strong>unavailable" in document
    assert "Token usage available</span><strong>0 / 1 tasks" in document


def test_difficulty_summary_and_task_metadata_render_from_stored_snapshot() -> None:
    records = sample_records()

    summary = calculate_summary(records)
    document = build_html(records, summary, "difficulty.jsonl")

    buckets = {item.bucket: item for item in summary.difficulty_buckets}
    assert (buckets["easy"].total, buckets["easy"].solved) == (1, 1)
    assert buckets["easy"].solve_rate == 100.0
    assert (buckets["medium"].total, buckets["medium"].solved) == (2, 0)
    assert buckets["medium"].solve_rate == 0.0
    assert summary.unknown_difficulty == 1
    assert "Solve rate by difficulty" in document
    assert '<th scope="row">Easy</th><td>1</td><td>1</td><td>100.0%' in document
    assert '<th scope="row">Medium</th><td>2</td><td>0</td><td>0.0%' in document
    assert '<th scope="row">Unknown</th><td>1</td>' in document
    assert "Source</dt><dd>lean_workbook" in document
    assert "Difficulty bucket</dt><dd>easy" in document
    assert "Difficulty score</dt><dd>0.2" in document
    assert "Difficulty method</dt><dd>static_v1" in document
    assert "Reference tactic count</dt><dd>3" in document


def test_legacy_result_without_task_metadata_remains_viewable() -> None:
    record = {
        "theorem_id": "legacy",
        "statement": "example : True",
        "verified": True,
        "normalized_proof": "by trivial",
    }

    summary = calculate_summary([record])
    document = build_html([record], summary, "legacy.jsonl")

    assert summary.difficulty_buckets == ()
    assert summary.unknown_difficulty == 1
    assert "UNKNOWN" not in document.split("Solve rate by difficulty", maxsplit=1)[0]
    assert "Unknown</th><td>1</td>" in document


def test_legacy_status_is_not_guessed_from_failed_proof_text() -> None:
    legacy_failed = {
        "theorem_id": "legacy_failed",
        "statement": "example : False",
        "normalized_proof": "by\n  sorry",
        "verified": False,
    }
    legacy_verified = {
        "theorem_id": "legacy_verified",
        "statement": "example : True",
        "normalized_proof": "by\n  trivial",
        "verified": True,
    }

    document = build_html(
        [legacy_failed, legacy_verified],
        calculate_summary([legacy_failed, legacy_verified]),
        "legacy.jsonl",
    )

    assert "Model alias" not in document
    assert "Provider model" in document
    assert "UNKNOWN / LEGACY" in document
    assert "VERIFIED (LEGACY)" in document
    assert "Proof output</h3>\n  <pre><code>by\n  sorry</code></pre>" in document


def test_lean_export_preserves_pass_failed_proofs_and_dataset_order() -> None:
    records = sample_records()
    document = build_lean_export(records, calculate_summary(records), "ordered.jsonl")

    passed_declaration = "example : True := by\n  trivial"
    failed_declaration = "example : False := by\n  exact nonexistent_theorem"
    assert document.startswith("import Mathlib\n")
    assert "theorem_id: passed_theorem\nstatus: VERIFIED\nbenchmark_verified: true" in document
    assert "theorem_id: failed_theorem\nstatus: REJECTED\nbenchmark_verified: false" in document
    assert passed_declaration in document
    assert failed_declaration in document
    assert document.index("theorem_id: passed_theorem") < document.index(
        "theorem_id: failed_theorem"
    )
    assert document.index("theorem_id: failed_theorem") < document.index("theorem_id: api_error")


def test_lean_export_preserves_incomplete_proof_exactly() -> None:
    record = sample_records()[3]

    document = build_lean_export([record], calculate_summary([record]), "incomplete.jsonl")

    assert "status: INCOMPLETE\nbenchmark_verified: false" in document
    assert "example : False := by\n  sorry" in document


def test_generation_error_is_comment_only_and_does_not_synthesize_proof() -> None:
    record = sample_records()[2]
    document = build_lean_export([record], calculate_summary([record]), "error.jsonl")

    assert "theorem_id: api_error\nstatus: NOT RUN\nbenchmark_verified: false" in document
    assert "No normalized proof was produced because generation failed." in document
    assert "example : 1 = 1" in document
    assert "error category:\ngeneration_error: RuntimeError" in document
    assert "provider unavailable" not in document
    assert "example : 1 = 1 :=" not in document
    assert "sorry" not in document


def test_lean_header_contains_only_safe_metadata(tmp_path: Path) -> None:
    records = sample_records()
    for record in records:
        record["model"] = "mock-/model"
    result_path = write_jsonl(tmp_path / "results" / "safe.jsonl", records)

    artifacts = write_result_artifacts(result_path)
    document = artifacts.lean_path.read_text(encoding="utf-8")
    header = document.split("theorem_id:", maxsplit=1)[0]

    assert "Lean Prover Agent generated inspection file." in header
    assert "source JSONL: safe.jsonl" in header
    assert "model alias: mock" in header
    assert "model: mock- /model" in header
    assert "total tasks: 4" in header
    assert "verified tasks: 1" in header
    assert "failed tasks: 3" in header
    assert str(tmp_path) not in header
    assert "http://" not in header
    assert "https://" not in header


def test_retry_summary_and_html_render_attempts_in_order() -> None:
    record = sample_retry_record()

    summary = calculate_summary([record])
    document = build_html([record], summary, "retry.jsonl")

    assert summary.strategy == "retry"
    assert summary.solved == 1
    assert "Strategy</span><strong>retry" in document
    assert "Generation budget</dt><dd>4" in document
    assert "Generations used</dt><dd>2" in document
    assert "Selected attempt</dt><dd>2" in document
    assert "All retry attempts (2)" in document
    assert document.index("Attempt 1") < document.index("Attempt 2")
    assert "REJECTED" in document
    assert "VERIFIED" in document
    assert "first attempt reasoning" in document
    assert "by\n  exact nonexistent_theorem" in document
    assert "by\n  trivial" in document


def test_retry_html_escapes_attempt_content() -> None:
    record = sample_retry_record()
    attempts = record["attempts"]
    assert isinstance(attempts, list)
    attempts[0]["raw_model_output"] = "<script>retry raw</script>"
    attempts[0]["reasoning_output"] = "<aside>retry reasoning</aside>"
    attempts[0]["lean_stderr"] = "</code><img src=x onerror=alert(1)>"

    document = build_html([record], calculate_summary([record]), "retry-unsafe.jsonl")

    assert "<script>retry raw</script>" not in document
    assert "&lt;script&gt;retry raw&lt;/script&gt;" in document
    assert "&lt;aside&gt;retry reasoning&lt;/aside&gt;" in document
    assert "&lt;/code&gt;&lt;img src=x onerror=alert(1)&gt;" in document


def test_retry_lean_export_selects_final_attempt_without_rewriting_proof() -> None:
    record = sample_retry_record()

    document = build_lean_export([record], calculate_summary([record]), "retry.jsonl")

    assert "strategy: retry" in document
    assert "selected_attempt: 2 of 2 used" in document
    assert "generation_budget: 4" in document
    assert "status: VERIFIED" in document
    assert "benchmark_verified: true" in document
    assert "example : True := by\n  trivial" in document
    assert "exact nonexistent_theorem" not in document


def test_retry_generation_error_export_remains_comment_only() -> None:
    record = sample_retry_record()
    record["solved"] = False
    record["final_verification_status"] = None
    attempts = record["attempts"]
    assert isinstance(attempts, list)
    attempts[-1] = {
        "attempt_index": 2,
        "raw_model_output": "",
        "reasoning_output": None,
        "proof_output": "",
        "normalized_proof": "",
        "verification_status": None,
        "verified": False,
        "has_sorry": False,
        "lean_stdout": "",
        "lean_stderr": "",
        "prompt_tokens": None,
        "completion_tokens": None,
        "generation_latency_ms": 15,
        "verification_latency_ms": 0,
        "total_attempt_latency_ms": 15,
        "error": "generation_error: RuntimeError: unavailable",
    }

    document = build_lean_export([record], calculate_summary([record]), "retry-error.jsonl")

    assert "status: NOT RUN" in document
    assert "No normalized proof was produced because generation failed." in document
    assert "example : True :=" not in document
    assert "sorry" not in document


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def sample_retry_record() -> dict[str, object]:
    return {
        "theorem_id": "retry_theorem",
        "statement": "example : True",
        "task_metadata": {
            "source": "lean_workbook",
            "source_id": "retry-source",
            "difficulty": {"score": 0.8, "bucket": "hard", "method": "static_v1"},
        },
        "strategy": "retry",
        "model_alias": "mock",
        "model": "mock-model",
        "generation_budget": 4,
        "attempts": [
            {
                "attempt_index": 1,
                "raw_model_output": "by\n  exact nonexistent_theorem",
                "reasoning_output": "first attempt reasoning",
                "proof_output": "by\n  exact nonexistent_theorem",
                "normalized_proof": "by\n  exact nonexistent_theorem",
                "verification_status": "rejected",
                "verified": False,
                "has_sorry": False,
                "lean_stdout": "",
                "lean_stderr": "unknown identifier",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "generation_latency_ms": 100,
                "verification_latency_ms": 50,
                "total_attempt_latency_ms": 150,
                "error": None,
            },
            {
                "attempt_index": 2,
                "raw_model_output": "by\n  trivial",
                "reasoning_output": None,
                "proof_output": "by\n  trivial",
                "normalized_proof": "by\n  trivial",
                "verification_status": "verified",
                "verified": True,
                "has_sorry": False,
                "lean_stdout": "",
                "lean_stderr": "",
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "generation_latency_ms": 110,
                "verification_latency_ms": 40,
                "total_attempt_latency_ms": 150,
                "error": None,
            },
        ],
        "final_verification_status": "verified",
        "solved": True,
        "generations_used": 2,
        "verifier_calls": 2,
        "prompt_tokens": 22,
        "completion_tokens": 8,
        "generation_latency_ms": 210,
        "verification_latency_ms": 90,
        "total_latency_ms": 300,
    }


def sample_records() -> list[dict[str, object]]:
    return [
        {
            "theorem_id": "passed_theorem",
            "statement": "example : True",
            "task_metadata": {
                "source": "lean_workbook",
                "source_id": "source-passed",
                "difficulty": {"score": 0.2, "bucket": "easy", "method": "static_v1"},
                "reference_tactic_count": 3,
            },
            "model_alias": "mock",
            "model": "mock-model",
            "raw_model_output": "```lean\nby\n  trivial\n```",
            "reasoning_output": "A direct proof closes True.",
            "proof_output": "```lean\nby\n  trivial\n```",
            "normalized_proof": "by\n  trivial",
            "verification_status": "verified",
            "verified": True,
            "has_sorry": False,
            "lean_stdout": "",
            "lean_stderr": "",
            "generation_latency_ms": 100,
            "verification_latency_ms": 50,
            "total_latency_ms": 150,
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "error": None,
        },
        {
            "theorem_id": "failed_theorem",
            "statement": "example : False",
            "task_metadata": {
                "source": "lean_workbook",
                "source_id": "source-failed",
                "difficulty": {"score": 0.5, "bucket": "medium", "method": "static_v1"},
            },
            "model_alias": "mock",
            "model": "mock-model",
            "raw_model_output": "by\n  exact nonexistent_theorem",
            "reasoning_output": None,
            "proof_output": "by\n  exact nonexistent_theorem",
            "normalized_proof": "by\n  exact nonexistent_theorem",
            "verification_status": "rejected",
            "verified": False,
            "has_sorry": False,
            "lean_stdout": "unknown identifier",
            "lean_stderr": "",
            "generation_latency_ms": 300,
            "verification_latency_ms": 150,
            "total_latency_ms": 450,
            "prompt_tokens": 10,
            "completion_tokens": None,
            "error": None,
        },
        {
            "theorem_id": "api_error",
            "statement": "example : 1 = 1",
            "model_alias": "mock",
            "model": "mock-model",
            "raw_model_output": "",
            "reasoning_output": None,
            "proof_output": "",
            "normalized_proof": "",
            "verification_status": None,
            "verified": False,
            "has_sorry": False,
            "lean_stdout": "",
            "lean_stderr": "",
            "generation_latency_ms": 25,
            "verification_latency_ms": 0,
            "total_latency_ms": 25,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": "generation_error: RuntimeError: provider unavailable",
        },
        {
            "theorem_id": "incomplete_theorem",
            "statement": "example : False",
            "task_metadata": {
                "source": "lean_workbook",
                "source_id": "source-incomplete",
                "difficulty": {"score": 0.6, "bucket": "medium", "method": "static_v1"},
            },
            "model_alias": "mock",
            "model": "mock-model",
            "raw_model_output": "<think>Use a placeholder.</think>\n\nby\n  sorry",
            "reasoning_output": "Use a placeholder.",
            "proof_output": "by\n  sorry",
            "normalized_proof": "by\n  sorry",
            "verification_status": "incomplete",
            "verified": False,
            "has_sorry": True,
            "lean_stdout": '{"kind":"hasSorry","severity":"warning"}',
            "lean_stderr": "",
            "generation_latency_ms": 200,
            "verification_latency_ms": 100,
            "total_latency_ms": 300,
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "error": None,
        },
    ]
