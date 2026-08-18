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


def test_default_paths_create_report_and_export_directories_without_changing_source(
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
    assert artifacts.html_path == tmp_path / "reports" / "example.html"
    assert artifacts.lean_path == tmp_path / "exports" / "example.lean"
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


def test_calculate_summary_uses_stored_millisecond_latencies() -> None:
    summary = calculate_summary(sample_records()[:2])

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


def test_html_renders_passed_failed_details_and_filters() -> None:
    records = sample_records()
    document = build_html(records, calculate_summary(records), "results.jsonl")

    assert 'data-status="passed"' in document
    assert 'data-status="failed"' in document
    assert ">PASS<" in document
    assert ">FAIL<" in document
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

    assert "source JSONL: safe.jsonl" in header
    assert "model: mock- /model" in header
    assert "total tasks: 4" in header
    assert "verified tasks: 1" in header
    assert "failed tasks: 3" in header
    assert str(tmp_path) not in header
    assert "http://" not in header
    assert "https://" not in header


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def sample_records() -> list[dict[str, object]]:
    return [
        {
            "theorem_id": "passed_theorem",
            "statement": "example : True",
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
