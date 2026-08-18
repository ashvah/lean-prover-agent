from __future__ import annotations

import json

from scripts.view_results import (
    build_html,
    calculate_summary,
    load_results,
    write_result_viewer,
)


def test_load_results_reads_valid_jsonl_without_changing_source(tmp_path) -> None:
    result_path = tmp_path / "results.jsonl"
    records = sample_records()
    result_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    original_bytes = result_path.read_bytes()

    loaded = load_results(result_path)
    output_path = write_result_viewer(result_path)

    assert loaded == records
    assert result_path.read_bytes() == original_bytes
    assert output_path == result_path.with_suffix(".html")
    assert output_path.is_file()


def test_calculate_summary_uses_stored_millisecond_latencies() -> None:
    summary = calculate_summary(sample_records())

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
    records[0]["normalized_proof"] = "</code><img src=x onerror=alert(1)>"
    records[0]["raw_model_output"] = "<b>raw</b>"
    records[0]["lean_stdout"] = "<em>stdout</em>"
    records[0]["lean_stderr"] = "<strong>stderr</strong>"
    summary = calculate_summary(records)

    document = build_html(records, summary, "unsafe.jsonl")

    assert '<script>alert("statement")</script>' not in document
    assert "</code><img src=x onerror=alert(1)>" not in document
    assert "&lt;script&gt;alert(&quot;statement&quot;)&lt;/script&gt;" in document
    assert "&lt;/code&gt;&lt;img src=x onerror=alert(1)&gt;" in document
    assert "&lt;b&gt;raw&lt;/b&gt;" in document
    assert "&lt;em&gt;stdout&lt;/em&gt;" in document
    assert "&lt;strong&gt;stderr&lt;/strong&gt;" in document


def test_html_renders_passed_failed_details_and_filters() -> None:
    records = sample_records()
    document = build_html(records, calculate_summary(records), "results.jsonl")

    assert 'data-status="passed"' in document
    assert 'data-status="failed"' in document
    assert ">PASS<" in document
    assert ">FAIL<" in document
    assert "Raw model output" in document
    assert "Lean stdout/stderr" in document
    assert 'data-filter="all"' in document
    assert 'data-filter="passed"' in document
    assert 'data-filter="failed"' in document
    assert 'id="result-search"' in document


def sample_records() -> list[dict[str, object]]:
    return [
        {
            "theorem_id": "passed_theorem",
            "statement": "example : True",
            "model": "mock-model",
            "raw_model_output": "```lean\nby trivial\n```",
            "normalized_proof": "by trivial",
            "verified": True,
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
            "raw_model_output": "by trivial",
            "normalized_proof": "by trivial",
            "verified": False,
            "lean_stdout": "unsolved goals",
            "lean_stderr": "",
            "generation_latency_ms": 300,
            "verification_latency_ms": 150,
            "total_latency_ms": 450,
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "error": None,
        },
    ]
