"""Generate read-only HTML and Lean artifacts from an experiment JSONL result file."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class ResultViewerError(ValueError):
    """Raised when an experiment result file cannot be rendered safely."""


@dataclass(frozen=True)
class ResultSummary:
    """Aggregate values displayed at the top of the result viewer."""

    model_alias: str | None
    model: str
    strategy: str
    total: int
    solved: int
    failed: int
    solve_rate: float
    average_generation_latency_ms: float
    average_verification_latency_ms: float
    average_total_latency_ms: float
    total_prompt_tokens: int | None
    total_completion_tokens: int | None
    total_tokens: int | None
    average_prompt_tokens: float | None
    average_completion_tokens: float | None
    average_total_tokens: float | None
    token_usage_available: int


@dataclass(frozen=True)
class GeneratedArtifacts:
    """Paths written from one immutable JSONL experiment result."""

    html_path: Path
    lean_path: Path


def load_results(result_path: str | Path) -> list[dict[str, object]]:
    """Load JSONL records without modifying or normalizing their contents."""

    path = Path(result_path)
    if not path.is_file():
        raise ResultViewerError(f"Result file does not exist: {path}")

    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ResultViewerError(f"Blank JSONL record at {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ResultViewerError(
                    f"Invalid JSON at {path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ResultViewerError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(record)

    if not records:
        raise ResultViewerError(f"Result file is empty: {path}")
    return records


def calculate_summary(records: Sequence[dict[str, object]]) -> ResultSummary:
    """Calculate solve counts and mean latencies from stored JSONL values."""

    if not records:
        raise ResultViewerError("Cannot summarize an empty result set")

    model_aliases = sorted(
        {_text(record.get("model_alias")) for record in records if _text(record.get("model_alias"))}
    )
    model_alias = ", ".join(model_aliases) if model_aliases else None
    models = sorted({_text(record.get("model")) or "unknown" for record in records})
    model = models[0] if len(models) == 1 else ", ".join(models)
    strategies = sorted({_text(record.get("strategy")) or "one_shot" for record in records})
    strategy = strategies[0] if len(strategies) == 1 else ", ".join(strategies)
    total = len(records)
    solved = sum(_benchmark_verified(record) for record in records)
    prompt_tokens = [_token_value(record.get("prompt_tokens")) for record in records]
    completion_tokens = [_token_value(record.get("completion_tokens")) for record in records]
    available_prompt_tokens = [value for value in prompt_tokens if value is not None]
    available_completion_tokens = [value for value in completion_tokens if value is not None]
    available_total_tokens = [
        prompt + completion
        for prompt, completion in zip(prompt_tokens, completion_tokens, strict=True)
        if prompt is not None and completion is not None
    ]
    return ResultSummary(
        model_alias=model_alias,
        model=model,
        strategy=strategy,
        total=total,
        solved=solved,
        failed=total - solved,
        solve_rate=100.0 * solved / total,
        average_generation_latency_ms=_average_latency(records, "generation_latency_ms"),
        average_verification_latency_ms=_average_latency(records, "verification_latency_ms"),
        average_total_latency_ms=_average_latency(records, "total_latency_ms"),
        total_prompt_tokens=_optional_sum(available_prompt_tokens),
        total_completion_tokens=_optional_sum(available_completion_tokens),
        total_tokens=_optional_sum(available_total_tokens),
        average_prompt_tokens=_optional_average(available_prompt_tokens),
        average_completion_tokens=_optional_average(available_completion_tokens),
        average_total_tokens=_optional_average(available_total_tokens),
        token_usage_available=len(available_total_tokens),
    )


def build_html(
    records: Sequence[dict[str, object]], summary: ResultSummary, source_name: str
) -> str:
    """Render escaped result records into a self-contained HTML document."""

    cards = "\n".join(_render_result_card(record) for record in records)
    token_chart = _render_token_chart(records)
    model_alias_html = (
        f"<div><span>Model alias</span><strong>{_escape(summary.model_alias)}</strong></div>"
        if summary.model_alias is not None
        else ""
    )
    summary_html = f"""
<section class="summary" aria-label="Experiment summary">
  {model_alias_html}
  <div><span>Provider model</span><strong>{_escape(summary.model)}</strong></div>
  <div><span>Strategy</span><strong>{_escape(summary.strategy)}</strong></div>
  <div><span>Total</span><strong>{summary.total}</strong></div>
  <div><span>Solved</span><strong>{summary.solved}</strong></div>
  <div><span>Failed</span><strong>{summary.failed}</strong></div>
  <div><span>Solve rate</span><strong>{summary.solve_rate:.1f}%</strong></div>
  <div><span>Avg generation</span><strong>{summary.average_generation_latency_ms:.1f} ms</strong></div>
  <div><span>Avg verification</span><strong>{summary.average_verification_latency_ms:.1f} ms</strong></div>
  <div><span>Avg total</span><strong>{summary.average_total_latency_ms:.1f} ms</strong></div>
  <div><span>Total prompt tokens</span><strong>{_format_token(summary.total_prompt_tokens)}</strong></div>
  <div><span>Total completion tokens</span><strong>{_format_token(summary.total_completion_tokens)}</strong></div>
  <div><span>Total tokens</span><strong>{_format_token(summary.total_tokens)}</strong></div>
  <div><span>Avg prompt tokens / available theorem</span><strong>{_format_optional_number(summary.average_prompt_tokens)}</strong></div>
  <div><span>Avg completion tokens / available theorem</span><strong>{_format_optional_number(summary.average_completion_tokens)}</strong></div>
  <div><span>Avg total tokens / available theorem</span><strong>{_format_optional_number(summary.average_total_tokens)}</strong></div>
  <div><span>Token usage available</span><strong>{summary.token_usage_available} / {summary.total} tasks</strong></div>
</section>""".strip()
    replacements = {
        "TITLE": _escape(f"Lean Prover Agent results — {source_name}"),
        "SOURCE": _escape(source_name),
        "SUMMARY": summary_html,
        "TOKEN_CHART": token_chart,
        "CARDS": cards,
    }
    return re.sub(
        r"__(TITLE|SOURCE|SUMMARY|TOKEN_CHART|CARDS)__",
        lambda match: replacements[match.group(1)],
        _HTML_TEMPLATE,
    )


def build_lean_export(
    records: Sequence[dict[str, object]], summary: ResultSummary, source_name: str
) -> str:
    """Render stored statements and normalized proofs for manual Lean inspection."""

    sections = [
        "import Mathlib",
        "",
        "/-",
        "Lean Prover Agent generated inspection file.",
        f"source JSONL: {_lean_comment_text(source_name)}",
        *(
            [f"model alias: {_lean_comment_text(summary.model_alias)}"]
            if summary.model_alias is not None
            else []
        ),
        f"model: {_lean_comment_text(summary.model)}",
        f"strategy: {_lean_comment_text(summary.strategy)}",
        f"total tasks: {summary.total}",
        f"verified tasks: {summary.solved}",
        f"failed tasks: {summary.failed}",
        "Failed model-generated proofs are intentionally preserved.",
        "-/",
        "",
    ]
    for record in records:
        sections.extend(_render_lean_record(record))
    return "\n".join(sections)


def default_artifact_paths(
    result_path: str | Path, *, strategy: str | None = None
) -> GeneratedArtifacts:
    """Place derived files in the matching strategy tree for conventional results."""

    source_path = Path(result_path)
    if source_path.parent.name == "results" and strategy in {"one_shot", "retry"}:
        parent_strategy_root = source_path.parent.parent
        artifacts_root = (
            parent_strategy_root.parent
            if parent_strategy_root.parent.name == "artifacts"
            else parent_strategy_root / "artifacts"
        )
        artifact_root = artifacts_root / strategy
    else:
        artifact_root = (
            source_path.parent.parent
            if source_path.parent.name == "results"
            else source_path.parent
        )
    return GeneratedArtifacts(
        html_path=artifact_root / "reports" / f"{source_path.stem}.html",
        lean_path=artifact_root / "exports" / f"{source_path.stem}.lean",
    )


def write_result_artifacts(
    result_path: str | Path,
    html_output_path: str | Path | None = None,
    lean_output_path: str | Path | None = None,
) -> GeneratedArtifacts:
    """Transform one JSONL result into HTML and Lean files without changing the source."""

    source_path = Path(result_path)
    records = load_results(source_path)
    summary = calculate_summary(records)
    defaults = default_artifact_paths(source_path, strategy=summary.strategy)
    artifacts = GeneratedArtifacts(
        html_path=Path(html_output_path) if html_output_path is not None else defaults.html_path,
        lean_path=Path(lean_output_path) if lean_output_path is not None else defaults.lean_path,
    )
    resolved_paths = {
        source_path.resolve(),
        artifacts.html_path.resolve(),
        artifacts.lean_path.resolve(),
    }
    if len(resolved_paths) != 3:
        raise ResultViewerError("JSONL, HTML, and Lean paths must be distinct")

    html_document = build_html(records, summary, source_path.name)
    lean_document = build_lean_export(records, summary, source_path.name)
    artifacts.html_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.lean_path.parent.mkdir(parents=True, exist_ok=True)
    with artifacts.html_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(html_document)
    with artifacts.lean_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(lean_document)
    return artifacts


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create offline HTML and Lean result artifacts")
    parser.add_argument("result", help="Path to an existing experiment JSONL result file")
    parser.add_argument("--html-output", help="Explicit HTML report path")
    parser.add_argument("--lean-output", help="Explicit Lean inspection path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        artifacts = write_result_artifacts(
            args.result,
            html_output_path=args.html_output,
            lean_output_path=args.lean_output,
        )
    except (OSError, ResultViewerError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"HTML report: {artifacts.html_path.as_posix()}")
    print(f"Lean export: {artifacts.lean_path.as_posix()}")
    return 0


def _render_result_card(record: dict[str, object]) -> str:
    attempts = _retry_attempts(record)
    selected_record = attempts[-1] if attempts else record
    passed = _benchmark_verified(record)
    benchmark_status = "PASS" if passed else "FAIL"
    benchmark_key = "passed" if passed else "failed"
    verification_status = _verification_status_label(record)
    verification_key = _verification_status_key(record)
    theorem_id = _text(record.get("theorem_id"))
    statement = _text(record.get("statement"))
    search_text = f"{theorem_id} {statement}".casefold()
    proof_output = (
        _text(selected_record.get("proof_output"))
        if "proof_output" in selected_record
        else _text(selected_record.get("normalized_proof"))
    )
    reasoning_output = _text(selected_record.get("reasoning_output"))
    reasoning_html = (
        f"""
  <details>
    <summary>Reasoning</summary>
    <pre><code>{_escape(reasoning_output)}</code></pre>
  </details>""".rstrip()
        if reasoning_output
        else ""
    )
    runner_error = _text(selected_record.get("error"))
    runner_error_html = (
        f"<h4>Runner diagnostic</h4><pre><code>{_escape(runner_error)}</code></pre>"
        if runner_error
        else ""
    )
    prompt_tokens = _token_value(record.get("prompt_tokens"))
    completion_tokens = _token_value(record.get("completion_tokens"))
    total_tokens = (
        prompt_tokens + completion_tokens
        if prompt_tokens is not None and completion_tokens is not None
        else None
    )
    retry_metadata_html = _render_retry_metadata(record, attempts)
    attempts_html = _render_retry_attempts(attempts)
    return f"""
<article class="result-card {benchmark_key}" data-status="{benchmark_key}" data-search="{_escape(search_text)}">
  <header>
    <h2>{_escape(theorem_id)}</h2>
    <span class="status {benchmark_key}">{benchmark_status}</span>
  </header>
  <p class="verification-status">Verifier status: <strong class="verifier-{verification_key}">{_escape(verification_status)}</strong></p>
  <p class="benchmark-result">Benchmark result: <strong>{benchmark_status}</strong></p>
  {retry_metadata_html}
  <dl class="latencies">
    <div><dt>Generation</dt><dd>{_format_latency(record.get("generation_latency_ms"))} ms</dd></div>
    <div><dt>Verification</dt><dd>{_format_latency(record.get("verification_latency_ms"))} ms</dd></div>
    <div><dt>Total</dt><dd>{_format_latency(record.get("total_latency_ms"))} ms</dd></div>
  </dl>
  <dl class="tokens">
    <div><dt>Prompt tokens</dt><dd>{_format_token(prompt_tokens)}</dd></div>
    <div><dt>Completion tokens</dt><dd>{_format_token(completion_tokens)}</dd></div>
    <div><dt>Total tokens</dt><dd>{_format_token(total_tokens)}</dd></div>
  </dl>
  <h3>Theorem statement</h3>
  <pre><code>{_escape(statement)}</code></pre>
  <h3>Proof output</h3>
  <pre><code>{_escape(proof_output)}</code></pre>
  <h3>Normalized Lean proof</h3>
  <pre><code>{_escape(selected_record.get("normalized_proof"))}</code></pre>
  {reasoning_html}
  <details>
    <summary>Raw model output</summary>
    <pre><code>{_escape(selected_record.get("raw_model_output"))}</code></pre>
  </details>
  <details>
    <summary>Lean diagnostics</summary>
    <h4>stdout</h4>
    <pre><code>{_escape(selected_record.get("lean_stdout"))}</code></pre>
    <h4>stderr</h4>
    <pre><code>{_escape(selected_record.get("lean_stderr"))}</code></pre>
    {runner_error_html}
  </details>
  {attempts_html}
</article>""".strip()


def _render_retry_metadata(record: dict[str, object], attempts: Sequence[dict[str, object]]) -> str:
    if not attempts:
        return ""
    return f"""
  <dl class="retry-metadata">
    <div><dt>Strategy</dt><dd>{_escape(record.get("strategy"))}</dd></div>
    <div><dt>Generation budget</dt><dd>{_escape(record.get("generation_budget"))}</dd></div>
    <div><dt>Generations used</dt><dd>{_escape(record.get("generations_used"))}</dd></div>
    <div><dt>Verifier calls</dt><dd>{_escape(record.get("verifier_calls"))}</dd></div>
    <div><dt>Selected attempt</dt><dd>{len(attempts)}</dd></div>
  </dl>""".strip()


def _render_retry_attempts(attempts: Sequence[dict[str, object]]) -> str:
    if not attempts:
        return ""
    attempt_sections = "\n".join(_render_retry_attempt(attempt) for attempt in attempts)
    return f"""
  <details class="retry-attempts">
    <summary>All retry attempts ({len(attempts)})</summary>
    {attempt_sections}
  </details>""".strip()


def _render_retry_attempt(attempt: dict[str, object]) -> str:
    attempt_index = _text(attempt.get("attempt_index"))
    status = _verification_status_label(attempt)
    status_key = _verification_status_key(attempt)
    proof_output = (
        _text(attempt.get("proof_output"))
        if "proof_output" in attempt
        else _text(attempt.get("normalized_proof"))
    )
    reasoning = _text(attempt.get("reasoning_output"))
    reasoning_html = (
        f"<details><summary>Reasoning</summary><pre><code>{_escape(reasoning)}</code></pre></details>"
        if reasoning
        else ""
    )
    prompt_tokens = _token_value(attempt.get("prompt_tokens"))
    completion_tokens = _token_value(attempt.get("completion_tokens"))
    total_tokens = (
        prompt_tokens + completion_tokens
        if prompt_tokens is not None and completion_tokens is not None
        else None
    )
    error = _text(attempt.get("error"))
    return f"""
    <section class="retry-attempt">
      <h4>Attempt {_escape(attempt_index)} — <span class="verifier-{status_key}">{_escape(status)}</span></h4>
      <dl class="latencies">
        <div><dt>Generation</dt><dd>{_format_latency(attempt.get("generation_latency_ms"))} ms</dd></div>
        <div><dt>Verification</dt><dd>{_format_latency(attempt.get("verification_latency_ms"))} ms</dd></div>
        <div><dt>Attempt total</dt><dd>{_format_latency(attempt.get("total_attempt_latency_ms"))} ms</dd></div>
      </dl>
      <dl class="tokens">
        <div><dt>Prompt tokens</dt><dd>{_format_token(prompt_tokens)}</dd></div>
        <div><dt>Completion tokens</dt><dd>{_format_token(completion_tokens)}</dd></div>
        <div><dt>Total tokens</dt><dd>{_format_token(total_tokens)}</dd></div>
      </dl>
      <h5>Proof output</h5>
      <pre><code>{_escape(proof_output)}</code></pre>
      <h5>Normalized Lean proof</h5>
      <pre><code>{_escape(attempt.get("normalized_proof"))}</code></pre>
      {reasoning_html}
      <details><summary>Raw model output</summary><pre><code>{_escape(attempt.get("raw_model_output"))}</code></pre></details>
      <details><summary>Lean diagnostics</summary><h5>stdout</h5><pre><code>{_escape(attempt.get("lean_stdout"))}</code></pre><h5>stderr</h5><pre><code>{_escape(attempt.get("lean_stderr"))}</code></pre><h5>runner error</h5><pre><code>{_escape(error)}</code></pre></details>
    </section>""".strip()


def _render_token_chart(records: Sequence[dict[str, object]]) -> str:
    totals = [_record_total_tokens(record) for record in records]
    available_totals = [value for value in totals if value is not None]
    maximum_total = max(available_totals, default=0)
    rows: list[str] = []
    for record, total in zip(records, totals, strict=True):
        theorem_id = _escape(record.get("theorem_id"))
        prompt = _token_value(record.get("prompt_tokens"))
        completion = _token_value(record.get("completion_tokens"))
        if total is None or maximum_total == 0 or prompt is None or completion is None:
            bar = '<span class="token-unavailable">unavailable</span>'
        else:
            prompt_width = 100.0 * prompt / maximum_total
            completion_width = 100.0 * completion / maximum_total
            bar = (
                '<span class="token-bar" aria-hidden="true">'
                f'<span class="token-prompt" style="width:{prompt_width:.3f}%"></span>'
                f'<span class="token-completion" style="width:{completion_width:.3f}%"></span>'
                "</span>"
                f'<span class="token-count">{total}</span>'
            )
        rows.append(
            f'<div class="token-row"><span class="token-theorem">{theorem_id}</span>{bar}</div>'
        )
    return (
        '<section class="token-chart" aria-label="Token usage by theorem">'
        "<h2>Token usage by theorem</h2>"
        '<p class="token-legend"><span>Prompt</span><span>Completion</span></p>'
        f"{''.join(rows)}"
        "</section>"
    )


def _render_lean_record(record: dict[str, object]) -> list[str]:
    attempts = _retry_attempts(record)
    selected_record = attempts[-1] if attempts else record
    status = _verification_status_label(record)
    benchmark_verified = _benchmark_verified(record)
    theorem_id = _lean_comment_text(record.get("theorem_id"))
    statement = _text(record.get("statement"))
    normalized_proof = _text(selected_record.get("normalized_proof"))
    error = _text(selected_record.get("error"))
    retry_metadata = (
        [
            f"strategy: {_lean_comment_text(record.get('strategy'))}",
            f"selected_attempt: {len(attempts)} of {_lean_comment_text(record.get('generations_used'))} used",
            f"generation_budget: {_lean_comment_text(record.get('generation_budget'))}",
        ]
        if attempts
        else []
    )
    if error.startswith("generation_error:") and not normalized_proof:
        return [
            "/-",
            f"theorem_id: {theorem_id}",
            *retry_metadata,
            f"status: {status}",
            f"benchmark_verified: {str(benchmark_verified).lower()}",
            "No normalized proof was produced because generation failed.",
            "statement:",
            _lean_comment_text(statement),
            "error category:",
            _lean_comment_text(_safe_error_category(error)),
            "-/",
            "",
        ]
    return [
        "/-",
        f"theorem_id: {theorem_id}",
        *retry_metadata,
        f"status: {status}",
        f"benchmark_verified: {str(benchmark_verified).lower()}",
        "-/",
        "",
        f"{statement} := {normalized_proof}",
        "",
    ]


def _average_latency(records: Sequence[dict[str, object]], field: str) -> float:
    return sum(_numeric_value(record.get(field)) for record in records) / len(records)


def _numeric_value(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _format_latency(value: object) -> str:
    return f"{_numeric_value(value):.1f}"


def _token_value(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _record_total_tokens(record: dict[str, object]) -> int | None:
    prompt = _token_value(record.get("prompt_tokens"))
    completion = _token_value(record.get("completion_tokens"))
    if prompt is None or completion is None:
        return None
    return prompt + completion


def _optional_sum(values: Sequence[int]) -> int | None:
    return sum(values) if values else None


def _optional_average(values: Sequence[int]) -> float | None:
    return sum(values) / len(values) if values else None


def _format_token(value: int | None) -> str:
    return str(value) if value is not None else "unavailable"


def _format_optional_number(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "unavailable"


def _benchmark_verified(record: dict[str, object]) -> bool:
    if _text(record.get("strategy")) == "retry":
        return record.get("solved") is True
    status = record.get("verification_status")
    if isinstance(status, str) and status:
        return status.lower() == "verified"
    return record.get("verified") is True


def _verification_status_label(record: dict[str, object]) -> str:
    status_field = (
        "final_verification_status"
        if _text(record.get("strategy")) == "retry"
        else "verification_status"
    )
    status = record.get(status_field)
    if isinstance(status, str) and status:
        return status.upper()
    attempts = _retry_attempts(record)
    selected_record = attempts[-1] if attempts else record
    if status_field in record and _text(selected_record.get("error")).startswith(
        "generation_error:"
    ):
        return "NOT RUN"
    if status_field not in record and record.get("verified") is True:
        return "VERIFIED (LEGACY)"
    return "UNKNOWN / LEGACY"


def _verification_status_key(record: dict[str, object]) -> str:
    status_field = (
        "final_verification_status"
        if _text(record.get("strategy")) == "retry"
        else "verification_status"
    )
    status = record.get(status_field)
    if isinstance(status, str) and status in {
        "verified",
        "incomplete",
        "rejected",
        "timeout",
        "execution_error",
    }:
        return status.replace("_", "-")
    return "unknown"


def _retry_attempts(record: dict[str, object]) -> list[dict[str, object]]:
    if _text(record.get("strategy")) != "retry":
        return []
    attempts = record.get("attempts")
    if not isinstance(attempts, list):
        return []
    return [attempt for attempt in attempts if isinstance(attempt, dict)]


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _escape(value: object) -> str:
    return html.escape(_text(value), quote=True)


def _lean_comment_text(value: object) -> str:
    return _text(value).replace("/-", "/ -").replace("-/", "- /")


def _safe_error_category(error: str) -> str:
    return ": ".join(part.strip() for part in error.split(":", maxsplit=2)[:2])


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
    body { max-width: 1100px; margin: 0 auto; padding: 2rem; background: #f5f7fa; color: #172033; }
    h1 { margin-bottom: .25rem; }
    .source { color: #596579; margin-top: 0; }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: .75rem; margin: 1.5rem 0; }
    .summary div { display: grid; gap: .25rem; padding: .9rem; border: 1px solid #d8dee9; border-radius: .5rem; background: white; }
    .summary span, dt { color: #667085; font-size: .82rem; }
    .controls { display: flex; flex-wrap: wrap; gap: .75rem; align-items: center; margin: 1.5rem 0; }
    .controls input { flex: 1 1 280px; padding: .65rem; border: 1px solid #aeb7c5; border-radius: .4rem; }
    button { padding: .6rem .9rem; border: 1px solid #8792a2; border-radius: .4rem; background: white; color: inherit; cursor: pointer; }
    button.active { background: #1f5eff; border-color: #1f5eff; color: white; }
    .result-card { margin: 1rem 0; padding: 1.25rem; border: 1px solid #d8dee9; border-left-width: .4rem; border-radius: .55rem; background: white; }
    .result-card.passed { border-left-color: #198754; }
    .result-card.failed { border-left-color: #c9362b; }
    .result-card header { display: flex; justify-content: space-between; gap: 1rem; align-items: center; }
    .result-card h2 { margin: 0; font-size: 1.15rem; overflow-wrap: anywhere; }
    .status { padding: .25rem .55rem; border-radius: 999px; color: white; font-weight: 700; }
    .status.passed { background: #198754; }
    .status.failed { background: #c9362b; }
    .verification-status, .benchmark-result { margin: .5rem 0; }
    .verifier-verified { color: #198754; }
    .verifier-incomplete { color: #ad6500; }
    .verifier-rejected, .verifier-timeout, .verifier-execution-error { color: #c9362b; }
    .verifier-unknown { color: #667085; }
    .latencies, .tokens, .retry-metadata { display: flex; flex-wrap: wrap; gap: 1.25rem; margin: .8rem 0; }
    .latencies div, .tokens div, .retry-metadata div { display: flex; gap: .35rem; }
    .latencies dd, .tokens dd, .retry-metadata dd { margin: 0; }
    .retry-attempts { margin-top: 1rem; }
    .retry-attempt { margin-top: 1rem; padding: 1rem; border: 1px solid #d8dee9; border-radius: .45rem; }
    .retry-attempt h4 { margin-top: 0; }
    .token-chart { margin: 1.5rem 0; padding: 1rem; border: 1px solid #d8dee9; border-radius: .5rem; background: white; }
    .token-chart h2 { margin-top: 0; }
    .token-legend { display: flex; gap: 1rem; font-size: .85rem; }
    .token-legend span:first-child::before, .token-legend span:last-child::before { content: ""; display: inline-block; width: .75rem; height: .75rem; margin-right: .3rem; }
    .token-legend span:first-child::before { background: #4f7cff; }
    .token-legend span:last-child::before { background: #8b5cf6; }
    .token-row { display: grid; grid-template-columns: minmax(120px, 1fr) 3fr auto; gap: .65rem; align-items: center; margin: .45rem 0; }
    .token-theorem { overflow-wrap: anywhere; }
    .token-bar { display: flex; height: .8rem; border-radius: 999px; overflow: hidden; background: #e5e7eb; }
    .token-prompt { background: #4f7cff; }
    .token-completion { background: #8b5cf6; }
    .token-count, .token-unavailable { font-variant-numeric: tabular-nums; color: #667085; }
    pre { overflow-x: auto; padding: .85rem; border-radius: .4rem; background: #111827; color: #e5e7eb; white-space: pre-wrap; overflow-wrap: anywhere; }
    details { margin-top: .75rem; }
    summary { cursor: pointer; font-weight: 650; }
    [hidden] { display: none !important; }
    @media (prefers-color-scheme: dark) {
      body { background: #111827; color: #e5e7eb; }
      .summary div, button, .result-card, .token-chart { background: #1f2937; border-color: #4b5563; }
      .source, .summary span, dt { color: #aeb7c5; }
    }
  </style>
</head>
<body>
  <h1>Lean Prover Agent results</h1>
  <p class="source">Source: __SOURCE__</p>
  __SUMMARY__
  __TOKEN_CHART__
  <section class="controls" aria-label="Result filters">
    <button type="button" class="active" data-filter="all">All</button>
    <button type="button" data-filter="passed">Passed</button>
    <button type="button" data-filter="failed">Failed</button>
    <input id="result-search" type="search" placeholder="Search theorem ID or statement" aria-label="Search results">
    <span id="visible-count" aria-live="polite"></span>
  </section>
  <main id="results">
    __CARDS__
  </main>
  <script>
    const cards = Array.from(document.querySelectorAll(".result-card"));
    const buttons = Array.from(document.querySelectorAll("[data-filter]"));
    const search = document.getElementById("result-search");
    const visibleCount = document.getElementById("visible-count");
    let activeFilter = "all";

    function applyFilters() {
      const query = search.value.trim().toLocaleLowerCase();
      let visible = 0;
      for (const card of cards) {
        const statusMatches = activeFilter === "all" || card.dataset.status === activeFilter;
        const searchMatches = !query || card.dataset.search.includes(query);
        card.hidden = !(statusMatches && searchMatches);
        if (!card.hidden) visible += 1;
      }
      visibleCount.textContent = `${visible} of ${cards.length}`;
    }

    for (const button of buttons) {
      button.addEventListener("click", () => {
        activeFilter = button.dataset.filter;
        for (const candidate of buttons) candidate.classList.toggle("active", candidate === button);
        applyFilters();
      });
    }
    search.addEventListener("input", applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
