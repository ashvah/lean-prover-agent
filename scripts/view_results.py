"""Generate a self-contained HTML viewer for a one-shot JSONL result file."""

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

    model: str
    total: int
    solved: int
    failed: int
    solve_rate: float
    average_generation_latency_ms: float
    average_verification_latency_ms: float
    average_total_latency_ms: float


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

    models = sorted({_text(record.get("model")) or "unknown" for record in records})
    model = models[0] if len(models) == 1 else ", ".join(models)
    total = len(records)
    solved = sum(record.get("verified") is True for record in records)
    return ResultSummary(
        model=model,
        total=total,
        solved=solved,
        failed=total - solved,
        solve_rate=100.0 * solved / total,
        average_generation_latency_ms=_average_latency(records, "generation_latency_ms"),
        average_verification_latency_ms=_average_latency(records, "verification_latency_ms"),
        average_total_latency_ms=_average_latency(records, "total_latency_ms"),
    )


def build_html(
    records: Sequence[dict[str, object]], summary: ResultSummary, source_name: str
) -> str:
    """Render escaped result records into a self-contained HTML document."""

    cards = "\n".join(_render_result_card(record) for record in records)
    summary_html = f"""
<section class="summary" aria-label="Experiment summary">
  <div><span>Model</span><strong>{_escape(summary.model)}</strong></div>
  <div><span>Total</span><strong>{summary.total}</strong></div>
  <div><span>Solved</span><strong>{summary.solved}</strong></div>
  <div><span>Failed</span><strong>{summary.failed}</strong></div>
  <div><span>Solve rate</span><strong>{summary.solve_rate:.1f}%</strong></div>
  <div><span>Avg generation</span><strong>{summary.average_generation_latency_ms:.1f} ms</strong></div>
  <div><span>Avg verification</span><strong>{summary.average_verification_latency_ms:.1f} ms</strong></div>
  <div><span>Avg total</span><strong>{summary.average_total_latency_ms:.1f} ms</strong></div>
</section>""".strip()
    replacements = {
        "TITLE": _escape(f"LeanProof results — {source_name}"),
        "SOURCE": _escape(source_name),
        "SUMMARY": summary_html,
        "CARDS": cards,
    }
    return re.sub(
        r"__(TITLE|SOURCE|SUMMARY|CARDS)__",
        lambda match: replacements[match.group(1)],
        _HTML_TEMPLATE,
    )


def write_result_viewer(result_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Read a JSONL result and write a sibling HTML file, leaving JSONL unchanged."""

    source_path = Path(result_path)
    destination = Path(output_path) if output_path is not None else source_path.with_suffix(".html")
    if source_path.resolve() == destination.resolve():
        raise ResultViewerError("HTML output path must differ from the JSONL source path")

    records = load_results(source_path)
    summary = calculate_summary(records)
    document = build_html(records, summary, source_path.name)
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        output.write(document)
    return destination


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an offline HTML JSONL result viewer")
    parser.add_argument("result", help="Path to an existing one-shot JSONL result file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        output_path = write_result_viewer(args.result)
    except (OSError, ResultViewerError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"HTML viewer: {output_path.as_posix()}")
    return 0


def _render_result_card(record: dict[str, object]) -> str:
    passed = record.get("verified") is True
    status = "PASS" if passed else "FAIL"
    status_key = "passed" if passed else "failed"
    theorem_id = _text(record.get("theorem_id"))
    statement = _text(record.get("statement"))
    search_text = f"{theorem_id} {statement}".casefold()
    runner_error = _text(record.get("error"))
    runner_error_html = (
        f"<h4>Runner diagnostic</h4><pre><code>{_escape(runner_error)}</code></pre>"
        if runner_error
        else ""
    )
    return f"""
<article class="result-card {status_key}" data-status="{status_key}" data-search="{_escape(search_text)}">
  <header>
    <h2>{_escape(theorem_id)}</h2>
    <span class="status {status_key}">{status}</span>
  </header>
  <dl class="latencies">
    <div><dt>Generation</dt><dd>{_format_latency(record.get("generation_latency_ms"))} ms</dd></div>
    <div><dt>Verification</dt><dd>{_format_latency(record.get("verification_latency_ms"))} ms</dd></div>
    <div><dt>Total</dt><dd>{_format_latency(record.get("total_latency_ms"))} ms</dd></div>
  </dl>
  <h3>Theorem statement</h3>
  <pre><code>{_escape(statement)}</code></pre>
  <h3>Normalized Lean proof</h3>
  <pre><code>{_escape(record.get("normalized_proof"))}</code></pre>
  <details>
    <summary>Raw model output</summary>
    <pre><code>{_escape(record.get("raw_model_output"))}</code></pre>
  </details>
  <details>
    <summary>Lean stdout/stderr</summary>
    <h4>stdout</h4>
    <pre><code>{_escape(record.get("lean_stdout"))}</code></pre>
    <h4>stderr</h4>
    <pre><code>{_escape(record.get("lean_stderr"))}</code></pre>
    {runner_error_html}
  </details>
</article>""".strip()


def _average_latency(records: Sequence[dict[str, object]], field: str) -> float:
    return sum(_numeric_value(record.get(field)) for record in records) / len(records)


def _numeric_value(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _format_latency(value: object) -> str:
    return f"{_numeric_value(value):.1f}"


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _escape(value: object) -> str:
    return html.escape(_text(value), quote=True)


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
    .latencies { display: flex; flex-wrap: wrap; gap: 1.25rem; margin: .8rem 0; }
    .latencies div { display: flex; gap: .35rem; }
    .latencies dd { margin: 0; }
    pre { overflow-x: auto; padding: .85rem; border-radius: .4rem; background: #111827; color: #e5e7eb; white-space: pre-wrap; overflow-wrap: anywhere; }
    details { margin-top: .75rem; }
    summary { cursor: pointer; font-weight: 650; }
    [hidden] { display: none !important; }
    @media (prefers-color-scheme: dark) {
      body { background: #111827; color: #e5e7eb; }
      .summary div, button, .result-card { background: #1f2937; border-color: #4b5563; }
      .source, .summary span, dt { color: #aeb7c5; }
    }
  </style>
</head>
<body>
  <h1>LeanProof-Agent results</h1>
  <p class="source">Source: __SOURCE__</p>
  __SUMMARY__
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
