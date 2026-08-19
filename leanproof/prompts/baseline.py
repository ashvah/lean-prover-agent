"""Visible output contracts for the one-shot and independent-retry baselines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ReasoningMode = Literal["prompted", "none"]

PROMPTED_BASELINE_PROMPT_TEMPLATE = """Complete the following Lean 4 theorem.

Before writing the proof, produce a concise proof plan.

Return exactly the following two blocks and nothing else:

<plan>
A concise high-level plan for proving the theorem.
</plan>
<proof>
by
  ...
</proof>

Plan requirements:
- Keep the plan concise: normally 1-4 short sentences.
- Describe the main proof idea rather than a detailed step-by-step derivation.
- Mention important reductions, case splits, induction principles, lemmas, or Lean tactics only when they are genuinely relevant.
- Do not include the final Lean proof inside the plan.
- Do not include hidden reasoning, scratch work, self-critique, or alternative attempts.

Proof requirements:
- The <proof> block must contain exactly one complete Lean 4 proof term.
- The first non-whitespace token inside <proof> must be `by`.
- The proof must be valid in an environment with Mathlib imported.
- Do not use `sorry` or `admit`.
- Do not repeat the theorem statement.
- Do not use Markdown code fences.
- Do not include explanations or text outside <plan> and <proof>.

Theorem:

{statement}"""

NONE_BASELINE_PROMPT_TEMPLATE = """Complete the following Lean 4 theorem.

Return exactly the following block and nothing else:

<proof>
by
  ...
</proof>

Proof requirements:
- The <proof> block must contain exactly one complete Lean 4 proof term.
- The first non-whitespace token inside <proof> must be `by`.
- The proof must be valid in an environment with Mathlib imported.
- Do not use `sorry` or `admit`.
- Do not repeat the theorem statement.
- Do not use Markdown code fences.
- Do not include a plan, explanation, analysis, commentary, or text outside <proof>.

Theorem:

{statement}"""

_PROMPTED_OUTPUT_PATTERN = re.compile(
    r"\A\s*<plan>(?P<plan>.*?)</plan>\s*<proof>(?P<proof>.*?)</proof>\s*\Z",
    flags=re.DOTALL,
)
_NONE_OUTPUT_PATTERN = re.compile(
    r"\A\s*<proof>(?P<proof>.*?)</proof>\s*\Z",
    flags=re.DOTALL,
)
_CONTRACT_TAGS = ("<plan>", "</plan>", "<proof>", "</proof>")


class PromptContractError(ValueError):
    """Raised when visible model output contains a malformed baseline wrapper."""


@dataclass(frozen=True)
class ParsedBaselineOutput:
    """Agent-visible plan and proof extracted from one baseline response."""

    plan_output: str | None
    proof_output: str


def build_baseline_prompt(statement: str, reasoning_mode: ReasoningMode = "none") -> str:
    """Build the baseline prompt for one explicit experiment reasoning condition."""

    mode = validate_reasoning_mode(reasoning_mode)
    template = (
        PROMPTED_BASELINE_PROMPT_TEMPLATE if mode == "prompted" else NONE_BASELINE_PROMPT_TEMPLATE
    )
    return template.format(statement=statement)


def parse_baseline_output(
    visible_output: str, reasoning_mode: ReasoningMode
) -> ParsedBaselineOutput:
    """Parse exact plan/proof wrappers, with fallback only for unwrapped legacy output."""

    mode = validate_reasoning_mode(reasoning_mode)
    if mode == "prompted":
        match = _PROMPTED_OUTPUT_PATTERN.fullmatch(visible_output)
        if match is not None and _has_exact_tag_counts(visible_output, include_plan=True):
            return ParsedBaselineOutput(
                plan_output=match.group("plan").strip(),
                proof_output=match.group("proof").strip(),
            )
    else:
        match = _NONE_OUTPUT_PATTERN.fullmatch(visible_output)
        if match is not None and _has_exact_tag_counts(visible_output, include_plan=False):
            return ParsedBaselineOutput(
                plan_output=None,
                proof_output=match.group("proof").strip(),
            )

    if any(tag in visible_output for tag in _CONTRACT_TAGS):
        raise PromptContractError(f"Malformed {mode} baseline output wrapper")
    return ParsedBaselineOutput(plan_output=None, proof_output=visible_output)


def validate_reasoning_mode(value: str) -> ReasoningMode:
    """Validate the only two Agent-visible reasoning experiment modes."""

    if value not in {"prompted", "none"}:
        raise ValueError("reasoning_mode must be one of: prompted, none")
    return value


def _has_exact_tag_counts(value: str, *, include_plan: bool) -> bool:
    proof_counts_are_exact = value.count("<proof>") == 1 and value.count("</proof>") == 1
    if not proof_counts_are_exact:
        return False
    expected_plan_count = 1 if include_plan else 0
    return (
        value.count("<plan>") == expected_plan_count
        and value.count("</plan>") == expected_plan_count
    )
