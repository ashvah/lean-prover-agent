from __future__ import annotations

import pytest

from leanproof.prompts import (
    NONE_BASELINE_PROMPT_TEMPLATE,
    PROMPTED_BASELINE_PROMPT_TEMPLATE,
    PromptContractError,
    build_baseline_prompt,
    parse_baseline_output,
)


def test_prompted_prompt_requires_concise_plan_and_proof_blocks() -> None:
    statement = "example (p : Prop) (h : p) : p"

    prompt = build_baseline_prompt(statement, "prompted")

    assert prompt == PROMPTED_BASELINE_PROMPT_TEMPLATE.format(statement=statement)
    assert "<plan>" in prompt
    assert "1-4 short sentences" in prompt
    assert "<proof>" in prompt
    assert "sorry" in prompt
    assert prompt.endswith(statement)


def test_none_prompt_requires_only_proof_block() -> None:
    statement = "example : True"

    prompt = build_baseline_prompt(statement, "none")

    assert prompt == NONE_BASELINE_PROMPT_TEMPLATE.format(statement=statement)
    assert "Do not include a plan" in prompt
    assert prompt.endswith(statement)


def test_prompted_output_separates_agent_visible_plan_from_proof() -> None:
    parsed = parse_baseline_output(
        "<plan>Use the hypothesis directly.</plan><proof>by\n  exact h</proof>",
        "prompted",
    )

    assert parsed.plan_output == "Use the hypothesis directly."
    assert parsed.proof_output == "by\n  exact h"


def test_none_output_extracts_only_proof() -> None:
    parsed = parse_baseline_output("<proof>\nby\n  trivial\n</proof>", "none")

    assert parsed.plan_output is None
    assert parsed.proof_output == "by\n  trivial"


@pytest.mark.parametrize(
    "output",
    [
        "<plan>idea</plan><proof>by trivial",
        "<plan>idea</plan><proof>by trivial</proof>extra",
        "<plan>idea</plan><proof>by trivial</proof><proof>by trivial</proof>",
    ],
)
def test_malformed_contract_tags_are_rejected(output: str) -> None:
    with pytest.raises(PromptContractError):
        parse_baseline_output(output, "prompted")


def test_unwrapped_legacy_proof_is_preserved_for_normalization() -> None:
    output = "```lean\nby\n  trivial\n```"

    parsed = parse_baseline_output(output, "none")

    assert parsed.plan_output is None
    assert parsed.proof_output == output
