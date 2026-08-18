from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from leanproof.model import (
    BASELINE_PROMPT_TEMPLATE,
    ConfigurationError,
    LLMConfig,
    OpenAICompatibleProofModel,
    ProofGenerationError,
    build_baseline_prompt,
    normalize_proof,
    split_leading_think_block,
)


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ("by\n  exact h", "by\n  exact h"),
        ("  by\n  exact h\n", "by\n  exact h"),
        ("```lean\nby\n  exact h\n```", "by\n  exact h"),
        ("\n```Lean\nby\n  exact h\n```\n", "by\n  exact h"),
        ("```\nby\n  exact h\n```", "by\n  exact h"),
    ],
)
def test_normalize_proof(raw_output: str, expected: str) -> None:
    assert normalize_proof(raw_output) == expected


def test_baseline_prompt_is_fixed_and_contains_only_statement() -> None:
    statement = "example (p : Prop) (h : p) : p"

    prompt = build_baseline_prompt(statement)

    assert prompt == BASELINE_PROMPT_TEMPLATE.format(statement=statement)
    assert prompt.endswith(statement)
    assert "previous" not in prompt.lower()
    assert "feedback" not in prompt.lower()


def test_config_validates_and_normalizes_direct_values() -> None:
    config = LLMConfig(
        api_key=" environment-key ",
        base_url=" https://file.example.com/v1/ ",
        model=" file-model ",
    )

    assert config.api_key == "environment-key"
    assert config.base_url == "https://file.example.com/v1"
    assert config.model == "file-model"
    assert config.reasoning_split is False


def test_config_fails_fast_when_required_value_is_missing() -> None:
    with pytest.raises(ConfigurationError, match="api_key"):
        LLMConfig("", "https://api.example.com/v1", "test-model")


def test_config_rejects_invalid_base_url() -> None:
    with pytest.raises(ConfigurationError, match="base_url"):
        LLMConfig("test-key", "not-a-url", "test-model")


def test_model_makes_one_request_and_preserves_raw_output() -> None:
    completions = FakeCompletions("  ```lean\nby\n  exact h\n```  ")
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = LLMConfig("test-key", "https://api.example.com/v1", "test-model")
    model = OpenAICompatibleProofModel(config, client=client)
    statement = "example (p : Prop) (h : p) : p"

    result = model.generate_proof(statement)

    assert result.raw_output == "  ```lean\nby\n  exact h\n```  "
    assert result.proof_output == result.raw_output
    assert result.reasoning_output is None
    assert normalize_proof(result.proof_output) == "by\n  exact h"
    assert result.prompt_tokens == 31
    assert result.completion_tokens == 7
    assert len(completions.calls) == 1
    assert completions.calls[0] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": build_baseline_prompt(statement)}],
    }


def test_reasoning_split_configuration_is_explicit() -> None:
    config = LLMConfig(
        "test-key",
        "https://api.example.com/v1",
        "test-model",
        reasoning_split=True,
    )
    completions = FakeCompletions("by\n  trivial")
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    OpenAICompatibleProofModel(config, client=client).generate_proof("example : True")

    assert config.reasoning_split is True
    assert completions.calls[0]["extra_body"] == {"reasoning_split": True}


def test_disabled_reasoning_split_does_not_send_provider_option() -> None:
    completions = FakeCompletions("by\n  trivial")
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = LLMConfig("test-key", "https://api.example.com/v1", "test-model")

    OpenAICompatibleProofModel(config, client=client).generate_proof("example : True")

    assert "extra_body" not in completions.calls[0]


def test_leading_think_block_is_split_without_changing_raw_output() -> None:
    raw_output = "<think>\nreasoning text\n</think>\n\nby\n  ring"
    completions = FakeCompletions(raw_output)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = LLMConfig("test-key", "https://api.example.com/v1", "test-model")

    result = OpenAICompatibleProofModel(config, client=client).generate_proof("example : True")

    assert result.raw_output == raw_output
    assert result.reasoning_output == "reasoning text"
    assert result.proof_output == "by\n  ring"


def test_malformed_think_block_is_preserved_conservatively() -> None:
    raw_output = "<think>unfinished reasoning\nby\n  ring"

    proof_output, reasoning_output = split_leading_think_block(raw_output)

    assert proof_output == raw_output
    assert reasoning_output is None


def test_dedicated_reasoning_details_are_preserved() -> None:
    completions = FakeCompletions(
        "by\n  ring",
        reasoning_details=[{"type": "reasoning.text", "text": "dedicated reasoning"}],
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = LLMConfig(
        "test-key",
        "https://api.example.com/v1",
        "test-model",
        reasoning_split=True,
    )

    result = OpenAICompatibleProofModel(config, client=client).generate_proof("example : True")

    assert result.raw_output == "by\n  ring"
    assert result.proof_output == "by\n  ring"
    assert result.reasoning_output == "dedicated reasoning"


def test_default_client_disables_sdk_retries() -> None:
    config = LLMConfig("test-key", "https://api.example.com/v1", "test-model")

    with patch("leanproof.model.OpenAI") as openai_class:
        OpenAICompatibleProofModel(config)

    openai_class.assert_called_once_with(
        api_key="test-key",
        base_url="https://api.example.com/v1",
        max_retries=0,
        timeout=60.0,
    )


def test_model_rejects_completion_without_text() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=None
    )
    completions = SimpleNamespace(create=lambda **_: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = LLMConfig("test-key", "https://api.example.com/v1", "test-model")

    with pytest.raises(ProofGenerationError, match="did not contain text"):
        OpenAICompatibleProofModel(config, client=client).generate_proof("example : True")


class FakeCompletions:
    def __init__(self, raw_output: str, *, reasoning_details=None) -> None:
        self.raw_output = raw_output
        self.reasoning_details = reasoning_details
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.raw_output,
                        reasoning_details=self.reasoning_details,
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=31, completion_tokens=7),
        )
