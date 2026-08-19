from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, Timeout

from leanproof.models.model import (
    ConfigurationError,
    GenerationRequestError,
    LLMConfig,
    OpenAICompatibleProofModel,
    ProofGenerationError,
    classify_request_failure,
    normalize_proof,
    split_leading_think_block,
)
from leanproof.prompts import build_baseline_prompt


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ("by\n  exact h", "by\n  exact h"),
        ("  by\n  exact h\n", "by\n  exact h"),
        ("```lean\nby\n  exact h\n```", "by\n  exact h"),
        ("\n```Lean\nby\n  exact h\n```\n", "by\n  exact h"),
        ("```Lean4\nby\n  exact h\n```", "by\n  exact h"),
        ("```big\nby\n  exact h\n```", "by\n  exact h"),
        ("```whatever\nby\n  exact h\n```", "by\n  exact h"),
        ("```\nby\n  exact h\n```", "by\n  exact h"),
        ("Here is the proof:\n\n```lean\nby\n  exact h\n```", "by\n  exact h"),
    ],
)
def test_normalize_proof(raw_output: str, expected: str) -> None:
    assert normalize_proof(raw_output) == expected


def test_normalize_proof_preserves_observed_arbitrary_fence_payload_exactly() -> None:
    raw_output = """```big
by
  use fun i => 1
  intro k hk
  simp [Finset.sum_const, Finset.card_range]
  <;> field_simp
  <;> ring
```"""

    assert (
        normalize_proof(raw_output)
        == """by
  use fun i => 1
  intro k hk
  simp [Finset.sum_const, Finset.card_range]
  <;> field_simp
  <;> ring"""
    )


def test_normalize_proof_rejects_ambiguous_or_non_proof_output() -> None:
    ambiguous = """```lean
by
  exact h
```
text
```other
by
  assumption
```"""

    with pytest.raises(ProofGenerationError, match="multiple plausible"):
        normalize_proof(ambiguous)
    with pytest.raises(ProofGenerationError, match="does not contain"):
        normalize_proof("I cannot solve this theorem.")


def test_normalize_proof_does_not_repair_proof_contents() -> None:
    proof = "by\n  exact nonexistent_theorem\n  )"

    assert normalize_proof(proof) == proof


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
    assert config.connect_timeout_seconds == 10.0
    assert config.read_timeout_seconds == 300.0
    assert config.write_timeout_seconds == 30.0
    assert config.pool_timeout_seconds == 10.0


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
    assert result.native_reasoning_output is None
    assert result.plan_output is None
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
    assert result.native_reasoning_output == "reasoning text"
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
    assert result.native_reasoning_output == "dedicated reasoning"


def test_provider_reasoning_and_agent_visible_plan_remain_separate() -> None:
    raw_output = (
        "<think>native scratch work</think>"
        "<plan>Apply ring normalization.</plan>"
        "<proof>by\n  ring</proof>"
    )
    completions = FakeCompletions(raw_output)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = LLMConfig("test-key", "https://api.example.com/v1", "test-model")

    result = OpenAICompatibleProofModel(config, client=client).generate_proof(
        "example (x : ℝ) : x = x", reasoning_mode="prompted"
    )

    assert result.raw_output == raw_output
    assert result.native_reasoning_output == "native scratch work"
    assert result.plan_output == "Apply ring normalization."
    assert result.proof_output == "by\n  ring"


def test_default_client_disables_sdk_retries() -> None:
    config = LLMConfig("test-key", "https://api.example.com/v1", "test-model")

    with patch("leanproof.models.model.OpenAI") as openai_class:
        OpenAICompatibleProofModel(config)

    call = openai_class.call_args.kwargs
    assert call["api_key"] == "test-key"
    assert call["base_url"] == "https://api.example.com/v1"
    assert call["max_retries"] == 0
    assert isinstance(call["timeout"], Timeout)
    assert call["timeout"].connect == 10.0
    assert call["timeout"].read == 300.0
    assert call["timeout"].write == 30.0
    assert call["timeout"].pool == 10.0


def test_model_rejects_completion_without_text() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=None
    )
    completions = SimpleNamespace(create=lambda **_: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = LLMConfig("test-key", "https://api.example.com/v1", "test-model")

    with pytest.raises(ProofGenerationError, match="did not contain text"):
        OpenAICompatibleProofModel(config, client=client).generate_proof("example : True")


def test_model_wraps_transport_failure_with_low_level_cause() -> None:
    request = httpx2.Request("POST", "https://api.example.com/v1/chat/completions")
    try:
        raise httpx2.ConnectTimeout("connection timed out", request=request)
    except httpx2.ConnectTimeout as cause:
        try:
            raise APIConnectionError(request=request) from cause
        except APIConnectionError as error:
            provider_error = error
    completions = RaisingCompletions(provider_error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = LLMConfig("secret-key", "https://api.example.com/v1", "test-model")

    with pytest.raises(GenerationRequestError) as captured:
        OpenAICompatibleProofModel(config, client=client).generate_proof("example : True")

    assert captured.value.transport is True
    assert captured.value.retryable is True
    assert captured.value.details.stage == "generation_request"
    assert captured.value.details.type == "APIConnectionError"
    assert captured.value.details.cause_type == "ConnectTimeout"
    assert "secret-key" not in captured.value.details.message


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [
        (408, True),
        (409, True),
        (429, True),
        (500, True),
        (503, True),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (422, False),
    ],
)
def test_status_failure_classification(status_code: int, retryable: bool) -> None:
    request = httpx2.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx2.Response(status_code, request=request)
    classification = classify_request_failure(
        APIStatusError("request failed", response=response, body=None)
    )

    assert classification.retryable is retryable
    assert classification.transport is False
    assert classification.transient_api is retryable


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


class RaisingCompletions:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def create(self, **kwargs):
        raise self.error
