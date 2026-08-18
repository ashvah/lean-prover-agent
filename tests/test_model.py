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


def test_config_loads_dotenv_without_overriding_environment(tmp_path, monkeypatch) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "LLM_API_KEY=file-key\nLLM_BASE_URL=https://file.example.com/v1/\nLLM_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_API_KEY", "environment-key")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    config = LLMConfig.from_env(dotenv_path)

    assert config.api_key == "environment-key"
    assert config.base_url == "https://file.example.com/v1"
    assert config.model == "file-model"


def test_config_fails_fast_when_required_value_is_missing(tmp_path, monkeypatch) -> None:
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        LLMConfig.from_env(tmp_path / "missing.env")


def test_config_rejects_invalid_base_url(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "not-a-url")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    with pytest.raises(ConfigurationError, match="LLM_BASE_URL"):
        LLMConfig.from_env(tmp_path / "missing.env")


def test_model_makes_one_request_and_preserves_raw_output() -> None:
    completions = FakeCompletions("  ```lean\nby\n  exact h\n```  ")
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = LLMConfig("test-key", "https://api.example.com/v1", "test-model")
    model = OpenAICompatibleProofModel(config, client=client)
    statement = "example (p : Prop) (h : p) : p"

    result = model.generate_proof(statement)

    assert result.raw_output == "  ```lean\nby\n  exact h\n```  "
    assert result.prompt_tokens == 31
    assert result.completion_tokens == 7
    assert len(completions.calls) == 1
    assert completions.calls[0] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": build_baseline_prompt(statement)}],
    }


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
    def __init__(self, raw_output: str) -> None:
        self.raw_output = raw_output
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.raw_output))],
            usage=SimpleNamespace(prompt_tokens=31, completion_tokens=7),
        )
