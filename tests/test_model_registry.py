from __future__ import annotations

import pytest

from leanproof.model import ConfigurationError, LLMConfig
from leanproof.model_registry import ModelRegistry


def test_multiple_complete_configurations_are_registered_and_sorted(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL_ALIASES", "beta,alpha,unused")
    _set_model(monkeypatch, "ALPHA", "alpha-key", "alpha-model", reasoning="true")
    _set_model(monkeypatch, "BETA", "beta-key", "beta-model", reasoning="false")
    for suffix in ("API_KEY", "BASE_URL", "MODEL", "REASONING_SPLIT"):
        monkeypatch.delenv(f"UNUSED_{suffix}", raising=False)

    registry = ModelRegistry.from_env(tmp_path / "missing.env")

    assert registry.names() == ("alpha", "beta")
    assert registry.get("alpha").model == "alpha-model"
    assert registry.get("alpha").reasoning_split is True
    assert registry.get("beta").reasoning_split is False


def test_incomplete_configuration_fails_without_exposing_credentials(tmp_path, monkeypatch) -> None:
    secret = "private-partial-key"
    monkeypatch.setenv("LLM_MODEL_ALIASES", "partial")
    monkeypatch.setenv("PARTIAL_API_KEY", secret)
    monkeypatch.delenv("PARTIAL_BASE_URL", raising=False)
    monkeypatch.delenv("PARTIAL_MODEL", raising=False)

    with pytest.raises(ConfigurationError) as captured:
        ModelRegistry.from_env(tmp_path / "missing.env")

    message = str(captured.value)
    assert "Incomplete model configuration for alias 'partial'" in message
    assert "PARTIAL_BASE_URL" in message
    assert "PARTIAL_MODEL" in message
    assert secret not in message


def test_unknown_alias_lists_available_models_without_credentials() -> None:
    secret = "private-registry-key"
    registry = ModelRegistry(
        {
            "alpha": LLMConfig(secret, "https://api.example.com/v1", "provider-alpha"),
            "beta": LLMConfig("other-key", "https://api.example.com/v1", "provider-beta"),
        }
    )

    with pytest.raises(
        ConfigurationError,
        match=r"Unknown model 'missing'\. Available models: alpha, beta",
    ) as captured:
        registry.get("missing")

    assert secret not in str(captured.value)


def test_invalid_reasoning_flag_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL_ALIASES", "alpha")
    _set_model(monkeypatch, "ALPHA", "alpha-key", "alpha-model", reasoning="sometimes")

    with pytest.raises(ConfigurationError, match="ALPHA_REASONING_SPLIT must be true or false"):
        ModelRegistry.from_env(tmp_path / "missing.env")


def _set_model(monkeypatch, prefix: str, api_key: str, model: str, *, reasoning: str) -> None:
    monkeypatch.setenv(f"{prefix}_API_KEY", api_key)
    monkeypatch.setenv(f"{prefix}_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv(f"{prefix}_MODEL", model)
    monkeypatch.setenv(f"{prefix}_REASONING_SPLIT", reasoning)
