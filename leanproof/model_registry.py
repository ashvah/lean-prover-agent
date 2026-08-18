"""Load multiple named model configurations and select one explicitly."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv

from leanproof.model import ConfigurationError, LLMConfig

MODEL_ALIASES_VARIABLE = "LLM_MODEL_ALIASES"
_ALIAS_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_REQUIRED_SUFFIXES = ("API_KEY", "BASE_URL", "MODEL")


class ModelRegistry:
    """Store validated model configurations under stable experiment-facing aliases."""

    def __init__(self, configurations: Mapping[str, LLMConfig]) -> None:
        registered: dict[str, LLMConfig] = {}
        for alias, config in configurations.items():
            _validate_alias(alias)
            if alias in registered:
                raise ConfigurationError(f"Duplicate model alias: {alias}")
            registered[alias] = config
        self._configurations = registered

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> ModelRegistry:
        """Load complete prefixed configurations from environment and an optional `.env`."""

        load_dotenv(dotenv_path=dotenv_path, override=False)
        aliases = _configured_aliases(os.getenv(MODEL_ALIASES_VARIABLE, ""))
        configurations: dict[str, LLMConfig] = {}
        for alias in aliases:
            prefix = alias.upper()
            values = {
                suffix: os.getenv(f"{prefix}_{suffix}", "").strip() for suffix in _REQUIRED_SUFFIXES
            }
            if not any(values.values()):
                continue
            missing = [f"{prefix}_{suffix}" for suffix, value in values.items() if not value]
            if missing:
                raise ConfigurationError(
                    f"Incomplete model configuration for alias '{alias}'; missing: "
                    + ", ".join(missing)
                )
            reasoning_variable = f"{prefix}_REASONING_SPLIT"
            configurations[alias] = LLMConfig(
                api_key=values["API_KEY"],
                base_url=values["BASE_URL"],
                model=values["MODEL"],
                reasoning_split=_parse_boolean(
                    os.getenv(reasoning_variable, ""), reasoning_variable
                ),
            )
        return cls(configurations)

    def names(self) -> tuple[str, ...]:
        """Return registered aliases in deterministic order."""

        return tuple(sorted(self._configurations))

    def get(self, alias: str) -> LLMConfig:
        """Return one selected configuration or fail without exposing its credentials."""

        try:
            return self._configurations[alias]
        except KeyError as error:
            available = ", ".join(self.names()) or "none"
            raise ConfigurationError(
                f"Unknown model '{alias}'. Available models: {available}"
            ) from error


def _configured_aliases(raw_aliases: str) -> tuple[str, ...]:
    aliases = tuple(part.strip() for part in raw_aliases.split(",") if part.strip())
    seen: set[str] = set()
    for alias in aliases:
        _validate_alias(alias)
        if alias in seen:
            raise ConfigurationError(f"Duplicate model alias in {MODEL_ALIASES_VARIABLE}: {alias}")
        seen.add(alias)
    return aliases


def _validate_alias(alias: str) -> None:
    if _ALIAS_PATTERN.fullmatch(alias) is None:
        raise ConfigurationError(
            f"Invalid model alias '{alias}'; use lowercase letters, digits, and underscores"
        )


def _parse_boolean(raw_value: str, variable_name: str) -> bool:
    value = raw_value.strip().lower()
    if not value:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{variable_name} must be true or false")
