"""Minimal frozen-model abstraction for the one-shot baseline."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI

BASELINE_PROMPT_TEMPLATE = """Complete the following Lean 4 theorem.

Return only a complete Lean proof beginning with `by`.

Do not return Markdown.
Do not explain the proof.
Do not repeat the theorem statement.

Theorem:

{statement}"""

_FENCED_PROOF_PATTERN = re.compile(
    r"\A```(?:lean)?[ \t]*\r?\n(?P<proof>.*)\r?\n```[ \t]*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


class ConfigurationError(ValueError):
    """Raised when required global LLM configuration is missing or invalid."""


class ProofGenerationError(RuntimeError):
    """Raised when a provider response does not contain a textual proof."""


@dataclass(frozen=True)
class LLMConfig:
    """Validated OpenAI-compatible provider configuration."""

    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> LLMConfig:
        """Load configuration from environment variables and an optional `.env` file."""

        load_dotenv(dotenv_path=dotenv_path, override=False)
        api_key = _required_environment_value("LLM_API_KEY")
        base_url = _required_environment_value("LLM_BASE_URL")
        model = _required_environment_value("LLM_MODEL")

        parsed_url = urlparse(base_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ConfigurationError(
                "LLM_BASE_URL must be an absolute HTTP(S) URL without a query or fragment"
            )

        return cls(api_key=api_key, base_url=base_url.rstrip("/"), model=model)


@dataclass(frozen=True)
class GenerationResult:
    """One provider generation with raw output, latency, and optional token usage."""

    raw_output: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ProofModel(Protocol):
    """Interface implemented by one-shot proof generators."""

    @property
    def model_name(self) -> str:
        """Return the configured provider model identifier."""

    def generate_proof(self, statement: str) -> GenerationResult:
        """Generate exactly one complete Lean proof for a theorem statement."""


class OpenAICompatibleProofModel:
    """Generate one proof through an OpenAI-compatible Chat Completions endpoint."""

    def __init__(self, config: LLMConfig, *, client: Any | None = None) -> None:
        self._config = config
        self._client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=0,
            timeout=60.0,
        )

    @property
    def model_name(self) -> str:
        """Return the configured provider model identifier."""

        return self._config.model

    def generate_proof(self, statement: str) -> GenerationResult:
        """Make one provider request and preserve its textual output unchanged."""

        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[{"role": "user", "content": build_baseline_prompt(statement)}],
        )
        latency_ms = round((time.perf_counter() - started) * 1000)

        try:
            raw_output = response.choices[0].message.content
        except (AttributeError, IndexError) as error:
            raise ProofGenerationError("Provider response did not contain a completion") from error

        if not isinstance(raw_output, str):
            raise ProofGenerationError("Provider completion did not contain text")

        usage = getattr(response, "usage", None)
        return GenerationResult(
            raw_output=raw_output,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )


def build_baseline_prompt(statement: str) -> str:
    """Insert a theorem statement into the fixed Phase 1 baseline prompt."""

    return BASELINE_PROMPT_TEMPLATE.format(statement=statement)


def normalize_proof(raw_output: str) -> str:
    """Remove surrounding whitespace and one optional outer Lean code fence."""

    stripped_output = raw_output.strip()
    fenced_match = _FENCED_PROOF_PATTERN.fullmatch(stripped_output)
    if fenced_match:
        return fenced_match.group("proof").strip()
    return stripped_output


def _required_environment_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"Missing required environment variable: {name}")
    return value
