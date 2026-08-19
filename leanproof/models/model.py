"""Minimal frozen-model abstraction for the one-shot baseline."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

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
_LEADING_THINK_PATTERN = re.compile(
    r"\A[ \t\r\n]*<think>(?P<reasoning>.*?)</think>(?P<answer>.*)\Z",
    flags=re.DOTALL,
)


class ConfigurationError(ValueError):
    """Raised when an LLM configuration is missing or invalid."""


class ProofGenerationError(RuntimeError):
    """Raised when a provider response does not contain a textual proof."""


@dataclass(frozen=True)
class LLMConfig:
    """Validated OpenAI-compatible provider configuration."""

    api_key: str
    base_url: str
    model: str
    reasoning_split: bool = False

    def __post_init__(self) -> None:
        """Validate values regardless of which registry or test created the configuration."""

        api_key = self.api_key.strip()
        base_url = self.base_url.strip()
        model = self.model.strip()
        if not api_key:
            raise ConfigurationError("api_key must not be empty")
        if not base_url:
            raise ConfigurationError("base_url must not be empty")
        if not model:
            raise ConfigurationError("model must not be empty")

        parsed_url = urlparse(base_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ConfigurationError(
                "base_url must be an absolute HTTP(S) URL without a query or fragment"
            )
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", base_url.rstrip("/"))
        object.__setattr__(self, "model", model)


@dataclass(frozen=True)
class GenerationResult:
    """One provider generation split into original, reasoning, and final-answer text."""

    raw_output: str
    proof_output: str
    reasoning_output: str | None
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
        request: dict[str, object] = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": build_baseline_prompt(statement)}],
        }
        if self._config.reasoning_split:
            request["extra_body"] = {"reasoning_split": True}
        response = self._client.chat.completions.create(
            **request,
        )
        latency_ms = round((time.perf_counter() - started) * 1000)

        try:
            message = response.choices[0].message
            raw_output = message.content
        except (AttributeError, IndexError) as error:
            raise ProofGenerationError("Provider response did not contain a completion") from error

        if not isinstance(raw_output, str):
            raise ProofGenerationError("Provider completion did not contain text")

        proof_output, tagged_reasoning = split_leading_think_block(raw_output)
        dedicated_reasoning = _dedicated_reasoning_output(message)
        usage = getattr(response, "usage", None)
        return GenerationResult(
            raw_output=raw_output,
            proof_output=proof_output,
            reasoning_output=dedicated_reasoning or tagged_reasoning,
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


def split_leading_think_block(raw_output: str) -> tuple[str, str | None]:
    """Conservatively split one complete leading ``<think>`` block from final output."""

    match = _LEADING_THINK_PATTERN.fullmatch(raw_output)
    if match is None:
        return raw_output, None
    return match.group("answer").lstrip(), match.group("reasoning").strip()


def _dedicated_reasoning_output(message: Any) -> str | None:
    reasoning_details = getattr(message, "reasoning_details", None)
    if isinstance(reasoning_details, list):
        text_parts: list[str] = []
        for detail in reasoning_details:
            text = detail.get("text") if isinstance(detail, dict) else getattr(detail, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)

    for field_name in ("reasoning_content", "reasoning"):
        value = getattr(message, field_name, None)
        if isinstance(value, str) and value:
            return value
    return None
