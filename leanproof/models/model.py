"""Minimal frozen-model abstraction for the one-shot baseline."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI, Timeout

from leanproof.prompts import (
    PromptContractError,
    ReasoningMode,
    build_baseline_prompt,
    parse_baseline_output,
)

DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_READ_TIMEOUT_SECONDS = 300.0
DEFAULT_WRITE_TIMEOUT_SECONDS = 30.0
DEFAULT_POOL_TIMEOUT_SECONDS = 10.0

_FENCED_BLOCK_PATTERN = re.compile(
    r"^```[^\r\n]*\r?\n(?P<proof>.*?)\r?\n```[ \t]*(?:\r?\n|\Z)",
    flags=re.DOTALL | re.MULTILINE,
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
class ErrorDetails:
    """Serializable failure details without provider credentials or request headers."""

    stage: str
    type: str
    cause_type: str | None
    message: str
    status_code: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a compact JSON-compatible error record."""

        record: dict[str, object] = {
            "stage": self.stage,
            "type": self.type,
            "cause_type": self.cause_type,
            "message": self.message,
        }
        if self.status_code is not None:
            record["status_code"] = self.status_code
        return record

    @classmethod
    def from_exception(cls, stage: str, error: Exception) -> ErrorDetails:
        """Describe a project-owned exception without exposing traceback internals."""

        cause = error.__cause__ or error.__context__
        return cls(
            stage=stage,
            type=type(error).__name__,
            cause_type=type(cause).__name__ if cause is not None else None,
            message=str(error),
            status_code=getattr(error, "status_code", None),
        )


class GenerationRequestError(RuntimeError):
    """Project-owned provider request failure with retry-relevant classification."""

    def __init__(
        self,
        details: ErrorDetails,
        *,
        retryable: bool,
        transport: bool,
        transient_api: bool = False,
        elapsed_ms: int | None = None,
    ) -> None:
        super().__init__(details.message)
        self.details = details
        self.retryable = retryable
        self.transport = transport
        self.transient_api = transient_api
        self.elapsed_ms = elapsed_ms


@dataclass(frozen=True)
class RequestFailureClassification:
    """Provider failure category used by the bounded request retry loop."""

    retryable: bool
    transport: bool
    transient_api: bool


def classify_request_failure(error: APIError) -> RequestFailureClassification:
    """Classify OpenAI-compatible failures without inspecting provider model names."""

    if isinstance(error, (APITimeoutError, APIConnectionError)):
        return RequestFailureClassification(
            retryable=True,
            transport=True,
            transient_api=False,
        )
    if isinstance(error, APIStatusError):
        status_code = error.status_code
        retryable = status_code in {408, 409, 429} or 500 <= status_code <= 599
        return RequestFailureClassification(
            retryable=retryable,
            transport=False,
            transient_api=retryable,
        )
    return RequestFailureClassification(
        retryable=False,
        transport=False,
        transient_api=False,
    )


@dataclass(frozen=True)
class LLMConfig:
    """Validated OpenAI-compatible provider configuration."""

    api_key: str
    base_url: str
    model: str
    reasoning_split: bool = False
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    read_timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS
    write_timeout_seconds: float = DEFAULT_WRITE_TIMEOUT_SECONDS
    pool_timeout_seconds: float = DEFAULT_POOL_TIMEOUT_SECONDS
    api_key_env: str | None = None

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
        for field_name in (
            "connect_timeout_seconds",
            "read_timeout_seconds",
            "write_timeout_seconds",
            "pool_timeout_seconds",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ConfigurationError(f"{field_name} must be greater than zero")

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
        if self.api_key_env is not None:
            api_key_env = self.api_key_env.strip()
            if not api_key_env:
                raise ConfigurationError("api_key_env must not be empty when provided")
            object.__setattr__(self, "api_key_env", api_key_env)


@dataclass(frozen=True)
class GenerationResult:
    """One provider generation split into original, reasoning, and final-answer text."""

    raw_output: str
    proof_output: str
    native_reasoning_output: str | None
    latency_ms: int
    plan_output: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ProofModel(Protocol):
    """Interface implemented by one-shot proof generators."""

    @property
    def model_name(self) -> str:
        """Return the configured provider model identifier."""

    def generate_proof(
        self, statement: str, *, reasoning_mode: ReasoningMode = "none"
    ) -> GenerationResult:
        """Generate exactly one complete Lean proof for a theorem statement."""


class OpenAICompatibleProofModel:
    """Generate one proof through an OpenAI-compatible Chat Completions endpoint."""

    def __init__(self, config: LLMConfig, *, client: Any | None = None) -> None:
        self._config = config
        self._client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=0,
            timeout=Timeout(
                connect=config.connect_timeout_seconds,
                read=config.read_timeout_seconds,
                write=config.write_timeout_seconds,
                pool=config.pool_timeout_seconds,
            ),
        )

    @property
    def model_name(self) -> str:
        """Return the configured provider model identifier."""

        return self._config.model

    def generate_proof(
        self, statement: str, *, reasoning_mode: ReasoningMode = "none"
    ) -> GenerationResult:
        """Make one provider request and preserve its textual output unchanged."""

        started = time.perf_counter()
        request: dict[str, object] = {
            "model": self._config.model,
            "messages": [
                {
                    "role": "user",
                    "content": build_baseline_prompt(statement, reasoning_mode),
                }
            ],
        }
        if self._config.reasoning_split:
            request["extra_body"] = {"reasoning_split": True}
        try:
            response = self._client.chat.completions.create(
                **request,
            )
        except APIError as error:
            classification = classify_request_failure(error)
            raise self._request_error(
                error,
                started,
                transport=classification.transport,
                retryable=classification.retryable,
                transient_api=classification.transient_api,
            ) from error
        latency_ms = round((time.perf_counter() - started) * 1000)

        try:
            message = response.choices[0].message
            raw_output = message.content
        except (AttributeError, IndexError) as error:
            raise ProofGenerationError("Provider response did not contain a completion") from error

        if not isinstance(raw_output, str):
            raise ProofGenerationError("Provider completion did not contain text")

        visible_output, tagged_reasoning = split_leading_think_block(raw_output)
        dedicated_reasoning = _dedicated_reasoning_output(message)
        try:
            parsed_output = parse_baseline_output(visible_output, reasoning_mode)
        except PromptContractError as error:
            raise ProofGenerationError(str(error)) from error
        usage = getattr(response, "usage", None)
        return GenerationResult(
            raw_output=raw_output,
            proof_output=parsed_output.proof_output,
            native_reasoning_output=dedicated_reasoning or tagged_reasoning,
            latency_ms=latency_ms,
            plan_output=parsed_output.plan_output,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    def _request_error(
        self,
        error: Exception,
        started: float,
        *,
        transport: bool,
        retryable: bool,
        transient_api: bool = False,
    ) -> GenerationRequestError:
        message = str(error).replace(self._config.api_key, "[REDACTED]")
        cause = error.__cause__ or error.__context__
        details = ErrorDetails(
            stage="generation_request",
            type=type(error).__name__,
            cause_type=type(cause).__name__ if cause is not None else None,
            message=message,
            status_code=getattr(error, "status_code", None),
        )
        return GenerationRequestError(
            details,
            retryable=retryable,
            transport=transport,
            transient_api=transient_api,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )


def normalize_proof(proof_output: str) -> str:
    """Extract one proof-shaped payload using presentation-only cleanup."""

    stripped_output = proof_output.strip()
    fenced_candidates = [
        match.group("proof").strip()
        for match in _FENCED_BLOCK_PATTERN.finditer(stripped_output)
        if _is_proof_shaped(match.group("proof"))
    ]
    if len(fenced_candidates) > 1:
        raise ProofGenerationError("Provider output contains multiple plausible fenced proofs")
    if fenced_candidates:
        return fenced_candidates[0]
    if _is_proof_shaped(stripped_output):
        return stripped_output
    raise ProofGenerationError("Provider output does not contain one proof beginning with `by`")


def split_leading_think_block(raw_output: str) -> tuple[str, str | None]:
    """Conservatively split one complete leading ``<think>`` block from final output."""

    match = _LEADING_THINK_PATTERN.fullmatch(raw_output)
    if match is None:
        return raw_output, None
    return match.group("answer").lstrip(), match.group("reasoning").strip()


def _is_proof_shaped(value: str) -> bool:
    return re.match(r"\Aby(?:\s|\Z)", value.lstrip()) is not None


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
