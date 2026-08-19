"""Model configuration, provider integration, and proof generation."""

from leanproof.models.model import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_POOL_TIMEOUT_SECONDS,
    DEFAULT_READ_TIMEOUT_SECONDS,
    DEFAULT_WRITE_TIMEOUT_SECONDS,
    ConfigurationError,
    ErrorDetails,
    GenerationRequestError,
    GenerationResult,
    LLMConfig,
    OpenAICompatibleProofModel,
    ProofGenerationError,
    ProofModel,
    RequestFailureClassification,
    classify_request_failure,
    normalize_proof,
    split_leading_think_block,
)
from leanproof.models.registry import ModelRegistry

__all__ = [
    "DEFAULT_CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_POOL_TIMEOUT_SECONDS",
    "DEFAULT_READ_TIMEOUT_SECONDS",
    "DEFAULT_WRITE_TIMEOUT_SECONDS",
    "ConfigurationError",
    "ErrorDetails",
    "GenerationRequestError",
    "GenerationResult",
    "LLMConfig",
    "ModelRegistry",
    "OpenAICompatibleProofModel",
    "ProofGenerationError",
    "ProofModel",
    "RequestFailureClassification",
    "classify_request_failure",
    "normalize_proof",
    "split_leading_think_block",
]
