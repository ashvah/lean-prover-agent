"""Python interface for Lean Prover Agent."""

from leanproof.model import (
    GenerationResult,
    LLMConfig,
    OpenAICompatibleProofModel,
    ProofModel,
    normalize_proof,
    split_leading_think_block,
)
from leanproof.model_registry import ModelRegistry
from leanproof.verifier import LeanResult, LeanVerifier, VerificationStatus

__all__ = [
    "GenerationResult",
    "LLMConfig",
    "LeanResult",
    "LeanVerifier",
    "ModelRegistry",
    "OpenAICompatibleProofModel",
    "ProofModel",
    "VerificationStatus",
    "normalize_proof",
    "split_leading_think_block",
]
