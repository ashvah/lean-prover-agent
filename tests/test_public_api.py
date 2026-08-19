from __future__ import annotations

import leanproof
from leanproof.lean import LeanResult, LeanVerifier, VerificationStatus
from leanproof.models import (
    GenerationResult,
    LLMConfig,
    ModelRegistry,
    OpenAICompatibleProofModel,
    ProofModel,
)


def test_top_level_public_facade_preserves_existing_symbols() -> None:
    assert leanproof.GenerationResult is GenerationResult
    assert leanproof.LLMConfig is LLMConfig
    assert leanproof.LeanResult is LeanResult
    assert leanproof.LeanVerifier is LeanVerifier
    assert leanproof.ModelRegistry is ModelRegistry
    assert leanproof.OpenAICompatibleProofModel is OpenAICompatibleProofModel
    assert leanproof.ProofModel is ProofModel
    assert leanproof.VerificationStatus is VerificationStatus
