"""Proof-generation strategies and their result records."""

from leanproof.strategies.common import (
    DEFAULT_MAX_TRANSPORT_RETRIES,
    DatasetError,
    ProgressCallback,
    ProofVerifier,
    RequestAttempt,
    TaskDifficulty,
    TaskMetadata,
    TheoremTask,
    load_dataset,
)
from leanproof.strategies.one_shot import (
    OneShotResult,
    OneShotSummary,
    default_output_path,
    run_one_shot,
)
from leanproof.strategies.retry import (
    DEFAULT_MAX_ATTEMPTS,
    RetryAttempt,
    RetryResult,
    RetrySummary,
    default_retry_output_path,
    run_retry,
)

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_TRANSPORT_RETRIES",
    "DatasetError",
    "OneShotResult",
    "OneShotSummary",
    "ProgressCallback",
    "ProofVerifier",
    "RequestAttempt",
    "RetryAttempt",
    "RetryResult",
    "RetrySummary",
    "TaskDifficulty",
    "TaskMetadata",
    "TheoremTask",
    "default_output_path",
    "default_retry_output_path",
    "load_dataset",
    "run_one_shot",
    "run_retry",
]
