"""Typed runtime configuration loaded from repository TOML and environment secrets."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from leanproof.models import ConfigurationError, LLMConfig, ModelRegistry

DEFAULT_CONFIG_PATH = Path("configs/default.toml")
DEFAULT_RAW_DATA_ROOT = "data/raw"
DEFAULT_PROCESSED_DATA_ROOT = "data/processed"
DEFAULT_SPLITS_ROOT = "data/splits"
DEFAULT_ARTIFACT_ROOT = "artifacts"
DEFAULT_ONE_SHOT_DATASET_PATH = "data/smoke.jsonl"
DEFAULT_RETRY_DATASET_PATH = "data/smoke.jsonl"
DEFAULT_GENERATION_TIMEOUT_SECONDS = 300.0
DEFAULT_VERIFICATION_TIMEOUT_SECONDS = 120.0
DEFAULT_RETRY_MAX_ATTEMPTS = 4

ExperimentWorkflow = Literal["run_one_shot", "run_retry"]


@dataclass(frozen=True)
class ModelDefinition:
    """One non-secret model definition whose API key is named indirectly."""

    base_url: str
    model: str
    api_key_env: str
    reasoning_split: bool = False
    generation_timeout_seconds: float = DEFAULT_GENERATION_TIMEOUT_SECONDS


@dataclass(frozen=True)
class PathConfig:
    """Shared repository-relative roots used to derive runtime paths."""

    raw_data: str
    processed_data: str
    splits: str
    artifacts: str


@dataclass(frozen=True)
class PrepareDatasetConfig:
    """Defaults owned by the dataset-preparation entrypoint."""

    source: str
    source_file: str
    limit: int | None


@dataclass(frozen=True)
class SampleDatasetConfig:
    """Defaults owned by the deterministic dataset-sampling entrypoint."""

    source: str
    source_file: str
    split: str
    bucket: str
    size: int
    seed: int


@dataclass(frozen=True)
class InspectDatasetConfig:
    """Defaults owned by the processed-dataset inspection entrypoint."""

    source: str
    source_file: str


@dataclass(frozen=True)
class OneShotWorkflowConfig:
    """Defaults owned by the one-shot experiment entrypoint."""

    dataset: str
    model: str | None
    limit: int | None
    verbose: bool


@dataclass(frozen=True)
class RetryWorkflowConfig:
    """Defaults owned by the independent-retry experiment entrypoint."""

    dataset: str
    model: str | None
    max_attempts: int
    limit: int | None
    verbose: bool


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated shared infrastructure and user-facing workflow defaults."""

    paths: PathConfig
    models: Mapping[str, ModelDefinition]
    verification_timeout_seconds: float
    prepare_dataset: PrepareDatasetConfig
    sample_dataset: SampleDatasetConfig
    inspect_dataset: InspectDatasetConfig
    run_one_shot: OneShotWorkflowConfig
    run_retry: RetryWorkflowConfig


@dataclass(frozen=True)
class ResolvedExperimentConfig:
    """Actual runtime values after one workflow's CLI overrides are applied."""

    dataset_path: str
    model_alias: str
    limit: int | None
    verbose: bool
    retry_max_attempts: int
    generation_timeout_seconds: float
    verification_timeout_seconds: float
    artifact_root: str


def load_config(config_path: str | Path) -> RuntimeConfig:
    """Load and validate one TOML runtime configuration with built-in fallbacks."""

    path = Path(config_path)
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    try:
        with path.open("rb") as source:
            raw_config = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"Could not load TOML configuration {path}: {error}") from error
    if not isinstance(raw_config, dict):
        raise ConfigurationError("TOML configuration root must be a table")

    paths = _table(raw_config, "paths")
    lean = _table(raw_config, "lean")
    models_table = _table(raw_config, "models")
    prepare = _table(raw_config, "prepare_dataset")
    sample = _table(raw_config, "sample_dataset")
    inspect = _table(raw_config, "inspect_dataset")
    one_shot = _table(raw_config, "run_one_shot")
    retry = _table(raw_config, "run_retry")

    models = {
        alias: _model_definition(alias, value) for alias, value in sorted(models_table.items())
    }
    one_shot_model = _optional_string(one_shot, "model", None)
    retry_model = _optional_string(retry, "model", None)
    _validate_workflow_model(one_shot_model, models, "run_one_shot")
    _validate_workflow_model(retry_model, models, "run_retry")
    return RuntimeConfig(
        paths=PathConfig(
            raw_data=_string(paths, "raw_data", DEFAULT_RAW_DATA_ROOT),
            processed_data=_string(paths, "processed_data", DEFAULT_PROCESSED_DATA_ROOT),
            splits=_string(paths, "splits", DEFAULT_SPLITS_ROOT),
            artifacts=_string(paths, "artifacts", DEFAULT_ARTIFACT_ROOT),
        ),
        models=models,
        verification_timeout_seconds=_positive_number(
            lean,
            "verification_timeout_seconds",
            DEFAULT_VERIFICATION_TIMEOUT_SECONDS,
        ),
        prepare_dataset=PrepareDatasetConfig(
            source=_string(prepare, "source", "lean_workbook"),
            source_file=_string(prepare, "source_file", "0000.parquet"),
            limit=_optional_limit(prepare, "limit"),
        ),
        sample_dataset=SampleDatasetConfig(
            source=_string(sample, "source", "lean_workbook"),
            source_file=_string(sample, "source_file", "0000.jsonl"),
            split=_string(sample, "split", "development"),
            bucket=_bucket(sample, "bucket", "medium"),
            size=_positive_integer(sample, "size", 50),
            seed=_integer(sample, "seed", 42),
        ),
        inspect_dataset=InspectDatasetConfig(
            source=_string(inspect, "source", "lean_workbook"),
            source_file=_string(inspect, "source_file", "0000.jsonl"),
        ),
        run_one_shot=OneShotWorkflowConfig(
            dataset=_string(one_shot, "dataset", DEFAULT_ONE_SHOT_DATASET_PATH),
            model=one_shot_model,
            limit=_optional_limit(one_shot, "limit"),
            verbose=_boolean(one_shot, "verbose", False),
        ),
        run_retry=RetryWorkflowConfig(
            dataset=_string(retry, "dataset", DEFAULT_RETRY_DATASET_PATH),
            model=retry_model,
            max_attempts=_positive_integer(
                retry,
                "max_attempts",
                DEFAULT_RETRY_MAX_ATTEMPTS,
            ),
            limit=_optional_limit(retry, "limit"),
            verbose=_boolean(retry, "verbose", False),
        ),
    )


def build_model_registry(
    config: RuntimeConfig,
    *,
    dotenv_path: str | Path | None = None,
    required_alias: str | None = None,
    generation_timeout_seconds: float | None = None,
    environ: Mapping[str, str] | None = None,
) -> ModelRegistry:
    """Resolve named API-key environment variables into selectable model configs."""

    load_dotenv(dotenv_path=dotenv_path, override=False)
    environment = os.environ if environ is None else environ
    configurations: dict[str, LLMConfig] = {}
    for alias, definition in config.models.items():
        api_key = environment.get(definition.api_key_env, "").strip()
        if not api_key:
            if alias == required_alias:
                raise ConfigurationError(
                    f"Model '{alias}' requires environment variable {definition.api_key_env}"
                )
            continue
        model_config = LLMConfig(
            api_key=api_key,
            base_url=definition.base_url,
            model=definition.model,
            reasoning_split=definition.reasoning_split,
            generation_timeout_seconds=definition.generation_timeout_seconds,
            api_key_env=definition.api_key_env,
        )
        if alias == required_alias and generation_timeout_seconds is not None:
            model_config = replace(
                model_config,
                generation_timeout_seconds=_validate_positive_override(
                    generation_timeout_seconds,
                    "generation_timeout_seconds",
                ),
            )
        configurations[alias] = model_config
    return ModelRegistry(configurations)


def resolve_experiment_config(
    config: RuntimeConfig,
    *,
    workflow: ExperimentWorkflow,
    dataset_path: str | None = None,
    model_alias: str | None = None,
    limit: int | None = None,
    verbose: bool | None = None,
    retry_max_attempts: int | None = None,
    generation_timeout_seconds: float | None = None,
    verification_timeout_seconds: float | None = None,
) -> ResolvedExperimentConfig:
    """Apply CLI > workflow TOML > built-in fallback for one experiment entrypoint."""

    workflow_config = config.run_one_shot if workflow == "run_one_shot" else config.run_retry
    selected_model = model_alias or workflow_config.model
    if selected_model is None:
        raise ConfigurationError(f"No model selected; use --model or configure {workflow}.model")
    if selected_model not in config.models:
        available = ", ".join(sorted(config.models)) or "none"
        raise ConfigurationError(f"Unknown model '{selected_model}'. Available models: {available}")
    definition = config.models[selected_model]
    selected_limit = workflow_config.limit if limit is None else limit
    if selected_limit is not None and selected_limit <= 0:
        raise ConfigurationError("limit must be greater than zero")
    configured_attempts = (
        config.run_retry.max_attempts if workflow == "run_retry" else DEFAULT_RETRY_MAX_ATTEMPTS
    )
    selected_attempts = configured_attempts if retry_max_attempts is None else retry_max_attempts
    if selected_attempts <= 0:
        raise ConfigurationError("retry_max_attempts must be greater than zero")
    return ResolvedExperimentConfig(
        dataset_path=dataset_path or workflow_config.dataset,
        model_alias=selected_model,
        limit=selected_limit,
        verbose=workflow_config.verbose if verbose is None else verbose,
        retry_max_attempts=selected_attempts,
        generation_timeout_seconds=_validate_positive_override(
            generation_timeout_seconds
            if generation_timeout_seconds is not None
            else definition.generation_timeout_seconds,
            "generation_timeout_seconds",
        ),
        verification_timeout_seconds=_validate_positive_override(
            verification_timeout_seconds
            if verification_timeout_seconds is not None
            else config.verification_timeout_seconds,
            "verification_timeout_seconds",
        ),
        artifact_root=config.paths.artifacts,
    )


def _validate_workflow_model(
    model_alias: str | None,
    models: Mapping[str, ModelDefinition],
    workflow: str,
) -> None:
    if model_alias is not None and model_alias not in models:
        raise ConfigurationError(
            f"Model '{model_alias}' configured by {workflow}.model is not defined in models"
        )


def _model_definition(alias: str, raw_value: object) -> ModelDefinition:
    if not isinstance(raw_value, dict):
        raise ConfigurationError(f"models.{alias} must be a table")
    return ModelDefinition(
        base_url=_string(raw_value, "base_url"),
        model=_string(raw_value, "model"),
        api_key_env=_string(raw_value, "api_key_env"),
        reasoning_split=_boolean(raw_value, "reasoning_split", False),
        generation_timeout_seconds=_positive_number(
            raw_value,
            "generation_timeout_seconds",
            DEFAULT_GENERATION_TIMEOUT_SECONDS,
        ),
    )


def _table(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a TOML table")
    return value


def _string(table: Mapping[str, object], name: str, fallback: str | None = None) -> str:
    value = table.get(name, fallback)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_string(table: Mapping[str, object], name: str, fallback: str | None) -> str | None:
    value = table.get(name, fallback)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string when provided")
    return value.strip()


def _boolean(table: Mapping[str, object], name: str, fallback: bool) -> bool:
    value = table.get(name, fallback)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be true or false")
    return value


def _positive_number(table: Mapping[str, object], name: str, fallback: float) -> float:
    value = table.get(name, fallback)
    return _validate_positive_override(value, name)


def _validate_positive_override(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return float(value)


def _positive_integer(table: Mapping[str, object], name: str, fallback: int) -> int:
    value = table.get(name, fallback)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _integer(table: Mapping[str, object], name: str, fallback: int) -> int:
    value = table.get(name, fallback)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{name} must be an integer")
    return value


def _optional_limit(table: Mapping[str, object], name: str) -> int | None:
    value = table.get(name, 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer")
    return value or None


def _bucket(table: Mapping[str, object], name: str, fallback: str) -> str:
    value = _string(table, name, fallback)
    if value not in {"easy", "medium", "hard", "all"}:
        raise ConfigurationError(f"{name} must be one of: easy, medium, hard, all")
    return value
