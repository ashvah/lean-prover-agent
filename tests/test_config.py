from __future__ import annotations

import json
from pathlib import Path

import pytest

from leanproof.config import (
    DEFAULT_GENERATION_TIMEOUT_SECONDS,
    DEFAULT_VERIFICATION_TIMEOUT_SECONDS,
    build_model_registry,
    load_config,
    resolve_experiment_config,
)
from leanproof.datasets import DataPaths
from leanproof.models import ConfigurationError, OpenAICompatibleProofModel
from scripts.inspect_dataset import build_argument_parser as build_inspect_parser
from scripts.prepare_dataset import build_argument_parser as build_prepare_parser
from scripts.run_one_shot import build_argument_parser as build_one_shot_parser
from scripts.run_retry import build_argument_parser as build_retry_parser
from scripts.sample_dataset import build_argument_parser as build_sample_parser

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_toml_loads_models_and_separate_timeouts() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.toml")

    assert config.paths.raw_data == "data/raw"
    assert config.paths.processed_data == "data/processed"
    assert config.paths.splits == "data/splits"
    assert config.paths.artifacts == "artifacts"
    assert tuple(config.models) == ("deepseek_c", "deepseek_r", "minimax", "qwen")
    assert config.models["minimax"].generation_timeout_seconds == 300.0
    assert config.verification_timeout_seconds == 120.0
    assert (config.prepare_dataset.source, config.prepare_dataset.source_file) == (
        "lean_workbook",
        "0000.parquet",
    )
    assert (
        config.sample_dataset.bucket,
        config.sample_dataset.size,
        config.sample_dataset.seed,
    ) == (
        "medium",
        50,
        42,
    )
    assert config.inspect_dataset.source_file == "0000.jsonl"
    assert config.run_one_shot.dataset == "data/smoke.jsonl"
    assert config.run_one_shot.model == "minimax"
    assert config.run_retry.dataset.endswith("0000_medium_50_seed42.jsonl")
    assert config.run_retry.model == "minimax"
    assert config.run_retry.max_attempts == 4
    assert config.run_one_shot.limit is None
    assert config.run_retry.limit is None
    assert config.run_one_shot.max_transport_retries == 2
    assert config.run_retry.max_transport_retries == 2


@pytest.mark.parametrize("invalid_limit", [0, -1])
def test_non_positive_limit_is_invalid_in_toml_instead_of_meaning_unlimited(
    tmp_path: Path, invalid_limit: int
) -> None:
    path = write_config(tmp_path / "invalid-limit.toml")
    contents = path.read_text(encoding="utf-8").replace(
        'model = "alpha"\nmax_transport_retries = 2',
        f'model = "alpha"\nlimit = {invalid_limit}\nmax_transport_retries = 2',
        1,
    )
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="limit must be a positive integer"):
        load_config(path)


def test_positive_toml_limit_is_preserved(tmp_path: Path) -> None:
    path = write_config(tmp_path / "positive-limit.toml")
    contents = path.read_text(encoding="utf-8").replace(
        'model = "alpha"\nmax_transport_retries = 2',
        'model = "alpha"\nlimit = 5\nmax_transport_retries = 2',
        1,
    )
    path.write_text(contents, encoding="utf-8")

    assert load_config(path).run_one_shot.limit == 5


def test_experiment_cli_limit_is_positive_and_transport_retries_are_non_negative() -> None:
    one_shot = build_one_shot_parser().parse_args(["--max-transport-retries", "0"])
    retry = build_retry_parser().parse_args(["--max-transport-retries", "3"])

    assert one_shot.max_transport_retries == 0
    assert retry.max_transport_retries == 3
    with pytest.raises(SystemExit):
        build_one_shot_parser().parse_args(["--limit", "0"])
    with pytest.raises(SystemExit):
        build_retry_parser().parse_args(["--max-transport-retries", "-1"])


def test_malformed_toml_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[model\ndefault = 'broken'", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Could not load TOML"):
        load_config(path)


def test_cli_overrides_toml_and_toml_overrides_built_in_fallback(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "runtime.toml",
        generation_timeout=450,
        verification_timeout=150,
        retry_dataset="data/from_retry_toml.jsonl",
        attempts=3,
    )
    config = load_config(path)
    from_toml = resolve_experiment_config(config, workflow="run_retry")
    from_cli = resolve_experiment_config(
        config,
        workflow="run_retry",
        dataset_path="data/from_cli.jsonl",
        model_alias="alpha",
        retry_max_attempts=2,
        generation_timeout_seconds=600,
        verification_timeout_seconds=90,
        verbose=True,
    )

    assert from_toml.dataset_path == "data/from_retry_toml.jsonl"
    assert from_toml.retry_max_attempts == 3
    assert from_toml.generation_timeout_seconds == 450.0
    assert from_toml.verification_timeout_seconds == 150.0
    assert from_cli.dataset_path == "data/from_cli.jsonl"
    assert from_cli.retry_max_attempts == 2
    assert from_cli.generation_timeout_seconds == 600.0
    assert from_cli.verification_timeout_seconds == 90.0
    assert from_cli.verbose is True


def test_built_in_timeout_fallbacks_are_independent(tmp_path: Path) -> None:
    path = write_config(tmp_path / "fallback.toml", include_timeouts=False)
    config = load_config(path)

    assert config.models["alpha"].generation_timeout_seconds == (DEFAULT_GENERATION_TIMEOUT_SECONDS)
    assert config.verification_timeout_seconds == DEFAULT_VERIFICATION_TIMEOUT_SECONDS
    generation_override = resolve_experiment_config(
        config, workflow="run_one_shot", generation_timeout_seconds=700
    )
    verification_override = resolve_experiment_config(
        config, workflow="run_one_shot", verification_timeout_seconds=30
    )
    assert generation_override.generation_timeout_seconds == 700.0
    assert generation_override.verification_timeout_seconds == 120.0
    assert verification_override.generation_timeout_seconds == 300.0
    assert verification_override.verification_timeout_seconds == 30.0


def test_workflows_keep_distinct_dataset_model_and_cli_defaults(tmp_path: Path) -> None:
    config = load_config(
        write_config(
            tmp_path / "workflows.toml",
            one_shot_dataset="data/one.jsonl",
            retry_dataset="data/retry.jsonl",
            one_shot_model="alpha",
            retry_model="beta",
            include_second_model=True,
        )
    )

    one_shot = resolve_experiment_config(config, workflow="run_one_shot")
    retry = resolve_experiment_config(config, workflow="run_retry")
    retry_override = resolve_experiment_config(
        config,
        workflow="run_retry",
        dataset_path="data/override.jsonl",
        model_alias="alpha",
        retry_max_attempts=2,
        verbose=True,
    )

    assert (one_shot.dataset_path, one_shot.model_alias) == ("data/one.jsonl", "alpha")
    assert (retry.dataset_path, retry.model_alias) == ("data/retry.jsonl", "beta")
    assert retry.retry_max_attempts == 4
    assert retry_override.dataset_path == "data/override.jsonl"
    assert retry_override.model_alias == "alpha"
    assert retry_override.retry_max_attempts == 2
    assert retry_override.verbose is True


def test_dataset_paths_are_derived_from_shared_roots() -> None:
    paths = DataPaths.from_configured_roots(
        Path("repo"),
        raw_data="data/raw",
        processed_data="data/processed",
        splits="data/splits",
    )

    assert paths.raw_dataset_path("lean_workbook", "lean-workbook.parquet") == Path(
        "repo/data/raw/lean_workbook/lean-workbook.parquet"
    )
    assert paths.processed_dataset_path("lean_workbook", "lean-workbook.parquet") == Path(
        "repo/data/processed/lean_workbook/lean-workbook.jsonl"
    )
    assert paths.dataset_manifest_path("lean_workbook", "lean-workbook.parquet") == Path(
        "repo/data/processed/lean_workbook/lean-workbook.manifest.json"
    )
    assert paths.processed_input_path("lean_workbook", "lean-workbook.jsonl") == Path(
        "repo/data/processed/lean_workbook/lean-workbook.jsonl"
    )
    assert paths.development_split_path(
        source_file="lean-workbook.jsonl",
        split="development",
        bucket="medium",
        size=50,
        seed=42,
    ) == Path("repo/data/splits/development/lean-workbook_medium_50_seed42.jsonl")


def test_dataset_workflow_cli_values_are_optional_toml_overrides() -> None:
    prepare = build_prepare_parser().parse_args([])
    sample = build_sample_parser().parse_args([])
    inspect = build_inspect_parser().parse_args([])

    assert (prepare.source, prepare.source_file, prepare.input, prepare.output) == (
        None,
        None,
        None,
        None,
    )
    assert (sample.source, sample.source_file, sample.bucket, sample.size, sample.seed) == (
        None,
        None,
        None,
        None,
        None,
    )
    assert (inspect.source, inspect.source_file, inspect.input) == (None, None, None)


def test_registry_resolves_named_secret_and_model_timeout_without_serializing_key(
    tmp_path: Path,
) -> None:
    secret = "never-serialize-this-key"
    config = load_config(write_config(tmp_path / "runtime.toml"))
    registry = build_model_registry(
        config,
        required_alias="alpha",
        generation_timeout_seconds=525,
        environ={"ALPHA_API_KEY": secret},
    )
    model_config = registry.get("alpha")

    assert registry.names() == ("alpha",)
    assert model_config.api_key == secret
    assert model_config.api_key_env == "ALPHA_API_KEY"
    assert model_config.generation_timeout_seconds == 525.0
    assert secret not in json.dumps(config, default=lambda value: value.__dict__)


def test_missing_required_api_key_names_variable_without_exposing_secrets(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path / "runtime.toml"))

    with pytest.raises(ConfigurationError, match="ALPHA_API_KEY") as captured:
        build_model_registry(config, required_alias="alpha", environ={})

    assert str(captured.value) == "Model 'alpha' requires environment variable ALPHA_API_KEY"
    assert "https://" not in str(captured.value)


def test_model_client_receives_resolved_generation_timeout(tmp_path: Path, monkeypatch) -> None:
    config = load_config(write_config(tmp_path / "runtime.toml", generation_timeout=480))
    registry = build_model_registry(
        config,
        required_alias="alpha",
        environ={"ALPHA_API_KEY": "test-key"},
    )
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("leanproof.models.model.OpenAI", FakeOpenAI)
    OpenAICompatibleProofModel(registry.get("alpha"))

    assert captured["timeout"] == 480.0
    assert captured["max_retries"] == 0


def write_config(
    path: Path,
    *,
    generation_timeout: int = 300,
    verification_timeout: int = 120,
    one_shot_dataset: str = "data/from_one_shot_toml.jsonl",
    retry_dataset: str = "data/from_retry_toml.jsonl",
    one_shot_model: str = "alpha",
    retry_model: str = "alpha",
    attempts: int = 4,
    include_timeouts: bool = True,
    include_second_model: bool = False,
) -> Path:
    model_timeout = (
        f"generation_timeout_seconds = {generation_timeout}\n" if include_timeouts else ""
    )
    lean_timeout = (
        f"verification_timeout_seconds = {verification_timeout}\n" if include_timeouts else ""
    )
    second_model = ""
    if include_second_model:
        second_model = """
[models.beta]
base_url = "https://api.example.com/v1"
model = "provider-beta"
api_key_env = "BETA_API_KEY"
reasoning_split = true
"""
    path.write_text(
        f'''[paths]
raw_data = "data/raw"
processed_data = "data/processed"
splits = "data/splits"
artifacts = "artifacts"

[models.alpha]
base_url = "https://api.example.com/v1"
model = "provider-alpha"
api_key_env = "ALPHA_API_KEY"
reasoning_split = false
{model_timeout}
{second_model}
[lean]
{lean_timeout}
[prepare_dataset]
source = "lean_workbook"
source_file = "raw.parquet"

[sample_dataset]
source = "lean_workbook"
source_file = "raw.jsonl"
split = "development"
bucket = "medium"
size = 50
seed = 42

[inspect_dataset]
source = "lean_workbook"
source_file = "raw.jsonl"

[run_one_shot]
dataset = "{one_shot_dataset}"
model = "{one_shot_model}"
max_transport_retries = 2
verbose = false

[run_retry]
dataset = "{retry_dataset}"
model = "{retry_model}"
max_attempts = {attempts}
max_transport_retries = 2
verbose = false
''',
        encoding="utf-8",
    )
    return path
