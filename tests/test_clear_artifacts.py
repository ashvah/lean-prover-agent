from __future__ import annotations

from pathlib import Path

import pytest

from scripts import clear_artifacts


def test_main_clears_only_expected_artifact_directories_and_recreates_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(clear_artifacts, "ARTIFACT_ROOT", artifact_root)
    outside_file = tmp_path / "data" / "keep.jsonl"
    outside_file.parent.mkdir()
    outside_file.write_text("source data", encoding="utf-8")
    for directory in clear_artifacts._artifact_directories():
        (directory / "nested").mkdir(parents=True)
        (directory / "generated.txt").write_text("generated", encoding="utf-8")
        (directory / "nested" / "generated.txt").write_text("generated", encoding="utf-8")

    exit_code = clear_artifacts.main()

    assert exit_code == 0
    assert outside_file.read_text(encoding="utf-8") == "source data"
    assert all(directory.is_dir() for directory in clear_artifacts._artifact_directories())
    assert all(
        not any(directory.iterdir()) for directory in clear_artifacts._artifact_directories()
    )


def test_clear_directory_rejects_path_outside_fixed_artifact_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(clear_artifacts, "ARTIFACT_ROOT", tmp_path / "artifacts")

    with pytest.raises(ValueError, match="Refusing to clear non-artifact directory"):
        clear_artifacts._clear_directory(tmp_path / "data")
