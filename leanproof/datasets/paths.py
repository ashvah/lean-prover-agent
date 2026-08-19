"""Deterministic path conventions for prepared and sampled theorem datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class DatasetPathError(ValueError):
    """Raised when dataset path intent cannot be converted into a safe path."""


@dataclass(frozen=True)
class DataPaths:
    """Repository-resolved dataset roots with domain-specific path derivation."""

    raw_data: Path
    processed_data: Path
    splits: Path

    @classmethod
    def from_configured_roots(
        cls,
        project_root: str | Path,
        *,
        raw_data: str | Path,
        processed_data: str | Path,
        splits: str | Path,
    ) -> DataPaths:
        """Resolve repository-relative roots while preserving explicit absolute roots."""

        root = Path(project_root)
        return cls(
            raw_data=_resolve_root(root, raw_data),
            processed_data=_resolve_root(root, processed_data),
            splits=_resolve_root(root, splits),
        )

    def raw_dataset_path(self, source: str, source_file: str) -> Path:
        """Derive one manually acquired raw source path."""

        return self.raw_data / _safe_component(source, "source") / _safe_filename(source_file)

    def processed_dataset_path(self, source: str, source_file: str) -> Path:
        """Derive canonical JSONL from a raw source filename stem."""

        stem = Path(_safe_filename(source_file)).stem
        return self.processed_data / _safe_component(source, "source") / f"{stem}.jsonl"

    def dataset_manifest_path(self, source: str, source_file: str) -> Path:
        """Derive a preparation manifest beside its canonical JSONL."""

        stem = Path(_safe_filename(source_file)).stem
        return self.processed_data / _safe_component(source, "source") / f"{stem}.manifest.json"

    def processed_input_path(self, source: str, source_file: str) -> Path:
        """Resolve one explicitly named canonical file below a source directory."""

        return self.processed_data / _safe_component(source, "source") / _safe_filename(source_file)

    def development_split_path(
        self,
        *,
        source_file: str,
        split: str,
        bucket: str,
        size: int,
        seed: int,
    ) -> Path:
        """Derive a deterministic human-readable sample output path."""

        if bucket not in {"easy", "medium", "hard", "all"}:
            raise DatasetPathError("bucket must be one of: easy, medium, hard, all")
        if size <= 0:
            raise DatasetPathError("size must be greater than zero")
        stem = Path(_safe_filename(source_file)).stem
        split_name = _safe_component(split, "split")
        return self.splits / split_name / f"{stem}_{bucket}_{size}_seed{seed}.jsonl"


def _resolve_root(project_root: Path, configured_root: str | Path) -> Path:
    path = Path(configured_root)
    return path if path.is_absolute() else project_root / path


def _safe_component(value: str, label: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) is None:
        raise DatasetPathError(f"{label} must be a single safe path component")
    return value


def _safe_filename(value: str) -> str:
    path = Path(value)
    if not value or path.name != value or value in {".", ".."}:
        raise DatasetPathError("source_file must be a filename without directories")
    return value
