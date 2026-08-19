"""Clear generated experiment artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
STRATEGIES = ("one_shot", "retry")
ARTIFACT_KINDS = ("results", "reports", "exports")


def _artifact_directories() -> tuple[Path, ...]:
    return tuple(
        ARTIFACT_ROOT / strategy / artifact_kind
        for strategy in STRATEGIES
        for artifact_kind in ARTIFACT_KINDS
    )


def _clear_directory(directory: Path) -> int:
    """Clear one fixed artifact directory after validating its resolved location."""

    artifact_root = ARTIFACT_ROOT.resolve()
    resolved_directory = directory.resolve()
    allowed_directories = {path.resolve() for path in _artifact_directories()}
    if resolved_directory not in allowed_directories or not resolved_directory.is_relative_to(
        artifact_root
    ):
        raise ValueError(f"Refusing to clear non-artifact directory: {directory}")

    directory.mkdir(parents=True, exist_ok=True)
    removed = 0
    for path in directory.iterdir():
        if not path.resolve().is_relative_to(artifact_root):
            raise ValueError(f"Refusing to clear path outside artifact root: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed += 1

    return removed


def main() -> int:
    total_removed = 0

    for directory in _artifact_directories():
        removed = _clear_directory(directory)
        total_removed += removed
        print(f"{directory.relative_to(PROJECT_ROOT)}: removed {removed} item(s)")

    print(f"Total removed: {total_removed} item(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
