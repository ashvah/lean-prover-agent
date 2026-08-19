"""Clear generated experiment artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_DIRECTORIES = (
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "exports",
    PROJECT_ROOT / "reports",
)


def clear_directory(directory: Path) -> int:
    """Remove all files and subdirectories inside one artifact directory."""

    if not directory.exists():
        return 0

    removed = 0

    for path in directory.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed += 1

    return removed


def main() -> int:
    total_removed = 0

    for directory in ARTIFACT_DIRECTORIES:
        removed = clear_directory(directory)
        total_removed += removed
        print(f"{directory.relative_to(PROJECT_ROOT)}: removed {removed} item(s)")

    print(f"Total removed: {total_removed} item(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
