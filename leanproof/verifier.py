"""Minimal subprocess-based Lean 4 verifier."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LeanResult:
    """Raw result of one Lean verification call."""

    success: bool
    stdout: str
    stderr: str
    elapsed_ms: int


class LeanVerifier:
    """Verify a theorem declaration and proof with ``lake env lean``."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        *,
        timeout_seconds: float = 60.0,
        lake_executable: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        if not (self.project_root / "lakefile.toml").is_file():
            raise ValueError(f"project_root must contain lakefile.toml: {self.project_root}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self.timeout_seconds = timeout_seconds
        self.lake_executable = str(lake_executable or self._find_lake())

    def verify(self, statement: str, proof: str) -> LeanResult:
        """Compile ``statement := proof`` in an isolated temporary Lean file."""

        source = f"import Mathlib\n\n{statement.rstrip()} := {proof.strip()}\n"
        started = time.perf_counter()

        with tempfile.TemporaryDirectory(prefix="leanproof-") as temp_dir:
            source_path = Path(temp_dir) / "Temp.lean"
            source_path.write_text(source, encoding="utf-8")
            command = [self.lake_executable, "env", "lean", str(source_path)]

            try:
                completed = subprocess.run(
                    command,
                    cwd=self.project_root,
                    env=self._subprocess_environment(),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                return LeanResult(
                    success=False,
                    stdout=self._output_text(error.stdout),
                    stderr=(
                        self._output_text(error.stderr) + f"Lean verification timed out after "
                        f"{self.timeout_seconds:g} seconds."
                    ),
                    elapsed_ms=self._elapsed_ms(started),
                )
            except OSError as error:
                return LeanResult(
                    success=False,
                    stdout="",
                    stderr=f"Failed to start Lean verifier: {error}",
                    elapsed_ms=self._elapsed_ms(started),
                )

        return LeanResult(
            success=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_ms=self._elapsed_ms(started),
        )

    @staticmethod
    def _find_lake() -> str:
        lake = shutil.which("lake")
        if lake:
            return lake

        executable_name = "lake.exe" if os.name == "nt" else "lake"
        elan_lake = Path.home() / ".elan" / "bin" / executable_name
        if elan_lake.is_file():
            return str(elan_lake)

        return "lake"

    @staticmethod
    def _subprocess_environment() -> dict[str, str]:
        environment = os.environ.copy()
        elan_home = Path.home() / ".elan"
        if "ELAN_HOME" not in environment and elan_home.is_dir():
            environment["ELAN_HOME"] = str(elan_home)
        return environment

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return round((time.perf_counter() - started) * 1000)

    @staticmethod
    def _output_text(output: str | bytes | None) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output
