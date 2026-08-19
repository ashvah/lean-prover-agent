"""Local Parquet adapter for the Lean-Workbook dataset."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pyarrow import parquet

from leanproof.datasets.schema import CanonicalTheorem

SOURCE_NAME = "lean_workbook"
_STATEMENT_COLUMNS = ("formal_statement", "lean_statement")
_SOURCE_ID_COLUMNS = ("id", "source_id", "problem_id")
_INFORMAL_COLUMNS = ("problem", "informal_statement", "natural_language_statement")
_ANSWER_COLUMNS = ("answer",)
_PROOF_COLUMNS = ("formal_proof", "proof", "tactic")
_TERMINAL_SORRY_PLACEHOLDER = re.compile(r"\s*:=\s*by\s+sorry\s*\Z")


class LeanWorkbookSchemaError(ValueError):
    """Raised when a Parquet file lacks the required Lean-Workbook schema."""


class RowMappingError(ValueError):
    """Raised for one recoverable malformed Lean-Workbook row."""


@dataclass(frozen=True)
class RawRow:
    index: int
    values: dict[str, object]


class LeanWorkbookAdapter:
    """Read deterministic source-order rows and map them to canonical theorems."""

    source = SOURCE_NAME

    def __init__(self, input_path: str | Path) -> None:
        self.input_path = Path(input_path)
        if not self.input_path.is_file():
            raise LeanWorkbookSchemaError(f"Lean-Workbook input does not exist: {self.input_path}")
        try:
            self._parquet_file = parquet.ParquetFile(self.input_path)
        except (OSError, ValueError) as error:
            raise LeanWorkbookSchemaError(
                f"Could not read Lean-Workbook Parquet file: {self.input_path}"
            ) from error
        columns = tuple(self._parquet_file.schema_arrow.names)
        self._statement_column = _first_available(columns, _STATEMENT_COLUMNS)
        if self._statement_column is None:
            expected = ", ".join(_STATEMENT_COLUMNS)
            available = ", ".join(columns) or "none"
            raise LeanWorkbookSchemaError(
                f"Lean-Workbook schema requires one formal statement column ({expected}); "
                f"available columns: {available}"
            )
        self._source_id_column = _first_available(columns, _SOURCE_ID_COLUMNS)
        self._informal_column = _first_available(columns, _INFORMAL_COLUMNS)
        self._answer_column = _first_available(columns, _ANSWER_COLUMNS)
        self._proof_column = _first_available(columns, _PROOF_COLUMNS)
        self._mapped_columns = {
            column
            for column in (
                self._statement_column,
                self._source_id_column,
                self._informal_column,
                self._answer_column,
                self._proof_column,
            )
            if column is not None
        }

    def iter_rows(self, *, limit: int | None = None) -> Iterator[RawRow]:
        """Yield local Parquet rows in stable file order without loading the whole table."""

        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero")
        row_index = 0
        for batch in self._parquet_file.iter_batches(batch_size=1024):
            for values in batch.to_pylist():
                if limit is not None and row_index >= limit:
                    return
                yield RawRow(index=row_index, values=values)
                row_index += 1

    def map_row(self, raw_row: RawRow) -> CanonicalTheorem:
        """Map one structured source row without rewriting its formal statement."""

        values = raw_row.values
        source_statement = values.get(self._statement_column)
        if not isinstance(source_statement, str):
            raise RowMappingError("formal_statement_not_string")
        if not source_statement.strip():
            raise RowMappingError("formal_statement_blank")
        statement = _remove_terminal_sorry_placeholder(source_statement)

        informal_statement = _optional_text(values, self._informal_column)
        answer = _optional_text(values, self._answer_column, stringify_scalars=True)
        reference_proof = _optional_text(values, self._proof_column)
        source_id_value = values.get(self._source_id_column) if self._source_id_column else None
        if source_id_value is None or not str(source_id_value).strip():
            source_id = _derived_source_id(statement, informal_statement, answer)
        else:
            source_id = str(source_id_value).strip()
        canonical_id = f"lean_workbook_{hashlib.sha256(source_id.encode('utf-8')).hexdigest()[:24]}"
        metadata = {
            key: safe_value
            for key, value in values.items()
            if key not in self._mapped_columns and (safe_value := _json_safe(value)) is not None
        }
        metadata["raw_row_index"] = raw_row.index
        if statement != source_statement:
            metadata["source_formal_statement"] = source_statement
        return CanonicalTheorem(
            id=canonical_id,
            source=SOURCE_NAME,
            source_id=source_id,
            statement=statement,
            informal_statement=informal_statement,
            answer=answer,
            reference_proof=reference_proof,
            metadata=metadata,
        )


def _first_available(columns: tuple[str, ...], candidates: tuple[str, ...]) -> str | None:
    return next((candidate for candidate in candidates if candidate in columns), None)


def _optional_text(
    values: dict[str, object], column: str | None, *, stringify_scalars: bool = False
) -> str | None:
    if column is None:
        return None
    value = values.get(column)
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if stringify_scalars and isinstance(value, (int, float, bool)):
        return str(value)
    raise RowMappingError(f"{column}_not_string")


def _derived_source_id(statement: str, informal_statement: str | None, answer: str | None) -> str:
    identity = json.dumps(
        [statement, informal_statement, answer], ensure_ascii=False, separators=(",", ":")
    )
    return f"sha256:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def _remove_terminal_sorry_placeholder(source_statement: str) -> str:
    """Remove only Lean-Workbook's known terminal ``:= by sorry`` placeholder."""

    match = _TERMINAL_SORRY_PLACEHOLDER.search(source_statement)
    return source_statement[: match.start()].rstrip() if match is not None else source_statement


def _json_safe(value: object) -> object | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        converted = [_json_safe(item) for item in value]
        return converted if all(item is not None for item in converted) else None
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        converted = {key: _json_safe(item) for key, item in value.items()}
        return converted if all(item is not None for item in converted.values()) else None
    return None
