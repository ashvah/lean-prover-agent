"""Theorem-level adapter for local Lean-Workbook tactic-transition Parquet data."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pyarrow import parquet

from leanproof.datasets.schema import CanonicalTheorem, ReferenceTrajectoryStep

SOURCE_NAME = "lean_workbook"
_STATEMENT_COLUMNS = ("formal_statement", "lean_statement")
_SOURCE_ID_COLUMNS = ("id", "source_id", "problem_id")
_STATUS_COLUMNS = ("status",)
_INFORMAL_COLUMNS = ("natural_language_statement", "informal_statement", "problem")
_ANSWER_COLUMNS = ("answer",)
_TACTIC_COLUMNS = ("tactic",)
_STATE_BEFORE_COLUMNS = ("state_before",)
_STATE_AFTER_COLUMNS = ("state_after",)
_ORDER_COLUMNS = ("step", "step_id", "tactic_index")
_TERMINAL_SORRY_PLACEHOLDER = re.compile(r"\s*:=\s*by\s+sorry\s*\Z")


class LeanWorkbookSchemaError(ValueError):
    """Raised when a Parquet file lacks the required Lean-Workbook schema."""


class GroupMappingError(ValueError):
    """Raised for one recoverable malformed theorem-level source group."""


RowMappingError = GroupMappingError


@dataclass(frozen=True)
class RawRow:
    index: int
    values: dict[str, object]


@dataclass(frozen=True)
class RawTheoremGroup:
    source_id: str
    rows: tuple[RawRow, ...]


@dataclass(frozen=True)
class GroupedRows:
    """Deterministic theorem groups plus truthful source traversal accounting."""

    source_tactic_rows_scanned: int
    raw_tactic_rows: int
    theorem_groups: tuple[RawTheoremGroup, ...]
    invalid_reasons: Mapping[str, int]


class LeanWorkbookAdapter:
    """Aggregate raw tactic rows by theorem ID and map one canonical record per group."""

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
        self._source_id_column = self._required_column(columns, _SOURCE_ID_COLUMNS, "theorem ID")
        self._statement_column = self._required_column(
            columns, _STATEMENT_COLUMNS, "formal statement"
        )
        self._status_column = self._required_column(columns, _STATUS_COLUMNS, "status")
        self._tactic_column = self._required_column(columns, _TACTIC_COLUMNS, "tactic")
        self._state_before_column = self._required_column(
            columns, _STATE_BEFORE_COLUMNS, "state_before"
        )
        self._state_after_column = self._required_column(
            columns, _STATE_AFTER_COLUMNS, "state_after"
        )
        self._informal_column = _first_available(columns, _INFORMAL_COLUMNS)
        self._answer_column = _first_available(columns, _ANSWER_COLUMNS)
        self._order_column = _first_available(columns, _ORDER_COLUMNS)
        self._mapped_columns = {
            column
            for column in (
                self._source_id_column,
                self._statement_column,
                self._status_column,
                self._tactic_column,
                self._state_before_column,
                self._state_after_column,
                self._informal_column,
                self._answer_column,
                self._order_column,
            )
            if column is not None
        }

    def load_groups(self, *, limit: int | None = None) -> GroupedRows:
        """Read all rows, group by source ID, and optionally retain the first N groups."""

        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero")
        groups: OrderedDict[str, list[RawRow]] = OrderedDict()
        invalid_reasons: Counter[str] = Counter()
        source_rows = 0
        for batch in self._parquet_file.iter_batches(batch_size=1024):
            for values in batch.to_pylist():
                row = RawRow(index=source_rows, values=values)
                source_rows += 1
                raw_source_id = values.get(self._source_id_column)
                if raw_source_id is None or not str(raw_source_id).strip():
                    invalid_reasons["source_id_missing"] += 1
                    continue
                source_id = str(raw_source_id).strip()
                groups.setdefault(source_id, []).append(row)

        selected_items = list(groups.items())[:limit]
        theorem_groups = tuple(
            RawTheoremGroup(source_id=source_id, rows=tuple(rows))
            for source_id, rows in selected_items
        )
        return GroupedRows(
            source_tactic_rows_scanned=source_rows,
            raw_tactic_rows=sum(len(group.rows) for group in theorem_groups),
            theorem_groups=theorem_groups,
            invalid_reasons=(dict(invalid_reasons) if limit is None else {}),
        )

    def map_group(self, group: RawTheoremGroup) -> CanonicalTheorem:
        """Validate group invariants and preserve every tactic transition in source order."""

        source_statement = self._consistent_text(
            group, self._statement_column, required=True, stringify_scalars=False
        )
        source_status = self._consistent_text(
            group, self._status_column, required=True, stringify_scalars=False
        ).lower()
        informal_statement = self._consistent_text(
            group, self._informal_column, required=False, stringify_scalars=False
        )
        answer = self._consistent_text(
            group, self._answer_column, required=False, stringify_scalars=True
        )
        ordered_rows = self._ordered_rows(group)
        trajectory = tuple(
            ReferenceTrajectoryStep(
                step=step,
                state_before=_optional_text(row.values, self._state_before_column),
                tactic=_optional_text(row.values, self._tactic_column),
                state_after=_optional_text(row.values, self._state_after_column),
            )
            for step, row in enumerate(ordered_rows)
        )
        statement = _remove_terminal_sorry_placeholder(source_statement)
        metadata = self._common_metadata(group)
        metadata["raw_row_indices"] = [row.index for row in ordered_rows]
        metadata["trajectory_order"] = self._order_column or "parquet_row_order"
        if statement != source_statement:
            metadata["source_formal_statement"] = source_statement
        canonical_id = (
            f"lean_workbook_{hashlib.sha256(group.source_id.encode('utf-8')).hexdigest()[:24]}"
        )
        return CanonicalTheorem(
            id=canonical_id,
            source=SOURCE_NAME,
            source_id=group.source_id,
            statement=statement,
            informal_statement=informal_statement,
            answer=answer,
            source_status=source_status,
            reference_trajectory=trajectory,
            reference_proof=None,
            metadata=metadata,
        )

    def _consistent_text(
        self,
        group: RawTheoremGroup,
        column: str | None,
        *,
        required: bool,
        stringify_scalars: bool,
    ) -> str | None:
        if column is None:
            if required:
                raise GroupMappingError("required_group_field_missing")
            return None
        values: list[str] = []
        for row in group.rows:
            value = row.values.get(column)
            if value is None or (isinstance(value, str) and not value.strip()):
                if required:
                    raise GroupMappingError(f"{column}_missing")
                continue
            if isinstance(value, str):
                normalized = value.strip() if column == self._status_column else value
            elif stringify_scalars and isinstance(value, (int, float, bool)):
                normalized = str(value)
            else:
                raise GroupMappingError(f"{column}_not_string")
            if normalized not in values:
                values.append(normalized)
        if not values:
            if required:
                raise GroupMappingError(f"{column}_missing")
            return None
        comparison_values = (
            {value.lower() for value in values} if column == self._status_column else set(values)
        )
        if len(comparison_values) > 1:
            raise GroupMappingError(f"{column}_conflict")
        return values[0]

    def _ordered_rows(self, group: RawTheoremGroup) -> tuple[RawRow, ...]:
        if self._order_column is None:
            return group.rows
        ordered: list[tuple[float, RawRow]] = []
        for row in group.rows:
            value = row.values.get(self._order_column)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise GroupMappingError(f"{self._order_column}_invalid")
            ordered.append((float(value), row))
        if len({order for order, _ in ordered}) != len(ordered):
            raise GroupMappingError(f"{self._order_column}_duplicate")
        return tuple(row for _, row in sorted(ordered, key=lambda item: item[0]))

    def _common_metadata(self, group: RawTheoremGroup) -> dict[str, object]:
        metadata: dict[str, object] = {}
        for key in group.rows[0].values:
            if key in self._mapped_columns:
                continue
            values = [_json_safe(row.values.get(key)) for row in group.rows]
            non_null_values = [value for value in values if value is not None]
            if non_null_values and all(value == non_null_values[0] for value in non_null_values):
                metadata[key] = non_null_values[0]
        return metadata

    @staticmethod
    def _required_column(columns: tuple[str, ...], candidates: tuple[str, ...], label: str) -> str:
        column = _first_available(columns, candidates)
        if column is None:
            expected = ", ".join(candidates)
            available = ", ".join(columns) or "none"
            raise LeanWorkbookSchemaError(
                f"Lean-Workbook schema requires a {label} column ({expected}); "
                f"available columns: {available}"
            )
        return column


def _first_available(columns: tuple[str, ...], candidates: tuple[str, ...]) -> str | None:
    return next((candidate for candidate in candidates if candidate in columns), None)


def _optional_text(values: dict[str, object], column: str) -> str | None:
    value = values.get(column)
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    raise GroupMappingError(f"{column}_not_string")


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
