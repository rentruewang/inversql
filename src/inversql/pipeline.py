# Copyright (c) The InverSQL Authors - All Rights Reserved

import dataclasses as dcls
import math
import typing
from collections import abc as cabc

import numpy as np
import pandas as pd
from sklearn import tree

from inversql.exprs import AndExpr, CmpExpr, CmpOp, DontCareExpr, Expr, OrExpr
from inversql.joins import Joiner, JoinResult, SharedColNameJoiner
from inversql.trees import sklearn_binary_tree_to_nodes

__all__ = [
    "CellRef",
    "CellSelection",
    "JoinSummary",
    "PipelineResult",
    "TableSpec",
    "TrainingSelection",
    "run_pipeline",
]


@dcls.dataclass(frozen=True)
class TableSpec:
    name: str
    df: pd.DataFrame


@dcls.dataclass(frozen=True)
class CellRef:
    row: int
    column: str


@dcls.dataclass(frozen=True)
class CellSelection:
    must_have: cabc.Sequence[CellRef | tuple[int, str]] = ()
    must_not_have: cabc.Sequence[CellRef | tuple[int, str]] = ()


@dcls.dataclass(frozen=True)
class TrainingSelection:
    positive_row_indices: cabc.Sequence[int] = ()


@dcls.dataclass(frozen=True)
class JoinSummary:
    how: str
    on: tuple[str, ...]
    row_count: int
    candidate_count: int


@dcls.dataclass(frozen=True)
class PipelineResult:
    joined_df: pd.DataFrame
    sql: str | None
    diagnostics: tuple[str, ...]
    join_summary: JoinSummary | None


@dcls.dataclass(frozen=True)
class _FeatureSpec:
    column: str
    kind: typing.Literal["numeric", "category"]
    value: object | None = None


def run_pipeline(
    left: TableSpec,
    right: TableSpec,
    cells: CellSelection | None = None,
    training: TrainingSelection | None = None,
    *,
    joiner: Joiner | None = None,
) -> PipelineResult:
    """
    Run the v1 InverSQL flow for two active tables.

    The facade intentionally keeps Streamlit out of the backend path. It joins the
    tables, derives binary labels from selected positive rows, trains a decision
    tree, and formats the resulting expression as SQL.
    """

    cells = cells or CellSelection()
    joiner = joiner or SharedColNameJoiner()

    diagnostics: list[str] = []
    selected_join, candidate_count = _first_valid_join(left.df, right.df, joiner)

    if selected_join is None:
        diagnostics.append("No valid join candidate was produced.")
        return PipelineResult(
            joined_df=pd.DataFrame(),
            sql=None,
            diagnostics=tuple(diagnostics),
            join_summary=None,
        )

    joined_df = _normalize_joined_df(selected_join.df)
    join_summary = JoinSummary(
        how=selected_join.how,
        on=tuple(selected_join.on),
        row_count=len(joined_df),
        candidate_count=candidate_count,
    )
    diagnostics.append(
        f"Using {selected_join.how} join on {_format_join_keys(selected_join.on)}."
    )

    positive_rows = _positive_rows(
        training=training,
        cells=cells,
        df=joined_df,
    )
    diagnostics.append(
        f"Training labels: {len(positive_rows)} positive, "
        f"{len(joined_df) - len(positive_rows)} negative."
    )
    cell_constraints, cell_diagnostics = _cell_constraints(
        joined_df, cells, left=left, right=right
    )
    diagnostics.extend(cell_diagnostics)

    if not positive_rows:
        if cell_constraints:
            diagnostics.append(
                "No positive rows were inferred; using cell constraints only."
            )
            sql = _build_sql(
                left=left,
                right=right,
                join=selected_join,
                output_columns=tuple(joined_df.columns),
                where_sql=cell_constraints,
            )
            return PipelineResult(joined_df, sql, tuple(diagnostics), join_summary)

        diagnostics.append("Select at least one must-have cell to generate SQL.")
        return PipelineResult(joined_df, None, tuple(diagnostics), join_summary)

    if len(positive_rows) == len(joined_df):
        if cell_constraints:
            diagnostics.append(
                "All joined rows are positive; using cell constraints only."
            )
            sql = _build_sql(
                left=left,
                right=right,
                join=selected_join,
                output_columns=tuple(joined_df.columns),
                where_sql=cell_constraints,
            )
            return PipelineResult(joined_df, sql, tuple(diagnostics), join_summary)

        diagnostics.append("Leave at least one row unselected to provide negatives.")
        return PipelineResult(joined_df, None, tuple(diagnostics), join_summary)

    feature_columns = tuple(joined_df.columns)
    if not feature_columns:
        diagnostics.append("No usable feature columns are available.")
        return PipelineResult(joined_df, None, tuple(diagnostics), join_summary)

    output_columns = tuple(joined_df.columns)

    x_train, feature_specs, encoding_diagnostics = _encode_features(
        joined_df, feature_columns
    )
    diagnostics.extend(encoding_diagnostics)

    if x_train.empty:
        diagnostics.append("No model features could be encoded.")
        return PipelineResult(joined_df, None, tuple(diagnostics), join_summary)

    y_train = np.zeros(len(joined_df), dtype=bool)
    y_train[list(positive_rows)] = True

    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(x_train.to_numpy(dtype=float), y_train)

    try:
        root = sklearn_binary_tree_to_nodes(clf)
        where_sql = _expr_to_sql(root.truth_exprs(), feature_specs, left, right)
        if cell_constraints:
            where_sql = f"({where_sql}) AND ({cell_constraints})"
    except Exception as exc:
        diagnostics.append(f"Decision tree could not be converted to SQL: {exc}")
        return PipelineResult(joined_df, None, tuple(diagnostics), join_summary)

    sql = _build_sql(
        left=left,
        right=right,
        join=selected_join,
        output_columns=output_columns,
        where_sql=where_sql,
    )
    return PipelineResult(joined_df, sql, tuple(diagnostics), join_summary)


def _first_valid_join(
    left: pd.DataFrame, right: pd.DataFrame, joiner: Joiner
) -> tuple[JoinResult | None, int]:
    candidate_count = 0
    selected: JoinResult | None = None

    for candidate in joiner(left, right):
        candidate_count += 1
        if candidate and selected is None:
            selected = candidate

    return selected, candidate_count


def _normalize_joined_df(df: pd.DataFrame) -> pd.DataFrame:
    if any(name is not None for name in df.index.names):
        return df.reset_index()

    return df.reset_index(drop=True)


def _valid_positive_rows(rows: cabc.Sequence[int], *, num_rows: int) -> tuple[int, ...]:
    return tuple(sorted({int(row) for row in rows if 0 <= int(row) < num_rows}))


def _positive_rows(
    *,
    training: TrainingSelection | None,
    cells: CellSelection,
    df: pd.DataFrame,
) -> tuple[int, ...]:
    if training is not None:
        return _valid_positive_rows(training.positive_row_indices, num_rows=len(df))

    rows = {
        cell.row
        for cell in map(_normalize_cell_ref, cells.must_have)
        if cell.column in df.columns and 0 <= cell.row < len(df)
    }
    return tuple(sorted(rows))


def _encode_features(
    df: pd.DataFrame, feature_columns: cabc.Sequence[str]
) -> tuple[pd.DataFrame, tuple[_FeatureSpec, ...], tuple[str, ...]]:
    encoded: dict[str, pd.Series] = {}
    specs: list[_FeatureSpec] = []
    diagnostics: list[str] = []

    for column in feature_columns:
        series = df[column]

        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            fill_value = float(numeric.median()) if numeric.notna().any() else 0.0
            encoded[column] = numeric.fillna(fill_value).astype(float)
            specs.append(_FeatureSpec(column=column, kind="numeric"))
            continue

        values = series.astype("string").fillna("<missing>")
        unique_values = list(pd.unique(values))

        if len(unique_values) > 32:
            diagnostics.append(
                f"Column {column!r} has {len(unique_values)} categories; "
                "using one-hot features for v1."
            )

        for value in unique_values:
            feature_name = f"{column}={value}"
            encoded[feature_name] = (values == value).astype(float)
            specs.append(_FeatureSpec(column=column, kind="category", value=value))

    return pd.DataFrame(encoded), tuple(specs), tuple(diagnostics)


def _cell_constraints(
    df: pd.DataFrame,
    cells: CellSelection,
    *,
    left: TableSpec,
    right: TableSpec,
) -> tuple[str, tuple[str, ...]]:
    diagnostics: list[str] = []
    must_have = _cell_values(df, cells.must_have)
    must_not_have = _cell_values(df, cells.must_not_have)

    diagnostics.append(
        f"Cell constraints: {sum(map(len, must_have.values()))} must-have, "
        f"{sum(map(len, must_not_have.values()))} must-not-have."
    )

    predicates: list[str] = []
    for column, values in must_have.items():
        predicates.append(
            _include_values_predicate(
                _source_ref(column, left, right),
                values,
            )
        )

    for column, values in must_not_have.items():
        must_have_values = must_have.get(column, ())
        allowed_values = tuple(
            value
            for value in values
            if not any(
                _same_sql_value(value, included) for included in must_have_values
            )
        )
        if len(allowed_values) != len(values):
            diagnostics.append(
                f"Ignored conflicting must-not-have values for column {column!r}."
            )

        if allowed_values:
            predicates.append(
                _exclude_values_predicate(
                    _source_ref(column, left, right),
                    allowed_values,
                )
            )

    return " AND ".join(predicates), tuple(diagnostics)


def _cell_values(
    df: pd.DataFrame, cells: cabc.Sequence[CellRef | tuple[int, str]]
) -> dict[str, tuple[object, ...]]:
    values_by_column: dict[str, list[object]] = {}

    for raw_cell in cells:
        cell = _normalize_cell_ref(raw_cell)
        if cell.column not in df.columns or not 0 <= cell.row < len(df):
            continue

        value = df.iloc[cell.row][cell.column]
        values = values_by_column.setdefault(cell.column, [])
        if not any(_same_sql_value(value, existing) for existing in values):
            values.append(value)

    return {column: tuple(values) for column, values in values_by_column.items()}


def _normalize_cell_ref(cell: CellRef | tuple[int, str]) -> CellRef:
    if isinstance(cell, CellRef):
        return cell

    row, column = cell
    return CellRef(row=int(row), column=str(column))


def _include_values_predicate(column: str, values: cabc.Sequence[object]) -> str:
    null_selected = any(_is_missing(value) for value in values)
    literals = tuple(
        _quote_literal(value) for value in values if not _is_missing(value)
    )

    predicates: list[str] = []
    if literals:
        predicates.append(_in_predicate(column, literals, include=True))

    if null_selected:
        predicates.append(f"{column} IS NULL")

    return " OR ".join(f"({predicate})" for predicate in predicates)


def _exclude_values_predicate(column: str, values: cabc.Sequence[object]) -> str:
    null_selected = any(_is_missing(value) for value in values)
    literals = tuple(
        _quote_literal(value) for value in values if not _is_missing(value)
    )

    predicates: list[str] = []
    if literals:
        predicates.append(
            f"({_in_predicate(column, literals, include=False)} OR {column} IS NULL)"
        )

    if null_selected:
        predicates.append(f"{column} IS NOT NULL")

    return " AND ".join(f"({predicate})" for predicate in predicates)


def _in_predicate(column: str, literals: cabc.Sequence[str], *, include: bool) -> str:
    if len(literals) == 1:
        op = "=" if include else "<>"
        return f"{column} {op} {literals[0]}"

    op = "IN" if include else "NOT IN"
    return f"{column} {op} ({', '.join(literals)})"


def _same_sql_value(left: object, right: object) -> bool:
    if _is_missing(left) and _is_missing(right):
        return True

    return left == right


def _expr_to_sql(
    expr: Expr,
    feature_specs: cabc.Sequence[_FeatureSpec],
    left: TableSpec,
    right: TableSpec,
) -> str:
    match expr:
        case CmpExpr(feat_idx=feat_idx, cmp=cmp, threshold=threshold):
            return _cmp_expr_to_sql(
                feature_specs[feat_idx], cmp, threshold, left, right
            )
        case AndExpr(left=left_expr, right=right_expr):
            left_sql = _expr_to_sql(left_expr, feature_specs, left, right)
            right_sql = _expr_to_sql(right_expr, feature_specs, left, right)
            return f"({left_sql}) AND ({right_sql})"
        case OrExpr(left=left_expr, right=right_expr):
            left_sql = _expr_to_sql(left_expr, feature_specs, left, right)
            right_sql = _expr_to_sql(right_expr, feature_specs, left, right)
            return f"({left_sql}) OR ({right_sql})"
        case DontCareExpr():
            return "TRUE"
        case _:
            raise TypeError(f"Unsupported expression type: {type(expr)}")


def _cmp_expr_to_sql(
    spec: _FeatureSpec,
    cmp: CmpOp,
    threshold: float,
    left: TableSpec,
    right: TableSpec,
) -> str:
    if spec.kind == "numeric":
        column = _source_ref(spec.column, left, right)
        return f"{column} {cmp.value} {_format_number(threshold)}"

    column = _source_ref(spec.column, left, right)
    value = _quote_literal(spec.value)

    if cmp in {CmpOp.LE, CmpOp.LT}:
        if threshold < 0:
            return "FALSE"
        if threshold < 1:
            return f"{column} <> {value}"
        return "TRUE"

    if cmp in {CmpOp.GT, CmpOp.GE}:
        if threshold < 0:
            return "TRUE"
        if threshold < 1:
            return f"{column} = {value}"
        return "FALSE"

    return f"{column} {cmp.value} {value}"


def _build_sql(
    *,
    left: TableSpec,
    right: TableSpec,
    join: JoinResult,
    output_columns: cabc.Sequence[str],
    where_sql: str,
) -> str:
    select_sql = ", ".join(
        f"{_source_ref(column, left, right)} AS {_quote_identifier(column)}"
        for column in output_columns
    )

    join_keys = tuple(join.on)
    join_sql = _join_sql(left=left, right=right, how=join.how, on=join_keys)

    return f"SELECT {select_sql}\nFROM {join_sql}\nWHERE {where_sql}"


def _join_sql(*, left: TableSpec, right: TableSpec, how: str, on: cabc.Sequence[str]):
    left_table = _quote_identifier(left.name)
    right_table = _quote_identifier(right.name)

    if how.lower() == "cross":
        return f"{left_table} CROSS JOIN {right_table}"

    conditions = " AND ".join(
        f"{left_table}.{_quote_identifier(column)} = "
        f"{right_table}.{_quote_identifier(column)}"
        for column in on
    )
    return f"{left_table} {how.upper()} JOIN {right_table} ON {conditions}"


def _source_ref(column: str, left: TableSpec, right: TableSpec) -> str:
    left_has = column in left.df.columns
    right_has = column in right.df.columns

    if left_has:
        return f"{_quote_identifier(left.name)}.{_quote_identifier(column)}"

    if right_has:
        return f"{_quote_identifier(right.name)}.{_quote_identifier(column)}"

    return _quote_identifier(column)


def _quote_identifier(identifier: str) -> str:
    escaped = str(identifier).replace('"', '""')
    return f'"{escaped}"'


def _quote_literal(value: object) -> str:
    if value is None:
        return "NULL"

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _format_number(float(value))

    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _format_number(value: float) -> str:
    if math.isfinite(value) and value.is_integer():
        return str(int(value))

    return repr(float(value))


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _format_join_keys(on: cabc.Sequence[str]) -> str:
    if not on:
        return "no columns"

    return ", ".join(on)
