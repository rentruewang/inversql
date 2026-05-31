# Copyright (c) The InverSQL Authors - All Rights Reserved

import dataclasses as dcls
import pathlib
import typing
from collections import abc as cabc

import pandas as pd
import streamlit as st

__all__ = [
    "CellRef",
    "MockSqlResult",
    "SelectionState",
    "SelectionTotals",
    "TableSpec",
    "_BUCKET_STYLES",
    "_highlight_selection",
    "_merge_selection_states",
    "_mock_sql",
    "_selection_mode_for",
    "_selection_state_from_event",
    "_selection_totals",
    "_subtract_selection_states",
    "main",
]

_LOGO_PATH = pathlib.Path(__file__).parents[2] / "assets" / "logo.svg"
_TARGETS = ("Cell", "Row", "Column")
_MODES = {"Cell": "multi-cell", "Row": "multi-row", "Column": "multi-column"}
_BUCKETS = {"must_have": "Must have", "must_not_have": "Must not have"}
_BUCKET_STYLES = {
    "must_have": "background-color: #fff3bf",
    "must_not_have": "background-color: #ffd6d6",
}


class TableSpec(typing.NamedTuple):
    name: str
    df: pd.DataFrame


class CellRef(typing.NamedTuple):
    row: int
    column: str


@dcls.dataclass(frozen=True)
class SelectionState:
    rows: cabc.Sequence[int] = ()
    columns: cabc.Sequence[str] = ()
    cells: cabc.Sequence[CellRef] = ()


@dcls.dataclass(frozen=True)
class SelectionTotals:
    rows: int = 0
    columns: int = 0
    cells: int = 0


@dcls.dataclass(frozen=True)
class MockSqlResult:
    main: str
    candidates: cabc.Sequence[str]


def main() -> None:
    st.set_page_config("InverSQL", layout="wide", initial_sidebar_state="collapsed")
    _css()
    _header()

    with st.container(border=True):
        st.subheader("Table Upload")
        uploaded = st.file_uploader("CSV files", "csv", accept_multiple_files=True)
        tables = _load_tables(uploaded)
        if not tables:
            st.info("Using demo tables. Upload CSV files to annotate your own data.")
            tables = _demo_tables()
        st.caption(
            " · ".join(f"{t.name}: {len(t.df)}x{len(t.df.columns)}" for t in tables)
        )

    annotation_col, sql_col = st.columns([1.65, 1], gap="large")
    with annotation_col, st.container(border=True):
        st.subheader("Table Annotation")
        target = st.segmented_control("Selection type", _TARGETS, default="Cell")
        target = str(target or "Cell")
        must_have, must_not_have = _annotation_workbench(tables, target)
        _metrics(must_have, must_not_have)

    with sql_col, st.container(border=True):
        _render_sql(_mock_sql(tables, must_have, must_not_have))


def _css() -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"], div[data-testid="collapsedControl"] {display:none;}
        .block-container {max-width:1500px; padding-top:1.4rem;}
        h1, h2, h3 {letter-spacing:0;}
        textarea {font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _header() -> None:
    _, center, _ = st.columns([2.2, 1, 2.2])
    with center:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH), use_container_width=True)
        st.markdown(
            "<h1 style='text-align:center'>inversql</h1>", unsafe_allow_html=True
        )


def _load_tables(files) -> list[TableSpec]:
    if not files:
        return []
    tables, used = [], set()
    cols = st.columns(min(len(files), 4))
    for idx, file in enumerate(files):
        default = pathlib.PurePath(file.name).stem or f"table_{idx + 1}"
        with cols[idx % len(cols)]:
            name = st.text_input(
                f"Name for {file.name}",
                default,
                key=f"name_{idx}_{file.name}",
            )
        name = _unique_table_name((name or default).strip(), used)
        used.add(name)
        try:
            file.seek(0)
            tables.append(TableSpec(name, pd.read_csv(file)))
        except Exception as exc:
            st.error(f"Could not read {file.name}: {exc}")
    return tables


def _demo_tables() -> list[TableSpec]:
    return [
        TableSpec(
            "customers", pd.DataFrame({"id": [1, 2, 3], "region": ["W", "E", "W"]})
        ),
        TableSpec(
            "orders", pd.DataFrame({"id": [10, 11, 12], "customer_id": [1, 2, 1]})
        ),
        TableSpec(
            "products", pd.DataFrame({"sku": ["A", "B", "C"], "active": [1, 1, 0]})
        ),
    ]


def _annotation_workbench(
    tables: cabc.Sequence[TableSpec],
    target: str,
) -> tuple[SelectionTotals, SelectionTotals]:
    tabs = st.tabs(list(_BUCKETS.values()))
    for tab, bucket in zip(tabs, _BUCKETS):
        with tab:
            for table_tab, table in zip(st.tabs([t.name for t in tables]), tables):
                with table_tab:
                    _annotation_table(bucket, table, target)
    return _bucket_totals("must_have", tables), _bucket_totals("must_not_have", tables)


def _annotation_table(bucket: str, table: TableSpec, target: str) -> None:
    state = _all_selection(bucket, table)
    event = st.dataframe(
        table.df.style.apply(
            lambda data: _highlight_selection(data, state, bucket), axis=None
        ),
        key=_key(bucket, target, table),
        on_select="rerun",
        selection_mode=_selection_mode_for(target),
        use_container_width=True,
        height=320,
    )
    current = _selection_state_from_event(event, target, table.df)
    if _store(bucket, target, table, current):
        st.rerun()

    remove_col, clear_col = st.columns(2)
    if remove_col.button(
        f"Remove selected {target.lower()}s",
        key=f"remove_{_key(bucket, target, table)}",
        use_container_width=True,
    ) and _remove(bucket, target, table, current):
        st.rerun()
    if clear_col.button(
        "Clear table",
        key=f"clear_{_key(bucket, target, table)}",
        use_container_width=True,
    ) and _clear(bucket, table):
        st.rerun()

    totals = _selection_totals([state])
    st.caption(
        f"{_BUCKETS[bucket]}: {totals.rows} rows, {totals.columns} columns, {totals.cells} cells"
    )


def _highlight_selection(
    data: pd.DataFrame, state: SelectionState, bucket: str
) -> pd.DataFrame:
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    style, columns = _BUCKET_STYLES[bucket], _column_positions(data)
    for row in state.rows:
        if 0 <= row < len(data):
            styles.iloc[row, :] = style
    for column in state.columns:
        if column in columns:
            styles.iloc[:, columns[column]] = style
    for cell in state.cells:
        if 0 <= cell.row < len(data) and cell.column in columns:
            styles.iat[cell.row, columns[cell.column]] = style
    return styles


def _metrics(must_have: SelectionTotals, must_not_have: SelectionTotals) -> None:
    cols = st.columns(6)
    for col, label, value in zip(
        cols,
        ["Have rows", "Have cols", "Have cells", "Not rows", "Not cols", "Not cells"],
        [*dcls.astuple(must_have), *dcls.astuple(must_not_have)],
    ):
        col.metric(label, value)


def _selection_mode_for(target: str) -> str:
    return _MODES.get(target, "multi-cell")


def _selection_state_from_event(event, target: str, df: pd.DataFrame) -> SelectionState:
    selection = _event_selection(event)
    if target == "Row":
        return SelectionState(rows=_selected_rows(selection, df))
    if target == "Column":
        return SelectionState(columns=_selected_columns(selection, df))
    return SelectionState(cells=_selected_cells(selection, df))


def _selected_rows(selection, df: pd.DataFrame) -> tuple[int, ...]:
    rows = []
    for row in _selection_values(selection, "rows"):
        try:
            row = int(row)
        except TypeError, ValueError:
            continue
        if row not in rows and 0 <= row < len(df):
            rows.append(row)
    return tuple(rows)


def _selected_columns(selection, df: pd.DataFrame) -> tuple[str, ...]:
    cols = []
    for column in _selection_values(selection, "columns"):
        column = _column_name(column, df)
        if column is not None and column not in cols:
            cols.append(column)
    return tuple(cols)


def _selected_cells(selection, df: pd.DataFrame) -> tuple[CellRef, ...]:
    cells = []
    for cell in _selection_values(selection, "cells"):
        cell = _normalize_selected_cell(cell, df)
        if cell is not None and cell not in cells:
            cells.append(cell)
    return tuple(cells)


def _normalize_selected_cell(cell, df: pd.DataFrame) -> CellRef | None:
    if isinstance(cell, dict):
        row, column = cell.get("row"), cell.get("column")
    else:
        try:
            row, column = cell
        except TypeError, ValueError:
            return None
    try:
        row = int(row)
    except TypeError, ValueError:
        return None
    column = _column_name(column, df)
    if column is None or not 0 <= row < len(df):
        return None
    return CellRef(row, column)


def _column_name(value, df: pd.DataFrame) -> str | None:
    text, columns = str(value), _column_positions(df)
    if text in columns:
        return text
    try:
        idx = int(value)
    except TypeError, ValueError:
        return None
    return str(df.columns[idx]) if 0 <= idx < len(df.columns) else None


def _selection_values(selection, name: str):
    values = getattr(selection, name, None)
    if values is None and isinstance(selection, dict):
        values = selection.get(name, [])
    return values or []


def _event_selection(event):
    return getattr(event, "selection", None) or (
        event.get("selection", {}) if isinstance(event, dict) else {}
    )


def _key(bucket: str, target: str, table: TableSpec) -> str:
    return f"{bucket}_{target.lower()}_{table.name}"


def _state_key(bucket: str, target: str, table: TableSpec) -> str:
    return f"stored_{_key(bucket, target, table)}"


def _store(
    bucket: str, target: str, table: TableSpec, selected: SelectionState
) -> bool:
    if not selected.rows and not selected.columns and not selected.cells:
        return False
    key = _state_key(bucket, target, table)
    merged = _merge_selection_states(
        st.session_state.get(key, SelectionState()), selected
    )
    if merged == st.session_state.get(key, SelectionState()):
        return False
    st.session_state[key] = merged
    return True


def _remove(
    bucket: str, target: str, table: TableSpec, selected: SelectionState
) -> bool:
    key = _state_key(bucket, target, table)
    old = st.session_state.get(key, SelectionState())
    new = _subtract_selection_states(old, selected)
    st.session_state[key] = new
    return old != new


def _clear(bucket: str, table: TableSpec) -> bool:
    old = [_all_selection(bucket, table)]
    for target in _TARGETS:
        st.session_state[_state_key(bucket, target, table)] = SelectionState()
    return old != [_all_selection(bucket, table)]


def _all_selection(bucket: str, table: TableSpec) -> SelectionState:
    return _merge_many(
        st.session_state.get(_state_key(bucket, target, table), SelectionState())
        for target in _TARGETS
    )


def _bucket_totals(bucket: str, tables: cabc.Sequence[TableSpec]) -> SelectionTotals:
    totals = [_selection_totals([_all_selection(bucket, table)]) for table in tables]
    return SelectionTotals(
        *(
            sum(getattr(total, field.name) for total in totals)
            for field in dcls.fields(SelectionTotals)
        )
    )


def _merge_many(states: cabc.Iterable[SelectionState]) -> SelectionState:
    out = SelectionState()
    for state in states:
        out = _merge_selection_states(out, state)
    return out


def _merge_selection_states(a: SelectionState, b: SelectionState) -> SelectionState:
    return SelectionState(
        _unique(a.rows, b.rows),
        _unique(a.columns, b.columns),
        _unique(a.cells, b.cells),
    )


def _subtract_selection_states(a: SelectionState, b: SelectionState) -> SelectionState:
    return SelectionState(
        _remove_values(a.rows, b.rows),
        _remove_values(a.columns, b.columns),
        _remove_values(a.cells, b.cells),
    )


def _unique(first: cabc.Sequence, second: cabc.Sequence) -> tuple:
    return tuple(dict.fromkeys([*first, *second]))


def _remove_values(values: cabc.Sequence, to_remove: cabc.Sequence) -> tuple:
    removed = set(to_remove)
    return tuple(value for value in values if value not in removed)


def _selection_totals(states: cabc.Sequence[SelectionState]) -> SelectionTotals:
    rows, cols, cells = set(), set(), set()
    for state in states:
        rows.update(state.rows)
        cols.update(state.columns)
        cells.update(state.cells)
    return SelectionTotals(len(rows), len(cols), len(cells))


def _column_positions(df: pd.DataFrame) -> dict[str, int]:
    return {str(column): idx for idx, column in enumerate(df.columns)}


def _mock_sql(
    tables: cabc.Sequence[TableSpec],
    must_have: SelectionTotals,
    must_not_have: SelectionTotals,
) -> MockSqlResult:
    names = [_sql_identifier(table.name) for table in tables]
    aliases = [f"t{i + 1}" for i in range(len(names))]
    select = ", ".join(f"{alias}.*" for alias in aliases)
    joins = "\n".join(
        f"JOIN {name} AS {alias} ON TRUE" for name, alias in zip(names[1:], aliases[1:])
    )
    comment = _annotation_comment(must_have, must_not_have)
    main = (
        f"SELECT {select}\nFROM {names[0]} AS {aliases[0]}\n"
        f"{joins}\nWHERE /* {comment} */ TRUE;"
    )
    candidates = tuple(f"-- Candidate {i}\n{main}" for i in range(1, 4))
    return MockSqlResult(main, candidates)


def _annotation_comment(
    must_have: SelectionTotals, must_not_have: SelectionTotals
) -> str:
    return f"must-have {must_have.rows} rows / {must_have.columns} columns / {must_have.cells} cells; must-not-have {must_not_have.rows} rows / {must_not_have.columns} columns / {must_not_have.cells} cells"


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _render_sql(sql: MockSqlResult) -> None:
    st.subheader("Generated SQL")
    st.text_area("SQL", sql.main, height=290, disabled=True)
    st.subheader("Candidate SQL")
    for i, candidate in enumerate(sql.candidates, 1):
        with st.expander(f"Candidate {i}", expanded=i == 1):
            st.code(candidate, "sql")


def _unique_table_name(name: str, used: set[str]) -> str:
    if name not in used:
        return name
    idx = 2
    while f"{name}_{idx}" in used:
        idx += 1
    return f"{name}_{idx}"


if __name__ == "__main__":
    main()
