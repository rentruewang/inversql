# Copyright (c) The InverSQL Authors - All Rights Reserved

import pathlib

import pandas as pd
import streamlit as st

from inversql.pipeline import (
    CellRef,
    CellSelection,
    TableSpec,
    run_pipeline,
)


def main() -> None:
    st.set_page_config(page_title="InverSQL", layout="wide")
    st.title("InverSQL")

    uploaded_files = st.sidebar.file_uploader(
        "CSV files",
        type="csv",
        accept_multiple_files=True,
    )

    tables = _load_tables(uploaded_files)
    if not tables:
        st.info("Upload at least two CSV files to build a query.")
        return

    _render_table_previews(tables)

    if len(tables) < 2:
        st.info("Upload one more CSV file to run the v1 two-table pipeline.")
        return

    left, right = _active_tables(tables)
    preview = run_pipeline(left, right)

    if preview.join_summary is None:
        st.subheader("Joined Preview")
        st.dataframe(preview.joined_df, use_container_width=True, height=360)
        _render_diagnostics(preview.diagnostics)
        return

    st.subheader("Joined Preview")
    must_have_event, must_not_have_event = _selection_events(
        preview.joined_df, left=left, right=right
    )
    cell_selection = CellSelection(
        must_have=_selected_cells(must_have_event, preview.joined_df),
        must_not_have=_selected_cells(must_not_have_event, preview.joined_df),
    )

    result = run_pipeline(
        left,
        right,
        cells=cell_selection,
    )

    _render_summary(result, cells=cell_selection)
    _render_sql(result.sql)
    _render_diagnostics(result.diagnostics)


def _load_tables(uploaded_files) -> list[TableSpec]:
    tables: list[TableSpec] = []
    used_names: set[str] = set()

    for idx, uploaded_file in enumerate(uploaded_files or []):
        default_name = pathlib.PurePath(uploaded_file.name).stem or f"table_{idx + 1}"
        raw_name = st.sidebar.text_input(
            f"Table {idx + 1} name",
            value=default_name,
            key=f"table_name_{idx}_{uploaded_file.name}",
        )
        table_name = _unique_table_name(raw_name.strip() or default_name, used_names)
        used_names.add(table_name)

        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.sidebar.error(f"Could not read {uploaded_file.name}: {exc}")
            continue

        tables.append(TableSpec(name=table_name, df=df))

    return tables


def _render_table_previews(tables: list[TableSpec]) -> None:
    st.subheader("Uploaded Tables")
    tabs = st.tabs([table.name for table in tables])

    for tab, table in zip(tabs, tables):
        with tab:
            st.dataframe(table.df, use_container_width=True, height=260)


def _active_tables(tables: list[TableSpec]) -> tuple[TableSpec, TableSpec]:
    table_names = [table.name for table in tables]
    table_by_name = {table.name: table for table in tables}

    left_name = st.sidebar.selectbox("Left table", table_names, index=0)
    right_options = [name for name in table_names if name != left_name]
    right_name = st.sidebar.selectbox("Right table", right_options, index=0)

    return table_by_name[left_name], table_by_name[right_name]


def _selection_events(df: pd.DataFrame, *, left: TableSpec, right: TableSpec):
    must_have_tab, must_not_have_tab = st.tabs(
        ["Must-have cells", "Must-not-have cells"]
    )

    with must_have_tab:
        must_have_event = st.dataframe(
            df,
            key=f"must_have_cells_{left.name}_{right.name}",
            on_select="rerun",
            selection_mode=["multi-row", "multi-column", "multi-cell"],
            use_container_width=True,
            height=420,
        )

    with must_not_have_tab:
        must_not_have_event = st.dataframe(
            df,
            key=f"must_not_have_cells_{left.name}_{right.name}",
            on_select="rerun",
            selection_mode=["multi-row", "multi-column", "multi-cell"],
            use_container_width=True,
            height=420,
        )

    return must_have_event, must_not_have_event


def _selected_cells(event, df: pd.DataFrame) -> tuple[CellRef, ...]:
    selection = _event_selection(event)
    selected: list[CellRef] = []
    seen: set[tuple[int, str]] = set()

    for row in _selection_values(selection, "rows"):
        try:
            row_idx = int(row)
        except TypeError, ValueError:
            continue

        for column in df.columns:
            _append_cell(selected, seen, row_idx, str(column), df)

    for column in _selection_values(selection, "columns"):
        if column not in df.columns:
            continue

        for row_idx in range(len(df)):
            _append_cell(selected, seen, row_idx, str(column), df)

    for cell in _selection_values(selection, "cells"):
        normalized = _normalize_selected_cell(cell)
        if normalized is not None:
            _append_cell(selected, seen, normalized.row, normalized.column, df)

    return tuple(selected)


def _append_cell(
    selected: list[CellRef],
    seen: set[tuple[int, str]],
    row: int,
    column: str,
    df: pd.DataFrame,
) -> None:
    key = row, column
    if key in seen or column not in df.columns or not 0 <= row < len(df):
        return

    selected.append(CellRef(row=row, column=column))
    seen.add(key)


def _selection_values(selection, name: str):
    values = getattr(selection, name, None)
    if values is None and isinstance(selection, dict):
        return selection.get(name, [])

    return values or []


def _normalize_selected_cell(cell) -> CellRef | None:
    if isinstance(cell, dict):
        row = cell.get("row")
        column = cell.get("column")
    else:
        try:
            row, column = cell
        except TypeError, ValueError:
            return None

    if row is None or column is None:
        return None

    return CellRef(row=int(row), column=str(column))


def _event_selection(event):
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        return event.get("selection", {})

    return selection


def _render_summary(
    result,
    *,
    cells: CellSelection,
) -> None:
    summary = result.join_summary
    if summary is None:
        return

    positive_rows = {cell.row for cell in cells.must_have}

    cols = st.columns(6)
    cols[0].metric("Joined rows", summary.row_count)
    cols[1].metric("Join type", summary.how.upper())
    cols[2].metric("Join keys", ", ".join(summary.on) or "None")
    cols[3].metric("Join candidates", summary.candidate_count)
    cols[4].metric("Positive rows", len(positive_rows))
    cols[5].metric("Cell rules", len(cells.must_have) + len(cells.must_not_have))


def _render_sql(sql: str | None) -> None:
    st.subheader("Generated SQL")

    if sql:
        st.code(sql, language="sql")
        return

    st.code("-- SQL will appear after selecting at least one must-have cell.", "sql")


def _render_diagnostics(diagnostics: tuple[str, ...]) -> None:
    with st.expander("Diagnostics", expanded=True):
        if not diagnostics:
            st.write("No diagnostics.")
            return

        for diagnostic in diagnostics:
            st.write(f"- {diagnostic}")


def _unique_table_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        return name

    idx = 2
    while f"{name}_{idx}" in used_names:
        idx += 1

    return f"{name}_{idx}"


if __name__ == "__main__":
    main()
