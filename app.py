# Copyright (c) The InverSQL Authors - All Rights Reserved

import collections
import contextlib as ctxl
import dataclasses as dcls
import pathlib
import typing
from collections import abc as cabc

import pandas as pd
import streamlit as st
from streamlit.runtime import uploaded_file_manager as up_man

from inversql.pipelines import Pipeline
from inversql.rels import CellLoc, SourceRelation

_LOGO_PATH = pathlib.Path(__file__).parent / "assets" / "logo.svg"


@dcls.dataclass
class SessState:
    KEY: typing.ClassVar[str] = "__session_state__"

    cells: dict[str, set[CellLoc]] = dcls.field(
        default_factory=lambda: collections.defaultdict(set)
    )
    "The selected cells, by tables."

    def commit(self):
        "Commit the changes back to state."
        st.session_state[self.KEY] = self

    @classmethod
    @ctxl.contextmanager
    def get(cls) -> cabc.Generator[typing.Self]:
        "Get the state, you can modify it. By end of scope it will commit back."
        state = cls.fetch()
        try:
            yield state
        finally:
            state.commit()

    @classmethod
    def fetch(cls) -> typing.Self:
        "Get the current states."
        if cls.KEY in st.session_state:
            result = st.session_state[cls.KEY]
            return result

        else:
            result = cls()
            st.session_state[cls.KEY] = result
            return result


def _css() -> None:
    st.html("""
        <style>
        section[data-testid="stSidebar"], div[data-testid="collapsedControl"] {display:none;}
        .block-container {max-width:1500px; padding-top:5rem;}
        h1, h2, h3 {letter-spacing:0;}
        textarea {font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;}
        </style>
        """)


def _header() -> None:
    _, center, _ = st.columns([2.2, 1, 2.2])
    with center:
        assert _LOGO_PATH.exists()
        st.image(str(_LOGO_PATH))


def _load_tables(csvs: list[up_man.UploadedFile]) -> dict[str, pd.DataFrame]:
    "Load the tables as our representation."

    if not csvs:
        return {}

    tables: dict[str, pd.DataFrame] = {}
    cols = st.columns(min(len(csvs), 4))

    for idx, file in enumerate(csvs):
        default = pathlib.PurePath(file.name).stem or f"table_{idx}"

        with cols[idx % len(cols)]:
            name = st.text_input(
                f"Name for {file.name}",
                default,
                key=f"name_{idx}_{file.name}",
            )

        try:
            file.seek(0)
            tables[name] = pd.read_csv(file)
        except Exception as exc:
            st.error(f"Could not read {file.name}: {exc}")

    return tables


def _render_table_tabs(*tables: SourceRelation):
    if not tables:
        st.info("No tables to show.")
        return

    for tab, table in zip(st.tabs([table.name for table in tables]), tables):
        with tab:
            _render_table(table)


def _highlight_coords(df: pd.DataFrame, coords: cabc.Iterable[CellLoc]) -> pd.DataFrame:
    style_df = pd.DataFrame("", index=df.index, columns=df.columns)

    for row, col in coords:
        style_df.iloc[row, col] = "background-color: #708090; font-weight: bold;"
    return style_df


def _event_selection(event):
    if result := getattr(event, "selection", None):
        return result

    if isinstance(event, dict):
        return event.get("selection", {})

    return {}


def _toggle_cell(state: set[CellLoc], cell: CellLoc) -> None:
    if cell in state:
        state.remove(cell)
    else:
        state.add(cell)


def _render_table(table: SourceRelation):
    with SessState.get() as state:
        _render_table_stateless(state, table)


def _render_table_stateless(state: SessState, table: SourceRelation):

    data = table.data()
    data_cols = list(data.columns)
    event = st.dataframe(
        data.style.apply(lambda _: _highlight_coords(data, table.cells), axis=None),
        on_select="rerun",
        selection_mode="single-cell",
        width="stretch",
        height=320,
        key=table.name,
    )

    selection = event.get("selection", {})
    cells = selection.get("cells", [])

    for row, col in cells:
        cell_loc = CellLoc(row_idx=row, col_idx=data_cols.index(col))
        selected = state.cells[table.name]

        _toggle_cell(selected, cell_loc)


def _make_sources(tables: dict[str, pd.DataFrame]) -> cabc.Generator[SourceRelation]:
    state = SessState.fetch()

    for name, df in tables.items():
        yield SourceRelation(name, df, cells=state.cells.get(name, ()))


def _gen_sql_and_render(*tables: SourceRelation) -> None:
    if not tables:
        st.info("No tables to show.")
        return

    if not any(any(table.cells) for table in tables):
        st.info("Should perform a selection to see SQL.")
        return

    pipeline = Pipeline()
    sqls = pipeline(*tables)

    main, *candidates = sqls
    st.subheader("Generated SQL")
    st.code(main, "sql", wrap_lines=True)


if __name__ == "__main__":

    st.set_page_config("InverSQL", layout="wide", initial_sidebar_state="collapsed")

    _css()
    _header()

    with st.container(border=True):
        st.subheader("Table Upload")
        uploaded = st.file_uploader("CSV files", "csv", accept_multiple_files=True)
        if not (tables := _load_tables(uploaded)):
            st.info("Upload CSV files to annotate your own data.")

        st.caption(
            " · ".join(
                f"{name}: {len(table)}x{len(table.columns)}"
                for name, table in tables.items()
            )
        )

    annotation_col, sql_col = st.columns([1.65, 1], gap="large")
    sources = list(_make_sources(tables))

    with annotation_col, st.container(border=True):
        _render_table_tabs(*sources)

    with sql_col, st.container(border=True):
        _gen_sql_and_render(*sources)
