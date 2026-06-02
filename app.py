# Copyright (c) The InverSQL Authors - All Rights Reserved

from inversql.pipelines import Pipeline
import html
import pathlib

import pandas as pd
import streamlit as st
from streamlit.runtime import uploaded_file_manager as up_man

from inversql.rels import SourceRelation

_LOGO_PATH = pathlib.Path(__file__).parent / "assets" / "logo.svg"


def _css() -> None:
    st.html("""
        <style>
        section[data-testid="stSidebar"], div[data-testid="collapsedControl"] {display:none;}
        .block-container {max-width:1500px; padding-top:1.4rem;}
        h1, h2, h3 {letter-spacing:0;}
        textarea {font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;}
        </style>
        """)


def _header() -> None:
    _, center, _ = st.columns([2.2, 1, 2.2])
    with center:
        assert _LOGO_PATH.exists()
        st.image(str(_LOGO_PATH), use_container_width=True)
        st.html("<h1 style='text-align:center'>inversql</h1>")


def _load_tables(csvs: list[up_man.UploadedFile]) -> list[SourceRelation]:
    "Load the tables as our representation."

    if not csvs:
        return []

    tables: list[SourceRelation] = []
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
            tables.append(SourceRelation(name, pd.read_csv(file)))
        except Exception as exc:
            st.error(f"Could not read {file.name}: {exc}")

    st.session_state.tables = tables
    return tables


def _render_sql(*sql: str) -> None:
    main, *candidates = sql
    st.subheader("Generated SQL")
    st.code(main, "sql")
    st.subheader("Candidate SQL")
    for i, candidate in enumerate(candidates, 1):
        with st.expander(f"Candidate {i}", expanded=i == 1):
            st.html(
                f"""
                <div style="
                    opacity: 0.55;
                    font-family: monospace;
                    background: #f6f8fa;
                    padding: 10px;
                    border-radius: 6px;
                    white-space: pre;
                    overflow-x: auto;
                ">
                {html.escape(candidate)}
                </div>
                """,
            )


if __name__ == "__main__":
    pipeline = Pipeline()

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
                f"{t.name}: {len(t.data())}x{len(t.data().columns)}" for t in tables
            )
        )

    annotation_col, sql_col = st.columns([1.65, 1], gap="large")
    sqls = pipeline(*st.session_state.tables)

    with sql_col, st.container(border=True):
        _render_sql(*sqls)
