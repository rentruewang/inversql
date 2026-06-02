# Copyright (c) The InverSQL Authors - All Rights Reserved

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

    return tables


if __name__ == "__main__":
    st.set_page_config("InverSQL", layout="wide", initial_sidebar_state="collapsed")

    _css()
    _header()

    with st.container(border=True):
        st.subheader("Table Upload")
        uploaded = st.file_uploader("CSV files", "csv", accept_multiple_files=True)
        tables = _load_tables(uploaded)
        if not tables:
            st.info("Using demo tables. Upload CSV files to annotate your own data.")
            tables = _render_tables()
        st.caption(
            " · ".join(f"{t.name}: {len(t.df)}x{len(t.df.columns)}" for t in tables)
        )

    annotation_col, sql_col = st.columns([1.65, 1], gap="large")
    with annotation_col, st.container(border=True):
        st.subheader("Table Annotation")
        target = st.segmented_control("Selection type", _TARGETS, default="Cell")
        target = str(target or "Cell")
        must_have, must_not_have = _annotation_workbench(tables, target)

    with sql_col, st.container(border=True):
        _render_sql(_mock_sql(tables, must_have, must_not_have))
