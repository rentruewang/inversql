# Copyright (c) The InverSQL Authors - All Rights Reserved

import pandas as pd

import app


def test_selection_mode_for_table_targets():
    assert app._selection_mode_for("Cell") == "multi-cell"
    assert app._selection_mode_for("Row") == "multi-row"
    assert app._selection_mode_for("Column") == "multi-column"
    assert app._selection_mode_for("unknown") == "multi-cell"


def test_selection_state_from_row_event():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    event = {"selection": {"rows": [1, 1, 7, "bad"]}}

    selected = app._selection_state_from_event(event, "Row", df)

    assert selected.rows == (1,)
    assert selected.columns == ()
    assert selected.cells == ()


def test_selection_state_from_column_event():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    event = {"selection": {"columns": ["b", "b", "missing", 0]}}

    selected = app._selection_state_from_event(event, "Column", df)

    assert selected.rows == ()
    assert selected.columns == ("b", "a")
    assert selected.cells == ()


def test_selection_state_from_cell_event():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    event = {
        "selection": {
            "cells": [
                (0, "a"),
                {"row": 1, "column": "b"},
                {"row": 0, "column": 1},
                (99, "a"),
                (0, "missing"),
                "bad",
            ]
        }
    }

    selected = app._selection_state_from_event(event, "Cell", df)

    assert selected.rows == ()
    assert selected.columns == ()
    assert selected.cells == (
        app.CellRef(0, "a"),
        app.CellRef(1, "b"),
        app.CellRef(0, "b"),
    )


def test_merge_selection_states_keeps_order_and_dedupes():
    existing = app.SelectionState(
        rows=(1,),
        columns=("a",),
        cells=(app.CellRef(0, "a"),),
    )
    selected = app.SelectionState(
        rows=(1, 2),
        columns=("b", "a"),
        cells=(app.CellRef(0, "a"), app.CellRef(1, "b")),
    )

    merged = app._merge_selection_states(existing, selected)

    assert merged.rows == (1, 2)
    assert merged.columns == ("a", "b")
    assert merged.cells == (app.CellRef(0, "a"), app.CellRef(1, "b"))


def test_subtract_selection_states_removes_selected_values():
    existing = app.SelectionState(
        rows=(1, 2, 3),
        columns=("a", "b"),
        cells=(app.CellRef(0, "a"), app.CellRef(1, "b")),
    )
    selected = app.SelectionState(
        rows=(2,),
        columns=("b",),
        cells=(app.CellRef(0, "a"),),
    )

    updated = app._subtract_selection_states(existing, selected)

    assert updated.rows == (1, 3)
    assert updated.columns == ("a",)
    assert updated.cells == (app.CellRef(1, "b"),)


def test_selection_totals_dedupes_across_modes():
    totals = app._selection_totals(
        [
            app.SelectionState(rows=(1, 2), columns=("a",)),
            app.SelectionState(rows=(2,), columns=("a", "b")),
            app.SelectionState(cells=(app.CellRef(0, "a"), app.CellRef(0, "a"))),
        ]
    )

    assert totals == app.SelectionTotals(rows=2, columns=2, cells=1)


def test_highlight_selection_marks_rows_columns_and_cells():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    state = app.SelectionState(
        rows=(0,),
        columns=("b",),
        cells=(app.CellRef(1, "a"),),
    )

    styles = app._highlight_selection(df, state, "must_have")

    assert styles.loc[0, "a"] == app._BUCKET_STYLES["must_have"]
    assert styles.loc[0, "b"] == app._BUCKET_STYLES["must_have"]
    assert styles.loc[1, "a"] == app._BUCKET_STYLES["must_have"]
    assert styles.loc[1, "b"] == app._BUCKET_STYLES["must_have"]


def test_mock_sql_is_deterministic_and_uses_selection_counts():
    tables = [
        app.TableSpec("left table", pd.DataFrame({"id": [1]})),
        app.TableSpec('right "table"', pd.DataFrame({"id": [1]})),
    ]
    must_have = app.SelectionTotals(rows=1, columns=2, cells=3)
    must_not_have = app.SelectionTotals(rows=4, columns=5, cells=6)

    first = app._mock_sql(tables, must_have, must_not_have)
    second = app._mock_sql(tables, must_have, must_not_have)

    assert first == second
    assert '"left table"' in first.main
    assert '"right ""table"""' in first.main
    assert "must-have 1 rows / 2 columns / 3 cells" in first.main
    assert "must-not-have 4 rows / 5 columns / 6 cells" in first.main
    assert len(first.candidates) == 3
