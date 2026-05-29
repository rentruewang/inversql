# Copyright (c) The InverSQL Authors - All Rights Reserved

import pandas as pd


def test_streamlit_app_imports_without_running_pipeline():
    import inversql.app

    assert callable(inversql.app.main)


def test_selected_cells_parses_streamlit_event_shape():
    from inversql.app import _selected_cells
    from inversql.pipeline import CellRef

    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "amount": [45.5, 120.0, 15.25, 89.99],
            "region": ["west", "east", "west", "east"],
        }
    )
    event = {
        "selection": {
            "cells": [
                (1, "name"),
                [2, "amount"],
                {"row": 3, "column": "region"},
            ]
        }
    }

    assert _selected_cells(event, df) == (
        CellRef(row=1, column="name"),
        CellRef(row=2, column="amount"),
        CellRef(row=3, column="region"),
    )


def test_selected_cells_expands_rows_and_columns():
    from inversql.app import _selected_cells
    from inversql.pipeline import CellRef

    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "amount": [45.5, 120.0, 15.25],
            "region": ["west", "east", "west"],
        }
    )
    event = {
        "selection": {
            "rows": [1],
            "columns": ["amount"],
            "cells": [(1, "amount"), (2, "region")],
        }
    }

    assert _selected_cells(event, df) == (
        CellRef(row=1, column="name"),
        CellRef(row=1, column="amount"),
        CellRef(row=1, column="region"),
        CellRef(row=0, column="amount"),
        CellRef(row=2, column="amount"),
        CellRef(row=2, column="region"),
    )
