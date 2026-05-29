# Copyright (c) The InverSQL Authors - All Rights Reserved

import pandas as pd

from inversql.pipeline import (
    CellRef,
    CellSelection,
    TableSpec,
    TrainingSelection,
    run_pipeline,
)


def _tables() -> tuple[TableSpec, TableSpec]:
    users = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "region": ["west", "east", "west", "east"],
        }
    )
    orders = pd.DataFrame(
        {
            "user_id": [1, 2, 2, 4],
            "order_id": [1001, 1002, 1003, 1004],
            "amount": [45.5, 120.0, 15.25, 89.99],
        }
    )

    return TableSpec("users", users), TableSpec("orders", orders)


def test_pipeline_joins_without_training_selection():
    left, right = _tables()

    result = run_pipeline(left, right)

    assert result.join_summary is not None
    assert result.join_summary.how == "left"
    assert result.join_summary.on == ("user_id",)
    assert len(result.joined_df) == 5
    assert result.sql is None
    assert "Select at least one must-have cell" in " ".join(result.diagnostics)


def test_pipeline_generates_sql_from_positive_rows():
    left, right = _tables()

    result = run_pipeline(
        left,
        right,
        training=TrainingSelection(positive_row_indices=(1,)),
    )

    assert result.sql is not None
    assert 'FROM "users" LEFT JOIN "orders" ON' in result.sql
    assert '"users"."user_id" = "orders"."user_id"' in result.sql
    assert "WHERE" in result.sql
    assert "1 positive, 4 negative" in " ".join(result.diagnostics)


def test_unselected_rows_are_negative_samples():
    left, right = _tables()

    result = run_pipeline(
        left,
        right,
        training=TrainingSelection(positive_row_indices=(0, 2)),
    )

    assert result.sql is not None
    assert "2 positive, 3 negative" in " ".join(result.diagnostics)


def test_pipeline_requires_at_least_one_negative_sample():
    left, right = _tables()

    result = run_pipeline(
        left,
        right,
        training=TrainingSelection(positive_row_indices=(0, 1, 2, 3, 4)),
    )

    assert result.sql is None
    assert "Leave at least one row unselected" in " ".join(result.diagnostics)


def test_must_have_cells_add_value_constraints_to_sql():
    left, right = _tables()

    result = run_pipeline(
        left,
        right,
        cells=CellSelection(must_have=(CellRef(row=1, column="region"),)),
    )

    assert result.sql is not None
    assert '"users"."region" = \'east\'' in result.sql
    assert "1 positive, 4 negative" in " ".join(result.diagnostics)
    assert "Cell constraints: 1 must-have, 0 must-not-have" in " ".join(
        result.diagnostics
    )


def test_must_have_cell_rows_are_positive_by_default():
    left, right = _tables()

    result = run_pipeline(
        left,
        right,
        cells=CellSelection(
            must_have=(
                CellRef(row=1, column="region"),
                CellRef(row=2, column="amount"),
            )
        ),
    )

    assert result.sql is not None
    assert "2 positive, 3 negative" in " ".join(result.diagnostics)


def test_must_not_have_cells_add_value_exclusions_to_sql():
    left, right = _tables()

    result = run_pipeline(
        left,
        right,
        cells=CellSelection(must_not_have=((0, "name"),)),
        training=TrainingSelection(positive_row_indices=(1,)),
    )

    assert result.sql is not None
    assert '("users"."name" <> \'Alice\' OR "users"."name" IS NULL)' in result.sql
    assert "Cell constraints: 0 must-have, 1 must-not-have" in " ".join(
        result.diagnostics
    )


def test_must_not_have_only_generates_constraint_sql():
    left, right = _tables()

    result = run_pipeline(
        left,
        right,
        cells=CellSelection(must_not_have=((0, "name"),)),
    )

    assert result.sql is not None
    assert '("users"."name" <> \'Alice\' OR "users"."name" IS NULL)' in result.sql
    assert "using cell constraints only" in " ".join(result.diagnostics)


def test_all_positive_rows_generate_constraint_only_sql():
    left, right = _tables()

    result = run_pipeline(
        left,
        right,
        cells=CellSelection(
            must_have=(
                CellRef(row=0, column="region"),
                CellRef(row=1, column="region"),
                CellRef(row=2, column="region"),
                CellRef(row=3, column="region"),
                CellRef(row=4, column="region"),
            )
        ),
    )

    assert result.sql is not None
    assert '"users"."region" IN' in result.sql
    assert "All joined rows are positive" in " ".join(result.diagnostics)


def test_must_have_and_must_not_have_conflicts_prefer_must_have():
    left, right = _tables()

    result = run_pipeline(
        left,
        right,
        cells=CellSelection(
            must_have=(CellRef(row=1, column="region"),),
            must_not_have=(CellRef(row=4, column="region"),),
        ),
        training=TrainingSelection(positive_row_indices=(1,)),
    )

    assert result.sql is not None
    assert '"users"."region" = \'east\'' in result.sql
    assert '"users"."region" <> \'east\'' not in result.sql
    assert "Ignored conflicting must-not-have values" in " ".join(result.diagnostics)


def test_pipeline_reports_when_no_valid_join_exists():
    left = TableSpec("left", pd.DataFrame({"left_id": [1, 2]}))
    right = TableSpec("right", pd.DataFrame({"right_id": [1, 2]}))

    result = run_pipeline(left, right)

    assert result.join_summary is None
    assert result.joined_df.empty
    assert result.sql is None
    assert "No valid join candidate" in " ".join(result.diagnostics)
