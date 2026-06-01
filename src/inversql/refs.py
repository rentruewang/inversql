# Copyright (c) The InverSQL Authors - All Rights Reserved

"Merging with tracking info."

import dataclasses as dcls
import typing

import pandas as pd

__all__ = ["pd_merge_with_suffix", "ColRef", "NumericDF", "AnnotatedDF"]

type _MergeHow = typing.Literal[
    "left", "right", "outer", "inner", "cross", "left_anti", "right_anti"
]

_ROW_SELECTED_MARKER = "__inversql_selected__"


class NumericDF:
    """
    The dataframe that converts all the fields to numeric data.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        df = df.copy()

        non_numeric_cols = df.select_dtypes(exclude=["number"]).columns

        mappings: dict[str, dict[int, pd.Categorical]] = {}

        # Convert the non numeric with categorical.
        for col in non_numeric_cols:
            df[col] = df[col].astype("category")

            mappings[col] = {
                code: category for code, category in enumerate(df[col].cat.categories)
            }

            df[col] = df[col].cat.codes

        self._numeric = df
        "The dataframe that is all numeric."

        self._mappings = mappings
        "Mapping of column -> codes -> category."

    def numeric(self) -> pd.DataFrame:
        return self._numeric

    @typing.no_type_check
    def revert(self, num_df: pd.DataFrame) -> pd.DataFrame:
        for column, code_category in self._mappings.items():
            if column not in num_df:
                continue

            num_df[column] = num_df[column].apply(lambda x: code_category[x])
        return num_df


@dcls.dataclass(frozen=True)
class AnnotatedDF:
    """
    The dataframes that are annotated.
    """

    name: str
    "The name of the dataframe."

    df: NumericDF
    "The dataframe to operate on."

    col_names: set[str] = dcls.field(default_factory=set)
    "Set of selected columns."

    row_idxs: set[int] = dcls.field(default_factory=set)
    "Set of selected rows."

    def dataframe(self) -> pd.DataFrame:
        df = self.df.numeric().copy()

        # Mark the row indices.
        df[_ROW_SELECTED_MARKER] = False
        df.loc[sorted(self.row_idxs), _ROW_SELECTED_MARKER] = True
        assert df.notna().all().all(), "DataFrame contains NaN values!"

        breakpoint()
        return df


@dcls.dataclass(frozen=True)
class ColRef:
    """
    The reference for a column, with some utilities.
    """

    table: str
    "Table that the column belongs to."

    column: str
    "The name of the column in that table."

    def __repr__(self):
        return f"{self.table}.{self.column}"

    @classmethod
    def attribute(cls, tables: dict[str, pd.DataFrame], col: str) -> typing.Self:
        "Find the column `col` in the original `tables`, with `ColRef`."

        if result := cls.parse(col):
            assert result.table in tables
            return result

        else:
            for key, df in tables.items():
                if col in df.columns:
                    return cls(table=key, column=col)

            raise RuntimeError(f"No {col=} found in tables.")

    @classmethod
    def parse(cls, name: str) -> typing.Self | None:
        try:
            *table, column = name.split(".")
        except ValueError:
            return None
        else:
            return cls(table=".".join(table), column=column)

    @classmethod
    def selected_marker(cls, table: str) -> typing.Self:
        return cls(table=table, column=_ROW_SELECTED_MARKER)


def pd_merge_with_suffix(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: _MergeHow,
    df_id_name: dict[int, str],
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
):
    """
    This handles merging in `pd`, with good suffix for parsing.

    Args:
        df_id_name: The mapping from id(dataframe) to their names. Used for suffixes.
        **kwargs: Same as `pd.merge`.
    """

    assert id(left) in df_id_name
    assert id(right) in df_id_name

    ls, rs = (df_id_name[id(t)] for t in [left, right])

    # If there is a conflict, the new column would be `name.table`.
    result = left.merge(
        right,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        suffixes=["." + ls, "." + rs],
    )

    map_left = {f"{col}.{ls}": f"{ls}.{col}" for col in left.columns}
    map_right = {f"{col}.{rs}": f"{rs}.{col}" for col in right.columns}
    result.rename(columns={**map_left, **map_right}, inplace=True)
    return result
