# Copyright (c) The InverSQL Authors - All Rights Reserved

"Relations."

import abc
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn import tree

from inversql.trees import sklearn_binary_tree_to_nodes

__all__ = [
    "ColRef",
    "ColLabel",
    "Relation",
    "SourceRelation",
    "JoinRelation",
    "SkLearnTreeRelation",
    "SelectRelation",
    "NumericDF",
]

type _SupportedJoinTypes = typing.Literal["left", "right", "outer", "inner", "cross"]
type _JoinKey = str | tuple[str, ...] | None

_ROW_MARKER = "__inversql_selected__"


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
    def marker(cls, table: str) -> typing.Self:
        return cls(table=table, column=_ROW_MARKER)


@dcls.dataclass(frozen=True)
class ColLabel(ColRef):
    """
    Column ref + label.
    """

    label: bool
    "Whether to select a column or not."

    def __repr__(self):
        return super().__repr__() + f" = {self.label}"

    def ref(self) -> ColRef:
        return ColRef(self.table, self.column)


class Relation(abc.ABC):
    """
    `Relation` is a relational construct with tracking info.
    """

    def data(self) -> pd.DataFrame:
        """
        Get the dataframe represented by the current `Relation`.
        Strip the internal tracking information.
        """

        df = self._to_pandas()
        df = df[[str(c.ref()) for c in self.columns]]

        return df.reset_index(drop=True)

    def row_labels(self) -> npt.NDArray[np.bool_]:
        """
        Get the labels of the tables, marked in the source with `row_idxs`.
        """

        df = self._to_pandas()
        markers = [f"{name}.{_ROW_MARKER}" for name in self.sources.keys()]

        # If any of the markers is true, it is included.

        y = np.zeros(len(df), dtype=bool)
        for marker in markers:
            assert marker in df.columns, {"marker": marker, "cols": df.columns}

            y |= df[marker].to_numpy()

        return y

    @property
    @abc.abstractmethod
    def columns(self) -> set[ColLabel]:
        """
        Get the labels (to select or not) of the columns.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def _to_pandas(self) -> pd.DataFrame:
        raise NotImplementedError

    @functools.cached_property
    def sources(self) -> dict[str, SourceRelation]:
        "Get all the sources that this relation has."

        return {source.name: source for source in self._sources()}

    @abc.abstractmethod
    def _sources(self) -> cabc.Generator[SourceRelation]:
        raise NotImplementedError


@typing.final
class SourceRelation(Relation):
    "The relation backed by an external table."

    def __init__(self, name: str, df: pd.DataFrame) -> None:
        self._name: str = name
        "The name of the dataframe."

        self._df: pd.DataFrame = self._qualify_df(df)
        "The dataframe to operate on."

        self._cells: set[tuple[int, int]] = set()
        "Set of selected cells."

    @property
    def name(self) -> str:
        return self._name

    @typing.override
    def _to_pandas(self) -> pd.DataFrame:
        df = self._df.copy()

        # Mark the row indices.
        marker = str(ColRef(self.name, _ROW_MARKER))

        df[marker] = False
        df.loc[self._row_idxs(), marker] = True
        assert df.notna().all().all(), "DataFrame contains NaN values!"
        return df

    @typing.override
    def _sources(self):
        yield self

    @property
    @typing.override
    def columns(self) -> set[ColLabel]:
        prefix = self.name + "."

        def drop_name_prefix(col_name: str):
            assert col_name.startswith(prefix)
            return col_name.removeprefix(prefix)

        selected = {col for _, col in self._cells}
        df_cols = [drop_name_prefix(c) for c in self._df.columns]
        return {
            ColLabel(table=self.name, column=col, label=idx in selected)
            for idx, col in enumerate(df_cols)
        }

    def _row_idxs(self) -> list[int]:
        rows = {row for row, _ in self._cells}
        return sorted(rows)

    def toggle(self, cell: tuple[int, int]) -> None:
        "Toggle the `cell` selection."

        if cell in self._cells:
            self._cells.remove(cell)
        else:
            self._cells.add(cell)

    def _qualify_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [f"{self.name}.{col}" for col in df.columns]
        return df


@typing.final
class JoinRelation(Relation):
    "The join based relations."

    def __init__(
        self,
        left: Relation,
        right: Relation,
        how: _SupportedJoinTypes,
        left_on: _JoinKey = None,
        right_on: _JoinKey = None,
    ) -> None:
        self._left = left
        self._right = right
        self._how = how
        self._left_on = left_on
        self._right_on = right_on

    def _to_pandas(self) -> pd.DataFrame:
        left = self._left._to_pandas()
        right = self._right._to_pandas()

        left_on = _join_key(self.left.columns, self.left_on)
        right_on = _join_key(self.right.columns, self.right_on)

        return pd.merge(left, right, how=self.how, left_on=left_on, right_on=right_on)

    @property
    def columns(self) -> set[ColLabel]:
        col_left = self._left.columns
        col_right = self._right.columns

        if not col_left.isdisjoint(col_right):
            raise ValueError(
                "Complex join types not supported yet. "
                f"{col_left=} and {col_right=} should be disjoint."
            )

        result = {*self._left.columns, *self._right.columns}
        assert len(result) == len(col_left) + len(col_right)
        return result

    def _sources(self):
        yield from self._left._sources()
        yield from self._right._sources()

    @property
    def how(self) -> _SupportedJoinTypes:
        return self._how

    @property
    def left(self) -> Relation:
        return self._left

    @property
    def right(self) -> Relation:
        return self._right

    @property
    def left_on(self) -> _JoinKey:
        return self._left_on

    @property
    def right_on(self) -> _JoinKey:
        return self._right_on


def _join_key(cols: set[ColLabel], target: _JoinKey) -> _JoinKey:
    def first(key: str) -> str:
        return next(str(col.ref()) for col in cols if col.column == key)

    match target:
        case None:
            return None
        case str():
            return next(str(col.ref()) for col in cols if col.column == target)
        case tuple():
            return tuple(first(k) for k in target)

    raise RuntimeError("Not reachable.")


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
        "Get the numeric version of the dataframe (pre computed)."
        return self._numeric

    @typing.no_type_check
    def revert(self, num_df: pd.DataFrame) -> pd.DataFrame:
        "Revert the numeric dataframe to the ones with cateogies."

        for column, code_category in self._mappings.items():
            if column not in num_df:
                continue

            num_df[column] = num_df[column].apply(lambda x: code_category[x])
            num_df[column] = num_df[column].astype("category")
        return num_df

    def original(self) -> pd.DataFrame:
        "Get the original dataframe that constructed this `NumericDF`."
        return self.revert(self.numeric())

    @property
    def columns(self) -> pd.Index[str]:
        "Pass through the columns of underlying df."
        return self._numeric.columns


@typing.final
class SkLearnTreeRelation(Relation):
    def __init__(self, input: Relation, clf: tree.DecisionTreeClassifier) -> None:
        self._input = input
        self._clf = clf
        self._save_node_expr()

    @property
    def columns(self) -> set[ColLabel]:
        return self._input.columns

    @typing.no_type_check
    def _to_pandas(self) -> pd.DataFrame:
        """
        Map to original and then do the conversion.
        """

        return self._numeric_df.original().iloc[self._predicted]

    def _save_node_expr(self) -> None:
        num_df = self._numeric_df.numeric()

        self.clf.fit(np.asarray(num_df), self.row_labels())
        self._node = sklearn_binary_tree_to_nodes(self.clf)
        self._predicted = self.clf.predict(np.asarray(num_df))
        assert np.all(self._predicted == self.row_labels())

    @property
    def input(self) -> Relation:
        return self._input

    @property
    def clf(self) -> tree.DecisionTreeClassifier:
        return self._clf

    @property
    def _numeric_df(self) -> NumericDF:
        return NumericDF(self.input.data())

    def _sources(self):
        yield from self.input._sources()


@typing.final
class SelectRelation(Relation):
    def __init__(self, input: Relation, *cols: str):
        self._input = input
        self._cols = cols

    def _to_pandas(self) -> pd.DataFrame:
        cols = [str(c.ref()) for c in self.columns]
        return self.input._to_pandas()[cols]

    def _sources(self):
        yield from self.input._sources()

    @property
    def columns(self) -> set[ColLabel]:
        name_to_label = {col.column: col for col in self.input.columns}
        return {name_to_label[c] for c in self.cols}

    @property
    def input(self) -> Relation:
        return self._input

    @property
    def cols(self) -> tuple[str, ...]:
        return self._cols
