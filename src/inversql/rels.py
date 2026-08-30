# Copyright (c) InverSQL Authors - All Rights Reserved

"Relations."

import abc
import dataclasses as dcls
import functools
import math
import operator
import typing
from collections import abc as cabc

import numpy as np
import pandas as pd
import sqlglot
from numpy import typing as npt
from sklearn import tree
from sqlglot import exp as sqlg_exp
from sqlglot import expressions as sqlg_expr

from inversql.exprs import AndExpr, CmpExpr, CmpOp, Expr, OrExpr
from inversql.trees import sklearn_binary_tree_to_nodes

__all__ = [
    "ColRef",
    "ColLabel",
    "CellLoc",
    "Relation",
    "SourceRelation",
    "JoinRelation",
    "SkLearnTreeRelation",
    "SelectRelation",
    "NumericDF",
]

_SupportedJoinTypes: typing.TypeAlias = typing.Literal[
    "left", "right", "inner", "cross"
]
_JoinKey: typing.TypeAlias = str | tuple[str, ...] | None

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

    def __str__(self):
        return str(self.to_sqlglot())

    @abc.abstractmethod
    def to_sqlglot(self) -> sqlg_expr.Select:
        """
        Return sqlglot objects.
        """

        raise NotImplementedError

    def data(self) -> pd.DataFrame:
        """
        Get the dataframe represented by the current `Relation`.
        Strip the internal tracking information.
        """

        df = self.to_pandas()
        df = df[sorted(str(c.ref()) for c in self.columns)]

        return df.reset_index(drop=True)

    def row_labels(self) -> npt.NDArray[np.bool_]:
        """
        Get the labels of the tables, marked in the source with `row_idxs`.
        """

        df = self.to_pandas()
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
    def to_pandas(self) -> pd.DataFrame:
        """
        The underlying dataframe passed. Contains tracking info like markers.
        """
        raise NotImplementedError

    @functools.cached_property
    def sources(self) -> dict[str, SourceRelation]:
        "Get all the sources that this relation has."

        return {source.name: source for source in self._sources()}

    @abc.abstractmethod
    def _sources(self) -> cabc.Generator[SourceRelation]:
        raise NotImplementedError


class CellLoc(typing.NamedTuple):
    "The index of a cell's location."

    row_idx: int
    "The row index."
    col_idx: int
    "The column index."


@typing.final
class SourceRelation(Relation):
    "The relation backed by an external table."

    def __init__(
        self, name: str, df: pd.DataFrame, cells: cabc.Iterable[CellLoc] = ()
    ) -> None:
        self._name: str = name
        "The name of the dataframe."

        self._df: pd.DataFrame = self._qualify_df(df)
        "The dataframe to operate on."

        self._cells: set[CellLoc] = set(cells)
        "Set of selected cells."

    def to_sqlglot(self) -> sqlg_expr.Select:
        return sqlglot.select("*").from_(self.name)

    @property
    def name(self) -> str:
        return self._name

    @typing.override
    def to_pandas(self) -> pd.DataFrame:
        df = self._df.copy()

        # Mark the row indices.
        marker = str(ColRef(self.name, _ROW_MARKER))

        df[marker] = False
        df.loc[self._row_idxs(), marker] = True
        assert df.notna().all().all(), "DataFrame contains NaN values!"

        assert marker in df.columns
        return df

    @typing.override
    def _sources(self):
        yield self

    def columns_all_selected(self) -> bool:
        df = self.data()
        rows = {row for row, _ in self.cells}
        return len(rows) == len(df)

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

    @property
    def cells(self) -> frozenset[CellLoc]:
        "A frozen view of the current cells."
        return frozenset(self._cells)

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

    def to_sqlglot(self) -> sqlg_expr.Select:
        left_sql = self.left.to_sqlglot()
        right_sql = self.right.to_sqlglot()

        # Cross does not need any keys.
        if self.how == "cross":
            return sqlglot.select("*").from_(left_sql).from_(right_sql)

        else:
            crit = self._gen_crit()
            return left_sql.join(right_sql, join_type=self.how.upper(), on=crit)

    def _gen_crit(self):
        generator = _CriterionGenerator()
        return generator(self)

    def to_pandas(self) -> pd.DataFrame:
        left = self._left.to_pandas()
        right = self._right.to_pandas()

        return pd.merge(
            left, right, how=self.how, left_on=self.left_on, right_on=self.right_on
        )

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
        return _join_key(self.left.columns, self._left_on)

    @property
    def right_on(self) -> _JoinKey:
        return _join_key(self.right.columns, self._right_on)


@dcls.dataclass(frozen=True)
class _CriterionGenerator:
    def __call__(self, rel: JoinRelation):
        # Handle the `str` case.
        if isinstance(rel.left_on, str):
            assert isinstance(rel.right_on, str)
            return self._criterion(rel.left_on, rel.right_on)

        # Handle the case where there are multiple join keys.
        elif _is_tuple_str(rel.left_on):
            assert _is_tuple_str(rel.right_on)
            assert len(rel.left_on) == len(rel.right_on)
            return functools.reduce(
                operator.and_,
                [self._criterion(l, r) for l, r in zip(rel.left_on, rel.right_on)],
            )

        else:
            raise ValueError(
                f"{rel.left_on} or {rel.right_on} invalid for join type {rel.how}."
            )

    def _criterion(self, left: str, right: str) -> sqlg_expr.Expr:
        left_ref = ColRef(*left.split("."))
        right_ref = ColRef(*right.split("."))

        left_expr = sqlg_exp.column(left_ref.column, table=left_ref.table)
        right_expr = sqlg_exp.column(right_ref.column, table=right_ref.table)
        return sqlg_exp.EQ(this=left_expr, expression=right_expr)


def _is_tuple_str(obj) -> typing.TypeIs[tuple[str, ...]]:
    return isinstance(obj, tuple) and all(isinstance(elem, str) for elem in obj)


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
        mappings: dict[str, list[str]] = {}

        # Convert the non numeric with categorical (only string supported).
        for col in non_numeric_cols:
            self._convert_str_columns(df, col, mappings)

        self._numeric = df
        "The dataframe that is all numeric."

        self._rev_mappings = mappings
        "Mapping of column -> list of categories (only strings)."

    def _convert_str_columns(
        self, df: pd.DataFrame, col: str, mappings: dict[str, list[str]]
    ) -> None:
        string_values: set[str] = set(df[col])

        if not all(isinstance(val, str) for val in string_values):
            raise NotImplementedError(
                "Only string columns or numeric columns are now supported."
            )

        sorted_vaues = sorted(string_values)

        string_idx = {string: idx for idx, string in enumerate(sorted_vaues)}
        indices = [string_idx[string] for string in df[col]]

        df[col] = indices
        mappings[col] = sorted_vaues

    def numeric(self) -> pd.DataFrame:
        "Get the numeric version of the dataframe (pre computed)."
        return self._numeric

    @typing.no_type_check
    def revert(self, num_df: pd.DataFrame) -> pd.DataFrame:
        "Revert the numeric dataframe to the ones with cateogies."

        for column, code_category in self._rev_mappings.items():
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

    def expr_to_sqlglot(self, expr: Expr, terms: list[str]) -> sqlg_expr.Select:
        # Visitor for AND / OR.
        def _and_or_to_sqlglot(expr: AndExpr | OrExpr):
            children = [self.expr_to_sqlglot(expr, terms) for expr in expr.exprs]
            return functools.reduce(type(expr).CLS_BIN_OP, children)

        # Visitor for CMP.
        def _cmp_to_sqlglot(expr: CmpExpr):
            cmp_op = expr.cmp
            threshold: int | float | str = expr.threshold

            if (feature := terms[expr.feat_idx]) in self._rev_mappings:
                feature_map = self._rev_mappings[feature]

                # Since this is categories, find the closest.
                # E.g. >= 1.5, find >= 2.

                match expr.cmp:
                    case CmpOp.EQ | CmpOp.NE:
                        cmp_op = expr.cmp
                        threshold = feature_map[int(expr.threshold)]
                    case CmpOp.GE | CmpOp.GT:
                        cmp_op = CmpOp.GE
                        threshold = feature_map[math.ceil(expr.threshold)]
                    case CmpOp.LE | CmpOp.LT:
                        cmp_op = CmpOp.LE
                        threshold = feature_map[math.floor(expr.threshold)]

            return cmp_op.op(sqlg_exp.column(terms[expr.feat_idx]), threshold)

        match expr:
            case AndExpr() | OrExpr():
                return _and_or_to_sqlglot(expr)
            case CmpExpr():
                return _cmp_to_sqlglot(expr)
            case _:
                raise TypeError(f"Not supported {type(expr)=}.")


@typing.final
class SkLearnTreeRelation(Relation):
    def __init__(self, input: Relation, clf: tree.DecisionTreeClassifier) -> None:
        self._input = input
        self._clf = clf
        self._save_predicted()

    def to_sqlglot(self) -> sqlg_expr.Select:
        df_cols = self._numeric_df.columns

        cols_labels_ordered = {str(col.ref()): col for col in self.columns}
        assert len(cols_labels_ordered) == len(self.columns)

        ordered_labels = [cols_labels_ordered[c] for c in df_cols]

        terms = [str(cl.ref()) for cl in ordered_labels]
        expr = self._tree_node.truth_exprs()
        # expr = simplify_expr(expr)
        criterion = self._numeric_df.expr_to_sqlglot(expr, terms)
        return self.input.to_sqlglot().where(criterion)

    @property
    def columns(self) -> set[ColLabel]:
        return self._input.columns

    @typing.no_type_check
    def to_pandas(self) -> pd.DataFrame:
        """
        Map to original and then do the conversion.
        """

        return self.input.to_pandas()

    def _save_predicted(self) -> None:
        num_df = self._numeric_df.numeric()

        self.clf.fit(np.asarray(num_df), self.row_labels())
        self._tree_node = sklearn_binary_tree_to_nodes(self.clf)
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

    def to_pandas(self) -> pd.DataFrame:
        cols = [str(c.ref()) for c in self.columns]
        return self.input.to_pandas()[cols]

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

    def to_sqlglot(self):
        sql = self.input.to_sqlglot()
        return sql.select(*self.cols)
