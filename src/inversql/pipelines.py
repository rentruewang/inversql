# Copyright (c) The InverSQL Authors - All Rights Reserved

import dataclasses as dcls

import numpy as np
import pandas as pd
from sklearn import tree

from inversql.exprs import simplify_expr
from inversql.joins import (
    Joiner,
    JoinerList,
    JoinResult,
    cross_joiner,
    shared_col_name_joiner,
)
from inversql.refs import AnnotatedDF, ColRef
from inversql.trees import sklearn_binary_tree_to_nodes

__all__ = ["default_joiners", "Pipeline"]


def default_joiners():
    """
    The default list of joiners. They can be found in `inversql.joins`.
    """

    return JoinerList(
        [
            cross_joiner,
            shared_col_name_joiner,
        ]
    )


@dcls.dataclass
class Pipeline:
    """
    The pipeline of `inversql`'s backend.
    It generates SQL for frontend to display.
    """

    joiner: Joiner = dcls.field(default_factory=default_joiners)
    "The joiner in the pipeline. Default to the `default_joiners` function."

    def __call__(self, *annotated: AnnotatedDF):
        tables = {a.name: a.dataframe() for a in annotated}

        for result in self.joiner(tables):
            clf = tree.DecisionTreeClassifier()

            train_x, train_y = train_test_pair(result, tables)
            clf.fit(train_x, train_y)

            nodes = sklearn_binary_tree_to_nodes(clf)
            truth_exprs = simplify_expr(nodes.truth_exprs())


def train_test_pair(
    result: JoinResult, tables: dict[str, pd.DataFrame]
) -> tuple[np.ndarray, np.ndarray]:
    selected = [ColRef.selected_marker(t) for t in tables]
    x = result.df.to_numpy()

    y = np.zeros(len(result.df)).astype(bool)
    for sel in selected:
        assert str(sel) in result.df.columns, result.df.columns
        marked = result.df[str(sel)].to_numpy()
        breakpoint()
        y |= marked

    return x, y
