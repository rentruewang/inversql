# Copyright (c) The InverSQL Authors - All Rights Reserved

import dataclasses as dcls

import pandas as pd
from sklearn import tree

from inversql.joins import Joiner, JoinerList, cross_joiner, shared_col_name_joiner

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


_SELECTED_MARKER = "__inversql_selected__"


@dcls.dataclass(frozen=True)
class AnnotatedDF:
    """
    The dataframes that are annotated.
    """

    name: str
    "The name of the dataframe."

    df: pd.DataFrame
    "The dataframe to operate on."

    col_names: set[str] = dcls.field(default_factory=set)
    "Set of selected columns."

    row_idxs: set[int] = dcls.field(default_factory=set)
    "Set of selected rows."

    def tagged_df(self, idx: int):
        df = self.df.copy()
        df.loc[[]]


@dcls.dataclass
class Pipeline:
    """
    The pipeline of `inversql`'s backend.
    It generates SQL for frontend to display.
    """

    joiner: Joiner = dcls.field(default_factory=default_joiners)
    "The joiner in the pipeline. Default to the `default_joiners` function."

    def __call__(self, *annotated: AnnotatedDF):
        df_dict = {a.name: a.df for a in annotated}
        results = self.joiner(df_dict)

        clf = tree.DecisionTreeClassifier()
