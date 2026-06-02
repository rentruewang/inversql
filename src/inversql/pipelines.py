# Copyright (c) The InverSQL Authors - All Rights Reserved

import dataclasses as dcls

from sklearn import tree

from inversql.joins import (
    Joiner,
    JoinerList,
    cross_joiner,
    shared_col_name_joiner,
)
from inversql.rels import SkLearnTreeRelation, SourceRelation

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

    def __call__(self, tables: dict[str, SourceRelation]):
        for result in self.joiner(tables):
            clf = SkLearnTreeRelation(result, tree.DecisionTreeClassifier())
            raise NotImplementedError
