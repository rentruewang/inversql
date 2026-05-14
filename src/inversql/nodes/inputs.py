# Copyright (c) The InverSQL Authors - All Rights Reserved

import dataclasses as dcls
import typing

from .nodes import QueryNode

__all__ = ["InputNode"]


@dcls.dataclass(frozen=True)
class InputNode(QueryNode, key="SOURCE"):
    """
    The input nodes are the sources to all the queries.
    """

    table: str
    """
    The table name to use.
    """

    @typing.override
    def __str__(self) -> str:
        return self.table

    @typing.override
    def children(self) -> tuple[()]:
        return ()
