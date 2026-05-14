# Copyright (c) The InverSQL Authors - All Rights Reserved

import abc
import dataclasses as dcls
import difflib
import typing

__all__ = ["QueryNode"]


@dcls.dataclass(frozen=True)
class QueryNode(abc.ABC):
    """
    The query tree nodes used by `inversql`.

    Note:
        I thought about directly using either exteral libary's interal representation,
        like `sqlalchemy` or `pypika`, for the query tree,
        s.t. I would not need to do it myself,
        but that query tree is:
            #. Not tailored to the same use cases
            #. Not under my control
            #. Doing something extra s.t. there would always be performance costs
        So I decided against it.
    """

    _SUBCLASSES: typing.ClassVar[dict[str, type["QueryNode"]]] = {}
    """
    The shared registry for `QueryNode`, storing all the subtypes by their keys.
    """

    @classmethod
    def __init_subclass__(cls, key: str) -> None:
        if existing := cls._SUBCLASSES.get(key):
            raise KeyError(
                "Duplicate keys not allowed for different classes. "
                f"{key=} already exists for class: {existing}, "
                f"but class: {cls} also uses the same key."
            )

        cls._SUBCLASSES[key] = cls

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        The generated SQL query.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def children(self) -> tuple["QueryNode", ...]:
        """
        The children nodes of the tree.
        """

        raise NotImplementedError

    @classmethod
    @typing.final
    def sub_type(cls, name: str) -> type["QueryNode"]:
        """
        Find a sub type of `QueryNode` class.
        """
        if found := cls._SUBCLASSES.get(name):
            return found

        err_msg = f"Class for name: {name} not found."

        if close_matches := difflib.get_close_matches(name, cls._SUBCLASSES, n=1):
            err_msg += f" Do you mean: {close_matches[0]}"

        raise KeyError(err_msg)
