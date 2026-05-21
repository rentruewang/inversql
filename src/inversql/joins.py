# Copyright (c) The InverSQL Authors - All Rights Reserved

import abc
import typing
import pandas as pd
from collections import abc as cabc


@typing.runtime_checkable
class Joiner(typing.Protocol):
    "The interface for joining 2 dataframes, in every ways you can imagine."

    def join(self, left: pd.DataFrame, right: pd.DataFrame, /) -> pd.DataFrame: ...


def explode_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame: ...
