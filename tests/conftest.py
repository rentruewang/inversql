# Copyright (c) The InverSQL Authors - All Rights Reserved

import pathlib

import pandas as pd
import pytest

from inversql.rels import CellLoc, SourceRelation


@pytest.fixture(params=range(42))
def seed(request):
    "Used for testing operations that are fast, but may need many seeds."
    return request.param


@pytest.fixture(scope="module")
def data() -> pathlib.Path:
    folder = pathlib.Path(__file__).parent / "data"
    assert folder.exists() and folder.is_dir()
    return folder


@pytest.fixture(scope="module")
def join_left_csv(data: pathlib.Path):
    csv = data / "join_left.csv"
    assert csv.exists() and csv.is_file()
    return csv


@pytest.fixture
def join_left_df(join_left_csv: pathlib.Path):
    return pd.read_csv(join_left_csv)


@pytest.fixture(scope="module")
def join_right_csv(data: pathlib.Path):
    csv = data / "join_right.csv"
    assert csv.exists() and csv.is_file()
    return csv


@pytest.fixture
def join_right_df(join_right_csv: pathlib.Path):
    return pd.read_csv(join_right_csv)


@pytest.fixture
def left_rel(join_left_df: pd.DataFrame):
    return SourceRelation("join_left", join_left_df, cells=[CellLoc(0, 1)])


@pytest.fixture
def right_rel(join_right_df: pd.DataFrame):
    return SourceRelation("join_right", join_right_df, cells=[CellLoc(1, 1)])
