# Copyright (c) The InverSQL Authors - All Rights Reserved

import pytest

from inversql.highlights import AxisInfo, CellHighLight, HeaderHighLight, HighLight


@pytest.fixture(params=[5, 7, 9])
def num_rows(request) -> int:
    "The number of rows."
    return request.param


@pytest.fixture(params=[5, 7, 9])
def num_cols(request) -> int:
    "The number of columns."
    return request.param


@pytest.fixture(scope="module")
def cell_hl_cls():
    "The class of `CellHighLight`."
    return CellHighLight


@pytest.fixture(scope="module")
def header_hl_cls():
    "The class of `HeaderHighLight`."
    return HeaderHighLight


class TestHighlight:
    @pytest.fixture(params=[cell_hl_cls.__name__, header_hl_cls.__name__])
    def highlight(self, num_rows, num_cols, request) -> HighLight:
        """
        The state of highlight. Must return a highlight object.
        All highlight classes have the same `__init__` signature `(num_rows, num_cols)`.
        """

        klass = request.getfixturevalue(request.param)
        return klass(num_rows=num_rows, num_cols=num_cols)

    def test_api(self, highlight):
        assert isinstance(highlight, HighLight)
        assert isinstance(highlight(0), AxisInfo)
        assert isinstance(highlight(1), AxisInfo)

    def test_magic_method(self, highlight):
        assert str(highlight(0)) == "row"
        assert str(highlight(1)) == "col"

        assert int(highlight(0)) == 0
        assert int(highlight(1)) == 1

    def test_init_state(self, highlight):
        assert not highlight(0)[3]
        assert not highlight(1)[3]

    @pytest.mark.parametrize("ax", [-100, 99])
    def test_axis_out_of_bounds(self, highlight, ax):
        with pytest.raises(IndexError):
            highlight(ax)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_api_fail(self, highlight, axis):
        with pytest.raises(IndexError):
            highlight(axis)[99]


class TestCell:
    @pytest.fixture
    def cell_hl(self, cell_hl_cls, num_rows, num_cols) -> CellHighLight:
        "The cell highlighter object."
        return cell_hl_cls(num_rows=num_rows, num_cols=num_cols)

    def test_object(self, cell_hl, num_rows, num_cols):
        assert cell_hl.cells.shape == (num_rows, num_cols)
        assert (cell_hl.cells == 0).all()

        assert isinstance(cell_hl(0), AxisInfo)
        assert isinstance(cell_hl(1), AxisInfo)

    def test_mutate(self, cell_hl):
        # Select rows (1, -1) and columns (2, -2).
        cell_hl[1, 2] = True
        cell_hl[-1, -2] = True

        # 2 rows / cols are selected.
        assert len(list(cell_hl(0))) == 2
        assert len(list(cell_hl(1))) == 2

        cell_hl[-1, 2] = True
        # Still, 2 rows / cols are selected.
        assert len(list(cell_hl(0))) == 2
        assert len(list(cell_hl(1))) == 2

        # Still 2 rows (1, -1), but only 1 column (2).
        cell_hl[-1, -2] = False
        assert len(list(cell_hl(0))) == 2
        assert list(cell_hl(1)) == [2]

    def test_mutate_out_of_bounds(self, cell_hl):
        with pytest.raises(IndexError):
            _ = cell_hl[10, 2]


class TestHeader:
    @pytest.fixture
    def header_hl(self, header_hl_cls, num_rows, num_cols) -> HeaderHighLight:
        "The header highlighter object."
        return header_hl_cls(num_rows=num_rows, num_cols=num_cols)

    def test_object(self, header_hl, num_rows, num_cols):
        assert header_hl.rows.shape == (num_rows,)
        assert header_hl.cols.shape == (num_cols,)

    def test_mutate(self, header_hl):
        # Select rows (1, -1) and columns (2, -2).
        header_hl(0)[1] = header_hl(0)[-1] = True
        header_hl(1)[2] = header_hl(1)[-2] = True

        # 2 rows / cols are selected.
        assert len(list(header_hl(0))) == 2
        assert len(list(header_hl(1))) == 2

        # Still 2 rows (1, -1), but only 1 column (2).
        header_hl(1)[-2] = False
        assert len(list(header_hl(0))) == 2
        assert list(header_hl(1)) == [2]
