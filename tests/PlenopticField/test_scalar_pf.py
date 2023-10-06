"""Unit test suite for scalar PF construction and manipulations."""

import numpy as np
import pytest

from WDF.PlenopticField import ScalarPF
from WDF.PlenopticField.scalar_plenoptic_field import PFError


@pytest.fixture
def setup_major_PFs():
    """Setup major PFs in the $x$- and $y$-directions."""

    # Initial coordinate arrays
    x = np.array([[-1, 0, 1]])
    y = np.array([[-3, -1.5, 0, 1.5, 3]])

    # Major plenoptic fields
    X = ScalarPF(arr=x, pf_type="major", dim="x", num=3)
    Y = ScalarPF(arr=y, pf_type="major", dim="y", num=2)
    return X._pf, Y._pf


@pytest.fixture
def setup_minor_PFs():
    """Setup minor PFs in the $x$- and $y$-directions."""

    # Initial coordinate arrays
    x = np.array([[-1, 0, 1]])
    y = np.array([[-3, -1.5, 0, 1.5, 3]])

    # Major plenoptic fields
    X = ScalarPF(x, "minor", "x", 3)
    Y = ScalarPF(y, "minor", "y", 2)
    return X._pf, Y._pf


@pytest.fixture
def setup_major_results():
    """Setup what we expect the results should be for major PFs."""

    X = np.array(
        [
            [-1, -1, -1, 0, 0, 0, 1, 1, 1],
            [-1, -1, -1, 0, 0, 0, 1, 1, 1],
            [-1, -1, -1, 0, 0, 0, 1, 1, 1],
            [-1, -1, -1, 0, 0, 0, 1, 1, 1],
            [-1, -1, -1, 0, 0, 0, 1, 1, 1],
            [-1, -1, -1, 0, 0, 0, 1, 1, 1],
            [-1, -1, -1, 0, 0, 0, 1, 1, 1],
            [-1, -1, -1, 0, 0, 0, 1, 1, 1],
            [-1, -1, -1, 0, 0, 0, 1, 1, 1],
        ]
    )

    Y = np.array(
        [
            [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
            [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
            [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
            [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        ]
    )

    return X, Y


@pytest.fixture
def setup_minor_results():
    """Setup what we expect the results should be for minor PFs."""

    X = np.array(
        [
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
            [-1, 0, 1, -1, 0, 1, -1, 0, 1],
        ]
    )

    Y = np.array(
        [
            [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
            [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
            [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        ]
    )

    return X, Y


def test_pf_empty_PF():
    """Test the construction of an empty plenoptic field."""

    empty = ScalarPF(arr=np.array([]))
    assert empty._pf.shape == (0, )


def test_pf_invalid_constructor():
    """Test throwing error when trying to pass mutliple args."""

    with pytest.raises(PFError) as exc:
        ScalarPF(arr=np.array([]), dim="x")

    assert exc.value.args[0] == (
        "***ERROR***:\tYou have passed invalid arguments for initializing" \
        " an empty plenoptic field: pf_type = , dim = x, num = -1"
    )


def test_pf_major_PF(setup_major_PFs, setup_major_results):
    """Test the construction of major plenoptic fields."""

    X_actual, Y_actual = setup_major_PFs
    X_result, Y_result = setup_major_results

    assert np.all(X_actual == X_result)
    assert np.all(Y_actual == Y_result)


def test_pf_minor_PF(setup_minor_PFs, setup_minor_results):
    """Test the construction of minor plenoptic fields."""

    X_actual, Y_actual = setup_minor_PFs
    X_result, Y_result = setup_minor_results

    assert np.all(X_actual == X_result)
    assert np.all(Y_actual == Y_result)


@pytest.mark.parametrize(
        "pf_type",
        [
            "major", "minor"
        ]
)
def test_pf_constructor_error_bad_dim(pf_type):
    with pytest.raises(PFError) as exc:
        ScalarPF(arr=np.zeros((1, 1)), pf_type=pf_type, dim="z", num=3)

    assert exc.value.args[0] == \
        "***ERROR***:\tYou passed a dimension not recognized\ndimension = z"
    
def test_pf_constructor_error_bad_type():
    with pytest.raises(PFError) as exc:
        ScalarPF(arr=np.zeros((1, 1)), pf_type="foo", dim="x", num=3)

    assert exc.value.args[0] == \
    "***ERROR***:\tThe plenoptic field type entered is invalid\npf_type = foo"
