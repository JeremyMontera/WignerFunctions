"""Unit test suite for scalar PF construction and manipulations."""

import numpy as np
import pytest

from WDF.PlenopticField import ScalarPF
from WDF.PlenopticField.scalar_plenoptic_field import PFError


BINOPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b}


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
    assert empty._pf.shape == (0,)


def test_pf_invalid_constructor():
    """Test throwing error when trying to pass mutliple args."""

    with pytest.raises(PFError) as exc:
        ScalarPF(arr=np.array([]), dim="x")

    assert exc.value.args[0] == (
        "***ERROR***:\tYou have passed invalid arguments for initializing"
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


@pytest.mark.parametrize("pf_type", ["major", "minor"])
def test_pf_constructor_error_bad_dim(pf_type):
    """Tests that an error is raised when passing an unknown dimension."""

    with pytest.raises(PFError) as exc:
        ScalarPF(arr=np.zeros((1, 1)), pf_type=pf_type, dim="z", num=3)

    assert (
        exc.value.args[0]
        == "***ERROR***:\tYou passed a dimension not recognized\ndimension = z" # noqa: W503
    )


def test_pf_constructor_error_bad_type():
    """Tests that an error is raised when passing an unknown pf type."""

    with pytest.raises(PFError) as exc:
        ScalarPF(arr=np.zeros((1, 1)), pf_type="foo", dim="x", num=3)

    assert (
        exc.value.args[0]
        == "***ERROR***:\tThe plenoptic field type entered is invalid\npf_type = foo" # noqa: W503
    )


@pytest.fixture
def add(setup_major_PFs, setup_minor_PFs):
    X_major, Y_major = setup_major_PFs
    X_minor, Y_minor = setup_minor_PFs

    ret_X: ScalarPF = X_major + X_minor
    ret_Y: ScalarPF = Y_major + Y_minor

    return ret_X, ret_Y


@pytest.fixture
def add_res():
    ret_X: np.ndarray = np.array(
        [
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
        ]
    )

    ret_Y: np.ndarray = np.array(
        [
            [-6, -6, -6, -6, -6, -6, -6, -6, -6, -6],
            [-4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5],
            [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        ]
    )

    return ret_X, ret_Y


@pytest.fixture
def sub(setup_major_PFs, setup_minor_PFs):
    X_major, Y_major = setup_major_PFs
    X_minor, Y_minor = setup_minor_PFs

    ret_X: ScalarPF = X_major + X_minor
    ret_Y: ScalarPF = Y_major - Y_minor

    return ret_X, ret_Y


@pytest.fixture
def sub_res():
    ret_X: np.ndarray = np.array(
        [
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
            [-2, -1, 0, -1, 0, 1, 0, 1, 2],
        ]
    )

    ret_Y: np.ndarray = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
            [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
            [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
            [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    return ret_X, ret_Y


@pytest.mark.parametrize(
    ("binop", "result"),
    [
        ("add", "add_res"),
        ("sub", "sub_res"),
    ]
)
def test_pf_pf_operators(binop, result, request):
    """Tests binary operations of plenoptic fields: addition/subtraction."""

    binop = request.getfixturevalue(binop)
    result = request.getfixturevalue(result)

    assert np.all(binop[0] == result[0])
    assert np.all(binop[1] == result[1])


@pytest.mark.parametrize(
    ("binop", ),
    [
        ("add", ),
        ("sub", ),
    ]
)
def test_pf_pf_operators_error_shape(binop):
    """Tests error thrown when trying to add/subtract two different shapes."""

    x1: ScalarPF = ScalarPF(arr=np.array([0.0, 1.0]), pf_type="major", dim="x", num=2)
    x2: ScalarPF = ScalarPF(arr=np.array([0.0, 1.0]), pf_type="minor", dim="y", num=3)

    with pytest.raises(PFError) as exc:
        BINOPS[binop](x1, x2)

    assert exc.value.args[0] == \
        "***ERROR***:\tThe shapes of the two PFs are not the same."
