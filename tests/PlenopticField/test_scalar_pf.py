"""Unit test suite for scalar PF construction and manipulations."""

import numpy as np
import pytest

from WDF.PlenopticField import ScalarPF

@pytest.fixture
def setup_major_PFs():
    """Setup major PFs in the $x$- and $y$-directions."""

    # Initial coordinate arrays
    x = np.array([[-1, 0, 1]])
    y = np.array([[-3, -1.5, 0, 1.5, 3]])

    # Major plenoptic fields
    X = ScalarPF(x, "major", "x", 3)
    Y = ScalarPF(y, "major", "y", 2)
    return X._pf, Y._pf

@pytest.fixture
def setup_results():
    """Setup what we expect the results should be."""

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
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        ]
    )

    return X, Y

def test_major_PF(setup_major_PFs, setup_results):
    """Test the construction of major plenoptic fields."""

    X_actual, Y_actual = setup_major_PFs
    X_result, Y_result = setup_results

    assert np.all(X_actual == X_result)
    assert np.all(Y_actual == Y_result)