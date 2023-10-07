"""Unit test suite for scalar PF construction and manipulations."""

import numpy as np
import pytest

from WDF.PlenopticField.scalar_plenoptic_field import (
    BuildOpt, SchemaValidator, ScalarPF
)


def test_scalar_pf_schema_validator_fails():
    input_args = {
        "arr": np.array([1, 2, 3]),
        "dim": "x",
        "num": 10.0,
        "pf_type": "major"
    }

    with pytest.raises(AssertionError) as exc:
        SchemaValidator().validate(input_args)

    assert exc.value.args[0] == "Given input doesn't have the required type!"

    input_args["num"] = 10
    input_args["foo"] = "bar"
    with pytest.raises(ValueError) as exc:
        SchemaValidator().validate(input_args)

    assert exc.value.args[0] == "You inputted an unexpected argument foo!"


def test_scalar_pf_factory_new_fails():
    with pytest.raises(AssertionError) as exc:
        ScalarPF(BuildOpt.New, pf_type="major", arr=np.array([0, 1, 2]), dim="x")

    assert exc.value.args[0] == "You didn't provide a number to build the PF!"
    with pytest.raises(AssertionError) as exc:
        ScalarPF(BuildOpt.New, pf_type="major", arr=np.array([0, 1, 2]), num=10)

    assert exc.value.args[0] == "You didn't provide a dimension of the PF!"
    with pytest.raises(AssertionError) as exc:
        ScalarPF(BuildOpt.New, pf_type="major", dim="x", num=10)

    assert exc.value.args[0] == "You didn't provide an array to build from!"
    with pytest.raises(ValueError) as exc:
        ScalarPF(BuildOpt.New, pf_type="eggs", arr=np.array([0, 1, 2]), dim="x", num=10)

    assert exc.value.args[0] == "You passed an invalid PF type eggs!"
