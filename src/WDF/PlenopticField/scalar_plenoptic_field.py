"""
This represents a "scalar" plenoptic field (PF). This is a PF-representation of a 1-D
quantity.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Optional, Tuple

import numpy as np


class SchemaValidator:

    @dataclasses.dataclass
    class Schema:
        arr = np.ndarray
        pf_type = str
        dim = str
        num = int
        old_pf = np.ndarray

    def validate(self, input_args: dict):
        check = self.Schema()
        for key in input_args.keys():
            if hasattr(check, key):
                assert isinstance(input_args[key], getattr(check, key)), \
                    "Given input doesn't have the required type!"
            else:
                raise ValueError(f"You inputted an unexpected argument {key}!")


class BuildOpt(enum.Enum):
    New: int = 0
    Old: int = 1


class ScalarPF:

    def __init__(self, option: BuildOpt, **kwargs: dict):
        SchemaValidator().validate(kwargs)
        if option == BuildOpt.New:
            self._construct_new(**kwargs)
        elif option == BuildOpt.Old:
            old_pf: np.ndarray = kwargs["old_pf"]
            self._construct_old(old_pf)

    def _construct_new(self, **kwargs: dict) -> None:
        assert "arr" in kwargs.keys(), "You didn't provide an array to build from!"
        assert "dim" in kwargs.keys(), "You didn't provide a dimension of the PF!"
        assert "num" in kwargs.keys(), "You didn't provide a number to build the PF!"

        arr, dim, num = kwargs["arr"], kwargs["dim"], kwargs["num"]
        if kwargs["pf_type"].lower() == "major":
            self._construct_new_major(arr, dim, num)
        elif kwargs["pf_type"].lower() == "minor":
            self._construct_new_minor(arr, dim, num)
        else:
            raise ValueError(f"You passed an invalid PF type {kwargs['pf_type']}!")

    def _construct_new_major(self, arr: np.ndarray, dim: str, num: int) -> None:
        ...

    def _construct_new_minor(self, arr: np.ndarray, dim: str, num: int) -> None:
        ...

    def _construct_old(self, old_pf: np.ndarray) -> None:
        ...