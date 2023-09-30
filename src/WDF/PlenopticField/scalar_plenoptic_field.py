"""
This represents a "scalar" plenoptic field (PF). This is a PF-representation of a 1-D
quantity.
"""

import numpy as np


class PFError(Exception):
    ...


class ScalarPF:
    def __init__(self, arr: np.ndarray, pf_type: str, dim: str):
        """
        Constructor...

        Args:
            arr:
                Array from which the scalar PF will be constructed from. The user
                should not have access to this object.
            pf_type:
                The type of PF: major or minor. A major PF is a PF where the indices of
                each micro-image correspond to the coordinates of the `arr`.
        """

        ...

    def _construct_major(self, arr: np.ndarray) -> np.ndarray:
        """
        This private function constructs a major PF from the provided array.

        Args:
            arr:
                ...

        Returns:
            ret:
                ...
        """

        self._pf: np.ndarray = np.zeros((1, 1))
        return self._pf
