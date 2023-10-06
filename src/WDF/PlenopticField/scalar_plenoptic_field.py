"""
This represents a "scalar" plenoptic field (PF). This is a PF-representation of a 1-D
quantity.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class PFError(Exception):
    ...


class ScalarPF:
    def __init__(
        self,
        arr: Optional[np.ndarray] = [],
        pf_type: Optional[str] = "",
        dim: Optional[str] = "",
        num: Optional[int] = -1,
    ):
        """
        Constructor...

        NOTE: this is explicitly called when constructing new scalar PFs representing
        coordinate arrays, coordinate vectors, etc.

        Args:
            arr:
                Array from which the scalar PF will be constructed from. The user
                should not have access to this object. Size: $N_s \times 1$. If the
                length of the array is zero, then the user is trying to build from
                another plenoptic field.
            pf_type:
                The type of PF: major or minor. A major PF is a PF where the indices of
                each micro-image correspond to the coordinates of the `arr`.
            dim:
                Direction that the corresponding coordinate varies in (either `'x'` for
                the `x`-direction, or `'y'` for the `y`-direction).
            num:
                Either the number of micro-images to generate for each major coordinate
                direction (assuming a minor PF) or the number pixels in each
                micro-image (assuming a major PF). It is assumed that all micro-images
                and PFs are square.
        """

        if len(arr) > 0:
            if pf_type.lower() == "major":
                self._construct_major(arr, num, dim)
            elif pf_type.lower() == "minor":
                self._construct_minor(arr, num, dim)
            else:
                raise PFError(
                    "***ERROR***:\tThe plenoptic field type entered is invalid\n"
                    f"pf_type = {pf_type}"
                )
        else:
            if not pf_type == "" or not dim == "" or not num == -1:
                raise PFError(
                    "***ERROR***:\tYou have passed invalid arguments for initializing"
                    f" an empty plenoptic field: pf_type = {pf_type}, dim = {dim},"
                    f" num = {num}"
                )

            self._pf: np.ndarray = np.array([])

    def __add__(self, other: ScalarPF) -> ScalarPF:
        """
        Overrides Python's built-in "add" operator. This will add two scalar PFs on a
        micro-image pixel-basis.

        Args:
            other:
                The scalar PF to add this one to.

        Returns:
            return:
                ...
        """

        ret: np.ndarray = self._pf + other._pf
        return ScalarPF(arr=ret)

    def __sub__(self, other: ScalarPF) -> ScalarPF:
        """
        Overrides Python's built-in "minus" operator. This will add two scalar PFs on a
        micro-image pixel-basis.

        Args:
            other:
                The scalar PF to add this one to.

        Returns:
            return:
                ...
        """

        ret: np.ndarray = self._pf - other._pf
        return ScalarPF(arr=ret)

    def _construct_major(self, arr: np.ndarray, num_pixels: int, dim: int) -> None:
        """
        This private function constructs a major PF from the provided array.

        Args:
            arr:
                Array from which the scalar PF will be constructed from. Size:
                $N_s \times 1$.
            num_pixels:
                The number pixels in each micro-image. It is assumed that all
                micro-images and PFs are square.
            dim:
                Direction that the corresponding coordinate varies in (either `'x'` for
                the `x`-direction, or `'y'` for the `y`-direction).

        Raises:
            PFError:
                You passed a dimension not recognized
                dimension = <...>

        Example:
            ```py
            >>> import WDF.PlenopticField
            >>> import numpy as np
            >>> arr = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
            >>> major_pf = WDF.PlenopticField.ScalarPF(arr, "major", "x", 2)
            >>> major_pf.shape
            (10, 10)
            ```
        """

        arr = arr.flatten()
        num_micro: int = arr.shape[0]
        self._pf: np.ndarray = np.zeros(
            (num_micro * num_pixels, num_micro * num_pixels)
        )

        for i in range(num_micro):
            qi = num_pixels * i

            for j in range(num_micro):
                qj = num_pixels * j

                if dim.lower() == "x":
                    self._pf[qi : qi + num_pixels, qj : qj + num_pixels] = arr[
                        j
                    ] * np.ones((num_pixels, num_pixels))
                elif dim.lower() == "y":
                    self._pf[qi : qi + num_pixels, qj : qj + num_pixels] = arr[
                        i
                    ] * np.ones((num_pixels))
                else:
                    raise PFError(
                        "***ERROR***:\tYou passed a dimension not recognized\n"
                        f"dimension = {dim.lower()}"
                    )

    def _construct_minor(self, arr: np.ndarray, num_micro: int, dim: int) -> None:
        """
        This private function constructs a minor PF from the provided array.

        Args:
            arr:
                Array from which the scalar PF will be constructed from. Size:
                $N_s \times 1$.
            num_micro:
                The number pixels in each micro-image. It is assumed that all
                micro-images and PFs are square.
            dim:
                Direction that the corresponding coordinate varies in (either `'x'` for
                the `x`-direction, or `'y'` for the `y`-direction).

        Raise:
            PFError:
                You passed a dimension not recognized
                dimension = <...>

        Example:
            ```py
            >>> import WDF.PlenopticField
            >>> import numpy as np
            >>> arr = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
            >>> minor_pf = WDF.PlenopticField.ScalarPF(arr, "minor", "x", 2)
            >>> minor_pf.shape
            (10, 10)
            ```
        """

        arr = arr.flatten()
        arr_x, arr_y = np.meshgrid(arr, arr)
        num_pixels: int = arr.shape[0]
        self._pf: np.ndarray = np.zeros(
            (num_micro * num_pixels, num_micro * num_pixels)
        )

        for i in range(num_micro):
            qi = num_pixels * i

            for j in range(num_micro):
                qj = num_pixels * j

                if dim.lower() == "x":
                    self._pf[qi : qi + num_pixels, qj : qj + num_pixels] = arr_x
                elif dim.lower() == "y":
                    self._pf[qi : qi + num_pixels, qj : qj + num_pixels] = arr_y
                else:
                    raise PFError(
                        "***ERROR***:\tYou passed a dimension not recognized\n"
                        f"dimension = {dim.lower()}"
                    )

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Gets the shape of the plenoptic field: (number of micro-images * number of
        pixels per micro-image). This is just forwarding `numpy.shape` from the
        underlying array.

        Type:
            Tuple[int, int]
        """

        return self._pf.shape
