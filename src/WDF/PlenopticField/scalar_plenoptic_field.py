"""
This represents a "scalar" plenoptic field (PF). This is a PF-representation of a 1-D
quantity.
"""

import numpy as np


class PFError(Exception):
    ...


class ScalarPF:
    def __init__(
        self,
        arr: np.ndarray,
        pf_type: str,
        dim: str,
        num: int,
    ):
        """
        Constructor...

        NOTE: this is explicitly called when constructing new scalar PFs representing
        coordinate arrays, coordinate vectors, etc.

        Args:
            arr:
                Array from which the scalar PF will be constructed from. The user
                should not have access to this object. Size: $N_s \times 1$.
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

        if pf_type.lower() == "major":
            self._construct_major(arr, num, dim)
        elif pf_type.lower() == "minor":
            self._construct_minor(arr, num, dim)

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
