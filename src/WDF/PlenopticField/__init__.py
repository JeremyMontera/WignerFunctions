"""
This will define the plenoptic field data structure and other related data structures.
A plenoptic field can be seen as a sort of block matrix: it is assembled by layering
sub-matrices (called micro-images) next to each other. This allows one to store 4-D
data as a 2-D array instead of as a 4-D array.
"""

from .scalar_plenoptic_field import ScalarPF  # noqa: F401
