"""
Basis functions for lattice models in kMCpy.

This module provides different basis functions for converting between species and numerical values
used in cluster expansion and other lattice-based calculations.
"""


class OccupationBasis:
    """
    Occupation basis function that maps between binary occupation states.
    Uses [0, 1] or [-1, 1] representation for site occupancy.
    """
    
    def __init__(self):
        # For occupation basis: 0 = same as template, 1 = different from template
        # In some contexts: -1 = same as template, +1 = different from template
        self.basis_function = [0, 1]  # Default occupation representation


class ChebychevBasis:
    """
    Chebyshev basis function that maps between [-1, +1] representation.
    Often used in cluster expansion for better numerical properties.
    """
    
    def __init__(self):
        # For Chebyshev basis: -1 = same as template, +1 = different from template
        self.basis_function = [-1, 1]
