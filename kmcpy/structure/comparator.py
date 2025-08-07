from pymatgen.analysis.structure_matcher import AbstractComparator


class SupercellComparator(AbstractComparator):
    """
    A Comparator that matches sites, given some overlap in the element
    composition.
    """

    def are_equal(self, sp1, sp2) -> bool:
        """
        True if sp1 and sp2 are considered equivalent according to site_specie_mapping.

        Args:
            sp1: First species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.
            sp2: Second species. A dict of {specie/element: amt} as per the
                definition in Site and PeriodicSite.

        Returns:
            True if sp1 and sp2 are allowed to be on the same site according to mapping.
        """
        return True

    def get_hash(self, composition):
        """Get the fractional composition."""
        return composition.fractional_composition