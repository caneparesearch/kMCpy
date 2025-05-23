"""
This is inherited from pymatgen.analysis.local_env
"""

from pymatgen.analysis.local_env import CutOffDictNN, NearNeighbors
from kmcpy.external.structure import StructureKMCpy
from monty.serialization import loadfn
import os

_directory = os.path.join(os.path.dirname(__file__))


class CutOffDictNNKMCpy(CutOffDictNN):
    def get_nn_info(self, structure, n):
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n in structure.

        Args:
            structure (Structure): input structure.
            n (integer): index of site for which to determine near-neighbor
                sites.

        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one
                of which represents a coordinated site, its image location,
                and its weight.
        """
        site = structure[n]

        neighs_dists = structure.get_neighbors(site, self._max_dist)

        nn_info = []

        if "wyckoff_sequence" in structure.site_properties:
            if "supercell" in structure.site_properties:
                for nn in neighs_dists:
                    n_site = nn
                    dist = nn.nn_distance
                    neigh_cut_off_dist = self._lookup_dict.get(
                        site.species_string, {}
                    ).get(n_site.species_string, 0.0)

                    if dist < neigh_cut_off_dist:
                        nn_info.append(
                            {
                                "site": n_site,
                                "image": self._get_image(structure, n_site),
                                "weight": dist,
                                "site_index": self._get_original_site(
                                    structure, n_site
                                ),
                                "wyckoff_sequence": n_site.properties[
                                    "wyckoff_sequence"
                                ],
                                "local_index": n_site.properties["local_index"],
                                "label": n_site.properties["label"],
                                "supercell": n_site.properties["supercell"],
                            }
                        )
            else:
                for nn in neighs_dists:
                    n_site = nn
                    dist = nn.nn_distance
                    neigh_cut_off_dist = self._lookup_dict.get(
                        site.species_string, {}
                    ).get(n_site.species_string, 0.0)

                    if dist < neigh_cut_off_dist:
                        nn_info.append(
                            {
                                "site": n_site,
                                "image": self._get_image(structure, n_site),
                                "weight": dist,
                                "site_index": self._get_original_site(
                                    structure, n_site
                                ),
                                "wyckoff_sequence": n_site.properties[
                                    "wyckoff_sequence"
                                ],
                                "local_index": n_site.properties["local_index"],
                                "label": n_site.properties["label"],
                            }
                        )
        else:
            for nn in neighs_dists:
                n_site = nn
                dist = nn.nn_distance
                neigh_cut_off_dist = self._lookup_dict.get(site.species_string, {}).get(
                    n_site.species_string, 0.0
                )

                if dist < neigh_cut_off_dist:
                    nn_info.append(
                        {
                            "site": n_site,
                            "image": self._get_image(structure, n_site),
                            "weight": dist,
                            "site_index": self._get_original_site(structure, n_site),
                        }
                    )

        return nn_info


class CutOffDictNNrange(NearNeighbors):
    """
    Jerry: Modified from CutOffDictNN in pymatgen so that it can search all pairs between a range of distances [d_min, d_max]
    c
    A basic NN class using a dictionary of fixed cut-off distances.
    Only pairs of elements listed in the cut-off dictionary are considered
    during construction of the neighbor lists.
    Omit passing a dictionary for a Null/Empty NN class.
    """

    def __init__(self, cut_off_dict=None):
        """
        Args:
            cut_off_dict (dict[str, float]): a dictionary
            of cut-off distances, e.g. {('Fe','O'): [2.0,3.0]} for
            a maximum Fe-O bond length between 2.0 and 3.0 Angstroms.
            Bonds will only be created between pairs listed
            in the cut-off dictionary.
            If your structure is oxidation state decorated,
            the cut-off distances will have to explicitly include
            the oxidation state, e.g. {('Fe2+', 'O2-'): [2.0,3.0]}
        """
        from collections import defaultdict

        self.cut_off_dict = cut_off_dict or {}
        # for convenience  Jerry: added minimum and maximum
        self._max_dist = 0.0
        self._min_dist = 1e3
        lookup_dict_max = defaultdict(dict)
        lookup_dict_min = defaultdict(dict)
        for (sp1, sp2), (dist_min, dist_max) in self.cut_off_dict.items():
            lookup_dict_max[sp1][sp2] = dist_max
            lookup_dict_max[sp2][sp1] = dist_max
            if dist_max > self._max_dist:
                self._max_dist = dist_max

            lookup_dict_min[sp1][sp2] = dist_min
            lookup_dict_min[sp2][sp1] = dist_min
            if dist_min < self._min_dist:
                self._min_dist = dist_min
        self._lookup_dict_max = lookup_dict_max
        self._lookup_dict_min = lookup_dict_min

    @property
    def structures_allowed(self):
        """
        Boolean property: can this NearNeighbors class be used with Structure
        objects?
        """
        return True

    @property
    def molecules_allowed(self):
        """
        Boolean property: can this NearNeighbors class be used with Molecule
        objects?
        """
        return True

    @property
    def extend_structure_molecules(self):
        """
        Boolean property: Do Molecules need to be converted to Structures to use
        this NearNeighbors class? Note: this property is not defined for classes
        for which molecules_allowed is False.
        """
        return True

    @staticmethod
    def from_preset(preset):
        """
        Initialise a CutOffDictNN according to a preset set of cut-offs.
        Args:
            preset (str): A preset name. The list of supported presets are:
                - "vesta_2019": The distance cut-offs used by the VESTA
                  visualisation program.
        Returns:
            A CutOffDictNN using the preset cut-off dictionary.
        """
        if preset == "vesta_2019":
            cut_offs = loadfn(os.path.join(_directory, "vesta_cutoffs.yaml"))
            return CutOffDictNNrange(cut_off_dict=cut_offs)

        raise ValueError(f"Unrecognised preset: {preset}")

    def get_nn_info(self, structure: StructureKMCpy, n: int):
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n in structure.
        Args:
            structure (StructureKMCpy): input structure.
            n (int): index of site for which to determine near-neighbor
                sites.
        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one
                of which represents a coordinated site, its image location,
                and its weight.
        """
        site = structure[n]

        neighs_dists = structure.get_neighbors(site, self._max_dist)

        nn_info = []
        for nn in neighs_dists:
            n_site = nn
            dist = nn.nn_distance
            neigh_cut_off_dist_max = self._lookup_dict_max.get(
                site.species_string, {}
            ).get(n_site.species_string, 0.0)
            neigh_cut_off_dist_min = self._lookup_dict_min.get(
                site.species_string, {}
            ).get(n_site.species_string, 0.0)
            if dist < neigh_cut_off_dist_max and dist > neigh_cut_off_dist_min:
                nn_info.append(
                    {
                        "site": n_site,
                        "image": self._get_image(structure, n_site),
                        "weight": dist,
                        "site_index": self._get_original_site(structure, n_site),
                    }
                )

        return nn_info
