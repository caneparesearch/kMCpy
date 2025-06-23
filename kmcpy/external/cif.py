"""
This is inherited from pymatgen.io.cif
"""

from pymatgen.io.cif import CifParser
from pymatgen.core.operations import MagSymmOp, SymmOp
import warnings
from pymatgen.util.coord import find_in_coord_list_pbc, in_coord_list_pbc
from io import StringIO
import re
import numpy as np
from pymatgen.electronic_structure.core import Magmom
import math
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupOperations
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list_pbc
from kmcpy.external import StructureKMCpy

class CifParserKMCpy(CifParser):

    @staticmethod
    def from_string(cif_string, **kwargs):
        """
        Creates a CifParser from a string.

        Args:
            cif_string (str): String representation of a CIF.
            **kwargs: Passthrough of all kwargs supported by CifParser.

        Returns:
            CifParser
        """
        stream = StringIO(cif_string)
        return CifParserKMCpy(stream, **kwargs)
    
    def _get_labeled_structure(self, data, primitive, symmetrized):
        """modified version from _get_structure, which can add the atom label to the site

        Args:
            data (dict): data parsed by the cif parser
            primitive (_type_): not implemented
            symmetrized (bool): not implemented
        """

        def get_num_implicit_hydrogens(sym):
            num_h = {"Wat": 2, "wat": 2, "O-H": 1}
            return num_h.get(sym[:3], 0)

        lattice = self.get_lattice(data)

        # if magCIF, get magnetic symmetry moments and magmoms
        # else standard CIF, and use empty magmom dict
        if self.feature_flags["magcif_incommensurate"]:
            raise NotImplementedError(
                "Incommensurate structures not currently supported."
            )
        if self.feature_flags["magcif"]:
            self.symmetry_operations = self.get_magsymops(data)
            magmoms = self.parse_magmoms(data, lattice=lattice)
        else:
            self.symmetry_operations = self.get_symops(data)
            magmoms = {}

        oxi_states = self._parse_oxi_states(data)

        coord_to_species = {}
        coord_to_magmoms = {}

        # add the coords_to_labels to record the label (Na1,Na2,Zr,S/P, etc.)
        coord_to_labels = {}

        def get_matching_coord(coord):
            keys = list(coord_to_species.keys())
            coords = np.array(keys)
            for op in self.symmetry_operations:
                c = op.operate(coord)
                inds = find_in_coord_list_pbc(coords, c, atol=self._site_tolerance)
                # can't use if inds, because python is dumb and np.array([0]) evaluates
                # to False
                if len(inds):
                    return keys[inds[0]]
            return False

        for i in range(len(data["_atom_site_label"])):

            try:
                # If site type symbol exists, use it. Otherwise, we use the
                # label.
                symbol = self._parse_symbol(data["_atom_site_type_symbol"][i])
                num_h = get_num_implicit_hydrogens(data["_atom_site_type_symbol"][i])
            except KeyError:
                symbol = self._parse_symbol(data["_atom_site_label"][i])
                num_h = get_num_implicit_hydrogens(data["_atom_site_label"][i])
            if not symbol:
                continue

            if oxi_states is not None:
                o_s = oxi_states.get(symbol, 0)
                # use _atom_site_type_symbol if possible for oxidation state
                if "_atom_site_type_symbol" in data.data.keys():
                    oxi_symbol = data["_atom_site_type_symbol"][i]
                    o_s = oxi_states.get(oxi_symbol, o_s)
                try:
                    el = Species(symbol, o_s)
                except Exception:
                    el = DummySpecies(symbol, o_s)
            else:
                el = get_el_sp(symbol)

            x = self.str2float(data["_atom_site_fract_x"][i])
            y = self.str2float(data["_atom_site_fract_y"][i])
            z = self.str2float(data["_atom_site_fract_z"][i])
            magmom = magmoms.get(data["_atom_site_label"][i], np.array([0, 0, 0]))

            try:
                occu = self.str2float(data["_atom_site_occupancy"][i])
            except (KeyError, ValueError):
                occu = 1

            if occu > 0:
                coord = (x, y, z)
                match = get_matching_coord(coord)
                comp_d = {el: occu}
                if num_h > 0:
                    comp_d["H"] = num_h
                    self.warnings.append(
                        "Structure has implicit hydrogens defined, "
                        "parsed structure unlikely to be suitable for use "
                        "in calculations unless hydrogens added."
                    )
                comp = Composition(comp_d)
                if not match:
                    coord_to_species[coord] = comp
                    coord_to_magmoms[coord] = magmom
                    coord_to_labels[coord] = data["_atom_site_label"][i]
                else:
                    coord_to_species[match] += comp
                    # disordered magnetic not currently supported
                    coord_to_magmoms[match] = None
                    coord_to_labels[coord] += data["_atom_site_label"][i]

        sum_occu = [
            sum(c.values())
            for c in coord_to_species.values()
            if not set(c.elements) == {Element("O"), Element("H")}
        ]
        if any(o > 1 for o in sum_occu):
            msg = (
                "Some occupancies ({}) sum to > 1! If they are within "
                "the occupancy_tolerance, they will be rescaled. "
                "The current occupancy_tolerance is set to: {}".format(
                    sum_occu, self._occupancy_tolerance
                )
            )
            warnings.warn(msg)
            self.warnings.append(msg)

        allspecies = []
        allcoords = []
        allmagmoms = []
        allhydrogens = []

        equivalent_indices = []

        site_properties = {}

        # initialize the wyckoff_sequence here
        site_properties["wyckoff_sequence"] = []

        site_properties["local_index"] = []

        site_properties["label"] = []

        local_index = 0

        # check to see if magCIF file is disordered
        if self.feature_flags["magcif"]:
            for k, v in coord_to_magmoms.items():
                if v is None:
                    # Proposed solution to this is to instead store magnetic
                    # moments as Species 'spin' property, instead of site
                    # property, but this introduces ambiguities for end user
                    # (such as unintended use of `spin` and Species will have
                    # fictitious oxidation state).
                    raise NotImplementedError(
                        "Disordered magnetic structures not currently supported."
                    )
        # print(coord_to_species)
        if coord_to_species.items():
            """
            for idx, (comp, group) in enumerate(
                    sorted(list(coord_to_species.items()), key=lambda x: x[1])
                ):
                print(comp,group)
            """

            for idx, (coord, comp) in enumerate(list(coord_to_species.items())):
                # I delete the weird group function
                # now just for every initial site, generate all sites

                # print(idx,comp)#debug

                tmp_coords = [coord]  # follow the fashion

                tmp_magmom = [coord_to_magmoms[tmp_coord] for tmp_coord in tmp_coords]
                tmp_label = [coord_to_labels[tmp_coord] for tmp_coord in tmp_coords]
                # print(tmp_coords,tmp_magmom,tmp_label)#debug

                if self.feature_flags["magcif"]:
                    coords, magmoms, _ = self._unique_coords(
                        tmp_coords, magmoms_in=tmp_magmom, lattice=lattice
                    )
                else:
                    coords, magmoms, _ = self._unique_coords(tmp_coords)

                # add wyckoff sequence here
                for wyckoff_sequence in range(0, len(coords)):
                    # tuple of wyckoff sequence, (label,sequence ) in format of  (str,int)

                    site_properties["wyckoff_sequence"].append(wyckoff_sequence)
                    site_properties["label"].append(coord_to_labels[tmp_coords[0]])
                    site_properties["local_index"].append(local_index)
                    local_index += 1

                if set(comp.elements) == {Element("O"), Element("H")}:
                    # O with implicit hydrogens
                    im_h = comp["H"]
                    species = Composition({"O": comp["O"]})
                else:
                    im_h = 0
                    species = comp

                # The following might be a more natural representation of equivalent indices,
                # but is not in the format expect by SymmetrizedStructure:
                #   equivalent_indices.append(list(range(len(allcoords), len(coords)+len(allcoords))))
                # The above gives a list like:
                #   [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11]] where the
                # integers are site indices, whereas the version used below will give a version like:
                #   [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
                # which is a list in the same order as the sites, but where if a site has the same integer
                # it is equivalent.
                equivalent_indices += len(coords) * [idx]

                allhydrogens.extend(len(coords) * [im_h])
                allcoords.extend(coords)
                allspecies.extend(len(coords) * [species])
                allmagmoms.extend(magmoms)

            # rescale occupancies if necessary
            for i, species in enumerate(allspecies):
                totaloccu = sum(species.values())
                if 1 < totaloccu <= self._occupancy_tolerance:
                    allspecies[i] = species / totaloccu

        if (
            allspecies
            and len(allspecies) == len(allcoords)
            and len(allspecies) == len(allmagmoms)
        ):
            # site_properties = {}
            if any(allhydrogens):
                assert len(allhydrogens) == len(allcoords)
                site_properties["implicit_hydrogens"] = allhydrogens

            if self.feature_flags["magcif"]:
                site_properties["magmom"] = allmagmoms

            if len(site_properties) == 0:
                site_properties = None

            struct = StructureKMCpy(
                lattice, allspecies, allcoords, site_properties=site_properties
            )

            if symmetrized:

                # Wyckoff labels not currently parsed, note that not all CIFs will contain Wyckoff labels
                # TODO: extract Wyckoff labels (or other CIF attributes) and include as site_properties
                wyckoffs = ["Not Parsed"] * len(struct)

                # Names of space groups are likewise not parsed (again, not all CIFs will contain this information)
                # What is stored are the lists of symmetry operations used to generate the structure
                # TODO: ensure space group labels are stored if present
                sg = SpacegroupOperations("Not Parsed", -1, self.symmetry_operations)

                return SymmetrizedStructure(struct, sg, equivalent_indices, wyckoffs)

            struct = struct.get_sorted_structure()

            if primitive and self.feature_flags["magcif"]:
                struct = struct.get_primitive_structure(use_site_props=True)
            elif primitive:
                struct = struct.get_primitive_structure()
                struct = struct.get_reduced_structure()

            return struct

    def get_labeled_structures(self, primitive=True, symmetrized=False):
        """

        This is a modified version of get_structures for KMCPY, which preserve the label and the wyckoff sequence information.

        2022.03.30


        Return list of structures in CIF file. primitive boolean sets whether a
        conventional cell structure or primitive cell structure is returned.

        Args:
            primitive (bool): Set to False to return conventional unit cells.
                Defaults to True. With magnetic CIF files, will return primitive
                magnetic cell which may be larger than nuclear primitive cell.
            symmetrized (bool): If True, return a SymmetrizedStructure which will
                include the equivalent indices and symmetry operations used to
                create the Structure as provided by the CIF (if explicit symmetry
                operations are included in the CIF) or generated from information
                in the CIF (if only space group labels are provided). Note that
                currently Wyckoff labels and space group labels or numbers are
                not included in the generated SymmetrizedStructure, these will be
                notated as "Not Parsed" or -1 respectively.

        Returns:
            List of Structures.
        """

        if primitive and symmetrized:
            raise ValueError(
                "Using both 'primitive' and 'symmetrized' arguments is not currently supported "
                "since unexpected behavior might result."
            )

        structures = []
        for i, d in enumerate(self._cif.data.values()):
            try:
                s = self._get_labeled_structure(d, primitive, symmetrized)
                if s:
                    structures.append(s)
            except (KeyError, ValueError) as exc:
                # Warn the user (Errors should never pass silently)
                # A user reported a problem with cif files produced by Avogadro
                # in which the atomic coordinates are in Cartesian coords.
                self.warnings.append(str(exc))
                warnings.warn(
                    f"No structure parsed for {i + 1} structure in CIF. Section of CIF file below."
                )
                warnings.warn(str(d))
                warnings.warn(f"Error is {str(exc)}.")

        if self.warnings:
            warnings.warn(
                "Issues encountered while parsing CIF: %s" % "\n".join(self.warnings)
            )
        if len(structures) == 0:
            raise ValueError("Invalid cif file with no structures!")
        return structures

    def _unique_coords(
        self,
        coords,
        magmoms=None,
        lattice=None,
        labels=None,
    ):
        """
        Generate unique coordinates using coord and symmetry positions
        and also their corresponding magnetic moments, if supplied.
        """
        coords_out: list[np.ndarray] = []
        labels_out = []
        labels = labels or {}

        if magmoms:
            magmoms_out = []
            if len(magmoms) != len(coords):
                raise ValueError
            for tmp_coord, tmp_magmom in zip(coords, magmoms):
                for op in self.symmetry_operations:
                    coord = op.operate(tmp_coord)
                    coord = np.array([i - math.floor(i) for i in coord])
                    if isinstance(op, MagSymmOp):
                        # Up to this point, magmoms have been defined relative
                        # to crystal axis. Now convert to Cartesian and into
                        # a Magmom object.
                        magmom = Magmom.from_moment_relative_to_crystal_axes(
                            op.operate_magmom(tmp_magmom), lattice=lattice
                        )
                    else:
                        magmom = Magmom(tmp_magmom)
                    if not in_coord_list_pbc(
                        coords_out, coord, atol=self._site_tolerance
                    ):
                        coords_out.append(coord)
                        magmoms_out.append(magmom)
                        labels_out.append(labels.get(tmp_coord))
            return coords_out, magmoms_out, labels_out

        for tmp_coord in coords:
            for op in self.symmetry_operations:
                coord = op.operate(tmp_coord)
                coord = np.array([i - math.floor(i) for i in coord])
                if not in_coord_list_pbc(coords_out, coord, atol=self._site_tolerance):
                    coords_out.append(coord)
                    labels_out.append(labels.get(tmp_coord))

        dummy_magmoms = [Magmom(0)] * len(coords_out)
        return coords_out, dummy_magmoms, labels_out

    @staticmethod
    def str2float(text):
        """
        Remove uncertainty brackets from strings and return the float.
        """

        try:
            # Note that the ending ) is sometimes missing. That is why the code has
            # been modified to treat it as optional. Same logic applies to lists.
            return float(re.sub(r"\(.+\)*", "", text))
        except TypeError:
            if isinstance(text, list) and len(text) == 1:
                return float(re.sub(r"\(.+\)*", "", text[0]))
        except ValueError as ex:
            if text.strip() == ".":
                return 0
            raise ex
        raise ValueError(f"{text} cannot be converted to float")
