from pathlib import Path
from typing import Any, Mapping, Sequence

from monty.serialization import dumpfn, loadfn
from pymatgen.core.structure import Structure
import numpy as np
from kmcpy.structure.basis import Occupation, get_basis
from abc import ABC
import logging
from kmcpy.structure.species import (
    SiteMapping,
    is_vacancy_species,
    normalize_species,
    species_equivalent,
    species_label,
)

logger = logging.getLogger(__name__) 


class LatticeStructure(ABC):
    '''The lattice of a kMC study: every site and the species it may hold.

    Think of it as the disordered structure: a site that "can be Na or a
    vacancy" is one lattice site with two possible states. A configuration
    (``State``) picks one species per site.

    Build it from a disordered structure or CIF, where partial occupancies
    define the possible species, or from an ordered structure plus
    ``site_mapping``::

        lattice = LatticeStructure.from_cif("Li_xCoO2.cif")          # Li: 0.5 -> Li or X
        lattice = LatticeStructure.from_cif(
            "nasicon.cif", site_mapping={"Na": ["Na", "X"], "Si": ["Si", "P"]}
        )
        lattice.make_supercell((2, 1, 1))
    '''
    def __init__(self, template_structure: Structure,
                 site_mapping: Mapping[Any, Any] | None = None,
                 basis_type: str = 'chebyshev',
                 supercell_shape: Sequence[int] = (1, 1, 1)):
        '''
        Args:
            template_structure: Structure with every site that can be occupied.
                Disordered sites (partial occupancies) define their possible
                species; a site whose occupancies sum to less than 1 can also
                be a vacancy ``"X"``. Ordered sites hold their species.
            site_mapping: Optional ``{species: [allowed species]}`` for the
                species whose sites vary, e.g. ``{"Na": ["Na", "X"]}``.
                Entries override possibilities derived from partial
                occupancies; species that are not listed are fixed. The order
                of the allowed species defines the state indices.
            basis_type: Occupation basis for cluster expansions. For
                'chebyshev', occupations store species-state indices and the
                LCE evaluates q - 1 Chebyshev site functions for q allowed
                species.
            supercell_shape: Repetitions of the template that are simulated
                (see :meth:`make_supercell`).
        '''
        template_structure, site_mapping = _ordered_template_and_mapping(
            template_structure, site_mapping
        )
        self.template_structure = template_structure
        self.supercell_shape = _supercell_shape(supercell_shape)

        mapping = SiteMapping(site_mapping)
        # Plain {label: [labels]} for every template species; fixed species map to themselves.
        self.site_mapping = site_mapping

        # allowed_species is like [["Na","X"],["Na","X"], ... ,["Sb","W"],["Sb","W"]];
        # sites without a site_mapping entry get None.
        self.allowed_species = [
            list(allowed) if allowed is not None else None
            for allowed in mapping.allowed_species_by_site(
                self.template_structure, strict=False
            )
        ]

        # Initialize basis using the registry.
        max_states = max(
            2,
            max(len(species) for species in self.allowed_species if species),
        )
        try:
            if basis_type == "chebyshev":
                self.basis = get_basis(basis_type, max_states=max_states)
            else:
                self.basis = get_basis(basis_type)
            self.basis_type = basis_type
        except ValueError as e:
            raise ValueError(f'Basis type {basis_type} not supported. {e}')

        try:
            self.active_site_order = self.get_active_site_order()
        except ValueError:
            self.active_site_order = None
 
    @classmethod
    def from_cif(
        cls,
        filename: str | Path,
        site_mapping: Mapping[Any, Any] | None = None,
        primitive: bool = False,
        supercell_shape: Sequence[int] = (1, 1, 1),
        basis_type: str = "chebyshev",
    ) -> "LatticeStructure":
        """Load the lattice from a CIF, keeping its site labels.

        Partial occupancies in the CIF define the possible species; see the
        constructor for ``site_mapping``. With ``primitive=True`` the CIF is
        reduced to its primitive cell first.
        """
        from kmcpy.io.cif import load_labeled_structure_from_cif

        structure = load_labeled_structure_from_cif(str(filename), primitive=primitive)
        return cls(
            structure,
            site_mapping,
            basis_type=basis_type,
            supercell_shape=supercell_shape,
        )

    def make_supercell(self, scaling_matrix, in_place: bool = True) -> "LatticeStructure":
        """Repeat the lattice into a supercell, like pymatgen's ``Structure.make_supercell``.

        ``scaling_matrix`` is an int, three ints ``(a, b, c)``, or a diagonal
        3x3 matrix; it multiplies the current ``supercell_shape``. The template
        stays the unit cell. Simulations run on the supercell; site indices in
        events, states, and results refer to its active sites.

        Returns the supercell: this object (``in_place=True``) or a new one.
        """
        matrix = np.array(scaling_matrix, dtype=int)
        if matrix.ndim == 0:
            shape = (int(matrix),) * 3
        elif matrix.shape == (3,):
            shape = tuple(int(value) for value in matrix)
        elif matrix.shape == (3, 3) and np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0:
            shape = tuple(int(value) for value in np.diag(matrix))
        else:
            raise ValueError(
                "scaling_matrix must be an int, three ints, or a diagonal 3x3 matrix; "
                f"got {scaling_matrix!r}"
            )
        shape = _supercell_shape(shape)
        supercell_shape = tuple(a * b for a, b in zip(self.supercell_shape, shape))
        if not in_place:
            return LatticeStructure(
                self.template_structure,
                self.site_mapping,
                basis_type=self.basis_type,
                supercell_shape=supercell_shape,
            )
        self.supercell_shape = supercell_shape
        self.active_site_order = self.get_active_site_order()
        return self

    @property
    def n_active_sites(self) -> int:
        """Number of sites whose occupation can change (in the supercell)."""
        return self.active_site_order.active_site_count

    @property
    def mobile_species(self) -> list[str]:
        """Species whose possible states include a vacancy."""
        return SiteMapping(self.site_mapping).mobile_species()

    def active_structure(self) -> Structure:
        """Supercell structure with only the active sites, in active-site order."""
        return self.active_site_order.active_structure()

    def occupations_from_structure(self, structure: Structure, tol: float = 0.1) -> list[int]:
        """Return active-site occupations (state indices) of an ordered structure.

        ``structure`` is one configuration of this supercell: the same
        lattice, with every atom on a lattice site (within ``tol`` Angstrom)
        and vacancies left out. Fixed sites are ignored.
        """
        order = self.active_site_order
        active = order.active_structure()
        if not np.allclose(structure.lattice.matrix, active.lattice.matrix, atol=tol):
            raise ValueError(
                "The structure's lattice does not match this supercell "
                f"(supercell_shape {self.supercell_shape})."
            )
        states = order.allowed_species_by_active_site
        occupations: list[int | None] = [None] * len(states)
        distances = active.lattice.get_all_distances(structure.frac_coords, active.frac_coords)
        for atom, site_distances in zip(structure, distances):
            site = int(np.argmin(site_distances))
            if site_distances[site] > tol:
                continue
            label = species_label(atom.specie)
            if label not in states[site]:
                raise ValueError(
                    f"{label} is not allowed on active site {site} (allowed: {list(states[site])})"
                )
            occupations[site] = states[site].index(label)
        for site, occupation in enumerate(occupations):
            if occupation is None:
                if "X" not in states[site]:
                    raise ValueError(f"Active site {site} has no atom and no vacancy state")
                occupations[site] = states[site].index("X")
        return occupations

    def structure_from_occupations(self, occupations: Sequence[int]) -> Structure:
        """Return the ordered supercell structure for active-site occupations.

        Vacancies are left out; fixed sites keep their species.
        """
        order = self.active_site_order
        states = order.allowed_species_by_active_site
        if len(occupations) != len(states):
            raise ValueError(
                f"Expected {len(states)} occupations for this supercell, got {len(occupations)}"
            )
        structure = order.full_structure_with_properties()
        vacancies = []
        for site, (original, occupation) in enumerate(zip(order.active_to_original, occupations)):
            label = states[site][int(occupation)]
            if is_vacancy_species(label):
                vacancies.append(original)
            else:
                structure.replace(
                    original,
                    normalize_species(label),
                    properties=structure[original].properties,
                )
        structure.remove_sites(vacancies)
        for name in [key for key in structure.site_properties if key.startswith("_kmcpy_")]:
            structure.remove_site_property(name)
        return structure

    def get_active_site_order(self, supercell_shape=None):
        """Return the compact active-site order (of ``supercell_shape``, default this supercell)."""
        from kmcpy.structure.active_site_order import ActiveSiteOrder

        return ActiveSiteOrder.from_lattice_structure(
            self,
            supercell_shape=supercell_shape if supercell_shape is not None else self.supercell_shape,
        )

    def get_active_lattice_structure(self, supercell_shape=None):
        """Return a lattice structure containing only mutable active sites."""
        active_site_order = self.get_active_site_order(supercell_shape)
        active_lattice_structure = LatticeStructure(
            active_site_order.active_structure(),
            self.site_mapping.copy(),
            self.basis_type,
        )
        active_lattice_structure.source_active_site_order = active_site_order
        return active_lattice_structure

    def get_occ_from_structure(
        self,
        structure: Structure,
        tol=0.1,
        angle_tol=5,
        sc_matrix=None,
        structure_site_mapping=None,
    ) -> Occupation:
        """
        get_occ_from_structure() returns an Occupation object based on a
        comparison with the instance's template_structure.

        The supercell relationship is inferred from lattice vectors unless
        ``sc_matrix`` is provided. Site mapping is inferred from fractional
        coordinates unless ``structure_site_mapping`` is provided.

        Args:
            structure (Structure): The input structure, which may be a supercell
                of the template and may contain vacancies.
            tol (float): Tolerance for structure matching.
            angle_tol (float): Kept for API compatibility.
            sc_matrix (np.ndarray, optional): Supercell matrix if known.
            structure_site_mapping (Sequence[int], optional): Explicit mapping
                from each input structure site to a site in the supercell
                template. If provided, ``structure_site_mapping[j]`` is the
                supercell-template index occupied by ``structure[j]``. Passing
                this skips automatic site matching.

        Returns:
            Occupation: The occupation object for the structure with proper basis.
        """
        # 1. Determine supercell matrix if not provided
        if sc_matrix is None:
            # Attempt to automatically detect supercell matrix
            # Compare lattice vectors to determine the supercell transformation
            template_lattice = self.template_structure.lattice.matrix
            structure_lattice = structure.lattice.matrix

            # Try to solve: structure_lattice = sc_matrix @ template_lattice
            try:
                sc_matrix_candidate = structure_lattice @ np.linalg.inv(template_lattice)

                # Round to nearest integers (supercell matrix should be integer)
                sc_matrix = np.round(sc_matrix_candidate).astype(int)

                # Validate that this is a good supercell matrix
                # Use stricter tolerance for automatic detection
                reconstructed = sc_matrix @ template_lattice
                strict_tol = min(tol, 0.01)  # Use at most 1% tolerance for supercell detection
                if not np.allclose(reconstructed, structure_lattice, rtol=strict_tol, atol=strict_tol):
                    logger.debug("Could not find integer supercell matrix, structures may be incompatible")
                    raise ValueError("No mapping found: cannot find valid supercell transformation")
                else:
                    logger.debug(f"Detected supercell matrix:\n{sc_matrix}")
            except np.linalg.LinAlgError:
                logger.debug("Singular matrix encountered")
                raise ValueError("No mapping found: lattice matrices are incompatible")

        logger.debug(f"Using supercell matrix:\n {sc_matrix}")
        
        # 2. Create the supercell template
        supercell_template = self.template_structure.copy()
        supercell_template.add_site_property(
            "_kmcpy_template_index",
            list(range(len(supercell_template))),
        )
        supercell_template.make_supercell(sc_matrix)
        template_indices = supercell_template.site_properties["_kmcpy_template_index"]
        supercell_allowed_species = [
            self.allowed_species[int(template_index)]
            for template_index in template_indices
        ]
        logger.debug(f"Supercell template has {len(supercell_template)} sites")
        logger.debug(f"Input structure has {len(structure)} sites")
        
        # Initialize missing sites as the vacancy state when available.
        occ_data = np.array(
            [
                self._missing_occupation_value(allowed_species)
                for allowed_species in supercell_allowed_species
            ],
            dtype=type(self.basis.match_value),
        )

        # Handle empty structure case (all sites are vacant)
        if len(structure) == 0:
            logger.debug("Empty structure - all sites are vacant")
            return Occupation(occ_data, basis=self.basis, validate=False)
        
        # 3. Validate structure compatibility
        # Check lattice compatibility
        if not np.allclose(supercell_template.lattice.matrix, structure.lattice.matrix,
                          rtol=tol, atol=tol):
            logger.debug("Lattice mismatch detected")
            raise ValueError("No mapping found: lattice parameters don't match within tolerance")

        if structure_site_mapping is None:
            template_site_indices = self._infer_structure_site_mapping(
                supercell_template,
                structure,
                tol=tol,
            )
        else:
            template_site_indices = np.array(structure_site_mapping, dtype=int)
            if len(template_site_indices) != len(structure):
                raise ValueError(
                    "structure_site_mapping length must match the number of "
                    "input structure sites"
                )
            if (
                len(template_site_indices) > 0
                and (
                    np.min(template_site_indices) < 0
                    or np.max(template_site_indices) >= len(supercell_template)
                )
            ):
                raise ValueError(
                    "structure_site_mapping contains indices outside the "
                    "supercell template"
                )

        logger.debug(f"Structure sites map to template sites: {template_site_indices}")

        if len(set(template_site_indices.tolist())) != len(template_site_indices):
            raise ValueError(
                "No mapping found: multiple atoms map to the same template site"
            )

        # 5. Create occupation vector from species at mapped sites. Missing
        # template sites remain mismatch/vacant.
        for structure_site_index, template_site_index in enumerate(template_site_indices):
            template_site_index = int(template_site_index)
            allowed_species = supercell_allowed_species[template_site_index]
            if not allowed_species:
                raise ValueError(
                    f"No allowed species defined for template site {template_site_index}"
                )

            actual_species = structure[structure_site_index].specie
            try:
                occ_data[template_site_index] = self.occupation_value_for_species(
                    template_site_index,
                    actual_species,
                    allowed_species=allowed_species,
                )
            except ValueError:
                raise ValueError(
                    "No mapping found: species "
                    f"{actual_species} is not allowed at template site "
                    f"{template_site_index}"
                ) from None
        
        logger.debug(f"Occupation vector: {occ_data}")
        
        # Return Occupation object
        return Occupation(occ_data, basis=self.basis, validate=False)

    @staticmethod
    def _infer_structure_site_mapping(
        supercell_template: Structure,
        structure: Structure,
        tol: float,
    ) -> np.ndarray:
        """Infer structure-site to supercell-template indices from fractional coordinates."""
        template_coords = supercell_template.frac_coords
        structure_coords = structure.frac_coords

        # distances[i, j] = minimum-image fractional distance from template site
        # i to structure site j. This keeps the historical tolerance semantics
        # while handling sites close to periodic boundaries.
        deltas = template_coords[:, None, :] - structure_coords[None, :, :]
        deltas -= np.round(deltas)
        distances = np.linalg.norm(deltas, axis=2)

        template_site_indices = np.argmin(distances, axis=0)
        min_distances = np.min(distances, axis=0)
        if np.any(min_distances > tol):
            logger.debug(
                "Some sites exceed tolerance: max distance = %s",
                np.max(min_distances),
            )
            raise ValueError(
                "No mapping found: some atoms are too far from template sites "
                f"(max distance: {np.max(min_distances):.4f}, tolerance: {tol})"
            )
        return template_site_indices

    def _missing_occupation_value(self, allowed_species):
        if not allowed_species:
            return self.basis.mismatch_value
        for state_index, specie in enumerate(allowed_species):
            if is_vacancy_species(specie):
                return self.basis.state_value(state_index, len(allowed_species))
        fallback_state = 1 if len(allowed_species) > 1 else 0
        return self.basis.state_value(fallback_state, len(allowed_species))

    def occupation_value_for_species(
        self,
        site_index: int,
        specie,
        allowed_species=None,
    ):
        """Return the occupation value for a species at a template site."""
        allowed_species = (
            self.allowed_species[int(site_index)]
            if allowed_species is None
            else allowed_species
        )
        if not allowed_species:
            raise ValueError(f"No allowed species defined for site {site_index}")
        for state_index, allowed in enumerate(allowed_species):
            if species_equivalent(specie, allowed):
                return self.basis.state_value(state_index, len(allowed_species))
        raise ValueError(f"Species {specie} is not allowed at site {site_index}")

    def species_for_occupation_value(self, site_index: int, value):
        """Return the allowed species represented by an occupation value."""
        allowed_species = self.allowed_species[int(site_index)]
        if not allowed_species:
            raise ValueError(f"No allowed species defined for site {site_index}")
        for state_index, specie in enumerate(allowed_species):
            if value == self.basis.state_value(state_index, len(allowed_species)):
                return specie
        raise ValueError(f"Unsupported occupation value {value} at site {site_index}")
        
    def copy(self):
        '''Create a copy of the LatticeStructure'''
        return LatticeStructure(self.template_structure.copy(),
                                self.site_mapping.copy(),
                                self.basis_type,
                                supercell_shape=self.supercell_shape)
    
    def __str__(self):
        return f"""LatticeStructure with {len(self.template_structure)} sites
        Template structure:\n {self.template_structure}
        Allowed species: {self.allowed_species}
        Site mapping: {self.site_mapping}
        Basis type: {self.basis_type}"""
    
    def __repr__(self):
        return self.__str__()
    
    def as_dict(self):
        """
        Convert the model object to a dictionary representation.
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "template_structure": self.template_structure.as_dict(),
            "site_mapping": self.site_mapping,
            "basis_type": self.basis_type,
            "supercell_shape": list(self.supercell_shape),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LatticeStructure":
        return cls(
            Structure.from_dict(data["template_structure"]),
            data.get("site_mapping"),
            basis_type=data.get("basis_type", "chebyshev"),
            supercell_shape=tuple(data.get("supercell_shape", (1, 1, 1))),
        )

    def to(self, filename: str | Path) -> None:
        """Write the lattice (including its structure) to a JSON file."""
        dumpfn(self.as_dict(), str(filename), indent=2)

    @classmethod
    def from_file(cls, filename: str | Path) -> "LatticeStructure":
        return cls.from_dict(loadfn(str(filename), cls=None))


def _supercell_shape(values: Sequence[int]) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in values)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("supercell_shape must contain three positive integers")
    return shape


def _ordered_template_and_mapping(
    structure: Structure, site_mapping: Mapping[Any, Any] | None
) -> tuple[Structure, dict[str, list[str]]]:
    """Return an ordered template and a complete ``{label: [labels]}`` site mapping.

    Disordered sites become their first species, and their species (plus a
    vacancy if the occupancies sum to less than 1) become the allowed states
    of that species. ``site_mapping`` entries override those; species left
    unmapped are fixed.
    """
    structure = _with_site_labels(structure)
    derived: dict[str, list[str]] = {}
    if not structure.is_ordered:
        first_species = []
        for site in structure:
            species = list(site.species.items())
            labels = [species_label(specie) for specie, _ in species]
            if sum(amount for _, amount in species) < 1 - 1e-4:
                labels.append("X")
            allowed = derived.setdefault(labels[0], [])
            allowed.extend(label for label in labels if label not in allowed)
            first_species.append(species[0][0])
        structure = Structure(
            structure.lattice,
            first_species,
            structure.frac_coords,
            site_properties=structure.site_properties,
        )

    mapping = dict(derived)
    for key, value in (site_mapping or {}).items():
        values = value if isinstance(value, (list, tuple)) else [value]
        mapping[species_label(normalize_species(key))] = [
            species_label(normalize_species(item)) for item in values
        ]
    for site in structure:
        label = species_label(site.specie)
        if not any(species_equivalent(site.specie, normalize_species(key)) for key in mapping):
            mapping[label] = [label]
    return structure, mapping


def _with_site_labels(structure: Structure) -> Structure:
    """Return ``structure`` with ``label``/``local_index``/``wyckoff_sequence`` site properties.

    Structures loaded from CIF already have them; event generation uses them.
    """
    properties = structure.site_properties
    if {"label", "local_index", "wyckoff_sequence"} <= set(properties):
        return structure
    structure = structure.copy()
    labels = properties.get("label") or [
        species_label(next(iter(site.species))) for site in structure
    ]
    counts: dict[str, int] = {}
    sequence = []
    for label in labels:
        sequence.append(counts.get(label, 0))
        counts[label] = sequence[-1] + 1
    structure.add_site_property("label", labels)
    structure.add_site_property("local_index", list(range(len(structure))))
    structure.add_site_property("wyckoff_sequence", sequence)
    return structure

