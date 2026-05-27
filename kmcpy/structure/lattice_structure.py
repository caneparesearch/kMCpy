from pymatgen.core.structure import Structure
import numpy as np
from kmcpy.structure.basis import Occupation, get_basis
from abc import ABC
import logging
from kmcpy.structure.species import (
    is_vacancy_species,
    normalize_species,
    species_equivalent,
    species_tokens,
)

logger = logging.getLogger(__name__) 


class LatticeStructure(ABC):
    '''LatticeStructure deal with the structure template which converts the structure to an occupation array and vice versa
    '''
    def __init__(self, template_structure: Structure,
                 site_mapping: dict,
                 basis_type: str = 'chebyshev'):
        '''Initialization of LatticeStructure
            Args:
            template_structure: pymatgen Structure object, this should include all possible sites (no doping, vacancy etc.)
            site_mapping: a dictionary mapping template species to allowed species; fixed sites have a single allowed species,
            e.g. {"Na":["Na","X"],"X":["Na","X"],"Sb":["Sb","W"],"W":["Sb","W"]} X is the vacancy site
            basis_type: str, the type of basis function. For 'chebyshev',
                occupations store species-state indices and the LCE evaluates
                q - 1 Chebyshev site functions for q allowed species.
        '''
        self.template_structure = template_structure

        self.site_mapping = {
            normalize_species(key): [
                normalize_species(item)
                for item in (value if isinstance(value, (list, tuple)) else [value])
            ]
            for key, value in site_mapping.items()
        }
                
        # Initialization of species for LatticeStructure
        # allowed_species is like [["Na","Va"],["Na","Va"],["Na","Va"],["Na","Va"], ... ,["Sb","W"],["Sb","W"],["Sb","W"]]
        self.allowed_species = []
        for site in self.template_structure:
            allowed_specie = None
            for mapped_species, mapped_allowed_species in self.site_mapping.items():
                if species_equivalent(site.specie, mapped_species):
                    allowed_specie = mapped_allowed_species
                    break
            self.allowed_species.append(allowed_specie)

        if len(self.allowed_species) != len(self.template_structure):
            raise ValueError(f"Species length {len(self.allowed_species)} does not match template structure length {len(self.template_structure)}!")

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
            self.active_site_index_map = self.get_active_site_index_map()
        except ValueError:
            self.active_site_index_map = None
 
    def get_active_site_index_map(self, supercell_shape=None):
        """Return the compact active-site index map for this lattice."""
        from kmcpy.structure.active_site_index_map import ActiveSiteIndexMap

        return ActiveSiteIndexMap.from_lattice_structure(
            self, supercell_shape=supercell_shape
        )

    def get_active_lattice_structure(self, supercell_shape=None):
        """Return a lattice structure containing only mutable active sites."""
        active_site_index_map = self.get_active_site_index_map(supercell_shape)
        active_lattice_structure = LatticeStructure(
            active_site_index_map.active_structure(),
            self.site_mapping.copy(),
            self.basis_type,
        )
        active_lattice_structure.source_active_site_index_map = active_site_index_map
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

    @staticmethod
    def _species_matches(actual, expected) -> bool:
        return species_equivalent(actual, expected)

    @staticmethod
    def _is_vacancy_species(specie) -> bool:
        return is_vacancy_species(specie)

    @staticmethod
    def _species_tokens(specie) -> set[str]:
        return species_tokens(specie)

    def _missing_occupation_value(self, allowed_species):
        if not allowed_species:
            return self.basis.mismatch_value
        for state_index, specie in enumerate(allowed_species):
            if self._is_vacancy_species(specie):
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
            if self._species_matches(specie, allowed):
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
        
    def get_structure_from_occ(self, occ: Occupation, sc_matrix=None) -> Structure:
        '''get_structure_from_occ() takes an Occupation object and returns a pymatgen Structure
        
        Args:
            occ: Occupation object containing site occupation data
            sc_matrix: Supercell matrix for creating the supercell
            
        Returns:
            Structure: pymatgen Structure with species assigned based on match/mismatch
        '''
        if sc_matrix is None:
            sc_matrix = np.eye(3, dtype=int)
        else:
            sc_matrix = np.array(sc_matrix, dtype=int)

        supercell_lattice_structure = self.copy()
        supercell_lattice_structure.make_supercell(sc_matrix)
        
        if len(occ) != len(supercell_lattice_structure.template_structure):
            raise ValueError(f"Occupation array length {len(occ)} does not match template structure length {len(supercell_lattice_structure.template_structure)}!")
        
        # Create a new structure based on the template
        new_lattice_structure = supercell_lattice_structure.copy()
        
        # Iterate through the sites and set species based on occupation
        for i, site in enumerate(new_lattice_structure.template_structure):
            occ_value = occ[i]  # Get occupation value at site i
            site.species = new_lattice_structure.species_for_occupation_value(
                i,
                occ_value,
            )
        
        # Remove vacancy sites in reverse order to avoid index shifting issues
        vacancy_indices = []
        for i, site in enumerate(new_lattice_structure.template_structure):
            if is_vacancy_species(site.specie):
                vacancy_indices.append(i)
        
        # Remove vacancy sites from the end to avoid index issues
        for i in reversed(vacancy_indices):
            new_lattice_structure.template_structure.remove_sites([i])

        return new_lattice_structure.template_structure

    def copy(self):
        '''Create a copy of the LatticeStructure'''
        return LatticeStructure(self.template_structure.copy(),
                                self.site_mapping.copy(),
                                self.basis_type)
    
    def make_supercell(self, sc_matrix: np.ndarray):
        '''Create a supercell of the template structure'''
        self.template_structure.make_supercell(sc_matrix)
        # Update allowed_species accordingly
        original_allowed_species = self.allowed_species.copy()
        self.allowed_species = []
        for i in range(sc_matrix[0,0]):
            for j in range(sc_matrix[1,1]):
                for k in range(sc_matrix[2,2]):
                    self.allowed_species.extend(original_allowed_species)
        if len(self.allowed_species) != len(self.template_structure):
            raise ValueError(f"After supercell, species length {len(self.allowed_species)} does not match template structure length {len(self.template_structure)}!")
        
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
            "template_structure": self.template_structure.as_dict(),
            "site_mapping": self.site_mapping,
            "basis_type": self.basis_type
        }
