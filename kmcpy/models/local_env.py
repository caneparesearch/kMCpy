from pymatgen.core import Structure, PeriodicSite, DummySpecies, Molecule
import numpy as np
import logging
from typing import TYPE_CHECKING
from abc import  abstractmethod

from kmcpy.models import LatticeModel

logger = logging.getLogger(__name__) 
logging.getLogger('pymatgen').setLevel(logging.WARNING)

class LocalEnvironment(LatticeModel):
    """
    Class to handle local environment around a site in a structure.
    """
    def __init__(self, template_structure:Structure, 
                 center, cutoff, 
                 specie_site_mapping=None,
                 basis_type = 'chebyshev',
                 is_write_basis=False, 
                 exclude_species=None):
        
        super().__init__(template_structure=template_structure, specie_site_mapping=specie_site_mapping,
                         basis_type=basis_type)
        self.cutoff = cutoff
        self.is_write_basis = is_write_basis

        if isinstance(center, int):
            self.center_site = template_structure[center]
        elif isinstance(center, list) or isinstance(center, tuple) or isinstance(center, np.array):
            self.center_site = PeriodicSite(species=DummySpecies('X'),
                              coords=center,
                              coords_are_cartesian=False,
                              lattice = self.template_structure.lattice.copy())
            logger.debug(f"Dummy site: {self.center_site}")
        else:
            raise ValueError("Center must be an index or a list of fractional coordinates.")
        template_structure.remove_oxidation_states()  # Remove oxidation states if present
        if exclude_species:
            template_structure.remove_species(exclude_species)

        local_env_sites = template_structure.get_sites_in_sphere(self.center_site.coords, cutoff, include_index=True)
        
        # Sort by species name
        local_env_sites.sort(key=lambda x: x[0].species_string)

        self.site_indices = [site[2] for site in local_env_sites]
        
        local_env_structure_sites = [site[0] for site in local_env_sites]

        local_env_structure = Molecule.from_sites(local_env_structure_sites)
        local_env_structure.translate_sites(np.arange(0, len(local_env_structure), 1).tolist(), -1 * self.center_site.coords)
        if is_write_basis:
            from pymatgen.symmetry.analyzer import PointGroupAnalyzer
            logger.info("Local environment: ")
            logger.info(local_env_structure)
            local_env_structure.to(fmt="xyz", filename="local_env.xyz")
            logger.info(
            "The point group of local environment is: %s",
            PointGroupAnalyzer(local_env_structure).sch_symbol,
            )
        self.structure = local_env_structure