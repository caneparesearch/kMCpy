#!/usr/bin/env python
"""
Code to build the structural model for local cluster expansion
Author: Zeyu Deng
Email: dengzeyu@gmail.com
"""
from kmcpy.model import LocalClusterExpansion


def main():
    local_cluster_expansion = LocalClusterExpansion()
    local_cluster_expansion.initialization(center_Na1_index=3,is_write_basis=True,cutoff_cluster=[6.5,4.8,0.0],cutoff_region=5)

    local_cluster_expansion.get_correlation_matrix_neb_cif('different_envs/*.cif')

    # Write representative clusters of all orbits as .xyz file
    local_cluster_expansion.write_representative_clusters('./orbits')

    local_cluster_expansion.to_json('local_cluster_expansion.json')

if __name__ == "__main__":
    main()