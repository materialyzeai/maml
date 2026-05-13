"""
Radial distribution functions for site features.
This was originally written in pymatgen-diffusion.
"""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    from pymatgen.core import Structure


class RadialDistributionFunction:
    """Calculator for radial distribution function."""

    def __init__(
        self,
        r_min: float = 0.0,
        r_max: float = 10.0,
        n_grid: int = 101,
        sigma: float = 0.0,
    ):
        """
        Fast radial distribution analysis. This method calculates
        rdf on `np.linspace(r_min, r_max, n_grid)` points.

        Args:
            r_min (float): minimal radius
            r_max (float): maximal radius
            n_grid (int): number of grid points, defaults to 101
            sigma (float): smooth parameter.
        """
        self.r_min = r_min
        self.r_max = r_max
        self.n_grid = n_grid

        self.dr = (self.r_max - self.r_min) / (self.n_grid - 1)  # end points are on grid
        self.r = np.linspace(self.r_min, self.r_max, self.n_grid)

        self.cutoff = self.r_max + self.dr / 2.0  # add a small shell to improve robustness
        self.sigma = ceil(sigma / self.dr)
        self.volumes = 4.0 * np.pi * self.r**2 * self.dr
        self.volumes[self.volumes < 1e-8] = 1e8  # avoid divide by zero

    def get_site_rdf(self, structure: Structure) -> tuple[np.ndarray, list[dict]]:
        """
        Args:
            structure (Structure): pymatgen structure
        Returns:
            r, rdfs, r is the radial points, and rdfs are a list of rdf dicts
            rdfs[0] is the rdfs of the first site. It is a dictionary of {atom_pair: pair_rdf}
            e.g.,
                {"Sr:O": [0, 0, 0.1, 0.2, ..]}.
        """
        pair_distances = get_pair_distances(structure, self.cutoff)
        density = self._get_specie_density(structure)

        # return full rdf information for site

        rdfs: list[dict] = [{}] * len(structure)
        for i, neighbors in enumerate(pair_distances):
            if len(neighbors["neighbors"]) == 0:
                continue
            c_specie = neighbors["specie"]
            temp_neighbors = neighbors["neighbors"]

            rdfs[i] = {
                f"{c_specie}:{specie}": _dist_to_counts(
                    temp_neighbors[specie],
                    r_min=self.r_min,
                    r_max=self.r_max,
                    n_grid=self.n_grid,
                )
                / self.volumes
                / density[specie]
                for specie in temp_neighbors
            }

            if self.sigma > 1e-8:
                rdfs[i] = {key: gaussian_filter1d(rdfs[i][key], self.sigma) for key in rdfs[i]}
        return self.r, rdfs

    def get_species_rdf(
        self,
        structure: Structure,
        ref_species: list | None = None,
        species: list | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get specie-wise rdf
        Args:
            structure (Structure): target structure
            ref_species (list of species or just single specie str): the reference species.
                The rdfs are calculated with these species at the center
            species (list of species or just single specie str): the species that we are interested in.
                The rdfs are calculated on these species.

        Returns: 2-D array of distances and rdf intensities.
        """
        all_species = list({str(i.specie) for i in structure.sites})
        density = self._get_specie_density(structure)
        n_atoms = structure.composition.to_data_dict["unit_cell_composition"]
        if ref_species is None:
            ref_species = all_species
        if species is None:
            species = all_species

        pair_distances = get_pair_distances(structure, self.cutoff)
        all_distances = []
        for _i, neighbors in enumerate(pair_distances):
            if neighbors["specie"] not in ref_species:
                continue

            for n_specie, distances in neighbors["neighbors"].items():
                if n_specie not in species:
                    continue
                all_distances.append(distances)

        if len(all_distances) == 0:
            return self.r, np.zeros_like(self.r)

        all_counts = [_dist_to_counts(d, r_min=self.r_min, r_max=self.r_max, n_grid=self.n_grid) for d in all_distances]
        sum_counts = np.sum(all_counts, axis=0)
        total_density = sum(density[i] for i in species)
        total_atoms = sum(n_atoms[i] for i in ref_species)
        rdf_temp = sum_counts / total_density / self.volumes / total_atoms
        if self.sigma > 1e-8:
            rdf_temp = gaussian_filter1d(rdf_temp, self.sigma)
        return self.r, rdf_temp

    def get_site_coordination(self, structure: Structure) -> tuple[np.ndarray, list[dict]]:
        """
        Get site wise coordination
        Args:
            structure (Structure): pymatgen Structure.

        Returns: r, cns where cns is a list of dictionary with specie_pair: pair_cn key:value pairs

        """
        _, rdfs = self.get_site_rdf(structure)
        density = self._get_specie_density(structure)
        cns: list[dict] = [{}] * len(structure)
        for i, rdf in enumerate(rdfs):
            if len(rdf) == 0:
                continue
            for pair, rdf_pair in rdf.items():
                _, specie = pair.split(":")
                cns[i][pair] = np.cumsum(rdf_pair * density[specie] * 4.0 * np.pi * self.r**2 * self.dr)
        return self.r, cns

    def get_species_coordination(
        self,
        structure: Structure,
        ref_species: list | None = None,
        species: list | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get specie-wise coordination number
        Args:
            structure (Structure): target structure
            ref_species (list of species or just single specie str): the reference species.
                The rdfs are calculated with these species at the center
            species (list of species or just single specie str): the species that we are interested in.
                The rdfs are calculated on these species.

        Returns: 2-D array of distances and respective specie-wise coordination number.
        """
        all_species = list({str(i.specie) for i in structure.sites})
        if ref_species is None:
            ref_species = all_species
        if species is None:
            species = all_species

        _, rdf = self.get_species_rdf(structure=structure, ref_species=ref_species, species=species)

        density = self._get_specie_density(structure)
        n_atoms = structure.composition.to_data_dict["unit_cell_composition"]
        total_density = sum(density[i] for i in species)
        total_atoms = sum(n_atoms[i] for i in ref_species)
        return self.r, np.cumsum(rdf * total_density * 4.0 * np.pi * self.r**2 * self.dr * total_atoms)

    @staticmethod
    def _get_specie_density(structure: Structure):
        n_atoms = structure.composition.to_data_dict["unit_cell_composition"]
        density = {}
        for i, j in n_atoms.items():
            density[i] = j / structure.volume
        return density


def _dist_to_counts(d: np.ndarray, r_min: float = 0.0, r_max: float = 8.0, n_grid: int = 100) -> np.ndarray:
    """
    Convert a distance array for counts in the bin
    Args:
        d (1D np.ndarray): distance array
        r_min (float): minimal radius
        r_max (float): maximum radius
    Returns:
        1D array of counts in the bins centered on grid.
    """
    counts = np.zeros((n_grid,))
    dr = (r_max - r_min) / (n_grid - 1)  # end points are on grid
    indices = np.array(np.floor((d - r_min + 0.5 * dr) / dr), dtype=int)

    unique, val_counts = np.unique(indices, return_counts=True)
    counts[unique] = val_counts
    return counts


def get_pair_distances(structure: Structure, r_max: float = 8.0) -> list[dict]:
    """
    Get pair distances from structure.
    The output will be a list of dictionary, for example
    [{"specie": "Mo",
    "neighbors": {"S": [1.0, 2.0, ...], "Fe": [1.2, 3.0, ...]}},
    {"specie": "Fe",
    "neighbors": {"Mo": [1.0, 3.0, ...]}}]
    it will be fairly easy to construct radial distribution func, etc
    from here.

    Args:
        structure (Structure): pymatgen Structure
        r_max (float): maximum radius to consider

    Returns: a list of dictionary with pair distance information of each specie.
    """
    index1, index2, _, distances = structure.get_neighbor_list(r_max)
    species = np.array([str(i.specie) for i in structure.sites])
    res = [{"specie": i, "neighbors": {}} for i in species]
    neighbor_species = species[index2]
    tuples = np.array(list(zip(index1, neighbor_species, strict=False)), dtype=[("index", "i4"), ("specie", "<U10")])
    unique_tuples, _indices = np.unique(tuples, return_inverse=True)
    for _index, unique_tuple in enumerate(unique_tuples):
        res[unique_tuple[0]]["neighbors"][unique_tuple[1]] = distances[tuples == unique_tuple]
    return res
