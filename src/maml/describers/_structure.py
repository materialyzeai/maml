"""Structure-wise describers. These describers include structural information."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from maml.base import BaseDescriber, describer_type

if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure

logger = logging.getLogger(__name__)


__all__ = [
    "CoulombEigenSpectrum",
    "CoulombMatrix",
    "DistinctSiteProperty",
    "RandomizedCoulombMatrix",
    "SortedCoulombMatrix",
]


@describer_type("structure")
class DistinctSiteProperty(BaseDescriber):
    r"""
    Constructs a describers based on properties of distinct sites in a
    structure. For now, this assumes that there is only one type of species in
    a particular Wyckoff site.

    Reference:
    @article{ye2018deep,
            title={Deep neural networks for accurate predictions of crystal stability},
            author={Ye, Weike and Chen, Chi and Wang, Zhenbin and
                Chu, Iek-Heng and Ong, Shyue Ping},
            journal={Nature communications},
            volume={9},
            number={1},
            pages={1--6},
            year={2018},
            publisher={Nature Publishing Group}}
    """

    # todo: generalize to multiple sites with the same Wyckoff.
    supported_properties = [
        "mendeleev_no",
        "electrical_resistivity",
        "velocity_of_sound",
        "reflectivity",
        "refractive_index",
        "poissons_ratio",
        "molar_volume",
        "thermal_conductivity",
        "boiling_point",
        "melting_point",
        "critical_temperature",
        "superconduction_temperature",
        "liquid_range",
        "bulk_modulus",
        "youngs_modulus",
        "brinell_hardness",
        "rigidity_modulus",
        "mineral_hardness",
        "vickers_hardness",
        "density_of_solid",
        "atomic_radius_calculated",
        "van_der_waals_radius",
        "coefficient_of_linear_thermal_expansion",
        "ground_state_term_symbol",
        "valence",
        "Z",
        "X",
        "atomic_mass",
        "block",
        "row",
        "group",
        "atomic_radius",
        "average_ionic_radius",
        "average_cationic_radius",
        "average_anionic_radius",
        "metallic_radius",
        "ionic_radii",
        "oxi_state",
        "max_oxidation_state",
        "min_oxidation_state",
        "is_transition_metal",
        "is_alkali",
        "is_alkaline",
        "is_chalcogen",
        "is_halogen",
        "is_lanthanoid",
        "is_metal",
        "is_metalloid",
        "is_noble_gas",
        "is_post_transition_metal",
        "is_quadrupolar",
        "is_rare_earth_metal",
        "is_actinoid",
    ]

    def __init__(
        self,
        properties: list[str],
        symprec: float = 0.1,
        wyckoffs: list[str] | None = None,
        feature_batch: str = "pandas_concat",
        **kwargs,
    ):
        """

        Args:
            properties (list): Sequence of specie properties. E.g.,
                ["atomic_radius"]. Look at pymatgen.core.periodic_table.Element and
                pymatgen.core.periodic_table.Species for support properties (there
                are a lot!)
            symprec (float): Symmetry precision for spacegroup determination.
            wyckoffs (list of wyckoff symbols):. E.g., ["48a", "24c"], if not provided,
                will get from input structure when doing transform
            feature_batch (str): way to batch a list of features into one
            **kwargs: keyword args to specify memory, verbose, and n_jobs.
        """
        self.properties = properties
        self.symprec = symprec
        self.wyckoffs = wyckoffs
        super().__init__(feature_batch=feature_batch, **kwargs)

    def transform_one(self, structure: Structure) -> pd.DataFrame:
        """

        Args:
            structure (pymatgen Structure): pymatgen structure for descriptor computation.

        Returns:
            pd.DataFrame that contains the distinct position labeled features

        """
        a = SpacegroupAnalyzer(structure, self.symprec)
        symm = a.get_symmetrized_structure()
        data = []
        names = []
        if self.wyckoffs is None:
            self.wyckoffs = symm.wyckoff_symbols
        for w in self.wyckoffs:
            site = symm.equivalent_sites[symm.wyckoff_symbols.index(w)][0]
            for p in self.properties:
                data.append(getattr(site.specie, p))
                names.append(f"{w}-{p}")
        return pd.DataFrame([data], columns=names)


@describer_type("structure")
class CoulombMatrix(BaseDescriber):
    r"""
    Coulomb Matrix to describe structure.

    Reference:
    @article{rupp2012fast,
            title={Fast and accurate modeling of molecular
                atomization energies with machine learning},
            author={Rupp, Matthias and Tkatchenko, Alexandre and M{\"u}ller,
                Klaus-Robert and Von Lilienfeld, O Anatole},
            journal={Physical review letters}, volume={108},
            number={5}, pages={058301},
            year={2012}, publisher={APS}}
    """

    def __init__(self, random_seed: int | None = None, max_atoms: int | None = None, is_ravel: bool = True, **kwargs):
        """
        Args:
            random_seed (int): random seed
            max_atoms (int): maximum number of atoms
            is_ravel (bool): whether to ravel the matrix to 1D
            **kwargs: keyword args to specify memory, verbose, and n_jobs.
        """
        self.max_atoms = max_atoms
        self.random_seed = random_seed
        self.is_ravel = is_ravel
        if "feature_batch" not in kwargs:
            kwargs["feature_batch"] = "pandas_concat"
        super().__init__(**kwargs)

    @staticmethod
    def _get_coulomb_mat(s: Molecule | Structure) -> np.ndarray:
        """
        Args:
            s (Molecule/Structure): input Molecule or Structure. Structure
                is not advised since the feature will depend on the supercell size.

        Returns:
            Coulomb matrix of the structure

        """
        dis = s.distance_matrix
        np.fill_diagonal(dis, np.inf)  # avoid dividing by zero
        zs = np.array([site.specie.Z for site in s])
        z_matrix = zs[:, None] * zs[None, :]
        z_diag = 0.5 * zs**2.4
        c = z_matrix / dis
        np.fill_diagonal(c, z_diag)
        return c

    def get_coulomb_mat(self, s: Molecule | Structure) -> np.ndarray:
        """
        Args:
            s (Molecule/Structure): input Molecule or Structure. Structure
                is not advised since the feature will depend on the supercell size
        Returns:
            Coulomb matrix of the structure.

        """
        c = self._get_coulomb_mat(s)
        num_sites = c.shape[0]
        if self.max_atoms is not None and self.max_atoms > num_sites:
            padding = self.max_atoms - num_sites
            return np.pad(c, (0, padding), mode="constant", constant_values=0)
        return c

    def transform_one(self, s: Molecule | Structure) -> pd.DataFrame:
        """
        Args:
            s (Molecule/Structure): pymatgen Molecule or Structure, Structure is not
                advised since the features will depend on supercell size.

        Returns:
            pandas.DataFrame.
            The column is index of the structure, which is 0 for single input
            df[0] returns the serials of coulomb_mat raval
        """
        c = self.get_coulomb_mat(s)
        if self.is_ravel:
            c = c.ravel()
        return pd.DataFrame(c)


@describer_type("structure")
class RandomizedCoulombMatrix(CoulombMatrix):
    r"""
    Randomized CoulombMatrix.

    Reference:
    @article{montavon2013machine,
            title={Machine learning of molecular electronic properties
                in chemical compound space},
            author={Montavon, Gr{\'e}goire and Rupp, Matthias and Gobre,
                Vivekanand and Vazquez-Mayagoitia, Alvaro and Hansen, Katja
                and Tkatchenko, Alexandre and M{\"u}ller, Klaus-Robert and
                Von Lilienfeld, O Anatole},
            journal={New Journal of Physics},
            volume={15}, number={9},pages={095003},
            year={2013},publisher={IOP Publishing}}
    """

    def __init__(self, random_seed: int | None = None, is_ravel: bool = True, **kwargs):
        """
        Args:
            random_seed (int): random seed
            is_ravel (bool): Passthrough to is_ravel for CoulombMatrix. Defaults to True.
            **kwargs: keyword args to specify memory, verbose, and n_jobs.
        """
        super().__init__(random_seed=random_seed, is_ravel=is_ravel, **kwargs)

    def get_randomized_coulomb_mat(self, s: Molecule | Structure) -> pd.DataFrame:
        """
        Returns the randomized matrix
        (i) take an arbitrary valid Coulomb matrix C
        (ii) compute the norm of each row of this Coulomb matrix: row_norms
        (iii) draw a zero-mean unit-variance noise vector ε of the same
            size as row_norms.
        (iv)  permute the rows and columns of C with the same permutation
            that sorts row_norms + ε.

        Args:
            s (Molecule/Structure): pymatgen Molecule or Structure, Structure is not
                advised since the features will depend on supercell size

        Returns:
            pd.DataFrame randomized Coulomb matrix
        """
        c = self.get_coulomb_mat(s)
        row_norms = np.linalg.norm(c, axis=1)
        rng = np.random.default_rng(self.random_seed)
        e = rng.normal(size=row_norms.size)
        p = np.argsort(row_norms + e)
        c = c[p][:, p]
        if self.is_ravel:
            c = c.ravel()
        return pd.DataFrame(c)

    def transform_one(self, s: Molecule | Structure) -> pd.DataFrame:
        """
        Transform one structure to descriptors
        Args:
            s (Molecule/Structure): pymatgen Molecule or Structure, Structure is not
                advised since the features will depend on supercell size.

        Returns: pandas dataframe descriptors

        """
        return self.get_randomized_coulomb_mat(s)


@describer_type("structure")
class SortedCoulombMatrix(CoulombMatrix):
    r"""
    Sorted CoulombMatrix.

    Reference:
    @inproceedings{montavon2012learning,
                title={Learning invariant representations
                    of molecules for atomization energy prediction},
                author={Montavon, Gr{\'e}goire and Hansen, Katja
                    and Fazli, Siamac and Rupp, Matthias and Biegler,
                    Franziska and Ziehe, Andreas and Tkatchenko, Alexandre
                    and Lilienfeld, Anatole V and M{\"u}ller, Klaus-Robert},
                booktitle={Advances in neural information processing systems},
                pages={440--448}, year={2012}}
    """

    def __init__(self, random_seed: int | None = None, is_ravel: bool = True, **kwargs):
        """
        Args:
            random_seed (int): random seed
            is_ravel (bool): Passthrough to is_ravel for CoulombMatrix. Defaults to True.
            **kwargs: keyword args to specify memory, verbose, and n_jobs.
        """
        super().__init__(random_seed=random_seed, is_ravel=is_ravel, **kwargs)

    def get_sorted_coulomb_mat(self, s: Molecule | Structure) -> pd.DataFrame:
        """
        Returns the matrix sorted by the row norm.

        Args:
            s (Molecule/Structure): pymatgen Molecule or Structure, Structure is not
                advised since the features will depend on supercell size

        Returns:
            pd.DataFrame, sorted Coulomb matrix
        """
        c = self.get_coulomb_mat(s)
        c = c[np.argsort(np.linalg.norm(c, axis=1))]
        if self.is_ravel:
            c = c.ravel()
        return pd.DataFrame(c)

    def transform_one(self, s: Molecule | Structure) -> pd.DataFrame:
        """
        Transform one structure into descriptor
        Args:
            s (Molecule/Structure): pymatgen Molecule or Structure, Structure is not
                advised since the features will depend on supercell size.

        Returns: pd.DataFrame descriptors

        """
        return self.get_sorted_coulomb_mat(s)


@describer_type("structure")
class CoulombEigenSpectrum(BaseDescriber):
    r"""
    Get the Coulomb Eigen Spectrum describers.

    Reference:
    @article{rupp2012fast,
            title={Fast and accurate modeling of molecular
                atomization energies with machine learning},
            author={Rupp, Matthias and Tkatchenko, Alexandre and M{\"u}ller,
                Klaus-Robert and Von Lilienfeld, O Anatole},
            journal={Physical review letters}, volume={108},
            number={5}, pages={058301},
            year={2012}, publisher={APS}}
    """

    def __init__(self, max_atoms: int | None = None, **kwargs):
        """
        This method calculates the Coulomb matrix of a molecule and
        then sort the eigen values of the Coulomb matrix as the vector
        features for the molecule.
        When multiple molecules are converted at the same time, the
        describers will stack the results. If the number of atoms is
        not the same for molecules, zeros will padded to the features.

        Args:
            max_atoms (int): maximum number of atoms
            **kwargs: Passthrough to parent class __init__.
        """
        feature_batch = "stack_padded" if max_atoms is None else "stack_first_dim"
        self.max_atoms = max_atoms
        super().__init__(feature_batch=feature_batch, **kwargs)

    def transform_one(self, mol: Molecule) -> np.ndarray:
        """
        Args:
            mol (Molecule): pymatgen molecule.

        Returns: np.ndarray the eigen value vectors of Coulob matrix

        """
        c_mat = CoulombMatrix._get_coulomb_mat(mol)
        eig_vals = np.linalg.eigvals(c_mat)
        eig_vals = np.abs(eig_vals)

        f = np.sort(eig_vals)[::-1]
        if self.max_atoms is not None:
            if self.max_atoms < len(f):
                raise RuntimeError("max_atoms is smaller than the size of current molecule")
            f = np.pad(f, (0, self.max_atoms - len(f)))
        return f
