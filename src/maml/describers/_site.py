"""This module provides local environment describers."""

from __future__ import annotations

import itertools
import logging
import re
import subprocess
from shutil import which

import numpy as np
import pandas as pd
from monty.io import zopen
from monty.tempfile import ScratchDir
from pymatgen.core import Composition, Element, Molecule, Structure
from pymatgen.core.periodic_table import get_el_sp

from maml.base import BaseDescriber, describer_type
from maml.utils import pool_from, to_composition

__all__ = [
    "BPSymmetryFunctions",
    "BispectrumCoefficients",
    "SiteElementProperty",
    "SmoothOverlapAtomicPosition",
]


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@describer_type("site")
class BispectrumCoefficients(BaseDescriber):
    r"""
    Bispectrum coefficients to describe the local environment of each atom.
    Lammps is required to perform this computation.

    Reference:
    @article{bartok2010gaussian,
             title={Gaussian approximation potentials: The
                accuracy of quantum mechanics, without the electrons},
             author={Bart{\'o}k, Albert P and Payne, Mike C
                and Kondor, Risi and Cs{\'a}nyi, G{\'a}bor},
             journal={Physical review letters},
             volume={104}, number={13}, pages={136403}, year={2010}, publisher={APS}}
    """

    def __init__(
        self,
        rcutfac: float,
        twojmax: int,
        element_profile: dict,
        quadratic: bool = False,
        pot_fit: bool = False,
        include_stress: bool = False,
        feature_batch: str = "pandas_concat",
        **kwargs,
    ):
        """
        Args:
            rcutfac (float): The rcutfac used in computing the cutoff between
                elements. true cutoff between i, j = rcutfac * (Ri + Rj),
                where Ri and Rj are cutoff for elements i and j
            twojmax (int): Band limit for bispectrum components.
            element_profile (dict): Parameters (cutoff radius 'r' and weight 'w')
                related to each element, e.g.,
                {'Na': {'r': 4.5, 'w': 0.9},
                 'Cl': {'r': 5.0, 'w': 3.0}}
            quadratic (bool): Whether including quadratic terms.
                Default to False.
            pot_fit (bool): Whether combine the dataframe for potential fitting.
            include_stress (bool): Whether to include stress components.
            feature_batch: way to batch together a list of features
            **kwargs: keyword args to specify memory, verbose, and n_jobs.
        """
        from maml.apps.pes import SpectralNeighborAnalysis

        self.calculator = SpectralNeighborAnalysis(
            rcutfac=rcutfac, twojmax=twojmax, element_profile=element_profile, quadratic=quadratic
        )
        self.rcutfac = rcutfac
        self.twojmax = twojmax
        self.elements = sorted(element_profile.keys(), key=Element)
        self.element_profile = element_profile
        self.quadratic = quadratic
        self.pot_fit = pot_fit
        self.include_stress = include_stress
        super().__init__(feature_batch=feature_batch, **kwargs)

    @property
    def subscripts(self) -> list:
        """
        The subscripts (2j1, 2j2, 2j) of all bispectrum components
        involved.
        """
        return self.calculator.get_bs_subscripts(self.twojmax)

    def transform_one(self, structure: Structure) -> pd.DataFrame:
        """
        Args:
            structure (Structure): Pymatgen Structure object.
        """
        columns = ["-".join([str(i) for i in s]) for s in self.subscripts]
        if self.quadratic:
            columns += [
                "-".join([f"{i}{j}{k}" for i, j, k in s])
                for s in itertools.combinations_with_replacement(self.subscripts, 2)
            ]

        raw_data = self.calculator.calculate([structure])

        def process(output, combine):
            b, db, vb, e = output
            df = pd.DataFrame(b, columns=columns)
            if combine:
                df_add = pd.DataFrame({"element": e, "n": np.ones(len(e))})
                df_b = df_add.join(df)
                n_atoms = df_b.shape[0]
                b_by_el = [df_b[df_b["element"] == e] for e in self.elements]
                sum_b = [df[df.columns[1:]].sum(axis=0) for df in b_by_el]
                hstack_b = pd.concat(sum_b, keys=self.elements)
                hstack_b = hstack_b.to_frame().T / n_atoms
                hstack_b = hstack_b.fillna(0)
                dbs = np.split(db, len(self.elements), axis=1)
                dbs = np.hstack([np.insert(d.reshape(-1, len(columns)), 0, 0, axis=1) for d in dbs])
                db_index = [f"{i}_{d}" for i in df_b.index for d in "xyz"]
                df_db = pd.DataFrame(dbs, index=db_index, columns=hstack_b.columns)
                if self.include_stress:
                    vbs = np.split(vb.sum(axis=0), len(self.elements))
                    vbs = np.hstack([np.insert(v.reshape(-1, len(columns)), 0, 0, axis=1) for v in vbs])
                    volume = structure.volume
                    vbs = vbs / volume * 160.21766208  # from eV to GPa
                    vb_index = ["xx", "yy", "zz", "yz", "xz", "xy"]
                    df_vb = pd.DataFrame(vbs, index=vb_index, columns=hstack_b.columns)
                    df = pd.concat([hstack_b, df_db, df_vb])
                else:
                    df = pd.concat([hstack_b, df_db])
            return df

        return process(raw_data[0], self.pot_fit)

    @property
    def feature_dim(self) -> int | None:
        """Bispectrum feature dimension."""
        n = 0
        for i in range(self.twojmax + 1):
            for j in range(i + 1):
                for k in range(j - i, min(self.twojmax, i + j) + 1, 2):
                    if k >= i:
                        n += 1
        if self.pot_fit:
            n += 1
        n_elements = len(self.elements)
        n *= n_elements
        return n


@describer_type("site")
class SmoothOverlapAtomicPosition(BaseDescriber):
    r"""
    Smooth overlap of atomic positions (SOAP) to describe the local environment
    of each atom.

    Reference:
    @article{bartok2013representing,
             title={On representing chemical environments},
             author={Bart{\'o}k, Albert P and Kondor, Risi and Cs{\'a}nyi, G{\'a}bor},
             journal={Physical Review B},
             volume={87}, number={18}, pages={184115}, year={2013}, publisher={APS}}
    """

    def __init__(
        self,
        cutoff: float,
        l_max: int = 8,
        n_max: int = 8,
        atom_sigma: float = 0.5,
        feature_batch: str = "pandas_concat",
        **kwargs,
    ):
        """

        Args:
            cutoff (float): Cutoff radius.
            l_max (int): The band limit of spherical harmonics basis function.
                Default to 8.
            n_max (int): The number of radial basis function. Default to 8.
            atom_sigma (float): The width of gaussian atomic density. Default to 0.5.
            feature_batch (str): way to batch together a list of features
            **kwargs: keyword args to specify memory, verbose, and n_jobs.
        """
        from maml.apps.pes import GAPotential

        self.operator = GAPotential()
        self.cutoff = cutoff
        self.l_max = l_max
        self.n_max = n_max
        self.atom_sigma = atom_sigma
        super().__init__(feature_batch=feature_batch, **kwargs)

    def transform_one(self, structure: Structure) -> pd.DataFrame:
        """
        Args:
            structure (Structure): Pymatgen Structure object.
        """
        if not which("quip"):
            raise RuntimeError(
                "quip has not been found.\n", "Please refer to https://github.com/libAtoms/QUIP for ", "further detail."
            )

        atoms_filename = "structure.xyz"

        exe_command = ["quip"]
        exe_command.append(f"atoms_filename={atoms_filename}")

        descriptor_command = ["soap"]
        descriptor_command.append(f"cutoff={self.cutoff}")
        descriptor_command.append(f"l_max={self.l_max}")
        descriptor_command.append(f"n_max={self.n_max}")
        descriptor_command.append(f"atom_sigma={self.atom_sigma}")

        atomic_numbers = [str(element.number) for element in sorted(np.unique(structure.species))]
        n_Z = len(atomic_numbers)
        n_species = len(atomic_numbers)
        Z = "{" + " ".join(atomic_numbers) + "}"
        species_Z = "{" + " ".join(atomic_numbers) + "}"
        descriptor_command.append(f"n_Z={n_Z}")
        descriptor_command.append(f"Z={Z}")
        descriptor_command.append(f"n_species={n_species}")
        descriptor_command.append(f"species_Z={species_Z}")

        exe_command.append("descriptor_str={" + " ".join(descriptor_command) + "}")

        with ScratchDir("."):
            _ = self.operator.write_cfgs(filename=atoms_filename, cfg_pool=pool_from([structure]))
            descriptor_output = "output"
            with open(descriptor_output, "w") as f, subprocess.Popen(exe_command, stdout=f) as p:
                stdout = p.communicate()[0]
                rc = p.returncode
            if rc != 0:
                error_msg = f"quip/soap exited with return code {rc}"
                msg = stdout.decode("utf-8").split("\n")[:-1]
                try:
                    error_line = next(i for i, m in enumerate(msg) if m.startswith("ERROR"))
                    error_msg += ", ".join(msg[error_line:])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)

            with zopen(descriptor_output, "rt") as f:  # type: ignore
                lines = f.read()

            descriptor_pattern = re.compile("DESC(.*?)\n", re.S)
            return pd.DataFrame([np.array(c.split(), dtype=np.float64) for c in descriptor_pattern.findall(lines)])


@describer_type("site")
class BPSymmetryFunctions(BaseDescriber):
    r"""
    Behler-Parrinello symmetry function to describe the local environment
    of each atom.

    Reference:
    @article{behler2007generalized,
            title={Generalized neural-network representation of
                high-dimensional potential-energy surfaces},
            author={Behler, J{\"o}rg and Parrinello, Michele},
            journal={Physical review letters},
            volume={98},
            number={14},
            pages={146401},
            year={2007},
            publisher={APS}}
    """

    def __init__(
        self,
        cutoff: float,
        r_etas: np.ndarray,
        r_shift: np.ndarray,
        a_etas: np.ndarray,
        zetas: np.ndarray,
        lambdas: np.ndarray,
        feature_batch: str = "pandas_concat",
        **kwargs,
    ):
        """
        Args:
            cutoff (float): The cutoff distance.
            r_etas (numpy.ndarray): η in radial function.
            r_shift (numpy.ndarray): Rs in radial function.
            a_etas (numpy.ndarray): η in angular function.
            zetas (numpy.ndarray): ζ in angular function.
            lambdas (numpy.ndarray): λ in angular function. Default to (1, -1).
            feature_batch: str = 'pandas_concat',
            **kwargs: keyword args to specify memory, verbose, and n_jobs.
        """
        self.cutoff = cutoff
        self.r_etas = np.array(r_etas)[None, :, None]
        self.r_shift = np.array(r_shift)[None, None, :]
        self.a_etas = np.array(a_etas)[None, :, None, None]
        self.zetas = np.array(zetas)[None, None, :, None]
        self.lambdas = np.array(lambdas)[None, None, None, :]
        super().__init__(feature_batch=feature_batch, **kwargs)

    def transform_one(self, structure: Structure) -> pd.DataFrame:
        """
        Args:
            structure (Structure): Pymatgen Structure object.
        """
        elements = sorted(structure.symbol_set, key=lambda sym: get_el_sp(sym).X)
        all_neighbors = structure.get_all_neighbors(self.cutoff)
        data = []

        for atom_idx, neighbors in enumerate(all_neighbors):
            center = structure[atom_idx].coords
            site_symmfuncs: list[np.ndarray] = []
            sorted_neighbors = sorted(neighbors, key=lambda neighbor: neighbor.species_string)
            temp_dict = {
                element: [(neighbor.coords, neighbor.nn_distance) for neighbor in group]
                for element, group in itertools.groupby(sorted_neighbors, key=lambda neighbor: neighbor.species_string)
            }
            for specie in elements:
                distances = np.array([nn_distance for _, nn_distance in temp_dict[specie]])[:, None, None]
                g2 = np.sum(
                    np.exp(-self.r_etas * (distances - self.r_shift) ** 2) * self._fc(distances),  # type:ignore[arg-type]
                    axis=0,  # type: ignore
                )
                site_symmfuncs.extend(g2.ravel().tolist())

            for specie1, specie2 in itertools.combinations_with_replacement(elements, 2):
                group1, group2 = temp_dict[specie1], temp_dict[specie2]

                c = (
                    itertools.combinations(range(len(group1)), 2)
                    if specie1 == specie2
                    else itertools.product(range(len(group1)), range(len(group2)))
                )
                index_combination = np.array(list(c)).T.tolist()

                coords_group1 = np.array([coords for coords, _ in group1])
                coords_group2 = np.array([coords for coords, _ in group2])
                distances_group1 = np.array([nn_distance for _, nn_distance in group1])
                distances_group2 = np.array([nn_distance for _, nn_distance in group2])

                v1 = coords_group1[index_combination[0]]
                v2 = coords_group2[index_combination[1]]
                d1 = distances_group1[index_combination[0]]
                d2 = distances_group2[index_combination[1]]
                d3 = np.linalg.norm(v1 - v2, axis=1)
                cosines = np.sum((v1 - center) * (v2 - center), axis=1) / (d1 * d2)
                cosines = cosines[:, None, None, None]
                distances = np.stack((d1, d2, d3))
                cutoffs = np.prod(self._fc(distances), axis=0)  # type: ignore
                cutoffs = np.atleast_1d(cutoffs)[:, None, None, None]
                powers = np.sum(distances**2, axis=0)[:, None, None, None]
                g4 = np.sum(
                    (1 + self.lambdas * cosines) ** self.zetas
                    * np.exp(-self.a_etas * powers)
                    * cutoffs
                    * 2.0 ** (1 - self.zetas),
                    axis=0,
                )
                site_symmfuncs.extend(g4.ravel().tolist())

            data.append(site_symmfuncs)
        return pd.DataFrame(data)

    def _fc(self, r: float) -> np.ndarray:
        """
        Cutoff function to decay the symmetry functions at vicinity of radial cutoff.

        Args:
            r (float): The pair distance.
        """
        return 0.5 * (np.cos(np.pi * r / self.cutoff) + 1) * (r <= self.cutoff)


@describer_type("site")
class SiteElementProperty(BaseDescriber):
    """
    Site specie property describers. For a structure or composition, return
    an unordered set of site specie properties.
    """

    def __init__(self, feature_dict: dict | None = None, output_weights: bool = False, **kwargs):
        """
        Args:
            feature_dict (dict): mapping from atomic number of feature vectors
            output_weights (bool): whether to output the site fraction,
                used in disordered compositions.
            **kwargs: passthrough.
        """
        self.feature_dict = feature_dict
        self.output_weights = output_weights
        super().__init__(feature_batch=None, **kwargs)

    @staticmethod
    def _get_keys(c: Composition) -> tuple[list[int], list[float]]:
        d = {str(i): j for i, j in c._data.items()}
        str_z = {str(i): i.Z for i in c.elements}
        elements = list(d.keys())
        z_values: list[int] = [str_z[i] for i in elements]
        weights: list[float] = [d[i] for i in elements]
        return z_values, weights

    def transform_one(
        self,
        obj: str | Composition | Structure | Molecule,  # type: ignore
    ) -> list[np.ndarray] | np.ndarray:
        """
        Transform one object to features.

        Args:
            obj (str/Composition/Structure/Molecule): object to transform

        Returns:
            features array
        """
        if isinstance(obj, Structure | Molecule):
            keys = []
            weights = []
            for i in obj:
                d = i.species._data
                for k, w in d.items():
                    keys.append(k.Z)
                    weights.append(w)
        else:
            comp = to_composition(obj)
            keys, weights = self._get_keys(comp)

        n = len(keys)

        features = [self.feature_dict[i] for i in keys] if self.feature_dict is not None else keys  # type: ignore
        features = np.reshape(features, (n, -1))  # type: ignore
        weights = np.reshape(weights, (n,))  # type: ignore
        if self.output_weights:
            return [features, weights]  # type: ignore

        int_weights = weights.astype(int)  # type: ignore
        if not np.allclose(int_weights, weights):
            raise ValueError(
                "Number of atoms are not integers and the describers"
                " cannot output single feature matrix. Try set "
                "output_weights = True"
            )
        return np.repeat(features, int_weights, axis=0)

    @property
    def feature_dim(self):
        """Feature dimension."""
        if self.feature_dict is None:
            return None
        key = next(iter(self.feature_dict.keys()))
        return np.array(self.feature_dict[key]).size
