"""Module implements the describers for GB entry."""

from __future__ import annotations

from functools import reduce
from math import gcd

import numpy as np
import pandas as pd
from monty.json import MSONable
from pymatgen.analysis.local_env import (
    BrunnerNN_real,
    BrunnerNN_reciprocal,
    BrunnerNN_relative,
    CovalentBondNN,
    Critic2NN,
    CrystalNN,
    CutOffDictNN,
    EconNN,
    JmolNN,
    MinimumDistanceNN,
    MinimumOKeeffeNN,
    MinimumVIRENN,
    NearNeighbors,
    OpenBabelNN,
    VoronoiNN,
)
from pymatgen.core import Element, Structure
from pymatgen.core.interface import GrainBoundary
from pymatgen.ext.matproj import MPRester, MPRestError
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import maml.apps.gbe.presetfeatures as preset
from maml.apps.gbe.utils import load_b0_dict, load_mean_delta_bl_dict
from maml.base import BaseDescriber


def convert_hcp_direction(rotation_axis: list, lat_type: str) -> np.ndarray:
    """
    Four index notion to three index notion for hcp and rhombohedral axis
    Args:
        rotation_axis (list): four index notion of axis
        lat_type(str): the
    Returns:
        rotation axis in three index notion.
    """
    u1 = rotation_axis[0]
    v1 = rotation_axis[1]
    w1 = rotation_axis[3]
    if lat_type.lower().startswith("h"):
        u = 2 * u1 + v1
        v = 2 * v1 + u1
        w = w1
        rotation_axis = [u, v, w]
    elif lat_type.lower().startswith("r"):
        u = 2 * u1 + v1 + w1
        v = v1 + w1 - u1
        w = w1 - 2 * v1 - u1
        rotation_axis = [u, v, w]

    # make sure gcd(rotation_axis)==1
    if reduce(gcd, rotation_axis) != 1:
        rotation_axis = [round(x / reduce(gcd, rotation_axis)) for x in rotation_axis]
    return np.array(rotation_axis)


def convert_hcp_plane(plane: list) -> np.ndarray:
    """
    Four index notion to three index notion for hcp and rhombohedral plane
    Args:
        plane (list):  four index notion.

    Returns:
        three index notion of plane

    """
    u1 = plane[0]
    v1 = plane[1]
    w1 = plane[3]
    plane = [u1, v1, w1]
    if reduce(gcd, plane) != 1:
        index = reduce(gcd, plane)
        plane = [round(x / index) for x in plane]
    return np.array(plane)


class GBDescriber(BaseDescriber):
    """
    The describers that describes the grain boundary db entry
    with selected structural and elemental features.
    """

    def __init__(
        self,
        structural_features: list | None = None,
        elemental_features: list | None = None,
        **kwargs,
    ):
        """

        Args:
            structural_features (list): list of structural features
            elemental_features (list): list of elemental features
            **kwargs (dict): parameters for BaseDescriber.
        """
        if not elemental_features:
            elemental_features = [
                preset.e_coh,
                preset.G,
                preset.a0,
                preset.ar,
                preset.mean_delta_bl,
                preset.mean_bl,
            ]
        if not structural_features:
            structural_features = [
                preset.d_gb,
                preset.d_rot,
                preset.sin_theta,
                preset.cos_theta,
            ]
        self.elem_features = elemental_features
        self.struc_features = structural_features
        super().__init__(**kwargs)

    def transform_one(
        self,
        db_entry: dict,
        inc_target: bool = True,
        inc_bulk_ref: bool = True,
        mp_api: str | None = None,
    ) -> pd.DataFrame:
        """
        Describe gb with selected structural and elemental features
        Args:
            db_entry (dict): entry from surfacedb.All_GB_properties_parallel_copy
            inc_target (bool): whether to include target in the df, default: False
            inc_bulk_ref (bool): whether to generate bulk reference
                bulk reference: i.e. the entry of the origin bulk of the GB,
                                the rotation angle (theta) = 0, gb_energy = 0
            mp_api (str): MP api key.


        Returns: pd.DataFrame of processed data, columns are the feature labels
                 The task_id is included and serve as the unique id of the data

        """
        structural = get_structural_feature(db_entry=db_entry, features=self.struc_features)
        elemental = get_elemental_feature(db_entry=db_entry, features=self.elem_features, mp_api=mp_api)
        df = pd.concat([structural, elemental], axis=1, join="inner")
        df["task_id"] = db_entry["task_id"]
        if inc_target:
            df[preset.e_gb.str_name] = db_entry["gb_energy"]
        if inc_bulk_ref:
            df_bulk = self.generate_bulk_ref(df, inc_target)
            return pd.concat([df, df_bulk], axis=0)
        return df

    def generate_bulk_ref(self, gb_df: pd.DataFrame, inc_target: bool = True) -> pd.DataFrame:
        """
        Generate the bulk reference for given gb entry
        Args:
            gb_df (pd.DataFrame): data for gb
            inc_target (bool) : whether to include target.

        Returns:
            bulk_df (pd.DataFrame): data for bulk

        """
        new_df = gb_df.copy()
        new_df[preset.sin_theta.str_name] = np.sin(0)
        new_df[preset.cos_theta.str_name] = np.cos(0)
        if preset.mean_delta_bl.str_name in new_df:
            new_df[preset.mean_delta_bl.str_name] = 0
        if inc_target:
            new_df[preset.e_gb.str_name] = 0
        return new_df


def get_structural_feature(db_entry: dict, features: list | None = None) -> pd.DataFrame:
    """
    The structural features:
    d_gb: interplanal distance of the gb_plane
    d_rot: interplanal distance of the gb_plane
        w/ smallest integer index ) normal to rotation axis
    theta: rotation angle (sin and cos).

    Args:
        db_entry (Dict): db entry
        features (List): list of features.


    Returns:
        pd.DataFrame of structural features

    """
    if features is None:
        features = [preset.d_gb, preset.d_rot, preset.sin_theta, preset.cos_theta]
    if isinstance(db_entry["bulk_conv"], dict):
        bulk_conv = Structure.from_dict(db_entry["bulk_conv"])
    else:
        bulk_conv = db_entry["bulk_conv"]
    sg = SpacegroupAnalyzer(bulk_conv)
    gb_plane = np.array(db_entry["gb_plane"])
    rotation_axis = db_entry["rotation_axis"]
    if gb_plane.shape[0] == 4:
        gb_plane = convert_hcp_plane(list(gb_plane))
        rotation_axis = convert_hcp_direction(rotation_axis, lat_type=sg.get_crystal_system())
    d_gb = bulk_conv.lattice.d_hkl(gb_plane)  # type: ignore[arg-type]
    d_rot = bulk_conv.lattice.d_hkl(rotation_axis)
    theta = db_entry["rotation_angle"]
    sin_theta = np.sin(theta * np.pi / 180)
    cos_theta = np.cos(theta * np.pi / 180)
    fvalues = []
    for f in features:
        if f == preset.sin_theta:
            fvalues.append(np.sin(theta * np.pi / 180))
        elif f == preset.cos_theta:
            fvalues.append(np.cos(theta * np.pi / 180))
        else:
            fvalues.append(locals()[f"{f.str_name}"])
    return pd.DataFrame([fvalues], columns=[f.str_name for f in features])


def get_elemental_feature(
    db_entry: dict,
    loc_algo: str = "crystalnn",
    features: list | None = None,
    mp_api: str | None = None,
) -> pd.DataFrame:
    """
    Function to get the elemental features
    Args:
        db_entry (Dict): db entry
        loc_algo(str): algorithm to determine local env.
                    Options: see GBBond.NNDict.keys()
        features(List): list of feature names
                e_coh: cohesive energy
                G: G_vrh shear modulus
                a0: bulk lattice parameter a
                ar: atomic radius
                b0: the bond length of the metal bulk
                mean_delta_bl: the mean bond length difference
                    between GB and the bulk
                bl: the mean bond length in GB
        mp_api(str): api key to MP.

    Returns:
        pd.DataFrame of elemental features
    """
    if features is None:
        features = [
            preset.e_coh,
            preset.G,
            preset.a0,
            preset.ar,
            preset.mean_delta_bl,
            preset.bdensity,
            preset.CLTE,
            preset.hb,
        ]
    if preset.mean_delta_bl in features or preset.mean_bl in features:
        mdbl = load_mean_delta_bl_dict(loc_algo=loc_algo)
    f_dict = {}
    if mp_api:
        rester = MPRester(mp_api)
    else:
        raise MPRestError("Please provide API key to access Materials Project")
    bulk = rester.materials.search(material_ids=[db_entry["material_id"]])[0]  # type: ignore[attr-defined]
    bulk_s = rester.get_structure_by_material_id(db_entry["material_id"])  # type: ignore[attr-defined]
    if bulk:
        f_dict[preset.bdensity.str_name] = bulk[0]["density"]
        f_dict[preset.G.str_name] = bulk[0]["elasticity"]["G_VRH"]
    el = Element(db_entry["pretty_formula"])
    f_dict[preset.ar.str_name] = el.atomic_radius
    f_dict[preset.a0.str_name] = bulk_s.lattice.a
    f_dict[preset.e_coh.str_name] = rester.get_cohesive_energy(db_entry["material_id"])  # type: ignore[attr-defined,misc]
    f_dict[preset.hb.str_name] = el.brinell_hardness
    f_dict[preset.CLTE.str_name] = el.coefficient_of_linear_thermal_expansion

    def get_mean_delta_bl(db_entry):
        return mdbl[str(db_entry["task_id"])]

    def get_bl(db_entry):
        b0_dict = load_b0_dict()
        b0 = b0_dict[db_entry["pretty_formula"]]
        return mdbl[str(db_entry["task_id"])] + b0

    fvalues = []
    for f in features:
        if f == preset.mean_delta_bl:
            # func = 'get_{}'.format(f.str_name).lower()
            # fvalues.append(locals()[func](db_entry))
            fvalues.append(get_mean_delta_bl(db_entry))
        elif f == preset.mean_bl:
            fvalues.append(get_bl(db_entry))
        else:
            fvalues.append(f_dict[f.str_name])
    return pd.DataFrame([fvalues], columns=[f.str_name for f in features])


class GBBond(MSONable):
    """
    This class describes the GB Bonding environment using provided
    local environment algorithm.
    Available algorithms: GBBond.NNDict.keys(), default CrystalNN.
    """

    NNDict = {
        i.__name__.lower(): i
        for i in [
            NearNeighbors,
            VoronoiNN,
            JmolNN,
            MinimumDistanceNN,
            OpenBabelNN,
            CovalentBondNN,
            MinimumVIRENN,
            MinimumOKeeffeNN,
            BrunnerNN_reciprocal,
            BrunnerNN_real,
            BrunnerNN_relative,
            EconNN,
            CrystalNN,
            CutOffDictNN,
            Critic2NN,
        ]
    }

    def __init__(
        self,
        gb: GrainBoundary,
        loc_algo: str = "crystalnn",
        bond_mat: np.ndarray | None = None,
        **kwargs,
    ):
        """

        Args:
            gb (GrainBoundary): the GrainBoundary Object
            loc_algo (str): algorithm to determine local env.
                    See options: GBBond.NNDict.keys()
                    Default: crystalnn
            bond_mat (np.ndarray): optional.
            **kwargs: pass to loc_algo.
        """
        self.loc_algo = self.NNDict[loc_algo](**kwargs)
        self.gb = gb
        if isinstance(bond_mat, np.ndarray):
            self.bond_mat = bond_mat
        else:
            self.bond_mat = self._get_bond_mat(gb)

    def _get_bond_mat(self, gb: GrainBoundary) -> np.ndarray:
        """

        Args:
            gb (GrainBoundary): the grain boundary structure object.

        Returns:
            bond matrix
            matrix of bond lengths
            1. bm[i][j] = bond length between atom i&j
            if i & j is bonded (determined by the loc_algo)
            2. bm[i][j] = bm[j][i]
            3. If not bonded, the bm[i][j] = np.nan

        """
        if isinstance(gb, dict):
            gb = GrainBoundary.from_dict(gb)
        bond_mat = np.zeros((gb.num_sites, gb.num_sites))
        all_nn_info = self.loc_algo.get_all_nn_info(gb)
        for i, _ in enumerate(gb.sites):
            # compute the all_nn_info first then call
            # the protected function is faster for multiple computation
            nn = self.loc_algo._get_nn_shell_info(structure=gb, site_idx=i, shell=1, all_nn_info=all_nn_info)
            for b in nn:
                bl = gb.get_distance(i, b["site_index"])
                bond_mat[i][b["site_index"]] = bl
        return bond_mat

    def get_breakbond_ratio(self, gb: GrainBoundary, b0: float, return_count: bool = False):
        """
        Get the breakbond ratio, i.e the ratio of shorter bonds vs longer bonds
        compared to the bulk bond length (b0)
        The algo to find the bonds can vary
        Meme: if use get_neighbors, require a hard set cutoff, which adds to
            arbitrariness.

        Args:
            gb (GrainBoundary): a GrainBoundary object
            b0 (float): cutoff to determine short vs. long bonds,
                        default the bulk bond length
            return_count(bool): whether to return count of

        Returns:
            ratio (float): shorter_bond / longer_bonds
            if return_count:
            shorter_bond: # of short bonds
            longer_bond: # of long bonds
        """
        if not self.bond_mat:
            self._get_bond_mat(gb)
        long_bond = (self.bond_mat > b0).sum()
        short_bond = (self.bond_mat < b0).sum()
        ratio = short_bond / long_bond if long_bond else 0
        if return_count:
            return ratio, short_bond, long_bond
        return ratio

    @property
    def min_bl(self) -> float:
        """The minimum bond length."""
        return self.bond_mat[self.bond_mat > 0].min()

    @property
    def max_bl(self) -> float:
        """The maximum bond length."""
        return self.bond_mat.max()

    @property
    def bond_matrix(self) -> np.ndarray:
        """The (padded) bond matrix."""
        return self.bond_mat

    def get_mean_bl_chg(self, b0: float) -> float:
        """
        Function to calculate the mean bond length difference between GB and the bulk
        Args:
            b0 (float): the bond length in bulk.

        Returns: the mean_bl_chg

        """
        return (self.bond_mat[self.bond_mat > 0] - b0).mean()

    def as_dict(self) -> dict:
        """
        Dict representation of the GBond class.

        Returns:
            dict of {"loc_algo": str, "bond_mat": bond matrix}
        """
        return_dict = super().as_dict()
        return_dict.update(
            {
                "loc_algo": self.loc_algo.__class__.__name__.lower(),
                "bond_mat": self.bond_mat,
                # "min_bl": self.min_bl,
                # "max_bl": self.max_bl
            }
        )
        return return_dict

    @classmethod
    def from_dict(cls, d):
        """

        Args:
            d (dict): Dict representation.

        Returns:
            GBBond

        """
        return cls(gb=d["gb"], loc_algo=d["loc_algo"], bond_mat=d["bond_mat"])
