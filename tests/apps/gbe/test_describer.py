from __future__ import annotations

import os
import shutil
import unittest
from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Element, Lattice, Structure
from pymatgen.core.interface import GrainBoundary, GrainBoundaryGenerator
from pymatgen.ext.matproj import MPRester, MPRestError
from pymatgen.util.testing import PymatgenTest

import maml.apps.gbe as gbe
import maml.apps.gbe.describer as describer_mod
from maml.apps.gbe.describer import (
    GBBond,
    GBDescriber,
    convert_hcp_direction,
    convert_hcp_plane,
    get_elemental_feature,
    get_structural_feature,
)
from maml.apps.gbe.presetfeatures import (
    CLTE,
    G,
    a0,
    ar,
    bdensity,
    cos_theta,
    d_gb,
    d_rot,
    e_coh,
    e_gb,
    hb,
    mean_bl,
    mean_delta_bl,
    sin_theta,
)
from maml.apps.gbe.utils import load_b0_dict, load_data, load_mean_delta_bl_dict

pjoin = os.path.join
module_dir = os.path.dirname(gbe.__file__)
REFS = pjoin(module_dir, "references")


class TestDescriber(PymatgenTest):
    def setUp(self) -> None:
        self.db_entries = [d for d in load_data() if d["material_id"] == "mp-91"]  # Contains 10 data
        self.b0_dict = load_b0_dict()

    def test_convert_hcp(self):
        assert np.allclose(convert_hcp_plane((2, -1, -1, 0)), (2, -1, 0))
        assert np.allclose(convert_hcp_direction((2, -1, -1, 0), "hexagonal"), (1, 0, 0))
        assert np.allclose(convert_hcp_direction((-1, -1, 2, 0), "hexagonal"), (-1, -1, 0))

    @unittest.skipUnless(os.getenv("MAPI_KEY"), "skip")
    def test_elemental_feature(self):
        # test case: W, theta 109.47122063449069 degree,
        df = get_elemental_feature(self.db_entries[0], mp_api=os.getenv("MAPI_KEY"))
        ans_dict = {
            e_coh: 8.301059,
            G: 147,
            a0: 2.760381,
            ar: 1.35,
            mean_delta_bl: -0.004265,
            hb: 2570.0,
            CLTE: 4.5 * 1e-6,
            bdensity: 18.854007561757314,
        }
        for k, v in ans_dict.items():
            assert df[k.str_name][0] == pytest.approx(v, 6)

    def test_structural_feature(self):
        df = get_structural_feature(self.db_entries[0])
        ans_dict = {
            d_gb: 1.0079488085907933,
            d_rot: 1.3012563165005835,
            sin_theta: 0.9428090415820635,
            cos_theta: -0.33333333333333315,
        }
        for k, v in ans_dict.items():
            assert df[k.str_name][0] == pytest.approx(v, 6)

    @unittest.skipUnless(os.getenv("MAPI_KEY"), "skip")
    def test_describer(self):
        describer = GBDescriber()
        df = describer.transform_one(
            self.db_entries[0], inc_target=True, inc_bulk_ref=True, mp_api=os.getenv("MAPI_KEY")
        )
        df_gb = df.iloc[0].to_frame().transpose()
        df_bulk = df.iloc[1].to_frame().transpose()
        ans_dict = {
            e_coh: 8.301059,
            G: 147,
            ar: 1.35,
            mean_delta_bl: -0.004265,
            d_gb: 1.0079488085907933,
            d_rot: 1.3012563165005835,
            sin_theta: 0.9428090415820635,
            cos_theta: -0.33333333333333315,
            "task_id": 5094,
            e_gb: 0.665409,
        }
        for k, v in ans_dict.items():
            if k == "task_id":
                assert df_gb["task_id"][0] == 5094
            else:
                assert df_gb[k.str_name][0] == pytest.approx(v, 6)

        ans_dict_bulk = {
            e_coh: 8.301059,
            G: 147,
            a0: 2.760381,
            ar: 1.35,
            mean_delta_bl: 0,
            # "breakbond_ratio": 4 / 328,
            d_gb: 1.0079488085907933,
            d_rot: 1.3012563165005835,
            sin_theta: np.sin(0),
            cos_theta: np.cos(0),
            "task_id": 5094,
            e_gb: 0,
        }
        for k, v in ans_dict_bulk.items():
            if k == "task_id":
                assert df_bulk["task_id"][0] == 5094
            else:
                assert df_bulk[k.str_name][0] == pytest.approx(v, 6)


class TestGBBond(PymatgenTest):
    def setUp(self) -> None:
        self.bulk = Structure.from_file(pjoin(REFS, "mp-91.cif"))
        self.gb_entry = next(d for d in load_data() if d["task_id"] == 5094)
        self.gb = self.gb_entry["initial_structure"]
        self.gbond = GBBond(self.gb, loc_algo="crystalnn")

    def test_loc_algo(self):
        assert len(self.gbond.loc_algo.get_nn_shell_info(structure=self.bulk, site_idx=0, shell=1)) == 8
        # With a warning of 'cannot locate an appropriate radius' "

    def test_bond_mat(self):
        bond_mat = self.gbond.bond_mat
        ans = np.array(
            [
                [
                    0.0,
                    2.76269883,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                    0.0,
                    2.76269957,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                    0.0,
                ],
                [
                    2.76269883,
                    0.0,
                    0.0,
                    0.0,
                    2.76270187,
                    0.0,
                    0.0,
                    2.76270031,
                    0.0,
                    2.76269893,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    2.76269883,
                    0.0,
                    0.0,
                    2.76269893,
                    0.0,
                    2.76270031,
                    0.0,
                    2.7627004,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    2.76269883,
                    0.0,
                    2.76269883,
                    2.76270031,
                    2.76270031,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    2.76270187,
                    0.0,
                    2.76269883,
                    0.0,
                    2.76269893,
                    0.0,
                    2.76270031,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                    2.76269893,
                    0.0,
                    2.60470048,
                    2.76269883,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    2.76269893,
                    2.76270031,
                    0.0,
                    2.60470048,
                    0.0,
                    0.0,
                    2.76269883,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    2.76270031,
                    0.0,
                    0.0,
                    2.76270031,
                    2.76269883,
                    0.0,
                    0.0,
                    0.0,
                    2.76269883,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    2.76270031,
                    0.0,
                    0.0,
                    0.0,
                    2.76269883,
                    0.0,
                    0.0,
                    0.0,
                    2.76269957,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                ],
                [
                    2.76270031,
                    2.76269893,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76269883,
                    0.0,
                    0.0,
                    0.0,
                    2.7627004,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    2.7627004,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76269957,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.7627004,
                ],
                [
                    2.76269957,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.7627004,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                    2.7627004,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76269883,
                    0.0,
                    0.0,
                    2.76270187,
                    2.76270031,
                    2.76269893,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                    0.0,
                    2.76269883,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76269957,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76269883,
                    2.76269883,
                    0.0,
                    0.0,
                    2.76270031,
                    2.76270031,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76269883,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76269893,
                    0.0,
                    2.76270031,
                    2.7627004,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76270187,
                    0.0,
                    2.76269883,
                    0.0,
                    0.0,
                    2.76270031,
                    0.0,
                    0.0,
                    2.76269893,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                    0.0,
                    2.76269883,
                    0.0,
                    2.76269883,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76269893,
                    2.76270031,
                    0.0,
                    0.0,
                    0.0,
                    2.76269883,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.7627004,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                    2.76269893,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.60470048,
                    2.76269883,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                    0.0,
                    2.76269893,
                    2.76269883,
                    0.0,
                    2.60470048,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                    0.0,
                    0.0,
                    0.0,
                    2.76269883,
                    0.0,
                    0.0,
                    2.76269957,
                    0.0,
                ],
                [
                    2.76270031,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.7627004,
                    0.0,
                    0.0,
                    0.0,
                    2.7627004,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76269957,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.76270031,
                    0.0,
                    2.7627004,
                    0.0,
                    0.0,
                    2.76269957,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.7627004,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )
        assert np.allclose(bond_mat, ans)
        dist_mat = self.gb.distance_matrix
        min_bl = (dist_mat[dist_mat > 0]).min()
        assert self.gbond.min_bl == min_bl

    def test_mean_bl_chg(self):
        b0 = load_b0_dict()["W"]
        assert self.gbond.get_mean_bl_chg(b0=b0) == pytest.approx(-0.004265041421307773)

    def testSerialization(self):
        test = GBBond.from_dict(self.gbond.as_dict())
        assert test.max_bl == self.gbond.max_bl
        assert test.min_bl == self.gbond.min_bl
        assert np.allclose(test.bond_mat, self.gbond.bond_mat)


W_BULK = Structure.from_file(pjoin(REFS, "mp-91.cif"))
FAKE_DENSITY = 19.25
FAKE_G_VRH = 147.0
FAKE_E_COH = 8.301059


class _FakeMPRester:
    """Offline stand-in for pymatgen.ext.matproj.MPRester."""

    def __init__(self, api_key=None, shear_modulus=None):
        self.api_key = api_key
        self.shear_modulus = {"vrh": FAKE_G_VRH} if shear_modulus is None else shear_modulus

    def get_summary_by_material_id(self, material_id, fields=None):
        return {"density": FAKE_DENSITY, "shear_modulus": self.shear_modulus}

    def get_structure_by_material_id(self, material_id):
        return W_BULK.copy()


class _FakeMPAPIRester:
    """Offline stand-in for mp_api.client.MPRester; get_cohesive_energy returns E_bulk - E_atoms (negative)."""

    contribs = object()

    def __init__(self, api_key=None):
        self.api_key = api_key

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get_cohesive_energy(self, material_ids, normalization="atom"):
        assert normalization == "atom"
        return dict.fromkeys(material_ids, -FAKE_E_COH)


@pytest.fixture
def fake_mprester(monkeypatch):
    monkeypatch.setattr(describer_mod, "MPRester", _FakeMPRester)
    monkeypatch.setattr(describer_mod, "MPAPIRester", _FakeMPAPIRester)


@pytest.fixture(scope="module")
def w_gb_entry():
    return next(d for d in load_data() if d["task_id"] == 5094)


@pytest.fixture(scope="module")
def w_gbond(w_gb_entry):
    return GBBond(w_gb_entry["initial_structure"], loc_algo="crystalnn")


def test_breakbond_ratio_with_ndarray_bond_mat(w_gb_entry, w_gbond):
    # Regression: `if not self.bond_mat` raised "truth value of an array is ambiguous".
    ratio, n_short, n_long = w_gbond.get_breakbond_ratio(w_gb_entry["initial_structure"], b0=2.76, return_count=True)
    assert n_long > 0
    assert ratio == pytest.approx(n_short / n_long)
    assert w_gbond.get_breakbond_ratio(w_gb_entry["initial_structure"], b0=2.76) == pytest.approx(ratio)


def test_breakbond_ratio_ignores_unbonded_pairs(w_gb_entry, w_gbond):
    # Regression: zero (unbonded) entries of bond_mat were counted as short bonds.
    b0 = load_b0_dict()["W"]
    ratio, n_short, n_long = w_gbond.get_breakbond_ratio(w_gb_entry["initial_structure"], b0=b0, return_count=True)
    n_bonds = int((w_gbond.bond_mat > 0).sum())
    assert (n_short, n_long) == (4, 92)
    assert n_short + n_long == n_bonds
    assert ratio == pytest.approx(4 / 92)


def test_breakbond_ratio_no_long_bonds(w_gb_entry, w_gbond):
    ratio, n_short, n_long = w_gbond.get_breakbond_ratio(w_gb_entry["initial_structure"], b0=100.0, return_count=True)
    assert ratio == 0
    assert n_long == 0
    assert n_short == int((w_gbond.bond_mat > 0).sum())


@pytest.fixture(scope="module")
def small_gb() -> GrainBoundary:
    """12-site bcc W Sigma3 [110] tilt GB; small enough to run every loc_algo quickly."""
    bulk = Structure(Lattice.cubic(3.17), ["W", "W"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return GrainBoundaryGenerator(bulk).gb_from_parameters([1, 1, 0], 70.528779, expand_times=1, vacuum_thickness=0)


@pytest.fixture(scope="module")
def pr_gb_entry():
    """An hcp (Pr) entry, which uses four-index Miller-Bravais notation."""
    return next(d for d in load_data() if d["task_id"] == 5640)


# ---------------------------------------------------------------------------
# Index conversions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("axis", "lat_type", "expected"),
    [
        ((2, -1, -1, 0), "hexagonal", (1, 0, 0)),
        ((0, 0, 0, 1), "Hexagonal", (0, 0, 1)),
        ((2, 0, -2, 2), "hexagonal", (2, 1, 1)),  # [4, 2, 2] reduced by gcd
        ((1, 0, -1, 0), "rhombohedral", (2, -1, -1)),
        ((1, 1, -2, 1), "rhombohedral", (4, 1, -2)),
        ((0, 0, 0, 3), "rhombohedral", (1, 1, 1)),  # [3, 3, 3] reduced by gcd
    ],
)
def test_convert_hcp_direction(axis, lat_type, expected):
    assert np.array_equal(convert_hcp_direction(list(axis), lat_type), expected)


@pytest.mark.parametrize(
    ("plane", "expected"),
    [((2, -1, -1, 0), (2, -1, 0)), ((2, 0, -2, 4), (1, 0, 2)), ((0, 0, 0, 1), (0, 0, 1))],
)
def test_convert_hcp_plane(plane, expected):
    assert np.array_equal(convert_hcp_plane(list(plane)), expected)


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------


def test_structural_feature_bulk_conv_as_dict(w_gb_entry):
    entry = dict(w_gb_entry)
    entry["bulk_conv"] = w_gb_entry["bulk_conv"].as_dict()
    pd.testing.assert_frame_equal(get_structural_feature(entry), get_structural_feature(w_gb_entry))


def test_structural_feature_hcp_four_index(pr_gb_entry):
    # (0001) twist GB about [0001]: both d_gb and d_rot reduce to the c lattice parameter.
    df = get_structural_feature(pr_gb_entry, features=[d_gb, d_rot, cos_theta])
    c = pr_gb_entry["bulk_conv"].lattice.c
    assert list(df.columns) == [d_gb.str_name, d_rot.str_name, cos_theta.str_name]
    assert df[d_gb.str_name][0] == pytest.approx(c)
    assert df[d_rot.str_name][0] == pytest.approx(c)
    assert df[cos_theta.str_name][0] == pytest.approx(np.cos(np.radians(pr_gb_entry["rotation_angle"])))


def test_structural_feature_trigonal_four_index():
    # Regression: 'trigonal' crystal system skipped the 4-index -> 3-index axis conversion.
    bulk = Structure.from_spacegroup("P-3m1", Lattice.hexagonal(4, 7), ["Cd", "I"], [[0, 0, 0], [1 / 3, 2 / 3, 0.25]])
    entry = {"bulk_conv": bulk, "gb_plane": [1, 0, -1, 0], "rotation_axis": [0, 0, 0, 1], "rotation_angle": 30}
    df = get_structural_feature(entry, features=[d_gb, d_rot])
    assert df[d_gb.str_name][0] == pytest.approx(4 * np.sqrt(3) / 2)
    assert df[d_rot.str_name][0] == pytest.approx(7)


# ---------------------------------------------------------------------------
# Elemental features (MPRester mocked; no network)
# ---------------------------------------------------------------------------


def test_elemental_feature_requires_api_key(w_gb_entry):
    with pytest.raises(MPRestError, match="API key"):
        get_elemental_feature(w_gb_entry, features=[ar], mp_api=None)


def test_elemental_feature_values(fake_mprester, w_gb_entry):
    features = [e_coh, G, a0, ar, mean_delta_bl, mean_bl, bdensity, CLTE, hb]
    df = get_elemental_feature(w_gb_entry, features=features, mp_api="fake-key")
    w = Element("W")
    mdbl = load_mean_delta_bl_dict()["5094"]
    expected = {
        e_coh: FAKE_E_COH,
        G: FAKE_G_VRH,
        a0: W_BULK.lattice.a,
        ar: w.atomic_radius,
        mean_delta_bl: mdbl,
        mean_bl: mdbl + load_b0_dict()["W"],
        bdensity: FAKE_DENSITY,
        CLTE: w.coefficient_of_linear_thermal_expansion,
        hb: w.brinell_hardness,
    }
    assert list(df.columns) == [f.str_name for f in features]
    assert df.shape == (1, len(features))
    for feat, val in expected.items():
        assert df[feat.str_name][0] == pytest.approx(float(val)), feat.str_name
    assert df[mean_delta_bl.str_name][0] == pytest.approx(-0.004265041421307773)


def test_elemental_feature_default_features(fake_mprester, w_gb_entry):
    df = get_elemental_feature(w_gb_entry, mp_api="fake-key")
    assert list(df.columns) == [
        e_coh.str_name,
        G.str_name,
        a0.str_name,
        ar.str_name,
        mean_delta_bl.str_name,
        bdensity.str_name,
        CLTE.str_name,
        hb.str_name,
    ]


def test_elemental_feature_no_elasticity(monkeypatch, fake_mprester, w_gb_entry):
    # Materials without elastic data have shear_modulus=None in the summary doc.
    monkeypatch.setattr(describer_mod, "MPRester", lambda api_key: _FakeMPRester(api_key, shear_modulus={}))
    df = get_elemental_feature(w_gb_entry, features=[a0, G], mp_api="fake-key")
    assert df[a0.str_name][0] == pytest.approx(W_BULK.lattice.a)
    assert df[G.str_name][0] is None


def test_elemental_feature_unknown_loc_algo(fake_mprester, w_gb_entry):
    with pytest.raises(ValueError, match="mean_delta_bond_length"):
        get_elemental_feature(w_gb_entry, loc_algo="voronoinn", features=[mean_delta_bl], mp_api="fake-key")


def test_elemental_feature_with_real_mprester_interface(monkeypatch, w_gb_entry):
    # Regression: the code called rester.materials.search and rester.get_cohesive_energy, which don't exist on
    # pymatgen's MPRester. Autospec the real clients so calls outside their interfaces fail.
    rester = create_autospec(MPRester, instance=True)
    rester.get_summary_by_material_id.return_value = {"density": FAKE_DENSITY, "shear_modulus": {"vrh": FAKE_G_VRH}}
    rester.get_structure_by_material_id.return_value = W_BULK.copy()
    mpr = create_autospec(describer_mod.MPAPIRester, instance=True)
    mpr.__enter__.return_value = mpr
    mpr.get_cohesive_energy.return_value = {w_gb_entry["material_id"]: -FAKE_E_COH}
    monkeypatch.setattr(describer_mod, "MPRester", lambda api_key: rester)
    monkeypatch.setattr(describer_mod, "MPAPIRester", lambda api_key: mpr)
    df = get_elemental_feature(w_gb_entry, features=[a0, G, bdensity, e_coh], mp_api="fake-key")
    assert df[a0.str_name][0] == pytest.approx(W_BULK.lattice.a)
    assert df[G.str_name][0] == pytest.approx(FAKE_G_VRH)
    assert df[bdensity.str_name][0] == pytest.approx(FAKE_DENSITY)
    assert df[e_coh.str_name][0] == pytest.approx(FAKE_E_COH)
    mpr.get_cohesive_energy.assert_called_once_with([w_gb_entry["material_id"]], normalization="atom")


def test_elemental_feature_skips_cohesive_energy_when_not_requested(monkeypatch, fake_mprester, w_gb_entry):
    def fail(api_key):
        raise AssertionError("mp_api should not be called when e_coh is not requested")

    monkeypatch.setattr(describer_mod, "MPAPIRester", fail)
    df = get_elemental_feature(w_gb_entry, features=[a0, G], mp_api="fake-key")
    assert e_coh.str_name not in df.columns


def test_cohesive_energy_requires_mpcontribs(monkeypatch, fake_mprester, w_gb_entry):
    class NoContribs(_FakeMPAPIRester):
        contribs = None

    monkeypatch.setattr(describer_mod, "MPAPIRester", NoContribs)
    with pytest.raises(ImportError, match="MPContribs client"):
        get_elemental_feature(w_gb_entry, features=[e_coh], mp_api="fake-key")


# ---------------------------------------------------------------------------
# GBDescriber
# ---------------------------------------------------------------------------


def test_gbdescriber_default_features():
    desc = GBDescriber()
    assert [f.str_name for f in desc.struc_features] == ["d_gb", "d_rot", "sin(theta)", "cos(theta)"]
    assert [f.str_name for f in desc.elem_features] == ["e_coh", "G", "a_0", "a_r", "mean_delta_bl", "mean_bl"]


def test_gbdescriber_custom_features():
    desc = GBDescriber(structural_features=[d_gb], elemental_features=[ar])
    assert desc.struc_features == [d_gb]
    assert desc.elem_features == [ar]


def test_gbdescriber_transform_one(fake_mprester, w_gb_entry):
    df = GBDescriber().transform_one(w_gb_entry, inc_target=True, inc_bulk_ref=True, mp_api="fake-key")
    assert df.shape[0] == 2
    gb_row, bulk_row = df.iloc[0], df.iloc[1]
    mdbl = load_mean_delta_bl_dict()["5094"]
    theta = np.radians(w_gb_entry["rotation_angle"])
    assert gb_row["task_id"] == bulk_row["task_id"] == 5094
    assert gb_row[e_gb.str_name] == pytest.approx(w_gb_entry["gb_energy"])
    assert gb_row[sin_theta.str_name] == pytest.approx(np.sin(theta))
    assert gb_row[cos_theta.str_name] == pytest.approx(np.cos(theta))
    assert gb_row[mean_delta_bl.str_name] == pytest.approx(mdbl)
    assert gb_row[mean_bl.str_name] == pytest.approx(mdbl + load_b0_dict()["W"])
    assert gb_row[d_gb.str_name] == pytest.approx(1.0079488085907933)
    # Bulk reference: zero rotation, zero GB energy, zero bond-length change; other features unchanged.
    assert bulk_row[sin_theta.str_name] == 0
    assert bulk_row[cos_theta.str_name] == 1
    assert bulk_row[e_gb.str_name] == 0
    assert bulk_row[mean_delta_bl.str_name] == 0
    for col in (d_gb, d_rot, e_coh, G, a0, ar, mean_bl):
        assert bulk_row[col.str_name] == gb_row[col.str_name]


def test_gbdescriber_transform_one_no_target_no_bulk(fake_mprester, w_gb_entry):
    desc = GBDescriber(structural_features=[d_gb, cos_theta], elemental_features=[ar])
    df = desc.transform_one(w_gb_entry, inc_target=False, inc_bulk_ref=False, mp_api="fake-key")
    assert df.shape == (1, 4)
    assert list(df.columns) == [d_gb.str_name, cos_theta.str_name, ar.str_name, "task_id"]


def test_generate_bulk_ref_without_bl_or_target():
    gb_df = pd.DataFrame({sin_theta.str_name: [0.5], cos_theta.str_name: [0.8], d_gb.str_name: [1.2]})
    bulk_df = GBDescriber().generate_bulk_ref(gb_df, inc_target=False)
    assert list(bulk_df.columns) == list(gb_df.columns)
    assert bulk_df[sin_theta.str_name][0] == 0
    assert bulk_df[cos_theta.str_name][0] == 1
    assert bulk_df[d_gb.str_name][0] == 1.2
    assert gb_df[sin_theta.str_name][0] == 0.5  # input not mutated


# ---------------------------------------------------------------------------
# GBBond and every NNDict loc_algo
# ---------------------------------------------------------------------------


def test_nndict_keys():
    assert set(GBBond.NNDict) == {
        "nearneighbors",
        "voronoinn",
        "jmolnn",
        "minimumdistancenn",
        "openbabelnn",
        "covalentbondnn",
        "minimumvirenn",
        "minimumokeeffenn",
        "brunnernnreciprocal",
        "brunnernnreal",
        "brunnernnrelative",
        "econnn",
        "crystalnn",
        "cutoffdictnn",
        "critic2nn",
    }


# (loc_algo, kwargs, expected number of bonds in the 12-site GB)
WORKING_LOC_ALGOS = [
    ("jmolnn", {}, 100),
    ("minimumdistancenn", {}, 60),
    ("minimumvirenn", {}, 60),
    ("minimumokeeffenn", {}, 60),
    ("brunnernnreciprocal", {}, 100),
    ("brunnernnreal", {}, 108),
    ("brunnernnrelative", {}, 100),
    ("econnn", {}, 100),
    ("crystalnn", {}, 76),
    ("cutoffdictnn", {"cut_off_dict": {("W", "W"): 2.8}}, 60),
]


@pytest.mark.parametrize(("loc_algo", "kwargs", "n_bonds"), WORKING_LOC_ALGOS, ids=[a[0] for a in WORKING_LOC_ALGOS])
def test_gbbond_loc_algos(small_gb, loc_algo, kwargs, n_bonds):
    gbond = GBBond(small_gb, loc_algo=loc_algo, **kwargs)
    assert isinstance(gbond.loc_algo, GBBond.NNDict[loc_algo])
    bm = gbond.bond_matrix
    assert bm.shape == (small_gb.num_sites, small_gb.num_sites)
    assert int((bm > 0).sum()) == n_bonds
    # Every recorded bond length equals the (minimum-image) interatomic distance.
    for i, j in zip(*np.nonzero(bm), strict=True):
        assert bm[i, j] == pytest.approx(small_gb.get_distance(i, j))
    dist = small_gb.distance_matrix
    assert gbond.min_bl == pytest.approx(dist[dist > 0].min())
    assert gbond.max_bl == bm.max()
    assert gbond.get_mean_bl_chg(b0=0.0) == pytest.approx(bm[bm > 0].mean())
    # Round trip keeps the algorithm name and bond matrix.
    d = gbond.as_dict()
    assert d["loc_algo"] == loc_algo
    restored = GBBond.from_dict(d)
    assert isinstance(restored.loc_algo, GBBond.NNDict[loc_algo])
    assert np.array_equal(restored.bond_mat, bm)


def test_gbbond_cutoffdictnn_without_cutoffs_finds_no_bonds(small_gb):
    gbond = GBBond(small_gb, loc_algo="cutoffdictnn")
    assert not gbond.bond_mat.any()
    assert gbond.max_bl == 0


@pytest.mark.parametrize(
    ("loc_algo", "exc"),
    [
        ("nearneighbors", NotImplementedError),  # abstract base class
        ("covalentbondnn", AttributeError),  # Molecule-only; needs Molecule.get_covalent_bonds
    ],
)
def test_gbbond_unsupported_loc_algos(small_gb, loc_algo, exc):
    with pytest.raises(exc):
        GBBond(small_gb, loc_algo=loc_algo)


@pytest.mark.skipif(shutil.which("critic2") is not None, reason="critic2 is installed")
def test_gbbond_critic2nn_requires_executable(small_gb):
    with pytest.raises(RuntimeError, match="critic"):
        GBBond(small_gb, loc_algo="critic2nn")


def test_gbbond_openbabelnn_requires_openbabel(small_gb):
    try:
        from openbabel import openbabel  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="openbabel"):
            GBBond(small_gb, loc_algo="openbabelnn")
    else:
        pytest.skip("openbabel is installed; OpenBabelNN only supports Molecules")


def test_gbbond_voronoinn(small_gb):
    # Regression: small_gb has frac coords of ~-1e-16, which made VoronoiNN raise "No Voronoi neighbors found".
    assert (small_gb.frac_coords < 0).any()
    gbond = GBBond(small_gb, loc_algo="voronoinn")
    i, j = np.nonzero(gbond.bond_mat)
    assert len(i) > 0
    assert gbond.bond_mat[i, j] == pytest.approx([small_gb.get_distance(a, b) for a, b in zip(i, j, strict=True)])


def test_gbbond_from_dict_structure(small_gb):
    # bond_mat=None with a dict gb exercises the dict -> GrainBoundary conversion in _get_bond_mat.
    ref = GBBond(small_gb, loc_algo="minimumdistancenn")
    gbond = GBBond.from_dict({"gb": small_gb.as_dict(), "loc_algo": "minimumdistancenn", "bond_mat": None})
    assert np.allclose(gbond.bond_mat, ref.bond_mat)


def test_gbbond_unknown_loc_algo(small_gb):
    with pytest.raises(KeyError):
        GBBond(small_gb, loc_algo="not_an_algo")


def test_breakbond_ratio_recomputes_missing_bond_mat(small_gb):
    gbond = GBBond(small_gb, loc_algo="minimumdistancenn")
    expected = gbond.bond_mat.copy()
    gbond.bond_mat = None
    ratio, n_short, n_long = gbond.get_breakbond_ratio(small_gb, b0=2.7, return_count=True)
    assert np.array_equal(gbond.bond_mat, expected)
    assert (n_short, n_long) == (((expected > 0) & (expected < 2.7)).sum(), (expected > 2.7).sum())
    assert n_short + n_long == 60
    assert ratio == pytest.approx(n_short / n_long)


if __name__ == "__main__":
    unittest.main()
