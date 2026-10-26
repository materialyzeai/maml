# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.
from __future__ import annotations

import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from shutil import which

import numpy as np
import pytest
from monty.serialization import loadfn
from pymatgen.io.lammps.data import LammpsData
from sklearn.linear_model import LinearRegression

from maml.apps.pes import SNAPotential, SpectralNeighborAnalysis, _lammps
from maml.base import SKLModel
from maml.describers import BispectrumCoefficients
from maml.utils import check_structures_forces_stresses, stress_format_change

CWD = os.getcwd()
DIR = os.path.abspath(os.path.dirname(__file__))
test_datapool = loadfn(os.path.join(DIR, "datapool.json"))
coeff_file = os.path.join(DIR, "SNAP", "SNAPotential.snapcoeff")
param_file = os.path.join(DIR, "SNAP", "SNAPotential.snapparam")


@unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found.")
class SNAPotentialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()
        os.chdir(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(CWD)
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        profile = {"Mo": {"r": 0.6, "w": 1.0}}
        self.describer1 = BispectrumCoefficients(
            rcutfac=4.6, twojmax=6, element_profile=profile, quadratic=False, pot_fit=True
        )
        model1 = SKLModel(describer=self.describer1, model=LinearRegression())
        self.potential1 = SNAPotential(model=model1, name="test")
        self.describer2 = BispectrumCoefficients(
            rcutfac=4.6, twojmax=6, element_profile=profile, quadratic=True, pot_fit=True
        )
        model2 = SKLModel(describer=self.describer2, model=LinearRegression())
        self.potential2 = SNAPotential(model=model2, name="test")
        self.test_pool = test_datapool
        self.test_structures = []
        self.test_energies = []
        self.test_forces = []
        self.test_stresses = []
        for d in self.test_pool:
            self.test_structures.append(d["structure"])
            self.test_energies.append(d["outputs"]["energy"])
            self.test_forces.append(d["outputs"]["forces"])
            self.test_stresses.append(d["outputs"]["virial_stress"])
        self.test_struct = self.test_pool[-1]["structure"]

    def test_train(self):
        self.potential1.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        assert len(self.potential1.model.model.coef_) == len(self.describer1.subscripts) + 1
        self.potential2.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        nss = len(self.describer2.subscripts)
        assert len(self.potential2.model.model.coef_) == nss + int((1 + nss) * nss / 2) + 1

    def test_evaluate(self):
        self.potential1.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        df_orig, df_tar = self.potential1.evaluate(
            test_structures=self.test_structures,
            test_energies=self.test_energies,
            test_forces=self.test_forces,
            test_stresses=self.test_stresses,
        )
        assert df_orig.shape[0] == df_tar.shape[0]

        self.potential2.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        df_orig, df_tar = self.potential2.evaluate(
            test_structures=self.test_structures,
            test_energies=self.test_energies,
            test_forces=self.test_forces,
            test_stresses=self.test_stresses,
        )
        assert df_orig.shape[0] == df_tar.shape[0]

    def test_predict_efs(self):
        self.potential1.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        energy, forces, stress = self.potential1.predict_efs(self.test_struct)
        assert len(forces) == len(self.test_struct)
        assert len(stress) == 6
        self.potential2.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        energy, forces, stress = self.potential2.predict_efs(self.test_struct)
        assert len(forces) == len(self.test_struct)
        assert len(stress) == 6

    def test_from_config(self):
        snap = SNAPotential.from_config(param_file, coeff_file)
        assert snap.model.model.coef_ is not None


# ---------------------------------------------------------------------------
# Tests below run without LAMMPS: the bispectrum calculation and the LAMMPS
# energy/force/stress run are replaced by deterministic fakes.
# ---------------------------------------------------------------------------

DUMP_HEADER = "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{n}\nITEM: BOX BOUNDS\n0 1\n0 1\n0 1\nITEM: ATOMS\n"


def _dump(filename, rows):
    rows = np.atleast_2d(rows)
    Path(filename).write_text(DUMP_HEADER.format(n=len(rows)) + "\n".join(" ".join(map(str, r)) for r in rows) + "\n")


def fake_lammps_handler(args):
    """Deterministic, structure-dependent stand-in for LAMMPS sna/atom and efs runs."""
    input_file = args[-1]
    struct = LammpsData.from_file("data.static", atom_style="charge").structure
    n_atoms = len(struct)
    text = Path(input_file).read_text()
    if input_file == "in.efs":
        Path("energy.txt").write_text(f"{-10.0 * n_atoms}\n")
        _dump("force.dump", np.zeros((n_atoms, 3)))
        Path("stress.txt").write_text("1 2 3 4 5 6\n")
        return
    twojmax = int(re.search(r"sna/atom\s+1 0.99363 (\d+)", text).group(1))
    quadratic = int(re.search(r"quadraticflag (\d)", text).group(1))
    elements = re.search(r"dump_modify\s+1 element (.*)", text).group(1).split()
    n_bs = len(SpectralNeighborAnalysis.get_bs_subscripts(twojmax))
    if quadratic:
        n_bs += n_bs * (n_bs + 1) // 2
    coords = struct.cart_coords
    k = np.arange(1, n_bs + 1)
    b = np.sin(np.outer(coords[:, 0], k) / 7) + np.cos(np.outer(coords[:, 1] + coords[:, 2], k) / 11)
    db = np.cos(np.outer(coords.sum(axis=1), np.arange(1, 3 * n_bs * len(elements) + 1)) / 13)
    vb = np.sin(np.outer(coords[:, 2], np.arange(1, 6 * n_bs * len(elements) + 1)) / 5)
    _dump("dump.element", [[str(sp)] for sp in struct.species])
    _dump("dump.sna", b)
    _dump("dump.snad", db)
    _dump("dump.snav", vb)
    return


@pytest.fixture
def snap_factory(mock_binaries, in_tmp):
    mock_binaries(_lammps, fake_lammps_handler)

    def make(quadratic=False, twojmax=2, name="test"):
        describer = BispectrumCoefficients(
            rcutfac=4.6,
            twojmax=twojmax,
            element_profile={"Mo": {"r": 0.6, "w": 1.0}},
            quadratic=quadratic,
            pot_fit=True,
        )
        return SNAPotential(model=SKLModel(describer=describer, model=LinearRegression()), name=name)

    return make


@pytest.mark.parametrize("quadratic", [False, True])
def test_train_write_param_from_config_roundtrip(snap_factory, train_data, quadratic):
    structures, energies, forces, stresses = train_data
    snap = snap_factory(quadratic=quadratic)
    snap.train(structures, energies, forces, stresses)
    n_bs = len(snap.model.describer.subscripts)
    if quadratic:
        n_bs += n_bs * (n_bs + 1) // 2
    coef = snap.model.model.coef_
    assert len(coef) == n_bs + 1

    pair_style, pair_coeff = snap.write_param()
    assert pair_style == "pair_style        snap"
    assert pair_coeff == "pair_coeff        * * test.snapcoeff test.snapparam Mo"
    params = Path("test.snapparam").read_text().splitlines()
    assert params == [
        "rcutfac 4.6",
        "twojmax 2",
        "rfac0 0.99363",
        "rmin0 0",
        f"quadraticflag {int(quadratic)}",
        "bzeroflag 0",
    ]
    coeff_lines = Path("test.snapcoeff").read_text().splitlines()
    assert coeff_lines[:2] == [f"1 {n_bs + 1}", "Mo 0.6 1.0"]

    loaded = SNAPotential.from_config("test.snapparam", "test.snapcoeff")
    np.testing.assert_allclose(loaded.model.model.coef_, coef)
    describer = loaded.model.describer
    assert (describer.rcutfac, describer.twojmax, describer.quadratic) == (4.6, 2, quadratic)
    assert describer.element_profile == {"Mo": {"r": 0.6, "w": 1.0}}
    assert loaded.elements == ["Mo"]


def test_from_config_shipped_files(snap_factory):
    snap = SNAPotential.from_config(param_file, coeff_file)
    assert snap.model.describer.twojmax == 6
    assert snap.model.describer.rcutfac == 5.0
    assert not snap.model.describer.quadratic
    assert snap.model.model.coef_.shape == (31,)
    assert snap.model.model.coef_[0] == pytest.approx(-22.66177458410766)


def test_from_config_without_quadraticflag(snap_factory, tmp_path):
    (tmp_path / "p.snapparam").write_text("rcutfac 4.0\ntwojmax 2\nrfac0 0.99363\n")
    (tmp_path / "c.snapcoeff").write_text("# comment\n1 6\nMo 0.5 1\n" + "\n".join(["0.5"] * 6))
    snap = SNAPotential.from_config(str(tmp_path / "p.snapparam"), str(tmp_path / "c.snapcoeff"))
    assert not snap.model.describer.quadratic
    np.testing.assert_allclose(snap.model.model.coef_, [0.5] * 6)


@pytest.mark.parametrize("include_stress", [False, True])
def test_evaluate_mocked(snap_factory, train_data, include_stress):
    structures, energies, forces, stresses = train_data
    snap = snap_factory()
    snap.model.describer.include_stress = include_stress
    snap.train(structures, energies, forces, stresses, include_stress=include_stress)
    df_orig, df_predict = snap.evaluate(structures, energies, forces, stresses, include_stress=include_stress)
    n_rows = sum(1 + 3 * len(s) + (6 if include_stress else 0) for s in structures)
    assert df_orig.shape[0] == df_predict.shape[0] == n_rows
    assert list(df_orig["dtype"]) == list(df_predict["dtype"])
    expected = snap.model.predict_objs(structures) * df_predict["n"]
    np.testing.assert_allclose(df_predict["y_orig"], expected)


def test_evaluate_respects_stress_format(snap_factory, datapool):
    # Regression: evaluate() ignored stress_format when rotating stresses of triclinic cells.
    entry = datapool[9]
    assert not entry["structure"].lattice.is_orthogonal
    structures, energies = [entry["structure"]], [entry["outputs"]["energy"]]
    forces, stresses = [entry["outputs"]["forces"]], [entry["outputs"]["virial_stress"]]
    snap = snap_factory()
    snap.model.describer.include_stress = True
    snap.train(structures * 2, energies * 2, forces * 2, stresses * 2, include_stress=True, stress_format="LAMMPS")
    df_orig, _ = snap.evaluate(structures, energies, forces, stresses, include_stress=True, stress_format="LAMMPS")
    _, _, rotated = check_structures_forces_stresses(structures, forces, stresses, stress_format="LAMMPS")
    expected = stress_format_change(rotated[0], from_format="LAMMPS", to_format="SNAP")
    np.testing.assert_allclose(df_orig[df_orig["dtype"] == "stress"]["y_orig"], expected)


def test_predict_efs_mocked(snap_factory, train_data):
    structures, energies, forces, stresses = train_data
    snap = snap_factory()
    snap.train(structures, energies, forces, stresses)
    energy, forces_pred, stress = snap.predict_efs(structures[0])
    assert energy == pytest.approx(-10.0 * len(structures[0]))
    assert forces_pred.shape == (len(structures[0]), 3)
    assert len(stress) == 6
