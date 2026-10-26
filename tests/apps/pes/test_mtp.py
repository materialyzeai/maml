# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from shutil import which

import numpy as np
import pytest
from monty.serialization import loadfn
from pymatgen.core import Structure

from maml.apps.pes import MTPotential, _mtp
from maml.apps.pes._mtp import MTini_params, feed

CWD = os.getcwd()
test_datapool = loadfn(os.path.join(os.path.abspath(os.path.dirname(__file__)), "datapool.json"))
config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "MTP", "fitted.mtp")


class MTPotentialTest(unittest.TestCase):
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
        self.potential = MTPotential(name="test")
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

    def test_write_read_cfgs(self):
        self.potential.elements = ["Mo"]
        self.potential.write_cfg("test.cfgs", cfg_pool=self.test_pool)
        datapool, df = self.potential.read_cfgs("test.cfgs")
        assert len(self.test_pool) == len(datapool)
        for data1, data2 in zip(self.test_pool, datapool):
            struct1 = data1["structure"]
            struct2 = Structure.from_dict(data2["structure"])
            assert struct1 == struct2
            energy1 = data1["outputs"]["energy"]
            energy2 = data2["outputs"]["energy"]
            self.assertAlmostEqual(energy1, energy2)
            forces1 = np.array(data1["outputs"]["forces"])
            forces2 = data2["outputs"]["forces"]
            np.testing.assert_array_almost_equal(forces1, forces2)

    @unittest.skipIf(not which("mlp"), "No MLIP cmd found.")
    def test_train(self):
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
            unfitted_mtp="08g.mtp",
            max_dist=3.0,
            max_iter=20,
        )
        assert self.potential.param

    @unittest.skipIf(not which("mlp"), "No MLIP cmd found.")
    def test_evaluate(self):
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
            unfitted_mtp="08g.mtp",
            max_dist=3.0,
            max_iter=20,
        )
        df_orig, df_tar = self.potential.evaluate(
            test_structures=self.test_structures,
            test_energies=self.test_energies,
            test_forces=self.test_forces,
            test_stresses=self.test_stresses,
        )
        assert df_orig.shape[0] == df_tar.shape[0]

    @unittest.skipIf(not which("mlp"), "No MLIP cmd found.")
    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found.")
    def test_predict_efs(self):
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
            unfitted_mtp="08g.mtp",
            max_dist=3.0,
            max_iter=20,
        )
        energy, forces, stress = self.potential.predict_efs(self.test_struct)
        assert len(forces) == len(self.test_struct)
        assert len(stress) == 6

    def test_from_config(self):
        mtp = MTPotential.from_config(config_file, elements=["Mo"])
        assert mtp.param is not None


# ---------------------------------------------------------------------------
# Tests below run without the MLIP `mlp` binary, which is replaced by a fake.
# ---------------------------------------------------------------------------


def _shift_energies(cfg_text, shift):
    """Add `shift` to every energy in a .cfgs file."""
    lines = cfg_text.split("\n")
    for i, line in enumerate(lines[:-1]):
        if line.strip() == "Energy":
            lines[i + 1] = f"{float(lines[i + 1]) + shift:>24.12f}"
    return "\n".join(lines)


def fake_mlp(args):
    command = args[1]
    if command == "mindist":
        return 0, "Reading configurations...\nGlobal mindist: 2.25\n", ""
    if command == "train":
        trained = next(a.split("=", 1)[1] for a in args if a.startswith("--trained-pot-name="))
        shutil.copyfile(config_file, trained)
        return 0, "BFGS iterations done\n", ""
    if command == "calc-efs":
        _, _, _, original, predicted = args
        Path(predicted).write_text(_shift_energies(Path(original).read_text(), 1.0))
        return 0, "", ""
    if command == "run":
        original = args[3].split("=", 1)[1]
        Path("predict.cfgs_0").write_text(_shift_energies(Path(original).read_text(), 2.0))
        return 0, "", ""
    raise AssertionError(f"unexpected mlp command {args}")


@pytest.fixture
def fake_mlp_bin(mock_binaries, in_tmp):
    def install(handler=fake_mlp, available=None):
        return mock_binaries(_mtp, handler, available=available)

    return install


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "1.0"),
        ({"Energy_equation_weight": 2.5}, "2.5"),
        # Regression: an explicit zero used to be replaced by the default value.
        ({"Energy_equation_weight": 0}, "0"),
        ({"Energy_equation_weight": None}, "1.0"),
    ],
)
def test_feed(kwargs, expected):
    fit = MTini_params["MLIP"]["Fit"]
    name, value, comment = feed("Energy_equation_weight", kwargs, fit, tab="\t\t")
    assert name == "\t\tmlip:fit:energy-weight"
    assert value == expected
    assert comment == fit["Energy_equation_weight"]["comment"]


def test_write_read_require_elements(in_tmp, datapool):
    mtp = MTPotential()
    with pytest.raises(ValueError, match="No species given"):
        mtp.write_cfg("x.cfgs", datapool[:1])
    with pytest.raises(ValueError, match="No species given"):
        mtp.read_cfgs("x.cfgs")


@pytest.mark.parametrize(("version", "header"), [("mlip-2", "PlusStress:  xx"), ("mlip-dev", "Stress:  xx")])
def test_write_read_cfgs_versions(in_tmp, datapool, version, header):
    mtp = MTPotential(version=version)
    mtp.elements = ["Mo"]
    pool = [{"structure": d["structure"].as_dict(), "outputs": d["outputs"]} for d in datapool[:2]]
    text = Path(mtp.write_cfg("test.cfgs", pool)).read_text()
    assert text.count("BEGIN_CFG") == 2
    assert header in text
    data_pool, df = mtp.read_cfgs("test.cfgs")
    for orig, read in zip(datapool[:2], data_pool, strict=True):
        np.testing.assert_allclose(read["outputs"]["virial_stress"], orig["outputs"]["virial_stress"], atol=1e-5)
        assert read["num_atoms"] == len(orig["structure"])
    assert (df["dtype"] == "stress").sum() == 12


def test_write_ini_mlip2(in_tmp):
    # Regression: integer/float options used to crash the string formatter.
    mtp = MTPotential()
    filename = mtp.write_ini(
        mtp_filename="pot.mtp",
        write_cfgs="out.cfgs",
        write_cfgs_skip=3,
        select=True,
        select_threshold=2.5,
        select_threshold_break=10,
        select_save_selected="selected.cfgs",
    )
    lines = [line.split() for line in Path(filename).read_text().splitlines()]
    assert lines == [
        ["mtp-filename", "pot.mtp"],
        ["write-cfgs", "out.cfgs"],
        ["write-cfgs:skip", "3"],
        ["select", "TRUE"],
        ["select:save-selected", "selected.cfgs"],
        ["select:threshold", "2.5"],
        ["select:threshold-break", "10"],
    ]
    mtp.write_ini()
    assert Path("mlip.ini").read_text().split() == ["mtp-filename", "fitted.mtp", "select", "FALSE"]


def test_write_ini_mlip_dev(in_tmp):
    mtp = MTPotential(version="mlip-dev")
    mtp.write_ini(
        Abinitio=2,
        r_min=2.5,
        load_from="pot.mtp",
        Calculate_EFS=True,
        Fit=True,
        Stress_equation_weight=0.0,
        Select=True,
        Write_cfgs="out.cfgs",
        Driver=2,
        Iteration_limit=50,
    )
    text = Path("mlip.ini").read_text()
    settings = {
        line.split()[0]: line.split()[1] for line in text.splitlines() if line.strip() and not line.startswith("#")
    }
    assert settings["abinitio"] == "2"
    assert settings["Abinitio:LJ:r_min"] == "2.5"
    assert settings["Abinitio:LJ:scale"] == "1.0"
    assert settings["mlip"] == "mtpr"
    assert settings["mlip:load-from"] == "pot.mtp"
    assert settings["calculate-efs"] == "TRUE"
    assert settings["MLIP:Fit"] == "true"
    # Regression: a zero stress weight must not be replaced by the default of 0.1.
    assert settings["mlip:fit:stress-weight"] == "0.0"
    assert settings["mlip:fit:energy-weight"] == "1.0"
    assert settings["MLIP:Select"] == "true"
    assert settings["write-cfgs"] == "out.cfgs"
    assert settings["driver"] == "2"
    assert settings["Driver:Relaxation:Iteration_limit"] == "50"


def test_from_config_write_param_roundtrip(in_tmp):
    mtp = MTPotential.from_config(config_file, elements=["Mo"])
    assert mtp.elements == ["Mo"]
    assert mtp.param["species_coeffs"] == [-9.118920883237898]
    assert len(mtp.param["moment_coeffs"]) == 9
    pair_style, pair_coeff = mtp.write_param(fitted_mtp="out.mtp")
    assert pair_style == "pair_style        mlip mlip.ini"
    assert pair_coeff == "pair_coeff        * *"
    assert Path("out.mtp").read_text().rstrip("\n") == Path(config_file).read_text().rstrip("\n")
    assert "out.mtp" in Path("mlip.ini").read_text()


def test_write_param_without_param():
    with pytest.raises(RuntimeError, match="parameters should be provided"):
        MTPotential().write_param()


@pytest.mark.parametrize("method", ["train", "evaluate"])
def test_missing_mlp(fake_mlp_bin, train_data, method):
    fake_mlp_bin(available=set())
    with pytest.raises(RuntimeError, match="mlp has not been found") as excinfo:
        getattr(MTPotential(), method)(*train_data)
    assert len(excinfo.value.args) == 1


def test_train(fake_mlp_bin, train_data):
    calls = fake_mlp_bin()
    mtp = MTPotential()
    assert mtp.train(*train_data, max_dist=4.5, radial_basis_size=6, max_iter=7, stress_weight=0) == 0
    assert mtp.elements == ["Mo"]
    assert calls[0] == ["mlp", "mindist", "train.cfgs"]
    train_cmd = calls[1]
    assert train_cmd[:4] == ["mlp", "train", "08g.mtp", "train.cfgs"]
    for arg in ["--max-iter=7", "--trained-pot-name=08g_fitted.mtp", "--stress-weight=0", "--weighting=vibration"]:
        assert arg in train_cmd
    unfitted = calls.file("08g.mtp")
    assert "species_count = 1" in unfitted
    assert "min_dist = 2.250000" in unfitted
    assert "max_dist = 4.500000" in unfitted
    assert "radial_basis_size = 6" in unfitted
    assert calls.file("train.cfgs").count("BEGIN_CFG") == len(train_data[0])
    assert mtp.param == MTPotential.from_config(config_file, ["Mo"]).param


def test_train_no_unfitted_mtp(fake_mlp_bin, train_data):
    fake_mlp_bin()
    with pytest.raises(RuntimeError, match="No specific parameter file"):
        MTPotential().train(*train_data, unfitted_mtp="")


def test_train_failure(fake_mlp_bin, train_data):
    def handler(args):
        if args[1] == "train":
            return 2, "ERROR: Too few configurations\n", ""
        return fake_mlp(args)

    fake_mlp_bin(handler)
    with pytest.raises(RuntimeError, match="MLP exited with return code 2: ERROR: Too few configurations"):
        MTPotential().train(*train_data)


@pytest.mark.parametrize(("version", "shift"), [("mlip-2", 1.0), ("mlip-dev", 2.0)])
def test_evaluate(fake_mlp_bin, train_data, version, shift):
    calls = fake_mlp_bin()
    mtp = MTPotential.from_config(config_file, elements=["Mo"])
    mtp.version = version
    df_orig, df_predict = mtp.evaluate(*train_data)
    if version == "mlip-2":
        assert calls[-1] == ["mlp", "calc-efs", "fitted.mtp", "original.cfgs", "predict.cfgs"]
    else:
        assert calls[-1] == ["mlp", "run", "mlip.ini", "--filename=original.cfgs"]
        assert "driver:cfg-reader:filename" in calls.file("mlip.ini")
    assert df_orig.shape == df_predict.shape
    is_energy = df_orig["dtype"] == "energy"
    np.testing.assert_allclose(df_predict[is_energy]["y_orig"], df_orig[is_energy]["y_orig"] + shift, atol=1e-6)
    np.testing.assert_allclose(df_predict[~is_energy]["y_orig"], df_orig[~is_energy]["y_orig"])


def test_evaluate_failure(fake_mlp_bin, train_data):
    fake_mlp_bin(lambda args: (1, "", ""))
    mtp = MTPotential.from_config(config_file, elements=["Mo"])
    with pytest.raises(RuntimeError, match=r"mlp exited with return code 1$"):
        mtp.evaluate(*train_data)
