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
from pymatgen.core import Structure

from maml.apps.pes import GAPotential, _gap

CWD = os.getcwd()
test_datapool = loadfn(os.path.join(os.path.abspath(os.path.dirname(__file__)), "datapool.json"))


class GAPotentialTest(unittest.TestCase):
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
        self.potential = GAPotential(name="test")
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
        self.test_struct = d["structure"]

    def test_write_read_cfgs(self):
        self.potential.write_cfgs("test.xyz", cfg_pool=self.test_pool)
        datapool, df = self.potential.read_cfgs("test.xyz")
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
            stress1 = np.array(data1["outputs"]["virial_stress"])
            stress2 = data2["outputs"]["virial_stress"]
            np.testing.assert_array_almost_equal(stress1, stress2)

    @unittest.skipIf(not which("gap_fit"), "No QUIP cmd found.")
    def test_train(self):
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        assert self.potential.param

    @unittest.skipIf(not which("quip"), "No QUIP cmd found.")
    def test_evaluate(self):
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        df_orig, df_tar = self.potential.evaluate(
            test_structures=self.test_structures,
            test_energies=self.test_energies,
            test_forces=self.test_forces,
            test_stresses=self.test_stresses,
        )
        assert df_orig.shape[0] == df_tar.shape[0]

    @unittest.skipIf(not which("gap_fit"), "No QUIP cmd found.")
    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found.")
    def test_predict(self):
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
        )
        energy, forces, stress = self.potential.predict_efs(self.test_struct)
        assert len(forces) == len(self.test_struct)
        assert len(stress) == 6


# ---------------------------------------------------------------------------
# Tests below run without QUIP/LAMMPS; gap_fit and quip are replaced by fakes.
# ---------------------------------------------------------------------------

GAP_XML = """<GAP_2017_6_17_60_4_3_56_165>
  <Potential label="GAP_2017_6_17_60_4_3_56_165" init_args="IP GAP label=GAP_2017_6_17_60_4_3_56_165"/>
  <GAP_params label="GAP_2017_6_17_60_4_3_56_165" gap_version="1496676150">
    <GAP_data do_core="F" e0="0.0"/>
    <gpSparse label="GAP_2017_6_17_60_4_3_56_165" n_coordinate="1">
      <gpCoordinates label="GAP_2017_6_17_60_4_3_56_1651" dimensions="3" signal_variance="1.0"
          sparseX_filename="{xml}.sparseX.GAP_2017_6_17_60_4_3_56_1651">
        <descriptor>soap l_max=12 n_max=10 cutoff=4.0 Z=42 n_species=1 species_Z={{42}}</descriptor>
      </gpCoordinates>
    </gpSparse>
  </GAP_params>
</GAP_2017_6_17_60_4_3_56_165>
"""
SPARSE_X = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def fake_gap_fit(args):
    xml = next(a.split("=", 1)[1] for a in args if a.startswith("gp_file="))
    Path(xml).write_text(GAP_XML.format(xml=xml))
    np.savetxt(f"{xml}.sparseX.GAP_2017_6_17_60_4_3_56_1651", SPARSE_X)
    return 0, "gap_fit finished\n", ""


def fake_quip(args):
    """Return the input frames with predicted energy (dft + 1) and forces (0.5 * dft) appended."""
    atoms_file = next(a.split("=", 1)[1] for a in args if a.startswith("atoms_filename="))
    lines = Path(atoms_file).read_text().split("\n")
    out, i = [], 0
    while i < len(lines):
        n_atoms = int(lines[i])
        comment = lines[i + 1]
        energy = float(re.search(r"dft_energy=(\S+)", comment).group(1))
        out.append(lines[i])
        out.append(comment.replace("dft_force:R:3", "dft_force:R:3:force:R:3") + f" energy={energy + 1:.6f}")
        for line in lines[i + 2 : i + 2 + n_atoms]:
            dft_force = [float(f) for f in line.split()[-3:]]
            out.append("AT " + line + "".join(f"{0.5 * f:>16f}" for f in dft_force))
        i += 2 + n_atoms
    return 0, "\n".join(out), ""


@pytest.fixture
def gap_binaries(mock_binaries, in_tmp):
    def install(handler=None, available=None):
        def dispatch(args):
            return (handler or {"gap_fit": fake_gap_fit, "quip": fake_quip}[args[0]])(args)

        return mock_binaries(_gap, dispatch, available=available)

    return install


@pytest.fixture
def trained_gap(gap_binaries, train_data):
    gap_binaries()
    gap = GAPotential(name="test")
    gap.train(*train_data)
    return gap


def test_line_up(datapool):
    entry = datapool[0]
    struct = entry["structure"]
    stress = [1, 2, 3, 4, 5, 6]
    lines = GAPotential._line_up(struct, -10.5, entry["outputs"]["forces"], stress).split("\n")
    assert lines[0] == str(len(struct))
    assert len(lines) == len(struct) + 2
    assert "dft_energy=-10.5" in lines[1]
    # Voigt [xx, yy, zz, xy, yz, xz] expanded to the full 3x3 matrix
    assert "dft_virial=1\t4\t6\t4\t2\t5\t6\t5\t3" in lines[1]
    assert lines[1].endswith("Properties=species:S:1:pos:R:3:Z:I:1:dft_force:R:3")
    assert lines[2].split()[0] == "Mo"
    assert lines[2].split()[4] == "42"


def test_write_cfgs_requires_xyz(in_tmp, datapool):
    with pytest.raises(RuntimeError, match=r"should end with \".xyz\""):
        GAPotential().write_cfgs("train.cfg", cfg_pool=datapool[:1])


def test_write_cfgs_accepts_dict_structures(in_tmp, datapool):
    pool = [{"structure": d["structure"].as_dict(), "outputs": d["outputs"]} for d in datapool[:2]]
    GAPotential().write_cfgs("dict.xyz", cfg_pool=pool)
    data_pool, df = GAPotential.read_cfgs("dict.xyz")
    assert len(data_pool) == 2
    assert df.shape[0] == sum(1 + 3 * len(d["structure"]) for d in datapool[:2])


@pytest.mark.parametrize(("binary", "method"), [("gap_fit", "train"), ("quip", "evaluate")])
def test_missing_binary(gap_binaries, train_data, binary, method):
    # Regression: the message used to be passed as a tuple of strings.
    gap_binaries(available=set())
    with pytest.raises(RuntimeError, match=f"{binary} has not been found") as excinfo:
        getattr(GAPotential(), method)(*train_data)
    assert len(excinfo.value.args) == 1
    assert "https://github.com/libAtoms/QUIP for further detail" in excinfo.value.args[0]


def test_train_command(gap_binaries, train_data):
    calls = gap_binaries()
    gap = GAPotential()
    assert gap.train(*train_data, l_max=8, n_sparse=50, use_stress=True) == 0
    (cmd,) = calls
    assert cmd[:2] == ["gap_fit", "at_file=train.xyz"]
    gap_arg = next(a for a in cmd if a.startswith("gap="))
    assert gap_arg.startswith("gap={soap l_max=8 n_max=10 atom_sigma=0.5 zeta=4 cutoff=4.0")
    assert "n_sparse=50" in gap_arg
    assert gap_arg.endswith("add_species=T}")
    assert "default_sigma={0.0005 0.1 0.05 0.01}" in cmd
    for arg in [
        "energy_parameter_name=dft_energy",
        "force_parameter_name=dft_force",
        "virial_parameter_name=dft_virial",
    ]:
        assert arg in cmd
    assert cmd[-1] == "gp_file=train.xml"
    assert "sparse_jitter=1e-08" in cmd
    # The training configurations are written for gap_fit
    assert calls.file("train.xyz").count("Properties=") == len(train_data[0])


def test_train_flags_off(gap_binaries, train_data):
    calls = gap_binaries()
    GAPotential().train(*train_data, use_energies=False, use_forces=False)
    assert not any("parameter_name" in a for a in calls[0])


def test_train_bad_sigma(gap_binaries, train_data):
    gap_binaries()
    with pytest.raises(ValueError, match="4 numbers"):
        GAPotential().train(*train_data, default_sigma=(0.1, 0.1))


def test_train_failure(gap_binaries, train_data):
    gap_binaries(handler=lambda args: (1, "SYSTEM ABORT: bad input\n", ""))
    with pytest.raises(RuntimeError, match="gap_fit exited with return code 1: SYSTEM ABORT: bad input"):
        GAPotential().train(*train_data)


def test_train_populates_param(trained_gap):
    assert trained_gap.elements == ["Mo"]
    assert trained_gap.param["potential_label"] == "GAP_2017_6_17_60_4_3_56_165"
    np.testing.assert_allclose(trained_gap.param["element_param"]["Mo"], SPARSE_X)


def test_write_param(trained_gap):
    pair_style, pair_coeff = trained_gap.write_param("pot.xml")
    assert pair_style == "pair_style        quip"
    assert pair_coeff == 'pair_coeff        * * pot.xml "Potential xml_label=GAP_2017_6_17_60_4_3_56_165" 42'
    np.testing.assert_allclose(np.loadtxt("Mo.soapparam"), SPARSE_X)
    assert 'sparseX_filename="Mo.soapparam"' in Path("pot.xml").read_text()


def test_write_param_without_param():
    with pytest.raises(RuntimeError, match="should be provided"):
        GAPotential().write_param()


def test_from_config_other_directory(trained_gap, tmp_path, monkeypatch):
    # Regression: from_config resolved the sparseX files relative to the cwd instead of the xml file.
    trained_gap.write_param("pot.xml")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    gap = GAPotential.from_config(str(tmp_path / "pot.xml"))
    assert gap.elements == ["Mo"]
    assert gap.param["potential_label"] == "GAP_2017_6_17_60_4_3_56_165"
    np.testing.assert_allclose(gap.param["element_param"]["Mo"], SPARSE_X)


def test_evaluate(trained_gap, gap_binaries, train_data):
    calls = gap_binaries()
    df_orig, df_predict = trained_gap.evaluate(*train_data, predict_stress=True)
    assert calls[0] == [
        "quip",
        "atoms_filename=original.xyz",
        "param_filename=predict.xml",
        "energy=T",
        "forces=T",
        "virial=T",
    ]
    assert df_orig.shape == df_predict.shape
    energies_orig = df_orig[df_orig["dtype"] == "energy"]["y_orig"].to_numpy()
    energies_pred = df_predict[df_predict["dtype"] == "energy"]["y_orig"].to_numpy()
    np.testing.assert_allclose(energies_pred, energies_orig + 1, atol=1e-5)
    forces_orig = df_orig[df_orig["dtype"] == "force"]["y_orig"].to_numpy()
    forces_pred = df_predict[df_predict["dtype"] == "force"]["y_orig"].to_numpy()
    np.testing.assert_allclose(forces_pred, 0.5 * forces_orig, atol=1e-5)


@pytest.mark.xfail(
    reason="GAPotential.save uses the removed ruamel.yaml dump() API, and param holds an ElementTree "
    "that YAML cannot represent; fixing needs a decision on the on-disk format.",
    strict=True,
    raises=AttributeError,
)
def test_save(trained_gap):
    filename = trained_gap.save("param.yaml")
    assert Path(filename).exists()
