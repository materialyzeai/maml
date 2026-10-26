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
from pymatgen.core import Element, Structure

from maml.apps.pes import NNPotential, _nnp

CWD = os.getcwd()
DIR = os.path.abspath(os.path.dirname(__file__))
test_datapool = loadfn(os.path.join(DIR, "datapool.json"))
input_file = os.path.join(DIR, "NNP", "input.nn")
scaling_file = os.path.join(DIR, "NNP", "scaling.data")
weights_file = os.path.join(DIR, "NNP", "weights.data")


class NNPitentialTest(unittest.TestCase):
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
        self.potential = NNPotential(name="test")
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
        self.potential.write_cfgs("input.data", cfg_pool=self.test_pool)
        datapool, df = self.potential.read_cfgs("input.data")
        assert len(self.test_pool) == len(datapool)
        for data1, data2 in zip(self.test_pool, datapool):
            struct1 = data1["structure"]
            struct2 = Structure.from_dict(data2["structure"])
            assert struct1 == struct2
            energy1 = data1["outputs"]["energy"]
            energy2 = data2["outputs"]["energy"]
            assert abs(energy1 - energy2) < 0.001
            forces1 = np.array(data1["outputs"]["forces"])
            forces2 = data2["outputs"]["forces"]
            np.testing.assert_array_almost_equal(forces1, forces2)

    @unittest.skipIf(not which("nnp-train"), "No nnp-train cmd found.")
    def test_train(self):
        hidden_layers = [15, 15]
        activations = "t"
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
            atom_energy={"Mo": -4.14},
            r_cut=5.0,
            hidden_layers=hidden_layers,
            activations=activations,
            epochs=1,
        )
        assert self.potential.train_energy_rmse is not None
        assert self.potential.train_forces_rmse is not None
        assert self.potential.validation_energy_rmse is not None
        assert self.potential.validation_forces_rmse is not None

    @unittest.skipIf(not which("nnp-train"), "No nnp-train cmd found.")
    @unittest.skipIf(not which("nnp-predict"), "No nnp-train cmd found.")
    def test_evaluate(self):
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
            atom_energy={"Mo": -4.14},
            r_cut=5.0,
            epochs=1,
        )

        df_orig, df_tar = self.potential.evaluate(
            test_structures=self.test_structures,
            test_energies=self.test_energies,
            test_forces=self.test_forces,
            test_stresses=self.test_stresses,
        )
        assert df_orig.shape[0] == df_tar.shape[0]

    @unittest.skipIf(not which("nnp-train"), "No nnp-train cmd found.")
    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found.")
    def test_predict_efs(self):
        self.potential.train(
            train_structures=self.test_structures,
            train_energies=self.test_energies,
            train_forces=self.test_forces,
            train_stresses=self.test_stresses,
            atom_energy={"Mo": -4.14},
            r_cut=5.0,
            epochs=1,
        )
        energy, forces, stress = self.potential.predict_efs(self.test_struct)
        assert len(forces) == len(self.test_struct)
        assert len(stress) == 6

    def test_from_config(self):
        nnp = NNPotential.from_config(
            input_filename=input_file, scaling_filename=scaling_file, weights_filenames=[weights_file]
        )
        assert nnp.fitted


# ---------------------------------------------------------------------------
# Tests below run without n2p2; nnp-scaling/nnp-train/nnp-predict are faked.
# ---------------------------------------------------------------------------


def _input_nn_info(text):
    """Return (elements, number of symmetry functions per element, hidden layers, epochs) of input.nn."""
    rows = [line.split() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    elements = next(r[1:] for r in rows if r[0] == "elements")
    n_sf = {el: sum(1 for r in rows if r[0] == "symfunction_short" and r[1] == el) for el in elements}
    hidden = [int(n) for n in next(r[1:] for r in rows if r[0] == "global_nodes_short")]
    epochs = int(next(r[1] for r in rows if r[0] == "epochs"))
    return elements, n_sf, hidden, epochs


def _weights_lines(layer_sizes):
    lines, index = [], 1
    for layer in range(1, len(layer_sizes)):
        for i in range(layer_sizes[layer - 1]):
            for j in range(layer_sizes[layer]):
                lines.append(f"{0.01 * index:.16E} a {index:>9d} {layer - 1:>5d} {i + 1:>5d} {layer:>5d} {j + 1:>5d}")
                index += 1
        for j in range(layer_sizes[layer]):
            lines.append(f"{-0.01 * index:.16E} b {index:>9d} {layer:>5d} {j + 1:>5d}")
            index += 1
    return lines


def fake_n2p2(args):
    elements, n_sf, hidden, epochs = _input_nn_info(Path("input.nn").read_text())
    if args[0] == "nnp-scaling":
        rows = [
            f"{e + 1:>4d}{s + 1:>5d}  {0.1:.16E} {0.9:.16E} {0.5:.16E} {0.2:.16E}"
            for e, el in enumerate(elements)
            for s in range(n_sf[el])
        ]
        Path("scaling.data").write_text("# scaling\n" + "\n".join(rows) + "\n")
        return 0, "", ""
    if args[0] == "nnp-train":
        for el in elements:
            layer_sizes = [n_sf[el], *hidden, 1]
            name = f"weights.{str(Element(el).Z).zfill(3)}.{str(epochs).zfill(6)}.out"
            Path(name).write_text("# weights\n" + "\n".join(_weights_lines(layer_sizes)) + "\n")
        out = "\n".join(
            f"ENERGY {epoch:>5d} {0.1 / (epoch + 1):.5E} {0.2 / (epoch + 1):.5E}\n"
            f"FORCE  {epoch:>5d} {0.3 / (epoch + 1):.5E} {0.4 / (epoch + 1):.5E}"
            for epoch in range(epochs + 1)
        )
        return 0, out + "\n", ""
    if args[0] == "nnp-predict":
        text = Path("input.data").read_text()
        Path("output.data").write_text(text)
        return 0, "", ""
    raise AssertionError(f"unexpected command {args}")


@pytest.fixture
def fake_nnp_bin(mock_binaries, in_tmp):
    def install(handler=fake_n2p2, available=None):
        return mock_binaries(_nnp, handler, available=available)

    return install


@pytest.fixture
def loaded_nnp():
    return NNPotential.from_config(
        input_filename=input_file, scaling_filename=scaling_file, weights_filenames=[weights_file]
    )


def test_write_cfgs_units(in_tmp, datapool):
    nnp = NNPotential()
    nnp.write_cfgs("input.data", cfg_pool=datapool[:1])
    assert nnp.elements == ["Mo"]
    lines = Path("input.data").read_text().splitlines()
    assert lines[0] == "begin"
    assert lines[-1] == "end"
    energy = float(next(line for line in lines if line.startswith("energy")).split()[1])
    assert energy == pytest.approx(datapool[0]["outputs"]["energy"] * NNPotential.eV_to_Ha, abs=1e-6)
    a_bohr = float(lines[1].split()[1])
    assert a_bohr == pytest.approx(datapool[0]["structure"].lattice.matrix[0, 0] / NNPotential.bohr_to_angstrom)


def test_write_input_defaults_and_rewrite(in_tmp):
    nnp = NNPotential()
    nnp.elements = ["Mo"]
    nnp.write_input(atom_energy={"Mo": -4.14}, hidden_layers=[5, 4], normalize_nodes=True)
    first = Path("input.nn").read_text()
    assert nnp.fitted
    _, n_sf, hidden, epochs = _input_nn_info(first)
    assert hidden == [5, 4]
    assert epochs == 100
    # 1 radial eta x 9 shifts + 1 neighbour pair x (angular etas x lambdas x zetas)
    assert n_sf["Mo"] == nnp.num_symm_functions
    assert nnp.layer_sizes == [nnp.num_symm_functions, 5, 4, 1]
    assert "atom_energy                     Mo -4.14" in first
    assert "global_activation_short         t t l" in first
    assert "scale_symmetry_functions" in first
    # Rewriting from the stored parameters must reproduce the same file.
    # Regression: normalize_nodes used to be dropped on the second write.
    nnp.write_input()
    assert Path("input.nn").read_text() == first
    assert "normalize_nodes" in first


def test_write_input_integer_r_shift(in_tmp):
    # Regression: integer r_shift values made the in-place unit conversion fail.
    nnp = NNPotential()
    nnp.elements = ["Mo"]
    nnp.write_input(r_shift=[1, 2], r_etas=[0.5], a_etas=[0.1], zetas=[1.0], lambdas=[1, -1], r_cut=5)
    rows = [line.split() for line in Path("input.nn").read_text().splitlines() if line.startswith("symfunction_short")]
    radial = [r for r in rows if r[2] == "2"]
    angular = [r for r in rows if r[2] == "3"]
    assert len(radial) == 2
    assert len(angular) == 2
    assert float(radial[1][5]) == pytest.approx(2 / NNPotential.bohr_to_angstrom, abs=1e-6)
    assert float(radial[0][6]) == pytest.approx(5 / NNPotential.bohr_to_angstrom, abs=1e-6)
    assert nnp.num_symm_functions == 4


def test_load_input(in_tmp):
    nnp = NNPotential()
    nnp.load_input(input_file)
    assert nnp.elements == ["Mo"]
    param = nnp.param
    assert param["atom_energy"] == {"Mo": -4.14}
    assert param["hidden_layers"] == [8, 8]
    assert param["activations"] == "t"
    assert param["scale_features"] == "1"
    assert param["epochs"] == 10
    assert param["test_fraction"] == pytest.approx(1e-6)
    assert param["r_cut"] == pytest.approx(4.8)  # 9.0706854 bohr
    assert nnp.layer_sizes[1:] == [8, 8, 1]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", "0"),
        ("scale_symmetry_functions", "1"),
        ("center_symmetry_functions", "2"),
        ("scale_symmetry_functions_sigma", "3"),
    ],
)
def test_load_input_scale_features(in_tmp, line, expected):
    # Regression: anything but scale_symmetry_functions became the int 0, which then
    # crashed write_input() and silently dropped center/sigma scaling.
    text = Path(input_file).read_text().replace("scale_symmetry_functions\n", f"{line}\n" if line else "")
    Path("custom.nn").write_text(text)
    nnp = NNPotential()
    nnp.load_input("custom.nn")
    assert nnp.param["scale_features"] == expected
    nnp.fitted = True
    nnp.write_input()
    written = Path("input.nn").read_text().splitlines()
    if line:
        assert line in written
    else:
        assert not any("symmetry_functions" in w for w in written)


def test_load_input_normalize_nodes_roundtrip(in_tmp):
    text = (
        Path(input_file)
        .read_text()
        .replace("scale_max_short                 1.0\n", "scale_max_short 1.0\nnormalize_nodes\n")
    )
    Path("norm.nn").write_text(text)
    nnp = NNPotential()
    nnp.load_input("norm.nn")
    assert nnp.param["normalize_nodes"]
    nnp.fitted = True
    nnp.write_input()
    assert "normalize_nodes" in Path("input.nn").read_text().splitlines()


def test_load_weights_twice_does_not_duplicate(loaded_nnp):
    # Regression: re-loading weights for a specie appended to the existing layers, and the
    # hidden -> output layer weights were never loaded.
    n_layers = len(loaded_nnp.weights["Mo"])
    loaded_nnp.load_weights(weights_file, "Mo")
    assert len(loaded_nnp.weights["Mo"]) == n_layers == len(loaded_nnp.layer_sizes) - 1
    assert len(loaded_nnp.bs["Mo"]) == n_layers
    for layer, w in enumerate(loaded_nnp.weights["Mo"]):
        assert w.shape == (loaded_nnp.layer_sizes[layer], loaded_nnp.layer_sizes[layer + 1])
        assert loaded_nnp.bs["Mo"][layer].shape == (loaded_nnp.layer_sizes[layer + 1],)


def test_from_config_wrong_number_of_weights():
    with pytest.raises(ValueError, match="1 weights files should be given to Mo"):
        NNPotential.from_config(input_file, scaling_file, [weights_file, weights_file])


def test_write_param_without_parameters():
    with pytest.raises(RuntimeError, match="parameters should be provided"):
        NNPotential().write_param()


def test_write_param(in_tmp, loaded_nnp):
    pair_style, pair_coeff = loaded_nnp.write_param()
    assert pair_style.startswith("pair_style        nnp")
    assert pair_coeff == f"pair_coeff        * * {loaded_nnp.param['r_cut'] + 0.01}"
    written = np.loadtxt("weights.042.data", usecols=0)
    original = np.loadtxt(weights_file, usecols=0)
    np.testing.assert_allclose(written, original)
    scaling = np.loadtxt("scaling.data")
    np.testing.assert_allclose(scaling, np.loadtxt(scaling_file)[: len(scaling)])
    assert len(scaling) == loaded_nnp.num_symm_functions
    assert Path("input.nn").exists()


def test_write_param_scaling_format(in_tmp, loaded_nnp):
    # Regression: the last scaling column was padded with "." instead of spaces.
    loaded_nnp.scaling_param.iloc[0, 5] = "0.5"
    loaded_nnp.write_param()
    first = Path("scaling.data").read_text().splitlines()[0]
    assert "." * 3 not in first
    assert first.split()[-1] == "0.5"


@pytest.mark.parametrize(("binary", "method"), [("nnp-train", "train"), ("nnp-predict", "evaluate")])
def test_missing_binary(fake_nnp_bin, train_data, binary, method):
    fake_nnp_bin(available=set())
    with pytest.raises(RuntimeError, match="has not been found"):
        getattr(NNPotential(), method)(*train_data)


def test_train(fake_nnp_bin, train_data):
    # Regression: reading the training log used zopen() without a mode, which raises TypeError.
    calls = fake_nnp_bin()
    nnp = NNPotential()
    rc = nnp.train(*train_data, atom_energy={"Mo": -4.14}, hidden_layers=[3, 2], epochs=2, r_cut=5.0)
    assert rc == 0
    assert calls == [["nnp-scaling", "100"], ["nnp-train"]]
    np.testing.assert_allclose(nnp.train_energy_rmse, [0.1, 0.05, 0.1 / 3], rtol=1e-5)
    np.testing.assert_allclose(nnp.validation_energy_rmse, [0.2, 0.1, 0.2 / 3], rtol=1e-5)
    np.testing.assert_allclose(nnp.train_forces_rmse, [0.3, 0.15, 0.1], rtol=1e-5)
    np.testing.assert_allclose(nnp.validation_forces_rmse, [0.4, 0.2, 0.4 / 3], rtol=1e-5)
    assert [w.shape for w in nnp.weights["Mo"]] == [(nnp.num_symm_functions, 3), (3, 2), (2, 1)]
    assert len(nnp.scaling_param) == nnp.num_symm_functions
    assert calls.file("input.data").count("begin") == len(train_data[0])


@pytest.mark.parametrize("failing", ["nnp-scaling", "nnp-train"])
def test_train_failure(fake_nnp_bin, train_data, failing):
    def handler(args):
        if args[0] == failing:
            return 1, "", "Some output\nERROR: Unknown keyword\n"
        return fake_n2p2(args)

    fake_nnp_bin(handler)
    with pytest.raises(RuntimeError, match="n2p2 exited with return code 1: ERROR: Unknown keyword"):
        NNPotential().train(*train_data, epochs=1)


def test_evaluate(fake_nnp_bin, train_data, loaded_nnp):
    calls = fake_nnp_bin()
    df_orig, df_predict = loaded_nnp.evaluate(*train_data)
    assert calls == [["nnp-predict", "input.nn"]] * len(train_data[0])
    assert df_orig.shape == df_predict.shape
    np.testing.assert_allclose(df_predict["y_orig"], df_orig["y_orig"])


def test_evaluate_failure(fake_nnp_bin, train_data, loaded_nnp):
    fake_nnp_bin(lambda args: (1, "", "Segmentation fault\n"))
    with pytest.raises(RuntimeError, match="return code 1: Segmentation fault"):
        loaded_nnp.evaluate(*train_data)
