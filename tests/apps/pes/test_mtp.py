# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.
from __future__ import annotations

import copy
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
from pymatgen.core import Lattice, Structure

from maml.apps.pes import MTPotential, _lammps, _mtp
from maml.apps.pes._mtp import MTini_params, _load_mtp_param, feed, min_interatomic_distance

CWD = os.getcwd()
test_datapool = loadfn(os.path.join(os.path.abspath(os.path.dirname(__file__)), "datapool.json"))
config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "MTP", "fitted.mtp")
config_file_4 = os.path.join(os.path.abspath(os.path.dirname(__file__)), "MTP", "fitted_4.mtp")


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
        datapool, _df = self.potential.read_cfgs("test.cfgs")
        assert len(self.test_pool) == len(datapool)
        for data1, data2 in zip(self.test_pool, datapool, strict=True):
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
        _energy, forces, stress = self.potential.predict_efs(self.test_struct)
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
    mtp = MTPotential(version="mlip-2")
    with pytest.raises(ValueError, match="No species given"):
        mtp.write_cfg("x.cfgs", datapool[:1])
    with pytest.raises(ValueError, match="No species given"):
        mtp.read_cfgs("x.cfgs")


@pytest.mark.parametrize(
    ("version", "header"),
    [("mlip-2", "PlusStress:  xx"), ("mlip-3", "PlusStress:  xx"), ("mlip-dev", "Stress:  xx")],
)
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
    mtp = MTPotential(version="mlip-2")
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
    mtp = MTPotential.from_config(config_file, elements=["Mo"], version="mlip-2")
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
        MTPotential(version="mlip-2").write_param()


@pytest.mark.parametrize("method", ["train", "evaluate"])
def test_missing_mlp(fake_mlp_bin, train_data, method):
    fake_mlp_bin(available=set())
    with pytest.raises(RuntimeError, match="mlp has not been found") as excinfo:
        getattr(MTPotential(version="mlip-2"), method)(*train_data)
    assert len(excinfo.value.args) == 1


def test_train(fake_mlp_bin, train_data):
    calls = fake_mlp_bin()
    mtp = MTPotential(version="mlip-2")
    assert mtp.train(*train_data, max_dist=4.5, radial_basis_size=6, max_iter=7, stress_weight=0) == 0
    assert mtp.elements == ["Mo"]
    # The minimum distance is computed in Python; `mlp mindist` is no longer called.
    assert [call[1] for call in calls] == ["train"]
    # MLIP-2 command line, unchanged by the addition of MLIP-3 support.
    assert calls[0] == [
        "mlp",
        "train",
        "08g.mtp",
        "train.cfgs",
        "--max-iter=7",
        "--trained-pot-name=08g_fitted.mtp",
        "--curr-pot-name=08g.mtp",
        "--energy-weight=1",
        "--force-weight=0.01",
        "--stress-weight=0",
        "--init-params=same",
        "--scale-by-force=0",
        "--bfgs-conv-tol=0.001",
        "--weighting=vibration",
    ]
    unfitted = calls.file("08g.mtp")
    assert "species_count = 1" in unfitted
    assert f"min_dist = {min_interatomic_distance(train_data[0]):f}" in unfitted
    assert "max_dist = 4.500000" in unfitted
    assert "radial_basis_size = 6" in unfitted
    assert calls.file("train.cfgs").count("BEGIN_CFG") == len(train_data[0])
    assert mtp.param == MTPotential.from_config(config_file, ["Mo"], version="mlip-2").param


def test_train_no_unfitted_mtp(fake_mlp_bin, train_data):
    fake_mlp_bin()
    with pytest.raises(RuntimeError, match="No specific parameter file"):
        MTPotential(version="mlip-2").train(*train_data, unfitted_mtp="")


def test_train_failure(fake_mlp_bin, train_data):
    def handler(args):
        if args[1] == "train":
            return 2, "ERROR: Too few configurations\n", ""
        return fake_mlp(args)

    fake_mlp_bin(handler)
    with pytest.raises(RuntimeError, match="mlp train exited with return code 2") as excinfo:
        MTPotential(version="mlip-2").train(*train_data)
    msg = str(excinfo.value)
    assert "ERROR: Too few configurations" in msg
    assert 'version="mlip-2"' in msg
    assert "mlip-dev" in msg
    assert "gitlab.com/ashapeev/mlip-2" in msg


def test_train_failure_reports_stderr(fake_mlp_bin, train_data):
    # E.g. an MLIP version whose `mlp train` does not accept maml's flags.
    def handler(args):
        if args[1] == "train":
            return 1, "", "Unknown option --bfgs-conv-tol\n"
        return fake_mlp(args)

    fake_mlp_bin(handler)
    with pytest.raises(RuntimeError, match="mlp train exited with return code 1") as excinfo:
        MTPotential(version="mlip-2").train(*train_data)
    assert "mlp stderr" in str(excinfo.value)
    assert "Unknown option --bfgs-conv-tol" in str(excinfo.value)


def test_train_without_mlp_mindist(fake_mlp_bin, train_data):
    # Regression for #601: an `mlp` without a `mindist` command (e.g. MLIP-3) printed an error
    # message whose text maml then tried to parse as a float.
    def handler(args):
        if args[1] == "mindist":
            return 0, "Error: command mindist does not exist.\n", ""
        return fake_mlp(args)

    calls = fake_mlp_bin(handler)
    mtp = MTPotential(version="mlip-2")
    assert mtp.train(*train_data) == 0
    assert all(call[1] != "mindist" for call in calls)
    assert f"min_dist = {min_interatomic_distance(train_data[0]):f}" in calls.file("08g.mtp")


def _nan_fitted_mtp(trained):
    """Write a fitted potential whose coefficients went NaN, as `mlp train` does on divergence."""
    lines = Path(config_file).read_text().rstrip("\n").split("\n")
    lines[-2] = "species_coeffs = {nan}"
    lines[-1] = "moment_coeffs = {nan, -nan, nan, nan, nan, nan, nan, nan, nan}"
    Path(trained).write_text("\n".join(lines))


def test_train_diverged(fake_mlp_bin, train_data):
    # Regression for #702: a diverged fit wrote `{nan, ...}` coefficients and train() crashed with
    # an unhelpful json.decoder.JSONDecodeError.
    def handler(args):
        if args[1] == "train":
            trained = next(a.split("=", 1)[1] for a in args if a.startswith("--trained-pot-name="))
            _nan_fitted_mtp(trained)
            return 0, "BFGS iter 1: f=nan\nBFGS iter 2: f=nan\nTraining complete\n", ""
        return fake_mlp(args)

    fake_mlp_bin(handler)
    mtp = MTPotential(version="mlip-2")
    with pytest.raises(RuntimeError, match="MTP training diverged") as excinfo:
        mtp.train(*train_data)
    msg = str(excinfo.value)
    assert "non-finite" in msg
    assert "mlp stdout" in msg
    assert "BFGS iter 2: f=nan" in msg
    assert mtp.param is None


@pytest.mark.parametrize(
    "last_line",
    [
        "moment_coeffs = {1.0, inf}",
        "moment_coeffs = {1.0, -Infinity}",
        "moment_coeffs = {1.0, NaN}",
        "moment_coeffs = {1.0, 2.0",  # truncated / unparsable
    ],
)
def test_train_unusable_fitted_mtp(fake_mlp_bin, train_data, last_line):
    def handler(args):
        if args[1] == "train":
            trained = next(a.split("=", 1)[1] for a in args if a.startswith("--trained-pot-name="))
            lines = Path(config_file).read_text().rstrip("\n").split("\n")
            Path(trained).write_text("\n".join([*lines[:-1], last_line]))
            return 0, "done\n", ""
        return fake_mlp(args)

    fake_mlp_bin(handler)
    with pytest.raises(RuntimeError, match="MTP training diverged"):
        MTPotential(version="mlip-2").train(*train_data)


def test_train_missing_fitted_mtp(fake_mlp_bin, train_data):
    def handler(args):
        if args[1] == "train":
            return 0, "something went wrong\n", ""
        return fake_mlp(args)

    fake_mlp_bin(handler)
    with pytest.raises(RuntimeError, match=r"did not write the fitted potential 08g_fitted\.mtp") as excinfo:
        MTPotential(version="mlip-2").train(*train_data)
    assert "something went wrong" in str(excinfo.value)


def test_from_config_keeps_order_and_warns():
    with pytest.warns(UserWarning, match="differ from the order MTPotential.train uses"):
        mtp = MTPotential.from_config(config_file_4, elements=["Nb", "O", "Li", "Cl"])
    assert mtp.elements == ["Nb", "O", "Li", "Cl"]
    assert mtp.param["species_coeffs"] is not None


@pytest.mark.parametrize("elements", [["Li", "Nb", "Cl", "O"], ("Li", "Nb", "Cl", "O")])
def test_from_config_training_order_no_warning(recwarn, elements):
    mtp = MTPotential.from_config(config_file_4, elements=elements)
    assert mtp.elements == ["Li", "Nb", "Cl", "O"]
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_from_config_default_element_ordering(recwarn):
    mtp = MTPotential.from_config(config_file_4, elements=["Nb", "O", "Li", "Cl"], default_element_ordering=True)
    assert mtp.elements == ["Li", "Nb", "Cl", "O"]
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize(
    ("elements", "match"),
    [
        (["Li", "Nb", "Cl"], "species_count = 4, but 3 elements"),
        (["Li", "Li", "Nb", "Cl", "O"], "duplicates"),
        (["Li", "Li", "Nb", "Cl"], "duplicates"),
    ],
)
def test_from_config_invalid_elements(elements, match):
    with pytest.raises(ValueError, match=match):
        MTPotential.from_config(config_file_4, elements=elements)


def test_from_config_missing_species_count(in_tmp):
    text = Path(config_file).read_text()
    Path("no_count.mtp").write_text("\n".join(line for line in text.splitlines() if "species_count" not in line))
    with pytest.raises(ValueError, match="no species_count"):
        MTPotential.from_config("no_count.mtp", elements=["Mo"])


def test_from_config_nan(in_tmp):
    _nan_fitted_mtp("nan.mtp")
    with pytest.raises(ValueError, match="non-finite"):
        MTPotential.from_config("nan.mtp", elements=["Mo"])


@pytest.mark.parametrize(
    ("structures", "expected"),
    [
        # Simple cubic, one atom: only distances to its own periodic images, = a.
        ([Structure(Lattice.cubic(2.5), ["Mo"], [[0, 0, 0]])], 2.5),
        # Conventional bcc: body-centre to corner, = sqrt(3) / 2 * a.
        ([Structure(Lattice.cubic(3.16), ["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]])], np.sqrt(3) / 2 * 3.16),
        # Primitive (non-orthogonal) bcc, one atom: nearest self-image at sqrt(3) / 2 * a.
        (
            [
                Structure(
                    Lattice(np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) * 3.16 / 2),
                    ["Mo"],
                    [[0, 0, 0]],
                )
            ],
            np.sqrt(3) / 2 * 3.16,
        ),
        # Pair across the periodic boundary along x: 0.1 * 10 = 1 A apart through the image, the
        # in-cell separation (9 A) is larger. Minimum over several structures is returned.
        (
            [
                Structure(Lattice.cubic(10), ["Ni", "Mo"], [[0.05, 0, 0], [0.95, 0, 0]]),
                Structure(Lattice.cubic(3), ["Ni"], [[0, 0, 0]]),
            ],
            1.0,
        ),
    ],
)
def test_min_interatomic_distance(structures, expected):
    assert min_interatomic_distance(structures) == pytest.approx(expected, abs=1e-8)


def test_min_interatomic_distance_empty():
    with pytest.raises(ValueError, match="At least one structure"):
        min_interatomic_distance([])


@pytest.mark.parametrize(("version", "shift"), [("mlip-2", 1.0), ("mlip-dev", 2.0)])
def test_evaluate(fake_mlp_bin, train_data, version, shift):
    calls = fake_mlp_bin()
    mtp = MTPotential.from_config(config_file, elements=["Mo"], version="mlip-2")
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
    mtp = MTPotential.from_config(config_file, elements=["Mo"], version="mlip-2")
    with pytest.raises(RuntimeError, match=r"mlp exited with return code 1$"):
        mtp.evaluate(*train_data)


# ---------------------------------------------------------------------------
# MLIP-3. The fake below follows the `mlp` command line of MLIP-3 (commit 7fe598d,
# src/mlp/mlp_commands.cpp, src/common/utils.cpp and src/mlp/mlp.cpp). It was written from the
# source code; no MLIP-3 binary was run.
# ---------------------------------------------------------------------------

# Options read by `mlp train` (MTPR_trainer, AnyTrainer and the train command body).
MLIP3_TRAIN_OPTIONS = {
    "save_to",
    "energy_weight",
    "force_weight",
    "stress_weight",
    "penalty_weight",
    "scale_by_force",
    "select_factor",
    "weight_scaling",
    "weight_scaling_forces",
    "iteration_limit",
    "skip_preinit",
    "tolerance",
    "init_random",
    "no_mindist_update",
    "al_mode",
    "log",
}
MLIP3_BOOLS = {"TRUE", "True", "true", "FALSE", "False", "false"}

# MLIP-3 appends the active-learning state to the trained potential: text weights, then binary
# maxvol matrices. The bytes below are not valid UTF-8 and include "nan", which must be ignored.
MLIP3_SELECTION_STATE = (
    b"\n#MVS_v1.1\nenergy_weight 1.0000000000000000e+00\nforce_weight 0.0000000000000000e+00\n"
    b"stress_weight 0.0000000000000000e+00\nsite_en_weight 0.0000000000000000e+00\nweight_scaling 2\n#"
    + bytes(range(256))
    + b"nan\n#\n"
)


def _parse_mlip3_argv(argv):
    """Split argv like MLIP-3's ParseOptions: `--key=value` options, `--key` = "TRUE", rest args."""
    args, opts = [], {}
    for word in argv[2:]:
        if word.startswith("--"):
            key, sep, value = word[2:].partition("=")
            opts[key] = value if sep else "TRUE"
        else:
            args.append(word)
    return args, opts


def _relabel(text, types):
    """Replace the 0..N-1 radial_coeffs labels of an .mtp file with the atom types `types`."""
    return re.sub(
        r"^(\s*)(\d+)-(\d+)$",
        lambda m: f"{m.group(1)}{types[int(m.group(2))]}-{types[int(m.group(3))]}",
        text,
        flags=re.M,
    )


def _cfg_types(cfg_text):
    """Atom types in a .cfgs file, sorted (the species order MLIP-3 trains with)."""
    types = set()
    for block in re.findall(r"AtomData:.*?\n(.*?)\n Energy", cfg_text, re.S):
        types.update(int(line.split()[1]) for line in block.split("\n"))
    return sorted(types)


def _mlip3_trained_potential(types=None):
    """The fitted test potential as MLIP-3 would save it: coefficients in scientific notation,
    radial_coeffs labelled with the training atom types, followed by the selection state."""
    text = Path(config_file).read_text()
    if types is not None:
        text = _relabel(text, types)
    lines = text.rstrip("\n").split("\n")
    for i in (-2, -1):
        key, value = lines[i].split(" = ")
        numbers = [float(x) for x in value.strip("{}").split(",")]
        lines[i] = f"{key} = {{{', '.join(f'{x:.15e}' for x in numbers)}}}"
    return "\n".join(lines).encode() + MLIP3_SELECTION_STATE


def fake_mlp3(argv, strict=True):
    command = argv[1]
    args, opts = _parse_mlip3_argv(argv)
    if command == "train":
        if len(args) != 2:
            return 0, "mlp train: 2 arguments are required\n", ""
        # Real MLIP-3 silently ignores unknown options; by default the fake rejects them to catch typos.
        if strict:
            assert set(opts) <= MLIP3_TRAIN_OPTIONS, set(opts) - MLIP3_TRAIN_OPTIONS
        for key in ("iteration_limit", "weight_scaling", "weight_scaling_forces"):
            if key in opts:
                int(opts[key])
        for key in ("energy_weight", "force_weight", "stress_weight", "scale_by_force", "tolerance"):
            if key in opts:
                float(opts[key])
        for key in ("init_random", "no_mindist_update", "skip_preinit"):
            if key in opts:
                assert opts[key] in MLIP3_BOOLS, opts[key]
        # MLIP-3 sorts the training atom types and uses them as the species labels. The test
        # potential has one species, so the training set must have one atom type.
        types = _cfg_types(Path(args[1]).read_text())
        assert len(types) == 1, types
        Path(opts.get("save_to") or args[0]).write_bytes(_mlip3_trained_potential(types))
        return 0, "training complete\n", ""
    if command == "calculate_efs":
        if len(args) != 2:
            return 0, "mlp calculate_efs: 2 arguments required\n", ""
        output = opts.get("output_filename") or args[1]
        Path(output).write_text(_shift_energies(Path(args[1]).read_text(), 3.0))
        return 0, "MTP loaded from fitted.mtp\nEFS calculation complete\n", ""
    if command == "mindist":
        return 0, "Global mindist: 2.25\nmindist(0, 0) = 2.25\n", ""
    # mlp prints this and still exits with 0 (main() ignores ExecuteCommand's return value).
    return 0, f"Error: command {command} does not exist.\n", ""


def test_invalid_version():
    with pytest.raises(ValueError, match="Unsupported mlip version 'mlip-4'"):
        MTPotential(version="mlip-4")


def test_train_mlip3(fake_mlp_bin, train_data):
    calls = fake_mlp_bin(fake_mlp3)
    mtp = MTPotential(version="mlip-3")
    assert mtp.train(*train_data, max_dist=4.5, radial_basis_size=6, max_iter=7, stress_weight=0) == 0
    # No `mlp mindist` call; min_dist is computed in Python (and MLIP-3 also updates it itself).
    assert calls == [
        [
            "mlp",
            "train",
            "08g.mtp",
            "train.cfgs",
            "--save_to=08g_fitted.mtp",
            "--iteration_limit=7",
            "--energy_weight=1",
            "--force_weight=0.01",
            "--stress_weight=0",
            "--init_random=false",
            "--scale_by_force=0",
            "--tolerance=0.001",
            "--weight_scaling=1",
            "--weight_scaling_forces=0",
        ]
    ]
    unfitted = calls.file("08g.mtp")
    assert f"min_dist = {min_interatomic_distance(train_data[0]):f}" in unfitted
    assert "max_dist = 4.500000" in unfitted
    # MLIP-3 needs PlusStress blocks, the format its cfg reader shares with MLIP-2.
    assert "PlusStress:  xx" in calls.file("train.cfgs")
    # MLIP-3 gets atomic numbers as atom types and records them in the fitted potential.
    assert _cfg_types(calls.file("train.cfgs")) == [42]
    assert mtp.atomic_number_types
    assert mtp.elements == ["Mo"]
    expected = MTPotential.from_config(config_file, ["Mo"], version="mlip-2").param
    assert mtp.param["safe"] == _relabel("\n".join(expected["safe"]), [42]).split("\n")
    assert not any("MVS" in line for line in mtp.param["safe"])
    np.testing.assert_allclose(mtp.param["species_coeffs"], expected["species_coeffs"], rtol=1e-14)
    np.testing.assert_allclose(mtp.param["moment_coeffs"], expected["moment_coeffs"], rtol=1e-14)


@pytest.mark.parametrize(
    ("weighting", "init_params", "expected"),
    [
        ("vibrations", "same", ["--init_random=false", "--weight_scaling=1", "--weight_scaling_forces=0"]),
        ("molecules", "random", ["--init_random=true", "--weight_scaling=0", "--weight_scaling_forces=0"]),
        ("structures", "same", ["--init_random=false", "--weight_scaling=2", "--weight_scaling_forces=1"]),
    ],
)
def test_train_mlip3_option_mapping(fake_mlp_bin, train_data, weighting, init_params, expected):
    calls = fake_mlp_bin(fake_mlp3)
    MTPotential(version="mlip-3").train(*train_data, weighting=weighting, init_params=init_params)
    for arg in expected:
        assert arg in calls[0]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"weighting": "atoms"}, "weighting must be one of"), ({"init_params": "zeros"}, "init_params must be")],
)
def test_train_mlip3_invalid_options(fake_mlp_bin, train_data, kwargs, match):
    calls = fake_mlp_bin(fake_mlp3)
    with pytest.raises(ValueError, match=match):
        MTPotential(version="mlip-3").train(*train_data, **kwargs)
    assert calls == []


def test_train_mlip3_diverged(fake_mlp_bin, train_data):
    def handler(argv):
        if argv[1] == "train":
            fake_mlp3(argv)
            trained = Path("08g_fitted.mtp")
            trained.write_bytes(trained.read_bytes().replace(b"moment_coeffs = {", b"moment_coeffs = {-nan, ", 1))
            return 0, "training complete\n", ""
        return fake_mlp3(argv)

    fake_mlp_bin(handler)
    with pytest.raises(RuntimeError, match="MTP training diverged"):
        MTPotential(version="mlip-3").train(*train_data)


def test_evaluate_mlip3(fake_mlp_bin, train_data):
    calls = fake_mlp_bin(fake_mlp3)
    mtp = MTPotential.from_config(config_file, elements=["Mo"], version="mlip-3")
    df_orig, df_predict = mtp.evaluate(*train_data)
    assert calls == [["mlp", "calculate_efs", "fitted.mtp", "original.cfgs", "--output_filename=predict.cfgs"]]
    assert "mlip.ini" not in calls.files[0]
    assert Path(config_file).read_text().rstrip("\n") == calls.file("fitted.mtp").rstrip("\n")
    is_energy = df_orig["dtype"] == "energy"
    np.testing.assert_allclose(df_predict[is_energy]["y_orig"], df_orig[is_energy]["y_orig"] + 3.0, atol=1e-6)
    np.testing.assert_allclose(df_predict[~is_energy]["y_orig"], df_orig[~is_energy]["y_orig"])


def test_from_config_mlip3_trained_file(in_tmp):
    Path("trained.almtp").write_bytes(_mlip3_trained_potential())
    mtp = MTPotential.from_config("trained.almtp", elements=["Mo"], version="mlip-3")
    assert mtp.version == "mlip-3"
    np.testing.assert_allclose(mtp.param["species_coeffs"], [-9.118920883237898], rtol=1e-14)
    assert len(mtp.param["moment_coeffs"]) == 9


def test_train_mlip2_relative_types(fake_mlp_bin, train_data):
    calls = fake_mlp_bin()
    mtp = MTPotential(version="mlip-2")
    mtp.train(*train_data)
    assert not mtp.atomic_number_types
    assert _cfg_types(calls.file("train.cfgs")) == [0]


def test_train_mlip3_atomic_number_limit(fake_mlp_bin, train_data):
    calls = fake_mlp_bin(fake_mlp3)
    structures = [s.copy() for s in train_data[0]]
    for s in structures:
        s.replace_species({"Mo": "Fm"})
    with pytest.raises(ValueError, match="atomic numbers up to 99"):
        MTPotential(version="mlip-3").train(structures, *train_data[1:])
    assert calls == []


def test_train_mlip3_unexpected_labels(fake_mlp_bin, train_data):
    def handler(argv):
        if argv[1] == "train":
            Path("08g_fitted.mtp").write_bytes(_mlip3_trained_potential())
            return 0, "training complete\n", ""
        return fake_mlp3(argv)

    fake_mlp_bin(handler)
    with pytest.raises(RuntimeError, match=r"has species types \[0\], but the training set contains \['Mo'\]"):
        MTPotential(version="mlip-3").train(*train_data)


def _labelled_4(path, types=(3, 41, 17, 8)):
    """fitted_4.mtp (Li, Nb, Cl, O) as fitted on atomic-number atom types."""
    Path(path).write_text(_relabel(Path(config_file_4).read_text(), list(types)))
    return path


@pytest.mark.parametrize("elements", [None, ["Li", "Nb", "Cl", "O"], ["O", "Cl", "Nb", "Li"]])
def test_from_config_reads_elements(in_tmp, recwarn, elements):
    mtp = MTPotential.from_config(_labelled_4("z.mtp"), elements=elements)
    # The file order wins; with atomic-number types the given order does not matter.
    assert mtp.elements == ["Li", "Nb", "Cl", "O"]
    assert mtp.atomic_number_types
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_from_config_reads_elements_mlip2(in_tmp):
    mtp = MTPotential.from_config(_labelled_4("z.mtp"), version="mlip-2")
    assert mtp.elements == ["Li", "Nb", "Cl", "O"]
    # MLIP-2 ignores the labels and indexes species by position.
    assert not mtp.atomic_number_types
    assert mtp._atom_type("Cl") == 2


def test_from_config_elements_mismatch(in_tmp):
    with pytest.raises(ValueError, match=r"atomic numbers \[3, 41, 17, 8\] \(\['Li', 'Nb', 'Cl', 'O'\]\)"):
        MTPotential.from_config(_labelled_4("z.mtp"), elements=["Li", "Nb", "Cl", "S"])


def test_from_config_requires_elements_without_labels():
    with pytest.raises(ValueError, match=r"numbers its species 0\.\.3 and does not record their elements"):
        MTPotential.from_config(config_file_4)


def test_write_read_cfgs_atomic_number_types(in_tmp, datapool):
    mtp = MTPotential.from_config(_labelled_4("z.mtp"))
    structure = datapool[0]["structure"].copy()
    structure.replace_species({"Mo": "Nb"})
    structure[0] = "O"
    pool = [{"structure": structure, "outputs": datapool[0]["outputs"]}]
    text = Path(mtp.write_cfg("test.cfgs", pool)).read_text()
    assert _cfg_types(text) == [8, 41]
    data_pool, _ = mtp.read_cfgs("test.cfgs")
    assert Structure.from_dict(data_pool[0]["structure"]).composition == structure.composition
    other = MTPotential.from_config(_labelled_4("other.mtp", types=(3, 42, 17, 8)))
    with pytest.raises(ValueError, match="contains atom type 41"):
        other.read_cfgs("test.cfgs")


def test_evaluate_mlip3_atomic_number_types(fake_mlp_bin, train_data):
    calls = fake_mlp_bin(fake_mlp3)
    Path("mo.mtp").write_bytes(_mlip3_trained_potential([42]))
    mtp = MTPotential.from_config("mo.mtp")
    assert mtp.elements == ["Mo"]
    df_orig, df_predict = mtp.evaluate(*train_data)
    assert _cfg_types(calls.file("original.cfgs")) == [42]
    is_energy = df_orig["dtype"] == "energy"
    np.testing.assert_allclose(df_predict[is_energy]["y_orig"], df_orig[is_energy]["y_orig"] + 3.0, atol=1e-6)


def test_write_ini_mlip3_not_implemented(in_tmp):
    mtp = MTPotential.from_config(config_file, elements=["Mo"], version="mlip-3")
    with pytest.raises(NotImplementedError, match=r"pair_style line\. Use write_param"):
        mtp.write_ini()
    assert not Path("mlip.ini").exists()


# ---------------------------------------------------------------------------
# MLIP-3 LAMMPS interface (interface-lammps-mlip-3, commit 68f02dd9, built against MLIP-3 commit
# 7fe598d). The emulation below follows its source code (LAMMPS/USER-MLIP/pair_MLIP.cpp, MLIP-3's
# src/external/interface.cpp and MLMTPR::Load/type_from_atomic_number); no binary was run.
# ---------------------------------------------------------------------------


def _parse_mlip3_pair_style(in_file):
    """Settings of `pair_style mlip key=value ...` as PairMLIP::settings builds them."""
    lines = [line.split() for line in Path(in_file).read_text().splitlines()]
    (pair_style,) = [line for line in lines if line[:2] == ["pair_style", "mlip"]]
    (pair_coeff,) = [line for line in lines if line[:1] == ["pair_coeff"]]
    assert pair_coeff[1:] == ["*", "*"]
    settings = {"mlip": "true", "calculate_efs": "true"}
    for arg in pair_style[2:]:
        assert arg.count("=") == 1, arg
        key, value = arg.split("=")
        settings["mlip:load_from" if key == "load_from" else key] = value
    return settings


def _mlip3_species_labels(potential):
    """MLMTPR::Load: species labels (atomic_numbers) in order of first appearance."""
    text = Path(potential).read_bytes().split(b"\n#MVS_v", 1)[0].decode()
    labels = []
    for line in text.splitlines():
        if match := re.match(r"^\s*(\d+)-(\d+)\s*$", line):
            for label in map(int, match.groups()):
                if label not in labels:
                    labels.append(label)
    return labels


def make_mlip3_lammps(species_elements):
    """Fake LAMMPS with MLIP-3's pair_style mlip. `species_elements[k]` is the element the fitted
    potential's species k was trained on. Checks that every atom is evaluated with its own species."""
    from pymatgen.io.lammps.data import LammpsData

    def handler(args):
        settings = _parse_mlip3_pair_style(args[-1])
        labels = _mlip3_species_labels(settings["mlip:load_from"])
        assert len(labels) == len(species_elements)
        data = LammpsData.from_file("data.static", atom_style="charge")
        for atom_type, site in zip(data.atoms["type"], data.structure, strict=True):
            # interface.cpp passes type - 1; type_from_atomic_number throws if the label is absent.
            assert atom_type - 1 in labels, f"Atomic number {atom_type - 1} is not present in the MTP potential!"
            assert species_elements[labels.index(atom_type - 1)] == str(site.specie)
        n_atoms = len(data.structure)
        Path("energy.txt").write_text(f"{-1.5 * n_atoms}\n")
        header = f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{n_atoms}\nITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\n"
        Path("force.dump").write_text(header + "ITEM: ATOMS\n" + "0.0 0.0 0.0\n" * n_atoms)
        Path("stress.txt").write_text("1 2 3 0 0 0\n")

    return handler


@pytest.fixture
def li_nb_cl_o():
    return Structure(
        Lattice.cubic(4.0), ["O", "Li", "Cl", "Nb"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0.5]]
    )


def _radial_labels(path):
    return re.findall(r"^\s*(\d+-\d+)\s*$", Path(path).read_text(), flags=re.M)


@pytest.mark.parametrize(
    ("make_mtp", "species_elements"),
    [
        # Fitted on atomic-number types, in an order that is not the LAMMPS type order.
        (lambda: MTPotential.from_config(_labelled_4("z.mtp", types=(8, 41, 3, 17))), ["O", "Nb", "Li", "Cl"]),
        # Fitted on 0..N-1 types with a non-electronegativity element order.
        (
            lambda: MTPotential.from_config(config_file_4, elements=["Nb", "O", "Li", "Cl"], version="mlip-3"),
            ["Nb", "O", "Li", "Cl"],
        ),
        # Fitted by MLIP-2 in the electronegativity order.
        (
            lambda: MTPotential.from_config(config_file_4, elements=["Li", "Nb", "Cl", "O"], version="mlip-3"),
            ["Li", "Nb", "Cl", "O"],
        ),
    ],
)
def test_write_param_mlip3(in_tmp, recwarn, make_mtp, species_elements):
    mtp = make_mtp()
    before = copy.deepcopy(mtp.param)
    assert mtp.lammps_elements() == ["Li", "Nb", "Cl", "O"]
    pair_style, pair_coeff = mtp.write_param(fitted_mtp="pot.almtp")
    assert pair_style == "pair_style        mlip load_from=pot.almtp"
    assert pair_coeff == "pair_coeff        * *"
    assert not Path("mlip.ini").exists()
    # Species k is labelled with the LAMMPS type (minus one) of its element.
    lammps_type = {el: i for i, el in enumerate(mtp.lammps_elements())}
    assert _mlip3_species_labels("pot.almtp") == [lammps_type[el] for el in species_elements]
    # Only the labels change: same blocks, same coefficients.
    written = _load_mtp_param("pot.almtp")
    assert written["species_coeffs"] == mtp.param["species_coeffs"]
    assert written["moment_coeffs"] == mtp.param["moment_coeffs"]
    assert len(_radial_labels("pot.almtp")) == 16
    original = [line for line in mtp.param["safe"] if not re.match(r"^\s*\d+-\d+\s*$", line)]
    assert [line for line in written["safe"] if not re.match(r"^\s*\d+-\d+\s*$", line)] == original
    # The potential itself is not modified, so evaluate() keeps using its own atom types.
    assert mtp.param == before


def test_write_param_mlip3_single_species(in_tmp):
    mtp = MTPotential.from_config(config_file, elements=["Mo"], version="mlip-3")
    assert mtp.selection_state is None
    assert mtp.write_param()[0] == "pair_style        mlip load_from=fitted.mtp"
    assert Path("fitted.mtp").read_text().rstrip("\n") == Path(config_file).read_text().rstrip("\n")


def test_write_param_mlip3_extrapolation_control(in_tmp):
    Path("trained.almtp").write_bytes(_mlip3_trained_potential([42]))
    mtp = MTPotential.from_config("trained.almtp")
    assert mtp.selection_state == MLIP3_SELECTION_STATE
    pair_style, _ = mtp.write_param(
        fitted_mtp="pot.almtp",
        extrapolation_control=True,
        threshold_save=2,
        threshold_break=10.0,
        save_extrapolative_to="preselected.cfg",
    )
    assert pair_style.split()[2:] == [
        "load_from=pot.almtp",
        "extrapolation_control=true",
        "threshold_save=2",
        "threshold_break=10.0",
        "save_extrapolative_to=preselected.cfg",
    ]
    settings = _parse_mlip3_pair_style_line(pair_style)
    assert settings["mlip:load_from"] == "pot.almtp"
    data = Path("pot.almtp").read_bytes()
    # Relabelled to LAMMPS type 1 (label 0); the active-learning state follows the parameters verbatim.
    assert _mlip3_species_labels("pot.almtp") == [0]
    assert data.endswith(MLIP3_SELECTION_STATE)
    assert data.count(b"#MVS_v") == 1
    assert _load_mtp_param("pot.almtp")["moment_coeffs"] == mtp.param["moment_coeffs"]


def _parse_mlip3_pair_style_line(pair_style):
    Path("in.test").write_text(f"{pair_style}\npair_coeff * *\n")
    return _parse_mlip3_pair_style("in.test")


def test_train_mlip3_keeps_selection_state(fake_mlp_bin, train_data):
    fake_mlp_bin(fake_mlp3)
    mtp = MTPotential(version="mlip-3")
    mtp.train(*train_data)
    assert mtp.selection_state == MLIP3_SELECTION_STATE
    pair_style, _ = mtp.write_param(extrapolation_control=True)
    assert "extrapolation_control=true" in pair_style
    assert Path("fitted.mtp").read_bytes().endswith(MLIP3_SELECTION_STATE)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"extrapolation_control": True}, "has none"),
        ({"extrapolation_control": "TRUE"}, "has none"),
        ({"load_from": "other.mtp"}, "load_from is set from fitted_mtp"),
        ({"save_extrapolative_to": "my file.cfg"}, "Invalid pair_style mlip argument"),
        ({"save_extrapolative_to": ""}, "Invalid pair_style mlip argument"),
        ({"threshold_save": "a=b"}, "Invalid pair_style mlip argument"),
    ],
)
def test_write_param_mlip3_invalid(in_tmp, kwargs, match):
    mtp = MTPotential.from_config(config_file, elements=["Mo"], version="mlip-3")
    with pytest.raises(ValueError, match=match):
        mtp.write_param(**kwargs)
    assert not Path("fitted.mtp").exists()


def test_write_param_mlip3_invalid_filename(in_tmp):
    mtp = MTPotential.from_config(config_file, elements=["Mo"], version="mlip-3")
    with pytest.raises(ValueError, match="must not contain whitespace"):
        mtp.write_param(fitted_mtp="my pot.mtp")


def test_write_param_mlip3_types_mismatch(in_tmp):
    mtp = MTPotential.from_config(_labelled_4("z.mtp"))
    mtp.elements = ["Li", "Nb", "Cl", "S"]
    with pytest.raises(ValueError, match=r"types \[8\] are not among the types"):
        mtp.write_param()


def test_write_param_mlip3_no_param():
    with pytest.raises(RuntimeError, match="parameters should be provided"):
        MTPotential(version="mlip-3").write_param()


@pytest.mark.parametrize(
    ("make_mtp", "species_elements"),
    [
        (lambda: MTPotential.from_config(_labelled_4("z.mtp", types=(8, 41, 3, 17))), ["O", "Nb", "Li", "Cl"]),
        (
            lambda: MTPotential.from_config(config_file_4, elements=["Nb", "O", "Li", "Cl"], version="mlip-3"),
            ["Nb", "O", "Li", "Cl"],
        ),
    ],
)
def test_predict_efs_mlip3(mock_binaries, in_tmp, recwarn, li_nb_cl_o, make_mtp, species_elements):
    mtp = make_mtp()
    calls = mock_binaries(_lammps, make_mlip3_lammps(species_elements))
    energy, forces, stress = mtp.predict_efs(li_nb_cl_o)
    assert calls == [["/fake/bin/lmp_serial", "-in", "in.efs"]]
    assert energy == pytest.approx(-6.0)
    assert forces.shape == (4, 3)
    assert len(stress) == 6


def test_predict_efs_mlip3_unrelabelled_fails(mock_binaries, in_tmp, li_nb_cl_o):
    # Without relabelling, the atomic-number labels are wrong for LAMMPS: LAMMPS types 1..4 are looked
    # up as labels 0..3. Here type 4 (O) silently becomes label 3, the Li species, and types 1-3 crash.
    mtp = MTPotential.from_config(_labelled_4("z.mtp", types=(8, 41, 3, 17)))
    mock_binaries(_lammps, make_mlip3_lammps(["O", "Nb", "Li", "Cl"]))
    mtp._write_param_mlip3 = lambda fitted_mtp, **kwargs: (
        mtp._write_mtp(fitted_mtp) or [f"pair_style mlip load_from={fitted_mtp}", "pair_coeff * *"]
    )
    with pytest.raises(AssertionError, match=r"is not present in the MTP potential|assert 'Li' == 'O'"):
        mtp.predict_efs(li_nb_cl_o)


def test_mlip2_version_with_mlip3_binary(fake_mlp_bin, train_data):
    # MLIP-3 ignores MLIP-2's --trained-pot-name and exits 0 for unknown commands, so the version
    # mismatch only shows up as missing output files.
    fake_mlp_bin(lambda argv: fake_mlp3(argv, strict=False))
    with pytest.raises(RuntimeError, match="did not write the fitted potential") as excinfo:
        MTPotential(version="mlip-2").train(*train_data)
    assert 'version="mlip-3"' in str(excinfo.value)

    mtp = MTPotential.from_config(config_file, elements=["Mo"], version="mlip-2")
    with pytest.raises(RuntimeError, match="mlp calc-efs did not write the predicted configurations") as excinfo:
        mtp.evaluate(*train_data)
    assert "Error: command calc-efs does not exist." in str(excinfo.value)


# ---------------------------------------------------------------------------
# Integration tests against a real MLIP-3 `mlp` (built in CI; skipped otherwise).
# ---------------------------------------------------------------------------

NOTEBOOK_DATA = Path(__file__).parents[3] / "notebooks" / "pes" / "mtp" / "data.json"


def _real_mlip3() -> bool:
    if not which("mlp"):
        return False
    import subprocess

    out = subprocess.run(["mlp", "list"], capture_output=True, text=True, check=False).stdout
    return "calculate_efs" in out


requires_mlip3 = pytest.mark.skipif(not _real_mlip3(), reason="real MLIP-3 mlp not on PATH")


@pytest.fixture(scope="module")
def notebook_data():
    data = loadfn(NOTEBOOK_DATA)
    return (
        [d["structure"] for d in data],
        [d["outputs"]["energy"] for d in data],
        [d["outputs"]["forces"] for d in data],
        [d["outputs"]["stress"] for d in data],
    )


@requires_mlip3
def test_real_mlip3_train_evaluate(in_tmp, notebook_data):
    # Regression for #702: the notebook example (no stresses, stress_weight=0) must train to a finite,
    # accurate potential with the default version.
    structures, energies, forces, stresses = notebook_data
    mtp = MTPotential()
    assert mtp.version == "mlip-3"
    mtp.train(structures, energies, forces, train_stresses=None, max_dist=5, stress_weight=0, max_iter=100)
    df_orig, df_pred = mtp.evaluate(structures, energies, forces, stresses)

    def per_atom_energy(df):
        e = df[df["dtype"] == "energy"]
        return (e["y_orig"] / e["n"]).to_numpy()

    def force_components(df):
        return df[df["dtype"] == "force"]["y_orig"].to_numpy()

    # The fitted potential records its elements as atomic-number species labels.
    mtp._write_mtp("trained.mtp")
    assert MTPotential.from_config("trained.mtp").elements == mtp.elements
    assert sorted(mtp.elements) == sorted({str(el) for s in structures for el in s.species})

    e_pred, f_pred = per_atom_energy(df_pred), force_components(df_pred)
    assert np.isfinite(e_pred).all()
    assert np.isfinite(f_pred).all()
    assert np.abs(per_atom_energy(df_orig) - e_pred).mean() < 0.02  # eV/atom
    assert np.abs(force_components(df_orig) - f_pred).mean() < 0.5  # eV/A


@requires_mlip3
def test_real_mlip3_rejects_mlip2_version(in_tmp, notebook_data):
    # MLIP-3 silently ignores MLIP-2 options (training with default weights), so the mismatch must be
    # reported instead of producing a bad or NaN potential.
    # MLIP-3 also ignores --max-iter, so keep the training set tiny.
    structures, energies, forces, _ = (x[:2] for x in notebook_data)
    with pytest.raises(RuntimeError, match='version="mlip-3"'):
        MTPotential(version="mlip-2").train(structures, energies, forces, None, max_dist=5, max_iter=5)


def _lammps_with_mlip3() -> bool:
    """Whether the default LAMMPS executable has MLIP-3's `pair_style mlip`."""
    import subprocess

    lmp = _lammps.get_default_lmp_exe()
    if not lmp:
        return False
    try:
        out = subprocess.run([lmp, "-h"], capture_output=True, text=True, check=False, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return re.search(r"\bmlip\b", out) is not None


requires_mlip3_lammps = pytest.mark.skipif(
    not (_real_mlip3() and _lammps_with_mlip3()),
    reason="real MLIP-3 mlp and LAMMPS with interface-lammps-mlip-3 not on PATH",
)


@requires_mlip3_lammps
def test_real_mlip3_lammps_matches_mlp(in_tmp, notebook_data):
    structures, energies, forces, _ = notebook_data
    # Se has a smaller atomic number but a larger electronegativity than Mo, so MLIP-3's species order
    # (sorted atom types) differs from the LAMMPS atom type order. The fit quality is irrelevant here.
    structures = [s.copy() for s in structures]
    for s in structures:
        s.replace_species({"Ni": "Se"})
    mtp = MTPotential()
    mtp.train(structures, energies, forces, None, max_dist=5, stress_weight=0, max_iter=50)
    assert mtp.elements == ["Se", "Mo"]
    assert mtp.lammps_elements() == ["Mo", "Se"]
    assert mtp.selection_state

    structure = structures[0]
    _, df_mlp = mtp.evaluate([structure], [energies[0]], [forces[0]])
    e_mlp = df_mlp[df_mlp["dtype"] == "energy"]["y_orig"].item()
    f_mlp = df_mlp[df_mlp["dtype"] == "force"]["y_orig"].to_numpy().reshape(-1, 3)

    energy, lmp_forces, stress = mtp.predict_efs(structure)
    assert energy == pytest.approx(e_mlp, abs=1e-3)
    np.testing.assert_allclose(lmp_forces, f_mlp, atol=1e-4)
    assert len(stress) == 6

    # Extrapolation control loads the active-learning state kept from training. ScratchDir does not
    # copy the working directory, so the potential is passed by absolute path.
    ff_settings = mtp.write_param(
        fitted_mtp=str(Path("pot.almtp").resolve()),
        extrapolation_control=True,
        threshold_save=2,
        threshold_break=100,
    )
    ((energy_ec, forces_ec, _),) = _lammps.EnergyForceStress(ff_settings).calculate([structure])
    assert energy_ec == pytest.approx(energy, abs=1e-8)
    np.testing.assert_allclose(forces_ec, lmp_forces, atol=1e-8)
