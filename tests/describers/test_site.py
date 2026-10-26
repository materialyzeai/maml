from __future__ import annotations

import itertools
import os
import tempfile
import unittest
from shutil import which

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Composition, Lattice, Molecule, Structure

from maml.describers import _site
from maml.describers._site import (
    BispectrumCoefficients,
    BPSymmetryFunctions,
    SiteElementProperty,
    SmoothOverlapAtomicPosition,
)


class BispectrumCoefficientsTest(unittest.TestCase):
    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found")
    def test_transform(self):
        s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]])
        profile = dict(Na=dict(r=0.3, w=0.9), Cl=dict(r=0.7, w=3.0))
        s *= [2, 2, 2]
        structures = [s] * 10
        for s in structures:
            n = np.random.randint(4)
            inds = np.random.randint(16, size=n)
            s.remove_sites(inds)

        bc_atom = BispectrumCoefficients(rcutfac=5, twojmax=3, element_profile=profile, quadratic=False, pot_fit=False)
        df_atom = bc_atom.transform(structures)
        for i, s in enumerate(structures):
            df_s = df_atom.xs(i, level="input_index")
            assert df_s.shape == (len(s), 8)
            assert df_s.equals(bc_atom.transform_one(s))

        bc_pot = BispectrumCoefficients(
            rcutfac=5, twojmax=3, element_profile=profile, quadratic=False, pot_fit=True, include_stress=True
        )
        df_pot = bc_pot.transform(structures)
        for i, s in enumerate(structures):
            df_s = df_pot.xs(i, level="input_index")
            assert df_s.shape == (1 + len(s) * 3 + 6, 18)
            assert df_s.equals(bc_pot.transform_one(s))
            sna = df_s.iloc[0]
            for specie in ["Na", "Cl"]:
                self.assertAlmostEqual(sna[specie, "n"], s.composition.fractional_composition[specie])
                np.testing.assert_array_equal(df_s[specie, "n"][1:], np.zeros(len(s) * 3 + 6))


class SmoothOverlapAtomicPositionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()
        os.chdir(cls.test_dir)

    def setUp(self):
        self.unary_struct = Structure.from_spacegroup("Im-3m", Lattice.cubic(3.4268), ["Li"], [[0, 0, 0]])
        self.binary_struct = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]]
        )
        self.describer = SmoothOverlapAtomicPosition(cutoff=4.8, l_max=8, n_max=8)

    @unittest.skipIf(not which("quip"), "No quip cmd found.")
    def test_transform_one(self):
        unary_descriptors = self.describer.transform_one(self.unary_struct)
        binary_descriptors = self.describer.transform_one(self.binary_struct)
        assert unary_descriptors.shape[0] == len(self.unary_struct)
        assert binary_descriptors.shape[0] == len(self.binary_struct)

    @unittest.skipIf(not which("quip"), "No quip cmd found.")
    def test_transform(self):
        unary_descriptors = self.describer.transform([self.unary_struct] * 3)
        assert unary_descriptors.shape[0] == len(self.unary_struct) * 3
        binary_descriptors = self.describer.transform([self.binary_struct] * 3)
        assert binary_descriptors.shape[0] == len(self.binary_struct) * 3


class BPSymmetryFunctionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()
        os.chdir(cls.test_dir)

    def setUp(self):
        self.unary_struct = Structure.from_spacegroup("Im-3m", Lattice.cubic(3.4268), ["Li"], [[0, 0, 0]])
        self.binary_struct = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]]
        )
        self.r_etas = [0.01, 0.02, 0.05]
        self.r_shift = [0]
        self.zetas = [1.0, 4.0]
        self.a_etas = [0.01, 0.05]
        self.lambdas = [-1, 1]
        self.describer = BPSymmetryFunctions(
            cutoff=4.8,
            r_etas=self.r_etas,
            r_shift=self.r_shift,
            lambdas=self.lambdas,
            a_etas=self.a_etas,
            zetas=self.zetas,
        )

    def test_transform_one(self):
        unary_descriptors = self.describer.transform_one(self.unary_struct)
        binary_descriptors = self.describer.transform_one(self.binary_struct)
        unary_elements = set(self.unary_struct.species)
        binary_elements = set(self.binary_struct.species)
        assert unary_descriptors.shape[0] == len(self.unary_struct)
        assert unary_descriptors.shape[1] == len(self.r_etas) * len(self.r_shift) * len(unary_elements) + len(
            self.a_etas
        ) * len(self.zetas) * len(self.lambdas) * len(list(itertools.combinations_with_replacement(unary_elements, 2)))
        assert binary_descriptors.shape[0] == len(self.binary_struct)
        assert binary_descriptors.shape[1] == len(self.r_etas) * len(self.r_shift) * len(binary_elements) + len(
            self.a_etas
        ) * len(self.zetas) * len(self.lambdas) * len(list(itertools.combinations_with_replacement(binary_elements, 2)))

    def test_transform(self):
        unary_descriptors = self.describer.transform([self.unary_struct] * 3)
        binary_descriptors = self.describer.transform([self.binary_struct] * 3)
        unary_elements = set(self.unary_struct.species)
        binary_elements = set(self.binary_struct.species)
        assert unary_descriptors.shape[0] == len(self.unary_struct) * 3
        assert unary_descriptors.shape[1] == len(self.r_etas) * len(self.r_shift) * len(unary_elements) + len(
            self.a_etas
        ) * len(self.zetas) * len(self.lambdas) * len(list(itertools.combinations_with_replacement(unary_elements, 2)))
        assert binary_descriptors.shape[0] == len(self.binary_struct) * 3
        assert binary_descriptors.shape[1] == len(self.r_etas) * len(self.r_shift) * len(binary_elements) + len(
            self.a_etas
        ) * len(self.zetas) * len(self.lambdas) * len(list(itertools.combinations_with_replacement(binary_elements, 2)))


class TestSiteSpecieProperty(unittest.TestCase):
    def test_unordered_site(self):
        s = Structure(Lattice.cubic(3), ["Mo", "S"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        udescriber = SiteElementProperty()
        np.testing.assert_array_almost_equal(udescriber.transform_one(s), np.array([[42, 16]]).T)

        udescriber2 = SiteElementProperty(feature_dict={16: [16, 16], 42: [42, 42]})
        np.testing.assert_array_almost_equal(udescriber2.transform_one(s), np.array([[42, 42], [16, 16]]))

        assert udescriber.describer_type == "site"

        udescriber = SiteElementProperty(output_weights=True)
        vec, weight = udescriber.transform_one(s)
        np.testing.assert_array_almost_equal(vec, np.array([[42, 16]]).T)
        np.testing.assert_array_almost_equal(weight, np.array([1, 1]))

        s2 = s.copy()
        s2.replace_species({"Mo": {"S": 0.1, "Mo": 0.9}})
        self.assertRaises(ValueError, udescriber2.transform_one, s2)
        vec, weight = udescriber.transform_one(s2)
        np.testing.assert_array_almost_equal(vec, np.array([[16, 42, 16]]).T)
        np.testing.assert_array_almost_equal(weight, np.array([0.1, 0.9, 1]))


# ---------------------------------------------------------------------------
# pytest-style tests with mocked LAMMPS / QUIP backends.
# ---------------------------------------------------------------------------

PROFILE = {"Na": {"r": 0.3, "w": 0.9}, "Cl": {"r": 0.7, "w": 3.0}}


@pytest.fixture
def nacl() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]])


def _canned_sna(structure: Structure, n_cols: int, n_elements: int, seed: int = 0):
    """Fake output of SpectralNeighborAnalysis.calculate for one structure: (b, db, vb, elements)."""
    rng = np.random.default_rng(seed)
    n_atoms = len(structure)
    b = rng.random((n_atoms, n_cols))
    db = rng.random((n_atoms, 3 * n_cols * n_elements))
    vb = rng.random((n_atoms, 6 * n_cols * n_elements))
    elements = np.array([site.specie.symbol for site in structure])
    return b, db, vb, elements


@pytest.fixture
def fake_lammps_exe(monkeypatch):
    """SpectralNeighborAnalysis requires a LAMMPS executable at construction; fake one for mocked tests."""
    from maml.apps.pes import _lammps

    monkeypatch.setattr(_lammps, "get_default_lmp_exe", lambda: "lmp_serial")
    monkeypatch.setattr(_lammps, "which", lambda cmd: f"/usr/bin/{cmd}")


def _mock_calculate(monkeypatch, describer, canned):
    calls = []

    def calculate(structures):
        calls.append(structures)
        return [canned]

    monkeypatch.setattr(describer.calculator, "calculate", calculate)
    return calls


@pytest.mark.parametrize("quadratic", [False, True])
def test_bispectrum_per_atom(fake_lammps_exe, monkeypatch, nacl, quadratic):
    bc = BispectrumCoefficients(rcutfac=5, twojmax=3, element_profile=PROFILE, quadratic=quadratic)
    n_bs = len(bc.subscripts)
    assert n_bs == 8
    n_cols = n_bs + n_bs * (n_bs + 1) // 2 if quadratic else n_bs
    canned = _canned_sna(nacl, n_cols, len(bc.elements))
    calls = _mock_calculate(monkeypatch, bc, canned)

    df = bc.transform_one(nacl)
    assert calls == [[nacl]]
    assert df.shape == (len(nacl), n_cols)
    assert bc.feature_dim == n_cols
    assert list(df.columns[:2]) == ["0-0-0", "1-0-1"]
    if quadratic:
        assert df.columns[n_bs] == "000-000"
    np.testing.assert_array_equal(df.to_numpy(), canned[0])


@pytest.mark.parametrize("quadratic", [False, True])
@pytest.mark.parametrize("include_stress", [False, True])
def test_bispectrum_pot_fit(fake_lammps_exe, monkeypatch, nacl, quadratic, include_stress):
    bc = BispectrumCoefficients(
        rcutfac=5,
        twojmax=3,
        element_profile=PROFILE,
        quadratic=quadratic,
        pot_fit=True,
        include_stress=include_stress,
    )
    n_bs = len(bc.subscripts)
    n_cols = n_bs + n_bs * (n_bs + 1) // 2 if quadratic else n_bs
    n_el = len(bc.elements)
    b, db, vb, elements = _canned_sna(nacl, n_cols, n_el)
    _mock_calculate(monkeypatch, bc, (b, db, vb, elements))

    df = bc.transform_one(nacl)
    n_atoms = len(nacl)
    n_rows = 1 + 3 * n_atoms + (6 if include_stress else 0)
    assert df.shape == (n_rows, (n_cols + 1) * n_el)
    assert bc.feature_dim == df.shape[1]

    # Energy row: per-element atom fraction and per-element bispectrum sums normalized by the atom count.
    energy_row = df.iloc[0]
    for el in bc.elements:
        mask = elements == el
        assert energy_row[el, "n"] == pytest.approx(mask.sum() / n_atoms)
        np.testing.assert_allclose(energy_row[el].to_numpy()[1:], b[mask].sum(axis=0) / n_atoms)

    # Force rows: per-element derivative blocks with a zero "n" column.
    force_rows = df.iloc[1 : 1 + 3 * n_atoms]
    assert list(force_rows.index[:4]) == ["0_x", "0_y", "0_z", "1_x"]
    for i_el, el in enumerate(bc.elements):
        np.testing.assert_array_equal(force_rows[el, "n"].to_numpy(), 0)
        block = db[:, i_el * 3 * n_cols : (i_el + 1) * 3 * n_cols].reshape(-1, n_cols)
        np.testing.assert_allclose(force_rows[el].to_numpy()[:, 1:], block)

    if include_stress:
        stress_rows = df.iloc[-6:]
        assert list(stress_rows.index) == ["xx", "yy", "zz", "yz", "xz", "xy"]
        vb_sum = vb.sum(axis=0)
        for i_el, el in enumerate(bc.elements):
            block = vb_sum[i_el * 6 * n_cols : (i_el + 1) * 6 * n_cols].reshape(-1, n_cols)
            expected = block / nacl.volume * 160.21766208
            np.testing.assert_allclose(stress_rows[el].to_numpy()[:, 1:], expected)
            np.testing.assert_array_equal(stress_rows[el, "n"].to_numpy(), 0)


def test_bispectrum_pot_fit_missing_element(fake_lammps_exe, monkeypatch):
    """Elements in the profile but absent from the structure get zero energy-row features."""
    li = Structure.from_spacegroup("Im-3m", Lattice.cubic(3.4268), ["Li"], [[0, 0, 0]])
    profile = {"Li": {"r": 0.5, "w": 1.0}, "O": {"r": 0.5, "w": 1.0}}
    bc = BispectrumCoefficients(rcutfac=5, twojmax=2, element_profile=profile, pot_fit=True)
    n_bs = len(bc.subscripts)
    _mock_calculate(monkeypatch, bc, _canned_sna(li, n_bs, 2))
    df = bc.transform_one(li)
    np.testing.assert_array_equal(df.iloc[0]["O"].to_numpy(), 0)
    assert df.iloc[0]["Li", "n"] == 1


class _FakePopen:
    """Stand-in for subprocess.Popen that writes canned quip output to the redirected stdout."""

    output = ""
    returncode_value = 0
    commands: list = []

    def __init__(self, cmd, stdout=None):
        _FakePopen.commands.append(cmd)
        self.stdout_file = stdout
        self.returncode = None

    def communicate(self):
        self.stdout_file.write(self.output)
        self.stdout_file.flush()
        self.returncode = self.returncode_value
        return None, None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def fake_quip(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_site, "which", lambda cmd: f"/usr/bin/{cmd}")
    _FakePopen.commands = []
    _FakePopen.output = ""
    _FakePopen.returncode_value = 0
    monkeypatch.setattr(_site.subprocess, "Popen", _FakePopen)
    return _FakePopen


def test_soap_quip_missing(monkeypatch, nacl):
    monkeypatch.setattr(_site, "which", lambda cmd: None)
    with pytest.raises(RuntimeError, match="quip has not been found"):
        SmoothOverlapAtomicPosition(cutoff=4.8).transform_one(nacl)


def test_soap_parses_output(fake_quip, nacl):
    descriptors = np.arange(len(nacl) * 3, dtype=float).reshape(len(nacl), 3)
    fake_quip.output = "libAtoms::Hello World\n" + "".join(f"DESC {' '.join(map(str, row))}\n" for row in descriptors)
    describer = SmoothOverlapAtomicPosition(cutoff=4.8, l_max=6, n_max=4, atom_sigma=0.4)
    df = describer.transform_one(nacl)
    assert isinstance(df, pd.DataFrame)
    np.testing.assert_array_equal(df.to_numpy(), descriptors)

    (cmd,) = fake_quip.commands
    assert cmd[:2] == ["quip", "atoms_filename=structure.xyz"]
    desc = cmd[2]
    assert desc.startswith("descriptor_str={soap ")
    for token in ["cutoff=4.8", "l_max=6", "n_max=4", "atom_sigma=0.4", "n_Z=2", "Z={11 17}", "species_Z={11 17}"]:
        assert token in desc

    batched = describer.transform([nacl, nacl])
    assert batched.shape == (2 * len(nacl), 3)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("some line\nERROR: bad descriptor\ndetails\n", "return code 1: ERROR: bad descriptor, details"),
        ("first\nlast line\n", "return code 1: last line"),
        ("", "return code 1$"),
    ],
)
def test_soap_quip_failure(fake_quip, nacl, output, expected):
    """Regression: quip failures used to raise AttributeError since stdout is redirected to a file."""
    fake_quip.output = output
    fake_quip.returncode_value = 1
    with pytest.raises(RuntimeError, match=expected):
        SmoothOverlapAtomicPosition(cutoff=4.8).transform_one(nacl)


def test_bp_symmetry_functions_values():
    """Check G2/G4 against a direct evaluation for an isolated Li3 right-angle cluster."""
    s = Structure(Lattice.cubic(20), ["Li", "Li", "Li"], [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]])
    cutoff = 4.0
    describer = BPSymmetryFunctions(
        cutoff=cutoff, r_etas=[0.5], r_shift=[0.0], a_etas=[0.1], zetas=[1.0], lambdas=[1.0]
    )
    df = describer.transform_one(s)
    assert df.shape == (3, 2)

    def fc(r):
        return 0.5 * (np.cos(np.pi * r / cutoff) + 1) * (r <= cutoff)

    # Site 0 has neighbors at 2 A along x and y (right angle, separation 2*sqrt(2) A).
    r1 = r2 = 2.0
    r3 = 2 * np.sqrt(2)
    g2 = 2 * np.exp(-0.5 * r1**2) * fc(r1)
    g4 = np.exp(-0.1 * (r1**2 + r2**2 + r3**2)) * fc(r1) * fc(r2) * fc(r3)
    np.testing.assert_allclose(df.iloc[0].to_numpy(), [g2, g4])


def test_bp_fc_cutoff():
    describer = BPSymmetryFunctions(cutoff=3.0, r_etas=[1], r_shift=[0], a_etas=[1], zetas=[1], lambdas=[1])
    np.testing.assert_allclose(describer._fc(np.array([0.0, 1.5, 3.0, 3.5])), [1.0, 0.5, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    ("obj", "expected"),
    [("Fe2O3", [[26], [26], [8], [8], [8]]), (Composition("LiH"), [[3], [1]])],
)
def test_site_element_property_composition(obj, expected):
    np.testing.assert_array_equal(SiteElementProperty().transform_one(obj), expected)


def test_site_element_property_feature_dim():
    assert SiteElementProperty().feature_dim is None
    assert SiteElementProperty(feature_dict={1: [1, 2, 3], 2: [4, 5, 6]}).feature_dim == 3


def test_site_element_property_molecule_weights():
    mol = Molecule(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    feats, weights = SiteElementProperty(output_weights=True).transform_one(mol)
    np.testing.assert_array_equal(feats.ravel(), [8, 1, 1])
    np.testing.assert_array_equal(weights, [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
