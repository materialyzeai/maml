from __future__ import annotations

import unittest

import numpy as np
import pytest
from monty.tempfile import ScratchDir
from pymatgen.core import Element, Lattice, Species, Structure
from pymatgen.io.lammps.data import LammpsData

from maml.utils._lammps import (
    _get_atomic_mass,
    _get_charge,
    check_structures_forces_stresses,
    stress_format_change,
    stress_list_to_matrix,
    write_data_from_structure,
)


class TestLAMMPS(unittest.TestCase):
    def test_stress_format_convert(self):
        vasp = ["xx", "yy", "zz", "xy", "yz", "xz"]
        lammps = stress_format_change(vasp, "VASP", "LAMMPS")
        self.assertListEqual(lammps.tolist(), ["xx", "yy", "zz", "xy", "xz", "yz"])
        snap = stress_format_change(vasp, "VASP", "SNAP")
        self.assertListEqual(snap.tolist(), ["xx", "yy", "zz", "yz", "xz", "xy"])

    def test_stress_list_matrix(self):
        a = [1, 2, 3, 4, 5, 6]
        matrix = stress_list_to_matrix(a, "VASP")
        matrix = matrix.tolist()
        self.assertListEqual(matrix[0], [1, 4, 6])
        self.assertListEqual(matrix[1], [4, 2, 5])
        self.assertListEqual(matrix[2], [6, 5, 3])

    def test_check_structures(self):
        s = Structure(
            Lattice(np.array([[3.16, 0.1, 0.2], [0.1, 3.17, 0.3], [0.1, 0.2, 3]])),
            ["Mo", "Mo"],
            [[0, 0, 0], [0.13, 0.4, 0.2]],
        )

        forces = np.array([[0.04844841, 0.08648062, 0.07070806], [-0.04844841, -0.08648062, -0.07070806]])

        stress = np.array([-0.22279327, -1.2809575, -0.44279698, -0.23345818, -0.37798718, -0.17676364])

        checked_force = np.array([[0.05552151, 0.09063424, 0.05940176], [-0.05552151, -0.09063424, -0.05940176]])

        checked_stress = np.array([-0.26319715, -1.3219795, -0.3613719, -0.30627516, -0.27276486, -0.17306383])

        new_structures, new_forces, new_stresses = check_structures_forces_stresses([s], [forces], [stress])

        # print(np.linalg.norm(checked_stress - new_stresses[0]))
        print(new_stresses[0], checked_stress)
        assert np.linalg.norm(checked_force - new_forces[0]) < 0.0001
        assert np.linalg.norm(checked_stress - new_stresses[0]) < 0.0001

        new_structures = check_structures_forces_stresses(structures=[s], return_none=False)

        assert len(new_structures) == 1
        assert isinstance(new_structures[0], Structure)

        new_structures, new_forces, new_stresses = check_structures_forces_stresses(structures=[s, s], return_none=True)

        assert len(new_forces) == 2
        assert new_forces[0] is None
        assert len(new_stresses) == 2
        assert new_stresses[0] is None

    def test_write(self):
        s = Structure(
            Lattice(np.array([[3.16, 0.1, 0.2], [0.1, 3.17, 0.3], [0.1, 0.2, 3]])),
            ["Mo", "Mo"],
            [[0, 0, 0], [0.13, 0.4, 0.2]],
        )
        with ScratchDir("."):
            write_data_from_structure(s, "test.data")
            lmp = LammpsData.from_file("test.data", atom_style="charge")
            lmp2 = LammpsData.from_structure(s)
            lmp2.write_file("test2.data")
            assert str(lmp) == str(lmp2)


def _rotated_orthogonal_structure():
    # orthogonal cell whose a vector points along y and b along -x (right-handed)
    return Structure(Lattice([[0, 3.0, 0], [-4.0, 0, 0], [0, 0, 5.0]]), ["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_check_structures_rotates_orthogonal_but_misaligned_lattice():
    """Regression: orthogonal cells were skipped even when not aligned with x, y, z."""
    s = _rotated_orthogonal_structure()
    force = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])  # along a
    stress = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])  # sigma_yy = 2 lies along a
    new_s, new_f, new_stress = check_structures_forces_stresses([s], [force], [stress])
    np.testing.assert_allclose(new_s[0].lattice.matrix, np.diag([3.0, 4.0, 5.0]), atol=1e-10)
    np.testing.assert_allclose(new_f[0], [[1.0, 0, 0], [-1.0, 0, 0]], atol=1e-10)
    np.testing.assert_allclose(new_stress[0], np.diag([2.0, 1.0, 3.0]), atol=1e-10)
    np.testing.assert_allclose(new_s[0].frac_coords, s.frac_coords, atol=1e-10)


def test_check_structures_aligned_lattice_untouched():
    s = Structure(Lattice.cubic(3.0), ["Mo"], [[0, 0, 0]])
    forces = [np.array([[0.1, 0.2, 0.3]])]
    stresses = [np.arange(6.0)]
    new_s, new_f, new_stress = check_structures_forces_stresses([s], forces, stresses)
    assert new_s[0] is s
    assert new_f[0] is forces[0]
    assert new_stress[0] is stresses[0]


@pytest.mark.parametrize(
    ("with_forces", "with_stresses", "n_out"),
    [(True, False, 2), (False, True, 2), (True, True, 3)],
)
def test_check_structures_return_without_none(with_forces, with_stresses, n_out):
    s = _rotated_orthogonal_structure()
    forces = [np.zeros((2, 3))] if with_forces else None
    stresses = [np.ones(6)] if with_stresses else None
    out = check_structures_forces_stresses([s], forces, stresses, return_none=False)
    assert len(out) == n_out
    assert isinstance(out[0][0], Structure)
    if with_stresses:
        # isotropic stress is invariant under rotation (in voigt format, off-diagonal terms are 1 too)
        assert np.shape(out[-1][0]) == (6,)


def test_write_data_unknown_element(tmp_path):
    s = Structure(Lattice.cubic(3.0), ["Mo"], [[0, 0, 0]])
    with pytest.raises(ValueError, match="not in ff_elements"):
        write_data_from_structure(s, str(tmp_path / "x.data"), ff_elements=["W"])


def test_write_data_with_oxidation_states(tmp_path):
    s = Structure(Lattice.cubic(4.2), ["Mg2+", "O2-"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    fname = tmp_path / "mgo.data"
    write_data_from_structure(s, str(fname), ff_elements=["Mg2+", "O2-"])
    lmp = LammpsData.from_file(str(fname), atom_style="charge")
    np.testing.assert_allclose(sorted(lmp.atoms["q"]), [-2, 2])
    np.testing.assert_allclose(lmp.masses["mass"].to_numpy(), [24.305, 15.9994], atol=1e-3)


@pytest.mark.parametrize(
    ("specie", "charge"),
    [("Fe2+", 2.0), ("O2-", -2.0), ("Fe", 0.0), (Species("O", -2), -2.0), (Element("Fe"), 0.0)],
)
def test_get_charge(specie, charge):
    """Regression: string species used the removed Species.from_string and always returned 0."""
    assert _get_charge(specie) == charge


@pytest.mark.parametrize(("specie", "mass"), [("Fe", 55.845), ("Fe2+", 55.845)])
def test_get_atomic_mass(specie, mass):
    assert _get_atomic_mass(specie) == pytest.approx(mass)


if __name__ == "__main__":
    unittest.main()
