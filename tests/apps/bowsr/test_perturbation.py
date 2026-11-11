from __future__ import annotations

import os
import unittest

import numpy as np
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from maml.apps.bowsr.perturbation import (
    LatticePerturbation,
    WyckoffPerturbation,
    crystal_system,
    get_standardized_structure,
)

test_lfpo = Structure.from_file(os.path.join(os.path.dirname(__file__), "test_structures", "test_lfpo.cif"))
test_lco = Structure.from_file(os.path.join(os.path.dirname(__file__), "test_structures", "test_lco.cif"))


class WyckoffPerturbationTest(unittest.TestCase):
    def setUp(self):
        self.test_lfpo = test_lfpo
        lfpo_sa = SpacegroupAnalyzer(test_lfpo)
        lfpo_sd = lfpo_sa.get_symmetry_dataset()
        lfpo_symm_ops = [
            SymmOp.from_rotation_and_translation(rotation, translation, tol=0.1)
            for rotation, translation in zip(lfpo_sd["rotations"], lfpo_sd["translations"], strict=True)
        ]
        self.test_refined_lco = get_standardized_structure(test_lco)
        lco_sa = SpacegroupAnalyzer(self.test_refined_lco)
        lco_sd = lco_sa.get_symmetry_dataset()
        lco_symm_ops = [
            SymmOp.from_rotation_and_translation(rotation, translation, tol=0.1)
            for rotation, translation in zip(lco_sd["rotations"], lco_sd["translations"], strict=True)
        ]
        self.wp1 = WyckoffPerturbation(62, wyckoff_symbol="a", symmetry_ops=lfpo_symm_ops)
        self.wp2 = WyckoffPerturbation(62, wyckoff_symbol="c", symmetry_ops=lfpo_symm_ops)
        self.wp3 = WyckoffPerturbation(62, wyckoff_symbol="d", symmetry_ops=lfpo_symm_ops)
        self.wp4 = WyckoffPerturbation(166, wyckoff_symbol="a", symmetry_ops=lco_symm_ops)
        self.wp5 = WyckoffPerturbation(166, wyckoff_symbol="b", symmetry_ops=lco_symm_ops)
        self.wp6 = WyckoffPerturbation(166, wyckoff_symbol="c", symmetry_ops=lco_symm_ops)

    def test_attributes(self):
        assert self.wp1.int_symbol == 62
        assert self.wp1.wyckoff_symbol == "a"
        assert self.wp1.dim == 0
        assert self.wp1.multiplicity == 4
        assert len(self.wp1.symmetry_ops) == 8
        assert self.wp2.wyckoff_symbol == "c"
        assert self.wp2.dim == 2
        assert self.wp2.multiplicity == 4
        assert self.wp3.wyckoff_symbol == "d"
        assert self.wp3.dim == 3
        assert self.wp3.multiplicity == 8

        assert self.wp4.int_symbol == 166
        assert self.wp4.wyckoff_symbol == "a"
        assert self.wp4.dim == 0
        assert self.wp4.multiplicity == 3
        assert len(self.wp4.symmetry_ops) == 36
        assert self.wp5.wyckoff_symbol == "b"
        assert self.wp5.dim == 0
        assert self.wp5.multiplicity == 3
        assert self.wp6.wyckoff_symbol == "c"
        assert self.wp6.dim == 1
        assert self.wp6.multiplicity == 6

    def test_get_orbit(self):
        orbits1 = self.wp1.get_orbit(self.test_lfpo[0].frac_coords)
        assert len(orbits1) == self.wp1.multiplicity
        orbits2 = self.wp2.get_orbit(self.test_lfpo[4].frac_coords)
        assert len(orbits2) == self.wp2.multiplicity
        orbits3 = self.wp3.get_orbit(self.test_lfpo[20].frac_coords)
        assert len(orbits3) == self.wp3.multiplicity
        orbits4 = self.wp3.get_orbit(self.test_lfpo[4].frac_coords)
        assert len(orbits4) != self.wp3.multiplicity

    def test_sanity_check(self):
        self.wp1.sanity_check(self.test_lfpo[8])
        assert self.wp1._fit_site
        self.wp1.sanity_check(self.test_lfpo[0])
        assert self.wp1._fit_site

        assert self.wp2._site is None
        assert not self.wp2._fit_site
        self.wp2.sanity_check(self.test_lfpo[8])
        assert self.wp2._site is not None
        assert self.wp2._fit_site
        assert self.wp3._site is None
        assert not self.wp3._fit_site
        self.wp3.sanity_check(self.test_lfpo[20])
        assert self.wp3._site is not None
        assert self.wp3._fit_site

        assert self.wp4._site is None
        assert not self.wp4._fit_site
        self.wp4.sanity_check(self.test_refined_lco[0])
        assert self.wp4._fit_site
        assert self.wp4._site is not None

        assert self.wp5._site is None
        assert not self.wp5._fit_site
        self.wp5.sanity_check(self.test_refined_lco[3])
        assert self.wp5._site is not None
        assert self.wp5._fit_site
        assert self.wp6._site is None
        assert not self.wp6._fit_site
        self.wp6.sanity_check(self.test_refined_lco[6])
        assert self.wp6._site is not None
        assert self.wp6._fit_site

    def test_standardize(self):
        self.wp1.sanity_check(self.test_lfpo[0])
        self.wp2.sanity_check(self.test_lfpo[8])
        self.wp3.sanity_check(self.test_lfpo[20])
        self.wp4.sanity_check(self.test_refined_lco[0])
        self.wp5.sanity_check(self.test_refined_lco[3])
        self.wp6.sanity_check(self.test_refined_lco[6])
        standardized_coords = self.wp1.standardize(self.test_lfpo[0].frac_coords)
        assert all(abs(standardized_coords - self.test_lfpo[0].frac_coords) < 0.01)
        standardized_coords = self.wp1.standardize(self.test_lfpo[2].frac_coords)
        assert all(abs(standardized_coords - self.test_lfpo[2].frac_coords) < 0.01)
        standardized_coords = self.wp2.standardize(self.test_lfpo[8].frac_coords)
        assert all(abs(standardized_coords - self.test_lfpo[10].frac_coords) < 0.01)
        for site in self.test_lfpo[20:]:
            np.testing.assert_array_almost_equal(self.wp3.standardize(site.frac_coords), site.frac_coords)


class LatticePerturbationTest(unittest.TestCase):
    def setUp(self):
        self.test_lfpo = test_lfpo
        self.test_lco = test_lco
        self.test_refined_lco = get_standardized_structure(test_lco)
        self.lp1 = LatticePerturbation(spg_int_symbol=62)
        self.lp2 = LatticePerturbation(spg_int_symbol=166)

    def test_crystal_system(self):
        assert self.lp1.crys_system == "orthorhombic"
        assert self.lp2.crys_system == "rhombohedral"
        assert crystal_system(72) == "orthorhombic"
        assert crystal_system(147) == "hexagonal"
        assert crystal_system(148) == "rhombohedral"
        assert crystal_system(229) == "cubic"
        assert crystal_system(12) == "monoclinic"
        assert crystal_system(1) == "triclinic"

    def test_sanity_check(self):
        assert self.lp1._lattice is None
        self.lp1.sanity_check(self.test_lfpo.lattice)
        assert self.lp1._lattice is not None
        assert self.lp1._lattice == self.test_lfpo.lattice

        assert self.lp2._lattice is None
        self.lp2.sanity_check(self.test_refined_lco.lattice)
        assert self.lp2._lattice is not None
        assert self.lp2._lattice == self.test_refined_lco.lattice


if __name__ == "__main__":
    unittest.main()
