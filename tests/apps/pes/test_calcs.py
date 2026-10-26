# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from shutil import which
from pymatgen.core import Lattice, Structure
from sklearn.linear_model import LinearRegression

from maml.apps.pes import (
    DefectFormation,
    ElasticConstant,
    EnergyForceStress,
    LatticeConstant,
    NudgedElasticBand,
    SNAPotential,
    SpectralNeighborAnalysis,
    get_default_lmp_exe,
)
from maml.base import SKLModel
from maml.describers import BispectrumCoefficients

CWD = os.getcwd()
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "coeff.json")) as f:
    coeff, intercept = json.load(f)
    coeff = np.array(coeff)


class SpectralNeighborAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()

    def setUp(self):
        self.structure1 = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.46873), ["Si"], [[0, 0, 0]])
        self.structure2 = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]]
        )
        self.structure3 = Structure.from_spacegroup(
            "Pm-3m", Lattice.cubic(3.88947), ["Ca", "Ti", "O"], [[0.5, 0.5, 0.5], [0, 0, 0], [0, 0, 0.5]]
        )

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.this_dir)
        shutil.rmtree(cls.test_dir)

    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found.")
    def test_calculate(self):
        profile1 = dict(Si=dict(r=0.5, w=1.9), C=dict(r=0.5, w=2.55))
        tjm1 = np.random.randint(1, 11)
        calculator1 = SpectralNeighborAnalysis(rcutfac=5, twojmax=tjm1, element_profile=profile1, quadratic=False)
        sna1, snad1, snav1, elem1 = calculator1.calculate([self.structure1])[0]
        n1 = calculator1.n_bs
        self.assertAlmostEqual(sna1[0][0], 585.920)
        assert sna1.shape == (len(self.structure1), n1)
        assert snad1.shape == (len(self.structure1), n1 * 3 * len(profile1))
        assert snav1.shape == (len(self.structure1), n1 * 6 * len(profile1))
        assert len(np.unique(elem1)) == 1

        calculator4 = SpectralNeighborAnalysis(rcutfac=5, twojmax=tjm1, element_profile=profile1, quadratic=True)
        sna4, snad4, snav4, elem4 = calculator4.calculate([self.structure1])[0]
        n4 = calculator4.n_bs
        n4 += int((1 + n4) * n4 / 2)
        self.assertAlmostEqual(sna4[0][0], 585.920)
        assert sna4.shape == (len(self.structure1), n4)
        assert snad4.shape == (len(self.structure1), n4 * 3 * len(profile1))
        assert snav4.shape == (len(self.structure1), n4 * 6 * len(profile1))
        assert len(np.unique(elem4)) == 1

        profile2 = dict(Na=dict(r=0.3, w=0.9), Cl=dict(r=0.7, w=3.0))
        tjm2 = np.random.randint(1, 11)
        calculator2 = SpectralNeighborAnalysis(rcutfac=5, twojmax=tjm2, element_profile=profile2, quadratic=False)
        sna2, snad2, snav2, elem2 = calculator2.calculate([self.structure2])[0]
        n2 = calculator2.n_bs
        self.assertAlmostEqual(sna2[0][0], 525.858)
        assert sna2.shape == (len(self.structure2), n2)
        assert snad2.shape == (len(self.structure2), n2 * 3 * len(profile2))
        assert snav2.shape == (len(self.structure2), n2 * 6 * len(profile2))
        assert len(np.unique(elem2)) == len(profile2)

        calculator5 = SpectralNeighborAnalysis(rcutfac=5, twojmax=tjm2, element_profile=profile2, quadratic=True)
        sna5, snad5, snav5, elem5 = calculator5.calculate([self.structure2])[0]
        n5 = calculator5.n_bs
        n5 += int((1 + n5) * n5 / 2)
        self.assertAlmostEqual(sna5[0][0], 525.858)
        assert sna5.shape == (len(self.structure2), n5)
        assert snad5.shape == (len(self.structure2), n5 * 3 * len(profile2))
        assert snav5.shape == (len(self.structure2), n5 * 6 * len(profile2))
        assert len(np.unique(elem5)) == len(profile2)

        profile3 = dict(Ca=dict(r=0.4, w=1.0), Ti=dict(r=0.3, w=1.5), O=dict(r=0.75, w=3.5))
        tjm3 = np.random.randint(1, 11)
        calculator3 = SpectralNeighborAnalysis(rcutfac=5, twojmax=tjm3, element_profile=profile3, quadratic=False)
        sna3, snad3, snav3, elem3 = calculator3.calculate([self.structure3])[0]
        n3 = calculator3.n_bs
        self.assertAlmostEqual(sna3[0][0], 25506.3)
        assert sna3.shape == (len(self.structure3), n3)
        assert snad3.shape == (len(self.structure3), n3 * 3 * len(profile3))
        assert snav3.shape == (len(self.structure3), n3 * 6 * len(profile3))
        assert len(np.unique(elem3)) == len(profile3)

        calculator6 = SpectralNeighborAnalysis(rcutfac=5, twojmax=tjm3, element_profile=profile3, quadratic=True)
        sna6, snad6, snav6, elem6 = calculator6.calculate([self.structure3])[0]
        n6 = calculator6.n_bs
        n6 += int((1 + n6) * n6 / 2)
        self.assertAlmostEqual(sna6[0][0], 25506.3)
        assert sna6.shape == (len(self.structure3), n6)
        assert snad6.shape == (len(self.structure3), n6 * 3 * len(profile3))
        assert snav6.shape == (len(self.structure3), n6 * 6 * len(profile3))
        assert len(np.unique(elem6)) == len(profile3)


class EnergyForceStressTest(unittest.TestCase):
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
        element_profile = {"Ni": {"r": 0.5, "w": 1}}
        describer1 = BispectrumCoefficients(
            rcutfac=4.1, twojmax=8, element_profile=element_profile, quadratic=False, pot_fit=True
        )
        model1 = SKLModel(describer=describer1, model=LinearRegression())
        model1.model.coef_ = coeff
        model1.model.intercept_ = intercept
        snap1 = SNAPotential(model=model1)
        self.ff_settings1 = snap1

        describer2 = BispectrumCoefficients(
            rcutfac=4.1, twojmax=8, element_profile=element_profile, quadratic=True, pot_fit=True
        )
        model2 = SKLModel(describer=describer2, model=LinearRegression())
        model2.model.coef_ = coeff
        model2.model.intercept_ = intercept
        snap2 = SNAPotential(model=model2)
        self.ff_settings2 = snap2

        self.struct = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.506), ["Ni"], [[0, 0, 0]])

    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found.")
    def test_calculate(self):
        calculator1 = EnergyForceStress(ff_settings=self.ff_settings1)
        energy1, forces1, stresses1 = calculator1.calculate([self.struct])[0]
        assert abs(energy1 - -23.1242962) < 0.01
        np.testing.assert_array_almost_equal(forces1, np.zeros((len(self.struct), 3)))
        assert len(stresses1) == 6


class ElasticConstantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()
        os.chdir(cls.test_dir)
        print(os.getcwd())

    @classmethod
    def tearDownClass(cls):
        os.chdir(CWD)
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        element_profile = {"Ni": {"r": 0.5, "w": 1}}
        describer = BispectrumCoefficients(rcutfac=4.1, twojmax=8, element_profile=element_profile, pot_fit=True)
        model = SKLModel(describer=describer, model=LinearRegression())
        model.model.coef_ = coeff
        model.model.intercept_ = intercept
        snap = SNAPotential(model=model)
        self.ff_settings = snap

    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found.")
    def test_calculate(self):
        calculator = ElasticConstant(ff_settings=self.ff_settings)
        struct = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.506), ["Ni"], [[0, 0, 0]])
        C11, C12, C44, bulkmodulus = calculator.calculate([struct])[0]
        assert abs(C11 - 276) / 276 < 0.1
        assert abs(C12 - 159) / 159 < 0.1
        assert abs(C44 - 132) / 132 < 0.1


class LatticeConstantTest(unittest.TestCase):
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
        element_profile = {"Ni": {"r": 0.5, "w": 1}}
        describer = BispectrumCoefficients(rcutfac=4.1, twojmax=8, element_profile=element_profile, pot_fit=True)
        model = SKLModel(describer=describer, model=LinearRegression())
        model.model.coef_ = coeff
        model.model.intercept_ = intercept
        snap = SNAPotential(model=model)
        self.struct = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.506), ["Ni"], [[0, 0, 0]])
        self.ff_settings = snap

    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS cmd found.")
    def test_calculate(self):
        calculator = LatticeConstant(ff_settings=self.ff_settings)
        a, b, c = self.struct.lattice.abc
        calc_a, calc_b, calc_c = calculator.calculate([self.struct])[0]
        np.testing.assert_almost_equal(calc_a, a, decimal=2)
        np.testing.assert_almost_equal(calc_b, b, decimal=2)
        np.testing.assert_almost_equal(calc_c, c, decimal=2)


class NudgedElasticBandTest(unittest.TestCase):
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
        element_profile = {"Ni": {"r": 0.5, "w": 1}}
        describer = BispectrumCoefficients(rcutfac=4.1, twojmax=8, element_profile=element_profile, pot_fit=True)
        model = SKLModel(describer=describer, model=LinearRegression())
        model.model.coef_ = coeff
        model.model.intercept_ = intercept
        snap = SNAPotential(model=model)
        self.struct = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.506), ["Ni"], [[0, 0, 0]])
        self.ff_settings = snap

    @unittest.skipIf(True, "No LAMMPS mpi cmd found.")
    def test_calculate(self):
        calculator = NudgedElasticBand(
            ff_settings=self.ff_settings, specie="Ni", lattice="fcc", alat=3.506, lmp_exe="lmp_mpi", num_replicas=3
        )
        print("NudgedElasticBand using %s" % calculator.LMP_EXE)
        migration_barrier = calculator.calculate()
        np.testing.assert_almost_equal(migration_barrier, 1.013, decimal=2)
        invalid_calculator = NudgedElasticBand(ff_settings=self.ff_settings, specie="Ni", lattice="sc", alat=3.506)
        self.assertRaises(ValueError, invalid_calculator.calculate)


class DefectFormationTest(unittest.TestCase):
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
        element_profile = {"Ni": {"r": 0.5, "w": 1}}
        describer = BispectrumCoefficients(rcutfac=4.1, twojmax=8, element_profile=element_profile, pot_fit=True)
        model = SKLModel(describer=describer, model=LinearRegression())
        model.model.coef_ = coeff
        model.model.intercept_ = intercept
        snap = SNAPotential(model=model)
        self.struct = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.506), ["Ni"], [[0, 0, 0]])
        self.ff_settings = snap

    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS serial cmd found.")
    def test_calculate(self):
        calculator = DefectFormation(ff_settings=self.ff_settings, specie="Ni", lattice="fcc", alat=3.506)
        defect_formation_energy = calculator.calculate()
        np.testing.assert_almost_equal(defect_formation_energy, 1.502, decimal=2)
        invalid_calculator = DefectFormation(ff_settings=self.ff_settings, specie="Ni", lattice="fccc", alat=3.506)
        self.assertRaises(ValueError, invalid_calculator.calculate)


class LMPTest(unittest.TestCase):
    @unittest.skipIf(not which("lmp_serial"), "No LAMMPS serial cmd found.")
    def test_get_lmp_exe(self):
        init_lmp = get_default_lmp_exe()
        lc = LatticeConstant(["dummy setting"])
        lc.set_lmp_exe(init_lmp)
        assert init_lmp == lc.LMP_EXE

        new_lmp = "It can be any directory"
        lc.set_lmp_exe(new_lmp)
        assert new_lmp == lc.LMP_EXE
        lc.set_lmp_exe(init_lmp)


if __name__ == "__main__":
    unittest.main()
