from __future__ import annotations

import unittest

import numpy as np
import pytest
from pymatgen.util.testing import PymatgenTest

from maml.describers import RadialDistributionFunction


class TestRDF(PymatgenTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rdf_instance = RadialDistributionFunction(r_max=4)
        cls.structure = cls.get_structure("SrTiO3")

    def test_get_site_rdf(self):
        r, site_rdfs = self.rdf_instance.get_site_rdf(self.structure)
        r_ref = np.linspace(self.rdf_instance.r_min, self.rdf_instance.r_max, self.rdf_instance.n_grid)
        assert np.allclose(r, r_ref)
        assert "Sr2+:O2-" in site_rdfs[0]
        assert "Sr2+:Sr2+" in site_rdfs[0]
        assert "Sr2+:Ti4+" in site_rdfs[0]

        a = self.structure.lattice.a
        sr_o = np.sqrt(2) / 2.0 * a
        sr_sr = a
        sr_ti = np.sqrt(3) / 2.0 * a
        sr_o_cal = r[np.argmax(site_rdfs[0]["Sr2+:O2-"])]
        sr_sr_cal = r[np.argmax(site_rdfs[0]["Sr2+:Sr2+"])]
        sr_ti_cal = r[np.argmax(site_rdfs[0]["Sr2+:Ti4+"])]
        assert sr_o_cal == pytest.approx(sr_o, 2)
        assert sr_sr_cal == pytest.approx(sr_sr, 1)
        assert sr_ti_cal == pytest.approx(sr_ti, 1)

    def test_get_species_rdf(self):
        r, rdf = self.rdf_instance.get_species_rdf(self.structure)
        assert self.structure.lattice.a * np.sqrt(2) / 2.0 == pytest.approx(r[np.argmax(rdf)], 2)

    def test_get_site_cn(self):
        r, cns = self.rdf_instance.get_site_coordination(self.structure)
        assert cns[0]["Sr2+:O2-"][-1].item() == pytest.approx(12)
        assert cns[0]["Sr2+:Ti4+"][-1].item() == pytest.approx(8)
        assert cns[0]["Sr2+:Sr2+"][-1].item() == pytest.approx(6)

    def test_get_specie_cn(self):
        r, cn = self.rdf_instance.get_species_coordination(self.structure)
        assert cn[-1] == 106


def test_isolated_site_has_empty_rdf():
    """A site with no neighbors within r_max yields an empty rdf/cn dict."""
    from pymatgen.core import Lattice, Structure

    s = Structure(Lattice.cubic(20), ["Li", "Li", "O"], [[0, 0, 0], [0.1, 0, 0], [0.5, 0.5, 0.5]])
    rdf = RadialDistributionFunction(r_max=4)
    _, rdfs = rdf.get_site_rdf(s)
    assert set(rdfs[0]) == {"Li:Li"}
    assert rdfs[2] == {}
    _, cns = rdf.get_site_coordination(s)
    assert cns[2] == {}
    assert cns[0]["Li:Li"][-1] == pytest.approx(1)


def test_species_rdf_filters_and_no_smoothing():
    structure = PymatgenTest.get_structure("SrTiO3")
    rdf = RadialDistributionFunction(r_max=4, sigma=0)
    assert rdf.sigma == 0
    r, sr_o = rdf.get_species_rdf(structure, ref_species=["Sr2+"], species=["O2-"])
    # Without smoothing, all Sr-O intensity sits in the grid bin at the Sr-O distance.
    assert r[np.argmax(sr_o)] == pytest.approx(structure.lattice.a * np.sqrt(2) / 2, abs=rdf.dr)
    assert np.count_nonzero(sr_o) <= 2
    # Species / reference species with no matching pairs give an all-zero rdf.
    _, none = rdf.get_species_rdf(structure, ref_species=["Sr2+"], species=["Xe"])
    np.testing.assert_array_equal(none, 0)
    _, none = rdf.get_species_rdf(structure, ref_species=["Xe"])
    np.testing.assert_array_equal(none, 0)


def test_gaussian_smoothing_preserves_peak_and_area():
    structure = PymatgenTest.get_structure("SrTiO3")
    sharp = RadialDistributionFunction(r_max=6, n_grid=301)
    smooth = RadialDistributionFunction(r_max=6, n_grid=301, sigma=0.1)
    assert smooth.sigma == 5  # sigma is expressed in grid points: ceil(0.1 / 0.02)

    r, rdf_sharp = sharp.get_species_rdf(structure, ref_species=["Sr2+"], species=["O2-"])
    _, rdf_smooth = smooth.get_species_rdf(structure, ref_species=["Sr2+"], species=["O2-"])
    assert rdf_smooth.max() < rdf_sharp.max()
    assert r[np.argmax(rdf_smooth)] == pytest.approx(r[np.argmax(rdf_sharp)], abs=2 * sharp.dr)
    assert rdf_smooth.sum() == pytest.approx(rdf_sharp.sum(), rel=1e-6)

    _, site_sharp = sharp.get_site_rdf(structure)
    _, site_smooth = smooth.get_site_rdf(structure)
    assert site_sharp[0].keys() == site_smooth[0].keys()
    for key in site_sharp[0]:
        assert site_smooth[0][key].sum() == pytest.approx(site_sharp[0][key].sum(), rel=1e-6)


if __name__ == "__main__":
    unittest.main()
