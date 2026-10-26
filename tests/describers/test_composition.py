from __future__ import annotations

import os
import unittest

import numpy as np
import pytest
from pymatgen.core import Composition, Molecule
from pymatgen.util.testing import PymatgenTest

from maml.describers import ElementProperty, ElementStats
from maml.describers._composition import _is_element_or_specie

CWD = os.path.abspath(os.path.dirname(__file__))


@unittest.skipIf(ElementProperty is None, "matminer package is needed")
class ElementPropertyTest(unittest.TestCase):
    s = PymatgenTest.get_structure("Li2O")

    def test_element_property(self):
        ep = ElementProperty.from_preset("magpie")
        ep.verbose = True
        res = ep.transform([self.s.composition])
        assert res.shape == (1, 132)


class ElementStatsTest(unittest.TestCase):
    def test_element_stats(self):
        test = ElementStats({"H": [4, 2, 3], "O": [2, 3, 4]}, stats=["min", "max", "moment:1:10"])
        assert test.transform(["H2O"]).shape == (1, 36)

        res = test.transform(["H2O", "H2O"])
        assert res.shape == (2, 36)

        res2 = test.transform([Composition("H2O"), Composition("H2O")])
        np.testing.assert_allclose(res.values, res2.values)

        dummy_h2o = Molecule(["H", "H", "O"], [[-0.5, 0, 0], [0.5, 0, 0], [0, 0, 0]])
        res3 = test.transform([dummy_h2o, dummy_h2o])
        np.testing.assert_allclose(res.values, res3.values)

        stats = ["min", "max", *["moment:%d:None" % i for i in range(1, 11)]]
        p0_names = ["p0_%s" % i for i in stats]
        p1_names = ["p1_%s" % i for i in stats]
        p2_names = ["p2_%s" % i for i in stats]
        all_names = []
        for i, j, k in zip(p0_names, p1_names, p2_names):
            all_names.extend([i, j, k])

        self.assertListEqual(list(res.columns), all_names)

    def test_from_data(self):
        es = ElementStats.from_data("megnet_1", stats=["moment:None:5"])
        d = es.transform_one("Fe2O3")
        assert d.shape == (1, 80)

        es = ElementStats.from_data(["megnet_1"], stats=["moment:None:5"])
        d = es.transform_one("Fe2O3")
        assert d.shape == (1, 80)

        es2 = ElementStats.from_data(["megnet_1", "megnet_3"], stats=["moment:None:5"])
        d = es2.transform_one("Fe2O3")
        assert d.shape == (1, 160)

    def test_error(self):
        with self.assertRaises(ValueError):
            _ = ElementStats(element_properties={"H": [1, 2], "O": [1, 2, 3]}, stats=["mean"])

        with self.assertRaises(ValueError):
            _ = ElementStats(element_properties={"H": [1, 2], "O": [1, 2]}, stats=["mean"], property_names=["p1"])

        with self.assertRaises(ValueError):
            _ = ElementStats(
                element_properties={"H": [1, 2], "O": [1, 2]}, stats=["super_std"], property_names=["p1", "p2"]
            )

        with self.assertRaises(ValueError):
            _ = ElementStats(
                element_properties={"H": [1, 2], "O": [1, 2]}, stats=["super_std:-1"], property_names=["p1", "p2"]
            )

        with self.assertRaises(ValueError):
            ElementStats.from_file(os.path.join(CWD, "test_data/wrong_dummy_property.json"), stats=["max"])

        with self.assertRaises(ValueError):
            ElementStats.from_data("megnet_22", stats=["max"])

    def test_keys(self):
        es = ElementStats.from_file(os.path.join(CWD, "test_data/dummy_property.json"), stats=["max"])
        self.assertListEqual(es.stats, ["mean"])

    def test_pca(self):
        es2 = ElementStats.from_data(["megnet_1", "megnet_3"], stats=["moment:None:5"], num_dim=4)
        d = es2.transform_one("Fe2O3")
        assert d.shape == (1, 20)

    def test_kpca(self):
        es2 = ElementStats.from_data(
            ["megnet_1", "megnet_3"], stats=["moment:None:5"], num_dim=2, reduction_algo="kpca"
        )
        d = es2.transform_one("Fe2O3")
        assert d.shape == (1, 10)

    def test_geometric_mean(self):
        es2 = ElementStats.from_data(["megnet_1", "megnet_3"], stats=["shifted_geometric_mean:100"])
        d = es2.transform_one("Fe2O3")
        assert d.shape == (1, 32)

    def test_initialization(self):
        self.assertRaises(
            TypeError,
            ElementStats,
            element_properties={"H": [1, 2], "O": [1, 2]},
            stats=["mean"],
            property_names=["p1", "p2"],
            some_random_variable="test",
        )
        es = ElementStats(
            element_properties={"H": [1, 2], "O": [1, 2]}, stats=["mean"], property_names=["p1", "p2"], n_jobs=-1
        )
        assert es.n_jobs > 0
        res = es.transform(["H2O", "H2O", "H2O", "H2O"])
        assert res.shape[0] == 4
        es.clear_cache()


def test_element_stats_property_name_mismatch():
    with pytest.raises(ValueError, match="Property name length is not consistent"):
        ElementStats(element_properties={"H": [1, 2], "He": [3, 4]}, property_names=["a"], stats=["mean"])


def test_element_stats_invalid_reduction_algo():
    with pytest.raises(ValueError, match="Reduction algorithm not available"):
        ElementStats(element_properties={"H": [1, 2], "He": [3, 4]}, stats=["mean"], num_dim=1, reduction_algo="tsne")


@pytest.mark.parametrize(("key", "expected"), [("D", True), ("T", True), ("Fe", True), ("Fe2+", True), ("Xx", False)])
def test_is_element_or_specie(key, expected):
    assert _is_element_or_specie(key) is expected


if __name__ == "__main__":
    unittest.main()
