from __future__ import annotations

import unittest

from pymatgen.core import SETTINGS
from pymatgen.util.testing import PymatgenTest

from maml.data._mp import MaterialsProject


@unittest.skipIf(not SETTINGS.get("PMG_MAPI_KEY"), "PMG_MAPI_KEY environment variable not set")
class MaterialsProjectTest(PymatgenTest):
    def test_get(self):
        mp = MaterialsProject(SETTINGS.get("PMG_MAPI_KEY"))
        features = ["band_gap", "formation_energy_per_atom", "energy_above_hull"]
        df = mp.get(criteria={"formula": "Li2O"}, properties=features)
        assert df.shape[0] >= 6
        assert df.shape[1] == len(features)
        assert set(df.columns) == set(features)


if __name__ == "__main__":
    unittest.main()
