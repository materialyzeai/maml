from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from pymatgen.core import Lattice, Structure

try:
    import dgl  # noqa: F401
    import matgl
except ImportError:
    matgl = None

from maml.describers import MatGLSite, MatGLStructure


@unittest.skipIf(matgl is None, "MatGL with DGL backend is required.")
class MatGLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]])

    def test_matgl_site_transform(self):
        atom_feat_2s = MatGLSite(feature_batch="pandas_concat").transform([self.s] * 2)
        self.assertListEqual(list(np.array(atom_feat_2s).shape), [16, 64])
        with pytest.raises(ValueError, match="Invalid output_layers"):
            MatGLSite(output_layers=["whatever"])
        atom_feat_2s_2l = MatGLSite(output_layers=["embedding", "gc_3"], feature_batch="pandas_concat").transform(
            [self.s] * 2
        )
        self.assertListEqual(list(np.array(atom_feat_2s_2l).shape), [16, 128])
        atom_feat_dict = MatGLSite(return_type=dict).transform_one(self.s)
        assert isinstance(atom_feat_dict, pd.DataFrame)

    def test_matgl_structure_transform(self):
        struct_feat_2s = MatGLStructure().transform([self.s] * 2)
        self.assertListEqual(list(np.array(struct_feat_2s).shape), [2, 128])
        # Tensorflow is tricky, as the M3GNet loaded here affects layers names of MEGNet loaded in MEGNet tests
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    unittest.main()
