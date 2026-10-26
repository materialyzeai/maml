from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pymatgen.util.testing import PymatgenTest
from sklearn.base import BaseEstimator

from maml.describers import wrap_matminer_describer
from maml.utils import to_composition

try:
    from matminer.featurizers import composition
except ImportError:
    composition = None


class FakeFeaturizer(BaseEstimator):
    """Fake matminer featurizer (matminer's BaseFeaturizer is an sklearn BaseEstimator)."""

    def __init__(self, scale: float = 1.0, prefix: str = "f"):
        self.scale = scale
        self.prefix = prefix

    def featurize(self, comp):
        return [comp.num_atoms * self.scale, len(comp.elements)]

    def feature_labels(self):
        return [f"{self.prefix}_natoms", f"{self.prefix}_nelements"]

    @classmethod
    def from_preset(cls, name: str):
        if name == "double":
            return cls(scale=2.0, prefix="p")
        raise ValueError(f"Unknown preset {name}")


@pytest.fixture
def wrapped():
    return wrap_matminer_describer("FakeFeaturizer", FakeFeaturizer, to_composition, describer_type="composition")


def test_wrapped_class_attributes(wrapped):
    assert wrapped.__name__ == "FakeFeaturizer"
    assert wrapped.__module__ == "maml.describers"
    assert wrapped.__doc__ == FakeFeaturizer.__doc__
    assert wrapped.describer_type == "composition"
    assert wrapped._get_param_names() == ["prefix", "scale"]


def test_transform_one(wrapped):
    describer = wrapped()
    df = describer.transform_one("Li2O")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["f_natoms", "f_nelements"]
    np.testing.assert_array_equal(df.to_numpy(), [[3.0, 2]])
    # Structures are converted to compositions by obj_conversion.
    s = PymatgenTest.get_structure("Li2O")
    np.testing.assert_array_equal(describer.transform_one(s).to_numpy(), [[s.num_sites, 2]])


def test_constructor_params_are_kept(wrapped):
    """Regression: constructor args used to be passed to a throwaway wrapped instance."""
    describer = wrapped(scale=10.0, prefix="x")
    assert describer.get_params() == {"scale": 10.0, "prefix": "x"}
    df = describer.transform_one("Li2O")
    assert list(df.columns) == ["x_natoms", "x_nelements"]
    np.testing.assert_array_equal(df.to_numpy(), [[30.0, 2]])


def test_base_describer_kwargs(wrapped):
    describer = wrapped(n_jobs=1, verbose=True)
    assert describer.n_jobs == 1
    assert describer.verbose is True
    df = describer.transform(["Li2O", "Fe2O3", "NaCl"])
    assert df.shape == (3, 2)
    np.testing.assert_array_equal(df["f_natoms"].to_numpy(), [3, 5, 2])


@pytest.mark.parametrize("key", ["feature_batch", "feature_concat"])
def test_feature_batch_kwarg(wrapped, key):
    """Regression: feature_batch used to be forwarded to the wrapped class and raise TypeError."""
    describer = wrapped(**{key: "stack_first_dim"})
    feats = describer.transform(["Li2O", "NaCl"])
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (2, 1, 2)


def test_invalid_kwarg(wrapped):
    with pytest.raises(TypeError):
        wrapped(bogus=1)


def test_from_preset(wrapped):
    describer = wrapped.from_preset("double", n_jobs=1)
    assert isinstance(describer, wrapped)
    assert describer.n_jobs == 1
    assert describer.get_params() == {"scale": 2.0, "prefix": "p"}
    np.testing.assert_array_equal(describer.transform_one("Li2O").to_numpy(), [[6.0, 2]])
    with pytest.raises(ValueError, match="Unknown preset"):
        wrapped.from_preset("bogus")


@pytest.mark.skipif(composition is None, reason="matminer package is needed")
class TestRealMatminer:
    def test_element_prop(self):
        ElementProperty = wrap_matminer_describer(
            "ElementProperty", composition.ElementProperty, to_composition, describer_type="composition"
        )
        ep = ElementProperty.from_preset("magpie")
        s_li2o = PymatgenTest.get_structure("Li2O")
        assert np.allclose(ep.transform_one(s_li2o).values, ep.transform_one("Li2O").values)
        assert ep._get_param_names() == ["data_source", "features", "stats"]

    def test_atomic_orbitals(self):
        AtomicOrbitals = wrap_matminer_describer(
            "AtomicOrbitals", composition.AtomicOrbitals, to_composition, describer_type="composition"
        )
        ao = AtomicOrbitals()
        s_lfp = PymatgenTest.get_structure("LiFePO4")
        for i, j in zip(ao.transform_one("LiFePO4").values, ao.transform_one(s_lfp).values, strict=True):
            for k, li in zip(i, j, strict=True):
                assert k == li
