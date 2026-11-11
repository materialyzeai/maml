"""Tests for the legacy TF M3GNet describers.

The TF-based ``m3gnet`` package is not installable alongside current TensorFlow, so it is replaced by
the lightweight fake in ``fake_m3gnet/`` to exercise maml's own logic.
"""

from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import parallel_config
from pymatgen.core import Lattice, Structure

from maml.describers import M3GNetSite, M3GNetStructure, _m3gnet

FAKE_M3GNET_DIR = Path(__file__).parent / "fake_m3gnet"
N_FEAT = 4


def _drop_m3gnet_modules():
    for name in [m for m in sys.modules if m == "m3gnet" or m.startswith("m3gnet.")]:
        del sys.modules[name]


@pytest.fixture
def fake_m3gnet(monkeypatch):
    """Put the fake ``m3gnet`` on sys.path, which joblib also hands to the worker processes it spawns."""
    _drop_m3gnet_modules()
    monkeypatch.syspath_prepend(str(FAKE_M3GNET_DIR))
    fake = importlib.import_module("m3gnet.models").M3GNet
    fake.loaded_from = []
    _m3gnet._cached_m3gnet_model.cache_clear()
    yield fake
    _m3gnet._cached_m3gnet_model.cache_clear()
    _drop_m3gnet_modules()


@pytest.fixture(scope="module")
def nacl() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]])


@pytest.mark.parametrize("cls", [M3GNetStructure, M3GNetSite])
def test_missing_m3gnet(monkeypatch, cls):
    monkeypatch.setitem(sys.modules, "m3gnet", None)
    monkeypatch.setitem(sys.modules, "m3gnet.models", None)
    with pytest.raises(ImportError):
        cls()


@pytest.mark.parametrize(("model_path", "expected"), [(None, str(_m3gnet.DEFAULT_MODEL)), ("/my/model", "/my/model")])
def test_model_path(fake_m3gnet, model_path, expected):
    describer = M3GNetStructure(model_path=model_path)
    assert describer.model_path == expected
    assert fake_m3gnet.loaded_from == [expected]


def test_structure_transform(fake_m3gnet, nacl):
    describer = M3GNetStructure()
    feat = describer.transform_one(nacl)
    # embedding 0.5, then +1, +2, +3 over the three blocks -> 6.5 per element.
    np.testing.assert_allclose(feat, np.full(5, 6.5 * len(nacl) * N_FEAT))
    assert np.array(describer.transform([nacl] * 2)).shape == (2, 5)


def test_site_transform(fake_m3gnet, nacl):
    df = M3GNetSite().transform_one(nacl)
    assert isinstance(df, pd.DataFrame)
    np.testing.assert_allclose(df.to_numpy(), 1.5)  # default gc_1

    df = M3GNetSite(output_layers=["embedding", "gc_3"]).transform_one(nacl)
    assert df.shape == (len(nacl), 2 * N_FEAT)
    np.testing.assert_allclose(df.iloc[:, :N_FEAT].to_numpy(), 0.5)
    np.testing.assert_allclose(df.iloc[:, N_FEAT:].to_numpy(), 6.5)

    as_dict = M3GNetSite(output_layers=["gc_2"], return_type={}).transform_one(nacl)
    assert list(as_dict) == ["gc_2"]
    np.testing.assert_allclose(as_dict["gc_2"], 3.5)

    batched = M3GNetSite(feature_batch="pandas_concat").transform([nacl] * 2)
    assert batched.shape == (2 * len(nacl), N_FEAT)


@pytest.mark.parametrize("output_layers", [["whatever"], ["gc_4"], "gc_1"])
def test_site_invalid_output_layers(fake_m3gnet, output_layers):
    with pytest.raises(ValueError, match="Invalid output_layers"):
        M3GNetSite(output_layers=output_layers)


@pytest.mark.parametrize("cls", [M3GNetStructure, M3GNetSite])
def test_pickle_drops_and_reloads_model(fake_m3gnet, nacl, cls):
    describer = cls(model_path="/my/model")
    with pytest.raises(TypeError, match="weakref"):
        pickle.dumps(describer.describer_model)
    restored = pickle.loads(pickle.dumps(describer))
    assert isinstance(restored.describer_model, fake_m3gnet)
    assert restored.describer_model is not describer.describer_model
    assert fake_m3gnet.loaded_from == ["/my/model", "/my/model"]
    assert restored.get_params() == describer.get_params()
    np.testing.assert_array_equal(np.asarray(restored.transform_one(nacl)), np.asarray(describer.transform_one(nacl)))
    # A second unpickle in the same process reuses the loaded model.
    assert pickle.loads(pickle.dumps(describer)).describer_model is restored.describer_model


@pytest.mark.parametrize(("cls", "kwargs"), [(M3GNetStructure, {}), (M3GNetSite, {"feature_batch": "pandas_concat"})])
def test_parallel_transform(fake_m3gnet, nacl, cls, kwargs):
    """Regression test for #637: M3GNet describers with n_jobs > 1 failed to pickle the TF model.

    The tasks go to real loky worker processes, as in the original report. A distinct ``idle_worker_timeout``
    makes joblib start fresh workers, which inherit the sys.path that holds the fake ``m3gnet``.
    """
    structures = [nacl, nacl.copy().scale_lattice(1.1 * nacl.volume)]
    serial = np.asarray(cls(**kwargs).transform(structures))
    with parallel_config(backend="loky", idle_worker_timeout=37):
        parallel = np.asarray(cls(n_jobs=2, **kwargs).transform(structures))
    np.testing.assert_array_equal(parallel, serial)
