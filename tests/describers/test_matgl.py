from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from pymatgen.core import Lattice, Structure

from maml.describers import MatGLSite, MatGLStructure, _matgl


@pytest.fixture(scope="module")
def nacl() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.69169), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]])


@pytest.fixture(scope="module")
def default_model():
    """Load the default pretrained model (~1.6 MB, cached by matgl after first download)."""
    try:
        model, _ = _matgl._load_matgl_model(None)
    except RuntimeError as exc:
        if isinstance(exc.__cause__, OSError):
            pytest.skip(f"Unable to download {_matgl.DEFAULT_MODEL}: {exc.__cause__}")
        raise
    return model


class FakeMatGLModel:
    """Minimal stand-in for a matgl M3GNet exposing predict_structure/feature_dict."""

    n_blocks = 3

    def __init__(self, n_feat: int = 4, n_readout: int = 6):
        self.n_feat = n_feat
        self.n_readout = n_readout
        self.feature_dict: dict = {}
        self.calls = 0

    def predict_structure(self, structure, **kwargs):
        self.calls += 1
        n = len(structure)
        fea: dict = {
            "embedding": {"node_feat": torch.zeros(n, self.n_feat)},
            "readout": torch.arange(self.n_readout, dtype=torch.float32)[None, :],
        }
        for i in range(self.n_blocks):
            fea[f"gc_{i + 1}"] = {"node_feat": torch.full((n, self.n_feat), float(i + 1))}
        self.feature_dict = fea
        return torch.tensor(0.0)


class FakeWrapper:
    def __init__(self, model):
        self.model = model


@pytest.fixture
def fake_load(monkeypatch):
    loaded_paths = []

    def _load_model(path):
        loaded_paths.append(path)
        return FakeWrapper(FakeMatGLModel())

    monkeypatch.setattr(_matgl.matgl, "load_model", _load_model)
    return loaded_paths


@pytest.mark.parametrize(("model_path", "expected"), [(None, _matgl.DEFAULT_MODEL), ("my/model", "my/model")])
def test_load_matgl_model_path(fake_load, model_path, expected):
    model, path = _matgl._load_matgl_model(model_path)
    assert path == expected
    assert fake_load == [expected]
    assert isinstance(model, FakeMatGLModel)


def test_load_matgl_model_unwrapped(monkeypatch):
    bare = FakeMatGLModel()
    monkeypatch.setattr(_matgl.matgl, "load_model", lambda path: bare)
    model, _ = _matgl._load_matgl_model(None)
    assert model is bare


def test_structure_mocked(fake_load, nacl):
    describer = MatGLStructure()
    assert describer.model_path == _matgl.DEFAULT_MODEL
    assert describer.describer_type == "structure"
    feat = describer.transform_one(nacl)
    np.testing.assert_array_equal(feat, np.arange(6))
    assert np.array(describer.transform([nacl] * 2)).shape == (2, 6)


def test_site_mocked(fake_load, nacl):
    describer = MatGLSite(output_layers=["embedding", "gc_2"])
    assert describer.describer_type == "site"
    df = describer.transform_one(nacl)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (len(nacl), 8)
    # embedding columns first (zeros), then gc_2 (all 2.0); order follows output_layers.
    np.testing.assert_array_equal(df.iloc[:, :4].to_numpy(), 0.0)
    np.testing.assert_array_equal(df.iloc[:, 4:].to_numpy(), 2.0)

    assert MatGLSite().output_layers == ["gc_1"]

    as_dict = MatGLSite(output_layers=["gc_3"], return_type={}).transform_one(nacl)
    assert set(as_dict) == {"gc_3"}
    np.testing.assert_array_equal(as_dict["gc_3"], np.full((len(nacl), 4), 3.0))

    batched = MatGLSite(feature_batch="pandas_concat").transform([nacl] * 2)
    assert batched.shape == (2 * len(nacl), 4)


@pytest.mark.parametrize("output_layers", [["whatever"], ["gc_4"], "gc_1", ("gc_1",)])
def test_site_invalid_output_layers(fake_load, output_layers):
    with pytest.raises(ValueError, match="Invalid output_layers"):
        MatGLSite(output_layers=output_layers)


def test_site_transform_real_model(default_model, monkeypatch, nacl):
    monkeypatch.setattr(_matgl, "_load_matgl_model", lambda path: (default_model, _matgl.DEFAULT_MODEL))
    atom_feat_2s = MatGLSite(feature_batch="pandas_concat").transform([nacl] * 2)
    assert np.array(atom_feat_2s).shape == (16, 64)
    atom_feat_2s_2l = MatGLSite(output_layers=["embedding", "gc_3"], feature_batch="pandas_concat").transform(
        [nacl] * 2
    )
    assert np.array(atom_feat_2s_2l).shape == (16, 128)
    # All sites of the same species are symmetry-equivalent in rock salt, so their features must match.
    df = MatGLSite().transform_one(nacl)
    na = [i for i, site in enumerate(nacl) if site.specie.symbol == "Na"]
    np.testing.assert_allclose(df.iloc[na].to_numpy(), np.tile(df.iloc[na[0]].to_numpy(), (len(na), 1)), atol=1e-5)


def test_structure_transform_real_model(default_model, monkeypatch, nacl):
    monkeypatch.setattr(_matgl, "_load_matgl_model", lambda path: (default_model, _matgl.DEFAULT_MODEL))
    struct_feat_2s = MatGLStructure().transform([nacl] * 2)
    feats = np.array(struct_feat_2s)
    assert feats.shape == (2, 128)
    np.testing.assert_allclose(feats[0], feats[1])
