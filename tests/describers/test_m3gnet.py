"""Tests for the legacy TF M3GNet describers.

The TF-based ``m3gnet`` package is not installable alongside current TensorFlow, so the
``m3gnet`` modules are replaced by lightweight fakes to exercise maml's own logic.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from maml.describers import M3GNetSite, M3GNetStructure, _m3gnet

ATOMS, BONDS = 0, 1
N_FEAT = 4


class _Converter:
    def convert(self, structure):
        return types.SimpleNamespace(as_list=lambda: [np.zeros((len(structure), N_FEAT)), "bonds", len(structure)])


class _GraphLayer:
    """Adds ``offset`` to atom features, so each block's output is distinguishable."""

    def __init__(self, offset: float):
        self.offset = offset

    def __call__(self, g):
        g = list(g)
        g[ATOMS] = g[ATOMS] + self.offset
        return g


class _Readout:
    def __init__(self, n_out: int):
        self.n_out = n_out

    def __call__(self, g):
        return np.full((1, self.n_out), g[ATOMS].sum())


class FakeM3GNet:
    n_blocks = 3
    threebody_cutoff = 4.0
    loaded_from: list[str] = []

    def __init__(self):
        self.graph_converter = _Converter()
        self.three_interactions = [lambda g, basis, cutoff: g for _ in range(self.n_blocks)]
        self.graph_layers = [_GraphLayer(i + 1) for i in range(self.n_blocks)]
        self.layers = [None, types.SimpleNamespace(layers=[_Readout(5)]), None]

    @classmethod
    def from_dir(cls, path):
        cls.loaded_from.append(path)
        return cls()

    def basis_expansion(self, graph):
        return "basis"

    def featurizer(self, graph):
        g = list(graph)
        g[ATOMS] = g[ATOMS] + 0.5  # embedding output
        return g

    def feature_adjust(self, g):
        return g


@pytest.fixture
def fake_m3gnet(monkeypatch):
    FakeM3GNet.loaded_from = []
    pkg = types.ModuleType("m3gnet")
    models = types.ModuleType("m3gnet.models")
    models.M3GNet = FakeM3GNet
    graph = types.ModuleType("m3gnet.graph")
    graph.Index = types.SimpleNamespace(ATOMS=ATOMS, BONDS=BONDS)
    graph.tf_compute_distance_angle = lambda g: g
    layers = types.ModuleType("m3gnet.layers")
    layers.polynomial = lambda bonds, cutoff: ("cutoff", bonds, cutoff)
    for name, mod in {"m3gnet": pkg, "m3gnet.models": models, "m3gnet.graph": graph, "m3gnet.layers": layers}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return FakeM3GNet


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
