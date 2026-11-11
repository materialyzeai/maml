"""Tests for the MEGNet energy-model wrapper with the megnet package faked."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest
from pymatgen.core import Element, Lattice, Structure
from pymatgen.core.periodic_table import DummySpecies

import maml.apps.bowsr.model.megnet as megnet_module


class FakeGaussianDistance:
    def __init__(self, centers, width):
        self.centers = centers
        self.width = width


class FakeCrystalGraph:
    def __init__(self, bond_converter=None, cutoff=None):
        self.bond_converter = bond_converter
        self.cutoff = cutoff
        self.converted = []

    def convert(self, structure):
        self.converted.append(structure)
        return {"atom": None, "bond": [], "n_sites": len(structure)}


class FakeMEGNetModel:
    loaded_from: list = []

    def __init__(self, nfeat_edge=None, nfeat_global=None, nfeat_node=None, npass=None, graph_converter=None):
        self.init_args = (nfeat_edge, nfeat_global, nfeat_node, npass)
        self.graph_converter = graph_converter
        rng = np.random.RandomState(0)
        self.weights = [rng.normal(size=(95, 16)), np.ones(3), np.zeros(2)]
        self.graphs = []

    @classmethod
    def from_file(cls, filename):
        cls.loaded_from.append(filename)
        return cls()

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def predict_structure(self, structure):
        return np.array([-1.5 * len(structure)])

    def predict_graph(self, graph):
        self.graphs.append(graph)
        return np.array([float(np.sum(graph["atom"]))])


@pytest.fixture
def megnet_wrapper(monkeypatch):
    """Reload the wrapper module with a fake ``megnet`` package installed."""
    pkg = types.ModuleType("megnet")
    data = types.ModuleType("megnet.data")
    crystal = types.ModuleType("megnet.data.crystal")
    graph = types.ModuleType("megnet.data.graph")
    models = types.ModuleType("megnet.models")
    crystal.CrystalGraph = FakeCrystalGraph
    graph.GaussianDistance = FakeGaussianDistance
    models.MEGNetModel = FakeMEGNetModel
    for name, mod in {
        "megnet": pkg,
        "megnet.data": data,
        "megnet.data.crystal": crystal,
        "megnet.data.graph": graph,
        "megnet.models": models,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    FakeMEGNetModel.loaded_from = []
    yield importlib.reload(megnet_module)
    monkeypatch.undo()
    with pytest.warns(UserWarning, match="megnet module should be installed"):
        importlib.reload(megnet_module)


@pytest.fixture
def cscl() -> Structure:
    return Structure(Lattice.cubic(4.1), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture
def disordered() -> Structure:
    return Structure(Lattice.cubic(4.1), [{"Fe": 0.25, "Co": 0.75}, "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_default_model_loaded_from_bundled_file(megnet_wrapper):
    model = megnet_wrapper.MEGNet()
    assert FakeMEGNetModel.loaded_from == [megnet_wrapper.model_filename]
    assert megnet_wrapper.model_filename.endswith("formation_energy.hdf5")
    assert isinstance(model.model, FakeMEGNetModel)
    assert not model.reconstruct
    assert model.embedding.shape == (95, 16)


def test_predict_ordered(megnet_wrapper, cscl):
    inner = FakeMEGNetModel()
    model = megnet_wrapper.MEGNet(model=inner)
    assert model.model is inner
    assert FakeMEGNetModel.loaded_from == []
    assert model.predict_energy(cscl) == pytest.approx(-3.0)


def test_predict_disordered_requires_reconstruct(megnet_wrapper, disordered):
    model = megnet_wrapper.MEGNet(model=FakeMEGNetModel())
    with pytest.raises(ValueError, match="reconstruct=True"):
        model.predict_energy(disordered)


def test_reconstruct_builds_new_model(megnet_wrapper):
    inner = FakeMEGNetModel()
    model = megnet_wrapper.MEGNet(model=inner, reconstruct=True, gaussian_cutoff=4, radius_cutoff=3.5, npass=3)
    assert model.model is not inner
    assert model.model.init_args == (100, 2, 16, 3)
    # The embedding layer is split off; the remaining weights are transferred.
    assert len(model.model.weights) == len(inner.weights) - 1
    converter = model.model.graph_converter
    assert converter.cutoff == 3.5
    assert converter.bond_converter.centers[-1] == 4
    assert len(converter.bond_converter.centers) == 100
    assert converter.bond_converter.width == 0.5


def test_reconstruct_predict_uses_occupancy_weighted_embedding(megnet_wrapper, disordered):
    model = megnet_wrapper.MEGNet(model=FakeMEGNetModel(), reconstruct=True)
    energy = model.predict_energy(disordered)

    emb = model.embedding
    expected_atom = [
        0.25 * emb[Element("Fe").Z] + 0.75 * emb[Element("Co").Z],
        emb[Element("O").Z],
    ]
    graph = model.model.graphs[-1]
    assert len(graph["atom"]) == 2
    for actual, expected in zip(graph["atom"], expected_atom, strict=True):
        assert np.allclose(actual, expected)
    assert energy == pytest.approx(float(np.sum(expected_atom)))
    # The graph is built from a dummy-species copy; the input is untouched.
    converted = model.model.graph_converter.converted[-1]
    assert all(isinstance(sp, DummySpecies) for sp in converted.species)
    assert not disordered.is_ordered


def test_requires_megnet():
    # Without megnet installed (the default test environment) the wrapper refuses to build.
    if megnet_module.megnet is not None:
        pytest.skip("megnet is installed")
    with pytest.raises(RuntimeError, match="megnet package needs to be installed"):
        megnet_module.MEGNet()
