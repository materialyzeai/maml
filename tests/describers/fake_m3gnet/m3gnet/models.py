"""Fake ``m3gnet.models`` with a deterministic M3GNet stand-in."""

from __future__ import annotations

import types
import weakref

import numpy as np

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


class M3GNet:
    n_blocks = 3
    threebody_cutoff = 4.0
    loaded_from: list[str] = []

    def __init__(self):
        # Like a TF/Keras model, the fake cannot be pickled (TypeError: cannot pickle 'weakref.ReferenceType').
        self._self_ref = weakref.ref(self)
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
