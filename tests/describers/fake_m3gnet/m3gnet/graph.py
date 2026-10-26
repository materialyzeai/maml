"""Fake ``m3gnet.graph``."""

from __future__ import annotations

import types

Index = types.SimpleNamespace(ATOMS=0, BONDS=1)


def tf_compute_distance_angle(graph):
    return graph
