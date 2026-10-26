"""Tests for the cvxpy / scipy selector dispatch in maml.apps.symbolic."""

from __future__ import annotations

import importlib
import sys

import pytest

import maml.apps.symbolic as symbolic
from maml.apps.symbolic import _selectors


@pytest.fixture
def symbolic_without_cvxpy(monkeypatch):
    """Reload maml.apps.symbolic as if cvxpy were not installed, restoring it afterwards."""
    monkeypatch.setitem(sys.modules, "cvxpy", None)  # makes `import cvxpy` raise ImportError
    yield importlib.reload(symbolic)
    monkeypatch.undo()
    importlib.reload(symbolic)


def test_falls_back_to_scipy_selectors(symbolic_without_cvxpy):
    assert symbolic_without_cvxpy.cp is None
    assert symbolic_without_cvxpy.Lasso is _selectors.Lasso
    assert symbolic_without_cvxpy.AdaptiveLasso is _selectors.AdaptiveLasso
    assert symbolic_without_cvxpy.DantzigSelector is _selectors.DantzigSelector


def test_uses_cvxpy_selectors_when_available():
    pytest.importorskip("cvxpy")
    from maml.apps.symbolic import _selectors_cvxpy

    assert symbolic.cp is not None
    assert symbolic.Lasso is _selectors_cvxpy.LassoCP
    assert symbolic.AdaptiveLasso is _selectors_cvxpy.AdaptiveLassoCP
    assert symbolic.DantzigSelector is _selectors_cvxpy.DantzigSelectorCP


def test_public_api():
    assert sorted(symbolic.__all__) == sorted(
        [
            "ISIS",
            "SCAD",
            "SIS",
            "AdaptiveLasso",
            "DantzigSelector",
            "FeatureGenerator",
            "L0BrutalForce",
            "Lasso",
            "Operator",
        ]
    )
    for name in symbolic.__all__:
        assert hasattr(symbolic, name)
