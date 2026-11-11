"""Symbolic learning."""

from __future__ import annotations

try:
    import cvxpy as cp
except ImportError:
    cp = None  # type: ignore[assignment]

from ._feature_generator import FeatureGenerator, Operator
from ._selectors import SCAD, L0BrutalForce
from ._sis import ISIS, SIS

if cp is None:
    from ._selectors import AdaptiveLasso, DantzigSelector, Lasso
else:
    # import from cvxpy alternatives
    from ._selectors_cvxpy import AdaptiveLassoCP as AdaptiveLasso  # type: ignore
    from ._selectors_cvxpy import DantzigSelectorCP as DantzigSelector  # type: ignore
    from ._selectors_cvxpy import LassoCP as Lasso  # type: ignore

__all__ = [
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
