"""
Describer for converting (structural) objects into models-readable
numeric vectors or tensors.
"""

from __future__ import annotations

from ._composition import ElementProperty, ElementStats
from ._m3gnet import M3GNetSite, M3GNetStructure
from ._matgl import MatGLSite, MatGLStructure
from ._matminer import wrap_matminer_describer
from ._megnet import MEGNetSite, MEGNetStructure
from ._rdf import RadialDistributionFunction
from ._site import BispectrumCoefficients, BPSymmetryFunctions, SiteElementProperty, SmoothOverlapAtomicPosition
from ._structure import (
    CoulombEigenSpectrum,
    CoulombMatrix,
    DistinctSiteProperty,
    RandomizedCoulombMatrix,
    SortedCoulombMatrix,
)

__all__ = [
    "BPSymmetryFunctions",
    "BispectrumCoefficients",
    "CoulombEigenSpectrum",
    "CoulombMatrix",
    "DistinctSiteProperty",
    "ElementProperty",
    "ElementStats",
    "M3GNetSite",
    "M3GNetStructure",
    "MEGNetSite",
    "MEGNetStructure",
    "MatGLSite",
    "MatGLStructure",
    "RadialDistributionFunction",
    "RandomizedCoulombMatrix",
    "SiteElementProperty",
    "SmoothOverlapAtomicPosition",
    "SortedCoulombMatrix",
    "wrap_matminer_describer",
]
