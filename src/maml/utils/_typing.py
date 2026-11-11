"""Define several typing for convenient use."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pymatgen.core import Molecule, Structure

OptStrOrCallable = str | Callable | None
StructureOrMolecule = Structure | Molecule
VectorLike = list[float] | np.ndarray
