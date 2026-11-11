from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Composition, Lattice, Molecule, Structure

from maml.utils._typing import OptStrOrCallable, StructureOrMolecule, VectorLike


@pytest.mark.parametrize(
    ("obj", "alias", "expected"),
    [
        (Structure(Lattice.cubic(3), ["Mo"], [[0, 0, 0]]), StructureOrMolecule, True),
        (Molecule(["H"], [[0, 0, 0]]), StructureOrMolecule, True),
        (Composition("H2O"), StructureOrMolecule, False),
        ("mean", OptStrOrCallable, True),
        (len, OptStrOrCallable, True),
        (None, OptStrOrCallable, True),
        (1, OptStrOrCallable, False),
    ],
)
def test_aliases_usable_at_runtime(obj, alias, expected):
    # The aliases are runtime union types, so they can be used in isinstance checks.
    assert isinstance(obj, alias) is expected


def test_vector_like_members():
    # list[float] is a parameterized generic, so VectorLike is for annotations only (no isinstance).
    assert VectorLike.__args__ == (list[float], np.ndarray)
