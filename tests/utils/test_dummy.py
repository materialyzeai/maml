from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Composition, Molecule, Structure

from maml.base import BaseDescriber, describer_type
from maml.utils._dummy import DUMMY_OBJECTS, feature_dim_from_test_system, get_describer_dummy_obj


class _StructureHinted(BaseDescriber):
    def transform_one(self, obj: Structure):
        return np.array([[len(obj), obj.volume, obj.density]])


class _CompositionHinted(BaseDescriber):
    def transform_one(self, obj: Composition):
        return np.array([obj.num_atoms])


@describer_type("Molecule")
class _MoleculeTyped(BaseDescriber):
    def transform_one(self, obj):
        return np.ones((len(obj), 4))


def test_dummy_obj_from_describer_type():
    assert get_describer_dummy_obj(_MoleculeTyped()) is DUMMY_OBJECTS["molecule"]


@pytest.mark.parametrize(
    ("describer", "expected_type"),
    [(_StructureHinted(), Structure), (_CompositionHinted(), Composition)],
)
def test_dummy_obj_from_type_hint(describer, expected_type):
    """Regression: hint-derived class names were not lower-cased, raising KeyError."""
    assert isinstance(get_describer_dummy_obj(describer), expected_type)


def test_dummy_obj_from_real_type_hint():
    def transform_one(obj):
        return obj

    transform_one.__annotations__ = {"obj": Molecule}

    class _Instance:
        pass

    inst = _Instance()
    inst.transform_one = transform_one
    assert get_describer_dummy_obj(inst) is DUMMY_OBJECTS["molecule"]


@pytest.mark.parametrize(
    ("describer", "dim"),
    [(_StructureHinted(), 3), (_CompositionHinted(), None), (_MoleculeTyped(), 4)],
)
def test_feature_dim_from_test_system(describer, dim):
    assert feature_dim_from_test_system(describer) == dim
