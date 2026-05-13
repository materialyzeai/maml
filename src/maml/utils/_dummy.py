"""Dummy test systems."""

from __future__ import annotations

from pymatgen.core import Composition, Lattice, Molecule, Structure

from ._data_conversion import to_array
from ._inspect import get_param_types

DUMMY_OBJECTS = {
    "str": "H2O",
    "composition": Composition("H2O"),
    "structure": Structure(Lattice.cubic(3.167), ["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]]),  # type: ignore[arg-type]
    "molecule": Molecule(["C", "O"], [[0, 0, 0], [1, 0, 0]]),  # type: ignore[arg-type]
}


def get_describer_dummy_obj(instance):
    """
    For a describers, get a dummy object for transform_one.
    This relies on the type hint.

    Args:
        instance (BaseDescriber): describers instance
    """
    obj_type = getattr(instance, "describer_type", None)
    if obj_type is not None:
        return DUMMY_OBJECTS[obj_type.lower()]
    arg_types = get_param_types(instance.transform_one)
    arg_type = next(iter(arg_types.values()))
    str_t = str(arg_type)
    if "." in str_t:
        str_t = str_t.rsplit(".", maxsplit=1)[-1]
    return DUMMY_OBJECTS[str_t]


def feature_dim_from_test_system(describer):
    """
    Get feature size from a test system.

    Args:
        describer (BaseDescriber): describers instance
    """
    dummy_obj = get_describer_dummy_obj(describer)
    features = to_array(describer.transform_one(dummy_obj))
    if features.ndim == 1:
        return None

    return features.shape[-1]
