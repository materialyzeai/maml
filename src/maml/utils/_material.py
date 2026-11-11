"""Materials utils."""

from __future__ import annotations

from pymatgen.core import Composition, Molecule, Structure


def to_composition(obj: Composition | Molecule | Structure | str) -> Composition:
    """
    Convert str/structure or composition to compositions.

    Args:
        obj (str/structure/composition): object to convert

    Returns:
        Composition object
    """
    if isinstance(obj, str):
        return Composition(obj)
    if isinstance(obj, Structure | Molecule):
        return obj.composition
    return obj
