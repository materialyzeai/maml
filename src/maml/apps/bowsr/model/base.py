"""Base class that expose a predict_energy method."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure


class EnergyModel:
    """
    Base energy model class. For any model used in BOWSR, it has to have
    a predict_energy method that returns a float.
    """

    def predict_energy(self, structure: Structure) -> float:
        """
        Predict the energy of a structure
        Args:
            structure: Pymatgen Structure object.

        Returns: (float) energy value.

        """
        raise NotImplementedError
