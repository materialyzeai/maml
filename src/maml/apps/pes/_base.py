# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.
"""Base classes for potentials."""

from __future__ import annotations

import abc

from maml.base import BaseModel


class PotentialMixin:
    """Abstract base class for a Interatomic Potential."""

    @abc.abstractmethod
    def train(self, train_structures, train_energies, train_forces, train_stresses, **kwargs):
        """
        Train interatomic potentials with energies, forces ann stresses corresponding
            to structures.

        Args:
            train_structures (list): List of Pymatgen Structure objects.
            train_energies (list): List of DFT-calculated total energies of each
                structure in structures list.
            train_forces (list): List of DFT-calculated (m, 3) forces of each
                structure with m atoms in structures list. m can be varied with
                each single structure case.
            train_stresses (list): List of DFT-calculated (6, ) virial stresses of
                each structure in structures list.
            **kwargs: optional keywords that can be added in the future.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, test_structures, test_energies, test_forces, test_stresses):
        """
        Evaluate energies, forces and stresses of structures with trained
        interatomic potentials.

        Args:
            test_structures (list): List of Pymatgen Structure Objects.
            test_energies (list): List of DFT-calculated total energies of each
                structure in structures list.
            test_forces (list): List of DFT-calculated (m, 3) forces of each
                structure with m atoms in structures list. m can be varied with
                each single structure case.
            test_stresses (list): List of DFT-calculated (6, ) viriral stresses of
                each structure in structures list.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_efs(self, structure):
        """
        Predict energy, forces and stresses of the structure.

        Args:
            structure (Structure): Pymatgen Structure object.

        Returns:
            energy, forces, stress
        """

    @abc.abstractmethod
    def write_param(self):
        """
        Write optimized weights file to perform energy and force prediction via
        lammps calculation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def from_config(self, **kwargs):
        """Initialize potentials with parameters file."""
        raise NotImplementedError


class Potential(PotentialMixin, BaseModel):  # type: ignore
    """
    Potential models that can be used to fit structure-[energy, force, stress]
    pairs.
    """


def _check_returncode(program: str, returncode: int, output: bytes | None) -> None:
    """
    Raise an informative RuntimeError if an external program failed.

    Args:
        program (str): Name of the program, used in the error message.
        returncode (int): Return code of the finished process.
        output (bytes | None): Captured stdout/stderr of the process. The lines
            from the first one starting with "ERROR" onwards are reported, or
            the last line if no such line exists.

    Raises:
        RuntimeError: If returncode is non-zero.
    """
    if returncode == 0:
        return
    error_msg = f"{program} exited with return code {returncode}"
    lines = [line for line in (output or b"").decode("utf-8", errors="replace").splitlines() if line.strip()]
    error_line = next((i for i, line in enumerate(lines) if line.startswith("ERROR")), None)
    if error_line is not None:
        error_msg += ": " + ", ".join(lines[error_line:])
    elif lines:
        error_msg += ": " + lines[-1]
    raise RuntimeError(error_msg)
