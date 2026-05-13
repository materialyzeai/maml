"""DFT wrapper."""

from __future__ import annotations

import os
import subprocess
from shutil import which
from typing import TYPE_CHECKING

from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.sets import MPStaticSet

from maml.apps.bowsr.model.base import EnergyModel

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

module_dir = os.path.dirname(__file__)
elements_filename = os.path.join(module_dir, "..", "regularization", "elements.json")
elements = loadfn(elements_filename)


class DFT(EnergyModel):
    """DFT static calculation wrapped as energy model."""

    def __init__(self, exe_path: str | None = None):
        """
        DFT wrapper
        Args:
            exe_path: VASP executable path.
        """
        if not exe_path:
            if not which("vasp_std"):
                raise RuntimeError("Vasp executable can not be found.")
            self.vasp_exe = which("vasp_std")
        else:
            self.vasp_exe = exe_path

    def predict_energy(self, structure: Structure):
        """
        Predict energy from structure.

        Args:
            structure: (pymatgen Structure).

        Returns: float
        """
        el_amt_dict = structure.composition.get_el_amt_dict()

        static = MPStaticSet(structure)
        with ScratchDir("."):
            static.write_input(".")
            with subprocess.Popen(
                [self.vasp_exe],  # type:ignore[list-item]
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # type: ignore
            ) as p_exe:
                stdout, stderr = p_exe.communicate()
                rc = p_exe.returncode
            if rc != 0:
                error_msg = f"vasp exited with return code {rc}"
                msg = stderr.decode("utf-8").split("\n")[:-1]
                try:
                    error_line = next(i for i, m in enumerate(msg) if m.startswith("ERROR"))
                    error_msg += ", ".join(msg[error_line:])
                except Exception:
                    error_msg += ", "
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)

            compat = MaterialsProjectCompatibility()
            vrun = Vasprun("vasprun.xml")
            entry = compat.process_entry(vrun.get_computed_entry())  # type: ignore[arg-type]
            return (
                entry.energy  # type: ignore[union-attr]
                - sum(elements[el]["energy_per_atom"] * amt for el, amt in el_amt_dict.items())
            ) / len(structure)
