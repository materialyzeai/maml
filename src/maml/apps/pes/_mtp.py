# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

"""This module provides MTP interatomic potential class."""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
from collections import OrderedDict
from shutil import which

import numpy as np
from monty.io import zopen
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from pymatgen.core import Lattice, Structure

from maml.utils import check_structures_forces_stresses, convert_docs, pool_from

from ._base import _check_returncode
from ._lammps import LammpsPotential

module_dir = os.path.dirname(__file__)
MTini_params = loadfn(os.path.join(module_dir, "params", "MTini.json"))


MLIP2_URL = "https://gitlab.com/ashapeev/mlip-2"
MLIP3_URL = "https://gitlab.com/ashapeev/mlip-3"
SUPPORTED_VERSIONS = ("mlip-2", "mlip-3", "mlip-dev")
SUPPORTED_VERSIONS_MSG = (
    'maml\'s MTPotential only supports the MLIP-2 (version="mlip-2"), MLIP-3 (version="mlip-3") and '
    '"mlip-dev" command-line interfaces, and `version` must match the installed mlp. '
    f"MLIP-2 is available at {MLIP2_URL} and MLIP-3 at {MLIP3_URL}."
)

# MLIP-3 `mlp train` has no --weighting option. Its --weight_scaling (energies and stresses) and
# --weight_scaling_forces options take the place of MLIP-2's --weighting, with the combinations
# recommended in MLIP-3's `mlp help train` for MD-sampled configurations, molecules and periodic
# structures.
MLIP3_WEIGHT_SCALING = {
    "vibration": (1, 0),
    "vibrations": (1, 0),
    "molecules": (0, 0),
    "structures": (2, 1),
}

# MLIP-3 appends its active-learning state (partly binary) to a trained potential after this marker.
_MLIP3_SELECTION_MARKER = b"\n#MVS_v"


def _output_tail(output: bytes | str | None, n_lines: int = 20) -> str:
    """Return the last ``n_lines`` non-empty lines of a process output stream."""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    lines = [line.rstrip() for line in (output or "").splitlines() if line.strip()]
    return "\n".join(lines[-n_lines:])


def _format_mlp_output(stdout: bytes | str | None, stderr: bytes | str | None, n_lines: int = 20) -> str:
    """Format the tails of ``mlp``'s stdout and stderr for an error message."""
    parts = []
    for name, stream in (("stdout", stdout), ("stderr", stderr)):
        tail = _output_tail(stream, n_lines)
        if tail:
            parts.append(f"--- mlp {name} (last {n_lines} lines) ---\n{tail}")
    return "\n".join(parts) if parts else "(mlp produced no output)"


def min_interatomic_distance(structures) -> float:
    """
    Minimum interatomic distance over a set of periodic structures.

    This is the quantity reported by ``mlp mindist`` and used as the ``min_dist`` of the MTP
    radial basis: the minimum over all pairs of atoms, including pairs between an atom and
    periodic images of itself or of other atoms.

    Args:
        structures ([Structure]): Pymatgen Structure objects.

    Returns:
        float: The minimum interatomic distance in Angstrom.

    Raises:
        ValueError: If no structures are given.
    """
    if not structures:
        raise ValueError("At least one structure is needed to compute the minimum interatomic distance.")
    min_dist = np.inf
    for structure in structures:
        # Every atom has periodic images of itself at the lattice vector lengths, so a cutoff just
        # above min(a, b, c) always finds a neighbour. Start smaller and grow the cutoff to keep
        # the neighbour list cheap for large cells.
        guaranteed = min(structure.lattice.abc) + 1e-3
        cutoff = min(3.0, guaranteed)
        while True:
            distances = structure.get_neighbor_list(cutoff)[3]
            if len(distances) > 0 or cutoff >= guaranteed:
                break
            cutoff = min(2 * cutoff, guaranteed)
        if len(distances) > 0:
            min_dist = min(min_dist, float(np.min(distances)))
    return float(min_dist)


_NON_FINITE = re.compile(r"(?<![\w.])[-+]?(?:nan|inf(?:inity)?)(?![\w.])", re.IGNORECASE)


def _load_mtp_param(filename) -> OrderedDict:
    """
    Read the parameters of an .mtp file.

    A potential trained by MLIP-3 is followed by its active-learning (selection) state, which
    starts with a "#MVS_v1.1" line and contains binary data. It is not needed to evaluate the
    potential and is dropped.

    Args:
        filename (str): The .mtp file.

    Returns:
        OrderedDict: "safe" holds all lines but the last two verbatim; the last two lines
            (species_coeffs and moment_coeffs) are parsed into lists of floats.

    Raises:
        ValueError: If the file contains NaN/inf values or cannot be parsed.
    """
    with open(filename, "rb") as f:
        data = f.read()
    text = data.split(_MLIP3_SELECTION_MARKER, 1)[0].decode("utf-8", errors="replace").rstrip()
    if match := _NON_FINITE.search(text):
        raise ValueError(f"{filename} contains non-finite parameter values (found {match.group()!r}).")
    lines = text.splitlines()
    param: OrderedDict = OrderedDict()
    param["safe"] = [line.rstrip() for line in lines[:-2]]
    try:
        for line in lines[-2:]:
            key, value = line.rstrip().split(" = ", 1)
            param[key] = json.loads(value.replace("{", "[").replace("}", "]"))
    except ValueError as exc:
        raise ValueError(f"Could not parse the parameters in {filename}: {exc}") from exc
    return param


def feed(attribute, kwargs, dictionary, tab="\t"):
    r"""
    Args:
        attribute (str): Attribute to be operated.
        kwargs (dict): Generic parameters.
        dictionary (dict): The default parameters dictionary.
        tab (str): '\t' or '\t\t', depend on orders of attribute.

    Return:
        (str).
    """
    # Explicit falsy values such as 0 or 0.0 must not be replaced by the default.
    tmp = kwargs[attribute] if kwargs.get(attribute) is not None else dictionary.get(attribute).get("value")
    return (
        tab + dictionary.get(attribute).get("name"),
        str(tmp),
        dictionary.get(attribute).get("comment"),
    )


class MTPotential(LammpsPotential):
    """
    This class implements moment tensor potentials.
    Installation of the mlip package is needed.
    Please refer to https://mlip.skoltech.ru.
    """

    pair_style = "pair_style        mlip {}"
    pair_coeff = "pair_coeff        * *"

    def __init__(self, name=None, param=None, version=None):
        """

        Args:
            name (str): Name of force field.
            param (dict): The parameter configuration of potentials.
            version (str): The version of mlip package: "mlip-3" (default), "mlip-2" or
                "mlip-dev". With "mlip-3", train() and evaluate() are supported, but the LAMMPS
                based methods (write_param, write_ini and everything built on them) are not, since
                MLIP-3's LAMMPS interface is a separate package with its own input format.
        """
        version = version if version else "mlip-3"
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported mlip version {version!r}. Choose from {SUPPORTED_VERSIONS}.")
        self.name = name if name else "MTPotential"
        self.mtp_stress_order = ["xx", "yy", "zz", "yz", "xz", "xy"]
        self.vasp_stress_order = ["xx", "yy", "zz", "xy", "yz", "xz"]
        self.param = param if param else None
        self.elements = None
        self.version = version

    def _line_up(self, structure, energy, forces, virial_stress):
        """
        Convert input structure, energy, forces, virial_stress to
        proper configuration format for mlip usage.

        Args:
            structure (Structure): Pymatgen Structure object.
            energy (float): DFT-calculated energy of the system.
            forces (list): The forces should have dimension (num_atoms, 3).
            virial_stress (list): stress should has 6 distinct
                elements arranged in order [xx, yy, zz, yz, xz, xy].
        """
        inputs = OrderedDict(
            Size=structure.num_sites,
            SuperCell=structure.lattice,
            AtomData=(structure, forces),
            Energy=energy,
            Stress=virial_stress,
        )

        lines = ["BEGIN_CFG"]

        if "Size" in inputs:
            lines.append(" Size")
            lines.append(f"{inputs['Size']:>7d}")
        if "SuperCell" in inputs:
            lines.append(" SuperCell")
            for vec in inputs["SuperCell"].matrix:
                lines.append(f"{vec[0]:>17.6f}{vec[1]:>14.6f}{vec[2]:>14.6f}")
        if "AtomData" in inputs:
            format_str = "{:>14s}{:>5s}{:>15s}{:>14s}{:>14s}{:>13s}{:>13s}{:>13s}"
            format_float = "{:>14d}{:>5d}{:>15f}{:>14f}{:>14f}{:>13f}{:>13f}{:>13f}"
            lines.append(
                format_str.format(
                    "AtomData:  id",
                    "type",
                    "cartes_x",
                    "cartes_y",
                    "cartes_z",
                    "fx",
                    "fy",
                    "fz",
                )
            )
            for i, (site, force) in enumerate(zip(structure, forces, strict=False)):
                lines.append(
                    format_float.format(
                        i + 1,
                        self.elements.index(str(site.specie)),
                        *site.coords,
                        *force,
                    )
                )
        if "Energy" in inputs:
            lines.append(" Energy")
            lines.append(f"{inputs['Energy']:>24.12f}")
        if "Stress" in inputs:
            # MLIP-3 reads (and writes) the same PlusStress block as MLIP-2.
            if not hasattr(self, "version") or self.version in ("mlip-2", "mlip-3"):
                format_str = "{:>16s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}"
                lines.append(format_str.format("PlusStress:  xx", "yy", "zz", "yz", "xz", "xy"))
            if self.version == "mlip-dev":
                format_str = "{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}"
                lines.append(format_str.format("Stress:  xx", "yy", "zz", "yz", "xz", "xy"))
            format_float = "{:>12f}{:>12f}{:>12f}{:>12f}{:>12f}{:>12f}"
            lines.append(format_float.format(*np.array(virial_stress)))

        lines.append("END_CFG")

        return "\n".join(lines)

    def write_cfg(self, filename, cfg_pool):
        """
        Write configurations to file
        Args:
            filename (str): filename
            cfg_pool (list): list of configurations.

        Returns: filename.

        """
        if not self.elements:
            raise ValueError("No species given.")

        lines = []
        for dataset in cfg_pool:
            if isinstance(dataset["structure"], dict):
                structure = Structure.from_dict(dataset["structure"])
            else:
                structure = dataset["structure"]
            energy = dataset["outputs"]["energy"]
            forces = dataset["outputs"]["forces"]
            virial_stress = dataset["outputs"]["virial_stress"]
            virial_stress = [virial_stress[self.vasp_stress_order.index(n)] for n in self.mtp_stress_order]
            lines.append(self._line_up(structure, energy, forces, virial_stress))

        with open(filename, "w") as f:
            f.write("\n".join(lines))

        return filename

    def write_ini(self, mtp_filename="fitted.mtp", **kwargs):
        """
        Write mlip.ini file for mlip packages of version mlip-2 or mlip-dev. Not supported for
        mlip-3, whose LAMMPS interface is a separate package.
        Supported keyword arguments are parallel with options stated in the mlip manuals.
        Please refer to https://mlip.skoltech.ru.

        Args:
            mtp_filename (str): Name of file with MTP to be loaded.
            **kwargs: Different kwargs for mlip-2 and mlip-dev.
            mlip-2:

                write_cfgs (str): Name of file for mlp processed configurations to be written to.
                write_cfgs_skip (int): Skipped number of processed configurations before writing.
                select (bool): activates or deactivates calculation of extrapolation grades and
                    optionally writing configurations with high extrapolation grades. False is
                    recommended for large-scale MD run.
                select_save_selected (str): Name of file for saving configurations with grade
                    exceeding select_threshold.
                select_threshold (float): Configurations with extrapolation grade exceeding the
                    value will be saved to the specified file.
                select_threshold_break (float): The mlip execution will be interrupted if the
                    extrapolation grade exceeds this value.
                select_load_state (str): Name of file for loading the active learning state,
                    typically created by the mlp calc-grade command.
                select_log (str): Name of file (or standard output stream stdout/stderr) for
                    writing a log of the configuration selection process.

            mlip-dev:
                Abinitio (int): Defines Ab-initio models. Default to 1.

                    0: If Ab-initio models is not required.
                    1: Used if driver provides EFS data with configurations.
                    2: Use embedded Lennard-Jones pair potentials.

                        r_min (float): Distance to minimum of pair function (in Angstroms).
                            Default to 2.0.
                        scale (float): Value of pair function in minimum (in eV).
                            Default to 1.0.
                        cutoff (float): Cutoff radius (in Angstroms). Default to 5.0.

                    3: Use DFT models by VASP. Linking via files exchange.

                        POSCAR (str): Relative path of POSCAR file.
                        OUTCAR (str): Relative path of OUTCAR file.
                        Start_command (str): Relative path of command file.

                    4: Use potentials calculating by LAMMPS. Linking via files exchange.

                        Input_file (str): File with configuration to be read by lammps.
                        Output_file (str): File with configuration and EFS data to be read by MLIP.
                        Start_command (str): Relative path of command file.

                    5: Use MTP as Ab-initio potentials.

                        MTP_filename (str): MTP file name.

                MLIP (str): MTP.

                    load_from (str): Potential filename.
                    Cacluate_EFS (bool): Whether to perform EFS calculation by MTP.
                    Fit (bool): Whether to perform MTP learning.

                        Save (str): Output MTP file name (for trained MTP).
                        Energy_equation_weight (float): Weight for energy equation in
                            fitting procedure. Default to 1.0.
                        Forces_equation_weight (float): Weight for forces equations in
                            fitting procedure. Default to 0.001.
                        Stress_equation_weight (float): Weight for stresses equations in
                            fitting procedure.  Default to 0.1.
                        Relative_forces_weight (float): If greater than zero, large forces
                            will be fitted less accurate than small. Default to 0.0.
                        Fit_log (str): File to write fitting log. No logging if not specified.
                            Default to None.

                    Select (bool): Whether to activate active learning. Default to False.

                        Site_E_weight (float): Weight for site energy equations in
                            selection procedure. Default to 1.0.
                        Energy_weight (float): Weight for energy equation in
                            selection procedure. Default to 0.0.
                        Forces_weight (float): Weight for forces equations in
                            selection procedure. Default to 0.0.
                        Stress_weight (float): Weight for stresses equations in
                            selection procedure. Default to 0.0.
                        Threshold_slct (float): Selection threshold - maximum
                            allowed extrapolation level. Default to 0.1.
                        Save_TS (str): Filename where selected configurations
                            will be saved. No configuration saving if not specified.
                            Default to None.
                        Save_state (str): Filename where state of the selection
                            will be saved. No saving if not specified. Default to None.
                        Load_state (str): Filename where state of the selection
                            will be loaded. No saving if not specified. Default to None.
                        Select_log (str): File to write fitting log. No logging
                            if not specified. Default to None.

                    LOFT (bool): Whether to perform learning on the fly. Default to False

                        EFSviaMTP (bool): Works only on LOFT regime. If True,
                            only MTP-calculated EFS will be passed to driver, else
                            pass to driver ab-initio EFS while LOTF when learning occurs.
                        Log (str): Filename to write log of learning on the fly process.
                            No logging if not specified. Default to None.

                    Check_errors (bool): If True, comparison and accumulation of
                        error statistics for EFS calculated by ab-initio models and MTP.
                        Default to False.

                        Log (str): Filename to write log of learning on the fly process.
                            No logging if not specified. Default to None.

                    Write_cfgs (bool): File for writing all processed configurations.
                        No configuration recording if not specified. Default to None.

                        Skip_N (int): The number of configurations to skip while writing.
                            Default to 0.

                    Log (str): Filename to write MLIP log. No logging if not specified.
                        Default to None.

                Driver (int): Defines the configuration driver. Default to 1.

                    0: No driver or external MD driver.
                    1: Read configurations from database file.

                        Database_filename (str): Configuration file name.
                        Max_count (int): Maximal number of configurations to read.
                        Log (str): Filename to write reading log. No logging
                            if not specified. Default to None.

                    2: Embedded algorithm for relaxation.

                        Pressure (float): External pressure (in GPa).
                            If not zero enthalpy is minimized. Default to 0.0.
                        Iteration_limit (int): Maximal number of iteration of
                            the relaxation process. Default to 500.
                        Min_dist (float): Minimal interatomic distance constraint
                            (in Angstroms). Default to 1.0.
                        Forces_tolerance (float): Forces on atoms in relaxed
                            configuration should be smaller than this value
                            (in eV/Angstroms). Default to 0.0001.
                        Stress_tolerance (float): Stresses in relaxed configuration
                            should be smaller than this value (in GPa). Default to 0.001.
                        Max_step (float): Maximal allowed displacement of atoms and
                            lattice vectors in Cartesian coordinates (in Angstroms).
                            Default to 0.5.
                        Min_step (float): Minimal displacement of atoms and
                            lattice vectors in Cartesian coordinates (in Angstroms).
                            Default to 1.0e-8.
                        BFGS_Wolfe_C1 (float): Wolfe condition constant on the function
                            decrease (linesearch stopping criteria). Default to 1.0e-3.
                        BFGS_Wolfe_C2 (float): Wolfe condition constant on the gradient
                            decrease (linesearch stopping criteria). Default to 0.7.
                        Save_relaxed (str): Filename for output results of relaxation.
                            No configuration will be saved if not specified.
                            Default to None.
                        Log (str): Filename to write relaxation log. No logging
                            if not specified. Default to None.
        """
        if getattr(self, "version", None) == "mlip-3":
            raise NotImplementedError(
                "mlip.ini/LAMMPS input is not supported for version='mlip-3': MLIP-3's LAMMPS "
                "interface (interface-lammps-mlip-3) is a separate package and has not been "
                "integrated into maml."
            )
        lines = []
        if not hasattr(self, "version") or self.version == "mlip-2":
            format_str = "{:<48s}{:<20s}"
            lines.append(format_str.format("mtp-filename", mtp_filename))
            if kwargs.get("write_cfgs"):
                lines.append(format_str.format("write-cfgs", str(kwargs.get("write_cfgs"))))
            if kwargs.get("write_cfgs_skip"):
                lines.append(format_str.format("write-cfgs:skip", str(kwargs.get("write_cfgs_skip"))))
            if not kwargs.get("select"):
                lines.append(format_str.format("select", "FALSE"))
            else:
                lines.append(format_str.format("select", "TRUE"))
                select_identifiers = [
                    "select:save-selected",
                    "select:threshold",
                    "select:threshold-break",
                    "select:load-state",
                    "select:log",
                ]
                for i, option in enumerate(
                    [
                        "select_save_selected",
                        "select_threshold",
                        "select_threshold_break",
                        "select_load_state",
                        "select_log",
                    ]
                ):
                    if kwargs.get(option):
                        lines.append(format_str.format("\t" + select_identifiers[i], str(kwargs.get(option))))

        elif self.version == "mlip-dev":
            format_str = "{:<48s}{:<20s}{}"
            Abinitio = kwargs.get("Abinitio") if kwargs.get("Abinitio") else 0
            MLIP = kwargs.get("MLIP") if kwargs.get("MLIP") else "MPT.mpt"
            Driver = kwargs.get("Driver") if kwargs.get("Driver") else 0
            PARAMS = {
                "Abinitio": {
                    0: [],
                    1: [],
                    2: ["r_min", "scale", "cutoff"],
                    3: ["POSCAR", "OUTCAR", "Start_command"],
                    4: ["Input_file", "Output_file", "Start_command"],
                    5: ["MTP_filename"],
                },
                "MLIP": {
                    "Calculate_EFS": [],
                    "Fit": [
                        "Save",
                        "Energy_equation_weight",
                        "Forces_equation_weight",
                        "Stress_equation_weight",
                        "Relative_forces_weight",
                        "Fit_log",
                    ],
                    "Select": [
                        "Site_E_weight",
                        "Energy_weight",
                        "Forces_weight",
                        "Stress_weight",
                        "Threshold_slct",
                        "Save_TS",
                        "Save_state",
                        "Load_state",
                        "Select_log",
                    ],
                    "Write_cfgs": [],
                    "Log": [],
                },
                "Driver": {
                    0: [],
                    1: ["Database_filename", "Database_log"],
                    2: [
                        "Pressure",
                        "Iteration_limit",
                        "Min_dist",
                        "Forces_tolerance",
                        "Stress_tolerance",
                        "Max_step",
                        "Min_step",
                        "BFGS_Wolfe_C1",
                        "BFGS_Wolfe_C2",
                        "Save_relaxed",
                        "Relaxation_log",
                    ],
                },
            }

            if Abinitio:
                lines.append(
                    format_str.format(
                        MTini_params.get("Abinitio").get("name"),
                        str(Abinitio),
                        MTini_params.get("Abinitio").get("comment"),
                    )
                )
                abinitio = MTini_params.get("Abinitio").get(str(Abinitio))
                lines.append(format_str.format(abinitio.get("name"), "", abinitio.get("comment")))
                for attribute in PARAMS["Abinitio"][Abinitio]:
                    lines.append(format_str.format(*feed(attribute, kwargs, abinitio)))

            if MLIP:
                lines.append(
                    format_str.format(
                        MTini_params.get("MLIP").get("name"),
                        "mtpr",
                        MTini_params.get("MLIP").get("comment"),
                    )
                )
                mlip = MTini_params.get("MLIP")
                if kwargs.get("load_from"):
                    load_from = mlip.get("load_from")
                    lines.append(
                        format_str.format(
                            "\t" + load_from.get("name"),
                            kwargs.get("load_from"),
                            load_from.get("comment"),
                        )
                    )
                if kwargs.get("Calculate_EFS"):
                    calc_efs = mlip.get("Calculate_EFS")
                    lines.append(format_str.format("\t" + calc_efs.get("name"), "TRUE", calc_efs.get("comment")))
                if kwargs.get("Fit"):
                    fit = mlip.get("Fit")
                    lines.append(format_str.format("\t" + fit.get("name"), "true", fit.get("comment")))
                    for attribute in PARAMS["MLIP"]["Fit"]:
                        lines.append(format_str.format(*feed(attribute, kwargs, fit, tab="\t\t")))

                if kwargs.get("Select"):
                    select = mlip.get("Select")
                    lines.append(format_str.format("\t" + select.get("name"), "true", select.get("comment")))
                    for attribute in PARAMS["MLIP"]["Select"]:
                        lines.append(format_str.format(*feed(attribute, kwargs, select, tab="\t\t")))

                if kwargs.get("Write_cfgs"):
                    write_cfgs = mlip.get("Write_cfgs")
                    lines.append(
                        format_str.format(
                            "\t" + write_cfgs.get("name"),
                            kwargs.get("Write_cfgs"),
                            write_cfgs.get("comment"),
                        )
                    )

            if Driver:
                lines.append(
                    format_str.format(
                        MTini_params.get("Driver").get("name"),
                        str(Driver),
                        MTini_params.get("Driver").get("comment"),
                    )
                )
                driver = MTini_params.get("Driver").get(str(Driver))
                lines.append(format_str.format(driver.get("name"), "", driver.get("comment")))
                for attribute in PARAMS["Driver"][Driver]:
                    lines.append(format_str.format(*feed(attribute, kwargs, driver, tab="\t\t")))

        filename = "mlip.ini"
        with open(filename, "w") as f:
            f.write("\n".join(lines))

        return filename

    def read_cfgs(self, filename):
        """

        Args:
            filename (str): The configuration file to be read.

        """

        def formatify(string):
            return [float(s) for s in string.split()]

        if not self.elements:
            raise ValueError("No species given.")

        data_pool = []
        with zopen(filename, "rt") as f:
            lines = f.read()

        block_pattern = re.compile("BEGIN_CFG\n(.*?)\nEND_CFG", re.S)
        size_pattern = re.compile("Size\n(.*?)\n SuperCell", re.S | re.I)
        lattice_pattern = re.compile("SuperCell\n(.*?)\n AtomData", re.S | re.I)
        position_pattern = re.compile("fz\n(.*?)\n Energy", re.S)
        energy_pattern = re.compile("Energy\n(.*?)\n (?=PlusStress|Stress)", re.S)
        stress_pattern = re.compile("xy\n(.*?)(?=\n|$)", re.S)
        for block in block_pattern.findall(lines):
            d = {"outputs": {}}
            size_str = size_pattern.findall(block)[0]
            size = int(size_str.lstrip())
            lattice_str = lattice_pattern.findall(block)[0]
            lattice = Lattice(np.array(list(map(formatify, lattice_str.split("\n")))))
            position_str = position_pattern.findall(block)[0]
            position = np.array(list(map(formatify, position_str.split("\n"))))
            species = np.array(self.elements)[position[:, 1].astype(np.int64)]
            forces = position[:, 5:8].tolist()
            position = position[:, 2:5]
            energy_str = energy_pattern.findall(block)[0]
            energy = float(energy_str.lstrip())
            stress_str = stress_pattern.findall(block)[0]
            virial_stress = (
                np.array(list(map(formatify, stress_str.split())))
                .reshape(
                    6,
                )
                .tolist()
            )
            virial_stress = [virial_stress[self.mtp_stress_order.index(n)] for n in self.vasp_stress_order]
            struct = Structure(
                lattice=lattice,
                species=species,
                coords=position,
                coords_are_cartesian=True,
            )
            d["structure"] = struct.as_dict()
            d["outputs"]["energy"] = energy
            assert size == struct.num_sites
            d["num_atoms"] = size
            d["outputs"]["forces"] = forces
            d["outputs"]["virial_stress"] = virial_stress

            data_pool.append(d)
        _, df = convert_docs(docs=data_pool, include_stress=True)
        return data_pool, df

    def train(
        self,
        train_structures,
        train_energies,
        train_forces,
        train_stresses,
        unfitted_mtp="08g.mtp",
        max_dist=5,
        radial_basis_size=8,
        max_iter=1000,
        energy_weight=1,
        force_weight=1e-2,
        stress_weight=1e-3,
        init_params="same",
        scale_by_force=0,
        bfgs_conv_tol=1e-3,
        weighting="vibration",
    ):
        """
        Training data with moment tensor method.

        Args:
            train_structures ([Structure]): The list of Pymatgen Structure object.
                energies ([float]): The list of total energies of each structure
                in structures list.
            train_energies ([float]): List of total energies of each structure in
                structures list.
            train_forces ([np.array]): List of (m, 3) forces array of each structure
                with m atoms in structures list. m can be varied with each single
                structure case.
            train_stresses (list): List of (6, ) virial stresses of each structure
                in structures list.
            unfitted_mtp (str): Define the initial mtp file. Default to the mtp file
                stored in .params directory.
            max_dist (float): The actual radial cutoff.
            radial_basis_size (int): Relevant to number of radial basis function.
            max_iter (int): The number of maximum iteration.
                (MLIP-3: --iteration_limit.)
            energy_weight (float): The weight of energy.
            force_weight (float): The weight of forces.
            stress_weight (float): The weight of stresses. Zero-weight can be assigned.
            init_params (str): How to initialize parameters if a potential was not
                pre-fitted. Choose from "same" and "random".
                (MLIP-3: --init_random=false for "same", --init_random=true for "random".)
            scale_by_force (float): Default=0. If >0 then configurations near equilibrium
               (with roughly force < scale_by_force) get more weight.
            bfgs_conv_tol (float): Stop training if error dropped by a factor smaller than this
                over 50 BFGS iterations. (MLIP-3: --tolerance.)
            weighting (str): How to weight configuration with different sizes relative to each other.
                Choose from "vibrations", "molecules" and "structures". (MLIP-3:
                --weight_scaling/--weight_scaling_forces = 1/0, 0/0 and 2/1 respectively.)

        With version="mlip-3" the remaining options map to MLIP-3's underscored flags
        (--energy_weight, --force_weight, --stress_weight, --scale_by_force), the fitted potential
        is written with --save_to, and MLIP-2's --curr-pot-name has no MLIP-3 equivalent. MLIP-3
        also resets the potential's min_dist to 0.99 times the minimum interatomic distance of the
        training set.

        Raises:
            RuntimeError: If mlp is not found, `mlp train` fails, or the fit diverges (the fitted
                potential contains NaN/inf values or cannot be parsed).
        """
        if not which("mlp"):
            raise RuntimeError("mlp has not been found.\nPlease refer to https://mlip.skoltech.ru for further detail.")
        (
            train_structures,
            train_forces,
            train_stresses,
        ) = check_structures_forces_stresses(train_structures, train_forces, train_stresses)
        train_pool = pool_from(train_structures, train_energies, train_forces, train_stresses)
        elements = sorted(set(itertools.chain(*[struct.species for struct in train_structures])))
        self.elements = [str(element) for element in elements]

        atoms_filename = "train.cfgs"

        with ScratchDir("."):
            atoms_filename = self.write_cfg(filename=atoms_filename, cfg_pool=train_pool)

            if not unfitted_mtp:
                raise RuntimeError("No specific parameter file provided.")
            MTP_file_path = os.path.join(module_dir, "params", unfitted_mtp)
            shutil.copyfile(MTP_file_path, os.path.join(os.getcwd(), unfitted_mtp))

            # Computed in Python rather than with `mlp mindist`, which does not exist in every MLIP
            # version (e.g. MLIP-3) and whose output format is not stable across versions.
            min_dist = min_interatomic_distance(train_structures)

            with open(unfitted_mtp) as f:
                template = f.read()

            s = template % (len(self.elements), min_dist, max_dist, radial_basis_size)
            with open(unfitted_mtp, "w") as f:
                f.write(s)

            save_fitted_mtp = ".".join([unfitted_mtp.split(".")[0] + "_fitted", unfitted_mtp.split(".")[1]])

            if self.version == "mlip-3":
                if init_params not in ("same", "random"):
                    raise ValueError(f"init_params must be 'same' or 'random', got {init_params!r}.")
                if weighting not in MLIP3_WEIGHT_SCALING:
                    raise ValueError(f"weighting must be one of {sorted(MLIP3_WEIGHT_SCALING)}, got {weighting!r}.")
                weight_scaling, weight_scaling_forces = MLIP3_WEIGHT_SCALING[weighting]
                cmd = [
                    "mlp",
                    "train",
                    unfitted_mtp,
                    atoms_filename,
                    f"--save_to={save_fitted_mtp}",
                    f"--iteration_limit={max_iter}",
                    f"--energy_weight={energy_weight}",
                    f"--force_weight={force_weight}",
                    f"--stress_weight={stress_weight}",
                    f"--init_random={'true' if init_params == 'random' else 'false'}",
                    f"--scale_by_force={scale_by_force}",
                    f"--tolerance={bfgs_conv_tol}",
                    f"--weight_scaling={weight_scaling}",
                    f"--weight_scaling_forces={weight_scaling_forces}",
                ]
            else:
                cmd = [
                    "mlp",
                    "train",
                    unfitted_mtp,
                    atoms_filename,
                    f"--max-iter={max_iter}",
                    f"--trained-pot-name={save_fitted_mtp}",
                    f"--curr-pot-name={unfitted_mtp}",
                    f"--energy-weight={energy_weight}",
                    f"--force-weight={force_weight}",
                    f"--stress-weight={stress_weight}",
                    f"--init-params={init_params}",
                    f"--scale-by-force={scale_by_force}",
                    f"--bfgs-conv-tol={bfgs_conv_tol}",
                    f"--weighting={weighting}",
                ]

            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as p:
                stdout, stderr = p.communicate()
                rc = p.returncode
            if rc != 0:
                raise RuntimeError(
                    f"mlp train exited with return code {rc}. {SUPPORTED_VERSIONS_MSG}\n"
                    + _format_mlp_output(stdout, stderr)
                )

            if not os.path.exists(save_fitted_mtp):
                raise RuntimeError(
                    f"mlp train finished but did not write the fitted potential {save_fitted_mtp}. "
                    f"{SUPPORTED_VERSIONS_MSG}\n" + _format_mlp_output(stdout, stderr)
                )
            try:
                self.param = _load_mtp_param(save_fitted_mtp)
            except ValueError as exc:
                raise RuntimeError(
                    f"MTP training diverged: {exc} This usually means the fit diverged (e.g. the loss "
                    "became NaN), not that the input data file is malformed. Check the training data "
                    "for inconsistent energies/forces, and try other weights or a different "
                    "max_dist/radial_basis_size.\n" + _format_mlp_output(stdout, stderr)
                ) from exc
        return rc

    def write_param(self, fitted_mtp="fitted.mtp", **kwargs):
        """
        Write fitted mtp parameter file to perform lammps calculation.

        Args:
            fitted_mtp (str): Filename to store xml formatted parameters.
            **kwargs: pass to write_ini method.

        Raises:
            NotImplementedError: For version="mlip-3", whose LAMMPS interface is not supported.
        """
        if getattr(self, "version", None) == "mlip-3":
            raise NotImplementedError(
                "LAMMPS calculations are not supported for version='mlip-3': MLIP-3's LAMMPS "
                "interface (interface-lammps-mlip-3) is a separate package and has not been "
                "integrated into maml. train() and evaluate() work with mlip-3."
            )
        self._write_mtp(fitted_mtp)
        ini_file = self.write_ini(mtp_filename=fitted_mtp, load_from=fitted_mtp, Calculate_EFS=True, **kwargs)
        return [self.pair_style.format(ini_file), self.pair_coeff]

    def _write_mtp(self, fitted_mtp):
        """Write the fitted parameters to an .mtp file."""
        if not self.param:
            raise RuntimeError("The parameters should be provided.")
        lines = [
            (
                " = ".join([key, json.dumps(value).replace("[", "{").replace("]", "}")])
                if key != "safe"
                else "\n".join(value)
            )
            for key, value in self.param.items()
        ]
        with open(fitted_mtp, "w") as f:
            f.write("\n".join(lines))

    def evaluate(self, test_structures, test_energies, test_forces, test_stresses=None, **kwargs):
        """
        Evaluate energies, forces and stresses of structures with trained
        interatomic potentials.

        Args:
            test_structures ([Structure]): List of Pymatgen Structure Objects.
            test_energies ([float]): List of DFT-calculated total energies of
                each structure in structures list.
            test_forces ([np.array]): List of DFT-calculated (m, 3) forces of
                each structure with m atoms in structures list. m can be varied
                with each single structure case.
            test_stresses (list): List of DFT-calculated (6, ) viriral stresses
                of each structure in structures list.
            kwargs: Parameters of write_param method. Ignored for version="mlip-3", which
                runs `mlp calculate_efs` on the potential file without an mlip.ini.
        """
        if not which("mlp"):
            raise RuntimeError("mlp has not been found.\nPlease refer to https://mlip.skoltech.ru for further detail.")
        fitted_mtp = "fitted.mtp"
        original_file = "original.cfgs"
        predict_file = "predict.cfgs"
        test_structures, test_forces, test_stresses = check_structures_forces_stresses(
            test_structures, test_forces, test_stresses
        )
        predict_pool = pool_from(test_structures, test_energies, test_forces, test_stresses)

        with ScratchDir("."):
            if self.version == "mlip-3":
                self._write_mtp(fitted_mtp)
            else:
                self.write_param(
                    fitted_mtp=fitted_mtp,
                    Abinitio=0,
                    Driver=1,
                    Write_cfgs=predict_file,
                    Database_filename=original_file,
                    **kwargs,
                )
            original_file = self.write_cfg(original_file, cfg_pool=predict_pool)
            _, df_orig = self.read_cfgs(original_file)

            if not hasattr(self, "version") or self.version == "mlip-2":
                cmd = ["mlp", "calc-efs", fitted_mtp, original_file, predict_file]
            elif self.version == "mlip-3":
                cmd = ["mlp", "calculate_efs", fitted_mtp, original_file, f"--output_filename={predict_file}"]
            elif self.version == "mlip-dev":
                cmd = ["mlp", "run", "mlip.ini", f"--filename={original_file}"]
            with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
                stdout = p.communicate()[0]
                rc = p.returncode
            _check_returncode("mlp", rc, stdout)
            if not os.path.exists(predict_file):
                predict_file = f"{predict_file}_0"
            if not os.path.exists(predict_file):
                # mlp exits with 0 even for unknown commands, e.g. MLIP-2's calc-efs under MLIP-3.
                raise RuntimeError(
                    f"{cmd[0]} {cmd[1]} did not write the predicted configurations. {SUPPORTED_VERSIONS_MSG}\n"
                    + _format_mlp_output(stdout, None)
                )
            _, df_predict = self.read_cfgs(predict_file)
        return df_orig, df_predict

    @staticmethod
    def from_config(filename, elements, version=None):
        """
        Initialize potentials with parameters file.

        Args:
            filename (str): The file storing parameters of potentials, filename should
                ends with ".mtp". Potentials trained by MLIP-2 or MLIP-3 can be read.
            elements (list): The list of elements.
            version (str): The version of mlip package, see MTPotential. Default is "mlip-3".

        Returns:
            MTPotential

        Raises:
            ValueError: If the file contains NaN/inf values or cannot be parsed.
        """
        mtp = MTPotential(param=_load_mtp_param(filename), version=version)
        mtp.elements = elements

        return mtp
