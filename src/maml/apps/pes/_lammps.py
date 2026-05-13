# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

"""This module provides basic LAMMPS calculator classes."""

from __future__ import annotations

import abc
import io
import itertools
import logging
import os
import subprocess
from shutil import which

import numpy as np
from monty.tempfile import ScratchDir
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.lammps.data import LammpsData

from maml.apps.pes._base import Potential
from maml.utils import get_lammps_lattice_and_rotation, stress_list_to_matrix, stress_matrix_to_list

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_default_lmp_exe():
    """
    Get lammps executable
    Returns: Lammps executable name.
    """
    for lmp_exe in ["lmp_serial", "lmp_mpi", "lmp_g++_serial", "lmp_g++_mpich", "lmp_intel_cpu_intelmpi", "lmp"]:
        lmp_exe_detected = which(lmp_exe)
        if lmp_exe_detected is not None:
            logger.info(f"Setting Lammps executable to {lmp_exe_detected}")
            return lmp_exe_detected
    return None


def _pretty_input(lines):
    def prettify(line):
        return (
            line.split()[0].ljust(width) + " ".join(line.split()[1:])
            if not (len(line.split()) == 0 or line.strip().startswith("#"))
            else line
        )

    clean_lines = [line.strip("\n") for line in lines]
    commands = [line for line in clean_lines if len(line.strip()) > 0]
    keys = [c.split()[0] for c in commands if not c.split()[0].startswith("#")]
    width = max(len(k) for k in keys) + 4
    new_lines = map(prettify, clean_lines)
    return "\n".join(new_lines)


def _read_dump(file_name, dtype="float_"):
    with open(file_name) as f:
        lines = f.readlines()[9:]
    return np.loadtxt(io.StringIO("".join(lines)), dtype=dtype)


class LMPStaticCalculator:
    """
    Abstract class to perform static structure property calculation
    using LAMMPS.
    """

    _COMMON_CMDS = ["units metal", "atom_style charge", "box tilt large", "read_data data.static", "run 0"]

    allowed_kwargs = ["lmp_exe"]

    def __init__(self, **kwargs):
        """
        Initializer for lammps calculator
        Allowed keyword args are lmp_exe string.

        """
        lmp_exe = kwargs.pop("lmp_exe", None)
        if lmp_exe is None:
            lmp_exe = get_default_lmp_exe()
        if not which(lmp_exe):
            raise ValueError(f"lammps executable {lmp_exe} not found")
        self.LMP_EXE = lmp_exe
        for i, j in kwargs.items():
            if i not in self.allowed_kwargs:
                raise TypeError(f"{i!s} not in supported kwargs {self.allowed_kwargs!s}")
            setattr(self, i, j)

    @abc.abstractmethod
    def _setup(self):
        """Setup a calculation, writing input files, etc."""
        return

    @abc.abstractmethod
    def _sanity_check(self, structure):
        """Check if the structure is valid for this calculation."""
        return

    @abc.abstractmethod
    def _parse(self):
        """Parse results from dump files."""
        return

    def calculate(self, structures):
        """
        Perform the calculation on a series of structures.

        Args:
            structures: Input structures in a list.

        Returns:
            List of computed data corresponding to each structure,
            varies with different subclasses.

        """
        for struct in structures:
            assert self._sanity_check(struct) is True, "Incompatible structure found"
        ff_elements = None
        if hasattr(self, "element_profile"):
            element_profile = self.element_profile
            ff_elements = element_profile.keys()
        if hasattr(self, "ff_settings"):
            ff_settings = self.ff_settings
            if hasattr(ff_settings, "elements"):
                ff_elements = ff_settings.elements

        with ScratchDir("."):
            input_file = self._setup()
            data = []
            for struct in structures:
                struct.remove_oxidation_states()
                ld = LammpsData.from_structure(struct, ff_elements, atom_style="charge")
                ld.write_file("data.static")
                with subprocess.Popen([self.LMP_EXE, "-in", input_file], stdout=subprocess.PIPE) as p:
                    stdout = p.communicate()[0]
                    rc = p.returncode
                if rc != 0:
                    error_msg = f"LAMMPS exited with return code {rc}"
                    msg = stdout.decode("utf-8").split("\n")[:-1]
                    try:
                        error_line = next(i for i, m in enumerate(msg) if m.startswith("ERROR"))
                        error_msg += ", ".join(msg[error_line:])
                    except Exception:
                        error_msg += msg[-1]
                    raise RuntimeError(error_msg)
                results = self._parse()
                data.append(results)
        return data

    def set_lmp_exe(self, lmp_exe: str) -> None:
        """
        Set lammps executable for the instance.

        Args:
            lmp_exe (str): lammps executable path
        Returns:
        """
        self.LMP_EXE = lmp_exe


class EnergyForceStress(LMPStaticCalculator):
    """Calculate energy, forces and virial stress of structures."""

    def __init__(self, ff_settings, **kwargs):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for LAMMPS
                calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
            **kwargs: Passthrough to init.
        """
        self.ff_settings = ff_settings
        super().__init__(**kwargs)

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), "templates", "efs")
        with open(os.path.join(template_dir, "in.efs")) as f:
            input_template = f.read()

        input_file = "in.efs"

        ff_settings = self.ff_settings.write_param() if isinstance(self.ff_settings, Potential) else self.ff_settings

        with open(input_file, "w") as f:
            f.write(input_template.format(ff_settings="\n".join(ff_settings)))
        return input_file

    def _sanity_check(self, structure):
        return True

    def calculate(self, structures):
        """
        Calculate the energy, forces and stresses of structures.
        Proper rotation of the results are applied when the structure
        is triclinic.

        Args:
            structures (list): a list of structures

        Returns: list of (energy, forces, stresses) tuple

        """
        results = super().calculate(structures=structures)
        final_results = []
        for res, s in zip(results, structures, strict=False):
            new_forces, new_stresses = self._rotate_force_stress(s, res[1], res[2])
            final_results.append((res[0], new_forces, new_stresses))
        return final_results

    @staticmethod
    def _rotate_force_stress(structure, forces, stresses):
        _, _symmop, rot_matrix = get_lammps_lattice_and_rotation(structure)
        inv_rot_matrix = np.linalg.inv(rot_matrix)
        forces = forces.dot(inv_rot_matrix.T)
        stresses = stress_list_to_matrix(stresses, stress_format="LAMMPS")
        stresses = inv_rot_matrix.dot(stresses.dot(inv_rot_matrix.T))
        stresses = stress_matrix_to_list(stresses, stress_format="LAMMPS")
        return forces, stresses

    def _parse(self):
        energy = float(np.loadtxt("energy.txt"))
        force = _read_dump("force.dump")
        stress = np.loadtxt("stress.txt")
        return energy, force, stress


class SpectralNeighborAnalysis(LMPStaticCalculator):
    """
    Calculator for bispectrum components to characterize the local
    neighborhood of each atom in a general way.

    Usage:
        [(b, db, e)] = sna.calculate([Structure])
        b: 2d NumPy array with shape (N, n_bs) containing bispectrum
            coefficients, where N is the No. of atoms in structure and
            n_bs is the No. of bispectrum components.
        db: 2d NumPy array with shape (N, 3 * n_bs * n_elements)
            containing the first order derivatives of bispectrum
            coefficients with respect to atomic coordinates,
            where n_elements is the No. of elements in element_profile.
        e: 2d NumPy array with shape (N, 1) containing the element of
            each atom.

    """

    _CMDS = [
        "pair_style lj/cut 10",
        "pair_coeff * * 1 1",
        "compute sna all sna/atom ",
        "compute snad all snad/atom ",
        "compute snav all snav/atom ",
        "dump 1 all custom 1 dump.element element",
        "dump_modify 1 sort id",
        "dump 2 all custom 1 dump.sna c_sna[*]",
        "dump_modify 2 sort id",
        "dump 3 all custom 1 dump.snad c_snad[*]",
        "dump_modify 3 sort id",
        "dump 4 all custom 1 dump.snav c_snav[*]",
        "dump_modify 4 sort id",
    ]

    def __init__(self, rcutfac, twojmax, element_profile, quadratic=False, **kwargs):
        """
        For more details on the parameters, please refer to the
        official documentation of LAMMPS.

        Notes:
            Despite this calculator uses compute sna(d)/atom command
            (http://lammps.sandia.gov/doc/compute_sna_atom.html), the
            parameter definition is in consistent with pair_style snap
            document (http://lammps.sandia.gov/doc/pair_snap.html).

        Args:
            rcutfac (float): The rcutfac in bispectrum coefficient calculations.
                The cutoff radius between element i and j are rcutfac * (R_i + R_j)
                where R_i and R_j are cutoff set for element i and j.
            twojmax (int): Band limit for bispectrum components.
            element_profile (dict): Parameters (cutoff radius 'r' and
                weight 'w') related to each element, e.g.,
                {'Na': {'r': 4.5, 'w': 0.9},
                 'Cl': {'r': 4.8, 'w': 3.0}}
            quadratic (bool): Whether including quadratic terms.
                Default to False.
            **kwargs: Passthrough to init.
        """
        self.rcutfac = rcutfac
        self.twojmax = twojmax
        self.element_profile = element_profile
        self.quadratic = quadratic
        super().__init__(**kwargs)

    @staticmethod
    def get_bs_subscripts(twojmax):
        """
        Method to list the subscripts 2j1, 2j2, 2j of bispectrum
        components.

        Args:
            twojmax (int): Band limit for bispectrum components.

        Returns:
            List of all subscripts [2j1, 2j2, 2j].

        """
        subs = itertools.product(range(twojmax + 1), repeat=3)

        filters = [lambda x: x[0] >= x[1], lambda x: x[2] >= x[0]]
        j_filter = [lambda x: x[2] in range(x[0] - x[1], min(twojmax, x[0] + x[1]) + 1, 2)]
        filters.extend(j_filter)
        for f in filters:
            subs = filter(f, subs)
        return list(subs)

    @property
    def n_bs(self):
        """Returns No. of bispectrum components to be calculated."""
        return len(self.get_bs_subscripts(self.twojmax))

    def _setup(self):
        def add_args(line):
            return line + compute_args if line.startswith("compute") else line

        compute_args = f"1 0.99363 {self.twojmax} "
        el_in_seq = sorted(self.element_profile.keys(), key=Element)
        cutoffs = [self.element_profile[e]["r"] * self.rcutfac for e in el_in_seq]
        weights = [self.element_profile[e]["w"] for e in el_in_seq]
        compute_args += " ".join([str(p) for p in cutoffs + weights])
        compute_args += f" rmin0 0 quadraticflag {int(self.quadratic)}"
        CMDS = list(map(add_args, self._CMDS))
        CMDS[2] += " bzeroflag 0"
        CMDS[3] += " bzeroflag 0"
        CMDS[4] += " bzeroflag 0"
        dump_modify = "dump_modify 1 element "
        dump_modify += " ".join(str(e) for e in el_in_seq)
        CMDS.append(dump_modify)
        ALL_CMDS = self._COMMON_CMDS[:]
        ALL_CMDS[-1:-1] = CMDS
        input_file = "in.sna"
        with open(input_file, "w") as f:
            f.write(_pretty_input(ALL_CMDS))
        return input_file

    def _sanity_check(self, structure):
        elements = set(structure.symbol_set)
        sna_elements = self.element_profile.keys()
        return elements.issubset(sna_elements)

    def _parse(self):
        element = np.atleast_1d(_read_dump("dump.element", "unicode"))
        b = np.atleast_2d(_read_dump("dump.sna"))
        db = np.atleast_2d(_read_dump("dump.snad"))
        vb = np.atleast_2d(_read_dump("dump.snav"))
        return b, db, vb, element


class ElasticConstant(LMPStaticCalculator):
    """Elastic constant calculator."""

    _RESTART_CONFIG = {
        "internal": {"write_command": "write_restart", "read_command": "read_restart", "restart_file": "restart.equil"},
        "external": {"write_command": "write_restart", "read_command": "read_restart", "restart_file": "restart.equil"},
    }

    def __init__(
        self,
        ff_settings,
        potential_type="external",
        deformation_size=1e-6,
        jiggle=1e-5,
        maxiter=400,
        maxeval=1000,
        full_matrix=False,
        **kwargs,
    ):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for LAMMPS
                calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
            potential_type (str): 'internal' indicates the internal potentials
                installed in lammps, 'external' indicates the external potentials
                outside of lammps.
            deformation_size (float): Finite deformation size. Usually range from
                1e-2 to 1e-8, to confirm the results not depend on it.
            jiggle (float): The amount of random jiggle for atoms to
                prevent atoms from staying on saddle points.
            maxiter (float): The maximum number of iteration. Default to 400.
            maxeval (float): The maximum number of evaluation. Default to 1000.
            full_matrix (bool): If False, only c11, c12, c44 and bulk modulus are returned.
                If True, 6 x 6 elastic matrices in the Voigt notation are returned.
            **kwargs: Passthrough to init.
        """
        self.ff_settings = ff_settings
        self.write_command = self._RESTART_CONFIG[potential_type]["write_command"]
        self.read_command = self._RESTART_CONFIG[potential_type]["read_command"]
        self.restart_file = self._RESTART_CONFIG[potential_type]["restart_file"]
        self.deformation_size = deformation_size
        self.jiggle = jiggle
        self.maxiter = maxiter
        self.maxeval = maxeval
        self.full_matrix = full_matrix
        super().__init__(**kwargs)

    def _setup(
        self,
    ):
        template_dir = os.path.join(os.path.dirname(__file__), "templates", "elastic")

        with open(os.path.join(template_dir, "in.elastic")) as f:
            input_template = f.read()
        with open(os.path.join(template_dir, "init.template")) as f:
            init_template = f.read()
        with open(os.path.join(template_dir, "potential.template")) as f:
            potential_template = f.read()
        with open(os.path.join(template_dir, "displace.template")) as f:
            displace_template = f.read()

        input_file = "in.elastic"
        ff_settings = self.ff_settings.write_param() if isinstance(self.ff_settings, Potential) else self.ff_settings

        with open(input_file, "w") as f:
            f.write(input_template.format(write_restart=self.write_command, restart_file=self.restart_file))

        with open("init.mod", "w") as f:
            f.write(
                init_template.format(
                    deformation_size=self.deformation_size,
                    jiggle=self.jiggle,
                    maxiter=self.maxiter,
                    maxeval=self.maxeval,
                )
            )
        with open("potential.mod", "w") as f:
            f.write(potential_template.format(ff_settings="\n".join(ff_settings)))
        with open("displace.mod", "w") as f:
            f.write(displace_template.format(read_restart=self.read_command, restart_file=self.restart_file))
        return input_file

    def _sanity_check(self, structure):
        """Check if the structure is valid for this calculation."""
        return True

    def _parse(self):
        """Parse results from dump files."""
        if self.full_matrix:
            return np.loadtxt("voigt_tensor.txt")

        C11, C12, C44, bulkmodulus = np.loadtxt("elastic.txt")
        return C11, C12, C44, bulkmodulus


class NudgedElasticBand(LMPStaticCalculator):
    """NudgedElasticBand migration energy calculator."""

    def __init__(self, ff_settings, specie, lattice, alat, num_replicas=7, **kwargs):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for
                LAMMPS calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
            specie (str): Name of specie.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
            num_replicas (int): Number of replicas to use.
            **kwargs: Passthrough to init.
        """
        self.ff_settings = ff_settings
        self.specie = specie
        self.lattice = lattice
        self.alat = alat
        self.num_replicas = num_replicas
        super().__init__(**kwargs)

    @staticmethod
    def get_unit_cell(specie, lattice, alat):
        """
        Get the unit cell from specie, lattice type and lattice constant.

        Args:
            specie (str): Name of specie.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
        """
        if lattice == "fcc":
            unit_cell = Structure.from_spacegroup(
                sg="Fm-3m", lattice=Lattice.cubic(alat), species=[specie], coords=[[0, 0, 0]]
            )
        elif lattice == "bcc":
            unit_cell = Structure.from_spacegroup(
                sg="Im-3m", lattice=Lattice.cubic(alat), species=[specie], coords=[[0, 0, 0]]
            )
        elif lattice == "diamond":
            unit_cell = Structure.from_spacegroup(
                sg="Fd-3m", lattice=Lattice.cubic(alat), species=[specie], coords=[[0, 0, 0]]
            )
        else:
            raise ValueError("Lattice type is invalid.")

        return unit_cell

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), "templates", "neb")

        with open(os.path.join(template_dir, "in.relax")) as f:
            relax_template = f.read()
        with open(os.path.join(template_dir, "in.neb")) as f:
            neb_template = f.read()

        unit_cell = self.get_unit_cell(specie=self.specie, lattice=self.lattice, alat=self.alat)
        lattice_calculator = LatticeConstant(ff_settings=self.ff_settings)
        a, _, _ = lattice_calculator.calculate([unit_cell])[0]
        unit_cell = self.get_unit_cell(specie=self.specie, lattice=self.lattice, alat=a)

        if self.lattice == "fcc":
            start_idx, final_idx = 95, 49
            scale_factor = [3, 3, 3]
        elif self.lattice == "bcc":
            start_idx, final_idx = 40, 14
            scale_factor = [3, 3, 3]
        elif self.lattice == "diamond":
            start_idx, final_idx = 7, 15
            scale_factor = [2, 2, 2]
        else:
            raise ValueError("Lattice type is invalid.")

        super_cell = unit_cell * scale_factor
        super_cell_ld = LammpsData.from_structure(
            super_cell, ff_elements=self.ff_settings.elements, atom_style="charge"
        )
        super_cell_ld.write_file("data.supercell")

        with open("in.relax", "w") as f:
            f.write(
                relax_template.format(
                    ff_settings="\n".join(self.ff_settings.write_param()),
                    lattice=self.lattice,
                    alat=a,
                    specie=self.specie,
                    del_id=start_idx + 1,
                    relaxed_file="initial.relaxed",
                )
            )

        with subprocess.Popen([self.LMP_EXE, "-in", "in.relax"], stdout=subprocess.PIPE) as p:
            stdout = p.communicate()[0]
            rc = p.returncode
        if rc != 0:
            error_msg = f"LAMMPS exited with return code {rc}"
            msg = stdout.decode("utf-8").split("\n")[:-1]
            try:
                error_line = next(i for i, m in enumerate(msg) if m.startswith("ERROR"))
                error_msg += ", ".join(msg[error_line:])
            except Exception:
                error_msg += msg[-1]
            raise RuntimeError(error_msg)

        with open("in.relax", "w") as f:
            f.write(
                relax_template.format(
                    ff_settings="\n".join(self.ff_settings.write_param()),
                    lattice=self.lattice,
                    alat=a,
                    specie=self.specie,
                    del_id=final_idx + 1,
                    relaxed_file="final.relaxed",
                )
            )

        with subprocess.Popen([self.LMP_EXE, "-in", "in.relax"], stdout=subprocess.PIPE) as p:
            stdout = p.communicate()[0]
            rc = p.returncode
        if rc != 0:
            error_msg = f"LAMMPS exited with return code {rc}"
            msg = stdout.decode("utf-8").split("\n")[:-1]
            try:
                error_line = next(i for i, m in enumerate(msg) if m.startswith("ERROR"))
                error_msg += ", ".join(msg[error_line:])
            except Exception:
                error_msg += msg[-1]
            raise RuntimeError(error_msg)

        final_relaxed_struct = LammpsData.from_file("final.relaxed", atom_style="charge").structure

        lines = [f"{final_relaxed_struct.num_sites}"]

        for idx, site in enumerate(final_relaxed_struct):
            if idx == final_idx:
                idx = final_relaxed_struct.num_sites
            elif idx == start_idx:
                idx = final_idx
            lines.append(f"{idx + 1}  {site.x:.3f}  {site.y:.3f}  {site.z:.3f}")

        with open("data.final_replica", "w") as f:
            f.write("\n".join(lines))

        input_file = "in.neb"

        with open(input_file, "w") as f:
            f.write(
                neb_template.format(
                    ff_settings="\n".join(self.ff_settings.write_param()),
                    start_replica="initial.relaxed",
                    final_replica="data.final_replica",
                )
            )

        return input_file

    def calculate(self):
        """Calculate the NEB barrier given Potential class."""
        with ScratchDir("."):
            input_file = self._setup()
            with subprocess.Popen(
                [
                    "mpirun",
                    "-n",
                    str(self.num_replicas),
                    self.LMP_EXE,
                    "-partition",
                    f"{self.num_replicas}x1",
                    "-in",
                    input_file,
                ],
                stdout=subprocess.PIPE,
            ) as p:
                stdout = p.communicate()[0]
                rc = p.returncode
            if rc != 0:
                error_msg = f"LAMMPS exited with return code {rc}"
                msg = stdout.decode("utf-8").split("\n")[:-1]
                try:
                    error_line = next(i for i, m in enumerate(msg) if m.startswith("ERROR"))
                    error_msg += ", ".join(msg[error_line:])
                except Exception:
                    logger.info(f"NudgedElasticBand error with message {msg}")
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)
            return self._parse()

    def _sanity_check(self, structure):
        """Check if the structure is valid for this calculation."""
        return True

    def _parse(self):
        """Parse results from dump files."""
        with open("log.lammps") as f:
            lines = f.readlines()[-1:]
        return float(lines[0].split()[6])


class DefectFormation(LMPStaticCalculator):
    """Defect formation energy calculator."""

    def __init__(self, ff_settings, specie, lattice, alat, **kwargs):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for
                LAMMPS calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
            specie (str): Name of specie.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
            **kwargs: Passthrough to init.
        """
        self.ff_settings = ff_settings
        self.specie = specie
        self.lattice = lattice
        self.alat = alat
        super().__init__(**kwargs)

    @staticmethod
    def get_unit_cell(specie, lattice, alat):
        """
        Get the unit cell from specie, lattice type and lattice constant.

        Args:
            specie (str): Name of specie.
            lattice (str): The lattice type of structure. e.g. bcc or diamond.
            alat (float): The lattice constant of specific lattice and specie.
        """
        if lattice == "fcc":
            unit_cell = Structure.from_spacegroup(
                sg="Fm-3m", lattice=Lattice.cubic(alat), species=[specie], coords=[[0, 0, 0]]
            )
        elif lattice == "bcc":
            unit_cell = Structure.from_spacegroup(
                sg="Im-3m", lattice=Lattice.cubic(alat), species=[specie], coords=[[0, 0, 0]]
            )
        elif lattice == "diamond":
            unit_cell = Structure.from_spacegroup(
                sg="Fd-3m", lattice=Lattice.cubic(alat), species=[specie], coords=[[0, 0, 0]]
            )
        else:
            raise ValueError("Lattice type is invalid.")

        return unit_cell

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), "templates", "defect")

        with open(os.path.join(template_dir, "in.defect")) as f:
            defect_template = f.read()

        unit_cell = self.get_unit_cell(specie=self.specie, lattice=self.lattice, alat=self.alat)
        lattice_calculator = LatticeConstant(ff_settings=self.ff_settings)
        a, _, _ = lattice_calculator.calculate([unit_cell])[0]
        unit_cell = self.get_unit_cell(specie=self.specie, lattice=self.lattice, alat=a)

        if self.lattice == "fcc":
            idx, scale_factor = 95, [3, 3, 3]
        elif self.lattice == "bcc":
            idx, scale_factor = 40, [3, 3, 3]
        elif self.lattice == "diamond":
            idx, scale_factor = 7, [2, 2, 2]
        else:
            raise ValueError("Lattice type is invalid.")

        super_cell = unit_cell * scale_factor
        efs_calculator = EnergyForceStress(ff_settings=self.ff_settings)
        energy_per_atom = efs_calculator.calculate([super_cell])[0][0] / len(super_cell)

        super_cell_ld = LammpsData.from_structure(
            super_cell, ff_elements=self.ff_settings.elements, atom_style="charge"
        )
        super_cell_ld.write_file("data.supercell")

        input_file = "in.defect"

        with open(input_file, "w") as f:
            f.write(
                defect_template.format(
                    ff_settings="\n".join(self.ff_settings.write_param()),
                    lattice=self.lattice,
                    alat=a,
                    specie=self.specie,
                    del_id=idx + 1,
                    relaxed_file="data.relaxed",
                )
            )

        return input_file, energy_per_atom, len(super_cell) - 1

    def calculate(self):
        """Calculate the vacancy formation given Potential class."""
        with ScratchDir("."):
            input_file, energy_per_atom, num_atoms = self._setup()
            with subprocess.Popen([self.LMP_EXE, "-in", input_file], stdout=subprocess.PIPE) as p:
                stdout = p.communicate()[0]
                rc = p.returncode
            if rc != 0:
                error_msg = f"LAMMPS exited with return code {rc}"
                msg = stdout.decode("utf-8").split("\n")[:-1]
                try:
                    error_line = next(i for i, m in enumerate(msg) if m.startswith("ERROR"))
                    error_msg += ", ".join(msg[error_line:])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)
            defect_energy, _, _ = self._parse()
        return defect_energy - energy_per_atom * num_atoms

    def _sanity_check(self, structure):
        return True

    def _parse(self):
        energy = float(np.loadtxt("energy.txt"))
        force = _read_dump("force.dump")
        stress = np.loadtxt("stress.txt")
        return energy, force, stress


class LMPRelaxationCalculator(LMPStaticCalculator):
    """Structural Relaxation Calculator."""

    def __init__(
        self,
        ff_settings,
        box_relax=True,
        box_relax_keywords="aniso 0.0 vmax 0.001",
        box_triclinic=False,
        min_style="cg",
        etol=1e-15,
        ftol=1e-15,
        maxiter=5000,
        maxeval=5000,
        **kwargs,
    ):
        """
        Args:
            ff_settings (list/Potential): Configure the force field settings for
                LAMMPS calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
            box_relax (bool): Whether to allow the box size and shape to vary
                during the minimization.
            box_relax_keywords (str): Keywords and values to define the conditions
                and constraints on the box size and shape.
            box_triclinic (bool): Whether to change box to triclinic so that
                LAMMPS can adjust x, y and z independently in relaxation.
            min_style (str): The minimization algorithm to use.
            etol (float): The stopping tolerance for energy during the minimization.
            ftol (float): The stopping tolerance for force during the minimization.
            maxiter (int): The max iterations of minimizer.
            maxeval (int): The max number of force/energy evaluations.
            **kwargs: Passthrough to init.
        """
        self.ff_settings = ff_settings
        self.box_relax = box_relax
        self.box_relax_keywords = box_relax_keywords
        self.box_triclinic = box_triclinic
        self.min_style = min_style
        self.etol = etol
        self.ftol = ftol
        self.maxiter = maxiter
        self.maxeval = maxeval
        super().__init__(**kwargs)

    def _setup(self):
        template_dir = os.path.join(os.path.dirname(__file__), "templates", "relax")

        with open(os.path.join(template_dir, "in.relax")) as f:
            input_template = f.read()

        input_file = "in.relax"

        ff_settings = self.ff_settings.write_param() if isinstance(self.ff_settings, Potential) else self.ff_settings

        box_relax_settings = ""
        if self.box_relax:
            box_relax_settings = f"fix   1 all box/relax {self.box_relax_keywords}"

        change_box = "" if not self.box_triclinic else "change_box all triclinic"

        inputs = input_template.format(
            ff_settings="\n".join(ff_settings),
            box_relax_settings=box_relax_settings,
            change_box=change_box,
            min_style=self.min_style,
            etol=self.etol,
            ftol=self.ftol,
            maxiter=self.maxiter,
            maxeval=self.maxeval,
        )

        with open(input_file, "w") as f:
            f.write(inputs)

        return input_file

    def _sanity_check(self, structure):
        """Check if the structure is valid for this calculation."""
        return True

    def _parse(self):
        """Parse results from dump files."""
        ld = LammpsData.from_file("data.relaxed", atom_style="charge")
        final_structure = ld.structure
        efs_calculator = EnergyForceStress(ff_settings=self.ff_settings)
        energy, forces, stresses = efs_calculator.calculate([final_structure])[0]
        return final_structure, energy, forces, stresses


class LatticeConstant(LMPRelaxationCalculator):
    """Lattice Constant Relaxation Calculator."""

    def calculate(self, structures):
        """
        Calculate the relaxed lattice parameters of a list of structures.

        Args:
            structures ([Structure]): Input structures in a list.

        Returns:
            List of relaxed lattice constants (a, b, c in Å) of the input structures.

        """
        results = super().calculate(structures)
        structures_relaxed = [r[0] for r in results]
        return [list(struct.lattice.abc) for struct in structures_relaxed]


class SurfaceEnergy(LMPRelaxationCalculator):
    """
    Surface energy Calculator.

    This calculator generate and calculate surface energies of slab structures based on inputs of
    bulk_structure and miller_indexes with the SlabGenerator in pymatgen:
    https://pymatgen.org/pymatgen.core.surface.html

    """

    def __init__(
        self,
        ff_settings,
        bulk_structure,
        miller_indexes,
        min_slab_size=15,
        min_vacuum_size=15,
        lll_reduce=False,
        center_slab=False,
        in_unit_planes=False,
        primitive=True,
        max_normal_search=None,
        reorient_lattice=True,
        box_relax=False,
        **kwargs,
    ):
        """
        Init.

        Args:
            ff_settings (list/Potential): Configure the force field settings for
                LAMMPS calculation, if given a Potential object, should apply
                Potential.write_param method to get the force field setting.
            bulk_structure (Structure): The bulk structure of target system. Slab structures
                will be generated based on inputs of bulk_structure and miller_indexes with
                the SlabGenerator in pymatgen.
            miller_indexes (list): A list of miller indexes of planes parallel to
                the surface of target slabs. Slab structures
                will be generated based on inputs of bulk_structure and miller_indexes with
                the SlabGenerator in pymatgen.
            min_slab_size (float): Minimum size in angstroms of layers containing atoms.
            min_vacuum_size (float): Minimize size in angstroms of layers containing vacuum.
            lll_reduce (bool): Whether the slabs will be orthogonalized.
            center_slab (bool): Whether the slabs will be centered between the vacuum layer.
            in_unit_planes (bool): Whether to set min_slab_size and min_vac_size in units
                of hkl planes (True) or Angstrom (False/default).
            primitive (bool): Whether to reduce any generated slabs to a primitive cell.
            max_normal_search (int): If set to a positive integer, the code will conduct a search
                for a normal lattice vector that is as perpendicular to the surface as possible by
                considering multiples linear combinations of lattice vectors up to max_normal_search.
            reorient_lattice (bool): Whether to reorient the lattice parameters such that
                the c direction is the third vector of the lattice matrix.
            box_relax (bool): Whether to allow the box size and shape of the slab structures to vary
                during the minimization. Normally, this should be turned off.
            **kwargs: Passthrough to init.
        """
        super().__init__(ff_settings=ff_settings, box_relax=True, **kwargs)
        bulk_structure_relaxed, bulk_energy, _, _ = super().calculate([bulk_structure])[0]
        self.bulk_energy_per_atom = bulk_energy / bulk_structure_relaxed.num_sites
        from pymatgen.core.surface import SlabGenerator

        slab_generators = [
            SlabGenerator(
                initial_structure=bulk_structure_relaxed,
                miller_index=miller_index,
                min_slab_size=min_slab_size,
                min_vacuum_size=min_vacuum_size,
                lll_reduce=lll_reduce,
                center_slab=center_slab,
                in_unit_planes=in_unit_planes,
                primitive=primitive,
                max_normal_search=max_normal_search,
                reorient_lattice=reorient_lattice,
            )
            for miller_index in miller_indexes
        ]
        slabs = [SG.get_slab() for SG in slab_generators]
        self.miller_indexes = miller_indexes
        self.slabs = slabs
        self.surface_areas = [s.surface_area for s in slabs]

        super().__init__(ff_settings=ff_settings, box_relax=box_relax, **kwargs)

    def calculate(self):
        """
        Calculate surface energies with the formula:
        E(Surface) = (E(Slab) - E(bulk)) / Area(surface). (J/m^2).

        Returns:
            List of miller_indexes with their respective relaxed slab structures and surface energies in J/m^2.

        """
        results = super().calculate(self.slabs)
        slabs_relaxed = [r[0] for r in results]
        slab_energies = [r[1] for r in results]
        surface_energies = [
            (slab_energy - slabs_relaxed[i].num_sites * self.bulk_energy_per_atom) / 2 / self.surface_areas[i]
            for i, slab_energy in enumerate(slab_energies)
        ]
        from pymatgen.core.units import Unit

        energy_unit_conversion_factor = Unit("eV").get_conversion_factor("J")
        length_unit_conversion_factor = Unit("ang").get_conversion_factor("m")
        surface_energies = [
            surface_energy * energy_unit_conversion_factor / length_unit_conversion_factor**2
            for surface_energy in surface_energies
        ]

        return list(zip(self.miller_indexes, slabs_relaxed, surface_energies, strict=False))


class LammpsPotential(Potential, abc.ABC):  # type: ignore
    """
    Lammps compatible potentials that call lammps executable for
    energy/force/stress calculations.
    """

    def predict_efs(self, structure):
        """
        Predict energy, forces and stresses of the structure.

        Args:
            structure (Structure): Pymatgen Structure object.

        Returns:
            energy, forces, stress
        """
        calculator = EnergyForceStress(self)
        energy, forces, stress = calculator.calculate(structures=[structure])[0]
        return energy, forces, stress
