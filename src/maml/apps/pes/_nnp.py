# Copyright (c) Materials Virtual Lab
# Distributed under the terms of the BSD License.

"""This module provides NNP interatomic potential class."""

from __future__ import annotations

import itertools
import os
import re
import subprocess
from collections import OrderedDict
from shutil import which

import numpy as np
import pandas as pd
from monty.io import zopen
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from pymatgen.core import Element, Lattice, Structure, units

from maml.utils import check_structures_forces_stresses, convert_docs, pool_from

from ._lammps import LammpsPotential

module_dir = os.path.dirname(__file__)
NNinput_params = loadfn(os.path.join(module_dir, "params", "NNinput.json"))


class NNPotential(LammpsPotential):
    """This class implements Neural Network Potential."""

    bohr_to_angstrom = units.bohr_to_angstrom
    eV_to_Ha = units.eV_to_Ha
    pair_style = (
        'pair_style        nnp dir "./" showew no showewsum 0 '
        "maxew 10000000 resetew yes cflength 1.8897261328 cfenergy 0.0367493254"
    )
    pair_coeff = "pair_coeff        * * {}"

    def __init__(self, name=None, param=None, weight_param=None, scaling_param=None):
        """

        Args:
            name (str): Name of force field.
            param (str): Input parameters.
            weight_param (dict): Atomic neural networks weights.
            scaling_param (dict): Scaling parameters for symmetry functions.
        """
        self.name = name if name else "NNPotential"
        self.elements = None
        self.weights = {}
        self.bs = {}
        self.atom_energy = None
        self.normalized_nodes = None
        self.epochs = None
        self.param = param if param else {}
        self.weight_param = weight_param if weight_param else {}
        self.scaling_param = scaling_param if scaling_param else None
        self.train_energy_rmse = None
        self.validation_energy_rmse = None
        self.train_forces_rmse = None
        self.validation_forces_rmse = None
        self.fitted = False

    def _line_up(self, structure, energy, forces, virial_stress):
        """
        Convert input structure, energy, forces, virial_stress to
        proper configuration format for n2p2 usage. Note that
        n2p2 takes bohr as length unit and Hartree as energy unit.

        Args:
            structure (Structure): Pymatgen Structure object.
            energy (float): DFT-calculated energy of the system.
            forces (list): The forces should have dimension
                (num_atoms, 3).
            virial_stress (list): stress should has 6 distinct
                elements arranged in order [xx, yy, zz, xy, yz, xz].
        """
        inputs = OrderedDict(
            Size=structure.num_sites,
            SuperCell=structure.lattice,
            AtomData=(structure, forces),
            Energy=energy,
            Stress=virial_stress,
        )

        lines = ["begin"]

        if "SuperCell" in inputs:
            bohr_matrix = inputs["SuperCell"].matrix / self.bohr_to_angstrom
            for vec in bohr_matrix:
                lines.append(f"lattice {vec[0]:>15.6f}{vec[1]:>15.6f}{vec[2]:>15.6f}")
        if "AtomData" in inputs:
            format_float = "atom{:>16.9f}{:>16.9f}{:>16.9f}{:>4s}{:>15.9f}{:>15.9f}{:>15.9f}{:>15.9f}{:>15.9f}"
            for _i, (site, force) in enumerate(zip(structure, forces, strict=False)):
                lines.append(
                    format_float.format(
                        *site.coords / self.bohr_to_angstrom,
                        site.species_string,
                        0.0,
                        0.0,
                        *np.array(force) * self.eV_to_Ha * self.bohr_to_angstrom,
                    )
                )
        if "Energy" in inputs:
            lines.append(f"energy  {energy * self.eV_to_Ha:f}")

        lines.append(f"charge  {structure.charge:f}")
        lines.append("end")

        return "\n".join(lines)

    def write_cfgs(self, filename, cfg_pool):
        """
        Write the formatted configuration file.

        Args:
            filename (str): The filename to be written.
            cfg_pool (list): The configuration pool contains
                structure and energy/forces properties.
        """
        lines = []
        elements = []
        for dataset in cfg_pool:
            if isinstance(dataset["structure"], dict):
                structure = Structure.from_dict(dataset["structure"])
            else:
                structure = dataset["structure"]
            energy = dataset["outputs"]["energy"]
            forces = dataset["outputs"]["forces"]
            virial_stress = dataset["outputs"]["virial_stress"]

            lines.append(self._line_up(structure, energy, forces, virial_stress))

            elements.extend(structure.species)

        if self.elements is None:
            self.elements = [element.name for element in sorted(set(elements))]

        with open(filename, "w") as f:
            f.write("\n".join(lines))

        return filename

    def write_input(self, **kwargs):
        """
        Write input.nn file to train the Neural Network Potential.

        Args:
            atom_energy (float): Atomic reference energy.

            kwargs:
                General nnp settings:
                    atom_energy (dict): Free atom reference energy for each specie.
                    cutoff_type (int): Type of cutoff function. Default to 1
                        (i.e., cosine function).
                    scale_features (int): Determine the method to scale the
                        symmetry function.
                        0: no scaling.
                        1: scale_symmetry_functions.
                        2: center_symmetry_functions.
                        3. scale_symmetry_functions_sigma.
                    scale_min_short (float): Minimum value for scaling.
                        Default to 0.0.
                    scale_max_short (float): Maximum value for scaling.
                        Default to 1.
                    hidden_layers (list): List of the numbers of
                        nodes in each hidden layer.
                    activations (str): Activation function for each hidden layer.
                        't': tanh, 's': logistic, 'p': softplus.
                    normalize_nodes (boolean): Whether to normalize input of nodes.

                Additional settings for training:
                    epoch (int): Number of training epochs.
                    updater_type (int): Weight update method
                        0: gradient Descent, 1: Kalman filter.
                    parallel_mode (int): Training parallelization used.
                        Default to serial mode.
                    jacobian_mode (int): Jacobian computation mode.
                        0: Summation to single gradient,
                        1: Per-task summed gradient,
                        2: Full Jacobian.
                    update_strategy (int): Update strategy.
                        0: combined, 1: per-element.
                    selection_mode (int): Update candidate selection mode.
                        0: random, 1: sort, 2: threshold
                    task_batch_size_energy (int): Number of energy update
                        candidates prepared per task for each update.
                    task_batch_size_force (int): Number of force update
                        candidates prepared per task for each update.
                    test_fraction (float): Fraction of structures kept for
                        testing.
                    force_weight (float): Weight of force updates relative
                        to energy updates. Default to 10.0
                    short_energy_fraction (float): Fraction of energy updates
                        per epoch. Default to 1.0.
                    short_force_fraction (float): Fraction of force updates
                        per epoch. Default to 0.02315.
                    short_energy_error_threshold (float): RMSE threshold for
                        energy update candidates. Default to 0.0.
                    short_force_error_threshold (float): RMSE threshold for
                        force update candidates. Default to 1.0.
                    rmse_threshold_trials (int): Maximum number of RMSE
                        threshold trials. Default to 3.
                    weights_min (float): Minimum value for initial random
                        weights. Default to -1.
                    weights_max (float): Maximum value for initial random
                        weights. Default to 1.
                    write_trainpoints (int): Write energy comparison every
                        this many epochs. Default to 1.
                    write_trainforces (int): Write force comparison every
                        this many epochs. Default to 1.
                    write_weights_epoch (int): Write weights every this many
                        epochs. Default to 1.
                    write_neuronstats (int): Write neuron statistics every
                        this many epochs. Default to 1.

                    # Kalman Filter
                    kalman_type (int): Kalman filter type. Default to 0.
                    kalman_epsilon (float): General Kalman filter parameter
                        epsilon. Default to 0.01.
                    kalman_q0 (float): General Kalman filter parameter q0.
                        Default to 0.01.
                    kalman_qtau (float): General Kalman filter parameter
                        qtau. Default to 2.302.
                    kalman_qmin (float): General Kalman filter parameter qmin.
                        Default to 1e-6.
                    kalman_eta (float): Standard Kalman filter parameter eta.
                        Default to 0.01.
                    kalman_etatau (float): Standard Kalman filter parameter
                        etatau. Defaults to 2.302.
                    kalman_etamax (float): Standard Kalman filter parameter
                        etamax. Default to 1.0.

                Symmetry functions:
                    r_cut (float): Cutoff distance (unit: Å).
                    r_etas (numpy.array): η in radial function.
                    r_shift (numpy.array): Rs in radial function.
                    a_etas (numpy.array): η in angular function.
                    zetas (numpy.array): ζ in angular function.
                    lambdas (numpy.array): λ in angular function. Default to (1, -1).
        """
        filename = "input.nn"

        head_formatter = "{:<32s}{value}"
        type2_format = "symfunction_short {central_atom}  2 {neighbor_atom}    {r_eta:.7f}    {rs:.7f}    {rcut:.7f}"
        type3_format = (
            "symfunction_short {central_atom}  3 {neighbor_atom1} "
            "{neighbor_atom2}    {a_eta:.7f} {lambd:>2d} {zeta:.7f}   "
            "{rcut:.7f}"
        )

        lines = [
            head_formatter.format("number_of_elements", value=len(self.elements)),
            head_formatter.format("elements", value=" ".join(self.elements)),
        ]

        PARAMS = {
            "general": ["cutoff_type", "scale_features", "scale_min_short", "scale_max_short", "hidden_layers"],
            "additional": [
                "epochs",
                "updater_type",
                "parallel_mode",
                "jacobian_mode",
                "update_strategy",
                "selection_mode",
                "task_batch_size_energy",
                "task_batch_size_force",
                "random_seed",
                "test_fraction",
                "force_weight",
                "short_energy_fraction",
                "short_force_fraction",
                "short_energy_error_threshold",
                "short_force_error_threshold",
                "rmse_threshold_trials",
                "weights_min",
                "weights_max",
                "write_trainpoints",
                "write_trainforces",
                "write_weights_epoch",
                "write_neuronstats",
                "kalman_type",
                "kalman_epsilon",
                "kalman_q0",
                "kalman_qtau",
                "kalman_qmin",
                "kalman_eta",
                "kalman_etatau",
                "kalman_etamax",
            ],
        }
        if self.fitted:
            if self.param.get("atom_energy"):
                for specie in self.elements:
                    lines.append(
                        head_formatter.format(
                            "atom_energy", value=" ".join([specie, str(self.param.get("atom_energy").get(specie))])
                        )
                    )
            for tag in PARAMS.get("general"):
                if tag == "scale_features":
                    lines.append(NNinput_params.get("general").get(tag).get(self.param.get(tag)))
                elif tag == "hidden_layers":
                    layers = self.param.get("hidden_layers")
                    activations = self.param.get("activations")
                    lines.append(head_formatter.format("global_hidden_layers_short", value=len(layers)))
                    lines.append(head_formatter.format("global_nodes_short", value=" ".join([str(i) for i in layers])))
                    lines.append(
                        head_formatter.format(
                            "global_activation_short", value=" ".join([activations] * len(layers) + ["l"])
                        )
                    )
                else:
                    lines.append(head_formatter.format(tag, value=self.param.get(tag)))
            if self.normalized_nodes:
                lines.append("normalize_nodes")

            for tag in PARAMS.get("additional"):
                lines.append(head_formatter.format(tag, value=self.param.get(tag)))
            lines.append("use_short_forces")

            r_cut = self.param.get("r_cut")
            r_cut /= self.bohr_to_angstrom
            r_shift = np.array(self.param.get("r_shift"))
            r_shift /= self.bohr_to_angstrom

            for central_atom in self.elements:
                for neighbor_atom1 in self.elements:
                    for r_eta, rs in itertools.product(self.param.get("r_etas"), r_shift):
                        lines.append(
                            type2_format.format(
                                central_atom=central_atom, neighbor_atom=neighbor_atom1, r_eta=r_eta, rs=rs, rcut=r_cut
                            )
                        )
            for central_atom in self.elements:
                for neighbor_atom1, neighbor_atom2 in itertools.combinations_with_replacement(self.elements, 2):
                    for a_eta, lambd, zeta in itertools.product(
                        self.param.get("a_etas"), self.param.get("lambdas"), self.param.get("zetas")
                    ):
                        lines.append(
                            type3_format.format(
                                central_atom=central_atom,
                                neighbor_atom1=neighbor_atom1,
                                neighbor_atom2=neighbor_atom2,
                                a_eta=a_eta,
                                lambd=lambd,
                                zeta=zeta,
                                rcut=r_cut,
                            )
                        )
        else:
            if kwargs.get("atom_energy"):
                for specie in self.elements:
                    lines.append(
                        head_formatter.format(
                            "atom_energy", value=" ".join([specie, str(kwargs.get("atom_energy").get(specie))])
                        )
                    )
                self.param.update({"atom_energy": kwargs.get("atom_energy")})
            for tag in PARAMS.get("general"):
                if tag == "scale_features":
                    value = kwargs.get(tag) if kwargs.get(tag) is not None else "1"
                    lines.append(NNinput_params.get("general").get(tag).get(value))
                    self.param.update({tag: value})
                elif tag == "hidden_layers":
                    layers = kwargs.get(tag, NNinput_params.get("general").get(tag))
                    self.param.update({tag: layers})
                    activations = kwargs.get("activations", NNinput_params.get("general").get("activations"))
                    self.param.update({"activations": activations})
                    lines.append(head_formatter.format("global_hidden_layers_short", value=len(layers)))
                    lines.append(head_formatter.format("global_nodes_short", value=" ".join([str(i) for i in layers])))
                    lines.append(
                        head_formatter.format(
                            "global_activation_short", value=" ".join([activations] * len(layers) + ["l"])
                        )
                    )
                else:
                    value = kwargs.get(tag, NNinput_params.get("general").get(tag))
                    lines.append(head_formatter.format(tag, value=value))
                    self.param.update({tag: value})
            if kwargs.get("normalize_nodes"):
                lines.append("normalize_nodes")
                self.param.update({"normalize_nodes": True})

            for tag in PARAMS.get("additional"):
                value = kwargs.get(tag, NNinput_params.get("additional").get(tag))
                lines.append(head_formatter.format(tag, value=value))
                self.param.update({tag: value})
            lines.append("use_short_forces")

            r_cut = (
                kwargs.get("r_cut")
                if kwargs.get("r_cut") is not None
                else NNinput_params.get("symmetry_function").get("r_cut")
            )
            self.param.update({"r_cut": r_cut})
            r_cut /= self.bohr_to_angstrom
            r_etas = (
                kwargs.get("r_etas")
                if kwargs.get("r_etas") is not None
                else NNinput_params.get("symmetry_function").get("r_etas")
            )
            self.param.update({"r_etas": r_etas})
            r_shift = (
                kwargs.get("r_shift")
                if kwargs.get("r_shift") is not None
                else NNinput_params.get("symmetry_function").get("r_shift")
            )
            self.param.update({"r_shift": r_shift})
            r_shift = np.array(r_shift)
            r_shift /= self.bohr_to_angstrom
            a_etas = (
                kwargs.get("a_etas")
                if kwargs.get("a_etas") is not None
                else NNinput_params.get("symmetry_function").get("a_etas")
            )
            self.param.update({"a_etas": a_etas})
            zetas = (
                kwargs.get("zetas")
                if kwargs.get("zetas") is not None
                else NNinput_params.get("symmetry_function").get("zetas")
            )
            self.param.update({"zetas": zetas})
            lambdas = (
                kwargs.get("lambdas")
                if kwargs.get("lambdas") is not None
                else NNinput_params.get("symmetry_function").get("lambdas")
            )
            self.param.update({"lambdas": lambdas})

            for central_atom in self.elements:
                for neighbor_atom1 in self.elements:
                    for r_eta, rs in itertools.product(self.param.get("r_etas"), r_shift):
                        lines.append(
                            type2_format.format(
                                central_atom=central_atom, neighbor_atom=neighbor_atom1, r_eta=r_eta, rs=rs, rcut=r_cut
                            )
                        )
            for central_atom in self.elements:
                for neighbor_atom1, neighbor_atom2 in itertools.combinations_with_replacement(self.elements, 2):
                    for a_eta, lambd, zeta in itertools.product(
                        self.param.get("a_etas"), self.param.get("lambdas"), self.param.get("zetas")
                    ):
                        lines.append(
                            type3_format.format(
                                central_atom=central_atom,
                                neighbor_atom1=neighbor_atom1,
                                neighbor_atom2=neighbor_atom2,
                                a_eta=a_eta,
                                lambd=lambd,
                                zeta=zeta,
                                rcut=r_cut,
                            )
                        )

            self.num_symm_functions = sum(len(list(itertools.product(r_etas, r_shift))) for _ in self.elements) + sum(
                len(list(itertools.product(a_etas, lambdas, zetas)))
                for _, _ in itertools.combinations_with_replacement(self.elements, 2)
            )

            self.layer_sizes = [self.num_symm_functions, *self.param.get("hidden_layers")]

        with open(filename, "w") as f:
            f.write("\n".join(lines))

        self.fitted = True

        return filename

    def load_input(self, filename="input.nn"):
        """
        Load input file from trained Neural Network Potential.

        Args:
            filename (str): The input filename.
        """
        PARAMS = {
            "general": ["cutoff_type", "scale_features", "scale_min_short", "scale_max_short", "hidden_layers"],
            "additional": [
                "epochs",
                "updater_type",
                "parallel_mode",
                "jacobian_mode",
                "update_strategy",
                "selection_mode",
                "task_batch_size_energy",
                "task_batch_size_force",
                "random_seed",
                "test_fraction",
                "force_weight",
                "short_energy_fraction",
                "short_force_fraction",
                "short_energy_error_threshold",
                "short_force_error_threshold",
                "rmse_threshold_trials",
                "weights_min",
                "weights_max",
                "write_trainpoints",
                "write_trainforces",
                "write_weights_epoch",
                "write_neuronstats",
                "kalman_type",
                "kalman_epsilon",
                "kalman_q0",
                "kalman_qtau",
                "kalman_qmin",
                "kalman_eta",
                "kalman_etatau",
                "kalman_etamax",
            ],
        }

        def str_formatify(string):
            return float(string) if "." in string or "e" in string else int(string)

        param = {}
        with open(filename) as f:
            lines = f.readlines()
        df = pd.DataFrame([line.split() for line in lines if "#" not in line])
        self.elements = sorted(
            (element for element in np.ravel(df[df[0] == "elements"])[1:] if element is not None),
            key=Element,
        )

        atom_energy = {}
        for atom, energy in zip(
            np.array(df[df[0] == "atom_energy"])[:, 1], np.array(df[df[0] == "atom_energy"])[:, 2], strict=False
        ):
            atom_energy[atom] = float(energy)
        param.update({"atom_energy": atom_energy})
        for tag in PARAMS.get("general"):
            if tag == "scale_features":
                scale_features = "1" if len(df[df[0] == "scale_symmetry_functions"]) != 0 else 0
                param.update({"scale_features": scale_features})
            elif tag == "hidden_layers":
                hidden_layers = [int(neuron) for neuron in np.array(df[df[0] == "global_nodes_short"])[0][1:] if neuron]
                param.update({"hidden_layers": hidden_layers})
                activations = np.array(df[df[0] == "global_activation_short"])[0][1]
                param.update({"activations": activations})
            else:
                value = str_formatify(np.array(df[df[0] == tag])[0][1])
                param.update({tag: value})
        if len(df[df[0] == "normalize_nodes"]) != 0:
            param.update({"normalize_nodes": True})

        for tag in PARAMS.get("additional"):
            value = str_formatify(np.array(df[df[0] == tag])[0][1])
            param.update({tag: value})

        r_cut = np.sort(np.array(df[(df[0] == "symfunction_short") & (df[2] == "2")][6], dtype=np.float64))[0]
        r_cut = float(f"{r_cut * units.bohr_to_angstrom:.1f}")
        param.update({"r_cut": r_cut})
        r_etas = np.sort(
            np.array(np.unique(df[(df[0] == "symfunction_short") & (df[2] == "2")][4]), dtype=np.float64)
        ).tolist()
        param.update({"r_etas": r_etas})
        r_shift = np.sort(
            np.array(np.unique(df[(df[0] == "symfunction_short") & (df[2] == "2")][5]), dtype=np.float64)
        ).tolist()
        r_shift = [float(f"{r * units.bohr_to_angstrom:.1f}") for r in r_shift]
        param.update({"r_shift": r_shift})
        a_etas = np.sort(
            np.array(np.unique(df[(df[0] == "symfunction_short") & (df[2] == "3")][5]), dtype=np.float64)
        ).tolist()
        param.update({"a_etas": a_etas})
        lambdas = np.sort(
            np.array(np.unique(df[(df[0] == "symfunction_short") & (df[2] == "3")][6]), dtype=np.int64)
        ).tolist()
        param.update({"lambdas": lambdas})
        zetas = np.sort(
            np.array(np.unique(df[(df[0] == "symfunction_short") & (df[2] == "3")][7]), dtype=np.float64)
        ).tolist()
        param.update({"zetas": zetas})
        self.num_symm_functions = sum(len(list(itertools.product(r_etas, r_shift))) for _ in self.elements) + sum(
            len(list(itertools.product(a_etas, lambdas, zetas)))
            for _, _ in itertools.combinations_with_replacement(self.elements, 2)
        )
        self.layer_sizes = [self.num_symm_functions, *hidden_layers]
        self.param = param

    def load_weights(self, weights_filename, specie):
        """
        Load weights file of trained Neural Network Potential.

        Args:
            weights_filename (str): The weights file.
            specie (str): The name of specie.
        """
        if not self.weights.get(specie) or not self.bs.get(specie) or not self.weight_param:
            self.weights[specie] = []
            self.bs[specie] = []
            self.weight_param[specie] = []

        with open(weights_filename) as f:
            weights_lines = f.readlines()

        weight_param = pd.DataFrame([line.split() for line in weights_lines if "#" not in line])
        weight_param.columns = ["value", "type", "index", "start_layer", "start_neuron", "end_layer", "end_neuron"]

        for layer_index in range(1, len(self.layer_sizes)):
            weights_group = weight_param[
                (weight_param["start_layer"] == str(layer_index - 1)) & (weight_param["end_layer"] == str(layer_index))
            ]

            weights = np.reshape(
                np.array(weights_group["value"], dtype=np.float64),
                (self.layer_sizes[layer_index - 1], self.layer_sizes[layer_index]),
            )
            self.weights[specie].append(weights)

            bs_group = weight_param[(weight_param["type"] == "b") & (weight_param["start_layer"] == str(layer_index))]
            bs = np.array(bs_group["value"], dtype=np.float64)
            self.bs[specie].append(bs)

        self.weight_param[specie] = weight_param

    def load_scaler(self, scaling_filename):
        """
        Load scaling info of trained Neural Network Potential.

        Args:
            scaling_filename (str): The scaling file.
        """
        with open(scaling_filename) as f:
            scaling_lines = f.readlines()
        scaling_param = pd.DataFrame([line.split() for line in scaling_lines if "#" not in line])
        self.scaling_param = scaling_param

    def read_cfgs(self, filename="output.data"):
        """
        Read the configuration file.

        Args:
            filename (str): The configuration file to be read.
        """
        data_pool = []
        with zopen(filename, "rt") as f:
            lines = f.read()

        block_pattern = re.compile("begin\n(.*?)end", re.S)
        lattice_pattern = re.compile("lattice(.*?)\n")
        position_pattern = re.compile("atom(.*?)\n")
        energy_pattern = re.compile("energy(.*?)\n")

        for block in block_pattern.findall(lines):
            d = {"outputs": {}}
            lattice_str = lattice_pattern.findall(block)
            lattice = Lattice(
                np.array([latt.split() for latt in lattice_str], dtype=np.float64) * self.bohr_to_angstrom
            )
            position_str = position_pattern.findall(block)
            positions = pd.DataFrame([pos.split() for pos in position_str])
            positions.columns = ["x", "y", "z", "specie", "charge", "atomic_energy", "fx", "fy", "fz"]
            coords = np.array(positions.loc[:, ["x", "y", "z"]], dtype=np.float64)
            coords = coords * self.bohr_to_angstrom
            species = np.array(positions["specie"])
            forces = np.array(positions.loc[:, ["fx", "fy", "fz"]], dtype=np.float64)
            forces = forces / self.eV_to_Ha / self.bohr_to_angstrom
            energy_str = energy_pattern.findall(block)[0]
            energy = float(energy_str.lstrip()) / self.eV_to_Ha
            struct = Structure(lattice=lattice, species=species, coords=coords, coords_are_cartesian=True)
            d["structure"] = struct.as_dict()
            d["outputs"]["energy"] = energy
            d["outputs"]["forces"] = forces
            d["num_atoms"] = len(struct)

            data_pool.append(d)
        _, df = convert_docs(docs=data_pool)
        return data_pool, df

    def write_param(self):
        """Write optimized weights file to perform energy and force prediction."""
        if self.weight_param is None or self.scaling_param is None:
            raise RuntimeError("The parameters should be provided.")
        for specie in self.elements:
            weights_filename = ".".join(["weights", str(Element(specie).number).zfill(3), "data"])
            weight_formatter = "{:>18s}{:>2s}{:>10s}{:>6s}{:>6s}{:>6s}{:>6s}"
            bias_formatter = "{:>18s}{:>2s}{:>10s}{:>6s}{:>6}"
            lines = []
            for i in range(self.weight_param[specie].shape[0]):
                if self.weight_param[specie].iloc[i]["type"] == "a":
                    lines.append(weight_formatter.format(*self.weight_param[specie].iloc[i]))
                else:
                    lines.append(bias_formatter.format(*self.weight_param[specie].iloc[i]))

            with open(weights_filename, "w") as f:
                f.writelines("\n".join(lines))

        scaling_filename = "scaling.data"
        scaling_formatter = "{:>4s}{:>5s}  {:>22s} {:>22s} {:>22s} {:.>22s}"
        scaling_lines = []
        for i in range(self.num_symm_functions * len(self.elements)):
            scaling_lines.append(scaling_formatter.format(*self.scaling_param.iloc[i]))
        with open(scaling_filename, "w") as f:
            f.writelines("\n".join(scaling_lines))

        self.write_input()

        return [self.pair_style, self.pair_coeff.format(self.param.get("r_cut") + 1e-2)]

    def train(self, train_structures, train_energies, train_forces, train_stresses=None, **kwargs):
        """
        Training data with moment tensor method.

        Args:
            train_structures ([Structure]): The list of Pymatgen Structure object.
                energies ([float]): The list of total energies of each structure
                in structures list.
            train_energies ([float]): List of total energies of each structure in
                structures list.
            train_forces ([np.array]): List of (m, 3) forces array of each structure
                with m atoms in structures list. m can be varied with each
                single structure case.
            train_stresses (list): List of (6, ) virial stresses of each
                structure in structures list.
            kwargs: Parameters in write_input method.
        """
        if not which("nnp-train"):
            raise RuntimeError("NNP Trainer has not been found.")
        train_structures, train_forces, train_stresses = check_structures_forces_stresses(
            train_structures, train_forces, train_stresses
        )
        train_pool = pool_from(train_structures, train_energies, train_forces, train_stresses)
        atoms_filename = "input.data"

        with ScratchDir("."):
            _ = self.write_cfgs(filename=atoms_filename, cfg_pool=train_pool)
            output = "training_output"

            self.write_input(**kwargs)
            with subprocess.Popen(["nnp-scaling", "100"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p_scaling:
                _stdout, stderr = p_scaling.communicate()
                rc = p_scaling.returncode
            if rc != 0:
                error_msg = f"n2p2 exited with return code {rc}"
                msg = stderr.decode("utf-8").split("\n")[:-1]
                try:
                    error_line = next(i for i, m in enumerate(msg) if m.startswith("ERROR"))
                    error_msg += ", ".join(msg[error_line:])
                except Exception:
                    error_msg += ", "
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)

            with open(output, "w") as f, subprocess.Popen(["nnp-train"], stdout=f, stderr=subprocess.PIPE) as p_train:
                _stdout, stderr = p_train.communicate()
                rc = p_train.returncode

            if rc != 0:
                error_msg = f"n2p2 exited with return code {rc}"
                msg = stderr.decode("utf-8").split("\n")[:-1]
                try:
                    error_line = next(i for i, m in enumerate(msg) if m.startswith("ERROR"))
                    error_msg += ", ".join(msg[error_line:])
                except Exception:
                    error_msg += ", "
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)

            with zopen(output) as f:
                error_lines = f.read()

            energy_rmse_pattern = re.compile(r"ENERGY\s*\S*\s*(\S*)\s*(\S*).*?\n")
            forces_rmse_pattern = re.compile(r"FORCE\s*\S*\s*(\S*)\s*(\S*).*?\n")
            errors = np.array(energy_rmse_pattern.findall(error_lines), dtype=np.float64).T.tolist()
            self.train_energy_rmse = errors[0]
            self.validation_energy_rmse = errors[1]

            errors = np.array(forces_rmse_pattern.findall(error_lines), dtype=np.float64).T.tolist()
            self.train_forces_rmse = errors[0]
            self.validation_forces_rmse = errors[1]

            for specie in self.elements:
                weights_filename = (
                    f"weights.{str(Element(specie).number).zfill(3)}.{str(self.param['epochs']).zfill(6)}.out"
                )
                self.weights[specie] = []
                self.bs[specie] = []
                self.weight_param[specie] = []
                self.load_weights(weights_filename, specie)
            self.load_scaler("scaling.data")

        return rc

    def evaluate(self, test_structures, test_energies, test_forces, test_stresses=None):
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
        """
        if not which("nnp-predict"):
            raise RuntimeError("NNP Predictor has not been found.")

        original_file = "input.data"
        predict_file = "output.data"
        test_structures, test_forces, test_stresses = check_structures_forces_stresses(
            test_structures, test_forces, test_stresses
        )
        predict_pool = pool_from(test_structures, test_energies, test_forces, test_stresses)
        with ScratchDir("."):
            _, _ = self.write_param()
            original_file = self.write_cfgs(original_file, cfg_pool=predict_pool)
            _, df_orig = self.read_cfgs(original_file)

            input_filename = self.write_input()

            dfs = []
            for data in predict_pool:
                _ = self.write_cfgs(original_file, cfg_pool=[data])
                with subprocess.Popen(
                    ["nnp-predict", input_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ) as p_evaluation:
                    _stdout, stderr = p_evaluation.communicate()
                    rc = p_evaluation.returncode
                if rc != 0:
                    error_msg = f"n2p2 exited with return code {rc}"
                    msg = stderr.decode("utf-8").split("\n")[:-1]
                    try:
                        error_line = next(i for i, m in enumerate(msg) if m.startswith("ERROR"))
                        error_msg += ", ".join(msg[error_line:])
                    except Exception:
                        error_msg += ", "
                        error_msg += msg[-1]
                    raise RuntimeError(error_msg)

                _, df = self.read_cfgs(predict_file)
                dfs.append(df)
            df_predict = pd.concat(dfs, ignore_index=True)

        return df_orig, df_predict

    @staticmethod
    def from_config(input_filename, scaling_filename, weights_filenames):
        """
        Initialize potentials with parameters file.

        Args:
            input_filename (str): The file storing the input configuration of
                Neural Network Potential.
            scaling_filename (str): The file storing scaling info of
                Neural Network Potential.
            weights_filenames (list): List of files storing weights of each specie in
                Neural Network Potential.
        """
        nnp = NNPotential()
        nnp.load_input(input_filename)
        nnp.load_scaler(scaling_filename)
        if len(nnp.elements) != len(weights_filenames):
            raise ValueError(f"{len(nnp.elements)} weights files should be given to " + " ".join(nnp.elements))
        for weights_filename, specie in zip(weights_filenames, nnp.elements, strict=False):
            nnp.load_weights(weights_filename, specie)
        nnp.fitted = True

        return nnp
