"""Convert data list to docs or pool existing data lists for training."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pymatgen.core import Structure


def doc_from(structure, energy=None, force=None, stress=None):
    """
    Method to convert structure and its properties into doc
    format for further processing. If properties are None, zeros
    array will be used.

    Args:
        structure (Structure): Pymatgen Structure object.
        energy (float): The total energy of the structure.
        force (np.array): The (m, 3) forces array of the structure
            where m is the number of atoms in structure.
        stress (list/np.array): The (6, ) stresses array of the
            structure.

    Returns:
        (dict)
    """
    energy = energy if energy is not None else 0
    force = force if force is not None else np.zeros((len(structure), 3))
    stress = stress if stress is not None else np.zeros(6)
    outputs = dict(energy=energy, forces=force, virial_stress=stress)
    return dict(structure=structure.as_dict(), num_atoms=len(structure), outputs=outputs)


def pool_from(structures, energies=None, forces=None, stresses=None):
    """
    Method to convert structures and their properties in to
    datapool format.

    Args:
        structures ([Structure]): The list of Pymatgen Structure object.
        energies ([float]): The list of total energies of each structure
            in structures list.
        forces ([np.array]): List of (m, 3) forces array of each structure
            with m atoms in structures list. m can be varied with each
            single structure case.
        stresses (list): List of (6, ) virial stresses of each
            structure in structures list.

    Returns:
        ([dict])
    """
    energies = energies if energies is not None else [None] * len(structures)
    forces = forces if forces is not None else [None] * len(structures)
    stresses = stresses if stresses is not None else [None] * len(structures)
    return [
        doc_from(structure, energy, force, stress)
        for structure, energy, force, stress in zip(structures, energies, forces, stresses, strict=False)
    ]


def convert_docs(docs, include_stress=False, **kwargs):
    """
    Method to convert a list of docs into objects, e.g.,
    Structure and DataFrame.

    Args:
        docs ([dict]): List of docs. Each doc should have the same
            format as one returned from .dft.parse_dir.
        include_stress (bool): Whether to include stress components.
        **kwargs: Passthrough.

    Returns:
        A list of structures, and a DataFrame with energy and force
        data in 'y_orig' column, data type ('energy' or 'force') in
        'dtype' column, No. of atoms in 'n' column sharing the same row
        of energy data while 'n' being 1 for the rows of force data.

    """
    structures, y_orig, n, dtype = [], [], [], []
    for d in docs:
        structure = Structure.from_dict(d["structure"]) if isinstance(d["structure"], dict) else d["structure"]
        outputs = d["outputs"]
        force_arr = np.array(outputs["forces"])
        assert force_arr.shape == (len(structure), 3), "Wrong force array not matching structure"
        force_arr = force_arr.ravel()
        if include_stress:
            virial_stress = outputs["virial_stress"]
            y = np.concatenate(([outputs["energy"]], force_arr, virial_stress))
            dtype.extend(["energy"] + ["force"] * len(force_arr) + ["stress"] * len(virial_stress))
        else:
            y = np.concatenate(([outputs["energy"]], force_arr))
            dtype.extend(["energy"] + ["force"] * len(force_arr))
        y_orig.append(y)
        n.append(np.insert(np.ones(len(y) - 1), 0, d["num_atoms"]))

        structures.append(structure)
    df = pd.DataFrame(dict(y_orig=np.concatenate(y_orig), n=np.concatenate(n), dtype=dtype))
    for k, v in kwargs.items():
        df[k] = v
    return structures, df


def to_array(x):
    """
    Convert x into numerical array
    Args:
        x: x can be a dataframe, a list or an array
    return np.ndarray.
    """
    if isinstance(x, pd.DataFrame):
        return x.to_numpy()
    if isinstance(x, list):
        return np.array([to_array(i) for i in x])
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, str | int | float):
        return x
    raise ValueError("Not recognized data type")
