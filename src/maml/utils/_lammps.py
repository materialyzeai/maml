"""LAMMPS utility."""

from __future__ import annotations

import logging

import numpy as np
from pymatgen.core import Element, Lattice, Species, Structure
from pymatgen.core.operations import SymmOp

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STRESS_FORMAT = {
    "VASP": ["xx", "yy", "zz", "xy", "yz", "xz"],
    "LAMMPS": ["xx", "yy", "zz", "xy", "xz", "yz"],
    "SNAP": ["xx", "yy", "zz", "yz", "xz", "xy"],
}


def check_structures_forces_stresses(
    structures: list[Structure],
    forces: list | None = None,
    stresses: list | None = None,
    stress_format: str = "VASP",
    return_none: bool = True,
):
    """
    Check structures, forces and stresses. The forces and stress are dependent
    on the lattice orientation. This function will rotate the structures
    and the corresponding forces and structures to lammps format
    [[ax, 0, 0],
    [bx, by, 0],
    [cx, cy, cz]].

    The lattice are formed by the row vectors.

    Args:
        structures (list): list of structures
        forces (list): list of force matrixs (m, 3)
        stresses (list): list of stress vectors
        stress_format (str): stress format, choose from
            "VASP", "LAMMPS", "SNAP"
        return_none (bool): whether to return list of None
            for forces and stresses

    Returns: structures [forces], [stresses]

    """
    new_structures = []
    new_forces = []
    new_stresses = []

    no_force = forces is None
    no_stress = stresses is None

    if forces is None:
        forces = [None] * len(structures)
    if stresses is None:
        stresses = [None] * len(structures)

    for i, s in enumerate(structures):
        # orthogonal structures do not need to rotate
        if s.lattice.is_orthogonal:
            new_structures.append(s)
            new_forces.append(forces[i])  # type: ignore
            new_stresses.append(stresses[i])  # type: ignore
            continue

        logger.info(f"Structure index {i} is rotated.")
        new_latt_matrix, symmop, rot_matrix = get_lammps_lattice_and_rotation(s, (0, 0, 0))
        coords = symmop.operate_multi(s.cart_coords)
        new_s = Structure(
            Lattice(new_latt_matrix),  # type: ignore
            s.species,  # type: ignore
            coords,  # type: ignore
            site_properties=s.site_properties,
            coords_are_cartesian=True,
        )
        new_structures.append(new_s)

        if not no_force:
            new_f = symmop.operate_multi(forces[i])
            new_forces.append(new_f)
        else:
            new_forces.append(None)

        if not no_stress:
            if np.size(stresses[i]) != 9:
                # voigt stress format
                stress_matrix = stress_list_to_matrix(stresses[i], stress_format)
                stress_matrix = rot_matrix.dot(stress_matrix).dot(rot_matrix.T)
                stress_matrix = stress_matrix_to_list(stress_matrix, stress_format)
            else:
                # 3x3 stress matrix
                stress_matrix = rot_matrix.dot(stresses[i]).dot(rot_matrix.T)
            # R \sigma R^T stress rotation
            new_stresses.append(stress_matrix)
        else:
            new_stresses.append(None)

    if return_none:
        return new_structures, new_forces, new_stresses

    out = [new_structures]
    if not no_force:
        out += [new_forces]
    if not no_stress:
        out += [new_stresses]
    if len(out) == 1:
        return out[0]
    return out


def stress_matrix_to_list(stress_matrix: np.ndarray, stress_format: str = "VASP") -> np.ndarray:
    """
    Stress matrix to list representation
    Args:
        stress_matrix (np.ndarray): stress matrix 3x3
        stress_format (str): stress list format
    Returns: list of float stress vector.
    """
    vasp_format = np.array(
        [
            stress_matrix[0, 0],
            stress_matrix[1, 1],
            stress_matrix[2, 2],
            stress_matrix[0, 1],
            stress_matrix[1, 2],
            stress_matrix[0, 2],
        ]
    )
    return stress_format_change(vasp_format, "VASP", stress_format)


def stress_list_to_matrix(stress: np.ndarray | list[float], stress_format: str = "VASP") -> np.ndarray:
    """
    Convert a length-6 stress list to stress matrix 3x3.

    Args:
        stress (list of float): list of stress
        stress_format (str): Supported formats are the follows
            VASP: xx, yy, zz, xy, yz, xz
            LAMMPS: xx, yy, zz, xy, zx, yz
            SNAP: xx, yy, zz, yz, xz, xy
    Returns: 3x3 stress matrix
    """
    s = stress_format_change(stress, from_format=stress_format, to_format="VASP")
    xx, yy, zz, xy, yz, xz = s
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def stress_format_change(stress: np.ndarray | list[float], from_format: str, to_format: str) -> np.ndarray:
    """
    Convert stress format from from_format to to_format
    Args:
        stress (list of float): length-6 stress vector
        from_format (str): choose from "VASP", "LAMMPS", "SNAP"
        to_format (str): choose from "VASP", "LAMMPS", "SNAP".

    Returns: list of float stress vector
    """
    from_order = STRESS_FORMAT[from_format.upper()]
    to_order = STRESS_FORMAT[to_format.upper()]
    mapping = [from_order.index(i) for i in to_order[3:]]
    return np.array([stress[0], stress[1], stress[2], *[stress[i] for i in mapping]])


def get_lammps_lattice_and_rotation(structure: Structure, origin=(0, 0, 0)) -> tuple[np.ndarray, SymmOp, np.ndarray]:
    """
    Transform structure to lammps compatible structure. The lattice and rotation
    matrix are returned.

    Args:
        structure (Structure): pymatgen structure
        origin (tuple): origin coordinates

    Returns: new lattice, rotation symmetry operator, rotation matrix

    """
    lattice = structure.lattice
    a, b, c = lattice.abc
    xlo, ylo, zlo = origin
    xhi = a + xlo
    m = lattice.matrix
    xy = np.dot(m[1], m[0] / a)
    yhi = np.sqrt(b**2 - xy**2) + ylo
    xz = np.dot(m[2], m[0] / a)
    yz = (np.dot(m[1], m[2]) - xy * xz) / (yhi - ylo)
    zhi = np.sqrt(c**2 - xz**2 - yz**2) + zlo
    # tilt = None if lattice.is_orthogonal else [xy, xz, yz]
    new_matrix = np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])
    rot_matrix = np.linalg.solve(new_matrix, m)
    symmop = SymmOp.from_rotation_and_translation(rot_matrix, origin)
    return new_matrix, symmop, rot_matrix


def write_data_from_structure(
    structure: Structure,
    filename: str,
    ff_elements: list[str] | None = None,
    significant_figures: int = 6,
    origin: tuple = (0, 0, 0),
):
    """
    Write structure to lammps data file, this is to speed up
    pymatgen LammpsData.

    Args:a
        structure (Structure): pymatgen structure
        filename (str): filename
        ff_elements (list of str): elements to be considered
        significant_figures (int): significant figures of floats in output
        origin (tuple): origin coordinates
    """
    new_matrix, symmop, _rot_matrix = get_lammps_lattice_and_rotation(structure=structure, origin=origin)
    lattice = structure.lattice
    xlo, ylo, zlo = origin
    xhi = new_matrix[0, 0] + xlo
    yhi = new_matrix[1, 1] + ylo
    zhi = new_matrix[2, 2] + zlo
    xy = new_matrix[1, 0]
    xz = new_matrix[2, 0]
    yz = new_matrix[2, 1]

    tilt = None if lattice.is_orthogonal else [xy, xz, yz]
    bounds = [[xlo, xhi], [ylo, yhi], [zlo, zhi]]

    lines = ["# Atom data generated by maml package"]

    elements_in_structure = [str(i.specie) for i in structure.sites]

    if ff_elements is None:
        ff_elements = list(set(elements_in_structure))
    else:
        if not set(elements_in_structure).issubset(set(ff_elements)):
            raise ValueError("Structure contains elements not in ff_elements")

    n_types = len(ff_elements)

    element_map = {i: j + 1 for j, i in enumerate(ff_elements)}
    # generate atom section
    lines.append(f"{len(structure)}    atoms\n")
    lines.append(f"{n_types}    atom types\n")

    ph = f"{{:.{significant_figures}f}}"

    for bound, d in zip(bounds, "xyz", strict=False):
        line = " ".join([ph.format(i) for i in bound] + [f"{d}{i}" for i in ["lo", "hi"]])
        lines.append(line)
    if tilt is not None:
        line = " ".join([ph.format(i) for i in tilt] + ["xy", "xz", "yz"])
        lines.append(line)

    lines.append("\nMasses\n")
    masses = [("{} " + ph).format(j, _get_atomic_mass(i)) for i, j in element_map.items()]
    lines.extend(masses)
    lines.append("\nAtoms\n")

    new_coords = symmop.operate_multi(structure.cart_coords)
    for i, (site, coords) in enumerate(zip(structure.sites, new_coords, strict=False)):
        charge = _get_charge(site.specie)
        line = "{} {} " + " ".join([ph] * 4)
        index = i + 1
        type = element_map[str(site.specie)]
        line = line.format(index, type, charge, *coords)
        lines.append(line)
    with open(filename, "w") as f:
        f.write("\n".join(lines))


def _get_atomic_mass(element_or_specie: str) -> float:  # type: ignore
    """
    Get atomic mass from element or specie string.

    Args:
        element_or_specie (str): specie or element string

    Returns: float mass

    """
    try:
        return Element(element_or_specie).atomic_mass  # type: ignore
    except Exception:
        return Species.from_str(element_or_specie).element.atomic_mass


def _get_charge(element_or_specie: str | Element | Species) -> float:  # type: ignore
    """
    Get charge from element or specie.

    Args:
        element_or_specie (str or Element or Species): element or specie

    Returns: charge float

    """
    if isinstance(element_or_specie, Species):
        return element_or_specie.oxi_state  # type: ignore
    if isinstance(element_or_specie, str):
        try:
            return Species.from_string(element_or_specie).oxi_state  # type: ignore
        except Exception:
            return 0.0
    return 0.0
