"""Crystal-system coverage for LatticePerturbation and WyckoffPerturbation edge cases."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from maml.apps.bowsr.perturbation import (
    LatticePerturbation,
    WyckoffPerturbation,
    crystal_system,
    get_standardized_structure,
    perturbation_mapping,
)


@pytest.mark.parametrize(
    ("int_number", "expected"),
    [
        (1, "triclinic"),
        (2, "triclinic"),
        (3, "monoclinic"),
        (15, "monoclinic"),
        (16, "orthorhombic"),
        (74, "orthorhombic"),
        (75, "tetragonal"),
        (142, "tetragonal"),
        (143, "hexagonal"),
        (146, "rhombohedral"),
        (147, "hexagonal"),
        (148, "rhombohedral"),
        (166, "rhombohedral"),
        (167, "rhombohedral"),
        (168, "hexagonal"),
        (194, "hexagonal"),
        (195, "cubic"),
        (230, "cubic"),
    ],
)
def test_crystal_system(int_number, expected):
    assert crystal_system(int_number) == expected


# (spg, lattice, dims, abc, x, expected perturbation)
FITTING_CASES = {
    "cubic": (225, Lattice.cubic(4.0), (1, 0), [4.0], [0.1], [0.1, 0.1, 0.1, 0, 0, 0]),
    "hexagonal": (194, Lattice.hexagonal(3.0, 5.0), (2, 0), [5.0, 3.0], [0.2, 0.1], [0.1, 0.1, 0.2, 0, 0, 0]),
    "rhombohedral-rhombo-setting": (
        166,
        Lattice.rhombohedral(5.0, 60.0),
        (1, 1),
        [5.0],
        [0.1, 2.0],
        [0.1, 0.1, 0.1, 2.0, 2.0, 2.0],
    ),
    "rhombohedral-hex-setting": (
        166,
        Lattice.hexagonal(3.0, 15.0),
        (2, 0),
        [15.0, 3.0],
        [0.2, 0.1],
        [0.1, 0.1, 0.2, 0, 0, 0],
    ),
    "tetragonal": (123, Lattice.tetragonal(3.0, 5.0), (2, 0), [5.0, 3.0], [0.2, 0.1], [0.1, 0.1, 0.2, 0, 0, 0]),
    "tetragonal-a-eq-c": (123, Lattice.cubic(3.0), (1, 0), [3.0], [0.1], [0.1, 0.1, 0.1, 0, 0, 0]),
    "orthorhombic": (
        62,
        Lattice.orthorhombic(3.0, 4.0, 5.0),
        (3, 0),
        [3.0, 4.0, 5.0],
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3, 0, 0, 0],
    ),
    "monoclinic": (
        14,
        Lattice.monoclinic(3.0, 4.0, 5.0, 100.0),
        (3, 1),
        [3.0, 4.0, 5.0],
        [0.1, 0.2, 0.3, 1.5],
        [0.1, 0.2, 0.3, 0, 1.5, 0],
    ),
    "monoclinic-gamma-unique": (
        14,
        Lattice.from_parameters(3.0, 4.0, 5.0, 90, 90, 110),
        (3, 1),
        [3.0, 4.0, 5.0],
        [0.1, 0.2, 0.3, 1.5],
        [0.1, 0.2, 0.3, 0, 0, 1.5],
    ),
    "monoclinic-orthogonal": (
        14,
        Lattice.orthorhombic(3.0, 4.0, 5.0),
        (3, 0),
        [3.0, 4.0, 5.0],
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3, 0, 0, 0],
    ),
    "triclinic": (
        2,
        Lattice.from_parameters(3.0, 4.0, 5.0, 80, 100, 110),
        (3, 3),
        [3.0, 4.0, 5.0],
        [0.1, 0.2, 0.3, 1, 2, 3],
        [0.1, 0.2, 0.3, 1, 2, 3],
    ),
}


@pytest.mark.parametrize(("spg", "lattice", "dims", "abc", "x", "expected"), FITTING_CASES.values(), ids=FITTING_CASES)
def test_lattice_perturbation_fits(spg, lattice, dims, abc, x, expected):
    lp = LatticePerturbation(spg)
    lp.sanity_check(lattice)
    assert lp.fit_lattice
    assert lp.dims == dims
    assert lp.abc == pytest.approx(abc)
    assert lp.lattice is lattice
    assert np.allclose(lp.perturbation_mode(np.array(x)), expected)
    # The perturbation keeps the lattice within its crystal system.
    params = np.array(lattice.abc + lattice.angles) + lp.perturbation_mode(np.array(x))
    perturbed = LatticePerturbation(spg)
    perturbed.sanity_check(Lattice.from_parameters(*params))
    assert perturbed.fit_lattice
    assert perturbed.dims == dims


NON_FITTING_CASES = {
    "cubic-not-equal": (225, Lattice.tetragonal(4.0, 5.0)),
    "hexagonal-not-hex": (194, Lattice.cubic(3.0)),
    "rhombohedral-orthorhombic": (166, Lattice.orthorhombic(3.0, 4.0, 5.0)),
    "tetragonal-oblique": (123, Lattice.from_parameters(3.0, 3.0, 5.0, 90, 90, 100)),
    "tetragonal-all-different": (123, Lattice.orthorhombic(3.0, 4.0, 5.0)),
    "orthorhombic-oblique": (62, Lattice.monoclinic(3.0, 4.0, 5.0, 100.0)),
    "monoclinic-triclinic": (14, Lattice.from_parameters(3.0, 4.0, 5.0, 80, 100, 110)),
}


@pytest.mark.parametrize(("spg", "lattice"), NON_FITTING_CASES.values(), ids=NON_FITTING_CASES)
def test_lattice_perturbation_does_not_fit(spg, lattice):
    lp = LatticePerturbation(spg)
    lp.sanity_check(lattice)
    assert not lp.fit_lattice
    assert lp.lattice is None
    assert repr(lp).endswith(f"crystal_system={lp.crys_system})\n")


def test_lattice_perturbation_without_symmetry():
    lattice = Lattice.cubic(4.0)
    lp = LatticePerturbation(225, use_symmetry=False)
    lp.sanity_check(lattice)
    assert lp.fit_lattice
    assert lp.dims == (3, 3)
    x = np.arange(6.0)
    assert np.array_equal(lp.perturbation_mode(x), x)
    assert repr(lp).startswith("LatticePerturbation(spg_int_number=225, crystal_system=cubic)\n")
    assert repr(lattice) in repr(lp)


def test_wyckoff_perturbation_without_symmetry():
    wp = WyckoffPerturbation(225, "a", use_symmetry=False)
    assert wp.dim == 3
    assert wp.multiplicity == 1
    assert len(wp.symmetry_ops) == 1
    x = np.array([0.1, 0.2, 0.3])
    assert np.array_equal(wp.perturbation_mode(x), x)
    assert wp.standard_mode(np.array([0.7, 0.2, 0.9]))
    orbit = wp.get_orbit([1.25, -0.5, 0.3])
    assert len(orbit) == 1
    assert np.allclose(orbit[0], [0.25, 0.5, 0.3])


def test_wyckoff_sanity_check_and_repr():
    structure = Structure(Lattice.cubic(4.0), ["Cu"], [[0.1, 0.2, 0.3]])
    wp = WyckoffPerturbation(225, "a")
    assert repr(wp) == "WyckoffPerturbation(spg_int_number=225, wyckoff_symbol=a)"
    # A general position has a 192-fold orbit, not the 4-fold orbit of 4a.
    wp.sanity_check(structure[0])
    assert not wp.fit_site
    assert wp.site is None

    origin = Structure(Lattice.cubic(4.0), ["Cu"], [[0, 0, 0]])
    wp.sanity_check(origin[0])
    assert wp.fit_site
    assert wp.site is origin[0]
    assert repr(wp) == "WyckoffPerturbation(spg_int_number=225, wyckoff_symbol=a) Cu [0.0000, 0.0000, 0.0000]"


@pytest.mark.parametrize(
    ("x", "fixed", "expected"),
    [
        ([1.0, 2.0], [1], [1.0, 0.0, 2.0]),
        ([5.0], [0, 2], [0.0, 5.0, 0.0]),
        ([1.0, 2.0, 3.0], [], [1.0, 2.0, 3.0]),
    ],
)
def test_perturbation_mapping(x, fixed, expected):
    assert np.array_equal(perturbation_mapping(np.array(x), fixed), expected)


def test_get_standardized_structure_from_primitive():
    prim = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.6), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).get_primitive_structure()
    assert prim.num_sites == 2
    std = get_standardized_structure(prim)
    assert std.num_sites == 8
    assert std.lattice.abc == pytest.approx((5.6, 5.6, 5.6))
    assert std.composition.reduced_formula == "NaCl"
    # Sorted by electronegativity: Na first.
    assert [site.specie.symbol for site in std] == ["Na"] * 4 + ["Cl"] * 4
