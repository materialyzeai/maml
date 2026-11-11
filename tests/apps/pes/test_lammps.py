"""Tests for the LAMMPS calculators with the LAMMPS executable mocked out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.lammps.data import LammpsData

from maml.apps.pes import _lammps
from maml.apps.pes._base import _check_returncode
from maml.apps.pes._lammps import (
    DefectFormation,
    ElasticConstant,
    EnergyForceStress,
    LammpsPotential,
    LatticeConstant,
    LMPRelaxationCalculator,
    NudgedElasticBand,
    SpectralNeighborAnalysis,
    SurfaceEnergy,
    _pretty_input,
    _read_dump,
    get_default_lmp_exe,
)

ENERGY_PER_ATOM = -4.5
STRESS = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
DUMP_HEADER = "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{n}\nITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\nITEM: ATOMS\n"


class DummyPotential(LammpsPotential):
    """Minimal LAMMPS-compatible potential used as ff_settings."""

    def __init__(self, elements=("Ni",)):
        self.elements = list(elements)

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def write_param(self):
        return ["pair_style lj/cut 3.0", "pair_coeff * * 1.0 1.0"]

    @staticmethod
    def from_config(**kwargs):
        return DummyPotential()


def _write_dump(filename, rows):
    rows = np.atleast_2d(rows)
    body = "\n".join(" ".join(str(v) for v in row) for row in rows)
    Path(filename).write_text(DUMP_HEADER.format(n=len(rows)) + body + "\n")


def _write_efs(n_atoms, extra_energy=0.0):
    Path("energy.txt").write_text(f"{ENERGY_PER_ATOM * n_atoms + extra_energy}\n")
    _write_dump("force.dump", np.tile([0.1, -0.2, 0.3], (n_atoms, 1)))
    Path("stress.txt").write_text(" ".join(map(str, STRESS)) + "\n")


def _remove_site(data_file, site_id, out_file):
    struct = LammpsData.from_file(data_file, atom_style="charge").structure
    struct.remove_sites([site_id - 1])
    LammpsData.from_structure(struct, atom_style="charge").write_file(out_file)


def make_lammps_handler(relax_scale=1.0, defect_energy=1.5, barrier=0.75, efs_extra=None):
    """Fake LAMMPS that writes the files each input script would produce."""

    def handler(args):
        if args[0] == "mpirun":
            Path("log.lammps").write_text("Step MaxReplicaForce MaxAtomForce GradV0 GradV1 GradVc EBF EBR RDT\n")
            with open("log.lammps", "a") as f:
                f.write(f"100 0.001 0.001 0.0 0.0 0.0 {barrier} 0.5 3.0\n")
            return
        input_file = args[args.index("-in") + 1]
        text = Path(input_file).read_text()
        if input_file == "in.efs":
            struct = LammpsData.from_file("data.static", atom_style="charge").structure
            _write_efs(len(struct), extra_energy=efs_extra(struct) if efs_extra else 0.0)
        elif input_file == "in.defect":
            n_atoms = len(LammpsData.from_file("data.supercell", atom_style="charge").structure) - 1
            _write_efs(n_atoms, extra_energy=defect_energy)
        elif input_file == "in.relax" and "data.supercell" in text:  # NEB end-point relaxation
            del_id = int(
                next(line.split()[-1] for line in text.splitlines() if line.startswith("group") and "del" in line)
            )
            out = next(line.split()[-1] for line in text.splitlines() if line.startswith("write_data"))
            _remove_site("data.supercell", del_id, out)
        elif input_file == "in.relax":
            struct = LammpsData.from_file("data.static", atom_style="charge").structure
            struct.scale_lattice(struct.volume * relax_scale**3)
            LammpsData.from_structure(struct, atom_style="charge").write_file("data.relaxed")
        elif input_file == "in.elastic":
            Path("elastic.txt").write_text("276.0 159.0 132.0 198.0\n")
            Path("voigt_tensor.txt").write_text("\n".join(" ".join(str(i * 6 + j) for j in range(6)) for i in range(6)))
        elif input_file == "in.sna":
            n_atoms = len(LammpsData.from_file("data.static", atom_style="charge").structure)
            _write_dump("dump.element", [["Na"]] * (n_atoms // 2) + [["Cl"]] * (n_atoms - n_atoms // 2))
            _write_dump("dump.sna", np.ones((n_atoms, 4)))
            _write_dump("dump.snad", np.ones((n_atoms, 3)))
            _write_dump("dump.snav", np.ones((n_atoms, 6)))
        else:
            raise AssertionError(f"Unexpected input file {input_file}")
        return

    return handler


@pytest.fixture
def fake_lammps(mock_binaries, in_tmp):
    def install(handler=None, available=None):
        return mock_binaries(_lammps, handler or make_lammps_handler(), available=available)

    return install


@pytest.fixture
def ni_fcc():
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.5), ["Ni"], [[0, 0, 0]])


def test_get_default_lmp_exe(monkeypatch):
    monkeypatch.setattr(_lammps, "which", lambda cmd: "/opt/bin/lmp_mpi" if cmd == "lmp_mpi" else None)
    assert get_default_lmp_exe() == "/opt/bin/lmp_mpi"
    monkeypatch.setattr(_lammps, "which", lambda cmd: None)
    assert get_default_lmp_exe() is None


def test_no_lammps_raises_value_error(fake_lammps):
    # Regression: with no LAMMPS on PATH, which(None) used to raise an opaque TypeError.
    fake_lammps(available=set())
    with pytest.raises(ValueError, match="lammps executable None not found"):
        EnergyForceStress(ff_settings=["pair_style lj/cut 3"])
    with pytest.raises(ValueError, match="lmp_custom not found"):
        EnergyForceStress(ff_settings=["pair_style lj/cut 3"], lmp_exe="lmp_custom")


def test_invalid_kwarg(fake_lammps):
    fake_lammps()
    with pytest.raises(TypeError, match="not in supported kwargs"):
        EnergyForceStress(ff_settings=[], foo=1)
    calc = EnergyForceStress(ff_settings=[], lmp_exe="lmp_serial")
    calc.set_lmp_exe("lmp_other")
    assert calc.LMP_EXE == "lmp_other"


def test_pretty_input():
    out = _pretty_input(["units metal", "# a comment", "", "pair_coeff * * 1 1\n"]).split("\n")
    width = len("pair_coeff") + 4
    assert out[0] == "units".ljust(width) + "metal"
    assert out[1] == "# a comment"
    assert out[2] == ""
    assert out[3] == "pair_coeff".ljust(width) + "* * 1 1"


def test_read_dump(in_tmp):
    # Regression: the default dtype "float_" is not understood by numpy >= 2.
    _write_dump("test.dump", [[1.0, 2.0], [3.0, 4.0]])
    data = _read_dump("test.dump")
    np.testing.assert_allclose(data, [[1.0, 2.0], [3.0, 4.0]])
    assert data.dtype == np.float64


@pytest.mark.parametrize("ff_settings", [["pair_style lj/cut 3.0", "pair_coeff * * 1.0 1.0"], DummyPotential()])
def test_energy_force_stress(fake_lammps, ff_settings, ni_fcc):
    calls = fake_lammps()
    calc = EnergyForceStress(ff_settings=ff_settings)
    ((energy, forces, stress),) = calc.calculate([ni_fcc])
    assert calls == [["/fake/bin/lmp_serial", "-in", "in.efs"]]
    assert energy == pytest.approx(ENERGY_PER_ATOM * len(ni_fcc))
    # cubic cell -> no rotation is applied
    np.testing.assert_allclose(forces, np.tile([0.1, -0.2, 0.3], (len(ni_fcc), 1)))
    np.testing.assert_allclose(stress, STRESS)
    assert "pair_style lj/cut 3.0\npair_coeff * * 1.0 1.0" in calls.file("in.efs")


def test_energy_force_stress_does_not_mutate_input(fake_lammps):
    # Regression: calculate() used to strip oxidation states from the caller's structures.
    fake_lammps()
    struct = Structure(Lattice.cubic(5.6), ["Na+", "Cl-"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    EnergyForceStress(ff_settings=["pair_style lj/cut 3"]).calculate([struct])
    assert [str(sp) for sp in struct.species] == ["Na+", "Cl-"]


def test_energy_force_stress_rotates_triclinic(fake_lammps):
    fake_lammps()
    struct = Structure(Lattice.from_parameters(3, 4, 5, 70, 80, 100), ["Ni", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    ((_, forces, stress),) = EnergyForceStress(ff_settings=[]).calculate([struct])
    # Rotation preserves the force norms and the stress trace
    np.testing.assert_allclose(np.linalg.norm(forces, axis=1), np.linalg.norm([0.1, -0.2, 0.3]))
    assert sum(stress[:3]) == pytest.approx(sum(STRESS[:3]))
    assert not np.allclose(forces[0], [0.1, -0.2, 0.3])


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("LAMMPS (29 Oct 2020)\nERROR: Unknown pair style\nLast command: pair_style foo\n", "ERROR: Unknown pair"),
        ("LAMMPS (29 Oct 2020)\nSomething went wrong\n", "Something went wrong"),
        ("", "return code 1$"),
    ],
)
def test_lammps_failure(fake_lammps, ni_fcc, stdout, expected):
    # Regression: empty stdout used to raise IndexError instead of RuntimeError.
    fake_lammps(lambda args: (1, stdout, ""))
    with pytest.raises(RuntimeError, match=expected):
        EnergyForceStress(ff_settings=[]).calculate([ni_fcc])


def test_check_returncode_success():
    assert _check_returncode("prog", 0, None) is None
    with pytest.raises(RuntimeError, match="prog exited with return code 2: ERROR: a, details"):
        _check_returncode("prog", 2, b"ok\nERROR: a\ndetails\n")


def test_spectral_neighbor_analysis(fake_lammps):
    calls = fake_lammps()
    profile = {"Na": {"r": 0.3, "w": 0.9}, "Cl": {"r": 0.7, "w": 3.0}}
    calc = SpectralNeighborAnalysis(rcutfac=5, twojmax=3, element_profile=profile, quadratic=True)
    # Known numbers of bispectrum components for common band limits
    assert len(calc.get_bs_subscripts(6)) == 30
    assert len(calc.get_bs_subscripts(8)) == 55
    assert calc.n_bs == len(calc.get_bs_subscripts(3))
    for j1, j2, j in calc.get_bs_subscripts(3):
        assert j1 >= j2
        assert j >= j1
        assert (j - (j1 - j2)) % 2 == 0
    struct = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.69), ["Na", "Cl"], [[0, 0, 0], [0, 0, 0.5]])
    b, db, vb, elements = calc.calculate([struct])[0]
    assert b.shape[0] == db.shape[0] == vb.shape[0] == len(struct)
    assert sorted(set(elements)) == ["Cl", "Na"]
    text = calls.file("in.sna")
    # elements sorted by electronegativity order of Element: Na before Cl; cutoffs = r * rcutfac
    assert "1.5 3.5 0.9 3.0 rmin0 0 quadraticflag 1 bzeroflag 0" in text
    assert "dump_modify 1 element Na Cl" in " ".join(text.split())


def test_sna_sanity_check(fake_lammps):
    fake_lammps()
    calc = SpectralNeighborAnalysis(rcutfac=5, twojmax=2, element_profile={"Si": {"r": 0.5, "w": 1}})
    struct = Structure(Lattice.cubic(5), ["Na"], [[0, 0, 0]])
    with pytest.raises(AssertionError, match="Incompatible structure"):
        calc.calculate([struct])


@pytest.mark.parametrize("full_matrix", [False, True])
def test_elastic_constant(fake_lammps, ni_fcc, full_matrix):
    calls = fake_lammps()
    calc = ElasticConstant(
        ff_settings=DummyPotential(), potential_type="internal", deformation_size=1e-3, full_matrix=full_matrix
    )
    result = calc.calculate([ni_fcc])[0]
    if full_matrix:
        assert result.shape == (6, 6)
        assert result[1, 2] == 8
    else:
        np.testing.assert_allclose(result, [276.0, 159.0, 132.0, 198.0])
    assert calls[0][-1] == "in.elastic"
    assert "0.001" in calls.file("init.mod")
    assert "pair_style lj/cut 3.0" in calls.file("potential.mod")
    assert "restart.equil" in calls.file("displace.mod")


def test_elastic_constant_bad_potential_type(fake_lammps):
    fake_lammps()
    with pytest.raises(KeyError):
        ElasticConstant(ff_settings=[], potential_type="unknown")


@pytest.mark.parametrize(
    ("box_relax", "box_triclinic", "expected", "unexpected"),
    [
        (True, True, ["fix   1 all box/relax aniso 0.0 vmax 0.001", "change_box all triclinic"], []),
        (False, False, ["min_style         fire"], ["box/relax", "change_box"]),
    ],
)
def test_relaxation_calculator(fake_lammps, ni_fcc, box_relax, box_triclinic, expected, unexpected):
    calls = fake_lammps(make_lammps_handler(relax_scale=1.01))
    calc = LMPRelaxationCalculator(
        ff_settings=DummyPotential(), box_relax=box_relax, box_triclinic=box_triclinic, min_style="fire"
    )
    final_structure, energy, forces, stress = calc.calculate([ni_fcc])[0]
    assert final_structure.lattice.a == pytest.approx(3.5 * 1.01)
    assert energy == pytest.approx(ENERGY_PER_ATOM * len(ni_fcc))
    assert forces.shape == (len(ni_fcc), 3)
    assert len(stress) == 6
    assert [c[-1] for c in calls] == ["in.relax", "in.efs"]
    text = calls.file("in.relax")
    for line in expected:
        assert line in text
    for line in unexpected:
        assert line not in text


def test_lattice_constant(fake_lammps, ni_fcc):
    fake_lammps(make_lammps_handler(relax_scale=1.02))
    ((a, b, c),) = LatticeConstant(ff_settings=DummyPotential()).calculate([ni_fcc])
    np.testing.assert_allclose([a, b, c], [3.5 * 1.02] * 3)


@pytest.mark.parametrize(("lattice", "n_sites"), [("fcc", 4), ("bcc", 2), ("diamond", 8)])
@pytest.mark.parametrize("cls", [NudgedElasticBand, DefectFormation])
def test_get_unit_cell(cls, lattice, n_sites):
    cell = cls.get_unit_cell(specie="Ni", lattice=lattice, alat=3.5)
    assert len(cell) == n_sites
    assert cell.lattice.abc == pytest.approx((3.5, 3.5, 3.5))


@pytest.mark.parametrize("cls", [NudgedElasticBand, DefectFormation])
def test_get_unit_cell_invalid(cls):
    with pytest.raises(ValueError, match="Lattice type is invalid"):
        cls.get_unit_cell(specie="Ni", lattice="sc", alat=3.5)


@pytest.mark.parametrize(("lattice", "n_supercell"), [("fcc", 108), ("bcc", 54), ("diamond", 64)])
def test_defect_formation(fake_lammps, lattice, n_supercell):
    calls = fake_lammps(make_lammps_handler(relax_scale=1.01, defect_energy=1.5))
    calc = DefectFormation(ff_settings=DummyPotential(), specie="Ni", lattice=lattice, alat=3.5)
    assert calc.calculate() == pytest.approx(1.5)
    assert [c[-1] for c in calls] == ["in.relax", "in.efs", "in.efs", "in.defect"]
    text = calls.file("in.defect")
    assert f"lattice         {lattice} {3.5 * 1.01}" in text
    assert f"{n_supercell}  atoms" in calls.file("data.supercell")


def test_defect_formation_invalid_lattice(fake_lammps):
    fake_lammps()
    calc = DefectFormation(ff_settings=DummyPotential(), specie="Ni", lattice="sc", alat=3.5)
    with pytest.raises(ValueError, match="Lattice type is invalid"):
        calc.calculate()


def test_nested_calculators_use_given_lmp_exe(fake_lammps):
    # Regression: inner LatticeConstant/EnergyForceStress ignored the user-supplied lmp_exe
    # and fell back to auto-detection.
    calls = fake_lammps(make_lammps_handler(), available={"lmp_custom"})
    calc = DefectFormation(ff_settings=DummyPotential(), specie="Ni", lattice="fcc", alat=3.5, lmp_exe="lmp_custom")
    calc.calculate()
    assert {c[0] for c in calls} == {"lmp_custom"}


@pytest.mark.parametrize(("lattice", "start_idx"), [("fcc", 95), ("bcc", 40), ("diamond", 7)])
def test_nudged_elastic_band(fake_lammps, lattice, start_idx):
    calls = fake_lammps(make_lammps_handler(barrier=0.75))
    calc = NudgedElasticBand(ff_settings=DummyPotential(), specie="Ni", lattice=lattice, alat=3.5, num_replicas=3)
    assert calc.calculate() == pytest.approx(0.75)
    assert calls[-1] == ["mpirun", "-n", "3", "/fake/bin/lmp_serial", "-partition", "3x1", "-in", "in.neb"]
    assert [c[-1] for c in calls[:-1]] == ["in.relax", "in.efs", "in.relax", "in.relax"]
    final_replica = calls.file("data.final_replica").splitlines()
    n_sites = int(final_replica[0])
    assert len(final_replica) == n_sites + 1
    # The final replica lists every atom of the initial state (all ids except the initial vacancy) once
    ids = sorted(int(line.split()[0]) for line in final_replica[1:])
    assert ids == [i for i in range(1, n_sites + 2) if i != start_idx + 1]
    assert "initial.relaxed" in calls.file("in.neb")


def test_nudged_elastic_band_failure(fake_lammps):
    def handler(args):
        if args[0] == "mpirun":
            return 1, "ERROR on proc 0: NEB failed\n", ""
        return make_lammps_handler()(args)

    fake_lammps(handler)
    calc = NudgedElasticBand(ff_settings=DummyPotential(), specie="Ni", lattice="fcc", alat=3.5)
    with pytest.raises(RuntimeError, match="NEB failed"):
        calc.calculate()


def test_nudged_elastic_band_relax_failure(fake_lammps):
    def handler(args):
        if args[-1] == "in.relax" and "data.supercell" in Path("in.relax").read_text():
            return 1, "ERROR: Lost atoms\n", ""
        return make_lammps_handler()(args)

    fake_lammps(handler)
    calc = NudgedElasticBand(ff_settings=DummyPotential(), specie="Ni", lattice="fcc", alat=3.5)
    with pytest.raises(RuntimeError, match="Lost atoms"):
        calc.calculate()


def test_nudged_elastic_band_invalid_lattice(fake_lammps):
    fake_lammps()
    calc = NudgedElasticBand(ff_settings=DummyPotential(), specie="Ni", lattice="sc", alat=3.5)
    with pytest.raises(ValueError, match="Lattice type is invalid"):
        calc.calculate()


def test_defect_formation_failure(fake_lammps):
    def handler(args):
        if args[-1] == "in.defect":
            return 3, "ERROR: Bad defect\n", ""
        return make_lammps_handler()(args)

    fake_lammps(handler)
    calc = DefectFormation(ff_settings=DummyPotential(), specie="Ni", lattice="fcc", alat=3.5)
    with pytest.raises(RuntimeError, match="return code 3: ERROR: Bad defect"):
        calc.calculate()


def test_surface_energy(fake_lammps, ni_fcc):
    from pymatgen.core.units import Unit

    # Slabs (with vacuum along c) get 2 eV on top of the extensive bulk energy.
    fake_lammps(make_lammps_handler(efs_extra=lambda s: 2.0 if s.lattice.c > 10 else 0.0))
    calc = SurfaceEnergy(
        ff_settings=DummyPotential(), bulk_structure=ni_fcc, miller_indexes=[[1, 0, 0], [1, 1, 1]], min_slab_size=5
    )
    assert calc.bulk_energy_per_atom == pytest.approx(ENERGY_PER_ATOM)
    assert not calc.box_relax
    assert len(calc.slabs) == 2
    results = calc.calculate()
    assert [r[0] for r in results] == [[1, 0, 0], [1, 1, 1]]
    ev_per_ang2_to_j_per_m2 = Unit("eV").get_conversion_factor("J") / Unit("ang").get_conversion_factor("m") ** 2
    for (_, slab, surface_energy), area in zip(results, calc.surface_areas, strict=True):
        assert isinstance(slab, Structure)
        assert surface_energy == pytest.approx(2.0 / 2 / area * ev_per_ang2_to_j_per_m2)


def test_lammps_potential_predict_efs(fake_lammps, ni_fcc):
    fake_lammps()
    energy, forces, stress = DummyPotential().predict_efs(ni_fcc)
    assert energy == pytest.approx(ENERGY_PER_ATOM * len(ni_fcc))
    assert forces.shape == (len(ni_fcc), 3)
    np.testing.assert_allclose(stress, STRESS)
