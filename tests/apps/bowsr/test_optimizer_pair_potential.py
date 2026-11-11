"""Tests for BayesianOptimizer driven by a cheap pair-potential energy model.

These run the full optimize/propose/get_optimized_structure loop without
requiring MEGNet, CGCNN or VASP.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic

import maml.apps.bowsr.model.cgcnn as cgcnn_module
import maml.apps.bowsr.model.dft as dft_module
import maml.apps.bowsr.model.megnet as megnet_module
import maml.apps.bowsr.optimizer as optimizer_module
from maml.apps.bowsr.acquisition import AcquisitionFunction
from maml.apps.bowsr.model.base import EnergyModel
from maml.apps.bowsr.optimizer import BayesianOptimizer, atoms_crowded, struct2perturbation
from maml.apps.bowsr.preprocessing import DummyScaler, StandardScaler


class LennardJones(EnergyModel):
    """Species-agnostic Lennard-Jones pair potential, in eV/atom."""

    def __init__(self, r0: float = 2.8, eps: float = 1.0, cutoff: float = 6.0):
        self.r0 = r0
        self.eps = eps
        self.cutoff = cutoff
        self.n_calls = 0

    def predict_energy(self, structure: Structure) -> float:
        self.n_calls += 1
        _, _, _, dists = structure.get_neighbor_list(self.cutoff)
        x6 = (self.r0 / dists) ** 6
        return float(self.eps * np.sum(x6 * x6 - 2 * x6) / 2 / len(structure))


class FailingModel(EnergyModel):
    def predict_energy(self, structure: Structure) -> float:
        raise RuntimeError("model blew up")


@pytest.fixture
def nacl() -> Structure:
    """Primitive rock-salt NaCl (Fm-3m, all Wyckoff positions fixed)."""
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.9), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture
def zno() -> Structure:
    """Wurtzite ZnO (P6_3mc), both sites on 2b with a free z coordinate."""
    return Structure.from_spacegroup(
        "P6_3mc", Lattice.hexagonal(3.25, 5.2), ["Zn", "O"], [[1 / 3, 2 / 3, 0], [1 / 3, 2 / 3, 0.38]]
    )


@pytest.fixture
def model() -> LennardJones:
    return LennardJones()


def nacl_lj_optimum(model: LennardJones, nacl: Structure) -> float:
    """Brute-force the LJ-optimal cubic lattice constant of NaCl."""
    grid = np.linspace(4.7, 7.1, 241)
    energies = [model.predict_energy(Structure(Lattice.cubic(a), nacl.species, nacl.frac_coords)) for a in grid]
    return float(grid[int(np.argmin(energies))])


def same_rows(coords1: np.ndarray, coords2: np.ndarray) -> bool:
    """Whether two sets of fractional coordinates are equal up to row order."""

    def canonical(coords):
        coords = np.round(np.asarray(coords) % 1, 6) % 1
        return coords[np.lexsort(coords.T[::-1])]

    return bool(np.allclose(canonical(coords1), canonical(coords2), atol=1e-5))


def test_struct2perturbation_symmetric(nacl):
    conv = SpacegroupAnalyzer(nacl).get_conventional_standard_structure()
    wps, indices, mapping, lp = struct2perturbation(conv)
    assert len(wps) == len(indices) == 2
    assert [wp.dim for wp in wps] == [0, 0]
    assert all(wp.fit_site for wp in wps)
    assert len(mapping) == conv.num_sites == 8
    assert set(mapping.values()) == set(indices)
    assert lp.crys_system == "cubic"
    assert lp.dims == (1, 0)
    assert lp.fit_lattice


def test_struct2perturbation_no_symmetry(zno):
    wps, indices, mapping, lp = struct2perturbation(zno, use_symmetry=False)
    assert indices == list(range(zno.num_sites))
    assert mapping == {i: i for i in range(zno.num_sites)}
    assert [wp.dim for wp in wps] == [3] * zno.num_sites
    assert lp.dims == (3, 3)


@pytest.mark.parametrize(("cutoff", "expected"), [(2.9, False), (3.0, True)])
def test_atoms_crowded(nacl, cutoff, expected):
    # Nearest-neighbour Na-Cl distance in rock salt with a = 5.9 is 2.95 A.
    assert atoms_crowded(nacl, cutoff_distance=cutoff) is expected


def test_atoms_crowded_single_atom():
    assert not atoms_crowded(Structure(Lattice.cubic(3.0), ["Cu"], [[0, 0, 0]]))


def test_init_standardizes_and_does_not_mutate_input(nacl, model):
    nacl = nacl.get_primitive_structure()
    nacl.add_oxidation_state_by_element({"Na": 1, "Cl": -1})
    optimizer = BayesianOptimizer(model, nacl, seed=0)
    # Primitive cell is converted to the conventional standard cell.
    assert optimizer.structure.num_sites == 8
    assert optimizer.structure.lattice.abc == pytest.approx((5.9, 5.9, 5.9))
    assert optimizer.wyckoff_dims == [0, 0]
    assert (optimizer.abc_dim, optimizer.angles_dim) == (1, 0)
    assert optimizer.space.dim == 1
    assert all(getattr(sp, "oxi_state", None) is None for sp in optimizer.structure.species)
    # Regression: the caller's structure must keep its oxidation states.
    assert nacl.num_sites == 2
    assert {sp.oxi_state for sp in nacl.species} == {1, -1}


def test_init_honours_use_symmetry(monkeypatch, zno, model):
    """Regression: the initial sanity check ignored use_symmetry."""
    seen = []
    orig = optimizer_module.struct2perturbation

    def spy(structure, use_symmetry=True, **kwargs):
        seen.append(use_symmetry)
        return orig(structure, use_symmetry, **kwargs)

    monkeypatch.setattr(optimizer_module, "struct2perturbation", spy)
    optimizer = BayesianOptimizer(model, zno, use_symmetry=False, seed=0)
    assert seen
    assert not any(seen)
    assert optimizer.space.dim == 3 * zno.num_sites + 6


def test_init_raises_if_not_standardizable(monkeypatch, nacl, model):
    class NeverFits:
        fit_site = False
        fit_lattice = False
        dims = (1, 0)

    monkeypatch.setattr(
        optimizer_module, "struct2perturbation", lambda *args, **kwargs: ([NeverFits()], [0], {0: 0}, NeverFits())
    )
    with pytest.raises(ValueError, match="Standard structures can not be obtained"):
        BayesianOptimizer(model, nacl)


def test_no_degrees_of_freedom_raises(nacl, model):
    with pytest.raises(ValueError, match="No degrees of freedom"):
        BayesianOptimizer(model, nacl, relax_coords=False, relax_lattice=False)


@pytest.mark.parametrize(
    ("relax_coords", "relax_lattice", "dim"),
    [(True, True, 4), (True, False, 2), (False, True, 2)],
)
def test_space_dims(zno, model, relax_coords, relax_lattice, dim):
    optimizer = BayesianOptimizer(model, zno, relax_coords=relax_coords, relax_lattice=relax_lattice, seed=0)
    assert optimizer.wyckoff_dims == [1, 1]
    assert (optimizer.abc_dim, optimizer.angles_dim) == (2, 0)
    assert optimizer.space.dim == dim


def test_get_derived_structure_identity(zno, model):
    optimizer = BayesianOptimizer(model, zno, seed=0)
    derived = optimizer.get_derived_structure(np.zeros(optimizer.space.dim))
    assert StructureMatcher().fit(derived, optimizer.structure)
    assert derived.lattice.abc == pytest.approx(optimizer.structure.lattice.abc)


def test_get_derived_structure_preserves_symmetry(zno, model):
    optimizer = BayesianOptimizer(model, zno, seed=0)
    x = np.array([0.0, 0.03, 0.0, 0.0])  # shift z of the second Wyckoff orbit
    derived = optimizer.get_derived_structure(x)
    assert SpacegroupAnalyzer(derived, symprec=1e-3).get_space_group_number() == 186
    # The atoms of the perturbed orbit move rigidly together along c (up to a
    # permutation of the symmetry-equivalent atoms within the orbit).
    idx = optimizer.indices[1]
    orbit = [k for k, v in optimizer.mapping.items() if v == idx]
    z_orig = np.sort(optimizer.structure.frac_coords[orbit, 2] % 1)
    z_new = np.sort(derived.frac_coords[orbit, 2] % 1)
    assert any(np.allclose(np.sort((z_orig + shift) % 1), z_new, atol=1e-6) for shift in (0.03, -0.03))
    assert same_rows(derived.frac_coords[orbit, :2], optimizer.structure.frac_coords[orbit, :2])
    others = [k for k in range(derived.num_sites) if k not in orbit]
    assert same_rows(derived.frac_coords[others], optimizer.structure.frac_coords[others])


def test_get_derived_structure_lattice(zno, model):
    optimizer = BayesianOptimizer(model, zno, seed=0)
    a, _, c = optimizer.structure.lattice.abc
    # lp.abc orders the two free lengths as [unique axis, repeated axis].
    assert optimizer.lp.abc == pytest.approx([c, a])
    derived = optimizer.get_derived_structure(np.array([0.0, 0.0, 0.2, -0.1]))
    assert derived.lattice.abc == pytest.approx((a - 0.1, a - 0.1, c + 0.2))
    assert derived.lattice.angles == pytest.approx(optimizer.structure.lattice.angles)


def test_fixed_coords_and_fixed_lattice(zno, model):
    fixed_coords = BayesianOptimizer(model, zno, relax_coords=False, seed=0)
    derived = fixed_coords.get_derived_structure(np.array([0.1, 0.1]))
    assert np.allclose(derived.frac_coords, fixed_coords.structure.frac_coords)
    assert derived.lattice.a == pytest.approx(fixed_coords.structure.lattice.a + 0.1)

    fixed_lattice = BayesianOptimizer(model, zno, relax_lattice=False, seed=0)
    derived = fixed_lattice.get_derived_structure(np.array([0.02, 0.02]))
    assert derived.lattice.abc == pytest.approx(fixed_lattice.structure.lattice.abc)
    assert not np.allclose(derived.frac_coords, fixed_lattice.structure.frac_coords)


def test_get_formation_energy_is_negated_model_energy(zno, model):
    optimizer = BayesianOptimizer(model, zno, seed=0)
    x = np.array([0.01, -0.01, 0.1, 0.05])
    assert optimizer.get_formation_energy(x) == pytest.approx(-model.predict_energy(optimizer.get_derived_structure(x)))


def test_add_query_registers_point(zno, model):
    optimizer = BayesianOptimizer(model, zno, use_scaler=False, seed=0)
    optimizer.set_bounds()
    x = optimizer.space.uniform_sample()
    target = optimizer.add_query(x)
    assert len(optimizer.space) == 1
    assert target == pytest.approx(optimizer.get_formation_energy(x))
    assert np.allclose(optimizer.space.params[0], x)
    optimizer.set_space_empty()
    assert len(optimizer.space) == 0


def test_add_query_swallows_model_errors(zno):
    optimizer = BayesianOptimizer(FailingModel(), zno, seed=0)
    assert optimizer.add_query(np.zeros(optimizer.space.dim)) == 0
    assert len(optimizer.space) == 0


def test_set_bounds(zno, model):
    optimizer = BayesianOptimizer(model, zno, seed=0)
    c, a = optimizer.lp.abc

    optimizer.set_bounds()
    assert np.allclose(optimizer.space.bounds[:2], [[-0.2, 0.2]] * 2)
    assert np.allclose(optimizer.space.bounds[2:], [[-0.2 * c, 0.2 * c], [-0.2 * a, 0.2 * a]])

    optimizer.set_bounds(abc_ratio=0.1, element_wise_wyckoff_bounds={"Zn": 0.05})
    zn_first = optimizer.wps[0].site.specie.symbol == "Zn"
    expected_wyc = [[-0.05, 0.05], [-0.2, 0.2]] if zn_first else [[-0.2, 0.2], [-0.05, 0.05]]
    assert np.allclose(optimizer.space.bounds[:2], expected_wyc)
    assert np.allclose(optimizer.space.bounds[2:], [[-0.1 * c, 0.1 * c], [-0.1 * a, 0.1 * a]])

    optimizer.set_bounds(abc_bound=0.5)
    assert np.allclose(optimizer.space.bounds[2:], [[-0.5, 0.5]] * 2)


def test_set_gpr_params(nacl, model):
    optimizer = BayesianOptimizer(model, nacl, seed=0)
    assert isinstance(optimizer.gpr.kernel, RationalQuadratic)
    optimizer.set_gpr_params(kernel=Matern(length_scale=2.0))
    assert isinstance(optimizer.gpr.kernel, Matern)
    assert optimizer.gpr.kernel.length_scale == 2.0


@pytest.mark.parametrize("use_scaler", [True, False])
@pytest.mark.parametrize("sampler", ["lhs", "uniform"])
def test_optimize_nacl_lattice(nacl, model, use_scaler, sampler):
    a_opt = nacl_lj_optimum(model, nacl)
    optimizer = BayesianOptimizer(model, nacl, use_scaler=use_scaler, seed=42)
    assert isinstance(optimizer.scaler, StandardScaler if use_scaler else DummyScaler)
    optimizer.set_bounds()
    optimizer.optimize(n_init=5, n_iter=5, n_warmup=200, sampler=sampler)

    assert len(optimizer.space) == 1 + 5 + 5
    # The unperturbed structure is always the first query.
    assert optimizer.space.target[0] == pytest.approx(-model.predict_energy(optimizer.structure))

    structure, energy = optimizer.get_optimized_structure_and_energy()
    assert energy == pytest.approx(model.predict_energy(structure))
    assert energy == pytest.approx(-optimizer.space.target.max())
    assert energy < model.predict_energy(optimizer.structure)
    assert abs(structure.lattice.a - a_opt) < abs(5.9 - a_opt)
    # Cubic symmetry is kept.
    assert structure.lattice.abc == pytest.approx((structure.lattice.a,) * 3)
    assert SpacegroupAnalyzer(structure).get_space_group_number() == 225

    # All evaluated points lie within the search box.
    params = np.asarray(optimizer.scaler.inverse_transform(optimizer.space.params))
    bounds = optimizer.space.bounds
    assert np.all(params >= bounds[:, 0] - 1e-9)
    assert np.all(params <= bounds[:, 1] + 1e-9)


@pytest.mark.parametrize("acq_type", ["ucb", "ei", "poi", "gp-ucb"])
@pytest.mark.parametrize("noisy", [True, False])
def test_optimize_zno(zno, model, acq_type, noisy):
    optimizer = BayesianOptimizer(model, zno, noisy=noisy, seed=0)
    optimizer.set_bounds()
    optimizer.optimize(n_init=4, n_iter=3, acq_type=acq_type, xi=0.01, n_warmup=100)
    assert len(optimizer.space) == 8
    structure, energy = optimizer.get_optimized_structure_and_energy()
    assert energy <= model.predict_energy(optimizer.structure) + 1e-12
    assert SpacegroupAnalyzer(structure, symprec=1e-3).get_space_group_number() == 186


def test_optimize_continue_and_restart(zno, model):
    optimizer = BayesianOptimizer(model, zno, seed=0)
    optimizer.set_bounds()
    optimizer.optimize(n_init=3, n_iter=2, n_warmup=50)
    assert len(optimizer.space) == 6
    gpr = optimizer.gpr
    optimizer.optimize(n_init=0, n_iter=2, n_warmup=50, is_continue=True)
    assert optimizer.gpr is gpr
    assert len(optimizer.space) == 8
    optimizer.optimize(n_init=0, n_iter=1, n_warmup=50, alpha=1e-4)
    assert optimizer.gpr is not gpr
    assert optimizer.gpr.alpha == 1e-4
    assert len(optimizer.space) == 9


def test_propose_within_bounds(zno, model):
    optimizer = BayesianOptimizer(model, zno, seed=1)
    optimizer.set_bounds()
    optimizer.optimize(n_init=5, n_iter=0)
    acq = AcquisitionFunction(acq_type="ei", kappa=2.576, xi=0.0)
    for sampler in ("lhs", "uniform"):
        x_next = optimizer.propose(acq, n_warmup=100, sampler=sampler)
        assert x_next.shape == (optimizer.space.dim,)
        assert np.all(optimizer.space.bounds[:, 0] <= x_next)
        assert np.all(x_next <= optimizer.space.bounds[:, 1])


def test_get_optimized_structure_skips_crowded(zno, model):
    optimizer = BayesianOptimizer(model, zno, use_scaler=False, seed=0)
    optimizer.set_bounds()
    small = np.array([0.0, 0.0, -0.5, -1.2])  # compressed, hence low LJ-energy is not guaranteed
    big = np.array([0.0, 0.0, 0.0, 0.0])
    optimizer.add_query(big)
    optimizer.add_query(small)
    # Force the compressed structure to look best.
    optimizer.space._target = np.array([-1.0, 10.0])
    crowded = optimizer.get_derived_structure(small)
    min_dist = np.min(crowded.distance_matrix[np.triu_indices(crowded.num_sites, 1)])

    structure, energy = optimizer.get_optimized_structure_and_energy(cutoff_distance=0)
    assert energy == -10.0
    assert structure.lattice.abc == pytest.approx(crowded.lattice.abc)

    structure, energy = optimizer.get_optimized_structure_and_energy(cutoff_distance=min_dist + 1e-3)
    assert energy == 1.0
    assert structure.lattice.abc == pytest.approx(optimizer.structure.lattice.abc)


@pytest.mark.xfail(
    reason="When every candidate is 'crowded', get_optimized_structure_and_energy falls through the loop and "
    "returns the *worst* (highest-energy) candidate instead of the best one or an error.",
    strict=True,
)
def test_get_optimized_structure_all_crowded_returns_best(zno, model):
    optimizer = BayesianOptimizer(model, zno, use_scaler=False, seed=0)
    optimizer.set_bounds()
    optimizer.optimize(n_init=4, n_iter=0)
    _, energy = optimizer.get_optimized_structure_and_energy(cutoff_distance=100.0)
    assert energy == pytest.approx(-optimizer.space.target.max())


def test_repr(zno, model):
    optimizer = BayesianOptimizer(model, zno, use_scaler=False, noisy=False, seed=0)
    text = repr(optimizer)
    assert text.startswith("BayesianOptimizer(relax_coords=True, relax_lattice=True, use_symmetry=True")
    assert "wyckoff_dims=[1, 1]" in text
    assert "scaler=DummyScaler" in text
    assert "noisy=False" in text


@pytest.fixture
def lj_as_dft(monkeypatch):
    """Make BayesianOptimizer.from_dict instantiate the LJ model for 'DFT'."""

    class DFT(LennardJones):
        pass

    monkeypatch.setattr(dft_module, "DFT", DFT)
    return DFT


@pytest.mark.parametrize(("module", "name"), [(megnet_module, "MEGNet"), (cgcnn_module, "CGCNN")])
def test_from_dict_dispatches_on_model_name(monkeypatch, zno, lj_as_dft, module, name):
    fake = type(name, (LennardJones,), {})
    monkeypatch.setattr(module, name, fake)
    optimizer = BayesianOptimizer(lj_as_dft(), zno, seed=0)
    optimizer.set_bounds()
    optimizer.optimize(n_init=2, n_iter=1, n_warmup=20)
    d = optimizer.as_dict()
    d["model"] = name
    assert isinstance(BayesianOptimizer.from_dict(d).model, fake)


def test_base_energy_model_is_abstract(zno):
    with pytest.raises(NotImplementedError):
        EnergyModel().predict_energy(zno)


@pytest.mark.parametrize("use_scaler", [True, False])
def test_as_dict_from_dict_roundtrip(zno, lj_as_dft, use_scaler):
    optimizer = BayesianOptimizer(lj_as_dft(), zno, use_scaler=use_scaler, noisy=False, seed=3)
    optimizer.set_bounds(abc_ratio=0.1)
    optimizer.optimize(n_init=4, n_iter=2, n_warmup=50)
    d = optimizer.as_dict()
    assert d["model"] == "DFT"
    assert d["scaler"]["@class"] == type(optimizer.scaler).__name__
    assert d["gpr"]["kernel"]["name"] == "RationalQuadratic"
    assert d["gpr"]["kernel"]["opt_params"] == optimizer.gpr.kernel_.get_params()

    restored = BayesianOptimizer.from_dict(d)
    assert isinstance(restored.model, lj_as_dft)
    assert restored.structure == optimizer.structure
    assert restored.noisy is False
    assert restored.use_symmetry
    assert np.allclose(restored.space.params, optimizer.space.params)
    assert np.allclose(restored.space.target, optimizer.space.target)
    assert np.allclose(restored.space.bounds, optimizer.space.bounds)
    assert restored.gpr.kernel == optimizer.gpr.kernel
    assert restored.gpr.kernel_ == optimizer.gpr.kernel_
    if use_scaler:
        assert np.allclose(restored.scaler.mean, optimizer.scaler.mean)
        assert np.allclose(restored.scaler.std, optimizer.scaler.std)
    assert restored.space.scaler is restored.scaler
    # Random state continues from where the original left off.
    assert restored.space.random_state.uniform() == optimizer.space.random_state.uniform()

    # Regression: the deserialized GPR used to lack _y_train_std / L_ and
    # could not predict.
    mean0, std0 = optimizer.gpr.predict(optimizer.space.params, return_std=True)
    mean1, std1 = restored.gpr.predict(restored.space.params, return_std=True)
    assert np.allclose(mean0, mean1)
    assert np.allclose(std0, std1)

    restored.optimize(n_init=0, n_iter=1, n_warmup=50, is_continue=True)
    assert len(restored.space) == len(optimizer.space) + 1


def test_as_dict_without_proposals(zno, model):
    """Regression: as_dict crashed when no acquisition had been evaluated (no _K_inv)."""
    optimizer = BayesianOptimizer(model, zno, seed=0)
    optimizer.set_bounds()
    optimizer.optimize(n_init=3, n_iter=0)
    optimizer.gpr.fit(optimizer.space.params, optimizer.space.target)
    d = optimizer.as_dict()
    assert d["gpr"]["_K_inv"] is None
    assert len(d["gpr"]["X_train_"]) == 4


def test_as_dict_unfitted_gpr(zno, model):
    """Regression: as_dict crashed before optimize() (unfitted GPR and scaler)."""
    optimizer = BayesianOptimizer(model, zno, seed=0)
    optimizer.set_bounds()
    d = optimizer.as_dict()
    assert d["scaler"]["params"] == {"mean": None, "std": None}
    assert d["gpr"]["X_train_"] is None
    assert d["gpr"]["kernel"]["opt_params"] is None
    assert d["space"]["params"] == []


def test_from_dict_legacy_gpr_dict(zno, lj_as_dft):
    """Dicts written before _y_train_std/L_ were serialized still load and predict means."""
    optimizer = BayesianOptimizer(lj_as_dft(), zno, seed=0)
    optimizer.set_bounds()
    optimizer.optimize(n_init=3, n_iter=1, n_warmup=50)
    d = optimizer.as_dict()
    for key in ("_y_train_std", "L_", "y_train_"):
        d["gpr"].pop(key)
    restored = BayesianOptimizer.from_dict(d)
    assert np.allclose(restored.gpr.predict(optimizer.space.params), optimizer.gpr.predict(optimizer.space.params))


def test_from_dict_unsupported_model(zno, model):
    optimizer = BayesianOptimizer(model, zno, seed=0)
    optimizer.set_bounds()
    optimizer.optimize(n_init=2, n_iter=1, n_warmup=20)
    d = optimizer.as_dict()
    assert d["model"] == "LennardJones"
    with pytest.raises(AttributeError, match="LennardJones is not supported"):
        BayesianOptimizer.from_dict(d)
