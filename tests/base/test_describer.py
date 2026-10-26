from __future__ import annotations

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

import pytest

from maml.base import BaseDescriber, DummyDescriber, SequentialDescriber, describer_type
from maml.base._describer import _transform_one

if TYPE_CHECKING:
    from pymatgen.core import Structure


class _SquareDescriber(BaseDescriber):
    def transform_one(self, obj: float) -> np.ndarray:
        return np.array([obj, obj**2])


class _MultiOutputDescriber(BaseDescriber):
    def transform_one(self, obj: float) -> list[np.ndarray]:
        return [np.array([obj]), np.array([obj, -obj])]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags


@describer_type("structure")
class _NumSitesDescriber(BaseDescriber):
    def transform_one(self, obj: Structure) -> np.ndarray:
        return np.array([[len(obj), obj.volume]])


def test_disallowed_kwarg():
    with pytest.raises(TypeError, match="foo not allowed as kwargs"):
        _SquareDescriber(foo=1)


def test_transform_one_not_implemented():
    class _Empty(BaseDescriber):
        pass

    with pytest.raises(NotImplementedError):
        _Empty().transform_one(1)


def test_fit_returns_self():
    desc = _SquareDescriber()
    assert desc.fit([1, 2]) is desc


@pytest.mark.parametrize("feature_batch", ["stack_first_dim", None])
def test_transform_serial(feature_batch):
    desc = _SquareDescriber(feature_batch=feature_batch)
    out = desc.transform([1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(out), [[1, 1], [2, 4], [3, 9]])


def test_transform_parallel_and_verbose_match_serial():
    # worker processes must unpickle the describer, hence an importable class
    objs = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    serial = DummyDescriber(feature_batch="stack_first_dim").transform(objs)
    parallel = DummyDescriber(n_jobs=2, verbose=True, feature_batch="stack_first_dim").transform(objs)
    np.testing.assert_allclose(parallel, serial)
    np.testing.assert_allclose(parallel, np.stack(objs))


@pytest.mark.parametrize("n_jobs", [-1, 10**6])
def test_n_jobs_clipped_to_cpu_count(n_jobs):
    from joblib import cpu_count

    assert _SquareDescriber(n_jobs=n_jobs).n_jobs == cpu_count()


def test_multi_output_transform():
    out = _MultiOutputDescriber(feature_batch="stack_first_dim").transform([1.0, 2.0])
    assert isinstance(out, list)
    assert len(out) == 2
    np.testing.assert_allclose(out[0], [[1], [2]])
    np.testing.assert_allclose(out[1], [[1, -1], [2, -2]])


def test_memory_cache_and_clear(tmp_path):
    # the cached describer must be picklable by joblib, hence an importable class
    desc = DummyDescriber(memory=str(tmp_path), feature_batch="stack_first_dim")
    cached = desc.memory.cache(_transform_one)
    assert not cached.check_call_in_cache(desc, 1)
    np.testing.assert_allclose(desc.transform([1, 2]), [1, 2])
    assert cached.check_call_in_cache(desc, 1)
    assert cached.check_call_in_cache(desc, 2)
    desc.clear_cache()
    assert not cached.check_call_in_cache(desc, 1)


def test_memory_true_creates_tempdir():
    desc = _SquareDescriber(memory=True)
    assert desc.memory.location is not None
    desc.clear_cache()


def test_clear_cache_without_memory_is_noop():
    desc = _SquareDescriber()
    assert desc.memory.location is None
    desc.clear_cache()


def test_feature_dim_from_describer_type():
    # describer_type="structure" -> dummy bcc Mo with two sites
    desc = _NumSitesDescriber()
    assert desc.describer_type == "structure"
    assert desc.feature_dim == 2


def test_feature_dim_none_for_1d_features():
    @describer_type("composition")
    class _Comp(BaseDescriber):
        def transform_one(self, obj):
            return np.array([obj.num_atoms])

    assert _Comp().feature_dim is None


def test_dummy_describer_returns_input():
    df = pd.DataFrame({"a": [1, 2]})
    assert DummyDescriber().transform_one(df) is df


def test_sequential_describer():
    seq = SequentialDescriber([_SquareDescriber(feature_batch="stack_first_dim"), DummyDescriber()])
    assert [name for name, _ in seq.steps] == ["_SquareDescriber", "DummyDescriber"]
    out = seq.fit_transform([2.0, 3.0])
    np.testing.assert_allclose(np.asarray(out), [[2, 4], [3, 9]])


def test_sequential_describer_get_params_and_clone():
    """Regression: get_params/clone failed with AttributeError 'describers'."""
    from sklearn.base import clone

    describers = [_SquareDescriber(), DummyDescriber()]
    seq = SequentialDescriber(describers)
    assert seq.get_params(deep=False)["describers"] is describers
    cloned = clone(seq)
    assert [name for name, _ in cloned.steps] == ["_SquareDescriber", "DummyDescriber"]
