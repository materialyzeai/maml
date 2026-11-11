"""Utilities package."""

from __future__ import annotations

from ._data_conversion import convert_docs, pool_from, to_array
from ._data_split import DataSplitter, ShuffleSplitter
from ._dummy import feature_dim_from_test_system, get_describer_dummy_obj
from ._inspect import get_full_args
from ._jit import njit
from ._lammps import (
    check_structures_forces_stresses,
    get_lammps_lattice_and_rotation,
    stress_format_change,
    stress_list_to_matrix,
    stress_matrix_to_list,
    write_data_from_structure,
)
from ._material import to_composition
from ._preprocessing import DummyScaler, Scaler, StandardScaler
from ._signal_processing import fft_magnitude, get_sp_method, spectrogram, wvd
from ._stats import STATS_KWARGS, Stats, get_full_stats_and_funcs, stats_list_conversion
from ._tempfile import MultiScratchDir
from ._value_profile import ConstantValue, LinearProfile, ValueProfile

__all__ = [
    "STATS_KWARGS",
    "ConstantValue",
    "DataSplitter",
    "DummyScaler",
    "LinearProfile",
    "MultiScratchDir",
    "Scaler",
    "ShuffleSplitter",
    "StandardScaler",
    "Stats",
    "ValueProfile",
    "check_structures_forces_stresses",
    "convert_docs",
    "cwt",
    "feature_dim_from_test_system",
    "fft_magnitude",
    "get_describer_dummy_obj",
    "get_full_args",
    "get_full_stats_and_funcs",
    "get_lammps_lattice_and_rotation",
    "get_sp_method",
    "njit",
    "pool_from",
    "spectrogram",
    "stats_list_conversion",
    "stress_format_change",
    "stress_list_to_matrix",
    "stress_matrix_to_list",
    "to_array",
    "to_composition",
    "write_data_from_structure",
    "wvd",
]
