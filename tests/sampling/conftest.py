from __future__ import annotations

import os
import pickle

import pytest

PC_weighted_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "MPF_2021_2_8_first10_test_PC_weighted.pickle",
)

PC_unweighted_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "MPF_2021_2_8_first10_test_PC_unweighted.pickle",
)

Birch_result_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "MPF_2021_2_8_first10_Birch_results.pickle",
)

feature_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "M3GNet_features_MPF_2021_2_8_first10_features_test.pickle",
)


@pytest.fixture(scope="session")
def MPF_2021_2_8_first10_features_test():
    with open(feature_file_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def Birch_results():
    with open(Birch_result_file_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def PC_unweighted():
    with open(PC_unweighted_file_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def PC_weighted():
    with open(PC_weighted_file_path, "rb") as f:
        return pickle.load(f)
