"""Utils for describers."""

from __future__ import annotations

import logging
from collections import Counter
from functools import partial

import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _add_allowed_stats(cls):
    """
    Decorate to add allowed_stats to the Stats class.

    Args:
        cls: Stats class

    Returns: Stats class with allowed_stats attributes

    """
    all_keys = list(cls.__dict__.keys())
    allowed = []
    for key in all_keys:
        if isinstance(cls.__dict__[key], staticmethod):
            allowed.append(key)
    cls.allowed_stats = allowed
    return cls


@_add_allowed_stats
class Stats:
    """
    Calculate the stats of a list of values.
    This is particularly helpful when you want to convert
    lists of values of different lengths to uniform length
    for machine learning purposes.

    supported
    """

    @staticmethod
    def max(data: list[float], weights: list[float] | None = None) -> float:
        """
        Max of value
        Args:31
            data (list): list of float data
            weights (list): optional weights.

        Returns: maximum value

        """
        return np.max(data)

    @staticmethod
    def min(data: list[float], weights: list[float] | None = None) -> float:
        """
        Min of value
        Args:
            data (list): list of float data
            weights (list): optional weights.

        Returns: minimum value

        """
        return np.min(data)

    @staticmethod
    def range(data: list[float], weights: list[float] | None = None) -> float:
        """
        Range of values
        Args:
            data (list): list of float data
            weights (list): optional weights.

        Returns: range of values, i.e., max - min

        """
        return Stats.max(data) - Stats.min(data)

    @staticmethod
    def mode(data: list[float], weights: list[float] | None = None) -> float:
        """
        Mode of data, if multiple entries have equal counts,
        compute the average of those.

        Args:
            data (list): list of float data
            weights (list): optional weights

        Returns: mode of values, i.e., max - min

        """
        if weights is None:
            counts = Counter(data)
            most_common = counts.most_common()
            max_count = most_common[0][1]
            modes = []
            for v, c in most_common:
                if c == max_count:
                    modes.append(v)
            return np.mean(modes).item()

        data_array = np.array(data)
        weights_array = np.array(weights)
        maxes = np.isclose(weights_array, weights_array.max())
        max_data = data_array[maxes]
        return np.mean(max_data).item()

    @staticmethod
    def mean_absolute_deviation(data: list[float], weights: list[float] | None = None) -> float:
        """
        Mean absolute deviation.

        Args:
            data (list): list of float data
            weights (list): optional weights

        Returns: mean absolute deviation
        """
        mean = Stats.mean(data, weights)
        data_sub = [abs(i - mean) for i in data]
        return Stats.mean(data_sub, weights)

    @staticmethod
    def mean_absolute_error(data: list[float], weights: list[float] | None = None) -> float:
        """
        Mean absolute error.

        Args:
            data (list): list of float data
            weights (list): optional weights

        Returns: mean absolute error
        """
        return Stats.mean_absolute_deviation(data=data, weights=weights)

    @staticmethod
    def moment(
        data: list[float],
        weights: list[float] | None = None,
        order: int | None = None,
        max_order: int | None = None,
    ):
        """
        Moment of probability mass function.

        order = 1 means weighted mean
        order = 2 means standard deviation
        order > 2 corresponds to higher order moment to
            the 1./order power

        Args:
            data (list): list of float data
            weights (list or None): weights for each data points
            order (int): moment order
            max_order (int): if set, it will overwrite order

        Returns: float or list of floats

        """
        # check if only a single value should be output
        single = max_order is None

        if weights is None:
            weights = [1.0 / len(data)] * len(data)

        if max_order is not None:
            if order is not None:
                logger.info("max_order will overwrite the order parameter")
            orders = list(range(1, max_order + 1))
        else:
            orders = [order or 1]

        stats = [_root_moment(data, weights, i) for i in orders]

        if not single:
            return stats
        return stats[0]

    @staticmethod
    def mean(data: list[float], weights: list[float] | None = None) -> float:
        """
        Weighted average.

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: average value

        """
        return Stats.moment(data, weights=weights, order=1)

    @staticmethod
    def inverse_mean(data: list[float], weights: list[float] | None = None) -> float:
        """
        Inverse mean.

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: average value

        """
        return Stats.mean([1.0 / x for x in data], weights=weights)

    @staticmethod
    def average(data: list[float], weights: list[float] | None = None) -> float:
        """
        Weighted average.

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: average value

        """
        return Stats.mean(data, weights=weights)

    @staticmethod
    def std(data: list[float], weights: list[float] | None = None) -> float:
        """
        Standard deviation.

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: Standard deviation

        """
        return Stats.moment(data, weights=weights, order=2)

    @staticmethod
    def skewness(data: list[float], weights: list[float] | None = None) -> float:
        """
        Skewness of the distribution.

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: Skewness of the distribution

        """
        third = Stats.moment(data, weights=weights, order=3)
        std = Stats.std(data, weights=weights)
        if std < 1e-4:
            std = 1.0
        return third**3 / std**3

    @staticmethod
    def kurtosis(data: list[float], weights: list[float] | None = None) -> float:
        """
        Kurtosis of the distribution.

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: Kurtosis of the distribution

        """
        fourth = Stats.moment(data, weights=weights, order=4)
        std = Stats.std(data, weights=weights)
        if std < 1e-4:
            std = 1.0
        return fourth**4 / std**4

    @staticmethod
    def geometric_mean(data: list[float], weights: list[float] | None = None) -> float:
        """
        Geometric mean of the data.

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: geometric mean of the distribution

        """
        return Stats.power_mean(data, weights, p=0)

    @staticmethod
    def power_mean(data: list[float], weights: list[float] | None = None, p: int = 1) -> float:
        """
        Power mean https://en.wikipedia.org/wiki/Generalized_mean.

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point
            p (int): power

        Returns: power mean of the distribution

        """
        if np.any(np.array(data) <= 0.0):
            raise ValueError("Not possible to calculate geometric means for negative values")

        weights = [1.0 / len(data)] * len(data) if weights is None else [(i / sum(weights)) for i in weights]

        assert abs(sum(weights) - 1) < 1e-3

        if p == 0:
            return np.prod([i**j for i, j in zip(data, weights, strict=False)]).item()

        s = np.sum([j * i**p for i, j in zip(data, weights, strict=False)])
        return s ** (1.0 / p)

    @staticmethod
    def shifted_geometric_mean(data: list[float], weights: list[float] | None = None, shift: float = 100) -> float:
        """
        Since we cannot calculate the geometric means on negative or zero values,
        we can first shift all values to positive and then calculate the geometric mean
        afterwards, we shift the computed geometric mean back by a shift value.

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point
            shift (float): shift value

        Returns: geometric mean of the distribution

        """
        data_new = [i + shift for i in data]
        return Stats.geometric_mean(data_new, weights=weights) - shift

    @staticmethod
    def harmonic_mean(data: list[float], weights: list[float] | None = None) -> float:
        """
        Harmonic mean of the data.

        Args:
            data (list): list of float data
            weights (list or None): weights for each data point

        Returns: harmonic mean of the distribution

        """
        weights = [1.0 / len(data)] * len(data) if weights is None else [(i / sum(weights)) for i in weights]
        return np.sum(weights) / np.sum(np.array(weights) / np.array(data))


def _root_moment(data, weights, order) -> float:
    """
    Auxiliary function to compute moment.

    Args:
        data (list): list of float data
        weights (list or None): weights for each data point
        order (int): order of moment

    Returns: moment of order

    """
    weights_sum = np.sum(weights)
    pmf = np.array(weights) / weights_sum
    mean = 0.0

    if order > 1:
        # central moments are taken about the weighted mean
        mean = float(np.sum(np.asarray(data, dtype=float) * pmf))

    moment = sum((i - mean) ** order * j for i, j in zip(data, pmf, strict=False))

    # when order is odd, moment can be negative
    sign = -1 if moment < 0 else 1

    # avoid error like np.power(-0.001, 1./3.)
    return (sign * moment) ** (1.0 / order) * sign


def _convert_a_or_b(v: str, a=int, b=None):
    try:
        return a(v)
    except ValueError:
        return b


def _moment_symbol_conversion(moment_symbol: str):
    splits = moment_symbol.split(":")
    max_order = _convert_a_or_b(splits[2], int, None)

    if max_order is None:
        return [moment_symbol]
    if max_order > 0:
        return [f"moment:{i}:None" for i in range(1, max_order + 1)]
    return ["moment:0:None"]


def stats_list_conversion(stats_list: list[str]) -> list[str]:
    """
    Convert a list of stats str into a fully expanded list.
    This applies mainly to stats that can return a list of values, e.g.,
    moment with max_order > 1.

    Args:
        stats_list (list): list of stats str

    Returns: list of expanded stats str

    """
    re_list = []
    for st in stats_list:
        if ":" not in st:
            re_list.append(st)
        elif "moment" in st:
            moment = _moment_symbol_conversion(st)
            re_list.extend(moment)
        else:
            re_list.append(st)
    return re_list


STATS_KWARGS = {"moment": [{"order": int}, {"max_order": int}], "shifted_geometric_mean": [{"shift": float}]}


def get_full_stats_and_funcs(stats: list) -> tuple[list[str], list]:
    """
    Get expanded stats function name str and the corresponding
    function callables.

    Args:
        stats (list): a list of stats names, e.g, ['mean', 'std', 'moment:1:None']

    Returns: list of stats names, list of stats callable

    """
    stats_func = []
    full_stats = stats_list_conversion(stats)

    for stat in full_stats:
        if ":" in stat:
            splits = stat.split(":")
            stat_name = splits[0]

            if stat_name.lower() not in getattr(Stats, "allowed_stats", []):  # type: ignore
                raise ValueError(f"{stat_name.lower()} not in available Stats")

            func = getattr(Stats, stat_name)
            args = splits[1:]
            arg_dict = {}
            for name_dict, arg in zip(STATS_KWARGS[stat_name], args, strict=False):  # type: ignore
                name = next(iter(name_dict.keys()))
                value_type = next(iter(name_dict.values()))
                try:
                    value = value_type(arg)
                except ValueError:
                    value = None  # type: ignore
                arg_dict[name] = value
            stats_func.append(partial(func, **arg_dict))
            continue

        if stat.lower() not in getattr(Stats, "allowed_stats", []):  # type: ignore
            raise ValueError(f"{stat.lower()} not in available Stats")
        stats_func.append(getattr(Stats, stat))
    return full_stats, stats_func
