"""Feature Generator."""

from __future__ import annotations

import warnings
from functools import partial
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd


def _update_df(df, op, fn1, fn2=None):
    """Helper function to update the dataframe with new generated feature array."""
    fnames = df.columns
    if op.is_unary:
        new_fname = op.gen_name(fn1)
        if new_fname not in fnames:
            if op.rep in ["sqrt", "log10"] and np.any(df[fn1] < 0):
                warnings.warn(
                    f"Data {fn1} Contains negative number, sqrt will return complex number. "
                    "Consider using abssqrt or abslog10"
                )
            df[new_fname] = df[fn1].apply(op)
    elif op.is_binary:
        new_fname = op.gen_name(fn1, fn2)
        if new_fname not in fnames:
            df[new_fname] = op(df[fn1], df[fn2])


def generate_feature(feature_df: pd.DataFrame, operators: list) -> pd.DataFrame:
    """
    Generate new features by applying operators to columns in feature_df.

    Args:
        feature_df(pd.DataFrame): dataframe of original features
        operators(list): list of str of operators (check Operator.support_op_rep for reference)

    Returns: dataframe of augmented features

    """
    fdf = feature_df.copy()
    if not np.all([(o in Operator.support_op_rep) for o in operators]):
        raise ValueError("Contain unsupported operators, check Operator.supported_op_rep")
    ops = [Operator.from_str(o) for o in operators]
    sop = [op for op in ops if op.is_unary]
    bop = [op for op in ops if op.is_binary]
    for fn1, fn2 in combinations_with_replacement(fdf.columns, r=2):
        if fn1 == fn2:
            if not sop:
                continue
            for op in sop:
                _update_df(fdf, op, fn1)
        else:
            if not bop:
                continue
            for op in bop:
                _update_df(fdf, op, fn1, fn2)
                if not op.commutative:
                    _update_df(fdf, op, fn2, fn1)
    return fdf


class FeatureGenerator:
    """FeatureGenerator class for feature augmentation before selection."""

    def __init__(self, feature_df: pd.DataFrame, operators: list):
        """

        Args:
            feature_df(pd.DataFrame): original features
            operators(list): list of operators(str).
        """
        self.fdf = feature_df
        self.operators = operators

    def augment(self, n: int = 1) -> np.ndarray:
        """
        Augment features
        Args:
            n(int): number of rounds of iteration.

        Returns: augmented dataframe

        """
        df = self.fdf.copy()
        for _ in range(n):
            ndf = generate_feature(df, self.operators)
            df = ndf
        return df


class Operator:
    """
    Operator class. Wrap math operators with more attributes including check
    is_unary, is_binary, and is_commutative, and generate name string
    for the output.
    """

    support_op_rep = [
        "^2",
        "^3",
        "sqrt",
        "abssqrt",
        "cbrt",
        "exp",
        "abs",
        "log10",
        "abslog10",
        "+",
        "-",
        "*",
        "/",
        "|+|",
        "|-|",
        "sum_power_2",
        "sum_exp",
    ]

    def __init__(self, operation: Callable[..., Any], rep: str, unary: bool, commutative: bool):
        """
        Args:
            operation(Callable): operation function
            rep(str): representations of the operator
            unary(bool): whether it is a unary operator
            commutative(bool): whether it is a commutative operator.
        """
        self.opt = operation
        self.rep = rep
        self.unary = unary
        self.commutative = commutative

    def compute(self, i1: np.ndarray, i2: np.ndarray | None = None) -> np.ndarray:
        """
        Compute the results
        Args:
            i1(np.ndarray): first input array
            i2(np.ndarray): second input array (for binary operators).

        Returns: array of computed results
        """
        if self.is_binary:
            return self.opt(i1, i2)
        return self.opt(i1)

    def gen_name(self, f1: str, f2: str | None = None) -> str:
        """
        Generate string representation for output
        Args:
            f1(str): name of the first input array
            f2(str): name of the second input array.

        Returns: name of the output

        """
        assert self.rep in operation_dict
        f_format = str(operation_dict[self.rep]["f_format"])
        if self.is_binary:
            return f_format.format(f1=f1, f2=f2)
        return f_format.format(f1=f1)

    @classmethod
    def from_str(cls, op_name: str):
        """
        Operator from name of the operator
        Args:
            op_name(str): string representation of the operator,
            check Operator.support_op_rep for reference.

        Returns: Operator

        """
        kwgs = operation_dict[op_name]["kwgs"]
        return cls(**kwgs)

    @property
    def is_unary(self) -> bool:
        """Returns: True if the operator takes one argument else False."""
        if not self.unary:
            if self.rep in ["^2", "^3", "sqrt", "abssqrt", "cbrt", "exp", "abs", "log10", "abslog10"]:
                self.unary = True
            elif self.rep in ["+", "-", "*", "/", "|+|", "|-|", "sum_power_2", "sum_exp"]:
                self.unary = False
        return self.unary

    @property
    def is_binary(self) -> bool:
        """Returns: True if the operator takes two arguments else False."""
        return not self.is_unary

    @property
    def is_commutative(self) -> bool:
        """Returns: True if the operator is commutative else False."""
        if not self.commutative:
            self.commutative = not (self.is_unary or self.rep in ["-", "/"])
        return self.commutative

    def __call__(self, i1: np.ndarray, i2: np.ndarray | None = None) -> np.ndarray:
        """
        Compute the results
        Args:
            i1(np.ndarray): first input array
            i2(np.ndarray): second input array (for binary operators).

        Returns: array of computed results

        """
        return self.compute(i1, i2)

    def __str__(self) -> str:
        return self.rep


def _my_power(x: float, n: float) -> float:
    return pow(x, n)


def _my_abs_sqrt(x):
    return np.sqrt(abs(x))


def _my_exp(x):
    return np.exp(x)


def _my_abs_log10(x):
    return np.log10(abs(x))


def _my_exp_power_2(x):
    return np.exp(pow(x, 2))


def _my_exp_power_3(x):
    return np.exp(pow(x, 3))


def _my_sum(x, y):
    return x + y


def _my_abs_sum(x, y):
    return abs(x + y)


def _my_mul(x, y):
    return x * y


def _my_diff(x, y):
    return x - y


def _my_abs_diff(x, y):
    return abs(x - y)


def _my_div(x, y):
    return x / y


def _my_sum_power_2(x, y):
    return pow((x + y), 2)


def _my_sum_power_3(x, y):
    return pow((x + y), 3)


def _my_sum_exp(x, y):
    return np.exp(x + y)


# def _my_sum_exp_power_2(x, y):
#     return np.exp(pow(x + y, 2))
#
#
# def _my_sum_exp_power_3(x, y):
#     return np.exp(pow(x + y, 3))


operation_dict: dict[str, Any]
operation_dict = {
    "^2": {
        "kwgs": {"operation": partial(_my_power, n=2), "rep": "^2", "unary": True, "commutative": False},
        "f_format": "({f1})^2",
    },
    "^3": {
        "kwgs": {"operation": partial(_my_power, n=3), "rep": "^3", "unary": True, "commutative": False},
        "f_format": "({f1})^3",
    },
    "abs": {"kwgs": {"operation": abs, "rep": "abs", "unary": True, "commutative": False}, "f_format": "abs({f1})"},
    "sqrt": {
        "kwgs": {"operation": partial(_my_power, n=1 / 2), "rep": "sqrt", "unary": True, "commutative": False},
        "f_format": "sqrt({f1})",
    },
    "abssqrt": {
        "kwgs": {"operation": _my_abs_sqrt, "rep": "abssqrt", "unary": True, "commutative": False},
        "f_format": "sqrt(|{f1}|)",
    },
    "cbrt": {
        "kwgs": {"operation": partial(_my_power, n=1 / 3), "rep": "cbrt", "unary": True, "commutative": False},
        "f_format": "cbrt({f1})",
    },
    "exp": {"kwgs": {"operation": _my_exp, "rep": "exp", "unary": True, "commutative": False}, "f_format": "exp({f1})"},
    "log10": {
        "kwgs": {"operation": np.log10, "rep": "log10", "unary": True, "commutative": False},
        "f_format": "log10({f1})",
    },
    "abslog10": {
        "kwgs": {"operation": _my_abs_log10, "rep": "abslog10", "unary": True, "commutative": False},
        "f_format": "log10(|{f1}|)",
    },
    "+": {
        "kwgs": {"operation": _my_sum, "rep": "+", "unary": False, "commutative": True},
        "f_format": "(({f1}) + ({f2}))",
    },
    "|+|": {
        "kwgs": {"operation": _my_abs_sum, "rep": "|+|", "unary": False, "commutative": True},
        "f_format": "|({f1}) + ({f2})|",
    },
    "-": {
        "kwgs": {"operation": _my_diff, "rep": "-", "unary": False, "commutative": False},
        "f_format": "(({f1}) - ({f2}))",
    },
    "|-|": {
        "kwgs": {"operation": _my_abs_diff, "rep": "|-|", "unary": False, "commutative": True},
        "f_format": "|({f1}) - ({f2})|",
    },
    "*": {
        "kwgs": {"operation": _my_mul, "rep": "*", "unary": False, "commutative": True},
        "f_format": "(({f1}) * ({f2}))",
    },
    "/": {
        "kwgs": {"operation": _my_div, "rep": "/", "unary": False, "commutative": False},
        "f_format": "(({f1}) / ({f2}))",
    },
    "sum_power_2": {
        "kwgs": {"operation": _my_sum_power_2, "rep": "sum_power_2", "unary": False, "commutative": True},
        "f_format": "(({f1}) + ({f2}))^2",
    },
    "sum_exp": {
        "kwgs": {"operation": _my_sum_exp, "rep": "sum_exp", "unary": False, "commutative": True},
        "f_format": "exp(({f1}) + ({f2}))",
    },
}
