from __future__ import annotations

import unittest
import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest
from pymatgen.util.testing import PymatgenTest

from maml.apps.symbolic import FeatureGenerator, Operator
from maml.apps.symbolic._feature_generator import (
    _my_abs_diff,
    _my_abs_sqrt,
    _my_abs_sum,
    _my_diff,
    _my_div,
    _my_exp,
    _my_exp_power_2,
    _my_exp_power_3,
    _my_mul,
    _my_power,
    _my_sum,
    _my_sum_exp,
    _my_sum_power_2,
    _my_sum_power_3,
    _update_df,
    generate_feature,
    operation_dict,
)

pow2 = Operator.from_str("^2")
pow3 = Operator.from_str("^3")
sqrt = Operator.from_str("sqrt")
cbrt = Operator.from_str("cbrt")
exp = Operator.from_str("exp")
log10 = Operator.from_str("log10")
my_abs = Operator.from_str("abs")
add = Operator.from_str("+")
sub = Operator.from_str("-")
mul = Operator.from_str("*")
div = Operator.from_str("/")
sum_power_2 = Operator.from_str("sum_power_2")
sum_exp = Operator.from_str("sum_exp")


class TestOperator(PymatgenTest):
    def setUp(self):
        x1 = np.array([2, 4, 6, 8, 10])
        x2 = np.array([1, 4, 9, 16, 25])
        x3 = np.array([1, 8, 27, 64, 125])
        x4 = np.array([1, -8, 27, -64, 125])
        x5 = np.array([1, -4, 9, -16, 25])
        self.df = pd.DataFrame({"i1": x1, "i2": x2, "i3": x3, "i4": x4, "i5": x5})

    def testFuncitons(self):
        assert np.allclose(_my_power(self.df["i1"], n=2), np.array([4, 16, 36, 64, 100]))
        assert np.allclose(_my_abs_sqrt(self.df["i5"]), np.array([1, 2, 3, 4, 5]))
        assert np.allclose(_my_exp(self.df["i1"]), np.exp([2, 4, 6, 8, 10]))
        assert np.allclose(_my_exp_power_2(self.df["i1"]), np.exp(self.df["i1"].pow(2)).values)
        assert np.allclose(_my_exp_power_3(self.df["i1"]), np.exp(self.df["i1"].pow(3)).values)
        assert np.allclose(_my_sum(self.df["i1"], self.df["i2"]), (self.df["i1"] + self.df["i2"]).values)
        assert np.allclose(_my_abs_sum(self.df["i1"], self.df["i5"]), abs((self.df["i1"] + self.df["i5"]).values))
        assert np.allclose(_my_mul(self.df["i1"], self.df["i2"]), (self.df["i1"] * self.df["i2"]).values)
        assert np.allclose(_my_diff(self.df["i1"], self.df["i2"]), (self.df["i1"] - self.df["i2"]).values)
        assert np.allclose(_my_abs_diff(self.df["i1"], self.df["i2"]), abs(self.df["i1"] - self.df["i2"]).values)
        assert np.allclose(_my_div(self.df["i1"], self.df["i2"]), (self.df["i1"] / self.df["i2"]).values)
        assert np.allclose(_my_sum_power_2(self.df["i1"], self.df["i2"]), (self.df["i1"] + self.df["i2"]).pow(2).values)
        assert np.allclose(_my_sum_power_3(self.df["i1"], self.df["i2"]), (self.df["i1"] + self.df["i2"]).pow(3).values)
        assert np.allclose(
            _my_sum_exp(self.df["i1"], self.df["i2"]), (self.df["i1"] + self.df["i2"]).apply(np.exp).values
        )

    def testSingularOperators(self):
        pow2_2 = Operator(operation=partial(_my_power, n=2), rep="^2", unary=True, commutative=False)
        pow2_3 = Operator(**operation_dict["^2"]["kwgs"])
        assert np.allclose(pow2_2(self.df["i1"]), np.array([4, 16, 36, 64, 100]))
        assert pow2_2.is_unary
        assert not pow2_2.is_binary
        assert not pow2_2.is_commutative
        assert np.allclose(pow2_3(self.df["i1"]), np.array([4, 16, 36, 64, 100]))
        assert pow2_3.is_unary
        assert not pow2_3.is_binary
        assert not pow2_3.is_commutative

        assert np.allclose(pow2(self.df["i1"]), np.array([4, 16, 36, 64, 100]))
        assert np.allclose(pow2.compute(self.df["i1"]), np.array([4, 16, 36, 64, 100]))
        assert np.allclose(self.df["i1"].apply(pow2), np.array([4, 16, 36, 64, 100]))
        assert pow2.__str__() == pow2.rep
        assert pow2.gen_name("i1") == "(i1)^2"
        assert pow2.is_unary
        assert not pow2.is_binary
        assert not pow2.is_commutative

        assert np.allclose(pow3(self.df["i1"]), np.array([8, 64, 36 * 6, 64 * 8, 1000]))
        assert pow3.gen_name("i1") == "(i1)^3"
        assert pow3.is_unary
        assert not pow3.is_binary
        assert not pow3.is_commutative

        assert np.allclose(sqrt(self.df["i2"]), np.array([1, 2, 3, 4, 5]))
        assert sqrt.gen_name("i1") == "sqrt(i1)"
        assert sqrt.is_unary
        assert not sqrt.is_binary
        assert not sqrt.is_commutative

        assert np.allclose(cbrt(self.df["i3"]), np.array([1, 2, 3, 4, 5]))
        assert cbrt.gen_name("i3") == "cbrt(i3)"
        assert cbrt.is_unary
        assert not cbrt.is_binary
        assert not cbrt.is_commutative

        assert np.allclose(log10(self.df["i1"]), self.df["i1"].apply(np.log10).values)
        assert log10.gen_name("i1") == "log10(i1)"
        assert log10.is_unary
        assert not log10.is_binary
        assert not log10.is_commutative

        assert np.allclose(exp(self.df["i1"]), self.df["i1"].apply(np.exp).values)
        assert exp.gen_name("i1") == "exp(i1)"
        assert exp.is_unary
        assert not exp.is_binary
        assert not exp.is_commutative

        assert np.allclose(my_abs(self.df["i4"]), self.df["i3"].values)
        assert my_abs.gen_name("i4") == "abs(i4)"
        assert my_abs.is_unary
        assert not my_abs.is_binary
        assert not my_abs.is_commutative

        assert np.allclose(add(self.df["i1"], self.df["i2"]), (self.df["i1"] + self.df["i2"]).values)
        assert add.gen_name("i1", "i2") == "((i1) + (i2))"
        assert not add.is_unary
        assert add.is_binary
        assert add.is_commutative

        assert np.allclose(mul(self.df["i1"], self.df["i2"]), (self.df["i1"] * self.df["i2"]).values)
        assert mul.gen_name("i1", "i2") == "((i1) * (i2))"
        assert not mul.is_unary
        assert mul.is_binary
        assert mul.is_commutative

        assert np.allclose(sub(self.df["i1"], self.df["i2"]), (self.df["i1"] - self.df["i2"]).values)
        assert sub.gen_name("i1", "i2") == "((i1) - (i2))"
        assert not sub.is_unary
        assert sub.is_binary
        assert not sub.is_commutative

        assert np.allclose(div(self.df["i1"], self.df["i2"]), (self.df["i1"] / self.df["i2"]).values)
        assert div.gen_name("i1", "i2") == "((i1) / (i2))"
        assert not div.is_unary
        assert div.is_binary
        assert not div.is_commutative

        assert np.allclose(sum_power_2(self.df["i1"], self.df["i2"]), (self.df["i1"] + self.df["i2"]).pow(2).values)
        assert sum_power_2.gen_name("i1", "i2") == "((i1) + (i2))^2"
        assert not sum_power_2.is_unary
        assert sum_power_2.is_binary
        assert sum_power_2.is_commutative

        assert np.allclose(sum_exp(self.df["i1"], self.df["i2"]), (self.df["i1"] + self.df["i2"]).apply(np.exp).values)
        assert sum_exp.gen_name("i1", "i2") == "exp((i1) + (i2))"
        assert not sum_exp.is_unary
        assert sum_exp.is_binary
        assert sum_exp.is_commutative

    def testBinaryOperators(self):
        assert np.allclose(add(self.df["i1"], self.df["i2"]), np.array([3, 8, 15, 24, 35]))
        assert add.__str__(), "+"
        assert add.gen_name("i1", "i2") == "((i1) + (i2))"
        assert not add.is_unary
        assert add.is_binary
        assert add.is_commutative

        assert np.allclose(sub(self.df["i1"], self.df["i2"]), np.array([1, 0, -3, -8, -15]))
        assert sub.__str__(), "-"
        assert sub.gen_name("i1", "i2") == "((i1) - (i2))"
        assert not sub.is_unary
        assert sub.is_binary
        assert not sub.is_commutative


class TestFeatureGenerator(PymatgenTest):
    def setUp(self):
        x1 = np.array([2, 4, 6, 8, 10])
        x2 = np.array([1, 4, 9, 16, 25])
        x3 = np.array([1, 8, 27, 64, 125])
        self.df = pd.DataFrame({"i1": x1, "i2": x2, "i3": x3})

    def testUpdateDf(self):
        test_df = self.df.copy()
        _update_df(test_df, pow2, "i1")
        assert "(i1)^2" in test_df.columns
        assert np.allclose(test_df["(i1)^2"], test_df["i1"].pow(2).values)

        # Test duplicate won't happen
        _update_df(test_df, pow2, "i1")
        assert list(test_df.columns) == ["i1", "i2", "i3", "(i1)^2"]

        _update_df(test_df, add, "i1", "i2")
        assert "((i1) + (i2))" in test_df.columns
        assert np.allclose(test_df["((i1) + (i2))"], (test_df["i1"] + test_df["i2"]).values)

        _update_df(test_df, add, "i1", "i2")
        assert list(test_df.columns) == ["i1", "i2", "i3", "(i1)^2", "((i1) + (i2))"]

        # Test negative with sqrt and log10
        test_df["i4"] = np.array([1, -8, 27, -64, 125])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _update_df(test_df, sqrt, "i4")
            assert len(w) == 1
            assert "abssqrt" in str(w[-1].message)

    def testGenerateFeature(self):
        ops = ["^2", "+", "-"]
        nf_df = generate_feature(self.df, ops)
        new_columns = [
            "i1",
            "i2",
            "i3",
            "(i1)^2",
            "((i1) + (i2))",
            "((i1) - (i2))",
            "((i2) - (i1))",
            "((i1) + (i3))",
            "((i1) - (i3))",
            "((i3) - (i1))",
            "(i2)^2",
            "((i2) + (i3))",
            "((i2) - (i3))",
            "((i3) - (i2))",
            "(i3)^2",
        ]
        assert list(nf_df.columns) == new_columns
        assert np.allclose(nf_df["(i1)^2"].values, self.df["i1"].pow(2).values)
        assert np.allclose(nf_df["((i1) + (i2))"].values, (self.df["i1"] + self.df["i2"]).values)
        assert np.allclose(nf_df["((i1) - (i2))"].values, (self.df["i1"] - self.df["i2"]).values)
        assert np.allclose(nf_df["((i2) - (i1))"].values, (self.df["i2"] - self.df["i1"]).values)

        ops = ["^2", "abcd"]
        with pytest.raises(ValueError, match="unsupported operators"):
            generate_feature(self.df, ops)

    def testFeatureGenerator(self):
        ops = ["^2", "+", "-"]
        fg = FeatureGenerator(self.df, ops)
        nf_df = fg.augment(n=1)
        new_columns = [
            "i1",
            "i2",
            "i3",
            "(i1)^2",
            "((i1) + (i2))",
            "((i1) - (i2))",
            "((i2) - (i1))",
            "((i1) + (i3))",
            "((i1) - (i3))",
            "((i3) - (i1))",
            "(i2)^2",
            "((i2) + (i3))",
            "((i2) - (i3))",
            "((i3) - (i2))",
            "(i3)^2",
        ]
        assert list(nf_df.columns) == new_columns
        assert np.allclose(nf_df["(i1)^2"].values, self.df["i1"].pow(2).values)
        assert np.allclose(nf_df["((i1) + (i2))"].values, (self.df["i1"] + self.df["i2"]).values)
        assert np.allclose(nf_df["((i1) - (i2))"].values, (self.df["i1"] - self.df["i2"]).values)
        assert np.allclose(nf_df["((i2) - (i1))"].values, (self.df["i2"] - self.df["i1"]).values)

        df = self.df[["i1", "i2"]]
        fg = FeatureGenerator(df, ["^2", "+"])
        new_f1 = ["i1", "i2", "(i1)^2", "((i1) + (i2))", "(i2)^2"]
        new_f2 = [
            "i1",
            "i2",
            "(i1)^2",
            "((i1) + (i2))",
            "(i2)^2",
            "((i1) + ((i1)^2))",
            "((i1) + (((i1) + (i2))))",
            "((i1) + ((i2)^2))",
            "((i2) + ((i1)^2))",
            "((i2) + (((i1) + (i2))))",
            "((i2) + ((i2)^2))",
            "((i1)^2)^2",
            "(((i1)^2) + (((i1) + (i2))))",
            "(((i1)^2) + ((i2)^2))",
            "(((i1) + (i2)))^2",
            "((((i1) + (i2))) + ((i2)^2))",
            "((i2)^2)^2",
        ]
        nf1_df = fg.augment(n=1)
        nf2_df = fg.augment(n=2)
        assert list(nf1_df.columns) == new_f1
        assert list(nf2_df.columns) == new_f2


@pytest.fixture
def two_feature_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})


@pytest.mark.parametrize(
    ("operators", "expected"),
    [
        # only binary operators: the diagonal (a, a) pairs are skipped
        (["-"], {"((a) - (b))": [-3.0, -3.0, -3.0], "((b) - (a))": [3.0, 3.0, 3.0]}),
        # only unary operators: off-diagonal (a, b) pairs are skipped
        (["^2"], {"(a)^2": [1.0, 4.0, 9.0], "(b)^2": [16.0, 25.0, 36.0]}),
    ],
    ids=["binary_only", "unary_only"],
)
def test_generate_feature_single_operator_kind(two_feature_df, operators, expected):
    out = generate_feature(two_feature_df, operators)
    assert list(out.columns) == ["a", "b", *expected]
    for name, values in expected.items():
        np.testing.assert_allclose(out[name], values)
    # input frame is not mutated
    assert list(two_feature_df.columns) == ["a", "b"]


@pytest.mark.parametrize(("rep", "unary"), [("^2", True), ("sqrt", True), ("+", False), ("/", False)])
def test_operator_infers_arity_from_rep(rep, unary):
    op = Operator(operation=operation_dict[rep]["kwgs"]["operation"], rep=rep, unary=False, commutative=False)
    assert op.is_unary is unary
    assert op.is_binary is not unary
    assert str(op) == rep


def test_abslog10_operator():
    op = Operator.from_str("abslog10")
    np.testing.assert_allclose(op(np.array([-100.0, 10.0, 0.01])), [2.0, 1.0, -2.0])
    assert op.gen_name("a") == "log10(|a|)"


if __name__ == "__main__":
    unittest.main()
