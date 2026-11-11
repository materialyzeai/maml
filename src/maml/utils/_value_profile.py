"""
ValueProfile return values according to certain settings. For
example, one can design a linearly increasing value profile,
a sinusoidal profile or a constant profile, depending on the
step, and previous values.
"""

from __future__ import annotations

import numpy as np


class ValueProfile:
    """
    Base class for ValueProfile. The base class has the following methods
    methods:
        increment_step(self): add one to step
        get_value(self): abstract method that return the value.
    """

    def __init__(self, max_steps: int | None = None, **kwargs):
        """
        Initializer for ValueProfile. It sets step to 0
        Args:
            **kwargs: any kwargs.
        """
        self.step = 0
        if max_steps is None:
            max_steps = np.inf  # type: ignore
        self.max_steps = max_steps

    def increment_step(self):
        """Increase step attribute by one."""
        self.step += 1
        if self.step > self.max_steps:
            raise RuntimeError("Step exceeding maximum")

    def get_value(self) -> float:
        """
        Abstract method that returns the current value
        Returns: value float.

        """
        raise NotImplementedError


class ConstantValue(ValueProfile):
    """Return constant value."""

    def __init__(self, value: float, **kwargs):
        """
        Initialize constant profile.

        Args:
            value (float): constant value
            **kwargs: pass to ValueProfile.
        """
        self.value = value
        super().__init__(**kwargs)

    def get_value(self) -> float:
        """Return constant value."""
        return self.value


class LinearProfile(ValueProfile):
    """
    LinearProfile by setting starting value and the rate of
    value change. The profile can be initialized either by
    [value_start, value_end, max_step] or [value_start, rate].
    """

    def __init__(self, value_start: float, value_end: float = 0.0, max_steps: int = 100, **kwargs):
        """

        Args:
            value_start (float):  start value
            value_end (float): end value, optional
            max_steps (int): number of steps, optional
            rate (float): rate of value change
            **kwargs: captures anything else.
        """
        self.value_start = value_start
        rate = kwargs.get("rate")
        if rate is None:
            self.rate = (value_end - self.value_start) / max_steps
        else:
            self.rate = rate
            max_steps = np.inf  # type: ignore
        super().__init__(max_steps=max_steps, **kwargs)

    def get_value(self) -> float:
        """
        Get LinearProfile value
        Returns: float.
        """
        return self.value_start + self.step * self.rate
