"""Signal processing utils."""

from __future__ import annotations

from math import ceil, floor
from typing import TYPE_CHECKING, Any

import numpy as np
from monty.dev import requires
from scipy import fft, signal

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import tftb
except ImportError:
    tftb = None


def fft_magnitude(z: np.ndarray) -> np.ndarray:
    """
    Discrete Fourier Transform the signal z and return
    the magnitude of the  coefficients
    Args:
        z (np.ndarray): 1D signal array
    Returns: 1D magnitude.
    """
    return np.absolute(fft.fft(z))


def spectrogram(z: np.ndarray, return_time_freq: bool = False) -> tuple | np.ndarray:
    """
    The spectrogram of the signal
    Args:
        z (np.ndarray): 1D signal array
        return_time_freq (bool): whether to return time and frequency
    Returns: 2D spectrogram.
    """
    nx = len(z)
    nsc = floor(nx / 4.5)  # use matlab values
    nov = floor(nsc / 2)
    nfft = max(256, 2 ** ceil(np.log2(nsc)))
    freq, time, s = signal.spectrogram(z, window=signal.get_window("hann", nsc), noverlap=nov, nfft=nfft)
    if return_time_freq:
        return freq, time, s
    return s


@requires(tftb is not None, "Requires installation of tftb package")
def wvd(z: np.ndarray, return_all: bool = False) -> tuple | np.ndarray:
    """
    Wigner Ville Distribution calculator
    Args:
        z (np.ndarray): signal 1D
        return_all (bool): whether to return time and freq info, default
            only return the wvd information
    Returns: NxN wvd matrix.
    """
    tfr = tftb.processing.WignerVilleDistribution(z)
    (
        res,
        f1,
        f2,
    ) = tfr.run()
    if return_all:
        return res, f1, f2
    return res


AVAILABLE_SP_METHODS = {"fft_magnitude": fft_magnitude, "spectrogram": spectrogram, "wvd": wvd}


def get_sp_method(sp_method: str | Callable[..., Any]) -> Callable[..., Any]:  # type: ignore
    """
    Providing a signal processing method name return the callable
    Args:
        sp_method (str): name of the sp function
    Returns: callable for signal processing.
    """
    if isinstance(sp_method, str):
        try:
            return AVAILABLE_SP_METHODS[sp_method]  # type: ignore
        except KeyError:
            raise KeyError(f"{sp_method} is not in available methods: {AVAILABLE_SP_METHODS.keys()}")
    else:
        return sp_method
