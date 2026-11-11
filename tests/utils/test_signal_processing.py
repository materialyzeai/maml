from __future__ import annotations

import os
import unittest

import numpy as np
import pytest

from maml.utils import fft_magnitude, get_sp_method, spectrogram, wvd

CWD = os.path.join(os.path.dirname(__file__))

try:
    import tftb
except ImportError:
    tftb = None


class TestSP(unittest.TestCase):
    x = np.load(os.path.join(CWD, "test_x.npy"))

    def test_fft(self):
        fft_mag = fft_magnitude(TestSP.x)
        assert fft_mag.shape == (100,)
        fft_mag = get_sp_method("fft_magnitude")(TestSP.x)
        assert fft_mag.shape == (100,)
        fft_mag = get_sp_method(fft_magnitude)(TestSP.x)
        assert fft_mag.shape == (100,)
        self.assertRaises(KeyError, get_sp_method, sp_method="wrong_method")

    def test_spectrogram(self):
        spec = spectrogram(TestSP.x)
        assert spec.shape == (129, 8)

        freq, time, spec = spectrogram(TestSP.x, return_time_freq=True)
        assert freq.shape == (129,)
        assert time.shape == (8,)
        assert spec.shape == (129, 8)

    @unittest.skipIf(tftb is None, "tftb is required to run wvd")
    def test_wvd(self):
        wvd_res = wvd(TestSP.x)
        assert wvd_res.shape == (100, 100)
        wvd_res, f1, _f2 = wvd(TestSP.x, return_all=True)
        assert wvd_res.shape == (100, 100)
        assert f1.shape == (100,)


@unittest.skipIf(tftb is not None, "tests the error path when tftb is missing")
def test_wvd_requires_tftb():
    with pytest.raises(RuntimeError, match="tftb"):
        wvd(np.ones(16))


if __name__ == "__main__":
    unittest.main()
