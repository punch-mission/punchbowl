import numpy as np
import pytest
from punchbowl.simulate.noise import Noise


def test_noise_generation():
    arr_dim = 2048
    arr = np.random.random([arr_dim, arr_dim]) * (2**16)
    noise_arr = Noise.gen_noise(arr)

    assert noise_arr.shape == arr.shape
