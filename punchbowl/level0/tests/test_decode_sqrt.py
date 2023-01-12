import numpy as np
import pytest

from punchbowl.level0.decode_sqrt import encode_sqrt, decode_sqrt_simple, decode_sqrt


def test_encoding():
    arr_dim = 2048
    arr = np.random.random([arr_dim, arr_dim]) * (2**16)

    encoded_arr = encode_sqrt(arr, from_bits=16, to_bits=10)

    assert encoded_arr.shape == arr.shape
    assert np.max(encoded_arr) <= 2**10


def test_decoding():
    arr_dim = 2048
    arr = np.random.random([arr_dim, arr_dim]) * (2**10)

    decoded_arr = decode_sqrt_simple(arr, from_bits=10, to_bits=16)

    assert decoded_arr.shape == arr.shape
    assert np.max(decoded_arr) <= 2**16


@pytest.mark.parametrize('from_bits, to_bits', [(16, 10), (16, 11), (16, 12)])
def test_encode_then_decode(from_bits, to_bits):
    arr_dim = 2048
    ccd_gain = 1.0 / 4.3    # DN/electron
    ccd_offset = 100        # DN
    ccd_read_noise = 17     # DN

    original_arr = (np.random.random([arr_dim, arr_dim]) * (2**from_bits)).astype(int)

    encoded_arr = encode_sqrt(original_arr, from_bits, to_bits)
    decoded_arr = decode_sqrt(encoded_arr,
                         from_bits = from_bits,
                         to_bits = to_bits,
                         ccd_gain = ccd_gain,
                         ccd_offset = ccd_offset,
                         ccd_read_noise = ccd_read_noise)

    noise_tolerance = np.sqrt(original_arr / ccd_gain) * ccd_gain

    test_coords = np.where(original_arr > 150)

    assert np.all(np.abs(original_arr[test_coords] - decoded_arr[test_coords]) <= noise_tolerance[test_coords])
