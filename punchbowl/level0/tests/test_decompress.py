from punchbowl.level0.decode import create_fake_ndcube

from ndcube import NDCube
import pytest
import numpy as np
from punchbowl.level0.decompress import encode, decode_simple, decode


def test_encoding():
    arr_dim = 2048
    arr = np.random.random([arr_dim, arr_dim]) * (2**16)

    encoded_arr = encode(arr, from_bits=16, to_bits=10)

    assert encoded_arr.shape == arr.shape
    assert np.max(encoded_arr) <= 2**10


def test_decoding():
    arr_dim = 2048
    arr = np.random.random([arr_dim, arr_dim]) * (2**10)

    decoded_arr = decode_simple(arr, from_bits=10, to_bits=16)

    assert decoded_arr.shape == arr.shape
    assert np.max(decoded_arr) <= 2**16


@pytest.mark.parametrize('from_bits, to_bits', [(16, 10), (16, 11), (16, 12)])
def test_encode_then_decode(from_bits, to_bits):
    arr_dim = 2048
    ccd_gain = 1.0 / 4.3  # DN/electron
    ccd_offset = 100  # DN
    ccd_read_noise = 17  # DN

    original_arr = (np.random.random([arr_dim, arr_dim]) * (2**from_bits)).astype(int)

    encoded_arr = encode(original_arr, from_bits, to_bits)
    decoded_arr = decode(encoded_arr,
                         from_bits=from_bits,
                         to_bits=to_bits,
                         ccd_gain=ccd_gain,
                         ccd_offset=ccd_offset,
                         ccd_read_noise=ccd_read_noise)

    # TODO: use a calculated value instead of 200
    assert np.all(np.abs(original_arr - decoded_arr) <= 200)  # np.allclose(original_arr, decoded_arr)
