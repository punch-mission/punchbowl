import numpy as np
import pytest
from imagecodecs import JpeglsError, jpegls_decode

from punchbowl.auto.flows.level0 import decode_image_packets


def test_image_decompression_succeeds():
    compression_settings = {
        'JPEG': True,
        'CMP_BYP': False,
    }
    path = "punchbowl/auto/flows/tests/data/packet_sample.bin"
    with open(path, 'rb') as f:
        packets = np.frombuffer(f.read(), dtype=np.uint8)

    image = decode_image_packets(packets, compression_settings)

    assert image.shape == (2048, 2048)

    # these are hand inspected values of the image
    assert image[0, 0] == 0
    assert image[1000, 45] == 252

def test_image_decompression_fails_on_bad_packets():
    compression_settings = {
        'JPEG': True,
        'CMP_BYP': False,
    }
    path = "punchbowl/auto/flows/tests/data/faulty_packet_sample.bin"
    with open(path, 'rb') as f:
        packets = np.frombuffer(f.read(), dtype=np.uint8)

    with pytest.raises(JpeglsError):
        decode_image_packets(packets, compression_settings)
